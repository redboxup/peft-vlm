import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
from typing import List, Optional

from clip import clip
from clip.model import CLIP
from models import BaseCLIP
from configs import ProjectConf
from models.text_encoder import TextEncoder
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

class PromptLearner(nn.Module):
    def __init__(self, configs: ProjectConf, clip_model: CLIP):
        super().__init__()
        self.device = configs.device
        self.configs = configs
        n_cls = len(configs.dataset.cxr_labels)
        n_ctx_pos = configs.trainer.N_CTX
        n_ctx_neg = configs.trainer.N_CTX
        ctx_init_pos = configs.trainer.CTX_INIT_POS # type: ignore
        ctx_init_neg = configs.trainer.CTX_INIT_NEG # type: ignore
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = configs.trainer.img_size
        assert cfg_imsize == clip_imsize,(
            f"cfg_imsize ({cfg_imsize}) must"
            f" equal to clip_imsize ({clip_imsize})"
        )

        if ctx_init_pos and ctx_init_neg:
            # use given words to initialize context vectors
            ctx_init_pos = ctx_init_pos.replace("_", " ")
            ctx_init_neg = ctx_init_neg.replace("_", " ")
            n_ctx_pos = len(ctx_init_pos.split(" "))
            n_ctx_neg = len(ctx_init_neg.split(" "))
            prompt_pos = clip.tokenize(ctx_init_pos) # [num_prompts, context_len]
            prompt_neg = clip.tokenize(ctx_init_neg) # [num_prompts, context_len]
            prompt_pos = prompt_pos.to(self.device)
            prompt_neg = prompt_neg.to(self.device)
            with torch.no_grad():
                embedding_pos = (
                    clip_model.token_embedding(prompt_pos).to(self.device)
                ) # [num_prompts, context_len, emb_dim]
                embedding_neg = (
                    clip_model.token_embedding(prompt_neg).to(self.device)
                ) # [num_prompts, context_len, emb_dim]

            ctx_vectors_pos = embedding_pos[0, 1 : 1 + n_ctx_pos, :]
            ctx_vectors_neg = embedding_neg[0, 1 : 1 + n_ctx_neg, :]
            prompt_prefix_pos = ctx_init_pos
            prompt_prefix_neg = ctx_init_neg
            # ctx_vectors: [n_ctx, ctx_dim]
            # if configs.trainer.CSC:
            #     if self.configs.verbose:
            #         print("Initializing class-specific contexts")
            #     ctx_vectors_pos_ = []
            #     ctx_vectors_neg_ = []
            #     for i in range(n_cls):
            #         ctx_vectors_pos_.append(deepcopy(ctx_vectors_pos))
            #         ctx_vectors_neg_.append(deepcopy(ctx_vectors_neg))
            #     ctx_vectors_pos = torch.stack(ctx_vectors_pos_, dim=0)
            #     ctx_vectors_neg = torch.stack(ctx_vectors_neg_, dim=0)
        elif ctx_init_pos and not ctx_init_neg:
            # use given words to initialize positive context vectors
            ctx_init_pos = ctx_init_pos.replace("_", " ")
            n_ctx_pos = len(ctx_init_pos.split(" "))
            prompt_pos = clip.tokenize(ctx_init_pos)
            prompt_pos = prompt_pos.to(self.device)
            with torch.no_grad():
                embedding_pos = (
                    clip_model.token_embedding(prompt_pos).to(self.device)
                )
            ctx_vectors_pos = embedding_pos[0, 1 : 1 + n_ctx_pos, :]
            prompt_prefix_pos = ctx_init_pos
            ctx_vectors_neg = torch.empty(n_ctx_neg, ctx_dim, device=self.device)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            prompt_prefix_neg = " ".join(["Y"] * n_ctx_neg)  # type: ignore
        
            # if configs.trainer.CSC:
            #     if self.configs.verbose:
            #         print("Initializing class-specific contexts")
            #     ctx_vectors_pos_ = []
            #     for i in range(n_cls):
            #         ctx_vectors_pos_.append(deepcopy(ctx_vectors_pos))
            #     ctx_vectors_pos = torch.stack(ctx_vectors_pos_, dim=0)

        elif not ctx_init_pos and ctx_init_neg:
            # use given words to initialize negative context vectors
            ctx_init_neg = ctx_init_neg.replace("_", " ")
            n_ctx_neg = len(ctx_init_neg.split(" "))
            prompt_neg = clip.tokenize(ctx_init_neg)
            prompt_neg = prompt_neg.to(self.device)
            with torch.no_grad():
                # print(f"device: {self.device}")
                # print(f"prompt_neg device: {prompt_neg.device}")
                # print(f"clip_model device: {clip_model.device}")
                embedding_neg = (
                    clip_model.token_embedding(prompt_neg).to(self.device)
                )
            ctx_vectors_neg = embedding_neg[0, 1 : 1 + n_ctx_neg, :]
            prompt_prefix_neg = ctx_init_neg
            ctx_vectors_pos = torch.empty(n_ctx_pos, ctx_dim, device=self.device)
            nn.init.normal_(ctx_vectors_pos, std=0.02)
            prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)
        else:
            # # random initialization
            # if configs.trainer.CSC:
            #     if self.configs.verbose:
            #         print("Initializing class-specific contexts")
            #     ctx_vectors_pos = torch.empty(
            #         n_cls, n_ctx_pos, ctx_dim, device=self.device
            #     )
            #     ctx_vectors_neg = torch.empty(
            #         n_cls, n_ctx_neg, ctx_dim, device=self.device
            #     )
            # else:
            if self.configs.verbose:
                print("Initializing a generic context")
            ctx_vectors_pos = torch.empty(
                n_ctx_pos, ctx_dim, device=self.device
            )
            ctx_vectors_neg = torch.empty(
                n_ctx_neg, ctx_dim, device=self.device
            )

            nn.init.normal_(ctx_vectors_pos, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)
            prompt_prefix_neg = " ".join(["Y"] * n_ctx_neg)  

        if self.configs.verbose:
            print(f'Initial context: "{prompt_prefix_pos}"')
            print(f'Initial context: "{prompt_prefix_neg}"')
            print(f"Number of positive context words (tokens): {n_ctx_pos}")
            print(f"Number of negative context words (tokens): {n_ctx_neg}")

        # ctx_(neg/pos) shape: 
        # Class-Specific Context: [n_cls, n_ctx_(neg/pos), ctx_dim]
        # Shared Context across classes: [n_ctx_(neg/pos), ctx_dim]
        self.ctx_pos = nn.Parameter(ctx_vectors_pos)
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)

        classnames = [
            name.replace("_", " ") for name in configs.dataset.cxr_labels
        ]
        name_lens = [
            len(_tokenizer.encode(name)) for name in classnames
        ]
        prompts_pos = [
            prompt_prefix_pos + " " + name  for name in classnames
        ]
        prompts_neg = [
            prompt_prefix_neg + " " + name for name in classnames
        ]

        tokenized_prompts_pos = []
        tokenized_prompts_neg = []
        for p_pos, p_neg in zip(prompts_pos, prompts_neg):
            tokenized_prompts_pos.append(
                clip.tokenize(p_pos).to(self.device)
            )
            tokenized_prompts_neg.append(
                clip.tokenize(p_neg).to(self.device)
            )

        # tokenized_prompts_(pos/neg): [n_cls, context_len]
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg)

        with torch.no_grad():
            # embedding_(pos/neg): [n_cls, context_len, emb_dim]
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer(
            "token_prefix_pos",
            embedding_pos[:, :1, :].to(self.device),
        )  # SOS
        self.register_buffer(
            "token_suffix_pos",
            embedding_pos[:, 1 + n_ctx_pos :, :].to(self.device),
        )  # CLS, EOS
        self.register_buffer(
            "token_prefix_neg",
            embedding_neg[:, :1, :].to(self.device),
        )
        self.register_buffer(
            "token_suffix_neg",
            embedding_neg[:, 1 + n_ctx_neg :, :].to(self.device),
        )

        self.n_cls = n_cls
        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg

        # tokenized_prompts: [n_cls * 2, context_len]
        tokenized_prompts = torch.cat(
            [tokenized_prompts_neg,tokenized_prompts_pos],
            dim = 0
        )
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.name_lens = name_lens
        self.class_token_position = configs.trainer.CLASS_TOKEN_POSITION

    def forward(self, cls_ids: Optional[List] = None):
        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg

        if ctx_pos.dim() == 2:
                if cls_ids is None:
                    ctx_pos = ctx_pos.unsqueeze(0).expand(self.n_cls,-1,-1)
                else:
                    ctx_pos = ctx_pos.unsqueeze(0).expand(len(cls_ids),-1,-1)
        else:
            if cls_ids is not None:
                ctx_pos = ctx_pos[cls_ids]

        if ctx_neg.dim() == 2:
                if cls_ids is None:
                    ctx_neg = ctx_neg.unsqueeze(0).expand(self.n_cls,-1,-1)
                else:
                    ctx_neg = ctx_neg.unsqueeze(0).expand(len(cls_ids),-1,-1)
        else:
            if cls_ids is not None:
                ctx_neg = ctx_neg[cls_ids]

        # ctx_neg: [n_cls, n_ctx, dim]
        # ctx_pos: [n_cls, n_ctx, dim]
        

        if cls_ids is None:
            prefix_pos = self.token_prefix_pos
            prefix_neg = self.token_prefix_neg
            suffix_pos = self.token_suffix_pos
            suffix_neg = self.token_suffix_neg
        else:
            prefix_pos = self.token_prefix_pos[cls_ids]
            prefix_neg = self.token_prefix_neg[cls_ids]
            suffix_pos = self.token_suffix_pos[cls_ids]
            suffix_neg = self.token_suffix_neg[cls_ids]

        if self.class_token_position == "end":
            prompts_pos = torch.cat(
                [
                    prefix_pos,  # (n_cls, 1, dim)
                    ctx_pos,     # (n_cls, n_ctx, dim)
                    suffix_pos,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            prompts_neg = torch.cat(
                [
                    prefix_neg,  # (n_cls, 1, dim)
                    ctx_neg,     # (n_cls, n_ctx, dim)
                    suffix_neg,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        prompts = torch.cat([prompts_neg, prompts_pos], dim=0) # type: ignore

        if cls_ids is not None:
            tokenized_prompts_pos = (
                self.tokenized_prompts[self.n_cls:][cls_ids]
            )
            tokenized_prompts_neg = (
                self.tokenized_prompts[:self.n_cls][cls_ids]
            )
            tokenized_prompts = torch.cat(
                [tokenized_prompts_neg, tokenized_prompts_pos],
                dim=0
            )
        else:
            tokenized_prompts = self.tokenized_prompts

        return prompts, tokenized_prompts

@BaseCLIP.register_model("dualcoop")
class DualCoOpCLIP(BaseCLIP):
    def __init__(self, configs: ProjectConf, clip_model: CLIP):
        super().__init__(configs, clip_model)
        self.prompt_learner = PromptLearner(configs, clip_model)
        self.text_encoder = TextEncoder(clip_model, device=configs.device)
        self.optim_params = self.prompt_learner.parameters()

    def forward(self, images):
        images = images.to(self.configs.device) 
        image_features = self.clip_model.visual(images)

        prompts = self.prompt_learner()[0].to(self.configs.device)
        tokenized_prompts = (
            self.prompt_learner
            .tokenized_prompts
            .to(self.configs.device)
        )
        # number of classes * 2, since we have positive and negative prompts
        n_cls_2 = prompts.shape[0] // 2

        # pos_prompts: [n_cls, context_len, output_dim]
        pos_prompts = prompts[n_cls_2:]

        # pos_tokenized_prompts: [n_cls, context_len]
        pos_tokenized_prompts = tokenized_prompts[n_cls_2:]

        # text_features: [n_cls, output_dim]
        text_features = self.text_encoder(pos_prompts, pos_tokenized_prompts)

        # normalize features
        text_features = (
            text_features / text_features.norm(dim=-1, keepdim=True)
        )
        image_features = (
            image_features / image_features.norm(dim=-1, keepdim=True)
        )

        logit_scale = 1 # self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

    def forward_with_neg(self, images):
        images = images.to(self.configs.device)

        # image_features: [batch, output_dim]
        # local_image_features shape: [batch, height, width, dim]
        # output_dim = 512, height = width = 7 for ViT-B/32
        image_features, local_image_features = (
            self.clip_model.visual.forward_feats(images)
        )

        # prompts: [n_cls * 2, context_len, output_dim]
        prompts = self.prompt_learner()[0].to(self.configs.device)
        tokenized_prompts = (
            self.prompt_learner.tokenized_prompts.to(self.configs.device)
        )
        # text_features: [n_cls * 2, output_dim]
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = (
            image_features / image_features.norm(dim=-1, keepdim=True)
        )
        local_image_features = (
            local_image_features /
            local_image_features.norm(dim=-1, keepdim=True)
        )
        text_features = (
            text_features / text_features.norm(dim=-1, keepdim=True)
        )

        # local_image_features: [batch, height * width, dim]
        local_image_features = local_image_features.view(
            local_image_features.shape[0],-1, local_image_features.shape[-1]
        )

        # image_features_all: [batch, height * width + 1, output_dim]
        image_features_all =  torch.cat(
            [image_features.unsqueeze(1), local_image_features], dim=1
        )

        # output: [batch, n_cls * 3, height * width + 1]
        output = F.conv1d(
            image_features_all.transpose(1, 2),
            text_features[:, :, None]
        )
        B, n_cls_2, N = output.shape

        # output_half: [batch, n_cls, height * width + 1]
        output_half = output[:, n_cls_2 // 2:, :]

        # w_half: [batch, n_cls, height * width + 1]
        w_half = F.softmax(output_half, dim=-1)

        # w: [batch, 2 * n_cls, height * width + 1]
        w = torch.cat([w_half, w_half], dim=1)

        # output: [batch, 2 * n_cls]
        output = 5 * (output * w).sum(dim = -1)

        # logits: [batch, 2, n_cls]
        logits = torch.cat(
            (
                output[:, 0:n_cls_2 // 2].unsqueeze(1),
                output[:, n_cls_2//2:].unsqueeze(1),
            ), 
            dim=1
        )
        
        return logits

    def forward_feats(self, images):
        images = images.to(self.configs.device)

        # image_features: [batch, output_dim]
        # local_image_features: [batch, height, width, dim]
        # output_dim = 512, height = width = 7 for ViT-B/32
        image_features, local_image_features = (
            self.clip_model.visual.forward_feats(images)
        )

        # prompts: [n_cls * 2, context_len, output_dim]
        prompts = self.prompt_learner()[0].to(self.configs.device)
        tokenized_prompts = (
            self.prompt_learner
            .tokenized_prompts
            .to(self.configs.device)
        )
        # text_features: [n_cls * 2, output_dim]
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = (
            image_features / image_features.norm(dim=-1, keepdim=True)
        )
        local_image_features = (
            local_image_features /
            local_image_features.norm(dim=-1, keepdim=True)
        )
        text_features = (
            text_features / text_features.norm(dim=-1, keepdim=True)
        )

        # local_image_features: [batch, height * width, dim]
        local_image_features = local_image_features.view(
            local_image_features.shape[0], -1, local_image_features.shape[-1]
        )
        # image_features_all: [batch, height * width + 1, output_dim]
        image_features_all = torch.cat(
            [image_features.unsqueeze(1), local_image_features], dim=1
        )

        # output: [batch, n_cls * 2, height * width + 1]
        output = F.conv1d(
            image_features_all.transpose(1, 2),
            text_features[:, :, None],
        )
        # output_{pos/neg} = [batch, n_cls, height * width + 1]
        output_neg = output[:, :output.shape[1] // 2, :]
        output_pos = output[:, output.shape[1] // 2:, :]

        # output: [batch, n_cls, height * width + 1, 2]
        output = torch.stack((output_neg, output_pos), dim=-1)

        return output
