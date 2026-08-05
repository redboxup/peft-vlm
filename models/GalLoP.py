import torch
import math 
from clip import clip
import numpy as np
import torch.nn as nn
from torch import Tensor
from clip.model_gallop import CLIP
import torchvision

from copy import deepcopy
import torch.nn.functional as F
import itertools

from models import BaseCLIP
from configs import ProjectConf
from typing import Type, Any, Dict, Optional, Union, List, Tuple
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

NoneType = Type[None]
KwargType = Dict[str, Any]

class Linear(nn.Module):
    def __init__(self, in_dim: int, identity_init: bool = True) -> NoneType:
        super().__init__()
        self.linear = nn.Linear(in_dim, in_dim, bias=False)
        if identity_init:
            nn.init.zeros_(self.linear.weight)
            self.linear.weight.data += torch.eye(in_dim)
        else:
            nn.init.normal_(self.linear.weight, std=in_dim**-0.5)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

TRAINABLE_PARAMS: List[str] = []
class PromptLearner(nn.Module):
    def __init__(self, configs: ProjectConf, clip_model: CLIP):
        super().__init__()
        self.configs: ProjectConf = configs
        self.device: str  = configs.device
        self.n_cls: int  = len(configs.dataset.cxr_labels)
        self.n_ctx: int = configs.trainer.N_CTX
        self.ctx_init: str = configs.trainer.CTX_INIT
        self.class_names : List[str] = configs.dataset.cxr_labels
        self.learn_local_proj: bool = configs.trainer.learn_local_proj
        self.learn_local_prompt: bool = configs.trainer.learn_local_prompt
        self.learn_global_prompt: bool = configs.trainer.learn_global_prompt
        self.n_global_prompts: int = configs.trainer.n_global_prompts
        self.n_local_prompts: int = configs.trainer.n_local_prompts
        self.prompts_batch_size: int = configs.trainer.prompts_batch_size
        self.clip_model: CLIP = clip_model
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.clip_imsize = clip_model.visual.input_resolution
        self.cfg_imsize = configs.trainer.img_size
        assert self.cfg_imsize == self.clip_imsize,(
            f"cfg_imsize ({self.cfg_imsize}) must"
            f" equal to clip_imsize ({self.clip_imsize})"
        )
        if self.ctx_init:
            ctx_init = self.ctx_init.replace("{}", " ").replace("_", " ").strip()
            tokenized_ctx = clip.tokenize(ctx_init)
            self.n_ctx = int(
                tokenized_ctx.argmax(dim=-1)) - 1
            if self.learn_global_prompt or self.n_global_prompts > 1:
                if self.learn_global_prompt:
                    TRAINABLE_PARAMS.append('prompt_learner.global_prompts')
                self.global_prompts = nn.Parameter(
                    torch.empty(
                        (self.n_global_prompts, self.n_ctx, self.ctx_dim),
                        device=self.device
                    )
                )

            if self.learn_local_prompt or self.n_global_prompts > 1:
                if self.learn_local_prompt:
                    TRAINABLE_PARAMS.append('prompt_learner.local_prompts')
                self.local_prompts = nn.Parameter(
                    torch.empty(
                        (self.n_local_prompts, self.n_ctx, self.ctx_dim),
                        device=self.device
                    )
                )
            print(f"Initializing {self.n_ctx} context vectors with: {ctx_init}")
        else:
            print(f"Initializing {self.n_ctx} context vectors randomly")
            if self.learn_global_prompt or self.n_global_prompts > 1:
                if self.learn_global_prompt:
                    TRAINABLE_PARAMS.append('prompt_learner.global_prompts')
                self.global_prompts = nn.Parameter(torch.empty(
                    (self.n_global_prompts, self.n_ctx, self.ctx_dim),
                    device=self.device
                    )
                )
                self.global_prompts = nn.init.normal_(self.global_prompts, std=0.01)

            if self.learn_local_prompt or self.n_global_prompts > 1:
                if self.learn_local_prompt:
                    TRAINABLE_PARAMS.append('prompt_learner.local_prompts')
                self.local_prompts = nn.Parameter(torch.empty(
                    (self.n_local_prompts, self.n_ctx, self.ctx_dim),
                    device=self.device
                    )
                )
                self.local_prompts = nn.init.normal_(self.local_prompts, std=0.01)

    def forward(self):
        # this is for the case when we don't want tp learn any prompts
        if not self.learn_global_prompt and not self.learn_local_prompt:
            text_features = self.default_encode_text(self.class_names)
            # [n_cls, 1, embed_dim]
            return text_features, text_features
        
        tokenized_text = clip.tokenize(self.class_names).cuda(non_blocking=True)
        eot_tokens = tokenized_text.argmax(dim=-1)

        with torch.no_grad():
            token_embeddings = self.clip_model.token_embedding(tokenized_text)

        prefix = token_embeddings[:, :1, :]
        suffix = token_embeddings[:, 1 : -self.n_ctx, :]

        if self.learn_global_prompt or self.n_global_prompts > 1:
            global_prompts = self.global_prompts
            if self.prompts_batch_size < self.n_global_prompts and self.training:
                idx_select = torch.randperm(self.n_global_prompts
                            )[: self.prompts_batch_size] 
                # we don't want to do this for local prompts
                global_prompts = self.global_prompts[idx_select]

            text_features = self._most_efficient_encode_text(prefix,
                                global_prompts, suffix, eot_tokens)
        else:
            text_features = self.default_encode_text(self.class_names)

        if self.learn_local_prompt or self.n_local_prompts > 1:
            local_text_features = self._most_efficient_encode_text(prefix,
                                self.local_prompts, suffix, eot_tokens)
        else:
            local_text_features = text_features
        # [n_cls, prompts_batch_size, embed_dim]
        # [n_cls, local_prompts, embed_dim]
        return text_features, local_text_features 
    
    def _encode_text(self, prefix: Tensor, prompt: Tensor, suffix: Tensor, eot_tokens: Tensor) -> Tensor:
        x = torch.cat([prefix, prompt, suffix], dim=1)
        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        # x = x.permute(1, 0, 2)  # NLD -> LND  # This is not needed as we are using batch_first=True
        # x, padding = self.pad_if_necessary(x)
        x, *_ = self.clip_model.transformer(x, batch_first=True)
        # x = self.unpad_if_necessary(x, padding)
        # x = x.permute(1, 0, 2)  # LND -> NLD  # This is not needed as we are using batch_first=True
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), eot_tokens + self.n_ctx] @ self.clip_model.text_projection # [n_cls * n_prompt, embed_dim]
        return x
    
    def pad_if_necessary(self, x: Tensor) -> Tensor:
        if not self.parallel_text_encoder:
            return x, 0

        n = x.size(0)
        if n % self.num_devices == 0:
            return x, 0

        pad = self.num_devices - n % self.num_devices
        return torch.cat([x, torch.zeros(pad, *x.shape[1:], device=x.device)], dim=0), pad
    
    def unpad_if_necessary(self, x: Tensor, pad: int) -> Tensor:
        if pad == 0:
            return x

        return x[:-pad]

    def _single_forward_encode_text(self, prefix: Tensor, prompts: Tensor, suffix: Tensor, eot_tokens: Tensor) -> Tensor:
        n_prompts = prompts.size(0)
        n_classes = prefix.size(0)
        text_features = self._encode_text(
            prefix.repeat_interleave(n_prompts, dim=0),
            prompts.repeat(n_classes, 1, 1),
            suffix.repeat_interleave(n_prompts, dim=0),
            eot_tokens.repeat_interleave(n_prompts),
        )
        text_features = text_features.unflatten(0, (n_classes, n_prompts))
        # [n_cls, ?, embed_dim] ? == prompts_batch_size for global prompt or local_prompts for local prompt
        return text_features

    def default_encode_text(self, class_names: List[str]) -> Tensor:
        if self.ctx_init:
            prompts = [self.ctx_init.format(name) for name in class_names]
        else:
            prompts = class_names
        tokenized_text = clip.tokenize(prompts).cuda(non_blocking=True)
        text_features = self.clip_model.encode_text(tokenized_text, batch_first=True) #[14, 512]

        return text_features.unsqueeze(1)


    def _most_efficient_encode_text(self, prefix: Tensor, prompts: Tensor, suffix: Tensor, eot_tokens: Tensor) -> Tensor:
        return self._single_forward_encode_text(prefix, prompts, suffix, eot_tokens)

@BaseCLIP.register_model("gallop")
class GalLoPCLIP(BaseCLIP):
    def __init__(self, configs, clip_model: CLIP):
        super().__init__(configs, clip_model)
        self.prompt_learner = PromptLearner(configs, clip_model)
        self.clip_model = clip_model
        self.optim_params = self.prompt_learner.parameters()
        self.learn_local_proj = configs.trainer.learn_local_proj
        self.local_proj =  Linear(clip_model.visual.proj.shape[0],
                            configs.trainer.identity_init)
        if self.learn_local_proj:
            TRAINABLE_PARAMS.append('local_proj.linear.weight')
            self.optim_params = itertools.chain(
                self.optim_params, self.local_proj.parameters()
            )

        self.trainable_params = TRAINABLE_PARAMS
        self.logit_scale = clip_model.logit_scale

    def forward(self, images: Tensor) -> Tensor:
        images = images.to(self.configs.device)
        image_features, local_features = self.encode_image_and_proj(
            images
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        local_features = local_features / local_features.norm(dim=-1, keepdim=True)
        # [batch_size, embed_dim]
        # [batch_size, grid ** 2, embed_dim]

        text_features, local_text_features = self.prompt_learner()
        # [n_cls, local_prompts, embed_dim]
        # [n_cls, prompts_batch_size, embed_dim]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        global_logits = torch.einsum("bd, kmd -> bkm",
                                     image_features, text_features)
        local_logits = torch.einsum("bpd, knd -> bpkn", local_features,
                                     local_text_features)
        # [batch_size, n_cls, prompts_batch_size]
        # [batch_size, grid**2, n_cls, local_prompts]
        global_logits = global_logits * self.logit_scale.exp()
        local_logits = local_logits * self.logit_scale.exp()
        
        return global_logits, local_logits

    def encode_image_and_proj(self, image: Tensor) -> Tuple[Tensor, Tensor]:
        visual = self.clip_model.visual
        image_features, local_features = visual(image)

        if self.learn_local_proj:
            local_features = self.local_proj(local_features)

        if hasattr(visual, "proj"):
            image_features = image_features @ visual.proj
            local_features = local_features @ visual.proj

        return image_features, local_features
