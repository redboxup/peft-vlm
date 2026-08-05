import torch
import torch.nn as nn

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
        n_cls = len(configs.dataset.cxr_labels)
        n_prompt =  configs.trainer.N_Prompt
        n_ctx = configs.trainer.N_CTX
        ctx_init = configs.trainer.CTX_INIT
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = configs.trainer.img_size
        self.configs = configs
        assert cfg_imsize == clip_imsize, (
            f"cfg_imsize ({cfg_imsize}) must equal to"
            f" clip_imsize ({clip_imsize})"
        )

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = (
                    clip_model.token_embedding(prompt)
                    .to(self.device)
                )
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initializaton
            if configs.trainer.CSC:
                if configs.verbose:
                    print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, device=self.device)
            else:
                if configs.verbose:
                    print("Initializing a generic context")
                ctx_vectors = torch.empty(n_prompt, n_ctx, ctx_dim, device=self.device)
            
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(['X'] * n_ctx)

        print(f"Initial context: '{prompt_prefix}'")
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors) # to be optimized

        classnames = [name.replace("_", " ") for name in 
                      configs.dataset.cxr_labels
                      ]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p).to(self.device) for p in prompts])
        # [n_cls, 77]
        if not configs.trainer.CSC:
            tokenized_prompts = tokenized_prompts.repeat(n_prompt, 1)
            # [n_cls * n_prompt, 77]
        # else [n_cls, 77]

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts)

        self.register_buffer(
            "token_prefix", embedding[:, :1, :].to(self.device)
        ) # SOS
        self.register_buffer(
            "token_suffix", embedding[:, 1 + n_ctx :, :].to(self.device)
        ) # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.n_prompt = n_prompt
        self.tokenized_prompts = tokenized_prompts.to(self.device)
        self.name_lens = name_lens
        self.class_token_position = configs.trainer.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        # print(f"ctx shape {ctx.shape} || ctx.dim {ctx.ndim} || csc: { self.configs.trainer.CSC}")
        if ctx.ndim == 3 and not self.configs.trainer.CSC:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
            # [n_prompt, n_ctx, ctx_dim] --> [n_cls, n_prompt, n_ctx, ctx_dim]
            # print(f"ctx:  {ctx.shape}")
            ctx = ctx.permute(1, 0, 2, 3) 
            ctx = ctx.contiguous().view(self.n_prompt*self.n_cls,self.n_ctx,ctx.shape[3])
            # [n_prompt, n_cls, n_ctx, ctx_dim] --> [n_prompt*n_cls, n_ctx, ctx_dim]
        # else the ctx shape: [n_cls, n_ctx, ctx_dim]
        prefix = self.token_prefix
        suffix = self.token_suffix
        # print(f"prefix: {prefix.shape} || ctx: {ctx.shape} || suffix: {suffix.shape} ")

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix, # (n_cls, 1, dim)
                    ctx,    # (n_cls, n_ctx, dim)
                    suffix, # (n_cls, *, dim)
                ],
                dim = 1,
            )

        return prompts # [n_cls, 77, ctx_dim] or [n_prompt*n_cls, 77, ctx_dim]

@BaseCLIP.register_model("plot")
class PLOTCLIP(BaseCLIP):
    def __init__(self, configs: ProjectConf, clip_model: CLIP):
        super().__init__(configs, clip_model)
        self.prompt_learner = PromptLearner(configs, clip_model)
        self.text_encoder = TextEncoder(clip_model, device=configs.device)
        self.optim_params = self.prompt_learner.parameters()
        self.configs = configs
        self.logit_scale = clip_model.logit_scale
        self.eps = 1e-1
        self.max_iter = 100
        n_cls = len(configs.dataset.cxr_labels)
        self.ln_layer1 = nn.Linear(n_cls, n_cls)

        # with torch.no_grad():
        #     self.ln_layer1.weight.copy_(torch.eye(n_cls))

        # import sys 
        # sys.exit(0)


    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        K = -K
        for i in range(self.max_iter):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break

        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

        return T

    def forward(self, images):
        images = images.to(self.configs.device)

        prompts = self.prompt_learner().to(self.configs.device) 
        # [n_cls, 77, ctx_dim] or [n_prompt*n_cls, 77, ctx_dim]
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        # [n_cls, 77, ctx_dim] or [n_prompt*n_cls, 77, ctx_dim]

        # print(f"prompts shape: {prompts.shape}")
        text_features = self.text_encoder(prompts, tokenized_prompts)
        # [n_cls, ctx_dim] or [n_prompt*n_cls, ctx_dim]

        # get the enseble of prompt
        text_features = text_features.contiguous().view(
            self.configs.trainer.N_Prompt,
            len(self.configs.dataset.cxr_labels), -1)
        # [n_prompt*n_cls, ctx_dim] --> [n_prompt, n_cls, ctx_dim]

        # print(f"Text features shape: {text_features.shape}")

        text_feature_pool = text_features.mean(dim=0)
        # [n_cls,ctx_dim] all prompts are averaged

        text_features = (
            text_features / text_features.norm(dim=-1, keepdim=True)
        )

        text_feature_pool = (
            text_feature_pool / text_feature_pool.norm(dim=-1, keepdim=True)
        )


        image_feature, local_image_features = (
            self.clip_model.visual.forward_feats(images)
        )
        # [B, dim] || [B, rdim, cdim, dim]
        image_feature = (
            image_feature / image_feature.norm(dim=-1, keepdim=True)
        )
        local_image_features = (
            local_image_features / local_image_features.norm(
                dim = -1, keepdim=True
            )
        )

        local_image_features = local_image_features.contiguous().view(
            local_image_features.shape[0], -1, local_image_features.shape[-1]
        )
        #[B, rdim*cdim, dim]
        local_image_features = local_image_features.permute(1,0,2)
        m, b, d = local_image_features.shape
        n, c, _ = text_features.shape

        # print(f"local image features || {local_image_features.shape} || text_features {text_features.shape}")

        # m -> num_patches, b->batch_size, c->num_cls, n->num_prompt, d -> ctx_dim
        sim = torch.einsum("mbd,ncd->mbnc", local_image_features, text_features)
        # [m, B, n, c]
        # print(f"sim shape {sim.shape}")
        sim = sim.permute(0, 2 , 1, 3)
        # [m, n, B, c]
        sim = sim.contiguous().view(sim.shape[0],sim.shape[1], sim.shape[2]*sim.shape[3])
        # [m, n, B*c]
        sim = sim.permute(2, 0, 1)
        # [B*c, m, n]
        wdist = 1.0 - sim

        xx=torch.zeros(b*c, m, dtype=sim.dtype, device=sim.device).fill_(1. / m)
        yy=torch.zeros(b*c, n, dtype=sim.dtype, device=sim.device).fill_(1. / n)   

        with torch.no_grad():
            KK = torch.exp(wdist / self.eps)
            T = self.Sinkhorn(KK,xx,yy)
        if torch.isnan(T).any():
            return None

        sim_op = torch.sum(T*sim, dim=(1,2))
        sim_op = sim_op.contiguous().view(b, c)

        logit_scale = 1 # self.logit_scale.exp()
        logits = logit_scale * image_feature @ text_feature_pool.t()
        logits2 = logit_scale * sim_op

        logits2 = logits + logits2 # [B, c]
        # logits = self.ln_layer1(logits2)

        return logits2

