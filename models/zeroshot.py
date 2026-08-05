import torch

from clip import clip
from clip.model import CLIP
from models import BaseCLIP
from configs import ProjectConf
from clip import clip
import torchvision
import pandas as pd

@BaseCLIP.register_model("zeroshot")
class ZeroShotCLIP(BaseCLIP):
    def __init__(self, configs: ProjectConf, clip_model: CLIP):
        super().__init__(configs, clip_model)
        self.clip_model: CLIP
        self.configs: ProjectConf
        self.optim_params = None

    def forward(self, images: torch.Tensor):
        logit_scale = self.clip_model.logit_scale.exp()
        image_features = self.clip_model.visual(images.type(self.clip_model.dtype))
        # text features
        tokenized_text = clip.tokenize([self.configs.trainer.pos_context.format(classname) for classname in self.configs.dataset.cxr_labels])

        text_features = self.clip_model.encode_text(tokenized_text.to(self.configs.device))

        negative_tokenized_text = clip.tokenize([self.configs.trainer.neg_context.format(classname) for classname in self.configs.dataset.cxr_labels])
        negative_text_features = self.clip_model.encode_text(negative_tokenized_text.to(self.configs.device))

        # normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        negative_features = negative_text_features / negative_text_features.norm(dim=-1, keepdim=True)

        logits = logit_scale * image_features @ text_features.t()
        negative_logits = logit_scale * image_features @ negative_features.t()
        return logits, negative_logits

    def forward_feats(self, images):
        image_features, local_image_features = self.clip_model.encode_image_with_feats(images)
        # text features
        tokenized_text = clip.tokenize([classname for classname in self.configs.dataset.cxr_labels])
        text_features = self.clip_model.encode_text(tokenized_text.to(self.configs.device))

        # normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        local_image_features = local_image_features / local_image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # compute logits
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        local_logits = logit_scale * local_image_features @ text_features.t()
        return logits, local_logits