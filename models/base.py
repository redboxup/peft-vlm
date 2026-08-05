import torch
import torch.nn as nn
from typing import Dict, Any

from clip.model import CLIP
from configs import ProjectConf

class BaseCLIP(nn.Module):
    registered_models: Dict[str, Any]
    def __init__(self, configs: ProjectConf, clip_model: CLIP):
        super().__init__()
        self.configs = configs
        self.clip_model = clip_model

    def forward(self, images: torch.Tensor):
        raise NotImplementedError("Forward method to be implemented in child class")

    def forward_feats(self, images: torch.Tensor):
        raise NotImplementedError("Forward method to be implemented in child class")

    @classmethod
    def register_model(cls, model_name: str):
        """Decorator to register custom CLIP models

        Args:
            method_name (str): Name of the clip method
        """
        def decorator(model_cls):
            cls.registered_models[model_name] = model_cls
            return model_cls
        return decorator

# register clip models
BaseCLIP.registered_models = {}