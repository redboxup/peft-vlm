import torch


def deterministic_randperm(length: int, seed: int = 42):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randperm(n=length, generator=g)
