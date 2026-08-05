import torch
from torch.nn import functional as F
from torch import nn

class EntropySelectTopK(nn.Module):
    def __init__(self, top_k: int, num_of_local_feature: int):
        super().__init__()
        self.top_k = top_k # 10
        self.n_local_feats = num_of_local_feature # = 7

    def forward(self, p: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Extract non-Top-K regions and calculate entropy.
        """
        # p: [b, h*w, n_cls]) || # label: [b]
        b = torch.zeros_like(p[:, :, 0], dtype=torch.float32)

        for i in range(p.shape[0]):
            b[i] = p[i, :, label[i]]
        b = F.softmax(b, dim = -1)
        # b shape: [b, h*w]
        pred_topk_ids = torch.topk(b, k=self.top_k, dim=1)[1]
        # pred_topk_ids shape: [b, top_k]
        selected_p = [[] for _ in range(b.shape[0])]
        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                if j not in pred_topk_ids[i]:
                    selected_p[i].append(b[i, j])
        selected_p = torch.stack([torch.stack(s) for s in selected_p], dim=0)
        if selected_p.shape[0] == 0:
            return torch.tensor([0]).cuda()

        return -torch.mean(
            torch.sum(selected_p * torch.log(selected_p+1e-5), 1)
        )

