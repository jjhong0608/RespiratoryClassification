from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, *, temperature: float, normalize: bool):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError(
                f"Expected features with shape (B, V, D), got {tuple(features.shape)}"
            )
        if features.shape[1] < 2:
            raise ValueError("Supervised contrastive loss requires at least 2 views")
        if labels.ndim != 1:
            raise ValueError(
                f"Expected labels with shape (B,), got {tuple(labels.shape)}"
            )
        if labels.shape[0] != features.shape[0]:
            raise ValueError("labels batch size must match features batch size")

        if self.normalize:
            features = F.normalize(features, dim=-1)

        batch_size, n_views, _ = features.shape
        contrast = torch.cat(torch.unbind(features, dim=1), dim=0)
        logits = contrast @ contrast.T
        logits = logits / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        label_mask = labels.view(-1, 1).eq(labels.view(1, -1)).to(logits.dtype)
        mask = label_mask.repeat(n_views, n_views)
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0.0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        positive_counts = mask.sum(dim=1)
        if torch.any(positive_counts == 0):
            raise ValueError("Each anchor must have at least one positive example")
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / positive_counts
        return -mean_log_prob_pos.mean()
