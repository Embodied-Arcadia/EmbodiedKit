import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossCE(nn.Module):
    """
    Plain Focal Loss (multi-class), optional class weights alpha
    - gamma=0, alpha=None  → same scale as nn.CrossEntropyLoss(mean)
    - gamma=0, alpha!=None → same scale as nn.CrossEntropyLoss(weight=alpha, mean)
    """
    def __init__(self, alpha=None, gamma=1.5, reduction="mean"):
        super().__init__()
        if alpha is not None and not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        # An empty tensor is used as a "no weights" signal
        self.register_buffer("alpha", alpha if alpha is not None else torch.tensor([]))
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits:  [B, C]
        targets: [B] (long)
        """
        targets = targets.long().to(logits.device)
        log_probs = F.log_softmax(logits, dim=1)                           # [B, C]
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)      # [B]
        pt = log_pt.exp().clamp_min(1e-8)                                   # [B]

        if self.alpha.numel() > 0:
            alpha_t = self.alpha.to(device=logits.device, dtype=logits.dtype)[targets]  # [B]
        else:
            alpha_t = torch.ones_like(pt)

        loss = -alpha_t * (1.0 - pt).pow(self.gamma) * log_pt               # [B]

        if self.reduction == "mean":
            # Match the scale of CE's "weighted mean": use the sum of weights as the denominator
            denom = (alpha_t.sum() if self.alpha.numel() > 0 else pt.new_tensor(len(pt)))
            return loss.sum() / (denom + 1e-12)
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
