import torch
import torch.nn as nn
import torch.nn.functional as F

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        return self.bce(inputs, targets)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        prob = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = prob * targets + (1 - prob) * (1 - targets)
        focal_factor = (1 - p_t) ** self.gamma
        loss = self.alpha * focal_factor * ce_loss
        return loss.mean()
