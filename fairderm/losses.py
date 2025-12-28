# custom loss functions

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    # focal loss - down-weights easy examples, focuses on hard ones

    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        # register alpha as buffer so it moves to GPU with the module
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def get_class_weights(df, label_col='label_num'):
    # inverse frequency weights - minority classes get higher weight
    class_counts = df[label_col].value_counts().sort_index()
    total = len(df)
    weights = total / (len(class_counts) * class_counts)
    return torch.tensor(weights.values, dtype=torch.float32)
