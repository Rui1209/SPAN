import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY

_reduction_modes = ['none', 'mean', 'sum']


@LOSS_REGISTRY.register()
class GradientLoss(nn.Module):
    """Gradient loss: L1 on the first-order image gradients in x and y.

    Args:
        loss_weight (float): Loss weight for gradient loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(GradientLoss, self).__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
        pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
        target_dy = target[..., 1:, :] - target[..., :-1, :]
        target_dx = target[..., :, 1:] - target[..., :, :-1]
        return self.loss_weight * (F.l1_loss(pred_dy, target_dy, reduction=self.reduction) +
                                   F.l1_loss(pred_dx, target_dx, reduction=self.reduction))
