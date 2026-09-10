import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY

_reduction_modes = ['none', 'mean', 'sum']


@LOSS_REGISTRY.register()
class FFTLoss(nn.Module):
    """Frequency-domain FFT loss: L1 on the 2D FFT magnitude.

    The FFT uses ``norm='ortho'`` so that the magnitude scale is independent
    of the patch size, keeping the loss weight consistent across different
    crop shapes. Only the magnitude is penalized; the phase is not matched.

    Args:
        loss_weight (float): Loss weight for FFT loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTLoss, self).__init__()
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
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        return self.loss_weight * F.l1_loss(
            torch.abs(pred_fft), torch.abs(target_fft), reduction=self.reduction)
