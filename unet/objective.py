import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLossWithWeights(nn.Module):
    """
    Simple implementation of the weighted per-pixel cross-entropy loss.
    Described in the paper "U-Net: Convolutional Networks for Biomedical Image Segmentation".

    Reference:
    U-Net: Convolutional Networks for Biomedical Image Segmentation, Ronneberger, Brox and Fischer (2015) 
    https://lmb.informatik.uni-freiburg.de/Publications/2015/RFB15a/
    """ 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 

    def forward(self, logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor|None=None) -> torch.Tensor:
        # logits: [1, 2, 388, 388], target: [1, 388, 388]
        cross_entropy = F.cross_entropy(logits, target, reduction='none')
        if weight is not None: 
            cross_entropy = weight * cross_entropy
        return cross_entropy.mean()