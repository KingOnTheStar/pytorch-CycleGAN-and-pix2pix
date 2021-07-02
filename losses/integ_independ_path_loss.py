import torch
import torch.nn as nn
import numpy as np
from data.complex_data_processing.integral_grad import *


class IntegIndepenPathLoss(nn.Module):
    def __init__(self) -> None:
        super(IntegIndepenPathLoss, self).__init__()

    def forward(self, input):
        grad_x_mtx, grad_y_mtx = IntegralGrad.grad_split(input)
        dPdy = grad_x_mtx[..., :, 1:, :-1] - grad_x_mtx[..., :, :-1, :-1]
        dQdx = grad_y_mtx[..., :, :-1, 1:] - grad_y_mtx[..., :, :-1, :-1]

        reduce_axes = (-3, -2, -1)
        res = (dPdy - dQdx).abs().sum(dim=reduce_axes)
        return res
