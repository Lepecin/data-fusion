import torch
import torch.nn as nn

from torch import Size
from typing import Union, List

_shape_t = Union[int, List[int], Size]


class MaskedLayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 0.00001,
        elementwise_affine: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.bias = bias
        self.elementwise_affine = elementwise_affine
        self.normalized_shape = normalized_shape

        self.bias = None
        self.weight = None
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape, requires_grad=True))
            if bias:
                self.bias = nn.Parameter(
                    torch.zeros(normalized_shape, requires_grad=True)
                )

        self.running_mean = torch.zeros(normalized_shape, requires_grad=True)
        self.running_variance = torch.zeros(normalized_shape, requires_grad=True)

    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mean = torch.zeros(self.normalized_shape, requires_grad=False)
        variance = torch.ones(self.normalized_shape, requires_grad=False)

        if mask.any():
            mean = input[mask].mean(0)
            variance = input[mask].var(0)

        input = (input - mean) / (self.eps + variance).sqrt()

        if self.elementwise_affine:
            input = input * self.weight + self.bias

        return input
