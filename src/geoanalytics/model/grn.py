import torch
import torch.nn as nn

from typing import Optional


class GLU(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()

        self.linear = nn.Linear(size, size)
        self.sigmoid_linear = nn.Sequential(
            nn.Linear(size, size),
            nn.Sigmoid(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.linear.forward(input).mul(self.sigmoid_linear.forward(input))


class GRN(nn.Module):
    def __init__(self, size: int, context_size: Optional[int] = None):
        super().__init__()

        self.linear = nn.Linear(size, size)
        self.sequence = nn.Sequential(
            nn.ELU(),
            nn.Linear(size, size),
            GLU(size),
        )
        self.norm = nn.LayerNorm(size)

        self.linear_context = None
        if context_size is not None:
            self.linear_context = nn.Linear(context_size, size, False)

    def forward(
        self, input: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        transformed = self.linear.forward(input)
        if context is not None and self.linear_context is not None:
            transformed = transformed.add(self.linear_context.forward(context))

        return self.norm.forward(input.add(self.sequence.forward(transformed)))
