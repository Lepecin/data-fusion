import torch
import torch.nn as nn

from .grn import GRN
from .varencoder import VariableEncoder


class VariableSelectionNetwork(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        continuous_features: list[str],
        categorical_features: dict[str, int],
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features

        self.encoder = VariableEncoder(
            hidden_size, continuous_features, categorical_features
        )

        self.features = list(categorical_features) + continuous_features

        self.feature_grns = nn.ModuleDict(
            {key: GRN(hidden_size) for key in self.features}
        )

        self.attention_grn = nn.Sequential(
            GRN(hidden_size * len(self.features)),
            nn.Linear(hidden_size * len(self.features), len(self.features)),
            nn.Softmax(-1),
        )

    def forward(
        self,
        categorical_input: torch.Tensor,
        continuous_input: torch.Tensor,
    ) -> torch.Tensor:
        encoding = self.encoder.forward(
            categorical_input, continuous_input
        )  # (N, S, F, E)

        variables = torch.concat(  # (N, S, F, E)
            [
                self.feature_grns[key].forward(
                    tensor,  # (N, S, 1, E)
                )
                for key, tensor in zip(self.features, torch.split(encoding, 1, -2))
            ],
            -2,
        )

        weights: torch.Tensor = self.attention_grn.forward(  # (N, S, F)
            encoding.flatten(-2, -1),  # (N, S, E*F)
        )

        return variables.mul(weights.unsqueeze(-1)).sum(-2)
