import torch
import torch.nn as nn


class VariableEncoder(nn.Module):
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

        self.continuous_encoder = nn.ModuleDict(
            {key: nn.Linear(1, hidden_size, False) for key in continuous_features}
        )

        self.categorical_encoder = nn.ModuleDict(
            {
                key: nn.Embedding(categorical_size, hidden_size)
                for key, categorical_size in categorical_features.items()
            }
        )

    def forward(self, input: dict[str, torch.Tensor]) -> torch.Tensor:
        categorical_output = torch.concat(
            [
                self.categorical_encoder[key].forward(tensor)
                for tensor, key in zip(
                    torch.split(input["categorical"], 1, -1), self.categorical_features
                )
            ],
            -2,
        )
        continuous_output = torch.concat(
            [
                self.continuous_encoder[key].forward(tensor.unsqueeze(-2))
                for tensor, key in zip(
                    torch.split(input["continuous"], 1, -1), self.continuous_features
                )
            ],
            -2,
        )

        return torch.concat([categorical_output, continuous_output], -2)
