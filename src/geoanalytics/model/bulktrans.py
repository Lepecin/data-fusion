import torch
import torch.nn as nn

from .varselector import VariableSelectionNetwork

from dataclasses import dataclass, field


@dataclass
class BulkTransformerConfig:
    HIDDEN_SIZE: int
    HEADS: int
    DROPOUT: float
    CLASSES: int
    LAYERS: int
    FEEDFORWARD_SIZE: int
    CONTINUOUS_FEATURES: list[str] = field(default_factory=list)
    CATEGORICAL_FEATURES: list[str] = field(default_factory=list)
    CATEGORICAL_SIZES: list[int] = field(default_factory=list)


class BulkTransformerModel(nn.Module):
    def __init__(self, config: BulkTransformerConfig) -> None:
        super().__init__()

        self.encoder = VariableSelectionNetwork(
            config.HIDDEN_SIZE,
            config.CONTINUOUS_FEATURES,
            dict(zip(config.CATEGORICAL_FEATURES, config.CATEGORICAL_SIZES)),
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                config.HIDDEN_SIZE,
                config.HEADS,
                config.FEEDFORWARD_SIZE,
                config.DROPOUT,
                batch_first=True,
            ),
            config.LAYERS,
        )

        self.mlp_classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_SIZE, config.CLASSES),
            nn.Softmax(-1),
        )

    def forward(self, input: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = input["mask"]
        categorical_input = input["categorical"]
        continuous_input = input["continuous"]

        output = self.encoder.forward(categorical_input, continuous_input)
        transoutput = self.transformer.forward(
            output, src_key_padding_mask=mask.bitwise_not()
        )
        ave_output = (
            transoutput.mul(mask.unsqueeze(-1)).sum(1).div(mask.sum(1).unsqueeze(-1))
        )

        return self.mlp_classifier.forward(ave_output)
