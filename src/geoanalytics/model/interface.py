import torch
import polars
from lightning import LightningModule

from .bulktrans import BulkTransformerModel, BulkTransformerConfig


class BulkTransformerInterface(LightningModule):
    def __init__(self, config: BulkTransformerConfig) -> None:
        super().__init__()

        self.config = config
        self.model = BulkTransformerModel(config)

    def forward(self, input: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model.forward(input)

    @staticmethod
    def loss_function(output: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-8):
        return (
            torch.log(torch.clip(output, eps, 1 - eps))
            .mul(target)
            .add(torch.log(torch.clip(1 - output, eps, 1 - eps)).mul(1 - target))
            .neg()
            .sum(1)
            .mean()
        )

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        loss = self.loss_function(self.forward(batch), batch["target"])
        self.log("train_loss", loss.detach().item())
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor]):
        loss = self.loss_function(self.forward(batch), batch["target"])
        self.log("val_loss", loss.detach().item())

    def predict_step(self, batch: dict[str, torch.Tensor]):
        return (
            polars.DataFrame(
                self.forward(batch).detach().numpy(),
                schema=self.config.TARGET_CLASSES,
            )
            .with_columns(customer_id=batch["customer_id"].numpy())
            .select("customer_id", polars.col("*").exclude("customer_id"))
        )
