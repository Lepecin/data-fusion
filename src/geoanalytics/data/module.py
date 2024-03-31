import polars
from lightning import LightningDataModule
from torch.utils.data import DataLoader

import pathlib
from typing import Optional, Literal
from dataclasses import dataclass, field

from .dataset import FeatureDataset, TrainingDataset
from .processing import get_feature_dataset, get_target_dataset, get_hexes


@dataclass
class CustomDataConfig:
    hexses_data_path: str
    hexses_target_path: str
    input_path: str
    target_path: Optional[str] = None
    categorical_features: list[str] = field(default_factory=list)
    continuous_features: list[str] = field(default_factory=list)
    data_directory: str = (pathlib.Path(__file__).parent / "data-store").as_posix()
    fraction: float = 1.0
    seed: Optional[int] = None
    deploy: bool = False


class CustomDataModule(LightningDataModule):
    feature_data_filename = "feature-data.parquet"
    target_data_filename = "target-data.parquet"

    def __init__(
        self,
        config: CustomDataConfig,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.set_config(config)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def set_config(self, config: CustomDataConfig):
        self.config = config
        return self

    def prepare_data(self) -> None:
        directory = pathlib.Path(self.config.data_directory)
        directory.mkdir(exist_ok=True)
        self.feature_path = (directory / self.feature_data_filename).as_posix()
        self.target_path = (directory / self.target_data_filename).as_posix()

        get_feature_dataset(
            self.config.hexses_data_path,
            self.config.input_path,
        ).write_parquet(self.feature_path)

        if self.config.target_path is not None:
            get_target_dataset(
                self.config.hexses_target_path,
                self.config.target_path,
            ).write_parquet(self.target_path)

    def get_target_hexes(self) -> list[str]:
        return get_hexes(self.config.hexses_target_path)

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        feature_data = polars.read_parquet(self.feature_path)
        target_hexes = self.get_target_hexes()

        if self.config.target_path is not None:
            target_data = polars.read_parquet(self.target_path)

        if (stage == "validate") or (stage == "fit" and not self.config.deploy):
            customer_ids = feature_data["customer_id"].unique()
            train_ids = customer_ids.sample(
                fraction=self.config.fraction,
                seed=self.config.seed,
            )
            valid_ids = customer_ids.filter(customer_ids.is_in(train_ids))

            self.train = TrainingDataset(
                feature_data.filter(polars.col("customer_id").is_in(train_ids)),
                target_data.filter(polars.col("customer_id").is_in(train_ids)),
                target_hexes,
                self.config.categorical_features,
                self.config.continuous_features,
            )

            self.valid = TrainingDataset(
                feature_data.filter(polars.col("customer_id").is_in(valid_ids)),
                target_data.filter(polars.col("customer_id").is_in(valid_ids)),
                target_hexes,
                self.config.categorical_features,
                self.config.continuous_features,
            )
        elif stage == "fit" and self.config.deploy:
            self.train = TrainingDataset(
                feature_data,
                target_data,
                target_hexes,
                self.config.categorical_features,
                self.config.continuous_features,
            )
        elif stage == "predict" or stage == "test":
            self.test = FeatureDataset(
                feature_data,
                self.config.categorical_features,
                self.config.continuous_features,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )
