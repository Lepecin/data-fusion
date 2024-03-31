import torch
import polars
from torch.utils.data import Dataset
from .sampler import FeatureSampler, TargetSampler


class FeatureDataset(Dataset):
    def __init__(
        self,
        data: polars.DataFrame,
        categorical_features: list[str],
        continuous_features: list[str],
    ) -> None:

        self.feature_sampler = FeatureSampler(
            data, categorical_features, continuous_features
        )
        self.customer_ids = self.feature_sampler.get_customer_ids()

    def __len__(self):
        return self.customer_ids.len()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.feature_sampler.get_sample(self.customer_ids[index])


class TrainingDataset(Dataset):
    def __init__(
        self,
        feature_data: polars.DataFrame,
        target_data: polars.DataFrame,
        target_hexes: list[str],
        categorical_features: list[str],
        continuous_features: list[str],
    ) -> None:

        self.feature_sampler = FeatureSampler(
            feature_data, categorical_features, continuous_features
        )
        self.target_sampler = TargetSampler(target_data, target_hexes)
        self.customer_ids = self.feature_sampler.get_customer_ids()
        assert self.customer_ids.is_in(self.target_sampler.get_customer_ids()).all()

    def __len__(self):
        return self.customer_ids.len()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return dict(
            **self.feature_sampler.get_sample(self.customer_ids[index]),
            **self.target_sampler.get_sample(self.customer_ids[index]),
        )
