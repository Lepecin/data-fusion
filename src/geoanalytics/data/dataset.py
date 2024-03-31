import torch
import polars
from torch.utils.data import Dataset
from .sampler import FeatureSampler


class FeatureDataset(FeatureSampler, Dataset):
    def __init__(
        self,
        data: polars.DataFrame,
        categorical_features: list[str],
        continuous_features: list[str],
    ) -> None:
        super().__init__(data, categorical_features, continuous_features)
        self.customer_ids = self.get_customer_ids()

    def __len__(self):
        return self.customer_ids.len()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return super().__getitem__(self.customer_ids[index])
