import torch
import polars


class FeatureSampler:
    customer_id = "customer_id"

    def __init__(
        self,
        data: polars.DataFrame,
        categorical_features: list[str],
        continuous_features: list[str],
    ) -> None:

        self.data = data
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features

        self.max_length: int = polars.select(
            data[self.customer_id].unique_counts().max()
        ).item()

    def pad_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        length = len(tensor)
        assert length != 0
        padding_size = self.max_length - length
        return torch.cat(
            [
                tensor,
                torch.zeros(padding_size, *tensor.size()[1:]).to(tensor),
            ]
        )

    def get_customer_ids(self):
        return self.data[self.customer_id].unique()

    def get_sample(self, customer_index: int) -> dict[str, torch.Tensor]:
        customer_data = self.data.filter(polars.col(self.customer_id) == customer_index)
        length = len(customer_data)

        sample = {
            "categorical": torch.from_numpy(
                customer_data[self.categorical_features].to_numpy()
            ),
            "continuous": torch.from_numpy(
                customer_data[self.continuous_features].to_numpy()
            ),
            "mask": torch.tensor([True] * length),
        }

        return {key: self.pad_tensor(tensor) for key, tensor in sample.items()}
