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

        sample = {
            "categorical": torch.from_numpy(
                customer_data[self.categorical_features].to_numpy()
            ).int(),
            "continuous": torch.from_numpy(
                customer_data[self.continuous_features].to_numpy()
            ).float(),
            "mask": torch.tensor([True] * len(customer_data)).bool(),
        }

        sample = {key: self.pad_tensor(tensor) for key, tensor in sample.items()}
        sample["customer_id"] = torch.tensor(customer_index).int()

        return sample


class TargetSampler:
    customer_id = "customer_id"
    location_id = "location_id"

    def __init__(
        self,
        data: polars.DataFrame,
        target_hexes: list[str],
    ) -> None:

        self.data = data
        self.target_hexes = target_hexes

    def get_customer_ids(self):
        return self.data[self.customer_id].unique()

    def get_sample(self, customer_index: int) -> dict[str, torch.Tensor]:

        label = torch.zeros(len(self.target_hexes)).float()

        indices = self.data.filter(polars.col(self.customer_id) == customer_index)[
            self.location_id
        ].to_list()

        label[indices] = 1.0

        return {"target": label}
