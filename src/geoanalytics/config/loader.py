import yaml
import pathlib
from typing import Any, TypeVar, Generic

T = TypeVar("T")


class YamlLoader(Generic[T]):

    def __init__(self, path: str) -> None:
        self.path = path

    def get_fields(self) -> dict[str, T]:
        with open(self.path, "r") as file:
            fields = yaml.load(file, yaml.CLoader)
        if not isinstance(fields, dict):
            return {}
        return fields

    def set_fields(self, **fields: T):
        with open(self.path, "w") as file:
            yaml.dump(fields, file, yaml.CDumper)

    def update_fields(self, **fields: T):
        old_fields = self.get_fields()
        old_fields.update(fields)
        with open(self.path, "w") as file:
            yaml.dump(old_fields, file, yaml.CDumper)

    def print_fields(self):
        for key, value in self.get_fields().items():
            print(f"{key}: {value}")


class ArgLoader(YamlLoader[Any]):
    def __init__(self) -> None:
        super().__init__(str(pathlib.Path(__file__).parent / "arguments.yaml"))
