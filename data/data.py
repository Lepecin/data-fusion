import pathlib
import polars

directory = pathlib.Path(__file__).parent


def get_hexes_data() -> list[str]:
    hexes_data_path = directory / "geophysical/hexses_data.lst"
    with open(hexes_data_path, "r") as file:
        return [line.strip() for line in file.readlines()]


def get_hexes_target() -> list[str]:
    hexes_target_path = directory / "geophysical/hexses_target.lst"
    with open(hexes_target_path, "r") as file:
        return [line.strip() for line in file.readlines()]


def get_bank_data() -> polars.DataFrame:
    bank_data_path = directory / "bank/transactions.parquet"
    return polars.read_parquet(bank_data_path)


def get_bank_target() -> polars.DataFrame:
    bank_target_path = directory / "bank/target.parquet"
    return polars.read_parquet(bank_target_path)


def get_moscow_data() -> polars.DataFrame:
    moscow_data = directory / "geophysical/moscow.parquet"
    return polars.read_parquet(moscow_data)


if __name__ == "__main__":
    print(get_bank_data())
