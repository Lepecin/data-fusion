import h3
import polars

from .data import get_hexes


def get_feature_dataset(
    hexses_data_path: str,
    input_path: str,
) -> polars.DataFrame:

    data_hexes = get_hexes(hexses_data_path)

    coord_dataframe = polars.DataFrame(
        {
            "h3_09": data_hexes,
            "lat": map(lambda x: h3.h3_to_geo(x)[0], data_hexes),
            "lng": map(lambda x: h3.h3_to_geo(x)[1], data_hexes),
        }
    )

    data = (
        polars.read_parquet(input_path)
        .join(coord_dataframe, on="h3_09")
        .drop("h3_09")
        .fill_null(0)
        .sort("customer_id")
    )

    return data
