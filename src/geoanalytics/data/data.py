def get_hexes(path: str) -> list[str]:
    with open(path, "r") as file:
        return [line.strip() for line in file.readlines()]


def get_full_hexes(data_path: str, target_path: str) -> list[str]:
    return list(set(get_hexes(target_path) + get_hexes(data_path)))
