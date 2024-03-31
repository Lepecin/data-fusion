import click


@click.command()
@click.option(
    "-ht",
    "--hexses-target-path",
    help="Список локаций таргета",
    type=str,
    required=True,
)
@click.option(
    "-hd",
    "--hexses-data-path",
    help="Список локаций транзакций",
    type=str,
    required=True,
)
@click.option(
    "-i",
    "--input-path",
    help="Входные данные",
    type=str,
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    help="Выходные данные",
    type=str,
    required=True,
)
def main(
    hexses_target_path: str,
    hexses_data_path: str,
    input_path: str,
    output_path: str,
):
    print(hexses_target_path)
    print(hexses_data_path)
    print(input_path)
    print(output_path)


if __name__ == "__main__":
    main()
