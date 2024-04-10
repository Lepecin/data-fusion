import subprocess
import pathlib
import sys

directory = pathlib.Path(__file__).parent

subprocess.run(
    [
        sys.executable,
        directory / "src/geoanalytics/naive_baseline/submit/run.py",
        "-ht",
        directory / "data/geophysical/hexses_target.lst",
        "-hd",
        directory / "data/geophysical/hexses_data.lst",
        "-i",
        directory / "data/bank/target.parquet",
        "-o",
        directory / "data/output.parquet",
    ]
)
