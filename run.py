import subprocess
import pathlib

directory = pathlib.Path(__file__).parent
python_path = directory / ".venv/bin/python"


subprocess.run(
    [
        str(python_path),
        directory / "src/geoanalytics/project/app.py",
        "-ht",
        directory / "data/geophysical/hexses_target.lst",
        "-hd",
        directory / "data/geophysical/hexses_data.lst",
        "-i",
        directory / "data/geophysical/moscow.parquet",
        "-o",
        directory / "data/output.parquet",
    ]
)
