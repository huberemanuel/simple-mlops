from pathlib import Path

import boto3

from wine_platform.data import get_raw_path


def download_data(destiny_path: str):
    s3 = boto3.client("s3")
    file_name = "winequalityN.csv"
    output_path = str(Path(destiny_path).joinpath(file_name))
    s3.download_file("wine-data-dev", file_name, output_path)


if __name__ == "__main__":
    download_data(get_raw_path())
