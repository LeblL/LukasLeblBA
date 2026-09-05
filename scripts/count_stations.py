from pathlib import Path

import pandas as pd


DATASET_PATH = Path(__file__).resolve().parents[1] / "datasets" / "data-2026-01.parquet"


def main() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    df = pd.read_parquet(DATASET_PATH)
    print(df["station_name"].nunique())



if __name__ == "__main__":
    main()
