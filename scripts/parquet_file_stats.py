from pathlib import Path

import pandas as pd


DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets"
STATION_COLUMN = "station_name"


def main() -> None:
    parquet_files = sorted(DATASETS_DIR.glob("*.parquet"))

    if not parquet_files:
        print(f"No parquet files found in {DATASETS_DIR}")
        return

    print(f"{'file':<24} {'rows':>12} {'unique_stations':>16}")
    print("-" * 54)

    for path in parquet_files:
        df = pd.read_parquet(path, columns=[STATION_COLUMN])
        rows = len(df)
        unique_stations = df[STATION_COLUMN].nunique()
        print(f"{path.name:<24} {rows:>12} {unique_stations:>16}")


if __name__ == "__main__":
    main()
