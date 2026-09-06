from pathlib import Path

import pandas as pd


DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets"
MONTH_FILES = [
    "data-2025-12.parquet",
    "data-2026-01.parquet",
    "data-2026-02.parquet",
    "data-2026-03.parquet",
    "data-2026-04.parquet",
    "data-2026-05.parquet",
]
COLUMNS = [
    "station_name",
    "eva",
    "train_type",
    "time",
]


def main() -> None:
    files = [DATASETS_DIR / filename for filename in MONTH_FILES]

    missing_files = [file for file in files if not file.exists()]
    if missing_files:
        missing_names = ", ".join(file.name for file in missing_files)
        raise FileNotFoundError(f"Fehlende Parquet-Dateien: {missing_names}")

    for file in files:
        print("\n" + "=" * 70)
        print(file.name)

        df = pd.read_parquet(file, columns=COLUMNS)

        print("Zeilen:", len(df))
        print("Zeitraum:", df["time"].min(), "bis", df["time"].max())

        print("\nZugtypen:")
        with pd.option_context("display.max_rows", None):
            print(df["train_type"].value_counts(dropna=False))

        print("\nAnzahl Bahnhoefe:")
        print(df["station_name"].nunique())

        print("\nAnzahl EVA-Nummern:")
        print(df["eva"].nunique())

        del df


if __name__ == "__main__":
    main()
