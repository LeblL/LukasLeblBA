from pathlib import Path

import pandas as pd


DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets"
COLUMNS = [
    "station_name",
    "eva",
    "train_type",
    "time",
]


def main() -> None:
    files = sorted(DATASETS_DIR.glob("*.parquet"))

    if not files:
        print(f"Keine Parquet-Dateien gefunden in: {DATASETS_DIR}")
        return

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
