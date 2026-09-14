from pathlib import Path

import pandas as pd


def find_project_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "datasets").is_dir():
            return parent

    raise FileNotFoundError("Projektordner mit datasets-Verzeichnis nicht gefunden.")


PROJECT_DIR = find_project_dir()
DATASETS_DIR = PROJECT_DIR / "datasets"
OLD_SCHEMA_DATASETS_DIR = DATASETS_DIR / "old schema datasets"
MONTH_FILES = [
    OLD_SCHEMA_DATASETS_DIR / "old-schema-data-2025-12.parquet",
    OLD_SCHEMA_DATASETS_DIR / "old-schema-data-2026-01.parquet",
    OLD_SCHEMA_DATASETS_DIR / "old-schema-data-2026-02.parquet",
    OLD_SCHEMA_DATASETS_DIR / "old-schema-data-2026-03.parquet",
    OLD_SCHEMA_DATASETS_DIR / "old-schema-data-2026-04.parquet",
    DATASETS_DIR / "data-2026-05.parquet",
]
COLUMNS = [
    "station_name",
    "eva",
    "train_type",
    "time",
]


def main() -> None:
    files = MONTH_FILES

    missing_files = [file for file in files if not file.exists()]
    if missing_files:
        missing_names = ", ".join(str(file.relative_to(PROJECT_DIR)) for file in missing_files)
        raise FileNotFoundError(f"Fehlende Parquet-Dateien: {missing_names}")

    for file in files:
        print("\n" + "=" * 70)
        print(file.relative_to(PROJECT_DIR))

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
