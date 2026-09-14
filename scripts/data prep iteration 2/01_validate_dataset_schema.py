from pathlib import Path
import sys

import pandas as pd
import pyarrow.parquet as pq


MONTH_FILES = [
    "data-2026-01.parquet",
    "data-2026-02.parquet",
    "data-2026-03.parquet",
    "data-2026-04.parquet",
    "data-2026-05.parquet",
    "data-2026-06.parquet",
]

IMPORTANT_COLUMNS = [
    "station_name",
    "xml_station_name",
    "eva",
    "train_type",
    "train_name",
    "train_number",
    "line_number",
    "final_destination_station",
    "delay_in_min",
    "time",
    "is_canceled",
    "arrival_is_canceled",
    "departure_is_canceled",
    "train_line_ride_id",
    "train_line_station_num",
    "arrival_planned_time",
    "arrival_change_time",
    "departure_planned_time",
    "departure_change_time",
    "id",
]

SPECIAL_COLUMNS = ["train_name", "train_number", "line_number"]


def find_project_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "datasets").is_dir():
            return parent

    raise FileNotFoundError("Projektordner mit datasets-Verzeichnis nicht gefunden.")


PROJECT_DIR = find_project_dir()
DATASETS_DIR = PROJECT_DIR / "datasets"
MONTHLY_DATASETS_DIR = DATASETS_DIR / "monthly"


def month_from_filename(filename: str) -> str:
    return filename.removeprefix("data-").removesuffix(".parquet")


def print_heading(title: str) -> None:
    print(f"\n{title}")


def print_table(rows: list[dict[str, object]] | pd.DataFrame) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        print("Keine.")
    else:
        print(df.to_string(index=False))


def validate_input_files() -> list[Path]:
    files = [MONTHLY_DATASETS_DIR / filename for filename in MONTH_FILES]
    missing = [path.name for path in files if not path.exists()]

    if missing:
        print("Fehlende Parquet-Dateien:")
        for filename in missing:
            print(f"- {filename}")
        sys.exit(1)

    return files


def read_metadata(path: Path) -> dict[str, object]:
    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema_arrow

    return {
        "month": month_from_filename(path.name),
        "path": path,
        "columns": schema.names,
        "types": {field.name: str(field.type) for field in schema},
        "rows": parquet_file.metadata.num_rows,
        "file_size_mb": round(path.stat().st_size / (1024 * 1024), 2),
    }


def build_presence_table(columns: list[str], metadata: list[dict[str, object]]) -> pd.DataFrame:
    rows = []

    for column in columns:
        row = {"column": column}
        for item in metadata:
            row[item["month"]] = column in item["columns"]
        rows.append(row)

    return pd.DataFrame(rows)


def find_type_differences(metadata: list[dict[str, object]]) -> list[dict[str, object]]:
    all_columns = sorted({column for item in metadata for column in item["columns"]})
    rows = []

    for column in all_columns:
        types = {item["types"][column] for item in metadata if column in item["types"]}
        if len(types) <= 1:
            continue

        row = {"column": column}
        for item in metadata:
            row[item["month"]] = item["types"].get(column, "<fehlt>")
        rows.append(row)

    return rows


def find_partial_columns(metadata: list[dict[str, object]]) -> list[dict[str, object]]:
    all_columns = sorted({column for item in metadata for column in item["columns"]})
    rows = []

    for column in all_columns:
        present = [item["month"] for item in metadata if column in item["columns"]]
        missing = [item["month"] for item in metadata if column not in item["columns"]]

        if present and missing:
            rows.append(
                {
                    "column": column,
                    "present_in": ", ".join(present),
                    "missing_in": ", ".join(missing),
                }
            )

    return rows


def print_schema_summary(metadata: list[dict[str, object]]) -> None:
    print_heading("Schema je Datei")
    rows = [
        {
            "month": item["month"],
            "columns": len(item["columns"]),
            "column_names": ", ".join(item["columns"]),
        }
        for item in metadata
    ]
    print_table(rows)


def print_file_stats(metadata: list[dict[str, object]]) -> None:
    print_heading("Dateigroesse und Zeilen")
    rows = [
        {
            "month": item["month"],
            "rows": item["rows"],
            "columns": len(item["columns"]),
            "file_size_mb": item["file_size_mb"],
        }
        for item in metadata
    ]
    print_table(rows)


def main() -> None:
    files = validate_input_files()
    metadata = [read_metadata(path) for path in files]
    partial_columns = find_partial_columns(metadata)
    type_differences = find_type_differences(metadata)

    print_schema_summary(metadata)

    print_heading("Wichtige Spalten vorhanden")
    print_table(build_presence_table(IMPORTANT_COLUMNS, metadata))

    print_heading("Train- und Linien-Spalten")
    print_table(build_presence_table(SPECIAL_COLUMNS, metadata))

    print_heading("Spalten nur in einzelnen Monaten")
    print_table(partial_columns)

    print_heading("Spalten mit unterschiedlichen Datentypen")
    print_table(type_differences)

    print_file_stats(metadata)


if __name__ == "__main__":
    main()
