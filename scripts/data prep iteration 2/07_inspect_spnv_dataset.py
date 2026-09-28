from pathlib import Path

import pandas as pd


DATASET_FILE = "spnv_base_koeln_bonn_2025-11_2026-08.parquet"

STATION_EVA_MAP = {
    "Köln Hbf": ("08000207",),
    "Köln Messe/Deutz": ("08003368", "08073368"),
    "Köln Süd": ("08003361",),
    "Köln/Bonn Flughafen": ("08003330",),
    "Brühl": ("08001215",),
    "Bonn Hbf": ("08000044",),
    "Bonn-Beuel": ("08001083",),
    "Troisdorf": ("08000135",),
}

IMPORTANT_COLUMNS = [
    "id",
    "eva",
    "train_type",
    "line_number",
    "train_line_ride_id",
    "delay_in_min",
    "arrival_planned_time",
    "departure_planned_time",
]


def find_project_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "datasets").is_dir():
            return parent

    raise FileNotFoundError("Projektordner mit datasets-Verzeichnis nicht gefunden.")


PROJECT_DIR = find_project_dir()
DATASET_PATH = PROJECT_DIR / "datasets" / "processed" / DATASET_FILE
EVA_TO_STATION = {
    eva: station
    for station, eva_numbers in STATION_EVA_MAP.items()
    for eva in eva_numbers
}


def main() -> None:
    spnv = pd.read_parquet(DATASET_PATH)
    time_values = pd.to_datetime(spnv["time"], errors="coerce")

    unique_ids = spnv["id"].nunique(dropna=True)
    duplicate_ids = int(spnv["id"].dropna().duplicated().sum())

    station_names = spnv["eva"].astype("string").str.zfill(8).map(EVA_TO_STATION)
    station_counts = (
        station_names.value_counts()
        .rename_axis("station_name")
        .reset_index(name="observations")
        .sort_values("observations", ascending=False)
        .reset_index(drop=True)
    )

    train_type_counts = (
        spnv.groupby("train_type", dropna=False)
        .size()
        .reset_index(name="observations")
        .sort_values("observations", ascending=False)
        .reset_index(drop=True)
    )
    train_type_counts["train_type"] = train_type_counts["train_type"].fillna("<NA>")

    missing_values = (
        spnv[IMPORTANT_COLUMNS]
        .isna()
        .sum()
        .reset_index(name="missing")
        .rename(columns={"index": "column"})
    )
    missing_values["missing_percent"] = (
        missing_values["missing"] / len(spnv) * 100
    ).round(2)
    missing_values = missing_values.sort_values(
        ["missing", "column"],
        ascending=[False, True],
    ).reset_index(drop=True)

    print("SPNV-Datensatz")
    print(f"Beobachtungen: {len(spnv)}")
    print(f"Spalten: {len(spnv.columns)}")
    print(f"Zeitraum: {time_values.min()} bis {time_values.max()}")
    print(f"Eindeutige IDs: {unique_ids}")
    print(f"Doppelte IDs: {duplicate_ids}")

    print("\nStationen")
    print(station_counts.to_string(index=False))

    print("\nTrain Types")
    print(train_type_counts.to_string(index=False))

    print("\nFehlende Werte")
    print(missing_values.to_string(index=False))


if __name__ == "__main__":
    main()
