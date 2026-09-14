from pathlib import Path

import pandas as pd


MONTH_FILES = [
    "data-2026-01.parquet",
    "data-2026-02.parquet",
    "data-2026-03.parquet",
    "data-2026-04.parquet",
    "data-2026-05.parquet",
    "data-2026-06.parquet",
]

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

SPNV_TYPES = {"S", "RB", "RE", "NX", "TR", "TRI"}
COLUMNS = ["eva", "train_type"]


def find_project_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "datasets").is_dir():
            return parent

    raise FileNotFoundError("Projektordner mit datasets-Verzeichnis nicht gefunden.")


PROJECT_DIR = find_project_dir()
DATASETS_DIR = PROJECT_DIR / "datasets"
MONTHLY_DATASETS_DIR = DATASETS_DIR / "monthly"
EVA_TO_STATION = {
    eva: station
    for station, eva_numbers in STATION_EVA_MAP.items()
    for eva in eva_numbers
}


def load_station_observations(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=COLUMNS)
    df["eva"] = df["eva"].astype("string").str.zfill(8)

    filtered = df[df["eva"].isin(EVA_TO_STATION)].copy()
    filtered["station_name"] = filtered["eva"].map(EVA_TO_STATION)

    return filtered


def main() -> None:
    monthly_parts = []

    for filename in MONTH_FILES:
        path = MONTHLY_DATASETS_DIR / filename
        monthly_parts.append(load_station_observations(path))

    study_area = pd.concat(monthly_parts, ignore_index=True)
    is_spnv = study_area["train_type"].isin(SPNV_TYPES)

    train_type_counts = (
        study_area.groupby("train_type", dropna=False)
        .size()
        .reset_index(name="observations")
    )
    train_type_counts["status"] = train_type_counts["train_type"].isin(SPNV_TYPES).map(
        {True: "behalten", False: "ausschließen"}
    )
    train_type_counts["train_type"] = train_type_counts["train_type"].fillna("<NA>")
    train_type_counts = train_type_counts.sort_values(
        ["observations", "train_type"],
        ascending=[False, True],
    ).reset_index(drop=True)

    station_order = list(STATION_EVA_MAP)
    before = study_area.groupby("station_name").size().reindex(station_order, fill_value=0)
    after = (
        study_area[is_spnv]
        .groupby("station_name")
        .size()
        .reindex(station_order, fill_value=0)
    )
    station_counts = pd.DataFrame(
        {
            "station_name": station_order,
            "vorher": before.astype(int).to_numpy(),
            "nach_filter": after.astype(int).to_numpy(),
        }
    )

    observations_before = len(study_area)
    observations_after = int(is_spnv.sum())
    kept_share = observations_after / observations_before * 100 if observations_before else 0

    print("Train Types")
    print(train_type_counts.to_string(index=False))

    print("\nStationen")
    print(station_counts.to_string(index=False))

    print("\nKurzes Ergebnis")
    print(f"Beobachtungen vorher: {observations_before}")
    print(f"Beobachtungen nach SPNV-Filter: {observations_after}")
    print(f"Anteil behalten: {kept_share:.2f} %")


if __name__ == "__main__":
    main()
