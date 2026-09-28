from pathlib import Path

import pandas as pd


MONTH_FILES = [
    "data-2025-11.parquet",
    "data-2025-12.parquet",
    "data-2026-01.parquet",
    "data-2026-02.parquet",
    "data-2026-03.parquet",
    "data-2026-04.parquet",
    "data-2026-05.parquet",
    "data-2026-06.parquet",
    "data-2026-07.parquet",
    "data-2026-08.parquet",
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

COLUMNS = ["station_name", "xml_station_name", "eva"]
DEUTZ_STATION = "Köln Messe/Deutz"


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


def month_from_filename(filename: str) -> str:
    return filename.removeprefix("data-").removesuffix(".parquet")


def load_study_area(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=COLUMNS)
    df["eva"] = df["eva"].astype("string").str.zfill(8)

    filtered = df[df["eva"].isin(EVA_TO_STATION)].copy()
    filtered["station_name"] = filtered["eva"].map(EVA_TO_STATION)

    return filtered


def main() -> None:
    monthly_parts = []

    for filename in MONTH_FILES:
        path = MONTHLY_DATASETS_DIR / filename
        month = month_from_filename(filename)

        filtered = load_study_area(path)
        filtered.insert(0, "month", month)
        monthly_parts.append(filtered)

    study_area = pd.concat(monthly_parts, ignore_index=True)
    station_order = list(STATION_EVA_MAP)

    station_counts = (
        study_area.groupby("station_name", dropna=False)
        .agg(
            months_present=("month", "nunique"),
            observations=("station_name", "size"),
        )
        .reindex(station_order)
        .reset_index()
    )

    eva_numbers = (
        study_area.groupby("station_name")["eva"]
        .apply(lambda values: ", ".join(sorted(values.dropna().astype(str).unique())))
        .reindex(station_order)
        .reset_index(name="eva_numbers")
    )

    deutz = (
        study_area[study_area["station_name"] == DEUTZ_STATION]
        .groupby(["eva", "xml_station_name"], dropna=False)
        .size()
        .reset_index(name="observations")
        .sort_values(["eva", "xml_station_name"])
        .reset_index(drop=True)
    )

    print("Stationen")
    print(station_counts.to_string(index=False))

    print("\nEVA-Nummern")
    print(eva_numbers.to_string(index=False))

    print("\nKöln Messe/Deutz")
    print(deutz.to_string(index=False))


if __name__ == "__main__":
    main()
