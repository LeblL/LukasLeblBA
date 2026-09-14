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

COLUMNS = ["eva", "train_type", "line_number"]


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


def load_station_services(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=COLUMNS)
    df["eva"] = df["eva"].astype("string").str.zfill(8)

    filtered = df[df["eva"].isin(EVA_TO_STATION)].copy()
    filtered["station_name"] = filtered["eva"].map(EVA_TO_STATION)

    return filtered


def main() -> None:
    monthly_parts = []

    for filename in MONTH_FILES:
        path = MONTHLY_DATASETS_DIR / filename
        monthly_parts.append(load_station_services(path))

    study_area = pd.concat(monthly_parts, ignore_index=True)
    study_area["station_name"] = pd.Categorical(
        study_area["station_name"],
        categories=list(STATION_EVA_MAP),
        ordered=True,
    )

    result = (
        study_area.groupby(
            ["station_name", "line_number", "train_type"],
            dropna=False,
            observed=True,
        )
        .size()
        .reset_index(name="observations")
        .sort_values(["station_name", "observations"], ascending=[True, False])
        .reset_index(drop=True)
    )

    result["station_name"] = result["station_name"].astype(str)
    result["line_number"] = result["line_number"].fillna("<NA>")

    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
