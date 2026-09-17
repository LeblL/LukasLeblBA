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
EXPECTED_OBSERVATIONS = 549360
OUTPUT_FILE = "spnv_base_koeln_bonn_2026-01_2026-06.parquet"


def find_project_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "datasets").is_dir():
            return parent

    raise FileNotFoundError("Projektordner mit datasets-Verzeichnis nicht gefunden.")


PROJECT_DIR = find_project_dir()
DATASETS_DIR = PROJECT_DIR / "datasets"
MONTHLY_DATASETS_DIR = DATASETS_DIR / "monthly"
PROCESSED_DIR = DATASETS_DIR / "processed"
OUTPUT_PATH = PROCESSED_DIR / OUTPUT_FILE
STATION_EVAS = {
    eva
    for eva_numbers in STATION_EVA_MAP.values()
    for eva in eva_numbers
}
CANONICAL_EVA_BY_RAW_EVA = {
    eva: eva_numbers[0]
    for eva_numbers in STATION_EVA_MAP.values()
    for eva in eva_numbers
}


def canonicalize_stations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eva = df["eva"].astype("string").str.zfill(8)
    df["eva"] = eva.map(CANONICAL_EVA_BY_RAW_EVA)

    return df


def load_filtered_month(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    eva = df["eva"].astype("string").str.zfill(8)
    mask = eva.isin(STATION_EVAS) & df["train_type"].isin(SPNV_TYPES)

    return canonicalize_stations(df[mask])


def main() -> None:
    monthly_parts = []

    for filename in MONTH_FILES:
        path = MONTHLY_DATASETS_DIR / filename
        monthly_parts.append(load_filtered_month(path))

    spnv = pd.concat(monthly_parts, ignore_index=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    spnv.to_parquet(OUTPUT_PATH, index=False)

    print("SPNV-Datensatz erstellt")
    print(f"Beobachtungen: {len(spnv)}")
    print(f"Spalten: {len(spnv.columns)}")
    print(f"Datei: {OUTPUT_PATH}")

    if len(spnv) != EXPECTED_OBSERVATIONS:
        print(
            "WARNUNG: Erwartet "
            f"{EXPECTED_OBSERVATIONS} Beobachtungen, gefunden {len(spnv)}."
        )


if __name__ == "__main__":
    main()
