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
COLUMNS = ["station_name", "eva"]
MONTH_FILES = [
    OLD_SCHEMA_DATASETS_DIR / "old-schema-data-2025-12.parquet",
    OLD_SCHEMA_DATASETS_DIR / "old-schema-data-2026-01.parquet",
    OLD_SCHEMA_DATASETS_DIR / "old-schema-data-2026-02.parquet",
    OLD_SCHEMA_DATASETS_DIR / "old-schema-data-2026-03.parquet",
    OLD_SCHEMA_DATASETS_DIR / "old-schema-data-2026-04.parquet",
    DATASETS_DIR / "data-2026-05.parquet",
]

# Exakte Stationsnamen des Untersuchungsraums.
STATIONS = [
    "K\u00f6ln Hbf",
    "K\u00f6ln Messe/Deutz",
    "K\u00f6ln S\u00fcd",
    "K\u00f6ln/Bonn Flughafen",
    "Br\u00fchl",
    "Bonn Hbf",
    "Bonn-Beuel",
    "Troisdorf",
]


def month_from_path(path: Path) -> str:
    return path.stem.removeprefix("old-schema-data-").removeprefix("data-")


def main() -> None:
    monthly_results = []
    missing_station_warnings = []

    for path in MONTH_FILES:
        month = month_from_path(path)

        df = pd.read_parquet(path, columns=COLUMNS)
        filtered = df[df["station_name"].isin(STATIONS)]

        # Pro Monat nur die aggregierten Treffer behalten.
        counts = (
            filtered.groupby(["station_name", "eva"], dropna=False)
            .size()
            .reset_index(name="observations")
        )
        counts.insert(0, "month", month)
        monthly_results.append(counts)

        found_stations = set(filtered["station_name"].dropna().unique())
        for station in STATIONS:
            if station not in found_stations:
                missing_station_warnings.append((month, station))

    result = (
        pd.concat(monthly_results, ignore_index=True)
        .sort_values(["month", "station_name", "eva"])
        .reset_index(drop=True)
    )

    print("Beobachtungen je Monat, Station und EVA")
    print(result.to_string(index=False))

    if missing_station_warnings:
        print("\nWARNUNG: Fehlende Stationen je Monat")
        for month, station in missing_station_warnings:
            print(f"- {month}: {station}")

    eva_counts = (
        result.groupby(["month", "station_name"])["eva"]
        .nunique(dropna=False)
        .reset_index(name="eva_count")
    )
    multi_eva_keys = eva_counts[eva_counts["eva_count"] > 1][["month", "station_name"]]
    multi_eva = result.merge(multi_eva_keys, on=["month", "station_name"], how="inner")

    print("\nStationen mit mehreren EVA-Nummern")
    if multi_eva.empty:
        print("Keine gefunden.")
    else:
        print(multi_eva.to_string(index=False))

    eva_summary = (
        result.groupby("station_name")["eva"]
        .apply(lambda values: ", ".join(sorted(values.astype(str).unique())))
        .reset_index(name="eva_numbers")
        .sort_values("station_name")
        .reset_index(drop=True)
    )

    print("\nEVA-Nummern je Station im gesamten Zeitraum")
    print(eva_summary.to_string(index=False))


if __name__ == "__main__":
    main()
