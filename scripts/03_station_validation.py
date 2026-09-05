from pathlib import Path

import pandas as pd


DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets"
COLUMNS = ["station_name", "eva"]
MONTH_FILES = [
    "data-2025-12.parquet",
    "data-2026-01.parquet",
    "data-2026-02.parquet",
    "data-2026-03.parquet",
    "data-2026-04.parquet",
    "data-2026-05.parquet",
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


def month_from_filename(filename: str) -> str:
    return filename.removeprefix("data-").removesuffix(".parquet")


def main() -> None:
    monthly_results = []
    missing_station_warnings = []

    for filename in MONTH_FILES:
        month = month_from_filename(filename)
        path = DATASETS_DIR / filename

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
