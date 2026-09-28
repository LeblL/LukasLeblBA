from pathlib import Path

import pandas as pd


DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "processed"
    / "spnv_base_koeln_bonn_2025-11_2026-08.parquet"
)

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

COLUMNS = [
    "id",
    "eva",
    "train_type",
    "line_number",
    "delay_in_min",
    "arrival_planned_time",
    "arrival_change_time",
    "departure_planned_time",
    "departure_change_time",
    "arrival_is_canceled",
    "departure_is_canceled",
]
OUTPUT_COLUMNS = [
    "station_name",
    "train_type",
    "line_number",
    "delay_in_min",
    "arrival_planned_time",
    "arrival_change_time",
    "departure_planned_time",
    "departure_change_time",
    "id",
]
EVA_TO_STATION = {
    eva: station
    for station, eva_numbers in STATION_EVA_MAP.items()
    for eva in eva_numbers
}


def main() -> None:
    spnv = pd.read_parquet(DATASET_PATH, columns=COLUMNS)

    departure_based = spnv["departure_planned_time"].notna()
    arrival_based = (
        spnv["departure_planned_time"].isna()
        & spnv["arrival_planned_time"].notna()
    )

    canceled_departure_based = (
        departure_based & spnv["departure_is_canceled"].eq(True)
    )
    canceled_arrival_based = (
        arrival_based & spnv["arrival_is_canceled"].eq(True)
    )
    relevant_canceled = canceled_departure_based | canceled_arrival_based

    delays = spnv.loc[~relevant_canceled].copy()
    delays["station_name"] = (
        delays["eva"].astype("string").str.zfill(8).map(EVA_TO_STATION)
    )

    print("Höchste Verspätungen")
    print()
    print(
        delays.sort_values("delay_in_min", ascending=False)
        .head(20)[OUTPUT_COLUMNS]
        .to_string(index=False)
    )

    print()
    print("Niedrigste Verspätungen")
    print()
    print(
        delays.sort_values("delay_in_min", ascending=True)
        .head(20)[OUTPUT_COLUMNS]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
