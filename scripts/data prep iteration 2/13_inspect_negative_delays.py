from pathlib import Path

import pandas as pd


DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "processed"
    / "spnv_base_koeln_bonn_2025-11_2026-08.parquet"
)
COLUMNS = [
    "eva",
    "train_type",
    "line_number",
    "train_line_ride_id",
    "delay_in_min",
    "arrival_planned_time",
    "departure_planned_time",
    "arrival_is_canceled",
    "departure_is_canceled",
]


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

    negative = spnv.loc[~relevant_canceled & spnv["delay_in_min"].lt(0)]
    negative_delay = negative["delay_in_min"]

    train_type_counts = (
        negative.groupby("train_type", dropna=False)
        .size()
        .reset_index(name="negative_observations")
        .sort_values("negative_observations", ascending=False)
        .reset_index(drop=True)
    )

    line_number_counts = (
        negative.groupby("line_number", dropna=False)
        .size()
        .reset_index(name="negative_observations")
        .sort_values("negative_observations", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )

    print("Negative Verspätungen")
    print()
    print(f"Negative Werte insgesamt: {len(negative)}")
    print(f"delay_in_min < -5: {int((negative_delay < -5).sum())}")
    print(f"delay_in_min < -10: {int((negative_delay < -10).sum())}")
    print(f"delay_in_min < -30: {int((negative_delay < -30).sum())}")
    print(f"delay_in_min < -60: {int((negative_delay < -60).sum())}")

    print()
    print("Nach train_type")
    print()
    print(train_type_counts.to_string(index=False))

    print()
    print("Nach line_number")
    print()
    print(line_number_counts.to_string(index=False))

    print()
    print("Zugfahrten")
    print()
    print(
        "Unterschiedliche train_line_ride_id mit negativen Werten: "
        f"{negative['train_line_ride_id'].nunique(dropna=True)}"
    )
    print(
        "Unterschiedliche train_line_ride_id mit delay_in_min < -10: "
        f"{negative.loc[negative_delay < -10, 'train_line_ride_id'].nunique(dropna=True)}"
    )


if __name__ == "__main__":
    main()
