from pathlib import Path

import pandas as pd


DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "processed"
    / "spnv_base_koeln_bonn_2026-01_2026-06.parquet"
)
COLUMNS = [
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

    delay = spnv.loc[~relevant_canceled, "delay_in_min"]

    print("Verspätung nach Ausfallfilter")
    print()
    print(f"Beobachtungen: {len(delay)}")
    print(f"Minimum: {delay.min()}")
    print(f"Median: {delay.median():.2f}")
    print(f"Mittelwert: {delay.mean():.2f}")
    print(f"95%-Quantil: {delay.quantile(0.95):.2f}")
    print(f"99%-Quantil: {delay.quantile(0.99):.2f}")
    print(f"Maximum: {delay.max()}")

    print()
    print("Einfache Verteilung")
    print()
    print(f"delay_in_min < 0: {int((delay < 0).sum())}")
    print(f"delay_in_min = 0: {int((delay == 0).sum())}")
    print(f"delay_in_min > 0: {int((delay > 0).sum())}")

    print()
    print("Auffällige Werte")
    print()
    print(f"delay_in_min < -10: {int((delay < -10).sum())}")
    print(f"delay_in_min > 60: {int((delay > 60).sum())}")
    print(f"delay_in_min > 120: {int((delay > 120).sum())}")
    print(f"delay_in_min > 180: {int((delay > 180).sum())}")


if __name__ == "__main__":
    main()
