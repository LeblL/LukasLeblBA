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
    "arrival_is_canceled",
    "departure_is_canceled",
]


def main() -> None:
    spnv = pd.read_parquet(DATASET_PATH, columns=COLUMNS)

    delay = spnv["delay_in_min"]
    arrival_canceled = spnv["arrival_is_canceled"].eq(True)
    departure_canceled = spnv["departure_is_canceled"].eq(True)
    any_canceled = arrival_canceled | departure_canceled
    canceled_delay = delay[any_canceled]

    print("Verteilung von delay_in_min")
    print()
    print(f"Beobachtungen: {len(spnv)}")
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
    print("Ausfälle")
    print()
    print(f"arrival_is_canceled = True: {int(arrival_canceled.sum())}")
    print(f"departure_is_canceled = True: {int(departure_canceled.sum())}")
    print(f"mindestens ein Ausfallindikator = True: {int(any_canceled.sum())}")

    print()
    print("delay_in_min bei mindestens einem Ausfallindikator")
    print()
    print(f"Minimum delay_in_min: {canceled_delay.min()}")
    print(f"Median delay_in_min: {canceled_delay.median():.2f}")
    print(f"Mittelwert delay_in_min: {canceled_delay.mean():.2f}")
    print(f"Maximum delay_in_min: {canceled_delay.max()}")


if __name__ == "__main__":
    main()
