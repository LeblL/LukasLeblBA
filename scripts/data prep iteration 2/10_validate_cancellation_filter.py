from pathlib import Path

import pandas as pd


DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "processed"
    / "spnv_base_koeln_bonn_2025-11_2026-08.parquet"
)
COLUMNS = [
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

    excluded = int((canceled_departure_based | canceled_arrival_based).sum())
    remaining = len(spnv) - excluded
    excluded_share = excluded / len(spnv) * 100 if len(spnv) else 0

    print("Grundlage von delay_in_min")
    print()
    print(f"Abfahrtsbasiert: {int(departure_based.sum())}")
    print(f"Ankunftsbasiert: {int(arrival_based.sum())}")

    print()
    print("Relevante Ausfälle")
    print()
    print(
        "Ausgefallene abfahrtsbasierte Beobachtungen: "
        f"{int(canceled_departure_based.sum())}"
    )
    print(
        "Ausgefallene ankunftsbasierte Beobachtungen: "
        f"{int(canceled_arrival_based.sum())}"
    )

    print()
    print(f"Insgesamt auszuschließen: {excluded}")
    print(f"Verbleibende Beobachtungen: {remaining}")
    print()
    print(f"Anteil auszuschließen: {excluded_share:.2f} %")


if __name__ == "__main__":
    main()
