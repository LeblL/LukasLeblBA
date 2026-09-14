from pathlib import Path

import pandas as pd


DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "processed"
    / "spnv_base_koeln_bonn_2026-01_2026-06.parquet"
)
COLUMNS = ["arrival_planned_time", "departure_planned_time"]


def main() -> None:
    spnv = pd.read_parquet(DATASET_PATH, columns=COLUMNS)

    arrival_present = spnv["arrival_planned_time"].notna()
    departure_present = spnv["departure_planned_time"].notna()

    both_present = int((arrival_present & departure_present).sum())
    only_arrival = int((arrival_present & ~departure_present).sum())
    only_departure = int((~arrival_present & departure_present).sum())
    no_planned_time = int((~arrival_present & ~departure_present).sum())
    at_least_one_planned_time = int((arrival_present | departure_present).sum())

    print("Geplante Zeitstruktur")
    print()
    print(f"Ankunft und Abfahrt vorhanden: {both_present}")
    print(f"Nur Ankunft vorhanden: {only_arrival}")
    print(f"Nur Abfahrt vorhanden: {only_departure}")
    print(f"Keine geplante Zeit vorhanden: {no_planned_time}")
    print()
    print(f"Gesamt: {len(spnv)}")
    print()
    print(f"Beobachtungen mit mindestens einer geplanten Zeit: {at_least_one_planned_time}")
    print(f"Beobachtungen ohne geplante Zeit: {no_planned_time}")


if __name__ == "__main__":
    main()
