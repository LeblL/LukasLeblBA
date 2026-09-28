from pathlib import Path

import pandas as pd


BASE_DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "processed"
    / "spnv_base_koeln_bonn_2025-11_2026-08.parquet"
)
CLEAN_DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "processed"
    / "spnv_clean_koeln_bonn_2025-11_2026-08.parquet"
)
EXPECTED_CLEAN_OBSERVATIONS = 849426


def main() -> None:
    spnv = pd.read_parquet(BASE_DATASET_PATH)

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

    clean = spnv.loc[~relevant_canceled].copy()
    negative_delay = clean["delay_in_min"].lt(0)
    clean["delay_target"] = clean["delay_in_min"].where(~negative_delay, 0)

    clean.to_parquet(CLEAN_DATASET_PATH, index=False)

    print("Bereinigter SPNV-Datensatz erstellt")
    print()
    print(f"Beobachtungen Basisdatensatz: {len(spnv)}")
    print(f"Relevante Ausfälle entfernt: {int(relevant_canceled.sum())}")
    print(f"Beobachtungen bereinigter Datensatz: {len(clean)}")
    print(
        "Negative delay_in_min auf delay_target = 0 gesetzt: "
        f"{int(negative_delay.sum())}"
    )
    print(f"Spalten: {len(clean.columns)}")
    print(f"Datei: {CLEAN_DATASET_PATH}")

    if (
        EXPECTED_CLEAN_OBSERVATIONS is not None
        and len(clean) != EXPECTED_CLEAN_OBSERVATIONS
    ):
        print(
            "WARNUNG: Erwartet "
            f"{EXPECTED_CLEAN_OBSERVATIONS} Beobachtungen, gefunden {len(clean)}."
        )


if __name__ == "__main__":
    main()
