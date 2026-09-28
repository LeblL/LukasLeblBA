from pathlib import Path

import pandas as pd


CLEAN_DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "processed"
    / "spnv_clean_koeln_bonn_2025-11_2026-08.parquet"
)
TEMPORAL_DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "processed"
    / "spnv_temporal_koeln_bonn_2025-11_2026-08.parquet"
)
EXPECTED_OBSERVATIONS = 849421


def main() -> None:
    spnv = pd.read_parquet(CLEAN_DATASET_PATH)

    departure_present = spnv["departure_planned_time"].notna()
    spnv["planned_event_time"] = spnv["departure_planned_time"].combine_first(
        spnv["arrival_planned_time"]
    )
    spnv["event_time_source"] = departure_present.map(
        {True: "departure", False: "arrival"}
    )

    planned_event_time = pd.to_datetime(spnv["planned_event_time"], errors="coerce")
    spnv["event_date"] = planned_event_time.dt.date
    spnv["event_hour"] = planned_event_time.dt.hour
    spnv["event_weekday"] = planned_event_time.dt.weekday
    spnv["event_month"] = planned_event_time.dt.month
    spnv["is_weekend"] = spnv["event_weekday"].isin([5, 6])

    observations_before_period_filter = len(spnv)
    start = pd.Timestamp("2025-11-01 00:00:00")
    end = pd.Timestamp("2026-08-31 23:59:59")
    inside_study_period = planned_event_time.between(start, end, inclusive="both")
    removed_outside_study_period = int((~inside_study_period).sum())

    spnv = spnv.loc[inside_study_period].copy()
    spnv.to_parquet(TEMPORAL_DATASET_PATH, index=False)

    missing_planned_event_time = int(spnv["planned_event_time"].isna().sum())
    source_counts = spnv["event_time_source"].value_counts()

    print("Zeitmerkmale ergänzt")
    print()
    print(
        "Beobachtungen vor zeitlicher Abgrenzung: "
        f"{observations_before_period_filter}"
    )
    print(
        "Außerhalb Untersuchungszeitraum entfernt: "
        f"{removed_outside_study_period}"
    )
    print(f"Beobachtungen nach zeitlicher Abgrenzung: {len(spnv)}")
    print()
    print(f"planned_event_time fehlend: {missing_planned_event_time}")
    print()
    print("Zeitquelle:")
    print(f"departure: {int(source_counts.get('departure', 0))}")
    print(f"arrival: {int(source_counts.get('arrival', 0))}")
    print()
    print("Zeitraum planned_event_time:")
    print(f"von: {spnv['planned_event_time'].min()}")
    print(f"bis: {spnv['planned_event_time'].max()}")
    print()
    print(f"Spalten: {len(spnv.columns)}")
    print(f"Datei: {TEMPORAL_DATASET_PATH}")

    if EXPECTED_OBSERVATIONS is not None and len(spnv) != EXPECTED_OBSERVATIONS:
        print(
            "WARNUNG: Erwartet "
            f"{EXPECTED_OBSERVATIONS} Beobachtungen, gefunden {len(spnv)}."
        )
    if missing_planned_event_time:
        print("WARNUNG: planned_event_time enthält fehlende Werte.")


if __name__ == "__main__":
    main()
