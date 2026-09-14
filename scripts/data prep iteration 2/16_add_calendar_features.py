from datetime import date
from pathlib import Path

import pandas as pd


TEMPORAL_DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "processed"
    / "spnv_temporal_koeln_bonn_2026-01_2026-06.parquet"
)
CALENDAR_DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "processed"
    / "spnv_calendar_koeln_bonn_2026-01_2026-06.parquet"
)
EXPECTED_OBSERVATIONS = 523938
EXPECTED_COLUMNS = 28
PUBLIC_HOLIDAYS_NRW = {
    date(2026, 1, 1): "Neujahr",
    date(2026, 4, 3): "Karfreitag",
    date(2026, 4, 6): "Ostermontag",
    date(2026, 5, 1): "Tag der Arbeit",
    date(2026, 5, 14): "Christi Himmelfahrt",
    date(2026, 5, 25): "Pfingstmontag",
    date(2026, 6, 4): "Fronleichnam",
}


def main() -> None:
    spnv = pd.read_parquet(TEMPORAL_DATASET_PATH)

    holiday_name = spnv["event_date"].map(PUBLIC_HOLIDAYS_NRW)
    spnv["is_public_holiday"] = holiday_name.notna()
    spnv["holiday_name"] = holiday_name.fillna(pd.NA)

    spnv.to_parquet(CALENDAR_DATASET_PATH, index=False)

    holiday_observations = int(spnv["is_public_holiday"].sum())
    non_holiday_observations = len(spnv) - holiday_observations
    missing_is_public_holiday = int(spnv["is_public_holiday"].isna().sum())
    missing_holiday_names = spnv["holiday_name"].isna()
    missing_holiday_name_on_holiday = int(
        (spnv["is_public_holiday"] & missing_holiday_names).sum()
    )
    holiday_name_on_non_holiday = int(
        ((~spnv["is_public_holiday"]) & spnv["holiday_name"].notna()).sum()
    )

    print("Kalendermerkmale ergänzt")
    print()
    print(f"Beobachtungen: {len(spnv)}")
    print(f"Spalten: {len(spnv.columns)}")
    print()
    print("Gesetzliche Feiertage:")
    for holiday_date, name in PUBLIC_HOLIDAYS_NRW.items():
        observations = int((spnv["event_date"] == holiday_date).sum())
        print(
            f"{holiday_date:%d.%m.%Y} - {name}: "
            f"{observations} Beobachtungen"
        )
    print()
    print(f"Beobachtungen an Feiertagen insgesamt: {holiday_observations}")
    print(f"Beobachtungen außerhalb von Feiertagen: {non_holiday_observations}")
    print()
    print(f"Datei: {CALENDAR_DATASET_PATH}")

    if len(spnv) != EXPECTED_OBSERVATIONS:
        print(
            "WARNUNG: Erwartet "
            f"{EXPECTED_OBSERVATIONS} Beobachtungen, gefunden {len(spnv)}."
        )
    if len(spnv.columns) != EXPECTED_COLUMNS:
        print(
            "WARNUNG: Erwartet "
            f"{EXPECTED_COLUMNS} Spalten, gefunden {len(spnv.columns)}."
        )
    if missing_is_public_holiday:
        print("WARNUNG: is_public_holiday enthält fehlende Werte.")
    if missing_holiday_name_on_holiday or holiday_name_on_non_holiday:
        print("WARNUNG: holiday_name ist nicht konsistent mit is_public_holiday.")


if __name__ == "__main__":
    main()
