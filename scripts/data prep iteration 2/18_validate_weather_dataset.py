from pathlib import Path

import pandas as pd


PROCESSED_DIR = Path(__file__).resolve().parents[2] / "datasets" / "processed"
WEATHER_PATH = PROCESSED_DIR / "spnv_weather_koeln_bonn_2025-11_2026-08.parquet"

WEATHER_COLUMNS = [
    "ff_wind_speed_ms",
    "ff_wind_direction_deg",
    "fx_wind_gust_ms",
    "tu_temperature_c",
    "tu_relative_humidity_pct",
    "rr_precipitation_mm",
    "rr_precipitation_indicator",
    "rr_precipitation_form",
]

MISSING_COLUMNS = [
    "ff_wind_speed_ms",
    "fx_wind_gust_ms",
    "tu_temperature_c",
    "tu_relative_humidity_pct",
    "rr_precipitation_mm",
    "rr_precipitation_form",
]

SHORT_NAMES = {
    "ff_wind_speed_ms": "ff_speed",
    "fx_wind_gust_ms": "fx_gust",
    "tu_temperature_c": "temp",
    "tu_relative_humidity_pct": "humidity",
    "rr_precipitation_mm": "rain_mm",
    "rr_precipitation_form": "rain_form",
}

RANGE_COLUMNS = [
    "ff_wind_speed_ms",
    "ff_wind_direction_deg",
    "fx_wind_gust_ms",
    "tu_temperature_c",
    "tu_relative_humidity_pct",
    "rr_precipitation_mm",
]


def print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def missing_by_group(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    observations = df.groupby(group_column, dropna=False).size()
    missing = df.groupby(group_column, dropna=False)[MISSING_COLUMNS].apply(
        lambda group: group.isna().sum()
    )
    percent = missing.div(observations, axis=0).mul(100).round(2)

    missing.columns = [f"{SHORT_NAMES[column]}_n" for column in missing.columns]
    percent.columns = [f"{SHORT_NAMES[column]}_pct" for column in percent.columns]

    return (
        pd.concat([observations.rename("observations"), missing, percent], axis=1)
        .reset_index()
        .sort_values(group_column)
    )


def main() -> None:
    df = pd.read_parquet(WEATHER_PATH)

    print_section("Fehlende Wetterwerte")
    missing = df[WEATHER_COLUMNS].isna().sum().reset_index(name="missing")
    missing = missing.rename(columns={"index": "column"})
    missing["missing_pct"] = (missing["missing"] / len(df) * 100).round(3)
    print(missing.to_string(index=False))

    print_section("Fehlende Wetterwerte nach Bahnhof")
    print(missing_by_group(df, "station_name").to_string(index=False))

    print_section("Fehlende Wetterwerte nach Monat")
    print(missing_by_group(df, "event_month").to_string(index=False))

    print_section("DWD-Fehlwertcodes (-999)")
    dwd_missing_codes = pd.DataFrame(
        {
            "column": WEATHER_COLUMNS,
            "count": [
                int(df[column].isin([-999, -999.0]).sum())
                for column in WEATHER_COLUMNS
            ],
        }
    )
    print(dwd_missing_codes.to_string(index=False))

    print_section("Wertebereiche")
    ranges = df[RANGE_COLUMNS].agg(["min", "median", "mean", "max"]).T
    ranges["quantile_95"] = df[RANGE_COLUMNS].quantile(0.95)
    ranges = ranges[["min", "median", "mean", "quantile_95", "max"]]
    ranges = ranges.reset_index().rename(columns={"index": "column"})
    print(ranges.round(3).to_string(index=False))

    print_section("Niederschlags-Codes")
    print("rr_precipitation_indicator")
    indicator_counts = (
        df["rr_precipitation_indicator"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("value")
        .reset_index(name="observations")
    )
    print(indicator_counts.to_string(index=False))
    print()
    print("rr_precipitation_form")
    form_counts = (
        df["rr_precipitation_form"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("value")
        .reset_index(name="observations")
    )
    print(form_counts.to_string(index=False))

    print_section("Zeit")
    planned_event_time = pd.to_datetime(df["planned_event_time"], errors="coerce")
    print(f"Minimum planned_event_time: {planned_event_time.min()}")
    print(f"Maximum planned_event_time: {planned_event_time.max()}")
    print(f"Fehlende planned_event_time: {int(planned_event_time.isna().sum())}")
    print(
        "Beobachtungen vor 01.11.2025: "
        f"{int((planned_event_time < pd.Timestamp('2025-11-01')).sum())}"
    )
    print(
        "Beobachtungen nach 31.08.2026: "
        f"{int((planned_event_time > pd.Timestamp('2026-08-31 23:59:59')).sum())}"
    )

    print()
    print("Validierungsübersicht abgeschlossen")


if __name__ == "__main__":
    main()
