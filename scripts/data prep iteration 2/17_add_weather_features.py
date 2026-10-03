from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_DIR / "datasets" / "processed"
METADATA_DIR = PROJECT_DIR / "datasets" / "metadata"
WEATHER_DIR = PROJECT_DIR / "datasets" / "weather"

CALENDAR_DATASET_PATH = (
    PROCESSED_DIR / "spnv_calendar_koeln_bonn_2025-11_2026-08.parquet"
)
WEATHER_DATASET_PATH = (
    PROCESSED_DIR / "spnv_weather_koeln_bonn_2025-11_2026-08.parquet"
)
MAPPING_PATH = (
    METADATA_DIR / "db_station_weather_station_mapping_koeln_bonn_adjusted.csv"
)

EXPECTED_OBSERVATIONS = 849421
EXPECTED_COLUMNS = 36
WEATHER_HOUR_START = pd.Timestamp("2025-11-01 00:00:00")
WEATHER_HOUR_END_EXCLUSIVE = pd.Timestamp("2026-09-01 00:00:00")

WEATHER_SOURCES = {
    "ff": {
        "directory": "FF",
        "columns": {
            "QN_3": "ff_quality_level",
            "F": "ff_wind_speed_ms",
            "D": "ff_wind_direction_deg",
        },
    },
    "tu": {
        "directory": "TU",
        "columns": {
            "QN_9": "tu_quality_level",
            "TT_TU": "tu_temperature_c",
            "RF_TU": "tu_relative_humidity_pct",
        },
    },
    "rr": {
        "directory": "RR",
        "columns": {
            "QN_8": "rr_quality_level",
            "R1": "rr_precipitation_mm",
            "RS_IND": "rr_precipitation_indicator",
            "WRTR": "rr_precipitation_form",
        },
    },
    "fx": {
        "directory": "FX",
        "columns": {
            "QN_8": "fx_quality_level",
            "FX_911": "fx_wind_gust_ms",
        },
    },
}

MAPPING_COLUMNS = [
    "ff_station_id",
    "ff_station_name",
    "ff_distance_km",
    "tu_station_id",
    "tu_station_name",
    "tu_distance_km",
    "rr_station_id",
    "rr_station_name",
    "rr_distance_km",
    "fx_station_id",
    "fx_station_name",
    "fx_distance_km",
]

STATION_ID_COLUMNS = [
    "ff_station_id",
    "tu_station_id",
    "rr_station_id",
    "fx_station_id",
]

NA_VALUES = ["-999", "-999.0", " -999", " -999.0"]

WEATHER_FEATURE_COLUMNS = [
    "ff_wind_speed_ms",
    "ff_wind_direction_deg",
    "fx_wind_gust_ms",
    "tu_temperature_c",
    "tu_relative_humidity_pct",
    "rr_precipitation_mm",
    "rr_precipitation_indicator",
    "rr_precipitation_form",
]


def read_mapping() -> pd.DataFrame:
    dtype = {"eva": str} | {column: str for column in STATION_ID_COLUMNS}
    mapping = pd.read_csv(MAPPING_PATH, encoding="utf-8-sig", dtype=dtype)

    required_columns = {"eva", *MAPPING_COLUMNS}
    missing_columns = required_columns - set(mapping.columns)
    if missing_columns:
        raise ValueError(
            "Weather station mapping is missing columns: "
            f"{', '.join(sorted(missing_columns))}"
        )

    mapping = mapping[["eva", *MAPPING_COLUMNS]].copy()
    mapping["eva"] = mapping["eva"].astype(str).str.zfill(8)
    for column in STATION_ID_COLUMNS:
        mapping[column] = mapping[column].astype(str).str.zfill(5)

    duplicate_evas = mapping["eva"].duplicated(keep=False)
    if duplicate_evas.any():
        duplicates = mapping.loc[duplicate_evas, "eva"].sort_values().unique()
        raise ValueError(
            "Weather station mapping contains duplicate EVA values: "
            f"{', '.join(duplicates)}"
        )

    return mapping


def read_weather_source(prefix: str, config: dict[str, object]) -> pd.DataFrame:
    source_dir = WEATHER_DIR / str(config["directory"])
    source_columns = dict(config["columns"])
    paths = sorted(source_dir.glob("produkt_*.txt"))

    if not paths:
        raise FileNotFoundError(f"No weather files found in {source_dir}.")

    frames = []
    for path in paths:
        weather = pd.read_csv(
            path,
            sep=";",
            skipinitialspace=True,
            na_values=NA_VALUES,
        )
        weather.columns = [column.strip() for column in weather.columns]

        required_columns = {"STATIONS_ID", "MESS_DATUM", *source_columns.keys()}
        missing_columns = required_columns - set(weather.columns)
        if missing_columns:
            raise ValueError(
                f"{path.name} is missing columns: "
                f"{', '.join(sorted(missing_columns))}"
            )

        weather = weather[["STATIONS_ID", "MESS_DATUM", *source_columns.keys()]]
        weather = weather.rename(columns=source_columns)
        weather[f"{prefix}_weather_station_id"] = (
            weather["STATIONS_ID"].astype(str).str.strip().str.zfill(5)
        )
        # DWD MESS_DATUM is provided in UTC.
        # Convert to German local time (Europe/Berlin) before joining with
        # DB planned_event_time, which is stored as timezone-naive local time.
        # Europe/Berlin automatically handles CET (+01:00) and CEST (+02:00).
        weather_hour_utc = pd.to_datetime(
            weather["MESS_DATUM"].astype(str).str.strip(),
            format="%Y%m%d%H",
            errors="coerce",
            utc=True,
        )
        weather["weather_hour"] = (
            weather_hour_utc.dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
        )
        weather[f"{prefix}_weather_record_available"] = True

        value_columns = list(source_columns.values())
        for column in value_columns:
            weather[column] = pd.to_numeric(weather[column], errors="coerce")

        invalid_dates = int(weather["weather_hour"].isna().sum())
        if invalid_dates:
            raise ValueError(f"{path.name} contains {invalid_dates} invalid dates.")

        # Keep only the analysis period before validating duplicate local hours.
        # The raw DWD files include fall-back DST hours outside this dataset.
        weather = weather.loc[
            weather["weather_hour"].ge(WEATHER_HOUR_START)
            & weather["weather_hour"].lt(WEATHER_HOUR_END_EXCLUSIVE)
        ].copy()

        frames.append(
            weather[
                [
                    f"{prefix}_weather_station_id",
                    "weather_hour",
                    f"{prefix}_weather_record_available",
                    *value_columns,
                ]
            ]
        )

    combined = pd.concat(frames, ignore_index=True)
    duplicates = combined.duplicated(
        [f"{prefix}_weather_station_id", "weather_hour"], keep=False
    )
    if duplicates.any():
        duplicate_keys = combined.loc[
            duplicates, [f"{prefix}_weather_station_id", "weather_hour"]
        ].drop_duplicates()
        raise ValueError(
            f"{prefix.upper()} weather source contains duplicate station/hour keys:\n"
            f"{duplicate_keys.head(20).to_string(index=False)}"
        )

    return combined


def add_station_mapping(spnv: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    spnv = spnv.copy()
    spnv["eva"] = spnv["eva"].astype(str).str.zfill(8)
    merged = spnv.merge(mapping, on="eva", how="left", validate="many_to_one")

    missing_mapping = merged[STATION_ID_COLUMNS].isna().any(axis=1)
    if missing_mapping.any():
        missing_evas = sorted(merged.loc[missing_mapping, "eva"].unique())
        raise ValueError(
            "Missing weather station mapping for EVA values: "
            f"{', '.join(missing_evas)}"
        )

    return merged


def add_weather_source(
    spnv: pd.DataFrame,
    prefix: str,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    station_id_column = f"{prefix}_station_id"
    weather_station_id_column = f"{prefix}_weather_station_id"
    availability_column = f"{prefix}_weather_record_available"

    merged = spnv.merge(
        weather,
        left_on=[station_id_column, "weather_hour"],
        right_on=[weather_station_id_column, "weather_hour"],
        how="left",
        validate="many_to_one",
    )
    merged = merged.drop(columns=[weather_station_id_column])
    merged[availability_column] = (
        merged[availability_column].fillna(False).astype(bool)
    )

    return merged


def main() -> None:
    spnv = pd.read_parquet(CALENDAR_DATASET_PATH)
    calendar_columns = list(spnv.columns)
    mapping = read_mapping()

    planned_event_time = pd.to_datetime(spnv["planned_event_time"], errors="coerce")
    spnv["weather_hour"] = planned_event_time.dt.floor("h")

    missing_weather_hour = int(spnv["weather_hour"].isna().sum())
    if missing_weather_hour:
        raise ValueError(f"weather_hour is missing for {missing_weather_hour} rows.")

    spnv = add_station_mapping(spnv, mapping)

    for prefix, config in WEATHER_SOURCES.items():
        weather = read_weather_source(prefix, config)
        spnv = add_weather_source(spnv, prefix, weather)

    output = spnv[[*calendar_columns, *WEATHER_FEATURE_COLUMNS]]
    output.to_parquet(WEATHER_DATASET_PATH, index=False)

    print("Weather features added")
    print()
    print(f"Input file: {CALENDAR_DATASET_PATH}")
    print(f"Output file: {WEATHER_DATASET_PATH}")
    print()
    print(f"Observations: {len(output)}")
    print(f"Columns: {len(output.columns)}")
    print(f"Weather hour from: {spnv['weather_hour'].min()}")
    print(f"Weather hour to: {spnv['weather_hour'].max()}")
    print()

    print("Weather record availability:")
    for prefix in WEATHER_SOURCES:
        availability_column = f"{prefix}_weather_record_available"
        missing_records = int((~spnv[availability_column]).sum())
        print(f"{prefix.upper()}: missing station/hour records = {missing_records}")

    print()
    print("Missing saved weather feature values:")
    for column in WEATHER_FEATURE_COLUMNS:
        print(f"{column}: {int(output[column].isna().sum())}")

    if EXPECTED_OBSERVATIONS is not None and len(output) != EXPECTED_OBSERVATIONS:
        print(
            "WARNING: Expected "
            f"{EXPECTED_OBSERVATIONS} observations, found {len(output)}."
        )
    if len(output.columns) != EXPECTED_COLUMNS:
        print(
            "WARNING: Expected "
            f"{EXPECTED_COLUMNS} columns, found {len(output.columns)}."
        )


if __name__ == "__main__":
    main()
