from pathlib import Path

import pandas as pd


PROCESSED_DIR = Path(__file__).resolve().parents[2] / "datasets" / "processed"

INPUT_PATH = PROCESSED_DIR / "spnv_weather_koeln_bonn_2025-11_2026-08.parquet"
OUTPUT_PATH = (
    PROCESSED_DIR / "modeling_dataset_final_koeln_bonn_2025-11_2026-08.parquet"
)

MODEL_COLUMNS = [
    "delay_target",
    "id",
    "concrete_ride_id",
    "planned_event_time",
    "event_time_source",
    "train_line_ride_id",
    "station_name",
    "line_number",
    "train_type",
    "train_line_station_num",
    "event_hour",
    "event_weekday",
    "event_month",
    "is_weekend",
    "is_public_holiday",
    "ff_wind_speed_ms",
    "ff_wind_direction_deg",
    "fx_wind_gust_ms",
    "tu_temperature_c",
    "tu_relative_humidity_pct",
    "rr_precipitation_mm",
    "rr_precipitation_indicator",
    "rr_precipitation_form",
]

df = pd.read_parquet(INPUT_PATH)
df["concrete_ride_id"] = df["id"].str.rsplit("-", n=1).str[0]
df = df[MODEL_COLUMNS].copy()
df.to_parquet(OUTPUT_PATH, index=False)
