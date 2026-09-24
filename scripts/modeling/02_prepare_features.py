from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]

SPLIT_DIR = PROJECT_DIR / "datasets" / "modeling" / "splits"
OUTPUT_DIR = PROJECT_DIR / "datasets" / "modeling" / "features"

TRAIN_INPUT_PATH = SPLIT_DIR / "train.parquet"
VALIDATION_INPUT_PATH = SPLIT_DIR / "validation.parquet"
TEST_INPUT_PATH = SPLIT_DIR / "test.parquet"

TRAIN_OUTPUT_PATH = OUTPUT_DIR / "train_features.parquet"
VALIDATION_OUTPUT_PATH = OUTPUT_DIR / "validation_features.parquet"
TEST_OUTPUT_PATH = OUTPUT_DIR / "test_features.parquet"

FEATURE_COLUMNS = [
    "station_name",
    "line_number",
    "train_type",
    "train_line_station_num",
    "event_hour",
    "event_weekday",
    "is_public_holiday",
    "ff_wind_speed_ms",
    "wind_direction_sin",
    "wind_direction_cos",
    "fx_wind_gust_ms",
    "tu_temperature_c",
    "tu_relative_humidity_pct",
    "rr_precipitation_mm",
    "rr_precipitation_indicator",
    "rr_precipitation_form",
]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    wind_direction_rad = np.deg2rad(df["ff_wind_direction_deg"])
    df["wind_direction_sin"] = np.sin(wind_direction_rad)
    df["wind_direction_cos"] = np.cos(wind_direction_rad)

    return df[["delay_target"] + FEATURE_COLUMNS].copy()


train = pd.read_parquet(TRAIN_INPUT_PATH)
validation = pd.read_parquet(VALIDATION_INPUT_PATH)
test = pd.read_parquet(TEST_INPUT_PATH)

train_features = prepare_features(train)
validation_features = prepare_features(validation)
test_features = prepare_features(test)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_features.to_parquet(TRAIN_OUTPUT_PATH, index=False)
validation_features.to_parquet(VALIDATION_OUTPUT_PATH, index=False)
test_features.to_parquet(TEST_OUTPUT_PATH, index=False)

print(f"Training observations: {len(train_features)}")
print(f"Validation observations: {len(validation_features)}")
print(f"Test observations: {len(test_features)}")
print(f"Features: {len(FEATURE_COLUMNS)}")
