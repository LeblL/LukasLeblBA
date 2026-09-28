from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_DIR
    / "datasets"
    / "processed"
    / "modeling_dataset_final_koeln_bonn_2025-11_2026-08.parquet"
)

OUTPUT_DIR = PROJECT_DIR / "datasets" / "modeling" / "splits"
TRAIN_PATH = OUTPUT_DIR / "train.parquet"
VALIDATION_PATH = OUTPUT_DIR / "validation.parquet"
TEST_PATH = OUTPUT_DIR / "test.parquet"
EXCLUDED_PATH = OUTPUT_DIR / "excluded_boundary_rides.parquet"


df = pd.read_parquet(INPUT_PATH)
df["planned_event_time"] = pd.to_datetime(df["planned_event_time"])

validation_start = pd.Timestamp("2026-07-01 00:00:00")
test_start = pd.Timestamp("2026-08-01 00:00:00")
test_end = pd.Timestamp("2026-09-01 00:00:00")

train = df.loc[df["planned_event_time"] < validation_start].copy()
validation = df.loc[
    (df["planned_event_time"] >= validation_start)
    & (df["planned_event_time"] < test_start)
].copy()
test = df.loc[
    (df["planned_event_time"] >= test_start)
    & (df["planned_event_time"] < test_end)
].copy()

train_ride_ids = set(train["concrete_ride_id"])
validation_ride_ids = set(validation["concrete_ride_id"])
test_ride_ids = set(test["concrete_ride_id"])

boundary_ride_ids = (
    (train_ride_ids & validation_ride_ids)
    | (train_ride_ids & test_ride_ids)
    | (validation_ride_ids & test_ride_ids)
)

excluded_boundary_rides = pd.concat(
    [
        train.loc[train["concrete_ride_id"].isin(boundary_ride_ids)],
        validation.loc[validation["concrete_ride_id"].isin(boundary_ride_ids)],
        test.loc[test["concrete_ride_id"].isin(boundary_ride_ids)],
    ],
    ignore_index=True,
)

train = train.loc[~train["concrete_ride_id"].isin(boundary_ride_ids)].copy()
validation = validation.loc[
    ~validation["concrete_ride_id"].isin(boundary_ride_ids)
].copy()
test = test.loc[~test["concrete_ride_id"].isin(boundary_ride_ids)].copy()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train.to_parquet(TRAIN_PATH, index=False)
validation.to_parquet(VALIDATION_PATH, index=False)
test.to_parquet(TEST_PATH, index=False)
excluded_boundary_rides.to_parquet(EXCLUDED_PATH, index=False)

print(f"Training observations: {len(train)}")
print(f"Validation observations: {len(validation)}")
print(f"Test observations: {len(test)}")
print(f"Removed observations: {len(excluded_boundary_rides)}")
