from pathlib import Path

import pandas as pd


METADATA_DIR = Path(__file__).resolve().parents[2] / "datasets" / "metadata"
INPUT_PATH = METADATA_DIR / "db_station_weather_station_mapping_koeln_bonn.csv"
OUTPUT_PATH = METADATA_DIR / "db_station_weather_station_mapping_koeln_bonn_adjusted.csv"


def main() -> None:
    df = pd.read_csv(
        INPUT_PATH,
        encoding="utf-8-sig",
        dtype={
            "eva": str,
            "ff_station_id": str,
            "tu_station_id": str,
            "rr_station_id": str,
            "fx_station_id": str,
        },
    )

    invalid_rr_ids = {"14180", "19539", "01327"}
    rr_mask = df["rr_station_id"].astype(str).str.zfill(5).isin(invalid_rr_ids)

    df.loc[rr_mask, "rr_station_id"] = df.loc[rr_mask, "ff_station_id"]
    df.loc[rr_mask, "rr_station_name"] = df.loc[rr_mask, "ff_station_name"]
    df.loc[rr_mask, "rr_distance_km"] = df.loc[rr_mask, "ff_distance_km"]

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Adjusted mapping saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
