from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
METADATA_DIR = PROJECT_DIR / "datasets" / "metadata"

DB_STATIONS_PATH = METADATA_DIR / "db_station_coordinates_koeln_bonn.csv"
OUTPUT_PATH = METADATA_DIR / "db_station_weather_station_mapping_koeln_bonn.csv"

STUDY_START = pd.Timestamp("2025-11-01")
STUDY_END = pd.Timestamp("2026-08-31")

DWD_COLUMNS = [
    "station_id",
    "from_date",
    "to_date",
    "height_m",
    "lat",
    "lon",
    "station_name",
    "state",
    "release",
]

WEATHER_SOURCES = [
    ("ff", "FF_Stundenwerte_Beschreibung_Stationen.txt"),
    ("tu", "TU_Stundenwerte_Beschreibung_Stationen.txt"),
    ("rr", "RR_Stundenwerte_Beschreibung_Stationen.txt"),
    ("fx", "FX_Stundenwerte_Beschreibung_Stationen.txt"),
]


def read_db_stations() -> pd.DataFrame:
    stations = pd.read_csv(DB_STATIONS_PATH, encoding="utf-8-sig", dtype={"eva": str})
    required_columns = {"eva", "station_name", "latitude", "longitude"}
    missing_columns = required_columns - set(stations.columns)

    if missing_columns:
        raise ValueError(
            "DB station file is missing columns: "
            f"{', '.join(sorted(missing_columns))}"
        )

    stations = stations.copy()
    stations["eva"] = stations["eva"].str.zfill(8)
    stations["latitude"] = pd.to_numeric(stations["latitude"], errors="coerce")
    stations["longitude"] = pd.to_numeric(stations["longitude"], errors="coerce")

    missing_coordinates = stations["latitude"].isna() | stations["longitude"].isna()
    if missing_coordinates.any():
        missing = stations.loc[missing_coordinates, ["eva", "station_name"]]
        raise ValueError(
            "DB station file contains missing coordinates:\n"
            f"{missing.to_string(index=False)}"
        )

    return stations[["eva", "station_name", "latitude", "longitude"]]


def read_dwd_stations(filename: str) -> pd.DataFrame:
    path = METADATA_DIR / filename
    stations = pd.read_fwf(
        path,
        encoding="utf-8",
        skiprows=2,
        header=None,
        names=DWD_COLUMNS,
        dtype={"station_id": str},
    )

    stations = stations.copy()
    stations["station_id"] = stations["station_id"].str.zfill(5)
    stations["from_date"] = pd.to_datetime(
        stations["from_date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )
    stations["to_date"] = pd.to_datetime(
        stations["to_date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )
    stations["lat"] = pd.to_numeric(stations["lat"], errors="coerce")
    stations["lon"] = pd.to_numeric(stations["lon"], errors="coerce")

    valid_coordinates = stations["lat"].notna() & stations["lon"].notna()
    active_in_study_period = (
        (stations["from_date"] <= STUDY_END)
        & (stations["to_date"] >= STUDY_START)
    )

    return stations.loc[valid_coordinates & active_in_study_period].reset_index(
        drop=True
    )


def haversine_km(
    origin_lat: float,
    origin_lon: float,
    candidate_lats: pd.Series,
    candidate_lons: pd.Series,
) -> np.ndarray:
    radius_km = 6371.0088

    origin_lat_rad = np.radians(origin_lat)
    origin_lon_rad = np.radians(origin_lon)
    candidate_lats_rad = np.radians(candidate_lats.to_numpy(dtype=float))
    candidate_lons_rad = np.radians(candidate_lons.to_numpy(dtype=float))

    delta_lat = candidate_lats_rad - origin_lat_rad
    delta_lon = candidate_lons_rad - origin_lon_rad

    a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(origin_lat_rad)
        * np.cos(candidate_lats_rad)
        * np.sin(delta_lon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))

    return radius_km * c


def nearest_weather_station(
    db_station: pd.Series,
    weather_stations: pd.DataFrame,
    prefix: str,
) -> dict[str, object]:
    if weather_stations.empty:
        return {
            f"{prefix}_station_id": pd.NA,
            f"{prefix}_station_name": pd.NA,
            f"{prefix}_distance_km": pd.NA,
        }

    distances = haversine_km(
        origin_lat=float(db_station["latitude"]),
        origin_lon=float(db_station["longitude"]),
        candidate_lats=weather_stations["lat"],
        candidate_lons=weather_stations["lon"],
    )
    nearest_index = int(np.argmin(distances))
    nearest = weather_stations.iloc[nearest_index]

    return {
        f"{prefix}_station_id": nearest["station_id"],
        f"{prefix}_station_name": nearest["station_name"],
        f"{prefix}_distance_km": round(float(distances[nearest_index]), 3),
    }


def build_mapping() -> pd.DataFrame:
    db_stations = read_db_stations()
    weather_stations_by_prefix = {
        prefix: read_dwd_stations(filename) for prefix, filename in WEATHER_SOURCES
    }

    rows = []
    for _, db_station in db_stations.iterrows():
        row = {
            "eva": db_station["eva"],
            "station_name": db_station["station_name"],
            "latitude": db_station["latitude"],
            "longitude": db_station["longitude"],
        }

        for prefix, _ in WEATHER_SOURCES:
            row.update(
                nearest_weather_station(
                    db_station=db_station,
                    weather_stations=weather_stations_by_prefix[prefix],
                    prefix=prefix,
                )
            )

        rows.append(row)

    output_columns = [
        "eva",
        "station_name",
        "latitude",
        "longitude",
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

    return pd.DataFrame(rows, columns=output_columns)


def main() -> None:
    mapping = build_mapping()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("Nearest weather station mapping created")
    print(f"Study period: {STUDY_START.date()} to {STUDY_END.date()}")
    print(f"Rows: {len(mapping)}")
    print()
    print(mapping.to_string(index=False))
    print()
    print(f"File: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
