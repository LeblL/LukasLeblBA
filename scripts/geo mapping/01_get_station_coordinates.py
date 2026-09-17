import os
from pathlib import Path

import pandas as pd
import requests


API_URL = (
    "https://apis.deutschebahn.com/db-api-marketplace/apis/"
    "station-data/v2/stations"
)
OUTPUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "metadata"
    / "db_station_coordinates_koeln_bonn.csv"
)
TARGET_STATIONS = [
    {"eva": "08000207", "station_name": "Köln Hbf"},
    {"eva": "08003368", "station_name": "Köln Messe/Deutz"},
    {"eva": "08003361", "station_name": "Köln Süd"},
    {"eva": "08003330", "station_name": "Köln/Bonn Flughafen"},
    {"eva": "08001215", "station_name": "Brühl"},
    {"eva": "08000044", "station_name": "Bonn Hbf"},
    {"eva": "08001083", "station_name": "Bonn-Beuel"},
    {"eva": "08000135", "station_name": "Troisdorf"},
]
TARGET_BY_EVA_INT = {int(item["eva"]): item for item in TARGET_STATIONS}


def get_headers() -> dict[str, str]:
    client_id = os.getenv("DB_CLIENT_ID")
    api_key = os.getenv("DB_API_KEY")
    if not client_id or not api_key:
        raise SystemExit("FEHLER: DB_CLIENT_ID oder DB_API_KEY ist nicht gesetzt.")

    return {
        "DB-Client-ID": client_id,
        "DB-Api-Key": api_key,
        "Accept": "application/json",
    }


def get_stations(data: object) -> list[dict]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        stations = data.get("result") or data.get("stations") or []
        if isinstance(stations, dict):
            return stations.get("stations") or []
        return stations

    return []


def extract_coordinates(item: dict) -> tuple[object, object]:
    coordinates = (
        item.get("geographicCoordinates")
        or item.get("geographicCoordinate")
        or {}
    )
    values = coordinates.get("coordinates") if isinstance(coordinates, dict) else None

    if not values or len(values) < 2:
        return pd.NA, pd.NA

    longitude, latitude = values[0], values[1]
    return latitude, longitude


def main() -> None:
    try:
        response = requests.get(API_URL, headers=get_headers(), timeout=30)
    except requests.RequestException as error:
        raise SystemExit(f"FEHLER: StaDa API-Aufruf fehlgeschlagen: {error}") from error

    if not response.ok:
        print("StaDa API-Aufruf fehlgeschlagen")
        print(f"HTTP-Status: {response.status_code}")
        print("Antwort:")
        print(response.text)
        raise SystemExit(1)

    print("StaDa API erfolgreich erreicht")
    print(f"HTTP-Status: {response.status_code}")

    try:
        data = response.json()
    except ValueError as error:
        raise SystemExit("FEHLER: API-Antwort ist kein gültiges JSON.") from error

    if isinstance(data, dict):
        print(f"JSON-Keys: {list(data.keys())}")

    stations = get_stations(data)
    if not stations:
        raise SystemExit("FEHLER: Keine Stationsliste in der API-Antwort gefunden.")

    found_by_eva = {}
    for station in stations:
        eva_numbers = station.get("evaNumbers") or station.get("eva_numbers") or []

        for eva_entry in eva_numbers:
            api_eva = eva_entry.get("number")
            if api_eva is None:
                continue

            try:
                api_eva_int = int(api_eva)
            except (TypeError, ValueError):
                continue

            if api_eva_int not in TARGET_BY_EVA_INT:
                continue

            target = TARGET_BY_EVA_INT[api_eva_int]
            latitude, longitude = extract_coordinates(eva_entry)
            if pd.isna(latitude) or pd.isna(longitude):
                latitude, longitude = extract_coordinates(station)

            found_by_eva[target["eva"]] = {
                "eva": target["eva"],
                "station_name": target["station_name"],
                "latitude": latitude,
                "longitude": longitude,
                "is_main_eva": eva_entry.get(
                    "isMain",
                    eva_entry.get("is_main", pd.NA),
                ),
            }

    rows = [
        found_by_eva[item["eva"]]
        for item in TARGET_STATIONS
        if item["eva"] in found_by_eva
    ]
    result = pd.DataFrame(
        rows,
        columns=[
            "eva",
            "station_name",
            "latitude",
            "longitude",
            "is_main_eva",
        ],
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    searched_evas = {item["eva"] for item in TARGET_STATIONS}
    found_evas = set(result["eva"])
    missing_evas = sorted(searched_evas - found_evas)
    missing_coordinates = result[
        result["latitude"].isna() | result["longitude"].isna()
    ]

    print()
    print("DB-Stationskoordinaten abgerufen")
    print()
    print(f"Gesuchte Stationen: {len(TARGET_STATIONS)}")
    print(f"Gefundene Stationen: {len(result)}")
    print()
    print(
        result[
            [
                "eva",
                "station_name",
                "latitude",
                "longitude",
            ]
        ].to_string(index=False)
    )

    if missing_evas:
        print()
        print("WARNUNG: Folgende EVA-Nummern wurden nicht gefunden:")
        for eva in missing_evas:
            print(eva)

    if not missing_coordinates.empty:
        print()
        print(
            "WARNUNG: "
            f"{len(missing_coordinates)} gefundene Stationen haben keine "
            "vollständigen Koordinaten."
        )

    print()
    print(f"Datei: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
