from pathlib import Path

import pandas as pd


DATASET_PATH = Path(__file__).resolve().parents[1] / "datasets" / "data-2026-01.parquet"
COLUMNS = ["station_name", "eva"]

# Eindeutige Suchbegriffe fuer die acht Stationen im Untersuchungsraum.
STATION_SEARCHES = {
    "Koeln Hbf": "Köln Hbf",
    "Koeln Messe/Deutz": "Köln Messe/Deutz",
    "Koeln Sued": "Köln Süd",
    "Koeln/Bonn Flughafen": "Köln/Bonn Flughafen",
    "Bruehl": "Brühl",
    "Bonn Hbf": "Bonn Hbf",
    "Bonn-Beuel": "Bonn-Beuel",
    "Troisdorf": "Troisdorf",
}


def main() -> None:
    df = pd.read_parquet(DATASET_PATH, columns=COLUMNS)

    matches = []
    missing = []

    for station_label, search_term in STATION_SEARCHES.items():
        # Bruehl wird exakt abgeglichen, weil der Teilstring auch Bruehl-Kierberg trifft.
        if station_label == "Bruehl":
            station_matches = df[df["station_name"].str.casefold() == search_term.casefold()]
        else:
            station_matches = df[
                df["station_name"].str.contains(search_term, case=False, na=False, regex=False)
            ]

        if station_matches.empty:
            missing.append(station_label)
            continue

        matches.append(station_matches)

    if matches:
        result = (
            pd.concat(matches, ignore_index=True)
            .drop_duplicates()
            .sort_values("station_name")
            .reset_index(drop=True)
        )
        print(result.to_string(index=False))
    else:
        print("Keine passenden Stationen gefunden.")

    if missing:
        print("\nWarnung: Nicht gefunden:")
        for station in missing:
            print(f"- {station}")


if __name__ == "__main__":
    main()
