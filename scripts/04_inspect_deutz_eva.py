from pathlib import Path

import pandas as pd


DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets"
MONTH_FILES = [
    "data-2025-12.parquet",
    "data-2026-01.parquet",
    "data-2026-02.parquet",
    "data-2026-03.parquet",
    "data-2026-04.parquet",
    "data-2026-05.parquet",
]
BASE_COLUMNS = [
    "station_name",
    "xml_station_name",
    "eva",
    "train_type",
    "train_line_ride_id",
    "train_line_station_num",
    "arrival_planned_time",
    "departure_planned_time",
]
TRAIN_NAME_COLUMNS = ["train_name", "train_number"]
STATION_NAME = "K\u00f6ln Messe/Deutz"
EVAS = ["08003368", "08073368"]
SAMPLE_COLUMNS = [
    "eva",
    "xml_station_name",
    "train_type",
    "train_name",
    "train_line_ride_id",
    "train_line_station_num",
    "arrival_planned_time",
    "departure_planned_time",
]


def month_from_filename(filename: str) -> str:
    return filename.removeprefix("data-").removesuffix(".parquet")


def print_table(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    if df.empty:
        print("Keine Daten.")
    else:
        print(df.to_string(index=False))


def read_month(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")

    for train_name_column in TRAIN_NAME_COLUMNS:
        try:
            df = pd.read_parquet(path, columns=BASE_COLUMNS + [train_name_column])
        except Exception as error:
            if "No match for FieldRef.Name" in str(error):
                continue
            raise

        # Einige Monatsdateien nennen den fachlichen train_name train_number.
        return df.rename(columns={train_name_column: "train_name"})

    raise ValueError(f"Keine train_name- oder train_number-Spalte gefunden in: {path}")


def value_counts_table(df: pd.DataFrame, column: str, top_n: int | None = None) -> pd.DataFrame:
    counts = (
        df.groupby(["eva", column], dropna=False)
        .size()
        .reset_index(name="observations")
        .sort_values(["eva", "observations", column], ascending=[True, False, True])
        .reset_index(drop=True)
    )

    if top_n is None:
        return counts

    return counts.groupby("eva", group_keys=False).head(top_n).reset_index(drop=True)


def main() -> None:
    monthly_counts = []
    filtered_parts = []

    for filename in MONTH_FILES:
        month = month_from_filename(filename)
        path = DATASETS_DIR / filename

        df = read_month(path)
        filtered = df[(df["station_name"] == STATION_NAME) & (df["eva"].isin(EVAS))].copy()

        counts = filtered.groupby("eva", dropna=False).size().reset_index(name="observations")
        counts.insert(0, "month", month)
        monthly_counts.append(counts)

        filtered.insert(0, "month", month)
        filtered_parts.append(filtered)

    deutz = pd.concat(filtered_parts, ignore_index=True)

    monthly_result = (
        pd.concat(monthly_counts, ignore_index=True)
        .sort_values(["month", "eva"])
        .reset_index(drop=True)
    )
    print_table("Beobachtungen je Monat und EVA", monthly_result)

    xml_names = value_counts_table(deutz, "xml_station_name")
    print_table("xml_station_name je EVA", xml_names)

    train_types = value_counts_table(deutz, "train_type")
    print_table("train_type-Verteilung je EVA", train_types)

    train_names = value_counts_table(deutz, "train_name", top_n=30)
    print_table("Die 30 haeufigsten train_name-Werte je EVA", train_names)

    train_name_sets = {
        eva: set(deutz.loc[deutz["eva"] == eva, "train_name"].dropna().unique())
        for eva in EVAS
    }
    only_first = train_name_sets[EVAS[0]] - train_name_sets[EVAS[1]]
    only_second = train_name_sets[EVAS[1]] - train_name_sets[EVAS[0]]
    in_both = train_name_sets[EVAS[0]] & train_name_sets[EVAS[1]]

    overlap = pd.DataFrame(
        [
            {"category": f"nur EVA {EVAS[0]}", "train_name_count": len(only_first)},
            {"category": f"nur EVA {EVAS[1]}", "train_name_count": len(only_second)},
            {"category": "bei beiden EVAs", "train_name_count": len(in_both)},
        ]
    )
    print_table("Ueberschneidung der train_name-Werte", overlap)

    station_nums = value_counts_table(deutz, "train_line_station_num", top_n=30)
    print_table("Haeufigste train_line_station_num-Werte je EVA", station_nums)

    samples = (
        deutz.sort_values(["eva", "departure_planned_time", "arrival_planned_time"])
        .groupby("eva", group_keys=False)
        .head(15)
        .loc[:, SAMPLE_COLUMNS]
        .reset_index(drop=True)
    )
    print_table("Beispieldatensaetze je EVA", samples)


if __name__ == "__main__":
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", None):
        main()
