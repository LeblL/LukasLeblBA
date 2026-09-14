from pathlib import Path

import pandas as pd


def find_project_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "datasets").is_dir():
            return parent

    raise FileNotFoundError("Projektordner mit datasets-Verzeichnis nicht gefunden.")


PROJECT_DIR = find_project_dir()
DATASETS_DIR = PROJECT_DIR / "datasets"
OLD_SCHEMA_DATASETS_DIR = DATASETS_DIR / "old schema datasets"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
DETAIL_OUTPUT_PATH = OUTPUTS_DIR / "05_spnv_service_details.csv"
MONTH_FILES = [
    OLD_SCHEMA_DATASETS_DIR / "old-schema-data-2025-12.parquet",
    OLD_SCHEMA_DATASETS_DIR / "old-schema-data-2026-01.parquet",
    OLD_SCHEMA_DATASETS_DIR / "old-schema-data-2026-02.parquet",
    OLD_SCHEMA_DATASETS_DIR / "old-schema-data-2026-03.parquet",
    OLD_SCHEMA_DATASETS_DIR / "old-schema-data-2026-04.parquet",
    DATASETS_DIR / "data-2026-05.parquet",
]
BASE_COLUMNS = ["station_name", "eva", "train_type"]
TRAIN_NAME_COLUMNS = ["train_number", "train_name"]

# Exakte Stationsnamen des Untersuchungsraums.
STATIONS = [
    "K\u00f6ln Hbf",
    "K\u00f6ln Messe/Deutz",
    "K\u00f6ln S\u00fcd",
    "K\u00f6ln/Bonn Flughafen",
    "Br\u00fchl",
    "Bonn Hbf",
    "Bonn-Beuel",
    "Troisdorf",
]

# Beide EVA-Nummern fuer Koeln Messe/Deutz bleiben in dieser Pruefung enthalten.
DEUTZ_STATION = "K\u00f6ln Messe/Deutz"
DEUTZ_EVAS = ["08003368", "08073368"]


def print_table(title: str, df: pd.DataFrame) -> None:
    if title:
        print(title)
    if df.empty:
        print("Keine Daten.")
    else:
        print(df.to_string(index=False))


def print_section(number: int, title: str) -> None:
    print("\n" + "=" * 50)
    print(f"{number}. {title}")
    print("=" * 50 + "\n")


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
        return normalize_columns(df.rename(columns={train_name_column: "train_name"}))

    raise ValueError(f"Keine train_name- oder train_number-Spalte gefunden in: {path}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # StringDtype erhaelt fehlende Werte als <NA> und verhindert Sortierprobleme
    # bei gemischten train_name-Werten aus train_name und train_number.
    for column in ["station_name", "eva", "train_type", "train_name"]:
        df[column] = df[column].astype("string")

    return df


def aggregate_month(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    filtered = df[df["station_name"].isin(STATIONS)]
    deutz = filtered[
        (filtered["station_name"] == DEUTZ_STATION) & (filtered["eva"].isin(DEUTZ_EVAS))
    ]

    return {
        "train_service": count_observations(filtered, ["train_type", "train_name"]),
        "train_type": count_observations(filtered, ["train_type"]),
        "station_train_type": count_observations(filtered, ["station_name", "train_type"]),
        "deutz_eva_train_type": count_observations(deutz, ["eva", "train_type"]),
    }


def count_observations(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return df.groupby(group_columns, dropna=False).size().reset_index(name="observations")


def combine_counts(parts: list[pd.DataFrame], group_columns: list[str]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=group_columns + ["observations"])

    return (
        pd.concat(parts, ignore_index=True)
        .groupby(group_columns, dropna=False)["observations"]
        .sum()
        .reset_index()
    )


def sort_by_observations(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return (
        df.sort_values(
            ["observations", *group_columns],
            ascending=[False, *([True] * len(group_columns))],
        )
        .reset_index(drop=True)
    )


def sort_within_groups(
    df: pd.DataFrame, leading_columns: list[str], remaining_columns: list[str]
) -> pd.DataFrame:
    return (
        df.sort_values(
            [*leading_columns, "observations", *remaining_columns],
            ascending=[
                *([True] * len(leading_columns)),
                False,
                *([True] * len(remaining_columns)),
            ],
        )
        .reset_index(drop=True)
    )


def add_share_column(
    df: pd.DataFrame,
    share_column: str,
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    result = df.copy()

    if group_columns is None:
        total = result["observations"].sum()
        result[share_column] = (
            (result["observations"] / total * 100).round(2) if total else 0.0
        )
        return result

    totals = result.groupby(group_columns, dropna=False)["observations"].transform("sum")
    result[share_column] = (result["observations"] / totals * 100).round(2)
    return result


def format_value(value: object) -> str:
    return "<NA>" if pd.isna(value) else str(value)


def is_numeric_train_name(series: pd.Series) -> pd.Series:
    return series.astype("string").str.fullmatch(r"\d+").fillna(False)


def save_detail_csv(train_service_counts: pd.DataFrame) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    train_service_counts.to_csv(DETAIL_OUTPUT_PATH, index=False)


def print_train_names_by_type(train_service_counts: pd.DataFrame) -> None:
    if train_service_counts.empty:
        print("Keine Daten.")
        return

    meaningful_counts = train_service_counts[
        ~is_numeric_train_name(train_service_counts["train_name"])
    ]

    if meaningful_counts.empty:
        print("Keine nicht-numerischen train_name-Werte.")
        return

    sorted_counts = sort_by_observations(meaningful_counts, ["train_type", "train_name"])

    for train_type, group in sorted_counts.groupby("train_type", dropna=False, sort=True):
        print(f"\ntrain_type = {format_value(train_type)}")
        print(group.head(10).loc[:, ["train_name", "observations"]].to_string(index=False))


def print_technical_summary(
    train_service_counts: pd.DataFrame,
    train_type_counts: pd.DataFrame,
) -> None:
    unique_train_names = train_service_counts["train_name"].dropna().drop_duplicates()
    numeric_train_name_count = int(is_numeric_train_name(unique_train_names).sum())
    non_numeric_train_name_count = len(unique_train_names) - numeric_train_name_count

    print(f"Gesamtbeobachtungen: {int(train_type_counts['observations'].sum())}")
    print(f"Eindeutige train_type: {train_type_counts['train_type'].nunique(dropna=True)}")
    print(f"Eindeutige train_name: {len(unique_train_names)}")
    print(f"Rein numerische train_name: {numeric_train_name_count}")
    print(f"Nicht-numerische train_name: {non_numeric_train_name_count}")


def main() -> None:
    monthly_train_service_counts = []
    monthly_train_type_counts = []
    monthly_station_train_type_counts = []
    monthly_deutz_eva_train_type_counts = []

    for path in MONTH_FILES:
        df = read_month(path)
        monthly_counts = aggregate_month(df)

        # Ab hier werden pro Monat nur noch gefilterte Aggregationen behalten.
        monthly_train_service_counts.append(monthly_counts["train_service"])
        monthly_train_type_counts.append(monthly_counts["train_type"])
        monthly_station_train_type_counts.append(monthly_counts["station_train_type"])
        monthly_deutz_eva_train_type_counts.append(monthly_counts["deutz_eva_train_type"])

    train_service_counts = sort_by_observations(
        combine_counts(monthly_train_service_counts, ["train_type", "train_name"]),
        ["train_type", "train_name"],
    )
    save_detail_csv(train_service_counts)

    train_type_counts = sort_by_observations(
        combine_counts(monthly_train_type_counts, ["train_type"]),
        ["train_type"],
    )
    train_type_distribution = add_share_column(train_type_counts, "share_percent")
    print_section(1, "Gesamtverteilung der train_type-Werte")
    print_table("", train_type_distribution)

    print_section(2, "Haeufigste aussagekraeftige train_name je train_type")
    print_train_names_by_type(train_service_counts)

    station_train_type_counts = sort_within_groups(
        combine_counts(
            monthly_station_train_type_counts, ["station_name", "train_type"]
        ),
        ["station_name"],
        ["train_type"],
    )
    station_train_type_distribution = add_share_column(
        station_train_type_counts,
        "share_within_station_percent",
        ["station_name"],
    )
    print_section(3, "train_type-Verteilung je Station")
    print_table("", station_train_type_distribution)

    deutz_eva_train_type_counts = sort_within_groups(
        combine_counts(monthly_deutz_eva_train_type_counts, ["eva", "train_type"]),
        ["eva"],
        ["train_type"],
    )
    deutz_eva_train_type_distribution = add_share_column(
        deutz_eva_train_type_counts,
        "share_within_eva_percent",
        ["eva"],
    )
    print_section(4, "Koeln Messe/Deutz nach EVA")
    print_table("", deutz_eva_train_type_distribution)

    print_section(5, "Technische Zusammenfassung")
    print_technical_summary(train_service_counts, train_type_counts)

    unique_train_types = (
        train_type_counts["train_type"].dropna().sort_values().reset_index(drop=True)
    )
    print_section(6, "Eindeutige train_type-Werte")
    if unique_train_types.empty:
        print("Keine Daten.")
    else:
        print(", ".join(format_value(value) for value in unique_train_types))

    print("\nDetaildatei gespeichert unter:")
    print(DETAIL_OUTPUT_PATH.relative_to(PROJECT_DIR))


if __name__ == "__main__":
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        None,
    ):
        main()
