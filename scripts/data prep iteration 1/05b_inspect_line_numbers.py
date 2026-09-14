from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def find_project_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "datasets").is_dir():
            return parent

    raise FileNotFoundError("Projektordner mit datasets-Verzeichnis nicht gefunden.")


PROJECT_DIR = find_project_dir()
DATASETS_DIR = PROJECT_DIR / "datasets"
OLD_SCHEMA_DATASETS_DIR = DATASETS_DIR / "old schema datasets"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
DETAIL_OUTPUT_PATH = OUTPUTS_DIR / "05b_line_number_details.csv"

MONTH_FILES = [
    OLD_SCHEMA_DATASETS_DIR / "old-schema-data-2025-12.parquet",
    OLD_SCHEMA_DATASETS_DIR / "old-schema-data-2026-01.parquet",
    OLD_SCHEMA_DATASETS_DIR / "old-schema-data-2026-02.parquet",
    OLD_SCHEMA_DATASETS_DIR / "old-schema-data-2026-03.parquet",
    OLD_SCHEMA_DATASETS_DIR / "old-schema-data-2026-04.parquet",
    DATASETS_DIR / "data-2026-05.parquet",
]

REQUIRED_COLUMNS = ["station_name", "eva", "train_type", "line_number"]
TRAIN_NAME_COLUMNS = ["train_name", "train_number"]

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

RELEVANT_TRAIN_TYPES = ["S", "RB", "RE", "NX", "TR", "TRI", "EST", "EUR", "MSM", "UEX"]
STATION_CHECK_TRAIN_TYPES = ["NX", "TR", "TRI"]


def month_from_path(path: Path) -> str:
    return path.stem.removeprefix("old-schema-data-").removeprefix("data-")


def print_section(number: int, title: str) -> None:
    print("\n" + "=" * 50)
    print(f"{number}. {title}")
    print("=" * 50 + "\n")


def print_table(df: pd.DataFrame) -> None:
    if df.empty:
        print("Keine Daten.")
    else:
        print(df.to_string(index=False))


def read_schema_columns(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")

    return set(pq.read_schema(path).names)


def inspect_schemas() -> pd.DataFrame:
    rows = []

    for path in MONTH_FILES:
        columns = read_schema_columns(path)

        rows.append(
            {
                "month": month_from_path(path),
                "path": path,
                "line_number_available": "line_number" in columns,
                "train_name_available": "train_name" in columns,
                "train_number_available": "train_number" in columns,
            }
        )

    return pd.DataFrame(rows)


def read_month(path: Path, schema_row: pd.Series) -> pd.DataFrame:
    columns = REQUIRED_COLUMNS.copy()

    if bool(schema_row["train_name_available"]):
        columns.append("train_name")
        train_name_column = "train_name"
    elif bool(schema_row["train_number_available"]):
        columns.append("train_number")
        train_name_column = "train_number"
    else:
        train_name_column = None

    df = pd.read_parquet(path, columns=columns)

    if train_name_column is not None:
        df = df.rename(columns={train_name_column: "train_name"})

    return normalize_columns(df)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        df[column] = df[column].astype("string")

    return df


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


def sort_by_train_type_and_observations(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(
            ["train_type", "observations", "line_number"],
            ascending=[True, False, True],
        )
        .reset_index(drop=True)
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


def top_n_per_group(df: pd.DataFrame, group_columns: list[str], top_n: int) -> pd.DataFrame:
    return df.groupby(group_columns, dropna=False, group_keys=False).head(top_n)


def format_value(value: object) -> str:
    return "<NA>" if pd.isna(value) else str(value)


def filter_study_area(df: pd.DataFrame) -> pd.DataFrame:
    # Die Deutz-EVA-Nummern werden nicht ausgeschlossen; die Stationsfilterung
    # bleibt exakt auf die acht Untersuchungsstationen begrenzt.
    return df[df["station_name"].isin(STATIONS)]


def aggregate_month(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    filtered = filter_study_area(df)
    deutz = filtered[
        (filtered["station_name"] == DEUTZ_STATION) & (filtered["eva"].isin(DEUTZ_EVAS))
    ]
    station_check = filtered[filtered["train_type"].isin(STATION_CHECK_TRAIN_TYPES)]

    return {
        "line_number": count_observations(filtered, ["train_type", "line_number"]),
        "coverage": build_coverage_counts(filtered),
        "station_check": count_observations(
            station_check, ["station_name", "train_type", "line_number"]
        ),
        "deutz_eva": count_observations(deutz, ["eva", "train_type", "line_number"]),
    }


def build_coverage_counts(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.assign(with_line_number=df["line_number"].notna())
        .groupby(["train_type", "with_line_number"], dropna=False)
        .size()
        .reset_index(name="observations")
    )


def build_coverage_summary(coverage_counts: pd.DataFrame) -> pd.DataFrame:
    if coverage_counts.empty:
        return pd.DataFrame(
            columns=[
                "train_type",
                "observations",
                "with_line_number",
                "missing_line_number",
                "coverage_percent",
            ]
        )

    summary = (
        coverage_counts.pivot_table(
            index="train_type",
            columns="with_line_number",
            values="observations",
            aggfunc="sum",
            fill_value=0,
            dropna=False,
        )
        .rename(columns={False: "missing_line_number", True: "with_line_number"})
        .reset_index()
    )

    for column in ["with_line_number", "missing_line_number"]:
        if column not in summary:
            summary[column] = 0

    summary["observations"] = summary["with_line_number"] + summary["missing_line_number"]
    summary["coverage_percent"] = (
        summary["with_line_number"] / summary["observations"] * 100
    ).round(2)

    return (
        summary.loc[
            :,
            [
                "train_type",
                "observations",
                "with_line_number",
                "missing_line_number",
                "coverage_percent",
            ],
        ]
        .sort_values(["observations", "train_type"], ascending=[False, True])
        .reset_index(drop=True)
    )


def print_relevant_train_types(line_number_counts: pd.DataFrame) -> None:
    for train_type in RELEVANT_TRAIN_TYPES:
        group = line_number_counts[line_number_counts["train_type"] == train_type]
        group = sort_by_observations(group, ["line_number"])

        print(f"\ntrain_type = {train_type}")
        print_table(group.loc[:, ["line_number", "observations"]])


def print_technical_summary(
    line_number_counts: pd.DataFrame, coverage_summary: pd.DataFrame
) -> None:
    total_observations = int(coverage_summary["observations"].sum())
    with_line_number = int(coverage_summary["with_line_number"].sum())
    missing_line_number = int(coverage_summary["missing_line_number"].sum())
    coverage_percent = (
        round(with_line_number / total_observations * 100, 2)
        if total_observations
        else 0.0
    )
    unique_line_numbers = line_number_counts["line_number"].nunique(dropna=True)

    print(f"Gesamtzahl Beobachtungen: {total_observations}")
    print(f"Beobachtungen mit vorhandener line_number: {with_line_number}")
    print(f"Beobachtungen ohne line_number: {missing_line_number}")
    print(f"Anteil mit vorhandener line_number in Prozent: {coverage_percent}")
    print(f"Anzahl eindeutiger line_number-Werte: {unique_line_numbers}")


def save_detail_csv(line_number_counts: pd.DataFrame) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    line_number_counts.to_csv(DETAIL_OUTPUT_PATH, index=False)


def main() -> None:
    schema_overview = inspect_schemas()

    print_section(1, "Schema-Pruefung")
    print_table(
        schema_overview.loc[
            :,
            [
                "month",
                "line_number_available",
                "train_name_available",
                "train_number_available",
            ],
        ]
    )

    available_months = schema_overview[schema_overview["line_number_available"]]
    missing_months = schema_overview[~schema_overview["line_number_available"]]

    if available_months.empty:
        print("\nline_number ist in den untersuchten Parquet-Dateien nicht vorhanden.")
        return

    if not missing_months.empty:
        print("\nline_number fehlt in folgenden Monaten:")
        print(", ".join(missing_months["month"]))
        print("Die folgenden Analysen verwenden nur Monate mit vorhandener line_number-Spalte.")

    monthly_line_number_counts = []
    monthly_coverage_counts = []
    monthly_station_check_counts = []
    monthly_deutz_eva_counts = []

    for _, schema_row in available_months.iterrows():
        path = schema_row["path"]
        df = read_month(path, schema_row)
        monthly_counts = aggregate_month(df)

        # Pro Monat werden nur die gefilterten Aggregationen behalten.
        monthly_line_number_counts.append(monthly_counts["line_number"])
        monthly_coverage_counts.append(monthly_counts["coverage"])
        monthly_station_check_counts.append(monthly_counts["station_check"])
        monthly_deutz_eva_counts.append(monthly_counts["deutz_eva"])

    line_number_counts = sort_by_train_type_and_observations(
        combine_counts(monthly_line_number_counts, ["train_type", "line_number"])
    )
    save_detail_csv(line_number_counts)

    print_section(2, "Gesamtuebersicht train_type und line_number")
    line_number_top = top_n_per_group(line_number_counts, ["train_type"], top_n=20)
    print_table(line_number_top)

    coverage_counts = combine_counts(
        monthly_coverage_counts, ["train_type", "with_line_number"]
    )
    coverage_summary = build_coverage_summary(coverage_counts)

    print_section(3, "Qualitaet von line_number")
    print_technical_summary(line_number_counts, coverage_summary)
    print()
    print_table(coverage_summary)

    print_section(4, "Besonders relevante train_type-Werte")
    print_relevant_train_types(line_number_counts)

    station_check_counts = sort_within_groups(
        combine_counts(
            monthly_station_check_counts, ["station_name", "train_type", "line_number"]
        ),
        ["station_name", "train_type"],
        ["line_number"],
    )
    station_check_top = top_n_per_group(
        station_check_counts, ["station_name", "train_type"], top_n=10
    ).reset_index(drop=True)

    print_section(5, "Stationsspezifische Pruefung fuer NX, TR und TRI")
    print_table(station_check_top)

    deutz_eva_counts = sort_within_groups(
        combine_counts(monthly_deutz_eva_counts, ["eva", "train_type", "line_number"]),
        ["eva"],
        ["train_type", "line_number"],
    )
    deutz_eva_top = top_n_per_group(deutz_eva_counts, ["eva"], top_n=20).reset_index(
        drop=True
    )

    print_section(6, "Koeln Messe/Deutz nach EVA")
    print_table(deutz_eva_top)

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
