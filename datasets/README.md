# Datasets

This directory contains local Parquet input files used by the analysis scripts.

The current files are large:

- `data-2026-01.parquet`
- `data-2026-02.parquet`

Large binary data files are usually better kept outside Git and restored from the original source when needed. The root `.gitignore` ignores future `*.parquet` files for that reason.
