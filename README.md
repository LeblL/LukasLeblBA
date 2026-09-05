# LukasLeblBA

Small Python analysis project using local Parquet datasets.

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

Count unique station names in the January 2026 dataset:

```powershell
python scripts\count_stations.py
```

## Project Structure

- `datasets/` contains local input data files.
- `scripts/` contains runnable analysis scripts.
- `requirements.txt` pins the Python dependencies.

