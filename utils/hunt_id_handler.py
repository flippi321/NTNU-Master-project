"""
hunt_id_handler.py

Centralised mapping between HUNT ID formats. Three IDs are tracked:

- ``short`` — NT4MRI 5-char ID (e.g. ``"12345"``). The canonical ID.
- ``long``  — PID115827 13-digit ID (e.g. ``1158270000000``). Used for
  questionnaire / health data.
- ``old``   — Long HUNT3 13-digit ID (e.g. ``9410000000000``). Deprecated;
  scheduled for removal but still appears in some legacy files.

Source mappings:

- ``short`` ↔ ``long`` : ``data/metadata/processed/id_mapping.csv``
  (columns ``NT4MRI``, ``PID115827``).
- ``short`` ↔ ``old``  : ``data/metadata/HUNT4.xlsx`` (columns
  ``HUNT4 MRI Participant number``, ``Long HUNT3 numbers``).

``long`` ↔ ``old`` is derived by routing through ``short``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_DEFAULT_XLSX = Path(__file__).parent / ".." / "data" / "metadata" / "HUNT4.xlsx"
_DEFAULT_CSV  = Path(__file__).parent / ".." / "data" / "metadata" / "processed" / "id_mapping.csv"

# ---------------------------------------------------------------------------
# Internal state (module-level singleton)
# ---------------------------------------------------------------------------

_xlsx_path: Path = _DEFAULT_XLSX
_csv_path:  Path = _DEFAULT_CSV
_df: pd.DataFrame | None = None  # columns: short_id, long_id (nullable), old_id (nullable)


def init(xlsx_path: str | Path | None = None, csv_path: str | Path | None = None) -> None:
    """Override source paths and reset the cached mapping."""
    global _xlsx_path, _csv_path, _df
    if xlsx_path is not None:
        _xlsx_path = Path(xlsx_path)
    if csv_path is not None:
        _csv_path = Path(csv_path)
    _df = None


def _load() -> pd.DataFrame:
    global _df
    if _df is not None:
        return _df

    # short ↔ old from HUNT4.xlsx
    xlsx = pd.read_excel(_xlsx_path).rename(columns={
        "HUNT4 MRI Participant number": "short_id",
        "Long HUNT3 numbers":           "old_id",
    })
    xlsx = xlsx.dropna(subset=["short_id", "old_id"])
    xlsx["short_id"] = xlsx["short_id"].astype(str).str.zfill(5)
    xlsx["old_id"]   = xlsx["old_id"].astype(np.int64)
    xlsx = xlsx[["short_id", "old_id"]]

    # short ↔ long from id_mapping.csv
    csv = pd.read_csv(_csv_path).rename(columns={
        "NT4MRI":    "short_id",
        "PID115827": "long_id",
    })
    csv = csv.dropna(subset=["short_id", "long_id"])
    csv["short_id"] = csv["short_id"].astype(str).str.zfill(5)
    csv["long_id"]  = csv["long_id"].astype(np.int64)
    csv = csv[["short_id", "long_id"]]

    _df = csv.merge(xlsx, on="short_id", how="outer").reset_index(drop=True)
    return _df


def _lookup(from_col: str, to_col: str, key, default):
    df = _load()
    if from_col == "short_id":
        key = str(key).zfill(5)
    else:
        key = int(key)
    row = df[df[from_col] == key]
    if row.empty:
        return default
    val = row[to_col].iloc[0]
    if pd.isna(val):
        return default
    return val if to_col == "short_id" else int(val)


def _map(from_col: str, to_col: str) -> dict:
    df = _load().dropna(subset=[from_col, to_col]).copy()
    if from_col != "short_id":
        df[from_col] = df[from_col].astype(int)
    if to_col != "short_id":
        df[to_col] = df[to_col].astype(int)
    return df.set_index(from_col)[to_col].to_dict()


# ---------------------------------------------------------------------------
# Single-value lookups (all 6 directions)
# ---------------------------------------------------------------------------

def short_to_long(short_id: int | str, default: int | None = None) -> int | None:
    """5-char short_id → 13-digit PID115827 long_id, or *default* if not found."""
    return _lookup("short_id", "long_id", short_id, default)


def short_to_old(short_id: int | str, default: int | None = None) -> int | None:
    """5-char short_id → 13-digit deprecated old_id, or *default* if not found."""
    return _lookup("short_id", "old_id", short_id, default)


def long_to_short(long_id: int | str, default: str | None = None) -> str | None:
    """13-digit PID115827 long_id → 5-char short_id, or *default* if not found."""
    return _lookup("long_id", "short_id", long_id, default)


def long_to_old(long_id: int | str, default: int | None = None) -> int | None:
    """13-digit PID115827 long_id → 13-digit deprecated old_id, or *default*."""
    return _lookup("long_id", "old_id", long_id, default)


def old_to_short(old_id: int | str, default: str | None = None) -> str | None:
    """13-digit deprecated old_id → 5-char short_id, or *default* if not found."""
    return _lookup("old_id", "short_id", old_id, default)


def old_to_long(old_id: int | str, default: int | None = None) -> int | None:
    """13-digit deprecated old_id → 13-digit PID115827 long_id, or *default*."""
    return _lookup("old_id", "long_id", old_id, default)


# ---------------------------------------------------------------------------
# Bulk map accessors (for batch operations)
# ---------------------------------------------------------------------------

def short_to_long_map() -> dict[str, int]:
    """Full dict: short_id (str) → long_id (int)."""
    return _map("short_id", "long_id")


def short_to_old_map() -> dict[str, int]:
    """Full dict: short_id (str) → old_id (int)."""
    return _map("short_id", "old_id")


def long_to_short_map() -> dict[int, str]:
    """Full dict: long_id (int) → short_id (str)."""
    return _map("long_id", "short_id")


def long_to_old_map() -> dict[int, int]:
    """Full dict: long_id (int) → old_id (int)."""
    return _map("long_id", "old_id")


def old_to_short_map() -> dict[int, str]:
    """Full dict: old_id (int) → short_id (str)."""
    return _map("old_id", "short_id")


def old_to_long_map() -> dict[int, int]:
    """Full dict: old_id (int) → long_id (int)."""
    return _map("old_id", "long_id")


# ---------------------------------------------------------------------------
# ID list helpers
# ---------------------------------------------------------------------------

def all_short_ids(skip_first: int = 0) -> list[str]:
    """Sorted list of all short_ids, optionally skipping the first N."""
    df = _load().dropna(subset=["short_id"])
    return sorted(df["short_id"].tolist())[skip_first:]


def all_long_ids(skip_first: int = 0) -> list[int]:
    """Sorted list of all long_ids (PID115827), optionally skipping the first N."""
    df = _load().dropna(subset=["long_id"])
    return sorted(df["long_id"].astype(int).tolist())[skip_first:]


def all_old_ids(skip_first: int = 0) -> list[int]:
    """Sorted list of all deprecated old_ids, optionally skipping the first N."""
    df = _load().dropna(subset=["old_id"])
    return sorted(df["old_id"].astype(int).tolist())[skip_first:]
