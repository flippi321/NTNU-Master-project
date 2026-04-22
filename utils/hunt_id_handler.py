"""
hunt_id_handler.py

Centralised mapping between HUNT ID formats, loaded from HUNT4.xlsx.

The short_id and long_id have no derivable digit relationship — HUNT4.xlsx
is the only authoritative source.

ID formats
----------
long_id  : full 13-digit MR_HUNT_ID (int),  e.g. 9410000010908
short_id : HUNT4 MRI Participant number (str, zero-padded to 5), e.g. "10215"

Usage
-----
    from utils.hunt_id_handler import long_to_short, short_to_long

    short = long_to_short(9410000010908)   # → "10215"
    long  = short_to_long("10215")         # → 9410000010908

    # Override default HUNT4.xlsx path once at startup:
    import utils.hunt_id_handler as hih
    hih.init("path/to/HUNT4.xlsx")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_DEFAULT_XLSX = Path(__file__).parent / ".." / "data" / "metadata" / "HUNT4.xlsx"

# ---------------------------------------------------------------------------
# Internal state (module-level singleton)
# ---------------------------------------------------------------------------

_xlsx_path: Path = _DEFAULT_XLSX
_df: pd.DataFrame | None = None  # columns: long_id (int), short_id (str)


def init(xlsx_path: str | Path) -> None:
    """Override the HUNT4.xlsx path and reset the cached mapping."""
    global _xlsx_path, _df
    _xlsx_path = Path(xlsx_path)
    _df = None


def _load() -> pd.DataFrame:
    global _df
    if _df is not None:
        return _df
    raw = pd.read_excel(_xlsx_path)
    raw = raw.rename(columns={
        "HUNT4 MRI Participant number": "short_id",
        "Long HUNT3 numbers":           "long_id",
    })
    raw = raw.dropna(subset=["long_id"])
    raw["long_id"]  = raw["long_id"].astype(np.int64)
    raw["short_id"] = raw["short_id"].astype(str).str.zfill(5)
    _df = raw[["long_id", "short_id"]].reset_index(drop=True)
    return _df


# ---------------------------------------------------------------------------
# Single-value lookups
# ---------------------------------------------------------------------------

def long_to_short(long_id: int | str, default: str | None = None) -> str | None:
    """long MR_HUNT_ID → 5-char short_id, or *default* if not found."""
    df = _load()
    row = df[df["long_id"] == int(long_id)]
    return row["short_id"].iloc[0] if not row.empty else default


def short_to_long(short_id: int | str, default: int | None = None) -> int | None:
    """5-char short_id → long MR_HUNT_ID (int), or *default* if not found."""
    df = _load()
    row = df[df["short_id"] == str(short_id).zfill(5)]
    return int(row["long_id"].iloc[0]) if not row.empty else default


# ---------------------------------------------------------------------------
# Bulk map accessors (for batch operations)
# ---------------------------------------------------------------------------

def long_to_short_map() -> dict[int, str]:
    """Return full dict: long_id (int) → short_id (str)."""
    df = _load()
    return df.set_index("long_id")["short_id"].to_dict()


def short_to_long_map() -> dict[str, int]:
    """Return full dict: short_id (str) → long_id (int)."""
    df = _load()
    return df.set_index("short_id")["long_id"].to_dict()


# ---------------------------------------------------------------------------
# ID list helpers
# ---------------------------------------------------------------------------

def all_short_ids(skip_first: int = 0) -> list[str]:
    """Sorted list of all short_ids, optionally skipping the first N."""
    df = _load()
    return sorted(df["short_id"].tolist())[skip_first:]


def all_long_ids(skip_first: int = 0) -> list[int]:
    """Sorted list of all long_ids, optionally skipping the first N."""
    df = _load()
    return sorted(df["long_id"].tolist())[skip_first:]
