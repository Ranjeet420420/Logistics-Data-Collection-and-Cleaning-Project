"""
preview.py - Import + quick profiling helpers (shape, dtypes, missing %, duplicates).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def load_flights() -> pd.DataFrame:
    """flights-20k.json is a JSON array of records."""
    # convert_dates=False keeps 'date' as the raw string so parsing is explicit (and auditable)
    return pd.read_json(RAW_DIR / "flights-20k.json", convert_dates=False)


def load_airports() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "airports.csv")


def load_shipments() -> pd.DataFrame:
    # dtype=str: stop pandas guessing types; we parse explicitly during cleaning
    return pd.read_csv(RAW_DIR / "shipments_raw.csv", dtype=str, keep_default_na=False)


def profile(df: pd.DataFrame, name: str) -> dict:
    """Return a compact data-quality profile."""
    empties = df.apply(
        lambda s: s.astype("string").str.strip().str.lower().isin(["", "n/a", "na", "null", "none", "-", "--"]).sum()
        if (s.dtype == object or str(s.dtype).startswith("str")) else 0
    )
    miss = (df.isna().sum() + empties).astype(int)
    return {
        "dataset": name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "missing_or_placeholder": {k: int(v) for k, v in miss.items() if v > 0},
        "unique_per_column": {k: int(v) for k, v in df.nunique(dropna=True).items()},
    }


def preview(df: pd.DataFrame, name: str, n: int = 5) -> None:
    print(f"\n=== {name} ===")
    print(f"shape: {df.shape}")
    print(df.head(n).to_string(index=False))
    print("\ndtypes:\n" + df.dtypes.astype(str).to_string())
    print("\nprofile:\n" + json.dumps(profile(df, name), indent=2))


if __name__ == "__main__":
    preview(load_flights(), "flights-20k")
    preview(load_airports(), "airports")
    preview(load_shipments(), "shipments_raw")
