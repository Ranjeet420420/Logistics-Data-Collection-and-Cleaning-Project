"""
cleaning.py - Reusable data cleaning module for logistics data.

Each step is a small pure function: DataFrame in -> DataFrame out. Every step
appends an entry to a CleaningLog so the pipeline is fully auditable
("what changed, how many rows/cells, why").
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PLACEHOLDERS = {"", "n/a", "na", "null", "none", "-", "--", "nan", "?", "unknown"}
LB_TO_KG = 0.45359237


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
@dataclass
class CleaningLog:
    steps: list[dict] = field(default_factory=list)

    def add(self, step: str, affected: int, detail: str = "") -> None:
        self.steps.append({"step": step, "affected": int(affected), "detail": detail})
        print(f"  - {step:<38} affected={int(affected):>6}  {detail}")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def standardize_columns(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """snake_case, trimmed column names."""
    new = [re.sub(r"\W+", "_", c.strip().lower()).strip("_") for c in df.columns]
    changed = sum(a != b for a, b in zip(df.columns, new))
    df = df.copy()
    df.columns = new
    log.add("standardize_columns", changed)
    return df


def strip_text(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Trim whitespace in every text column."""
    df = df.copy()
    changed = 0
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith("str"):
            s = df[c].astype("string")
            stripped = s.str.strip()
            changed += int((s.fillna("") != stripped.fillna("")).sum())
            df[c] = stripped
    log.add("strip_whitespace", changed)
    return df


def replace_placeholders(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Turn 'N/A', '-', 'null', '' ... into real missing values."""
    df = df.copy()
    total = 0
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith("str"):
            s = df[c].astype("string")
            mask = s.str.strip().str.lower().isin(PLACEHOLDERS)
            total += int(mask.sum())
            df[c] = s.mask(mask, pd.NA)
    log.add("placeholders_to_NaN", total)
    return df


# --------------------------------------------------------------------------- #
# Type / format normalisation
# --------------------------------------------------------------------------- #
DATE_FORMATS = [
    "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M",   # ISO-like (unambiguous)
    "%d-%b-%Y",                                           # 12-Jan-2024 (unambiguous)
    "%d/%m/%Y", "%m/%d/%Y",                               # slash dates: day-first / month-first (AMBIGUOUS)
]
AMBIGUOUS_FORMATS = ("%d/%m/%Y", "%m/%d/%Y")


def parse_dates(df: pd.DataFrame, cols: list[str], log: CleaningLog, max_fail_pct: float = 20.0) -> pd.DataFrame:
    """
    Parse mixed formats; formats tried in fixed priority order (deterministic).
    Slash dates such as 05/06/2024 are ambiguous (5 June vs 6 May) -> day-first wins here;
    use resolve_ambiguous_dates() afterwards to use row context.
    Fails loudly if too many values cannot be parsed (guards against a wrong format list).
    """
    df = df.copy()
    for c in cols:
        raw = df[c].astype("string")
        out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        for fmt in DATE_FORMATS:
            todo = out.isna() & raw.notna()
            if not todo.any():
                break
            out.loc[todo] = pd.to_datetime(raw[todo], format=fmt, errors="coerce")
        failed = int((out.isna() & raw.notna()).sum())
        pct = 100 * failed / max(1, int(raw.notna().sum()))
        if pct > max_fail_pct:
            raise ValueError(f"parse_dates[{c}]: {pct:.1f}% unparseable - check DATE_FORMATS")
        df[c] = out
        log.add(f"parse_dates[{c}]", int(raw.notna().sum()) - failed, f"unparseable={failed}")
    return df


def resolve_ambiguous_dates(
    df_raw: pd.DataFrame, df: pd.DataFrame, cols: list[str], log: CleaningLog, max_gap_days: int = 30
) -> pd.DataFrame:
    """
    Use business logic to resolve dd/mm vs mm/dd ambiguity.
    For each row, build the candidate dates of every slash-formatted cell and pick the first
    combination (day-first preferred) that satisfies  order <= ship <= delivery  AND keeps the
    gap between consecutive stages within max_gap_days (a 10-month "transit" is not plausible).
    Rows with no consistent combination keep the day-first default and are caught later by
    validate_business_rules().
    """
    import itertools

    df = df.copy()
    slash = r"^\d{1,2}/\d{1,2}/\d{4}$"
    amb_cells = {c: df_raw[c].astype("string").str.match(slash, na=False) for c in cols}
    amb_rows = np.where(np.any([m.to_numpy() for m in amb_cells.values()], axis=0))[0]

    def cands(c, i):
        cur = df[c].iloc[i]
        if not amb_cells[c].iloc[i]:
            return [cur]
        txt = df_raw[c].iloc[i]
        out = []
        for fmt in AMBIGUOUS_FORMATS:
            ts = pd.to_datetime(txt, format=fmt, errors="coerce")
            if pd.notna(ts) and ts not in out:
                out.append(ts)
        return out or [cur]

    def consistent(vals):
        known = [v for v in vals if pd.notna(v)]
        return all(a <= b and (b - a).days <= max_gap_days for a, b in zip(known, known[1:]))

    changed = 0
    for i in amb_rows:
        options = [cands(c, i) for c in cols]
        for combo in itertools.product(*options):
            if consistent(combo):
                break
        else:
            continue
        for c, v in zip(cols, combo):
            if pd.notna(v) and df[c].iloc[i] != v:
                df.iloc[i, df.columns.get_loc(c)] = v
                changed += 1
    log.add("resolve_ambiguous_dates", changed, f"rows_with_slash_dates={len(amb_rows)}")
    return df


def parse_weight_kg(df: pd.DataFrame, col: str, log: CleaningLog) -> pd.DataFrame:
    """'12.5 kg', '27 lb', '30' -> float kg. Unparseable -> NaN."""
    df = df.copy()
    s = df[col].astype("string").str.lower().str.replace(",", "", regex=False)
    num = pd.to_numeric(s.str.extract(r"(-?\d+(?:\.\d+)?)")[0], errors="coerce")
    is_lb = s.str.contains(r"\blb", na=False) | s.str.contains("pound", na=False)
    converted = int(is_lb.sum())
    num = num.where(~is_lb, num * LB_TO_KG)
    df[col] = num.round(2)
    log.add(f"parse_weight[{col}]", int(num.notna().sum()), f"lb->kg converted={converted}")
    return df


def parse_currency(df: pd.DataFrame, col: str, log: CleaningLog) -> pd.DataFrame:
    """'₹1,250.00', 'Rs. 1250', 'INR 1250' -> float."""
    df = df.copy()
    s = (
        df[col].astype("string")
        .str.replace(r"(?i)\b(rs|inr)\b\.?|₹", "", regex=True)  # drop currency tokens first ("Rs." has a dot!)
        .str.replace(",", "", regex=False)
        .str.replace(r"[^\d.\-]", "", regex=True)
    )
    df[col] = pd.to_numeric(s, errors="coerce")
    log.add(f"parse_currency[{col}]", int(df[col].notna().sum()))
    return df


def to_numeric(df: pd.DataFrame, col: str, log: CleaningLog) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    log.add(f"to_numeric[{col}]", int(df[col].notna().sum()))
    return df


# --------------------------------------------------------------------------- #
# Categorical normalisation
# --------------------------------------------------------------------------- #
def normalize_categories(
    df: pd.DataFrame, col: str, canonical: dict[str, list[str]], log: CleaningLog,
    cutoff: float = 0.8,
) -> pd.DataFrame:
    """
    Map free-text variants to canonical labels.
    1) exact match on a lower-cased, punctuation-free key (alias table)
    2) fuzzy match (difflib) against canonical names + aliases for typos
    Anything still unknown is left as NaN so it is visible instead of silently wrong.
    """
    key = lambda x: re.sub(r"[^a-z0-9]", "", str(x).lower())
    lookup: dict[str, str] = {}
    for name, aliases in canonical.items():
        lookup[key(name)] = name
        for a in aliases:
            lookup[key(a)] = name

    keys = list(lookup)
    fuzzy_hits = 0

    def _map(x):
        nonlocal fuzzy_hits
        if pd.isna(x):
            return pd.NA
        k = key(x)
        if k in lookup:
            return lookup[k]
        m = difflib.get_close_matches(k, keys, n=1, cutoff=cutoff)
        if m:
            fuzzy_hits += 1
            return lookup[m[0]]
        return pd.NA

    before = df[col].nunique(dropna=True)
    df = df.copy()
    mapped = df[col].map(_map)
    unmapped = int((mapped.isna() & df[col].notna()).sum())
    df[col] = mapped.astype("string")
    log.add(
        f"normalize_categories[{col}]",
        before - df[col].nunique(dropna=True),
        f"unique {before}->{df[col].nunique(dropna=True)}, fuzzy={fuzzy_hits}, unmapped={unmapped}",
    )
    return df


# --------------------------------------------------------------------------- #
# Duplicates
# --------------------------------------------------------------------------- #
def remove_duplicates(df: pd.DataFrame, key: str, log: CleaningLog) -> pd.DataFrame:
    """
    Run AFTER normalisation: near-duplicates ("DELHIVERY" vs "Delhivery") only become
    exact duplicates once formats are harmonised. Keeps the most complete record.
    """
    n0 = len(df)
    exact = int(df.duplicated().sum())
    df = df.copy()
    df["_completeness"] = df.notna().sum(axis=1)
    df = df.sort_values("_completeness", ascending=False, kind="stable")
    df = df.drop_duplicates(subset=[key], keep="first").drop(columns="_completeness")
    df = df.sort_index()
    log.add("remove_duplicates", n0 - len(df), f"exact_dupes_before_key_dedupe={exact}")
    return df


# --------------------------------------------------------------------------- #
# Validation & outliers
# --------------------------------------------------------------------------- #
def validate_business_rules(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Impossible values become NaN (not silently corrected) and are flagged."""
    df = df.copy()

    for col in ["weight_kg", "distance_km", "freight_cost_inr"]:
        bad = df[col] <= 0
        log.add(f"non_positive[{col}] -> NaN", bad.sum())
        df.loc[bad, col] = np.nan

    df["flag_date_order"] = False
    if {"order_date", "ship_date"}.issubset(df.columns):
        bad = df["ship_date"] < df["order_date"]
        df.loc[bad, "flag_date_order"] = True
        df.loc[bad, "ship_date"] = pd.NaT
        log.add("ship_before_order -> NaT", bad.sum())
    if {"ship_date", "delivery_date", "order_date"}.issubset(df.columns):
        ref = df["ship_date"].fillna(df["order_date"])
        bad = df["delivery_date"] < ref
        df.loc[bad, "flag_date_order"] = True
        df.loc[bad, "delivery_date"] = pd.NaT
        log.add("delivery_before_ship -> NaT", bad.sum())
    return df


def iqr_bounds(s: pd.Series, k: float = 3.0) -> tuple[float, float]:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def handle_outliers(df: pd.DataFrame, cols: list[str], log: CleaningLog, k: float = 3.0) -> pd.DataFrame:
    """
    Flag with an IQR fence (k=3 -> only extreme values, avoids deleting genuinely heavy
    shipments), then set to NaN so they are re-imputed from group medians.
    Original value is preserved in <col>_outlier_flag so nothing is lost.
    """
    df = df.copy()
    for c in cols:
        # fence on log scale: freight/weight are right-skewed
        vals = df[c].astype(float).clip(lower=0)
        s = np.log1p(vals.dropna())
        lo, hi = iqr_bounds(s, k)
        lv = np.log1p(vals)
        mask = (lv < lo) | (lv > hi)
        df[f"{c}_outlier_flag"] = mask.fillna(False)
        df.loc[mask.fillna(False), c] = np.nan
        log.add(f"outliers[{c}] -> NaN", mask.sum(), f"log-IQR k={k}, upper~{np.expm1(hi):,.0f}")
    return df


# --------------------------------------------------------------------------- #
# Missing values
# --------------------------------------------------------------------------- #
def impute_numeric_group_median(
    df: pd.DataFrame, col: str, group: list[str], log: CleaningLog
) -> pd.DataFrame:
    """Fill with group median (e.g. cost by route) with global median fallback; flag imputed rows."""
    df = df.copy()
    flag = f"{col}_imputed"
    df[flag] = df[col].isna()
    gm = df.groupby(group)[col].transform("median")
    df[col] = df[col].fillna(gm).fillna(df[col].median())
    log.add(f"impute_median[{col}]", df[flag].sum(), f"by {group}")
    return df


def impute_categorical_mode(df: pd.DataFrame, col: str, log: CleaningLog, fill: str = "Unknown") -> pd.DataFrame:
    """Categorical gaps are labelled explicitly instead of guessing the mode."""
    df = df.copy()
    n = int(df[col].isna().sum())
    df[col] = df[col].fillna(fill)
    log.add(f"fill_unknown[{col}]", n)
    return df


def recompute_distance(df: pd.DataFrame, coords: dict[str, tuple[float, float]], log: CleaningLog) -> pd.DataFrame:
    """Fill missing distance from city coordinates (haversine x 1.25 road factor)."""
    df = df.copy()
    miss = df["distance_km"].isna()
    o = df["origin_city"].map(lambda c: coords.get(c, (np.nan, np.nan)))
    d = df["destination_city"].map(lambda c: coords.get(c, (np.nan, np.nan)))
    lat1, lon1 = o.str[0].astype(float), o.str[1].astype(float)
    lat2, lon2 = d.str[0].astype(float), d.str[1].astype(float)
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    est = 2 * 6371.0 * np.arcsin(np.sqrt(a)) * 1.25
    fill = miss & est.notna()
    df.loc[fill, "distance_km"] = est[fill].round(1)
    log.add("recompute_distance", fill.sum(), "haversine x1.25 from city coordinates")
    return df


def add_derived_features(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    df = df.copy()
    df["transit_days"] = (df["delivery_date"] - df["ship_date"]).dt.days
    df["cost_per_km"] = (df["freight_cost_inr"] / df["distance_km"]).round(2)
    log.add("derived[transit_days,cost_per_km]", len(df))
    return df
