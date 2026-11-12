"""
run_pipeline.py - End-to-end: collect -> preview/profile -> clean -> validate -> save.

Usage:  python src/run_pipeline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cleaning as C
import collect
import generate_shipments as G
import preview as P

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

# Cleaning rules (alias tables) are defined independently of the data generator: they encode
# domain knowledge (e.g. Bombay = Mumbai). Typos not listed here must be caught by fuzzy matching.
CITIES = {
    "Delhi": ["New Delhi"], "Mumbai": ["Bombay"], "Jaipur": [], "Bengaluru": ["Bangalore", "BLR"],
    "Chennai": ["Madras"], "Kolkata": ["Calcutta"], "Hyderabad": ["Hyd"], "Pune": ["Poona"],
    "Ahmedabad": [], "Lucknow": ["Lakhnau"],
}
CARRIERS = {
    "BlueDart": ["Blue Dart"], "Delhivery": [], "DTDC": [], "Ecom Express": ["EcomExpress"],
}
STATUSES = {
    "Delivered": [], "In Transit": ["In-Transit"], "Returned": ["RTO"], "Delayed": [],
}


# --------------------------------------------------------------------------- #
# Dataset A: real public flights data
# --------------------------------------------------------------------------- #
def clean_flights(flights: pd.DataFrame, airports: pd.DataFrame) -> tuple[pd.DataFrame, C.CleaningLog]:
    log = C.CleaningLog()
    df = C.standardize_columns(flights, log)
    df = C.strip_text(df, log)
    df = C.replace_placeholders(df, log)
    df = C.parse_dates(df, ["date"], log)

    for col in ["origin", "destination"]:
        df[col] = df[col].astype("string").str.upper()

    # referential integrity: every airport code must exist in the airports reference table
    valid = set(airports["iata"])
    bad = ~(df["origin"].isin(valid) & df["destination"].isin(valid))
    log.add("unknown_airport_codes", bad.sum(), "rows kept but flagged")
    df["flag_unknown_airport"] = bad

    # exact duplicates (no flight number in this extract, so only identical records are dropped)
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log.add("remove_exact_duplicates", n0 - len(df))

    # domain checks
    bad_dist = df["distance"] <= 0
    log.add("non_positive_distance", bad_dist.sum())
    df.loc[bad_dist, "distance"] = np.nan

    # delay outliers: flagged, NOT removed (extreme delays are real operational events)
    lo, hi = C.iqr_bounds(df["delay"].dropna(), k=3.0)
    df["flag_delay_outlier"] = (df["delay"] < lo) | (df["delay"] > hi)
    log.add("flag_delay_outliers", df["flag_delay_outlier"].sum(), f"fence=[{lo:.0f},{hi:.0f}] min")

    # enrichment from airport table
    ap = airports.set_index("iata")[["state", "latitude", "longitude"]]
    df = df.join(ap.add_prefix("origin_"), on="origin").join(ap.add_prefix("dest_"), on="destination")
    df["month"] = df["date"].dt.month
    df["hour"] = df["date"].dt.hour
    df["is_delayed_15"] = df["delay"] > 15
    log.add("enrich_airport_attributes", len(df))
    return df, log


# --------------------------------------------------------------------------- #
# Dataset B: synthetic shipments with injected defects
# --------------------------------------------------------------------------- #
def clean_shipments(raw: pd.DataFrame) -> tuple[pd.DataFrame, C.CleaningLog]:
    log = C.CleaningLog()
    df = C.standardize_columns(raw, log)
    df = C.strip_text(df, log)
    df = C.replace_placeholders(df, log)

    date_cols = ["order_date", "ship_date", "delivery_date"]
    before_parse = df.copy()
    df = C.parse_dates(df, date_cols, log)
    df = C.resolve_ambiguous_dates(before_parse, df, date_cols, log)
    df = C.parse_weight_kg(df, "weight_kg", log)
    df = C.parse_currency(df, "freight_cost_inr", log)
    df = C.to_numeric(df, "distance_km", log)

    df = C.normalize_categories(df, "origin_city", CITIES, log)
    df = C.normalize_categories(df, "destination_city", CITIES, log)
    df = C.normalize_categories(df, "carrier", CARRIERS, log)
    df = C.normalize_categories(df, "status", STATUSES, log)
    df["vehicle_type"] = df["vehicle_type"].str.title()

    df["shipment_id"] = df["shipment_id"].str.upper()
    df = C.remove_duplicates(df, "shipment_id", log)

    df = C.validate_business_rules(df, log)
    df = C.handle_outliers(df, ["freight_cost_inr", "weight_kg"], log, k=3.0)

    coords = G.CITIES
    df = C.recompute_distance(df, coords, log)
    df["route"] = df["origin_city"].astype("string") + "->" + df["destination_city"].astype("string")
    df = C.impute_numeric_group_median(df, "weight_kg", ["vehicle_type"], log)
    df = C.impute_numeric_group_median(df, "freight_cost_inr", ["route"], log)
    for col in ["carrier", "vehicle_type"]:
        df = C.impute_categorical_mode(df, col, log)

    df = C.add_derived_features(df, log)
    return df, log


# --------------------------------------------------------------------------- #
# Validation against known ground truth (possible only because defects are injected)
# --------------------------------------------------------------------------- #
def validate_against_truth(clean: pd.DataFrame, truth: pd.DataFrame, ledger: dict) -> dict:
    """Possible only because defects were injected: compare cleaned output with the known truth."""
    t = truth.copy()
    t["shipment_id"] = t["shipment_id"].str.upper()
    m = clean.merge(t, on="shipment_id", suffixes=("", "_true"))
    res = {"truth_rows": len(t), "clean_rows": len(clean), "matched_ids": len(m),
           "row_count_recovered": len(clean) == len(t)}

    # accuracy of fields that were NOT imputed / flagged as outliers (i.e. parsed from source)
    # cost tolerance is Rs 0.51 because the "Rs. 1,250" source format rounds paise away
    for col, tol in [("weight_kg", 0.1), ("freight_cost_inr", 0.51), ("distance_km", 0.1)]:
        keep = ~m.get(f"{col}_imputed", pd.Series(False, index=m.index)).astype(bool)
        diff = (m.loc[keep, col] - m.loc[keep, f"{col}_true"]).abs()
        res[f"{col}_parsed_correct_pct"] = round(float((diff <= tol).mean() * 100), 2)

    for col in ["carrier", "status", "origin_city", "destination_city"]:
        known = m[col] != "Unknown"
        res[f"{col}_correct_pct"] = round(float((m.loc[known, col] == m.loc[known, f"{col}_true"]).mean() * 100), 2)

    for col in ["order_date", "ship_date", "delivery_date"]:
        ok = m[col].notna()
        tv = pd.to_datetime(m[f"{col}_true"])
        res[f"{col}_correct_pct_of_non_null"] = round(float((m.loc[ok, col] == tv[ok]).mean() * 100), 2)
        res[f"{col}_null_after_cleaning"] = int((~ok).sum())

    # detection quality for injected defects
    def recall(flag_series, ids):
        ids = {i.upper() for i in ids}
        hit = m.loc[m["shipment_id"].isin(ids), flag_series]
        return int(hit.sum()), len(ids)

    for name, flag, ids in [
        ("cost_outliers", "freight_cost_inr_outlier_flag", ledger["cost_outlier"]),
        ("weight_outliers", "weight_kg_outlier_flag", ledger["weight_outlier"]),
    ]:
        hit, tot = recall(flag, ids)
        flagged = int(m[flag].sum())
        res[f"{name}_recall"] = f"{hit}/{tot}"
        res[f"{name}_flagged_total"] = flagged
        res[f"{name}_precision_pct"] = round(100 * hit / max(1, flagged), 1)

    neg = m.loc[m["shipment_id"].isin({i.upper() for i in ledger["negative_weight"]}), "weight_kg_imputed"]
    res["negative_weight_recall"] = f"{int(neg.sum())}/{len(ledger['negative_weight'])}"
    bad = m.loc[m["shipment_id"].isin({i.upper() for i in ledger["delivery_before_order"]}), "delivery_date"]
    res["delivery_before_order_nulled"] = f"{int(bad.isna().sum())}/{len(ledger['delivery_before_order'])}"
    return res


def main() -> None:
    for d in (PROC, REPORTS):
        d.mkdir(parents=True, exist_ok=True)

    print("STEP 1 - COLLECT")
    manifest = collect.collect_all()
    G.generate()

    print("\nSTEP 2 - LOAD & PROFILE")
    flights, airports, ship_raw = P.load_flights(), P.load_airports(), P.load_shipments()
    prof_before = {
        "flights": P.profile(flights, "flights-20k"),
        "airports": P.profile(airports, "airports"),
        "shipments_raw": P.profile(ship_raw, "shipments_raw"),
    }

    print("\nSTEP 3 - CLEAN: flights (real public data)")
    flights_clean, flog = clean_flights(flights, airports)

    print("\nSTEP 4 - CLEAN: shipments (synthetic, injected defects)")
    ship_clean, slog = clean_shipments(ship_raw)

    print("\nSTEP 5 - VALIDATE vs ground truth")
    truth = pd.read_csv(G.RAW_DIR / "shipments_ground_truth.csv")
    ledger = json.loads((G.RAW_DIR / "defect_ledger.json").read_text())
    val = validate_against_truth(ship_clean, truth, ledger)
    print(json.dumps(val, indent=2))

    print("\nSTEP 6 - SAVE")
    flights_clean.to_csv(PROC / "flights_clean.csv", index=False)
    ship_clean.to_csv(PROC / "shipments_clean.csv", index=False)
    flog.to_frame().to_csv(REPORTS / "cleaning_log_flights.csv", index=False)
    slog.to_frame().to_csv(REPORTS / "cleaning_log_shipments.csv", index=False)

    prof_after = {
        "flights_clean": P.profile(flights_clean, "flights_clean"),
        "shipments_clean": P.profile(ship_clean, "shipments_clean"),
    }
    (REPORTS / "quality_report.json").write_text(
        json.dumps(
            {"manifest": manifest, "before": prof_before, "after": prof_after, "validation": val},
            indent=2, default=str,
        )
    )
    print("done.")


if __name__ == "__main__":
    main()
