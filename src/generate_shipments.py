"""
generate_shipments.py - Synthetic "dirty" shipment dataset.

Public logistics datasets are usually already tidy. To practise (and demonstrate)
real-world cleaning problems, this script builds a shipment-level dataset for
Indian road freight and then injects *known* defects with a fixed random seed.
Because the injected defects are known, the cleaning pipeline can be measured
objectively (how many were found / fixed).

Injected defect types
  1. exact duplicate rows
  2. near duplicates (same shipment_id, different case / whitespace)
  3. placeholder / missing values ("N/A", "-", "null", empty)
  4. mixed date formats
  5. inconsistent categorical spellings (carrier, status, city)
  6. weight with mixed units ("12.5 kg", "27 lb") and cost with currency symbols
  7. impossible values (negative weight, delivery before shipping)
  8. extreme outliers (typing errors such as extra zeros)
  9. leading / trailing whitespace
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

SEED = 42
N = 5000

CITIES = {  # city: (lat, lon)
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Jaipur": (26.9124, 75.7873),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
    "Lucknow": (26.8467, 80.9462),
}
CITY_VARIANTS = {
    "Delhi": ["Delhi", "New Delhi", "delhi", "DELHI"],
    "Mumbai": ["Mumbai", "Bombay", "mumbai"],
    "Jaipur": ["Jaipur", "jaipur", "JAIPUR"],
    "Bengaluru": ["Bengaluru", "Bangalore", "BLR", "bengaluru"],
    "Chennai": ["Chennai", "Madras", "chennai"],
    "Kolkata": ["Kolkata", "Calcutta", "kolkata"],
    "Hyderabad": ["Hyderabad", "hyderabad", "Hyd"],
    "Pune": ["Pune", "pune", "Poona"],
    "Ahmedabad": ["Ahmedabad", "ahmedabad", "Ahmadabad", "Ahemdabad"],
    "Lucknow": ["Lucknow", "lucknow", "Lakhnau"],
}
CARRIERS = {
    "BlueDart": ["BlueDart", "Blue Dart", "BLUEDART", "bluedart", "Bluedrat"],
    "Delhivery": ["Delhivery", "delhivery", "DELHIVERY", "Delhivary", "Delhivry"],
    "DTDC": ["DTDC", "dtdc", "D.T.D.C"],
    "Ecom Express": ["Ecom Express", "EcomExpress", "ecom express"],
}
STATUSES = {
    "Delivered": ["Delivered", "delivered", "DELIVRD", "Delivered "],
    "In Transit": ["In Transit", "in transit", "In-Transit", "INTRANSIT"],
    "Returned": ["Returned", "returned", "RTO"],
    "Delayed": ["Delayed", "delayed", "DELAYD"],
}
VEHICLES = ["Truck", "Mini Truck", "Van", "Container"]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def build_clean(rng: np.random.Generator) -> pd.DataFrame:
    names = list(CITIES)
    origin = rng.choice(names, N)
    dest = rng.choice(names, N)
    same = origin == dest
    while same.any():  # an order must go somewhere
        dest[same] = rng.choice(names, same.sum())
        same = origin == dest

    lat1 = np.array([CITIES[c][0] for c in origin])
    lon1 = np.array([CITIES[c][1] for c in origin])
    lat2 = np.array([CITIES[c][0] for c in dest])
    lon2 = np.array([CITIES[c][1] for c in dest])
    dist = haversine_km(lat1, lon1, lat2, lon2) * 1.25  # road factor
    dist = np.round(dist, 1)

    order_date = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, N), unit="D")
    ship_date = order_date + pd.to_timedelta(rng.integers(0, 4, N), unit="D")
    transit_days = np.maximum(1, np.round(dist / 450 + rng.normal(1.0, 0.8, N))).astype(int)
    delivery_date = ship_date + pd.to_timedelta(transit_days, unit="D")

    weight = np.round(rng.lognormal(mean=4.0, sigma=0.8, size=N), 1)  # kg
    cost = np.round(500 + dist * 4.5 + weight * 6 + rng.normal(0, 150, N), 2)
    cost = np.maximum(cost, 300)

    return pd.DataFrame(
        {
            "shipment_id": [f"SHP{100000 + i}" for i in range(N)],
            "order_date": order_date,
            "ship_date": ship_date,
            "delivery_date": delivery_date,
            "origin_city": origin,
            "destination_city": dest,
            "carrier": rng.choice(list(CARRIERS), N, p=[0.3, 0.35, 0.2, 0.15]),
            "vehicle_type": rng.choice(VEHICLES, N, p=[0.35, 0.3, 0.25, 0.1]),
            "status": rng.choice(list(STATUSES), N, p=[0.7, 0.1, 0.07, 0.13]),
            "weight_kg": weight,
            "distance_km": dist,
            "freight_cost_inr": cost,
        }
    )


def inject_defects(df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    d = df.copy()
    n = len(d)
    ids = d["shipment_id"].copy()
    ledger: dict[str, list[str]] = {}

    # Convert to object/strings so we can corrupt them
    for c in ["order_date", "ship_date", "delivery_date"]:
        d[c] = d[c].dt.strftime("%Y-%m-%d")
    for c in ["weight_kg", "distance_km", "freight_cost_inr"]:
        d[c] = d[c].astype(object)

    # 4. mixed date formats
    fmts = ["%Y-%m-%d", "%d/%m/%Y", "%d-%b-%Y", "%m/%d/%Y"]
    for c in ["order_date", "ship_date", "delivery_date"]:
        idx = rng.choice(n, int(n * 0.35), replace=False)
        parsed = pd.to_datetime(d.loc[idx, c])
        d.loc[idx, c] = [ts.strftime(rng.choice(fmts[1:])) for ts in parsed]

    # 5. inconsistent categoricals
    d["origin_city"] = [rng.choice(CITY_VARIANTS[v]) if rng.random() < 0.4 else v for v in d["origin_city"]]
    d["destination_city"] = [rng.choice(CITY_VARIANTS[v]) if rng.random() < 0.4 else v for v in d["destination_city"]]
    d["carrier"] = [rng.choice(CARRIERS[v]) if rng.random() < 0.45 else v for v in d["carrier"]]
    d["status"] = [rng.choice(STATUSES[v]) if rng.random() < 0.4 else v for v in d["status"]]

    # 8. outliers (extra zeros / typing errors) - applied while values are still numeric
    out = rng.choice(n, 20, replace=False)
    d.loc[out, "freight_cost_inr"] = [round(float(v) * 100, 2) for v in d.loc[out, "freight_cost_inr"]]
    out_w = rng.choice(n, 10, replace=False)
    d.loc[out_w, "weight_kg"] = [round(float(v) * 100, 1) for v in d.loc[out_w, "weight_kg"]]
    ledger["cost_outlier"] = ids.iloc[out].tolist()
    ledger["weight_outlier"] = ids.iloc[out_w].tolist()

    # 6. weight with units, cost with currency symbols / commas
    w_idx = rng.choice(n, int(n * 0.3), replace=False)
    for i in w_idx:
        w = d.at[i, "weight_kg"]
        if rng.random() < 0.5:
            d.at[i, "weight_kg"] = f"{w} kg"
        else:
            d.at[i, "weight_kg"] = f"{round(w * 2.20462, 1)} lb"
    c_idx = rng.choice(n, int(n * 0.35), replace=False)
    for i in c_idx:
        v = d.at[i, "freight_cost_inr"]
        d.at[i, "freight_cost_inr"] = rng.choice([f"₹{v:,.2f}", f"Rs. {v:,.0f}", f"INR {v}"])

    # 3. missing values / placeholders
    placeholders = ["N/A", "-", "null", "", "NA", "none"]
    for c, frac in [("weight_kg", 0.05), ("freight_cost_inr", 0.04), ("delivery_date", 0.03),
                    ("distance_km", 0.03), ("vehicle_type", 0.02), ("carrier", 0.01)]:
        idx = rng.choice(n, int(n * frac), replace=False)
        d.loc[idx, c] = rng.choice(placeholders, len(idx))

    # 7. impossible values
    neg = rng.choice(n, 25, replace=False)
    d.loc[neg, "weight_kg"] = "-" + d.loc[neg, "weight_kg"].astype(str)
    ledger["negative_weight"] = ids.iloc[neg].tolist()
    bad = rng.choice(n, 60, replace=False)
    # delivery date earlier than shipping date: shift delivery to the order date minus 1 day
    ord_ts = pd.to_datetime(df["order_date"]).iloc[bad] - pd.Timedelta(days=1)
    d.loc[bad, "delivery_date"] = ord_ts.dt.strftime("%Y-%m-%d").to_numpy()
    ledger["delivery_before_order"] = ids.iloc[bad].tolist()

    # 9. whitespace noise
    for c in ["origin_city", "destination_city", "carrier", "vehicle_type", "shipment_id"]:
        idx = rng.choice(n, int(n * 0.05), replace=False)
        d.loc[idx, c] = d.loc[idx, c].astype(str).map(lambda s: f"  {s} ")

    # 1 & 2. duplicates
    exact = d.sample(150, random_state=SEED)
    near = d.sample(60, random_state=SEED + 1).copy()
    near["carrier"] = near["carrier"].astype(str).str.upper()
    near["origin_city"] = near["origin_city"].astype(str).str.lower() + " "
    d = pd.concat([d, exact, near], ignore_index=True)
    ledger["duplicate_rows_added"] = [len(exact), len(near)]
    return d.sample(frac=1, random_state=SEED).reset_index(drop=True), ledger


def generate() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    clean = build_clean(rng)
    dirty, ledger = inject_defects(clean, rng)
    clean.to_csv(RAW_DIR / "shipments_ground_truth.csv", index=False)  # only for validation
    import json
    (RAW_DIR / "defect_ledger.json").write_text(json.dumps(ledger))
    out = RAW_DIR / "shipments_raw.csv"
    dirty.to_csv(out, index=False)
    print(f"[ok] wrote {out.name}: {len(dirty):,} rows (ground truth: {len(clean):,})")
    return out


if __name__ == "__main__":
    generate()
