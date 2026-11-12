"""
make_charts.py - Figures for the report (before/after cleaning).
Run after run_pipeline.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cleaning as C

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREY = "#1f5fa8", "#d9822b", "#8a8f98"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def fig_missing(report: dict) -> None:
    before = report["before"]["shipments_raw"]["missing_or_placeholder"]
    after_df = pd.read_csv(ROOT / "data" / "processed" / "shipments_clean.csv", dtype={"vehicle_type": str})
    cols = ["weight_kg", "freight_cost_inr", "delivery_date", "distance_km", "vehicle_type", "carrier"]
    b = [before.get(c, 0) for c in cols]
    a = [int(after_df[c].isna().sum()) for c in cols]
    x = np.arange(len(cols))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(x - 0.2, b, 0.4, label="Raw (missing + placeholders)", color=ORANGE)
    ax.bar(x + 0.2, a, 0.4, label="After cleaning", color=BLUE)
    ax.set_xticks(x, [c.replace("_", "\n") for c in cols])
    ax.set_ylabel("Records")
    ax.set_title("Missing values per column: before vs after")
    ax.legend(frameon=False)
    for xi, v in zip(x - 0.2, b):
        ax.text(xi, v + 3, str(v), ha="center", fontsize=8)
    for xi, v in zip(x + 0.2, a):
        ax.text(xi, v + 3, str(v), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "missing_before_after.png", dpi=160)
    plt.close(fig)


def fig_cost_outliers() -> None:
    raw = pd.read_csv(ROOT / "data" / "raw" / "shipments_raw.csv", dtype=str, keep_default_na=False)
    log = C.CleaningLog()
    raw_cost = C.parse_currency(raw[["freight_cost_inr"]].replace("", pd.NA), "freight_cost_inr", log)["freight_cost_inr"].dropna()
    clean = pd.read_csv(ROOT / "data" / "processed" / "shipments_clean.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=False)
    axes[0].hist(np.log10(raw_cost.clip(lower=1)), bins=50, color=ORANGE)
    axes[0].set_title("Raw freight cost (log10 INR)")
    axes[0].set_xlabel("log10(cost)")
    axes[0].set_ylabel("Shipments")
    axes[1].hist(np.log10(clean["freight_cost_inr"].clip(lower=1)), bins=50, color=BLUE)
    axes[1].set_title("Cleaned freight cost (log10 INR)")
    axes[1].set_xlabel("log10(cost)")
    fig.suptitle("Extreme values (typing errors) removed before imputation", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "cost_outliers.png", dpi=160)
    plt.close(fig)


def fig_flight_delay() -> None:
    df = pd.read_csv(ROOT / "data" / "processed" / "flights_clean.csv")
    lo, hi = C.iqr_bounds(df["delay"].dropna(), k=3.0)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.hist(df["delay"].clip(-150, 400), bins=120, color=BLUE)
    ax.axvline(lo, color=ORANGE, ls="--")
    ax.axvline(hi, color=ORANGE, ls="--", label=f"IQR fence (k=3): [{lo:.0f}, {hi:.0f}] min")
    ax.set_yscale("log")
    ax.set_xlabel("Departure delay (min, clipped to [-150, 400] for display)")
    ax.set_ylabel("Flights (log scale)")
    ax.set_title("Real data: flight delay distribution with outlier fence")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "flight_delay_outliers.png", dpi=160)
    plt.close(fig)


def fig_log_effect() -> None:
    log = pd.read_csv(ROOT / "reports" / "cleaning_log_shipments.csv")
    # keep only *corrective* steps (parse/derive steps report successes, not fixes)
    corrective = ~log["step"].str.startswith(("parse_", "to_numeric", "derived", "standardize"))
    log = log[corrective & (log["affected"] > 0)].sort_values("affected")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(log["step"], log["affected"], color=BLUE)
    ax.set_xlabel("Records / values affected")
    ax.set_title("Shipment dataset: effect of each cleaning step")
    ax.tick_params(axis="y", labelsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "cleaning_steps_effect.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    report = json.loads((ROOT / "reports" / "quality_report.json").read_text())
    fig_missing(report)
    fig_cost_outliers()
    fig_flight_delay()
    fig_log_effect()
    print("figures written to", FIG)
