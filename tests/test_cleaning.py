"""Unit tests for the cleaning module. Run:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cleaning as C


def L():
    return C.CleaningLog()


def test_placeholders_become_nan():
    df = pd.DataFrame({"a": ["x", "N/A", "-", "", "null", " none "]})
    out = C.replace_placeholders(df, L())
    assert out["a"].isna().sum() == 5


def test_parse_weight_units():
    df = pd.DataFrame({"w": ["12.5 kg", "22.05 lb", "30", "-4 kg", None]})
    out = C.parse_weight_kg(df, "w", L())["w"]
    assert out.iloc[0] == 12.5
    assert abs(out.iloc[1] - 10.0) < 0.01
    assert out.iloc[2] == 30
    assert out.iloc[3] == -4          # sign preserved; rule validation rejects it later
    assert pd.isna(out.iloc[4])


def test_parse_currency_variants():
    df = pd.DataFrame({"c": ["₹1,250.50", "Rs. 1,250", "INR 1250.5", "1250"]})
    out = C.parse_currency(df, "c", L())["c"].tolist()
    assert out == [1250.5, 1250.0, 1250.5, 1250.0]


def test_parse_dates_mixed_and_fail_guard():
    df = pd.DataFrame({"d": ["2024-01-31", "31/01/2024", "31-Jan-2024", None]})
    out = C.parse_dates(df, ["d"], L())["d"]
    assert out.iloc[:3].nunique() == 1 and pd.isna(out.iloc[3])
    bad = pd.DataFrame({"d": ["garbage"] * 10})
    try:
        C.parse_dates(bad, ["d"], L())
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_ambiguous_date_uses_row_context():
    raw = pd.DataFrame({
        "order_date": ["2024-03-04"], "ship_date": ["2024-03-05"], "delivery_date": ["04/03/2024"],
    })
    cols = list(raw.columns)
    parsed = C.parse_dates(raw, cols, L())            # day-first default -> 4 March
    assert parsed["delivery_date"].iloc[0] == pd.Timestamp("2024-03-04")
    fixed = C.resolve_ambiguous_dates(raw, parsed, cols, L())   # 4 March < ship date -> use 3 April
    assert fixed["delivery_date"].iloc[0] == pd.Timestamp("2024-04-03")


def test_category_normalisation_fuzzy_and_unmapped():
    df = pd.DataFrame({"c": ["Delhivery", "delhivry", "BLUE DART", "Totally Other"]})
    canon = {"Delhivery": [], "BlueDart": ["Blue Dart"]}
    out = C.normalize_categories(df, "c", canon, L())["c"]
    assert out.tolist()[:3] == ["Delhivery", "Delhivery", "BlueDart"]
    assert pd.isna(out.iloc[3])


def test_dedupe_keeps_most_complete():
    df = pd.DataFrame({"id": ["A", "A", "B"], "x": [np.nan, 5, 1], "y": [np.nan, 6, 2]})
    out = C.remove_duplicates(df, "id", L())
    assert len(out) == 2 and out.loc[out["id"] == "A", "x"].iloc[0] == 5


def test_business_rules_flag_impossible_values():
    df = pd.DataFrame({
        "weight_kg": [-1.0, 5.0], "distance_km": [10.0, 10.0], "freight_cost_inr": [100.0, 100.0],
        "order_date": pd.to_datetime(["2024-01-05", "2024-01-05"]),
        "ship_date": pd.to_datetime(["2024-01-06", "2024-01-06"]),
        "delivery_date": pd.to_datetime(["2024-01-07", "2024-01-04"]),
    })
    out = C.validate_business_rules(df, L())
    assert pd.isna(out.loc[0, "weight_kg"])
    assert pd.isna(out.loc[1, "delivery_date"]) and out.loc[1, "flag_date_order"]


def test_group_median_imputation_flags_rows():
    df = pd.DataFrame({"g": ["a", "a", "a", "b"], "v": [1.0, 3.0, np.nan, np.nan]})
    out = C.impute_numeric_group_median(df, "v", ["g"], L())
    assert out.loc[2, "v"] == 2.0 and out["v_imputed"].sum() == 2
    assert not out["v"].isna().any()
