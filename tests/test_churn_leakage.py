"""Tests that the churn dataset cannot see the future. Run: python -m pytest -q"""
import pandas as pd
import pytest

from ml.churn_dataset import FEATURES, build_snapshot

SNAP = "2011-03-01"


def make_orders(rows):
    df = pd.DataFrame(rows, columns=["invoice_no", "customer_id", "order_date", "order_total", "n_lines", "n_products"])
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


def make_returns(rows=()):
    df = pd.DataFrame(rows, columns=["customer_id", "return_date", "line_total"])
    df["return_date"] = pd.to_datetime(df["return_date"])
    return df


CUSTOMERS = pd.DataFrame({"customer_id": [1, 2, 3, 4], "country_name": ["United Kingdom", "France", "United Kingdom", "Germany"]})

BASE_ORDERS = [
    # customer 1: orders before AND inside the 90-day window -> not churned
    ("A1", 1, "2011-01-10 10:00", 100.0, 5, 5),
    ("A2", 1, "2011-02-10 10:00", 50.0, 2, 2),
    ("A3", 1, "2011-04-01 10:00", 70.0, 3, 3),
    # customer 2: orders only before -> churned
    ("B1", 2, "2010-12-20 10:00", 200.0, 8, 6),
    # customer 3: first order only after the snapshot -> must not appear at all
    ("C1", 3, "2011-03-15 10:00", 30.0, 1, 1),
    # padding order so the data extends past snapshot + 90 days
    ("D1", 4, "2011-07-01 10:00", 10.0, 1, 1),
]


def test_label_and_eligibility():
    out = build_snapshot(make_orders(BASE_ORDERS), make_returns(), CUSTOMERS, SNAP, window_days=90).set_index("customer_id")
    assert set(out.index) == {1, 2}, "customers with no history must not appear"
    assert out.loc[1, "churned"] == 0
    assert out.loc[2, "churned"] == 1


def test_features_ignore_everything_after_the_snapshot():
    base = build_snapshot(make_orders(BASE_ORDERS), make_returns(), CUSTOMERS, SNAP).set_index("customer_id")
    # add big orders and a return far in the future for customers 1 and 2
    future = BASE_ORDERS + [
        ("F1", 1, "2011-03-20 10:00", 9999.0, 50, 40),
        ("F2", 2, "2011-05-05 10:00", 8888.0, 40, 30),
    ]
    changed = build_snapshot(make_orders(future), make_returns([(1, "2011-04-02", 500.0)]), CUSTOMERS, SNAP).set_index("customer_id")
    pd.testing.assert_frame_equal(base[FEATURES].loc[[1, 2]], changed[FEATURES].loc[[1, 2]])
    assert changed.loc[2, "churned"] == 0, "the label, unlike the features, should react to future orders"


def test_order_on_snapshot_day_counts_as_future_not_history():
    rows = BASE_ORDERS + [("E1", 2, "2011-03-01 00:00", 40.0, 1, 1)]
    out = build_snapshot(make_orders(rows), make_returns(), CUSTOMERS, SNAP).set_index("customer_id")
    assert out.loc[2, "n_orders_total"] == 1   # the snapshot-day order is not a feature...
    assert out.loc[2, "churned"] == 0          # ...but it does decide the label


def test_returns_after_snapshot_are_ignored():
    out = build_snapshot(make_orders(BASE_ORDERS), make_returns([(1, "2011-02-20", 25.0), (1, "2011-03-05", 99.0)]),
                         CUSTOMERS, SNAP).set_index("customer_id")
    assert out.loc[1, "return_value"] == 25.0
    assert out.loc[1, "n_return_lines"] == 1


def test_snapshot_too_close_to_end_of_data_is_rejected():
    with pytest.raises(ValueError):
        build_snapshot(make_orders(BASE_ORDERS), make_returns(), CUSTOMERS, "2011-06-20", window_days=90)


def test_active_filter_drops_dormant_customers():
    out = build_snapshot(make_orders(BASE_ORDERS), make_returns(), CUSTOMERS, SNAP, active_within_days=30).set_index("customer_id")
    assert set(out.index) == {1}  # customer 2 last ordered 71 days before the snapshot
