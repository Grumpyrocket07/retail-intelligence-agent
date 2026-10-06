"""Milestone 2a: build the churn dataset.

For each snapshot date we freeze the business on that day:
  - features  = what each customer had done BEFORE the date
  - label     = did they place no order in the WINDOW_DAYS AFTER the date (churned = 1)

Run:  python -m ml.churn_dataset
Writes data/ml/churn_dataset.csv and data/ml/churn_dataset_summary.json
"""
import json
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from etl.db import get_engine
from ml import config

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "ml"

log = logging.getLogger("ml.churn_dataset")

ORDERS_SQL = """
SELECT o.invoice_no, o.customer_id, o.order_date,
       COALESCE(i.order_total, 0) AS order_total,
       COALESCE(i.n_lines, 0)     AS n_lines,
       COALESCE(i.n_products, 0)  AS n_products
FROM orders o
LEFT JOIN (
    SELECT invoice_no, SUM(line_total) AS order_total, COUNT(*) AS n_lines,
           COUNT(DISTINCT stock_code) AS n_products
    FROM order_items GROUP BY invoice_no
) i ON i.invoice_no = o.invoice_no
WHERE o.customer_id IS NOT NULL
"""
RETURNS_SQL = "SELECT customer_id, return_date, line_total FROM returns WHERE customer_id IS NOT NULL"
CUSTOMERS_SQL = """
SELECT c.customer_id, co.country_name
FROM customers c JOIN countries co ON co.country_id = c.country_id
"""

FEATURES = [
    "recency_days", "tenure_days",
    "n_orders_total", "n_orders_90d", "n_orders_180d", "n_orders_365d",
    "total_spend", "spend_90d", "spend_180d", "spend_365d", "avg_order_value",
    "avg_gap_days", "recency_over_avg_gap",
    "avg_basket_lines", "avg_basket_products",
    "n_return_lines", "return_value", "return_ratio",
    "is_uk",
]


def build_features(orders, returns, customers, snapshot, active_within_days=None) -> pd.DataFrame:
    """The clues for every customer who had ordered before `snapshot`. Pure function: no database access.

    Leakage rule: only rows dated strictly before `snapshot` are read. There is no label here, so the same
    function builds both the training data (via build_snapshot) and the live scores served by the API.
    """
    snap = pd.Timestamp(snapshot)
    history = orders[orders["order_date"] < snap]
    past_returns = returns[returns["return_date"] < snap]

    feats = history.groupby("customer_id").agg(
        first_order=("order_date", "min"),
        last_order=("order_date", "max"),
        n_orders_total=("invoice_no", "nunique"),
        total_spend=("order_total", "sum"),
        avg_basket_lines=("n_lines", "mean"),
        avg_basket_products=("n_products", "mean"),
    )
    feats["recency_days"] = (snap - feats["last_order"]).dt.days
    feats["tenure_days"] = (snap - feats["first_order"]).dt.days
    feats["avg_order_value"] = feats["total_spend"] / feats["n_orders_total"]

    for days in (90, 180, 365):
        recent = history[history["order_date"] >= snap - pd.Timedelta(days=days)]
        by_cust = recent.groupby("customer_id")
        feats[f"n_orders_{days}d"] = by_cust["invoice_no"].nunique()
        feats[f"spend_{days}d"] = by_cust["order_total"].sum()
        feats[f"n_orders_{days}d"] = feats[f"n_orders_{days}d"].fillna(0).astype(int)
        feats[f"spend_{days}d"] = feats[f"spend_{days}d"].fillna(0.0)

    # Average gap between distinct order days (NaN for customers with a single order day)
    days_df = (
        history.assign(day=history["order_date"].dt.normalize())
        .drop_duplicates(["customer_id", "day"])
        .sort_values(["customer_id", "day"])
    )
    gaps = days_df.groupby("customer_id")["day"].diff().dt.days
    feats["avg_gap_days"] = gaps.groupby(days_df["customer_id"]).mean()
    ratio = feats["recency_days"] / feats["avg_gap_days"]
    feats["recency_over_avg_gap"] = ratio.where(feats["avg_gap_days"] > 0)

    ret = past_returns.groupby("customer_id").agg(
        n_return_lines=("line_total", "size"), return_value=("line_total", "sum")
    )
    feats = feats.join(ret)
    feats["n_return_lines"] = pd.to_numeric(feats["n_return_lines"], errors="coerce").fillna(0).astype(int)
    feats["return_value"] = pd.to_numeric(feats["return_value"], errors="coerce").fillna(0.0)
    feats["return_ratio"] = (feats["return_value"] / feats["total_spend"]).where(feats["total_spend"] > 0, 0.0)

    country = customers.set_index("customer_id")["country_name"]
    feats["is_uk"] = (feats.index.map(country) == "United Kingdom").astype(int)

    if active_within_days is not None:
        feats = feats[feats["recency_days"] <= active_within_days]

    feats["snapshot_date"] = snap.date().isoformat()
    return feats[["snapshot_date"] + FEATURES].reset_index()


def build_snapshot(orders, returns, customers, snapshot, window_days=config.WINDOW_DAYS,
                   active_within_days=None) -> pd.DataFrame:
    """One row per eligible customer for one snapshot date: the clues (build_features) plus the churn label.

    The label only reads rows from `snapshot` up to (not including) snapshot + window_days.
    """
    snap = pd.Timestamp(snapshot)
    window_end = snap + pd.Timedelta(days=window_days)
    data_end = orders["order_date"].max().normalize() + pd.Timedelta(days=1)
    if window_end > data_end:
        raise ValueError(
            f"Snapshot {snap.date()} + {window_days} days runs past the end of the data "
            f"({data_end.date()}); labels would be wrong. Choose an earlier snapshot."
        )

    feats = build_features(orders, returns, customers, snap, active_within_days)
    future = orders[(orders["order_date"] >= snap) & (orders["order_date"] < window_end)]
    feats["churned"] = (~feats["customer_id"].isin(set(future["customer_id"]))).astype(int)
    return feats


def load_tables(engine):
    with engine.connect() as conn:
        orders = pd.read_sql(text(ORDERS_SQL), conn, parse_dates=["order_date"])
        returns = pd.read_sql(text(RETURNS_SQL), conn, parse_dates=["return_date"])
        customers = pd.read_sql(text(CUSTOMERS_SQL), conn)
    for col in ("order_total",):
        orders[col] = orders[col].astype(float)
    returns["line_total"] = returns["line_total"].astype(float)
    return orders, returns, customers


def build_dataset() -> pd.DataFrame:
    orders, returns, customers = load_tables(get_engine())
    log.info("Loaded %s orders from identified customers", f"{len(orders):,}")

    parts, summary = [], []
    for split, dates in config.SNAPSHOTS.items():
        for snap in dates:
            full = build_snapshot(orders, returns, customers, snap)
            alt = full[full["recency_days"] <= config.COMPARISON_ACTIVE_WITHIN_DAYS]
            chosen = full if config.ACTIVE_WITHIN_DAYS is None else full[full["recency_days"] <= config.ACTIVE_WITHIN_DAYS]
            summary.append({
                "split": split, "snapshot": snap,
                "customers_all": int(len(full)), "churn_rate_all": round(float(full["churned"].mean()), 4),
                f"customers_active_{config.COMPARISON_ACTIVE_WITHIN_DAYS}d": int(len(alt)),
                f"churn_rate_active_{config.COMPARISON_ACTIVE_WITHIN_DAYS}d": round(float(alt["churned"].mean()), 4),
                "customers_used": int(len(chosen)), "churn_rate_used": round(float(chosen["churned"].mean()), 4),
            })
            parts.append(chosen.assign(split=split))

    dataset = pd.concat(parts, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUT_DIR / "churn_dataset.csv", index=False)
    (OUT_DIR / "churn_dataset_summary.json").write_text(json.dumps(
        {"window_days": config.WINDOW_DAYS, "active_within_days": config.ACTIVE_WITHIN_DAYS,
         "features": FEATURES, "snapshots": summary}, indent=2))
    return dataset, summary


def main():
    dataset, summary = build_dataset()
    log.info("Wrote %s rows x %d columns to data/ml/churn_dataset.csv", f"{len(dataset):,}", dataset.shape[1])
    print(pd.DataFrame(summary).to_string(index=False))
    print("\nMissing values per feature (expected: avg_gap_days and recency_over_avg_gap for one-order customers):")
    na = dataset[FEATURES].isna().sum()
    print(na[na > 0].to_string() if (na > 0).any() else "none")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
