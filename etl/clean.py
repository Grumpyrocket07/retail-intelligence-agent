"""Milestone 1, step 1: clean the UCI Online Retail II workbook.

Reads data/raw/online_retail_II.xlsx and writes:
  data/processed/*.csv            one CSV per target table
  data/processed/expected_counts.json   numbers the load step is validated against
  data/audit/dropped_rows.csv     every row removed by a cleaning rule, with the reason
  data/audit/cleaning_report.json summary of what each rule did

Run:  python -m etl.clean
"""
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "online_retail_II.xlsx"
OUT_DIR = ROOT / "data" / "processed"
AUDIT_DIR = ROOT / "data" / "audit"

COLUMN_MAP = {
    "Invoice": "invoice_no",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "unit_price",
    "Customer ID": "customer_id",
    "Country": "country",
}

# Service / accounting entries that are not products (rule 4).
# Real merchandise with unusual codes (DCGS*, GIFT_0001_*, PADS, SP1002) is deliberately kept.
NON_PRODUCT_CODES = {
    "POST", "DOT", "M", "D", "C2", "C3", "S", "B",
    "BANK CHARGES", "ADJUST", "ADJUST2", "AMAZONFEE", "CRUK",
    "TEST001", "TEST002", "GIFT",
}

OUTLIER_QUANTILE = 0.999  # rule 9: flag, do not drop

log = logging.getLogger("etl.clean")


def load_raw(path: Path) -> tuple[pd.DataFrame, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Put the UCI Online Retail II Excel file there first."
        )
    log.info("Reading %s (Excel is slow, about a minute)...", path.name)
    sheets = pd.read_excel(path, sheet_name=None)
    sheet_rows = {name: len(frame) for name, frame in sheets.items()}
    df = pd.concat(sheets.values(), ignore_index=True)
    missing = set(COLUMN_MAP) - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected columns in workbook, missing: {sorted(missing)}")
    return df.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())], sheet_rows


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Rule 2: trim and uppercase codes, tidy descriptions."""
    df = df.copy()
    df["invoice_no"] = df["invoice_no"].astype(str).str.strip().str.upper()
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
    desc = df["description"].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    df["description"] = desc.mask(desc == "")
    df["country"] = df["country"].astype(str).str.strip()
    df["customer_id"] = df["customer_id"].astype("Int64")
    df["unit_price"] = df["unit_price"].astype(float).round(3)
    return df


def classify(df: pd.DataFrame) -> pd.DataFrame:
    """Rules 3, 4, 5: tag each row with the reason it is removed, or '' to keep."""
    df = df.copy()
    df["is_cancel"] = df["invoice_no"].str.startswith("C")
    df["drop_reason"] = np.select(
        [
            df["stock_code"].isin(NON_PRODUCT_CODES) | df["invoice_no"].str.startswith("A"),
            df["unit_price"] <= 0,
            (~df["is_cancel"]) & (df["quantity"] <= 0),
            df["is_cancel"] & (df["quantity"] >= 0),
        ],
        [
            "non_product_code",
            "zero_or_negative_price",
            "non_cancellation_with_nonpositive_quantity",
            "cancellation_with_nonnegative_quantity",
        ],
        default="",
    )
    return df


def pick_most_common(frame: pd.DataFrame, key: str, value: str, date_col: str) -> pd.DataFrame:
    """Most frequent `value` per `key`; ties go to the most recent one (rules 2 and 8)."""
    counts = (
        frame.dropna(subset=[value])
        .groupby([key, value])
        .agg(n=(value, "size"), last=(date_col, "max"))
        .reset_index()
        .sort_values([key, "n", "last"], ascending=[True, False, False])
    )
    return counts.drop_duplicates(key)[[key, value]]


def build_tables(kept: pd.DataFrame) -> tuple[dict, dict]:
    sales = kept[~kept["is_cancel"]].copy()
    rets = kept[kept["is_cancel"]].copy()

    # ---- orders: one row per invoice (rule 6 keeps anonymous invoices, rule 8 uses earliest timestamp)
    g = sales.groupby("invoice_no")
    assert (g["customer_id"].nunique() <= 1).all(), "an invoice maps to more than one customer"
    assert (g["country"].nunique() == 1).all(), "an invoice maps to more than one country"
    orders = g.agg(
        customer_id=("customer_id", "first"),
        country=("country", "first"),
        order_date=("invoice_date", "min"),
    ).reset_index()

    ret_inv = (
        rets.groupby("invoice_no")
        .agg(customer_id=("customer_id", "first"), country=("country", "first"),
             order_date=("invoice_date", "min"))
        .reset_index()
    )

    # ---- customers: each customer's most frequent country across all their invoices (rule 8)
    all_inv = pd.concat([orders, ret_inv], ignore_index=True).dropna(subset=["customer_id"])
    inv_country = (
        all_inv.groupby(["customer_id", "country"])
        .agg(n=("invoice_no", "nunique"), last=("order_date", "max"))
        .reset_index()
        .sort_values(["customer_id", "n", "last"], ascending=[True, False, False])
        .drop_duplicates("customer_id")
    )

    # ---- countries
    country_names = sorted(set(kept["country"]))
    countries = pd.DataFrame({"country_id": range(1, len(country_names) + 1),
                              "country_name": country_names})
    cid = dict(zip(countries["country_name"], countries["country_id"]))

    customers = pd.DataFrame({
        "customer_id": inv_country["customer_id"].astype("Int64"),
        "country_id": inv_country["country"].map(cid).astype(int),
    }).sort_values("customer_id").reset_index(drop=True)

    orders["country_id"] = orders["country"].map(cid).astype(int)
    orders = orders[["invoice_no", "customer_id", "country_id", "order_date"]]

    # ---- products: most common cleaned description per code
    best = pick_most_common(kept, "stock_code", "description", "invoice_date")
    codes = pd.DataFrame({"stock_code": sorted(set(kept["stock_code"]))})
    products = codes.merge(best, on="stock_code", how="left")
    unknown_desc = int(products["description"].isna().sum())
    products["description"] = products["description"].fillna("UNKNOWN PRODUCT")

    # ---- outlier thresholds (rule 9), measured on sales lines
    qty_cut = float(sales["quantity"].quantile(OUTLIER_QUANTILE))
    price_cut = float(sales["unit_price"].quantile(OUTLIER_QUANTILE))

    def flag(frame):
        return ((frame["quantity"].abs() > qty_cut) | (frame["unit_price"] > price_cut)).astype(int)

    # ---- order_items (rule 7: price lives here)
    sales = sales.sort_values("invoice_no", kind="stable")
    order_items = pd.DataFrame({
        "invoice_no": sales["invoice_no"].values,
        "stock_code": sales["stock_code"].values,
        "quantity": sales["quantity"].values,
        "unit_price": sales["unit_price"].values,
    })
    order_items["line_total"] = (order_items["quantity"] * order_items["unit_price"]).round(3)
    order_items["is_outlier"] = flag(order_items)

    # ---- returns
    returns = pd.DataFrame({
        "invoice_no": rets["invoice_no"].values,
        "customer_id": rets["customer_id"].values,
        "country_id": rets["country"].map(cid).astype(int).values,
        "stock_code": rets["stock_code"].values,
        "quantity": rets["quantity"].abs().values,
        "unit_price": rets["unit_price"].values,
        "return_date": rets["invoice_date"].values,
    })
    returns["line_total"] = (returns["quantity"] * returns["unit_price"]).round(3)
    returns["is_outlier"] = flag(returns)
    returns = returns[["invoice_no", "customer_id", "country_id", "stock_code", "quantity",
                       "unit_price", "line_total", "return_date", "is_outlier"]]

    tables = {
        "countries": countries, "customers": customers, "products": products,
        "orders": orders, "order_items": order_items, "returns": returns,
    }
    extras = {
        "unknown_description_products": unknown_desc,
        "outlier_quantity_threshold": qty_cut,
        "outlier_price_threshold": price_cut,
        "anonymous_orders": int(orders["customer_id"].isna().sum()),
    }
    return tables, extras


def run() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    raw, sheet_rows = load_raw(RAW_FILE)
    report = {"raw_rows": int(len(raw)), "rows_per_sheet": sheet_rows}

    # Rule 1: sheet overlap + exact duplicates
    before = len(raw)
    df = raw.drop_duplicates().reset_index(drop=True)
    report["rule1_exact_duplicates_dropped"] = int(before - len(df))
    log.info("Rule 1: dropped %s exact duplicate rows (includes the sheet overlap)", f"{before - len(df):,}")

    df = classify(normalise(df))

    dropped = df[df["drop_reason"] != ""]
    report["dropped_by_reason"] = {k: int(v) for k, v in dropped["drop_reason"].value_counts().items()}
    for reason, n in report["dropped_by_reason"].items():
        log.info("Dropped %s rows: %s", f"{n:,}", reason)
    dropped.drop(columns=["is_cancel"]).to_csv(AUDIT_DIR / "dropped_rows.csv", index=False)

    kept = df[df["drop_reason"] == ""]
    tables, extras = build_tables(kept)
    report.update(extras)
    report["kept_rows"] = int(len(kept))

    for name, frame in tables.items():
        frame.to_csv(OUT_DIR / f"{name}.csv", index=False)
        log.info("Wrote %-12s %9s rows", name, f"{len(frame):,}")

    oi, rt = tables["order_items"], tables["returns"]
    expected = {
        "rows": {name: int(len(frame)) for name, frame in tables.items()},
        "order_items_line_total_sum": round(float(oi["line_total"].sum()), 3),
        "returns_line_total_sum": round(float(rt["line_total"].sum()), 3),
        "anonymous_orders": extras["anonymous_orders"],
        "outlier_items": int(oi["is_outlier"].sum()),
        "outlier_returns": int(rt["is_outlier"].sum()),
        "first_order": str(tables["orders"]["order_date"].min()),
        "last_order": str(tables["orders"]["order_date"].max()),
    }
    report["expected"] = expected
    (OUT_DIR / "expected_counts.json").write_text(json.dumps(expected, indent=2))
    (AUDIT_DIR / "cleaning_report.json").write_text(json.dumps(report, indent=2))
    log.info("Done in %.0fs. Report: data/audit/cleaning_report.json", time.time() - t0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run()
