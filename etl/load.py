"""Milestone 1, step 2: create the schema and load the cleaned CSVs into MySQL.

Run:  python -m etl.load --reset
(--reset drops and recreates the six tables first; without it, existing tables are kept.)
"""
import argparse
import logging
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from etl.db import get_engine

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SCHEMA_FILE = ROOT / "sql" / "schema.sql"

# Parents before children so foreign keys are satisfied.
LOAD_ORDER = ["countries", "customers", "products", "orders", "order_items", "returns"]
DATE_COLUMNS = {"orders": ["order_date"], "returns": ["return_date"]}
CHUNK = 10_000

log = logging.getLogger("etl.load")


def create_schema(engine, reset: bool) -> None:
    with engine.begin() as conn:
        if reset:
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
            for table in reversed(LOAD_ORDER):
                conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
            log.info("Dropped existing tables")
        statements = [s.strip() for s in SCHEMA_FILE.read_text().split(";") if s.strip()]
        for stmt in statements:
            conn.execute(text(stmt))
    log.info("Schema ready (%d statements)", len(statements))


def read_table(name: str) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / f"{name}.csv", parse_dates=DATE_COLUMNS.get(name, []))
    if "customer_id" in df.columns:
        # nullable ints -> plain Python ints / None so the MySQL driver accepts them
        col = df["customer_id"].astype("Int64")
        df["customer_id"] = col.astype(object).where(col.notna(), None)
    return df


def load_all(engine) -> None:
    for name in LOAD_ORDER:
        t0 = time.time()
        df = read_table(name)
        with engine.begin() as conn:  # one transaction per table: all or nothing
            df.to_sql(name, conn, if_exists="append", index=False, method="multi", chunksize=CHUNK)
        log.info("Loaded %-12s %9s rows in %.0fs", name, f"{len(df):,}", time.time() - t0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="drop and recreate tables first")
    args = parser.parse_args()
    if not (PROCESSED / "expected_counts.json").exists():
        raise SystemExit("No cleaned data found. Run: python -m etl.clean")
    engine = get_engine()
    create_schema(engine, args.reset)
    load_all(engine)
    log.info("Load complete. Next: python -m etl.validate")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
