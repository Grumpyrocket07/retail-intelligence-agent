"""Milestone 1, step 3: prove the database matches what the cleaning step produced.

Run:  python -m etl.validate
Exits with a non-zero status if any check fails.
"""
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from etl.db import get_engine

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ROOT / "data" / "processed" / "expected_counts.json"
EXAMPLES = ROOT / "sql" / "example_queries.sql"
TOLERANCE = 0.01

log = logging.getLogger("etl.validate")


def scalar(conn, sql: str):
    return conn.execute(text(sql)).scalar()


def run_checks(conn, exp: dict) -> list[tuple[str, bool, str]]:
    results = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))

    for table, n in exp["rows"].items():
        got = scalar(conn, f"SELECT COUNT(*) FROM {table}")
        check(f"row count: {table}", got == n, f"db={got:,} expected={n:,}")

    for table, key in [("order_items", "order_items_line_total_sum"), ("returns", "returns_line_total_sum")]:
        got = float(scalar(conn, f"SELECT COALESCE(SUM(line_total),0) FROM {table}"))
        check(f"sum of line_total: {table}", abs(got - exp[key]) <= TOLERANCE,
              f"db={got:,.2f} expected={exp[key]:,.2f}")

    got = scalar(conn, "SELECT COUNT(*) FROM orders WHERE customer_id IS NULL")
    check("anonymous orders", got == exp["anonymous_orders"], f"db={got:,} expected={exp['anonymous_orders']:,}")

    check("outlier items flagged", scalar(conn, "SELECT SUM(is_outlier) FROM order_items") == exp["outlier_items"])
    check("outlier returns flagged", scalar(conn, "SELECT SUM(is_outlier) FROM returns") == exp["outlier_returns"])

    first, last = scalar(conn, "SELECT MIN(order_date) FROM orders"), scalar(conn, "SELECT MAX(order_date) FROM orders")
    check("order date range", str(first) == exp["first_order"] and str(last) == exp["last_order"], f"{first} to {last}")

    orphan_sql = {
        "order_items without an order": "SELECT COUNT(*) FROM order_items oi LEFT JOIN orders o ON o.invoice_no=oi.invoice_no WHERE o.invoice_no IS NULL",
        "order_items without a product": "SELECT COUNT(*) FROM order_items oi LEFT JOIN products p ON p.stock_code=oi.stock_code WHERE p.stock_code IS NULL",
        "orders without items": "SELECT COUNT(*) FROM orders o LEFT JOIN order_items oi ON oi.invoice_no=o.invoice_no WHERE oi.item_id IS NULL",
        "customers with no orders or returns": "SELECT COUNT(*) FROM customers c WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id=c.customer_id) AND NOT EXISTS (SELECT 1 FROM returns r WHERE r.customer_id=c.customer_id)",
        "products never sold or returned": "SELECT COUNT(*) FROM products p WHERE NOT EXISTS (SELECT 1 FROM order_items oi WHERE oi.stock_code=p.stock_code) AND NOT EXISTS (SELECT 1 FROM returns r WHERE r.stock_code=p.stock_code)",
    }
    for name, sql in orphan_sql.items():
        n = scalar(conn, sql)
        check(f"integrity: no {name}", n == 0, f"found {n:,}")

    for table in ("order_items", "returns"):
        bad = scalar(conn, f"SELECT COUNT(*) FROM {table} WHERE quantity <= 0 OR unit_price <= 0 OR ABS(line_total - quantity*unit_price) > 0.01")
        check(f"values valid: {table}", bad == 0, f"bad rows={bad:,}")

    dup = scalar(conn, "SELECT COUNT(*) FROM (SELECT invoice_no FROM orders GROUP BY invoice_no HAVING COUNT(*)>1) d")
    check("integrity: invoice numbers unique", dup == 0)
    return results


def run_examples(conn) -> None:
    blocks = re.split(r"^-- name: (.+)$", EXAMPLES.read_text(), flags=re.M)[1:]
    for title, body in zip(blocks[0::2], blocks[1::2]):
        sql = "\n".join(l for l in body.splitlines() if not l.startswith("--")).strip().rstrip(";")
        print(f"\n== {title.strip()} ==")
        print(pd.read_sql(text(sql), conn).to_string(index=False))


def main() -> None:
    if not EXPECTED.exists():
        raise SystemExit("Missing expected_counts.json. Run: python -m etl.clean")
    exp = json.loads(EXPECTED.read_text())
    engine = get_engine()
    with engine.connect() as conn:
        results = run_checks(conn, exp)
        width = max(len(n) for n, _, _ in results)
        print()
        for name, ok, detail in results:
            print(f"[{'PASS' if ok else 'FAIL'}] {name.ljust(width)}  {detail}")
        failed = [r for r in results if not r[1]]
        print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
        if not failed:
            run_examples(conn)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
