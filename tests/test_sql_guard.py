"""The SQL safety check. Run: python -m pytest -q"""
import pytest

from sql_agent.guard import MAX_ROWS, SQLRejected, validate_and_limit


@pytest.mark.parametrize("sql", [
    "SELECT COUNT(*) FROM orders",
    "select * from orders",
    "SELECT * FROM retail_agent.orders",
    "WITH t AS (SELECT customer_id, COUNT(*) n FROM orders GROUP BY customer_id) SELECT * FROM t WHERE n > 10",
    "SELECT country_id FROM customers UNION SELECT country_id FROM orders",
    "SELECT a FROM (SELECT customer_id a FROM orders) x",
    "SELECT DATE_FORMAT(order_date, '%Y-%m'), COUNT(*) FROM orders GROUP BY 1",
    "SELECT 1;",
])
def test_harmless_queries_pass(sql):
    assert validate_and_limit(sql).upper().startswith(("SELECT", "WITH"))


@pytest.mark.parametrize("sql", [
    "DROP TABLE orders", "SELECT 1; DROP TABLE orders", "DELETE FROM returns", "UPDATE products SET description = 'x'",
    "INSERT INTO customers VALUES (1, 1)", "CREATE TABLE t (a INT)", "GRANT ALL ON *.* TO 'x'", "SHOW TABLES", "SET @a = 1",
    "SELECT * FROM mysql.user", "SELECT * FROM information_schema.tables", "SELECT * FROM performance_schema.users",
    "SELECT SLEEP(30)", "SELECT BENCHMARK(100000000, MD5('a'))", "SELECT LOAD_FILE('/etc/passwd')",
    "SELECT VERSION()", "SELECT USER()", "SELECT CURRENT_USER", "SELECT DATABASE()", "SELECT @@version", "SELECT @x",
    "SELECT * FROM orders INTO OUTFILE '/tmp/x'", "SELECT * FROM orders FOR UPDATE",
    "SELECT 1 FROM orders; SELECT 2 FROM orders",
])
def test_dangerous_queries_are_unsafe(sql):
    with pytest.raises(SQLRejected) as err:
        validate_and_limit(sql)
    assert err.value.kind == "unsafe"


@pytest.mark.parametrize("sql", ["SELECT * FROM sales", "SELEC * FROM", "", "   ", "SELECT * FROM orders LIMIT x"])
def test_mistakes_are_fixable_not_unsafe(sql):
    with pytest.raises(SQLRejected) as err:
        validate_and_limit(sql)
    assert err.value.kind == "fixable"


def test_row_limit_is_added_lowered_and_kept():
    assert f"LIMIT {MAX_ROWS}" in validate_and_limit("SELECT * FROM orders")
    assert f"LIMIT {MAX_ROWS}" in validate_and_limit("SELECT * FROM orders LIMIT 99999")
    assert "LIMIT 5" in validate_and_limit("SELECT * FROM orders LIMIT 5")


def test_union_is_wrapped_so_the_limit_covers_everything():
    out = validate_and_limit("SELECT customer_id FROM orders UNION SELECT customer_id FROM returns")
    assert out.upper().startswith("SELECT * FROM (") and f"LIMIT {MAX_ROWS}" in out


def test_comments_are_removed_from_what_runs():
    out = validate_and_limit("SELECT COUNT(*) /* DROP TABLE orders */ FROM orders -- ; DELETE FROM orders")
    assert "DROP" not in out.upper() and "DELETE" not in out.upper() and "--" not in out


def test_cte_names_are_allowed_but_unknown_tables_are_not():
    assert validate_and_limit("WITH big AS (SELECT * FROM orders) SELECT * FROM big")
    with pytest.raises(SQLRejected):
        validate_and_limit("WITH big AS (SELECT * FROM orders) SELECT * FROM secrets")
