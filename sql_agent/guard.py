"""Safety check for SQL written by an AI.

The question and the AI's answer are both untrusted. This module is one of the layers between them and the database:
  1. the SQL must parse as exactly ONE statement, and that statement must be a plain SELECT (WITH / UNION allowed);
  2. only the six project tables may be read (no system schemas such as mysql or information_schema);
  3. dangerous functions (SLEEP, LOAD_FILE, VERSION, ...) and SELECT ... INTO / FOR UPDATE are refused;
  4. a row limit is always added (or lowered) to MAX_ROWS.
What is executed is the SQL regenerated from the checked syntax tree, never the original text, so what was
checked is exactly what runs (comments are dropped on the way). The read-only database user is the final layer.
"""
import re
from typing import Optional

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

ALLOWED_TABLES = frozenset({"countries", "customers", "products", "orders", "order_items", "returns"})
DATABASE_NAME = "retail_agent"
MAX_ROWS = 200
MAX_SQL_CHARS = 4000


class SQLRejected(ValueError):
    """kind='unsafe': never retry, the request itself is a problem. kind='fixable': the AI may try again."""

    def __init__(self, reason: str, kind: str = "unsafe"):
        super().__init__(reason)
        self.reason, self.kind = reason, kind


def _types(*names):
    return tuple(getattr(exp, n) for n in names if hasattr(exp, n))


FORBIDDEN_NODES = _types(
    "Insert", "Update", "Delete", "Drop", "Create", "Alter", "Command", "Merge", "TruncateTable", "Set", "Use",
    "Describe", "Show", "Grant", "Revoke", "Into", "Lock", "Parameter", "SessionParameter", "LoadData", "Kill",
    "Transaction", "Commit", "Rollback", "Copy", "Pragma",
)
SET_OPERATIONS = _types("SetOperation", "Union", "Intersect", "Except")

# Names are compared upper-case with underscores removed, so LOAD_FILE and loadfile both match.
BLOCKED_FUNCTIONS = frozenset({
    "SLEEP", "BENCHMARK", "LOADFILE", "GETLOCK", "RELEASELOCK", "ISFREELOCK", "ISUSEDLOCK", "MASTERPOSWAIT",
    "VERSION", "CURRENTVERSION", "USER", "CURRENTUSER", "SESSIONUSER", "SYSTEMUSER", "DATABASE", "SCHEMA",
    "CURRENTSCHEMA", "CONNECTIONID", "LASTINSERTID", "EXTRACTVALUE", "UPDATEXML", "SYSEXEC", "SYSEVAL", "PGSLEEP",
})

_OUTFILE = re.compile(r"\binto\s+(outfile|dumpfile)\b", re.I)


def _norm(name: str) -> str:
    return (name or "").upper().replace("_", "")


def validate_and_limit(sql: str, max_rows: int = MAX_ROWS) -> str:
    """Returns the SQL that is safe to run, or raises SQLRejected."""
    if not sql or not sql.strip():
        raise SQLRejected("The SQL was empty.", "fixable")
    if len(sql) > MAX_SQL_CHARS:
        raise SQLRejected(f"The SQL is longer than {MAX_SQL_CHARS} characters.", "fixable")
    if _OUTFILE.search(sql):
        raise SQLRejected("Writing query results to files is not allowed.")

    try:
        statements = [s for s in sqlglot.parse(sql, read="mysql") if s is not None]
    except SqlglotError as exc:
        raise SQLRejected(f"The SQL could not be parsed as MySQL: {str(exc).splitlines()[0][:200]}", "fixable")
    if len(statements) != 1:
        raise SQLRejected("Exactly one SQL statement is allowed per request.")
    root = statements[0]

    if not isinstance(root, (exp.Select,) + SET_OPERATIONS):
        raise SQLRejected(f"Only SELECT queries are allowed (got {type(root).__name__.upper()}).")

    cte_names = {cte.alias_or_name.lower() for cte in root.find_all(exp.CTE)}
    for node in root.find_all(exp.Expression):
        if isinstance(node, FORBIDDEN_NODES):
            raise SQLRejected(f"This kind of statement or clause is not allowed: {type(node).__name__.upper()}.")
        if isinstance(node, exp.Func):
            names = {_norm(node.key), _norm(getattr(node, "name", ""))}
            if names & BLOCKED_FUNCTIONS:
                raise SQLRejected(f"The function {node.sql(dialect='mysql').split('(')[0].upper()} is not allowed.")
        if isinstance(node, exp.Table):
            schema = (node.db or "").lower()
            if schema and schema != DATABASE_NAME:
                raise SQLRejected(f"Reading from the '{schema}' schema is not allowed.")
            if node.catalog:
                raise SQLRejected("Catalog-qualified table names are not allowed.")
            name = node.name.lower()
            if name not in ALLOWED_TABLES and name not in cte_names:
                raise SQLRejected(
                    f"Table '{node.name}' does not exist. Use only: {', '.join(sorted(ALLOWED_TABLES))}.", "fixable")

    return _apply_row_limit(root, max_rows).sql(dialect="mysql", comments=False)


def _apply_row_limit(root: exp.Expression, max_rows: int) -> exp.Expression:
    if isinstance(root, SET_OPERATIONS):
        # LIMIT on a UNION is easy to misread, so wrap it: SELECT * FROM (<union>) AS _q LIMIT n
        return exp.select("*").from_(root.subquery("_q")).limit(max_rows)
    limit: Optional[exp.Expression] = root.args.get("limit")
    if limit is None:
        return root.limit(max_rows)
    value = limit.expression
    if isinstance(value, exp.Literal) and value.is_int:
        if int(value.name) > max_rows:
            return root.limit(max_rows)
        return root
    raise SQLRejected("LIMIT must be a plain number.", "fixable")
