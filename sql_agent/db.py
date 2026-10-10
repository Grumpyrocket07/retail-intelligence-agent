"""Read-only database connection for AI-written queries (the last safety layer)."""
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import URL, Engine

ROOT = Path(__file__).resolve().parents[1]
# Longest a query may run. Whole-table totals over a million order lines took up to ~4 s on a fast test machine,
# so leave generous headroom for slower laptops. Override with SQL_TIMEOUT_SECONDS in .env.
QUERY_TIMEOUT_SECONDS = int(os.getenv("SQL_TIMEOUT_SECONDS", "30"))

# Privileges a read-only user must not hold.
_DANGEROUS_GRANT = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|FILE|SUPER|PROCESS|SHUTDOWN|GRANT OPTION|ALL PRIVILEGES|EXECUTE|REFERENCES|INDEX|LOCK TABLES)\b",
    re.I)


def get_readonly_engine() -> Engine:
    load_dotenv(ROOT / ".env", override=True)
    timeout = int(os.getenv("SQL_TIMEOUT_SECONDS", str(QUERY_TIMEOUT_SECONDS)))
    user, password = os.getenv("DB_RO_USER"), os.getenv("DB_RO_PASSWORD")
    if not user or not password:
        raise RuntimeError("Set DB_RO_USER and DB_RO_PASSWORD in .env (create the user with sql/01_create_readonly_user.sql).")
    if user == os.getenv("DB_USER"):
        raise RuntimeError("DB_RO_USER must be a separate read-only user, not the same user the ETL uses to write.")
    for key in ("DB_HOST", "DB_PORT", "DB_NAME"):
        if not os.getenv(key):
            raise RuntimeError(f"Missing {key} in .env")
    engine = create_engine(
        URL.create("mysql+pymysql", username=user, password=password, host=os.environ["DB_HOST"],
                   port=int(os.environ["DB_PORT"]), database=os.environ["DB_NAME"], query={"charset": "utf8mb4"}),
        pool_pre_ping=True,
    )

    @event.listens_for(engine, "connect")
    def _limit_session(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        try:
            try:  # MySQL 5.7.8+ / 8: milliseconds
                cursor.execute(f"SET SESSION max_execution_time = {timeout * 1000}")
            except Exception:  # MariaDB: seconds
                cursor.execute(f"SET SESSION max_statement_time = {timeout}")
            cursor.execute("SET SESSION TRANSACTION READ ONLY")
        finally:
            cursor.close()

    return engine


def assert_read_only(engine: Engine) -> None:
    """Refuse to run if the database user can do anything besides read."""
    with engine.connect() as conn:
        grants = [str(row[0]) for row in conn.execute(text("SHOW GRANTS FOR CURRENT_USER()"))]
    for grant in grants:
        privileges = grant.split(" ON ")[0]
        if _DANGEROUS_GRANT.search(privileges):
            raise RuntimeError(f"The read-only database user has write privileges ({grant}). Fix its grants before continuing.")
