"""Database connection helper. Reads credentials from .env (see .env.example)."""
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL, Engine

ROOT = Path(__file__).resolve().parents[1]


def get_engine() -> Engine:
    load_dotenv(ROOT / ".env")
    required = ["DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_NAME"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing settings in .env: {', '.join(missing)}")
    url = URL.create(
        "mysql+pymysql",
        username=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        host=os.environ["DB_HOST"],
        port=int(os.environ["DB_PORT"]),
        database=os.environ["DB_NAME"],
        query={"charset": "utf8mb4"},
    )
    return create_engine(url, pool_pre_ping=True)
