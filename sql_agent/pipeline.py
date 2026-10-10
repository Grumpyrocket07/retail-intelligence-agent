"""Text-to-SQL: question in plain English -> checked SQL -> rows.

Flow for each question:
  AI writes SQL -> guard checks it -> read-only database runs it -> rows come back.
If the guard rejects the SQL in a fixable way, or the database reports an error, the problem is sent back to the AI
and it tries again (at most max_attempts times). Requests that look unsafe are stopped immediately, with no retry.
"""
import datetime as dt
import decimal
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from sql_agent.guard import MAX_ROWS, SQLRejected, validate_and_limit
from sql_agent.schema_context import FEEDBACK_DB_ERROR, FEEDBACK_REJECTED, SYSTEM_PROMPT

log = logging.getLogger("sql_agent")

MAX_QUESTION_CHARS = 500
_FENCE = re.compile(r"```(?:sql|mysql)?\s*(.*?)```", re.I | re.S)


@dataclass
class Attempt:
    number: int
    reply: str
    stage: str                      # "refused", "guard", "database" or "ok"
    sql_run: Optional[str] = None   # the checked SQL that was (or would have been) executed
    error: Optional[str] = None
    llm_seconds: float = 0.0        # time spent waiting for the AI
    db_seconds: float = 0.0         # time spent in the database


@dataclass
class Result:
    question: str
    status: str                     # "ok", "refused", "blocked" or "failed"
    message: str = ""
    sql: Optional[str] = None
    columns: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    attempts: list = field(default_factory=list)
    seconds: float = 0.0
    service_error: bool = False     # True when the AI service itself failed (overloaded, rate-limited, offline)

    def to_dict(self) -> dict:
        return asdict(self)


def extract_sql(reply: str) -> str:
    match = _FENCE.search(reply)
    return (match.group(1) if match else reply).strip()


def jsonable(value):
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat(sep=" ") if isinstance(value, dt.datetime) else value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


class Text2SQL:
    def __init__(self, llm, engine, max_attempts: int = 3, max_rows: int = MAX_ROWS):
        self.llm, self.engine, self.max_attempts, self.max_rows = llm, engine, max_attempts, max_rows

    def ask(self, question: str) -> Result:
        start = time.perf_counter()
        result = self._ask(question)
        result.seconds = round(time.perf_counter() - start, 2)
        log.info("question=%r status=%s attempts=%d rows=%d seconds=%.2f", question[:80], result.status,
                 len(result.attempts), result.row_count, result.seconds)
        return result

    def _ask(self, question: str) -> Result:
        question = (question or "").strip()
        result = Result(question=question, status="failed")
        if not question:
            result.status, result.message = "refused", "The question was empty."
            return result
        if len(question) > MAX_QUESTION_CHARS:
            result.status, result.message = "refused", f"The question is longer than {MAX_QUESTION_CHARS} characters."
            return result

        messages = [("system", SYSTEM_PROMPT), ("human", question)]
        for number in range(1, self.max_attempts + 1):
            t_llm = time.perf_counter()
            try:
                reply = self.llm(messages).strip()
            except Exception as exc:  # AI service unreachable, rate-limited, timed out, ...
                error = f"{type(exc).__name__}: {str(exc)[:300]}"
                result.attempts.append(Attempt(number=number, reply="", stage="llm", error=error,
                                               llm_seconds=round(time.perf_counter() - t_llm, 2)))
                result.status, result.message = "failed", f"The AI service did not answer: {error}"
                result.service_error = True
                return result
            attempt = Attempt(number=number, reply=reply, stage="guard", llm_seconds=round(time.perf_counter() - t_llm, 2))
            result.attempts.append(attempt)

            if reply.upper().startswith("CANNOT_ANSWER") or "CANNOT_ANSWER:" in reply.upper()[:60]:
                attempt.stage = "refused"
                result.status = "refused"
                result.message = reply.split(":", 1)[1].strip() if ":" in reply else "This cannot be answered from the data."
                return result

            try:
                safe_sql = validate_and_limit(extract_sql(reply), self.max_rows)
            except SQLRejected as exc:
                attempt.error = exc.reason
                if exc.kind == "unsafe":
                    result.status, result.message = "blocked", f"Blocked for safety: {exc.reason}"
                    return result
                messages += [("ai", reply), ("human", FEEDBACK_REJECTED.format(reason=exc.reason))]
                result.message = exc.reason
                continue

            attempt.sql_run = safe_sql
            t_db = time.perf_counter()
            try:
                with self.engine.connect() as conn:
                    cursor = conn.execute(text(safe_sql))
                    columns = list(cursor.keys())
                    rows = [[jsonable(v) for v in row] for row in cursor.fetchall()]
            except SQLAlchemyError as exc:
                attempt.db_seconds = round(time.perf_counter() - t_db, 2)
                error = str(getattr(exc, "orig", exc))[:300]
                attempt.stage, attempt.error = "database", error
                messages += [("ai", reply), ("human", FEEDBACK_DB_ERROR.format(error=error))]
                result.message = error
                continue

            attempt.db_seconds = round(time.perf_counter() - t_db, 2)
            attempt.stage = "ok"
            result.status, result.message, result.sql = "ok", "", safe_sql
            result.columns, result.rows, result.row_count = columns, rows, len(rows)
            result.truncated = len(rows) >= self.max_rows
            return result

        result.status = "failed"
        result.message = f"No working query after {self.max_attempts} attempts. Last problem: {result.message}"
        return result
