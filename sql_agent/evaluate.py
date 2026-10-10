"""Measure how often the Text-to-SQL pipeline gets it right.

  python -m sql_agent.evaluate                  run all questions (needs the Gemini key and the read-only database user)
  python -m sql_agent.evaluate --only dates     one category
  python -m sql_agent.evaluate --resume         redo only the questions the AI service failed on last time
  python -m sql_agent.evaluate --limit 5 --delay 10   first 5 questions, 10 seconds apart

Each question in sql_agent/eval_questions.json has an expected behaviour:
  answer  -> the result must match the result of a hand-written reference query ("execution accuracy")
  refuse  -> the system must say it cannot answer
  block   -> an attack: nothing may run, and the database must be unchanged afterwards
  limited -> must succeed but be cut to the row limit
"""
import argparse
import datetime as dt
import decimal
import json
import logging
import math
import time
from collections import defaultdict
from pathlib import Path

from sqlalchemy import text

from sql_agent.guard import MAX_ROWS
from sql_agent.pipeline import Text2SQL, jsonable

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = Path(__file__).with_name("eval_questions.json")
DOCS = ROOT / "docs"
TABLES = ("countries", "customers", "products", "orders", "order_items", "returns")


# --------------------------------------------------------------------------- comparing results
def _norm(value):
    if value is None:
        return None
    if isinstance(value, (bool, int, float, decimal.Decimal)):
        return float(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return str(value)[:19]
    return str(value).strip().lower()


def _equal(a, b) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-4, abs_tol=0.051)   # tolerates rounding to 1-2 decimals
    if isinstance(a, str) and isinstance(b, str):
        return a == b or (len(a) == 10 and b.startswith(a)) or (len(b) == 10 and a.startswith(b))  # date vs datetime
    return False


def results_match(gold_rows: list, got_rows: list) -> bool:
    """Same number of rows, and every reference row's values appear in a distinct returned row.
    Extra columns, different column names and different row order are fine; wrong or missing values are not."""
    if len(gold_rows) != len(got_rows):
        return False
    remaining = [[_norm(v) for v in row] for row in got_rows]
    for gold_row in ([_norm(v) for v in row] for row in gold_rows):
        for candidate in remaining:
            pool, ok = list(candidate), True
            for value in gold_row:
                idx = next((i for i, other in enumerate(pool) if _equal(value, other)), None)
                if idx is None:
                    ok = False
                    break
                pool.pop(idx)
            if ok:
                remaining.remove(candidate)
                break
        else:
            return False
    return True


# --------------------------------------------------------------------------- running
def run_sql(engine, sql: str) -> list:
    with engine.connect() as conn:
        return [[jsonable(v) for v in row] for row in conn.execute(text(sql)).fetchall()]


def fingerprint(engine) -> dict:
    return {t: run_sql(engine, f"SELECT COUNT(*) FROM {t}")[0][0] for t in TABLES}


def judge(question: dict, result, engine) -> tuple:
    """Returns (passed, reason)."""
    expect = question["expect"]
    note = f": {result.message[:150]}" if result.message else ""
    if expect == "answer":
        if result.status != "ok":
            return False, f"status {result.status}{note}"
        gold = run_sql(engine, question["gold_sql"])
        if results_match(gold, result.rows):
            return True, ""
        return False, f"wrong result (expected {gold[:3]}, got {result.rows[:3]})"
    if expect == "refuse":
        return result.status == "refused", f"status {result.status}{note}"
    if expect == "block":
        return result.status in ("refused", "blocked"), f"status {result.status}{note}"
    if expect == "limited":
        ok = result.status == "ok" and 0 < result.row_count <= MAX_ROWS
        return ok, f"status {result.status}, rows {result.row_count}{note}"
    raise ValueError(f"unknown expectation {expect}")


def _error_outcome(q: dict, reason: str, attempts: int = 0) -> dict:
    """The question could not be scored (AI service down, rate-limited, ...). It is NOT counted as a wrong answer."""
    return {"id": q["id"], "category": q["category"], "question": q["question"], "expect": q["expect"],
            "passed": None, "reason": reason, "status": "error", "sql": None, "attempts": attempts, "seconds": 0.0}


def evaluate(llm, engine, questions: list, delay: float = 0.0, progress=print, retries: int = 3, backoff: float = 20.0,
             sleep=time.sleep, previous=None, save=None, give_up_after: int = 3) -> dict:
    """Runs every question. When the AI service fails, waits (backoff, 2x backoff, ...) and retries; a question that still
    cannot be answered is reported as 'not scored' rather than wrong. `previous` = outcomes from an earlier run whose
    scored questions are reused (resume); `save(outcomes)` is called after every question. If `give_up_after` questions in a row
    cannot be answered the run stops early (the service is probably out of quota) and the rest are left for --resume."""
    agent = Text2SQL(llm, engine)
    before = fingerprint(engine)
    kept = {o["id"]: o for o in (previous or []) if o["status"] != "error"}
    outcomes, in_a_row = [], 0
    for i, q in enumerate(questions, 1):
        tag = f"[{i}/{len(questions)}]"
        if q["id"] in kept:   # results from an earlier run are always kept, even after the run has given up
            outcomes.append(kept[q["id"]])
            progress(f"{tag} kept  {q['id']} {q['category']:14s} (result from an earlier run)")
            continue
        if in_a_row >= give_up_after:
            outcomes.append(_error_outcome(q, "not attempted: the AI service kept failing, so the run stopped early"))
            continue
        result, last_error = None, ""
        for attempt in range(retries + 1):
            try:
                result = agent.ask(q["question"])
                if not result.service_error:
                    break
                last_error = result.message
            except Exception as exc:
                result, last_error = None, f"{type(exc).__name__}: {str(exc)[:200]}"
            if attempt < retries:
                wait = backoff * (2 ** attempt)
                progress(f"      AI service problem; waiting {wait:.0f}s, then retry {attempt + 1} of {retries}")
                sleep(wait)
        if result is None or result.service_error:
            outcome = _error_outcome(q, last_error[:250])
        else:
            try:
                passed, reason = judge(q, result, engine)
                outcome = {"id": q["id"], "category": q["category"], "question": q["question"], "expect": q["expect"],
                           "passed": passed, "reason": reason, "status": result.status, "sql": result.sql,
                           "attempts": len(result.attempts), "seconds": result.seconds}
            except Exception as exc:  # e.g. a reference query timed out: a problem with the check, not with the AI
                outcome = _error_outcome(q, f"could not check the answer: {type(exc).__name__}: {str(exc)[:150]}")
        in_a_row = in_a_row + 1 if outcome["passed"] is None else 0
        outcomes.append(outcome)
        label = "ERROR" if outcome["passed"] is None else "PASS " if outcome["passed"] else "FAIL "
        progress(f"{tag} {label} {q['id']} {q['category']:14s} {q['question'][:60]}")
        if save:
            save(outcomes)
        if in_a_row >= give_up_after:
            progress(f"\nStopping early: {give_up_after} questions in a row got no answer from the AI service. This usually means the free "
                     "quota is used up or the service is overloaded. Progress is saved; wait a while, then run with --resume.")
        if delay and i < len(questions) and in_a_row < give_up_after:
            sleep(delay)
    if save:
        save(outcomes)
    after = fingerprint(engine)

    scored = [o for o in outcomes if o["passed"] is not None]
    errors = [o for o in outcomes if o["passed"] is None]
    by_cat = defaultdict(lambda: {"passed": 0, "total": 0, "not_scored": 0})
    for o in outcomes:
        c = by_cat[o["category"]]
        if o["passed"] is None:
            c["not_scored"] += 1
        else:
            c["total"] += 1
            c["passed"] += int(o["passed"])
    answered = [o for o in scored if o["expect"] == "answer"]
    return {
        "questions": len(outcomes), "scored": len(scored), "not_scored": len(errors),
        "total": len(scored), "passed": sum(bool(o["passed"]) for o in scored),
        "accuracy": round(sum(bool(o["passed"]) for o in scored) / max(len(scored), 1), 4),
        "answerable_accuracy": round(sum(bool(o["passed"]) for o in answered) / max(len(answered), 1), 4),
        "answerable_scored": len(answered),
        "by_category": dict(by_cat),
        "database_unchanged": before == after,
        "mean_attempts": round(sum(o["attempts"] for o in scored) / max(len(scored), 1), 2),
        "mean_seconds": round(sum(o["seconds"] for o in scored) / max(len(scored), 1), 2),
        "outcomes": outcomes,
    }


def write_report(report: dict, model_note: str = "") -> None:
    DOCS.mkdir(exist_ok=True)
    (DOCS / "sql_eval_report.json").write_text(json.dumps(report, indent=2, default=str))
    lines = ["# Text-to-SQL evaluation", "", model_note, "",
             f"**{report['passed']} of {report['total']} scored questions passed ({report['accuracy']:.0%}).** "
             f"On questions with a data answer: {report['answerable_accuracy']:.0%} of {report['answerable_scored']}. "
             f"Database unchanged afterwards: {'yes' if report['database_unchanged'] else 'NO'}. "
             f"Mean attempts per question: {report['mean_attempts']}. Mean time: {report['mean_seconds']} s."]
    if report["not_scored"]:
        lines += ["", f"**{report['not_scored']} of {report['questions']} questions could not be scored** because the AI service "
                      "failed (not counted as wrong). Rerun with `--resume` to try only those again."]
    lines += ["", "| Category | Passed | Scored | Not scored |", "|---|---|---|---|"]
    lines += [f"| {c} | {v['passed']} | {v['total']} | {v['not_scored']} |" for c, v in report["by_category"].items()]
    failures = [o for o in report["outcomes"] if o["passed"] is False]
    if failures:
        lines += ["", "## Failures", ""]
        for o in failures:
            lines += [f"- **{o['id']}** ({o['category']}): {o['question']}", f"  - {o['reason']}",
                      f"  - SQL: `{o['sql']}`" if o["sql"] else "  - no SQL was run"]
    errs = [o for o in report["outcomes"] if o["passed"] is None]
    if errs:
        lines += ["", "## Not scored (AI service problems)", ""]
        lines += [f"- **{o['id']}**: {o['question']}  \n  {o['reason']}" for o in errs]
    (DOCS / "sql_eval_report.md").write_text("\n".join(lines) + "\n")


PROGRESS_FILE = DOCS / "sql_eval_progress.json"


def main() -> None:
    from sql_agent.db import assert_read_only, get_readonly_engine
    from sql_agent.llm import DEFAULT_MODEL, get_llm
    import os

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="run only this category")
    parser.add_argument("--limit", type=int, help="run only the first N questions")
    parser.add_argument("--delay", type=float, default=6.0, help="seconds to wait between questions (default 6)")
    parser.add_argument("--retries", type=int, default=3, help="retries per question when the AI service fails (default 3)")
    parser.add_argument("--backoff", type=float, default=20.0, help="first wait before a retry, doubling each time (default 20 s)")
    parser.add_argument("--resume", action="store_true", help="reuse scored results from the previous run, redo only the rest")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    questions = json.loads(QUESTIONS.read_text())
    if args.only:
        questions = [q for q in questions if q["category"] == args.only]
    if args.limit:
        questions = questions[: args.limit]

    previous = None
    if args.resume and PROGRESS_FILE.exists():
        previous = json.loads(PROGRESS_FILE.read_text())

    def save(outcomes):
        DOCS.mkdir(exist_ok=True)
        PROGRESS_FILE.write_text(json.dumps(outcomes, indent=2, default=str))

    engine = get_readonly_engine()
    assert_read_only(engine)
    report = evaluate(get_llm(), engine, questions, delay=args.delay, retries=args.retries, backoff=args.backoff,
                      previous=previous, save=save)
    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    write_report(report, f"Model: `{model}`. Run on {dt.date.today()}.")
    print(f"\n{report['passed']}/{report['total']} scored questions passed ({report['accuracy']:.0%}); "
          f"questions with a data answer {report['answerable_accuracy']:.0%} of {report['answerable_scored']}; "
          f"database unchanged: {report['database_unchanged']}")
    if report["not_scored"]:
        print(f"{report['not_scored']} of {report['questions']} questions could NOT be scored (AI service errors). "
              "Run again with --resume to retry just those.")
    for c, v in report["by_category"].items():
        extra = f"  (+{v['not_scored']} not scored)" if v["not_scored"] else ""
        print(f"  {c:15s} {v['passed']}/{v['total']}{extra}")
    print("Details: docs/sql_eval_report.md")


if __name__ == "__main__":
    main()
