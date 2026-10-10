"""The evaluation harness: matching rules, question file, and a full run with a scripted AI."""
import json
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from sql_agent.evaluate import evaluate, results_match
from sql_agent.guard import validate_and_limit

QUESTIONS = json.loads((Path(__file__).parents[1] / "sql_agent" / "eval_questions.json").read_text())


def test_matching_ignores_order_names_and_extra_columns_but_not_wrong_values():
    gold = [["UK", 100.0], ["FR", 50.0]]
    assert results_match(gold, [["FR", 50.0], ["UK", 100.0]])
    assert results_match(gold, [["UK", 100.004, "extra"], ["FR", 50.0, "extra"]])
    assert not results_match(gold, [["UK", 100.0], ["DE", 50.0]])
    assert not results_match(gold, [["UK", 100.0]])
    assert not results_match([[7.4]], [[7.0]])
    assert results_match([[7.4]], [[7.39]])
    assert results_match([["2011-12-09 12:50:00"]], [["2011-12-09"]])
    assert results_match([[None]], [[None]]) and not results_match([[None]], [[0.0]])


def test_question_file_is_well_formed():
    ids = [q["id"] for q in QUESTIONS]
    assert len(ids) == len(set(ids)) and len(QUESTIONS) >= 30
    for q in QUESTIONS:
        assert q["expect"] in {"answer", "refuse", "block", "limited"}
        if q["expect"] == "answer":
            validate_and_limit(q["gold_sql"])   # reference queries must satisfy the same guard as AI queries
        else:
            assert not q.get("gold_sql")


def test_harness_scores_right_wrong_refused_and_blocked_correctly():
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    for t in ("countries", "customers", "products", "orders", "order_items", "returns"):
        pd.DataFrame({"x": [1, 2, 3]}).to_sql(t, engine, index=False)
    questions = [
        {"id": "a", "category": "simple", "question": "right", "expect": "answer", "gold_sql": "SELECT COUNT(*) FROM orders"},
        {"id": "b", "category": "simple", "question": "wrong", "expect": "answer", "gold_sql": "SELECT COUNT(*) FROM orders"},
        {"id": "c", "category": "not_answerable", "question": "weather", "expect": "refuse"},
        {"id": "d", "category": "adversarial", "question": "attack", "expect": "block"},
        {"id": "e", "category": "adversarial", "question": "attack that gets answered", "expect": "block"},
    ]
    script = {"right": "SELECT COUNT(*) FROM orders", "wrong": "SELECT COUNT(*) + 1 FROM orders",
              "weather": "CANNOT_ANSWER: not in the data", "attack": "DROP TABLE orders",
              "attack that gets answered": "SELECT * FROM orders"}
    report = evaluate(lambda msgs: script[msgs[-1][1]], engine, questions, progress=lambda *_: None)
    verdict = {o["id"]: o["passed"] for o in report["outcomes"]}
    assert verdict == {"a": True, "b": False, "c": True, "d": True, "e": False}
    assert report["database_unchanged"] is True and report["passed"] == 3


# ---------------------------------------------------------------- outages are retried, resumable, and never counted as wrong
def _small_engine():
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    for t in ("countries", "customers", "products", "orders", "order_items", "returns"):
        pd.DataFrame({"x": [1, 2, 3]}).to_sql(t, engine, index=False)
    return engine


QS = [
    {"id": "a", "category": "simple", "question": "count", "expect": "answer", "gold_sql": "SELECT COUNT(*) FROM orders"},
    {"id": "b", "category": "simple", "question": "refuse me", "expect": "refuse"},
]


def test_ai_service_hiccup_is_retried_after_waiting():
    calls, waits = {"n": 0}, []

    def flaky(msgs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("503 UNAVAILABLE: high demand")
        return "SELECT COUNT(*) FROM orders" if msgs[-1][1] == "count" else "CANNOT_ANSWER: no"

    report = evaluate(flaky, _small_engine(), QS, progress=lambda *_: None, sleep=waits.append, backoff=10)
    assert report["passed"] == 2 and report["not_scored"] == 0
    assert waits == [10, 20], "waits should double each time"


def test_persistent_outage_is_not_scored_and_not_counted_as_wrong():
    def down(msgs):
        if msgs[-1][1] == "count":
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return "CANNOT_ANSWER: no"

    report = evaluate(down, _small_engine(), QS, progress=lambda *_: None, sleep=lambda s: None, retries=2)
    verdict = {o["id"]: o["passed"] for o in report["outcomes"]}
    assert verdict == {"a": None, "b": True}
    assert report["not_scored"] == 1 and report["scored"] == 1 and report["accuracy"] == 1.0
    assert "429" in report["outcomes"][0]["reason"]


def test_resume_reuses_scored_results_and_only_redoes_errors():
    engine = _small_engine()
    previous = [
        {"id": "a", "category": "simple", "question": "count", "expect": "answer", "passed": True, "reason": "",
         "status": "ok", "sql": "SELECT 1", "attempts": 1, "seconds": 1.0},
        {"id": "b", "category": "simple", "question": "refuse me", "expect": "refuse", "passed": None, "reason": "503",
         "status": "error", "sql": None, "attempts": 0, "seconds": 0.0},
    ]
    asked, saved = [], []

    def llm(msgs):
        asked.append(msgs[-1][1])
        return "CANNOT_ANSWER: no"

    report = evaluate(llm, engine, QS, progress=lambda *_: None, sleep=lambda s: None, previous=previous,
                      save=lambda outs: saved.append(len(outs)))
    assert asked == ["refuse me"], "the already-scored question must not be asked again"
    assert report["passed"] == 2 and report["not_scored"] == 0 and saved[-1] == 2


def test_run_stops_early_when_the_service_keeps_failing_but_keeps_earlier_results():
    qs = [{"id": f"q{i}", "category": "simple", "question": f"refuse {i}", "expect": "refuse"} for i in range(1, 8)]
    previous = [{"id": "q7", "category": "simple", "question": "refuse 7", "expect": "refuse", "passed": True, "reason": "",
                 "status": "refused", "sql": None, "attempts": 1, "seconds": 1.0}]
    asked, saved = [], []

    def down(msgs):
        asked.append(msgs[-1][1])
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    report = evaluate(down, _small_engine(), qs, progress=lambda *_: None, sleep=lambda s: None, retries=0,
                      give_up_after=3, previous=previous, save=lambda outs: saved.append(len(outs)))
    assert asked == ["refuse 1", "refuse 2", "refuse 3"], "after 3 failures in a row the AI must not be asked any more"
    assert report["questions"] == 7 and report["not_scored"] == 6 and report["passed"] == 1   # 3 failed + 3 never attempted
    assert report["outcomes"][-1]["passed"] is True, "an earlier scored result must survive the early stop"
    assert saved[-1] == 7, "the final state, including not-attempted questions, is saved for --resume"
