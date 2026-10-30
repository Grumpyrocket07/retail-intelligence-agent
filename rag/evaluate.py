"""Measure the document Q&A system.

  python -m rag.evaluate --retrieval-only    no AI calls: does the search find the right document? (also suggests the threshold)
  python -m rag.evaluate                     full run with the AI (uses Gemini quota); results saved after every question
  python -m rag.evaluate --resume            redo only the questions the AI service failed on
  python -m rag.evaluate --split test        score only the 'test' questions (the 'dev' ones were used to set the threshold)
  python -m rag.evaluate --set v2 --auto-threshold
                                             second, fresh question set: the "too far away" threshold is computed from the dev
                                             questions only (no AI calls), then the test questions are scored with it

What each question expects (rag/eval_questions.json):
  answer     -> status ok, the answer contains every required fact ("must_include" groups), and nothing forbidden
  no_answer  -> the system says "I don't know"
  resist     -> the system is not talked into the forbidden claim ("must_not_include"), and still gives the right fact if one is required
Retrieval is scored separately: was a document that should contain the answer among the top-k passages?
"""
import argparse
import datetime as dt
import json
import time
from collections import defaultdict
from pathlib import Path

from rag.chunking import load_chunks
from rag.pipeline import DEFAULT_MIN_SIMILARITY, DEFAULT_TOP_K, RagPipeline
from rag.store import Index

ROOT = Path(__file__).resolve().parents[1]
QUESTION_FILES = {"v1": Path(__file__).with_name("eval_questions.json"), "v2": Path(__file__).with_name("eval_questions_v2.json")}
DOCUMENTS = Path(__file__).with_name("documents")
ADVERSARIAL = Path(__file__).with_name("documents_adversarial")
DOCS = ROOT / "docs"


# --------------------------------------------------------------------------- checking answers
def _normalise(text: str) -> str:
    return (text.lower().replace("\u2013", "-").replace("\u2014", "-").replace("\u00a0", " ").replace("**", ""))


def check_answer(question: dict, answer: str) -> tuple:
    """Returns (ok, reason) for the wording of an answer."""
    text = _normalise(answer)
    for group in question["must_include"]:
        if not any(_normalise(option) in text for option in group):
            return False, f"missing one of {group}"
    for bad in question["must_not_include"]:
        if _normalise(bad) in text:
            return False, f"contains forbidden text '{bad}'"
    return True, ""


def judge(question: dict, result) -> tuple:
    expect = question["expect"]
    if expect == "no_answer":
        return result.status == "no_answer", f"status {result.status}: {result.answer[:100]}"
    if expect == "resist":
        if question["must_include"] and result.status != "ok":
            return False, f"status {result.status}: {result.answer[:100]}"
        return check_answer(question, result.answer)
    if result.status != "ok":
        return False, f"status {result.status}: {(result.answer or result.message)[:120]}"
    return check_answer(question, result.answer)


# --------------------------------------------------------------------------- retrieval only
def recommend_threshold(dev_rows: list, margin: float = 0.10) -> dict:
    """dev_rows: [{'top1': best-match similarity, 'answerable': bool}].

    The threshold only exists to skip the AI for questions that are obviously unrelated to the documents. Wrongly declining a real
    question is worse than letting an off-topic one through (the AI can still say "I don't know"), so the threshold is set on the safe
    side: a safety margin below the LEAST similar legitimate dev question.
    """
    answerable = sorted(r["top1"] for r in dev_rows if r["answerable"])
    unanswerable = sorted(r["top1"] for r in dev_rows if not r["answerable"])
    if not answerable:
        return {"threshold": DEFAULT_MIN_SIMILARITY, "note": "no answerable dev questions"}
    threshold = round(max(answerable[0] - margin, 0.0), 2)
    return {"threshold": threshold, "lowest_answerable_dev": answerable[0], "margin": margin,
            "highest_unanswerable_dev": unanswerable[-1] if unanswerable else None,
            "unanswerable_dev_declined_early": sum(u < threshold for u in unanswerable), "unanswerable_dev": len(unanswerable)}


def evaluate_retrieval(indexes: dict, questions: list, top_k: int = DEFAULT_TOP_K, min_similarity: float = DEFAULT_MIN_SIMILARITY) -> dict:
    rows = []
    for q in questions:
        found = indexes[q["corpus"]].search(q["question"], top_k)
        top1 = found[0]["similarity"] if found else 0.0
        row = {"id": q["id"], "split": q["split"], "category": q["category"], "answerable": q["expect"] in ("answer",) or
               (q["expect"] == "resist" and bool(q["expected_sources"])), "top1": top1,
               "retrieved": [(r["source"], r["similarity"]) for r in found]}
        if row["answerable"]:
            row["hit"] = any(r["source"] in q["expected_sources"] for r in found)
        elif q["expect"] == "no_answer":
            row["correctly_declined"] = top1 < min_similarity
        rows.append(row)
    hits = [r for r in rows if "hit" in r]
    declines = [r for r in rows if "correctly_declined" in r]
    dev_rows = [r for r in rows if r["split"] == "dev" and (r["answerable"] or "correctly_declined" in r)]
    return {
        "top_k": top_k, "min_similarity": min_similarity,
        "hit_rate": round(sum(r["hit"] for r in hits) / max(len(hits), 1), 4), "hits": sum(r["hit"] for r in hits), "answerable": len(hits),
        "misses": [{"id": r["id"], "retrieved": r["retrieved"]} for r in hits if not r["hit"]],
        "declined_by_threshold": sum(r["correctly_declined"] for r in declines), "unanswerable": len(declines),
        "wrongly_declined_answerable": [r["id"] for r in hits if r["top1"] < min_similarity],
        "threshold_suggestion_from_dev": recommend_threshold(dev_rows), "rows": rows,
    }


# --------------------------------------------------------------------------- full run
def _error_outcome(q: dict, reason: str) -> dict:
    return {"id": q["id"], "split": q["split"], "category": q["category"], "question": q["question"], "expect": q["expect"],
            "passed": None, "reason": reason, "status": "error", "answer": "", "sources": [], "retrieval_hit": None,
            "cited_expected": None, "grounded": None, "llm_called": True, "seconds": 0.0}


def evaluate_answers(pipelines: dict, questions: list, delay: float = 0.0, progress=print, retries: int = 3, backoff: float = 20.0,
                     sleep=time.sleep, previous=None, save=None, give_up_after: int = 3) -> dict:
    """Same safeguards as the Text-to-SQL harness: waits and retries when the AI service fails, a question that still cannot
    be answered is 'not scored' (never wrong), earlier scored results are kept (resume), and the run stops early when the service
    keeps failing."""
    kept = {o["id"]: o for o in (previous or []) if o["status"] != "error"}
    outcomes, in_a_row = [], 0
    for i, q in enumerate(questions, 1):
        tag = f"[{i}/{len(questions)}]"
        if q["id"] in kept:
            outcomes.append(kept[q["id"]])
            progress(f"{tag} kept  {q['id']} (result from an earlier run)")
            continue
        if in_a_row >= give_up_after:
            outcomes.append(_error_outcome(q, "not attempted: the AI service kept failing, so the run stopped early"))
            continue
        pipeline, result, last_error = pipelines[q["corpus"]], None, ""
        for attempt in range(retries + 1):
            result = pipeline.ask(q["question"])
            if not result.service_error:
                break
            last_error = result.message
            if attempt < retries:
                wait = backoff * (2 ** attempt)
                progress(f"      AI service problem; waiting {wait:.0f}s, then retry {attempt + 1} of {retries}")
                sleep(wait)
        if result.service_error:
            outcome = _error_outcome(q, last_error[:250])
        else:
            passed, reason = judge(q, result)
            expected = q["expected_sources"]
            outcome = {"id": q["id"], "split": q["split"], "category": q["category"], "question": q["question"], "expect": q["expect"],
                       "passed": passed, "reason": reason, "status": result.status, "answer": result.answer, "sources": result.sources,
                       "retrieval_hit": (any(r["source"] in expected for r in result.retrieved) if expected else None),
                       "cited_expected": (any(s in expected for s in result.sources) if expected and result.status == "ok" else None),
                       "grounded": result.grounded if result.status == "ok" else None, "llm_called": result.llm_called,
                       "seconds": result.seconds}
        in_a_row = in_a_row + 1 if outcome["passed"] is None else 0
        outcomes.append(outcome)
        label = "ERROR" if outcome["passed"] is None else "PASS " if outcome["passed"] else "FAIL "
        progress(f"{tag} {label} {q['id']} {q['category']:13s} {q['question'][:60]}")
        if save:
            save(outcomes)
        if in_a_row >= give_up_after:
            progress(f"\nStopping early: {give_up_after} questions in a row got no answer from the AI service (quota used up or overloaded). "
                     "Progress is saved; wait a while, then run with --resume.")
        if delay and result is not None and result.llm_called and i < len(questions) and in_a_row < give_up_after:
            sleep(delay)
    if save:
        save(outcomes)
    return summarise(outcomes)


def summarise(outcomes: list) -> dict:
    scored = [o for o in outcomes if o["passed"] is not None]
    rate = lambda rows: round(sum(bool(o["passed"]) for o in rows) / max(len(rows), 1), 4)
    by = lambda key: {k: {"passed": sum(bool(o["passed"]) for o in scored if o[key] == k), "scored": sum(o[key] == k for o in scored),
                          "not_scored": sum(o[key] == k and o["passed"] is None for o in outcomes)} for k in sorted({o[key] for o in outcomes})}
    with_expected = [o for o in scored if o["retrieval_hit"] is not None]
    answered = [o for o in scored if o["status"] == "ok" and o["expect"] == "answer"]
    return {
        "questions": len(outcomes), "scored": len(scored), "not_scored": len(outcomes) - len(scored),
        "passed": sum(bool(o["passed"]) for o in scored), "accuracy": rate(scored),
        "by_split": by("split"), "by_category": by("category"),
        "retrieval_hit_rate": round(sum(o["retrieval_hit"] for o in with_expected) / max(len(with_expected), 1), 4),
        "cited_expected_source_rate": round(sum(bool(o["cited_expected"]) for o in answered) / max(len(answered), 1), 4),
        "grounded_rate": round(sum(bool(o["grounded"]) for o in answered) / max(len(answered), 1), 4),
        "outcomes": outcomes,
    }


def write_report(report: dict, note: str = "", suffix: str = "") -> None:
    DOCS.mkdir(exist_ok=True)
    (DOCS / f"rag_eval_report{suffix}.json").write_text(json.dumps(report, indent=2, default=str))
    lines = ["# Document Q&A (RAG) evaluation", "", note, "",
             f"**{report['passed']} of {report['scored']} scored questions passed ({report['accuracy']:.0%}).** "
             f"Search found a relevant document in the top passages for {report['retrieval_hit_rate']:.0%} of answerable questions. "
             f"Answers that cited a correct source: {report['cited_expected_source_rate']:.0%}."]
    if report["not_scored"]:
        lines += ["", f"**{report['not_scored']} of {report['questions']} questions could not be scored** because the AI service failed. Rerun with `--resume`."]
    lines += ["", "| Split | Passed | Scored |", "|---|---|---|"] + [f"| {k} | {v['passed']} | {v['scored']} |" for k, v in report["by_split"].items()]
    lines += ["", "| Category | Passed | Scored |", "|---|---|---|"] + [f"| {k} | {v['passed']} | {v['scored']} |" for k, v in report["by_category"].items()]
    failures = [o for o in report["outcomes"] if o["passed"] is False]
    if failures:
        lines += ["", "## Failures", ""]
        for o in failures:
            lines += [f"- **{o['id']}** ({o['category']}, {o['split']}): {o['question']}", f"  - {o['reason']}",
                      f"  - answer: {o['answer'][:200]}"]
    (DOCS / f"rag_eval_report{suffix}.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- command line
def build_indexes(embedder) -> dict:
    from rag.store import DEFAULT_PATH

    main_index = Index(embedder, DEFAULT_PATH)
    if main_index.count() == 0:
        raise SystemExit("The index is empty. Build it first:  python -m rag.ingest")
    adversarial = Index(embedder)   # in memory: the normal documents plus one with a hidden instruction
    adversarial.build(load_chunks(DOCUMENTS, ADVERSARIAL))
    return {"main": main_index, "adversarial": adversarial}


def main() -> None:
    from rag.embedder import SentenceTransformerEmbedder

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=["v1", "v2"], default="v1", help="which question set (v1 = first 35, v2 = fresh 41)")
    parser.add_argument("--retrieval-only", action="store_true", help="no AI calls: check the search and suggest a threshold")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    parser.add_argument("--auto-threshold", action="store_true", help="compute the threshold from the dev questions, then score the test questions")
    parser.add_argument("--delay", type=float, default=6.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--backoff", type=float, default=20.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--min-similarity", type=float, default=DEFAULT_MIN_SIMILARITY)
    args = parser.parse_args()

    suffix = "" if args.set == "v1" else f"_{args.set}"
    progress_file = DOCS / f"rag_eval_progress{suffix}.json"
    all_questions = json.loads(QUESTION_FILES[args.set].read_text())
    indexes = build_indexes(SentenceTransformerEmbedder())

    min_similarity = args.min_similarity
    if args.auto_threshold:
        dev = [q for q in all_questions if q["split"] == "dev"]
        suggestion = evaluate_retrieval(indexes, dev, min_similarity=min_similarity)["threshold_suggestion_from_dev"]
        min_similarity = suggestion["threshold"]
        print(f"Threshold computed from the {len(dev)} dev questions only: {min_similarity}  {suggestion}")
        if args.split == "all":
            args.split = "test"
    questions = all_questions if args.split == "all" else [q for q in all_questions if q["split"] == args.split]

    if args.retrieval_only:
        rep = evaluate_retrieval(indexes, questions, min_similarity=min_similarity)
        print(f"Search found a relevant document in the top {rep['top_k']} for {rep['hits']} of {rep['answerable']} answerable questions ({rep['hit_rate']:.0%}).")
        for m in rep["misses"]:
            print("  missed:", m["id"], m["retrieved"])
        print(f"With threshold {min_similarity}: {rep['declined_by_threshold']} of {rep['unanswerable']} unanswerable questions would be declined "
              f"before reaching the AI; answerable questions wrongly declined: {rep['wrongly_declined_answerable'] or 'none'}")
        print("Threshold suggested by the dev questions:", rep["threshold_suggestion_from_dev"])
        return

    from sql_agent.llm import DEFAULT_MODEL, get_llm
    import os

    llm = get_llm()
    pipelines = {k: RagPipeline(v, llm, min_similarity=min_similarity) for k, v in indexes.items()}
    previous = json.loads(progress_file.read_text()) if args.resume and progress_file.exists() else None

    def save(outcomes):
        DOCS.mkdir(exist_ok=True)
        progress_file.write_text(json.dumps(outcomes, indent=2, default=str))

    report = evaluate_answers(pipelines, questions, delay=args.delay, retries=args.retries, backoff=args.backoff, previous=previous, save=save)
    write_report(report, f"Question set: {args.set}. Model: `{os.getenv('GEMINI_MODEL', DEFAULT_MODEL)}`, embeddings: all-MiniLM-L6-v2, "
                         f"similarity threshold {min_similarity}, split: {args.split}. Run on {dt.date.today()}.", suffix)
    print(f"\n{report['passed']}/{report['scored']} scored questions passed ({report['accuracy']:.0%}); "
          f"retrieval hit rate {report['retrieval_hit_rate']:.0%}; cited a correct source {report['cited_expected_source_rate']:.0%}")
    if report["not_scored"]:
        print(f"{report['not_scored']} of {report['questions']} could NOT be scored (AI service errors). Run again with --resume"
              f"{' --set ' + args.set if args.set != 'v1' else ''}{' --auto-threshold' if args.auto_threshold else ''}.")
    for k, v in report["by_category"].items():
        print(f"  {k:14s} {v['passed']}/{v['scored']}")
    declined_early = [o for o in report["outcomes"] if o["status"] == "no_answer" and not o["llm_called"]]
    print(f"Questions stopped by the threshold before reaching the AI: {len(declined_early)} "
          f"({', '.join(o['id'] for o in declined_early) or 'none'}); wrongly: "
          f"{', '.join(o['id'] for o in declined_early if o['expect'] == 'answer') or 'none'}")
    print(f"Details: docs/rag_eval_report{suffix}.md")


if __name__ == "__main__":
    main()
