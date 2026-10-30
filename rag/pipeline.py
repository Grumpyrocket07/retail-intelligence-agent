"""Question in, grounded answer out.

  1. find the passages closest in meaning to the question
  2. if even the best one is not close enough, say "I don't know" WITHOUT asking the AI (saves quota, avoids guessing)
  3. otherwise ask the AI to answer only from those passages and cite them
"""
import re
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from rag.prompts import NO_ANSWER_TEXT, SYSTEM_PROMPT, user_message
from rag.store import Index

MAX_QUESTION_CHARS = 500
DEFAULT_TOP_K = 4
DEFAULT_MIN_SIMILARITY = 0.20   # computed on the second question set's 13 dev questions only: 0.10 below the least similar legitimate
                                 # question (see rag/evaluate.py --set v2 --auto-threshold). It only skips the AI for clearly unrelated questions.
_CITATION = re.compile(r"\[([a-z0-9\-]+)\]")


@dataclass
class RagResult:
    question: str
    status: str                      # "ok", "no_answer" or "failed"
    answer: str = ""
    sources: List[str] = field(default_factory=list)       # sources the answer cites (only ones that were retrieved)
    grounded: bool = False           # True when the answer cites at least one retrieved source
    retrieved: List[dict] = field(default_factory=list)
    llm_called: bool = False
    service_error: bool = False
    message: str = ""
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class RagPipeline:
    def __init__(self, index: Index, llm, top_k: int = DEFAULT_TOP_K, min_similarity: float = DEFAULT_MIN_SIMILARITY):
        self.index, self.llm, self.top_k, self.min_similarity = index, llm, top_k, min_similarity

    def ask(self, question: str) -> RagResult:
        start = time.perf_counter()
        result = self._ask((question or "").strip())
        result.seconds = round(time.perf_counter() - start, 2)
        return result

    def _ask(self, question: str) -> RagResult:
        result = RagResult(question=question, status="failed")
        if not question or len(question) > MAX_QUESTION_CHARS:
            result.status, result.answer = "no_answer", NO_ANSWER_TEXT
            result.message = "The question was empty or too long."
            return result

        result.retrieved = self.index.search(question, self.top_k)
        best: Optional[float] = result.retrieved[0]["similarity"] if result.retrieved else None
        if best is None or best < self.min_similarity:
            result.status, result.answer = "no_answer", NO_ANSWER_TEXT
            result.message = f"Nothing in the documents is close enough to the question (best match {best})."
            return result

        messages = [("system", SYSTEM_PROMPT), ("human", user_message(question, result.retrieved))]
        try:
            reply = self.llm(messages).strip()
        except Exception as exc:   # AI service overloaded, rate-limited, offline ...
            result.service_error, result.llm_called = True, True
            result.message = f"The AI service did not answer: {type(exc).__name__}: {str(exc)[:300]}"
            return result
        result.llm_called = True

        if reply.lower().startswith("i don't know") or reply.lower().startswith("i do not know"):
            result.status, result.answer = "no_answer", reply
            return result

        retrieved_sources = {r["source"] for r in result.retrieved}
        cited = []
        for name in _CITATION.findall(reply):
            if name in retrieved_sources and name not in cited:
                cited.append(name)
        result.status, result.answer, result.sources, result.grounded = "ok", reply, cited, bool(cited)
        return result
