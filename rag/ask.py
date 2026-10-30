"""Ask the documents a question.

  python -m rag.ask "How long do I have to return an item?"
  python -m rag.ask                   (no question: type several, empty line to quit)
"""
import sys

from rag.embedder import SentenceTransformerEmbedder
from rag.pipeline import RagPipeline
from rag.store import DEFAULT_PATH, Index
from sql_agent.llm import get_llm   # the same Gemini connection the Text-to-SQL pipeline uses


def show(result) -> None:
    print(f"\n{result.answer or result.message}")
    if result.sources:
        print("Sources:", ", ".join(result.sources))
    print(f"\n[{result.status}, {result.seconds}s, AI called: {result.llm_called}]")
    for r in result.retrieved:
        print(f"   {r['similarity']:.2f}  {r['source']} / {r['section']}")


def main() -> None:
    index = Index(SentenceTransformerEmbedder(), DEFAULT_PATH)
    if index.count() == 0:
        sys.exit("The index is empty. Build it first:  python -m rag.ingest")
    pipeline = RagPipeline(index, get_llm())
    if len(sys.argv) > 1:
        show(pipeline.ask(" ".join(sys.argv[1:])))
        return
    while True:
        question = input("\nQuestion (empty to quit): ").strip()
        if not question:
            break
        show(pipeline.ask(question))


if __name__ == "__main__":
    main()
