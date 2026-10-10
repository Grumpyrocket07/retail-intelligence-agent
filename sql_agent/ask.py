"""Ask the database a question from the command line.

  python -m sql_agent.ask "Which 5 countries have the highest revenue?"
  python -m sql_agent.ask            (no question: type several, empty line to quit)
"""
import logging
import sys

import pandas as pd

from sql_agent.db import assert_read_only, get_readonly_engine
from sql_agent.llm import get_llm
from sql_agent.pipeline import Text2SQL


def show(result) -> None:
    print(f"\nStatus: {result.status}   (attempts: {len(result.attempts)}, {result.seconds}s)")
    for a in result.attempts:
        line = f"  attempt {a.number}: {a.stage:8s} AI {a.llm_seconds:5.1f}s   database {a.db_seconds:5.1f}s"
        print(line + (f"   problem: {a.error}" if a.error else ""))
    if result.sql:
        print(f"SQL:\n{result.sql}\n")
    if result.status == "ok":
        print(pd.DataFrame(result.rows, columns=result.columns).to_string(index=False, max_rows=30))
        if result.truncated:
            print("(result was cut to the row limit)")
    elif result.message:
        print(result.message)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    engine = get_readonly_engine()
    assert_read_only(engine)
    agent = Text2SQL(get_llm(), engine)
    if len(sys.argv) > 1:
        show(agent.ask(" ".join(sys.argv[1:])))
        return
    while True:
        question = input("\nQuestion (empty to quit): ").strip()
        if not question:
            break
        show(agent.ask(question))


if __name__ == "__main__":
    main()
