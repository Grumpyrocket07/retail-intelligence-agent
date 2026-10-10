"""The language model that writes SQL. Gemini through LangChain; swap get_llm() to use a different provider.

An 'llm' in this project is just a function: it takes a list of (role, text) messages and returns the reply text.
That keeps the rest of the code (and the tests) independent of any particular provider.
"""
import os
import warnings
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
LLM = Callable[[list], str]

DEFAULT_MODEL = "gemini-3.8-flash"


def content_to_text(content) -> str:
    """Newer Gemini models can return a list of content blocks instead of a plain string."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def get_llm() -> LLM:
    load_dotenv(ROOT / ".env", override=True)
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY in .env (get a free key at https://aistudio.google.com/apikey).")
    from langchain_google_genai import ChatGoogleGenerativeAI

    settings = dict(
        model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
        temperature=float(os.getenv("LLM_TEMPERATURE", "1.0")),  # Google recommends leaving Gemini 3 models at 1.0
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "1")),
        timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),     # give up on a call after this long
        google_api_key=key,
    )
    # Gemini 3 models "think" before answering, which can be slow. GEMINI_REASONING_EFFORT = minimal | low | medium | high
    effort = os.getenv("GEMINI_REASONING_EFFORT")
    try:
        model = ChatGoogleGenerativeAI(**settings, reasoning_effort=effort) if effort else ChatGoogleGenerativeAI(**settings)
    except Exception as exc:  # an older langchain-google-genai may not know this setting
        warnings.warn(f"GEMINI_REASONING_EFFORT was ignored: {exc}")
        model = ChatGoogleGenerativeAI(**settings)

    def call(messages: list) -> str:
        return content_to_text(model.invoke(messages).content)

    return call
