"""The language model used across the project. Gemini through LangChain; change get_chat_model() to use a different provider.

Two shapes are offered:
  get_llm()         a plain function: list of (role, text) messages in, reply text out (SQL, document Q&A)
  get_chat_model()  the LangChain chat model itself, needed by the agent for tool calling
"""
import os
import warnings
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
LLM = Callable[[list], str]

# Some Gemini models ignore the temperature setting and the library warns about it on every call.
warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*")

DEFAULT_MODEL = "gemini-3.5-flash-lite"   # gemini-3.8-flash often timed out (504) on the free tier


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


def get_chat_model():
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
        return ChatGoogleGenerativeAI(**settings, reasoning_effort=effort) if effort else ChatGoogleGenerativeAI(**settings)
    except Exception as exc:  # an older langchain-google-genai may not know this setting
        warnings.warn(f"GEMINI_REASONING_EFFORT was ignored: {exc}")
        return ChatGoogleGenerativeAI(**settings)


def get_llm() -> LLM:
    model = get_chat_model()

    def call(messages: list) -> str:
        return content_to_text(model.invoke(messages).content)

    return call
