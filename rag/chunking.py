"""Splits the markdown documents into small, self-describing pieces ("chunks") for search.

Each '## Section' becomes one chunk (long sections are split on paragraph breaks). Every chunk starts with the
document title and section name, so a piece such as "Orders placed before 2 pm ..." still says it is about dispatch times.
"""
import re
from dataclasses import dataclass
from pathlib import Path

MAX_CHARS = 900


@dataclass
class Chunk:
    id: str
    source: str      # file name without .md, used as the citation, e.g. "returns-and-refunds"
    title: str
    section: str
    text: str        # what is embedded and shown to the AI


def _split_long(body: str, limit: int) -> list:
    if len(body) <= limit:
        return [body]
    pieces, current = [], ""
    for paragraph in re.split(r"\n\s*\n", body):
        if current and len(current) + len(paragraph) + 2 > limit:
            pieces.append(current.strip())
            current = ""
        current += paragraph + "\n\n"
    if current.strip():
        pieces.append(current.strip())
    return pieces


def chunk_markdown(source: str, markdown: str, max_chars: int = MAX_CHARS) -> list:
    lines = markdown.strip().splitlines()
    title = next((l[2:].strip() for l in lines if l.startswith("# ")), source)
    sections, heading, buffer = [], None, []
    for line in lines:
        if line.startswith("## "):
            if heading is not None:
                sections.append((heading, "\n".join(buffer).strip()))
            heading, buffer = line[3:].strip(), []
        elif heading is not None:
            buffer.append(line)
    if heading is not None:
        sections.append((heading, "\n".join(buffer).strip()))

    chunks = []
    for heading, body in sections:
        if not body:
            continue
        for part in _split_long(body, max_chars):
            n = len(chunks)
            chunks.append(Chunk(id=f"{source}::{n:02d}", source=source, title=title, section=heading,
                                text=f"{title} - {heading}\n{part}"))
    return chunks


def load_chunks(*folders: Path) -> list:
    chunks = []
    for folder in folders:
        for path in sorted(Path(folder).glob("*.md")):
            chunks += chunk_markdown(path.stem, path.read_text(encoding="utf-8"))
    return chunks
