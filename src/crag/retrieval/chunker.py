"""HTML -> clean text -> overlapping chunks with article attribution."""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

from src.crag.models import Chunk
from src.crag.retrieval.zim_reader import ZimArticle

# Non-content elements in Kiwix/Wikipedia HTML that would pollute retrieval.
_STRIP_TAGS = ("script", "style", "nav", "footer", "header", "table", "figure", "sup")
_WHITESPACE_RE = re.compile(r"\s+")


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(_STRIP_TAGS):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return _WHITESPACE_RE.sub(" ", text).strip()


def chunk_article(article: ZimArticle, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split an article into overlapping character-window chunks.

    Windows are extended to the next sentence/word boundary where possible so
    chunks stay coherent for the grader and generator.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = html_to_text(article.html)
    if not text:
        return []

    chunks: list[Chunk] = []
    step = chunk_size - chunk_overlap
    start = 0
    index = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Prefer to end on a sentence boundary within the last 20% of the window.
        window = text[start:end]
        if end < len(text):
            boundary = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
            if boundary > int(chunk_size * 0.8):
                end = start + boundary + 1
                window = text[start:end]
        piece = window.strip()
        if piece:
            chunks.append(
                Chunk(
                    article_path=article.path,
                    article_title=article.title,
                    chunk_index=index,
                    text=piece,
                )
            )
            index += 1
        if end >= len(text):
            break
        start = end - chunk_overlap if end - chunk_overlap > start else start + step
    return chunks
