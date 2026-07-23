"""Read-only access to the offline Wikipedia ZIM archive.

Uses ``libzim`` for both content retrieval (:class:`libzim.reader.Archive`)
and candidate search via the ZIM file's embedded Xapian full-text index
(:class:`libzim.search.Searcher`). No network access of any kind.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from libzim.reader import Archive
from libzim.search import Query, Searcher

logger = logging.getLogger(__name__)


@dataclass
class ZimArticle:
    path: str
    title: str
    html: str


class ZimReaderError(RuntimeError):
    """Raised when the ZIM archive is missing, unreadable, or unsearchable."""


class ZimReader:
    """Thin, defensive wrapper around a single ZIM archive.

    The underlying :class:`Archive` is opened once and reused; libzim archives
    are cheap to query and memory-map their content.
    """

    def __init__(self, zim_path: Path) -> None:
        if not Path(zim_path).is_file():
            raise ZimReaderError(f"ZIM file not found: {zim_path}")
        try:
            self._archive = Archive(str(zim_path))
        except Exception as exc:  # libzim raises plain RuntimeError
            raise ZimReaderError(f"Failed to open ZIM archive {zim_path}: {exc}") from exc

        if not self._archive.has_fulltext_index:
            raise ZimReaderError(
                f"ZIM archive {zim_path} has no embedded Xapian full-text index; "
                "hybrid retrieval requires one."
            )
        self._searcher = Searcher(self._archive)

    # ------------------------------------------------------------------ meta
    @property
    def provenance(self) -> dict:
        """Knowledge-base lineage metadata for EU AI Act provenance stamping."""

        def _meta(key: str) -> Optional[str]:
            try:
                return bytes(self._archive.get_metadata(key)).decode("utf-8", errors="replace")
            except Exception:
                return None

        return {
            "kb_type": "zim",
            "kb_title": _meta("Title"),
            "kb_name": _meta("Name"),
            "kb_date": _meta("Date"),  # e.g. "2026-06-.." — the snapshot version
            "kb_language": _meta("Language"),
            "kb_uuid": str(self._archive.uuid),
            "kb_entry_count": self._archive.entry_count,
            "kb_filename": Path(self._archive.filename).name,
        }

    # ---------------------------------------------------------------- search
    def search(self, query: str, limit: int) -> list[str]:
        """Full-text search against the embedded Xapian index.

        Returns article paths ordered by Xapian relevance. Never raises for an
        empty result — an empty list is a valid CRAG signal (triggers rewrite).
        """
        try:
            search = self._searcher.search(Query().set_query(query))
            estimated = search.getEstimatedMatches()
            if estimated == 0:
                return []
            return [str(path) for path in search.getResults(0, limit)]
        except Exception as exc:
            logger.warning("Xapian search failed for query %r: %s", query, exc)
            return []

    # ----------------------------------------------------------------- fetch
    def get_article(self, path: str) -> Optional[ZimArticle]:
        """Fetch one article by path; returns None for redirects/missing/non-HTML."""
        try:
            entry = self._archive.get_entry_by_path(path)
            if entry.is_redirect:
                entry = entry.get_redirect_entry()
            item = entry.get_item()
            if not item.mimetype.startswith("text/html"):
                return None
            html = bytes(item.content).decode("utf-8", errors="replace")
            return ZimArticle(path=entry.path, title=entry.title, html=html)
        except KeyError:
            logger.warning("Article path not found in ZIM: %r", path)
            return None
        except Exception as exc:
            logger.warning("Failed to read article %r: %s", path, exc)
            return None
