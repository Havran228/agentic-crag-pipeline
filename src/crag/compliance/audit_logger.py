"""EU AI Act Article 12 — automated audit logging and record keeping.

Append-only JSONL records in the local ``logs/`` directory, one file per day.
Logged per interaction: the (PII-scrubbed) user query, retrieved chunk
identifiers and article paths, grader confidence scores, the exact final
prompt sent to the generator, the outcome, and the verification score.

Absolutely NO PII may be persisted: every free-text field passes through
``scrub_pii`` before being written.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.crag.models import PipelineResult

# Conservative PII patterns. Order matters: more specific first.
_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"), "[EMAIL]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[CARD]"),                    # payment cards
    (re.compile(r"\+?\d{1,3}[ .-]?\(?\d{1,4}\)?(?:[ .-]?\d{2,4}){2,4}"), "[PHONE]"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IPV4]"),
    (re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){4,7}[0-9a-fA-F]{1,4}\b"), "[IPV6]"),
    (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), "[IBAN]"),
]


def scrub_pii(text: str) -> str:
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class AuditLogger:
    """Thread-safe JSONL audit writer (Streamlit reruns share one instance)."""

    def __init__(self, log_dir: Path) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _log_file(self) -> Path:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self._log_dir / f"audit_{day}.jsonl"

    def _write(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with open(self._log_file(), "a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    # ------------------------------------------------------------ interfaces
    def new_trace_id(self) -> str:
        return uuid.uuid4().hex

    def log_event(self, trace_id: str, event: str, payload: dict) -> None:
        """Low-level event record (retrieval, grading, rewrite, ...)."""
        self._write({
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "trace_id": trace_id,
            "event": event,
            **payload,
        })

    def log_result(self, result: PipelineResult, final_prompt: str | None) -> None:
        """Full interaction record — the Article 12 core requirement."""
        self._write({
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "trace_id": result.trace_id,
            "event": "pipeline_result",
            "outcome": result.outcome.value,
            "query": scrub_pii(result.query),
            "rewrite_history": [scrub_pii(q) for q in result.rewrite_history],
            "retrieved_chunks": [
                {
                    "chunk_id": g.chunk.chunk_id,
                    "article_path": g.chunk.article_path,
                    "article_title": g.chunk.article_title,
                    "semantic_score": round(g.chunk.semantic_score, 4),
                    "grader_relevant": g.relevant,
                    "grader_confidence": round(g.confidence, 4),
                }
                for g in result.graded_chunks
            ],
            "final_prompt": scrub_pii(final_prompt) if final_prompt else None,
            "answer": scrub_pii(result.answer) if result.answer else None,
            "refusal_reason": result.refusal_reason,
            "verification_score": result.verification.score if result.verification else None,
            "provenance": result.provenance,
            "disclosure": result.disclosure,
        })
