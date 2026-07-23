"""Shared data models passed between pipeline stages."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


@dataclass
class Chunk:
    """A slice of a ZIM article, the retrieval unit of the pipeline."""

    article_path: str
    article_title: str
    chunk_index: int
    text: str
    semantic_score: float = 0.0

    @property
    def chunk_id(self) -> str:
        """Stable identifier used for audit logging and embedding cache keys."""
        digest = hashlib.sha256(
            f"{self.article_path}#{self.chunk_index}#{self.text}".encode("utf-8")
        ).hexdigest()
        return digest[:16]


@dataclass
class GradedChunk:
    chunk: Chunk
    relevant: bool
    confidence: float  # 0.0 - 1.0, as reported by the Ollama grader
    reasoning: str = ""


@dataclass
class VerificationReport:
    """Output of the offline fact-checking agent (ZIM cross-reference only)."""

    score: int  # 0-100 correctness / anti-hallucination score
    claims_checked: int
    claims_supported: int
    details: list[dict] = field(default_factory=list)


class PipelineOutcome(str, Enum):
    ANSWERED = "answered"
    REFUSED = "refused"  # Article 14 structured fallback
    ERROR = "error"


@dataclass
class PipelineResult:
    """Final, user-facing result. Always carries EU AI Act metadata."""

    outcome: PipelineOutcome
    query: str
    answer: Optional[str] = None
    refusal_reason: Optional[str] = None
    graded_chunks: list[GradedChunk] = field(default_factory=list)
    rewrite_history: list[str] = field(default_factory=list)
    verification: Optional[VerificationReport] = None
    # Populated by the compliance layer:
    disclosure: dict = field(default_factory=dict)   # Article 50
    provenance: dict = field(default_factory=dict)   # knowledge-base lineage
    trace_id: str = ""
