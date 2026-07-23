"""Context Evaluator / Document Grader.

Asks the (small, fast) Ollama grader model whether each retrieved ZIM chunk
actually helps answer the user's query. Structured JSON output; confidences
feed both the CRAG corrective loop and the Article 12 audit trail.
"""

from __future__ import annotations

import logging

from src.crag.llm.ollama_client import OllamaClient, OllamaError
from src.crag.models import Chunk, GradedChunk

logger = logging.getLogger(__name__)

_GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reasoning": {"type": "string"},
    },
    "required": ["relevant", "confidence", "reasoning"],
}

_SYSTEM_PROMPT = (
    "You are a strict document relevance grader in a retrieval pipeline. "
    "Given a user question and a document excerpt from an offline Wikipedia "
    "snapshot, decide whether the excerpt contains information that helps "
    "answer the question. Be conservative: partial keyword overlap without "
    "substantive content is NOT relevant. Respond only with the requested JSON."
)


class DocumentGrader:
    def __init__(self, client: OllamaClient, model: str) -> None:
        self._client = client
        self._model = model

    def grade(self, query: str, chunks: list[Chunk]) -> list[GradedChunk]:
        graded: list[GradedChunk] = []
        for chunk in chunks:
            user_prompt = (
                f"Question:\n{query}\n\n"
                f"Document excerpt (from article '{chunk.article_title}'):\n"
                f"{chunk.text}"
            )
            try:
                verdict = self._client.chat_json(
                    model=self._model,
                    system=_SYSTEM_PROMPT,
                    user=user_prompt,
                    json_schema=_GRADE_SCHEMA,
                )
                graded.append(
                    GradedChunk(
                        chunk=chunk,
                        relevant=bool(verdict["relevant"]),
                        confidence=max(0.0, min(1.0, float(verdict["confidence"]))),
                        reasoning=str(verdict.get("reasoning", "")),
                    )
                )
            except (OllamaError, KeyError, TypeError, ValueError) as exc:
                # A chunk we cannot grade is treated as irrelevant — the
                # pipeline must never generate from unvetted context.
                logger.warning("Grading failed for chunk %s: %s", chunk.chunk_id, exc)
                graded.append(
                    GradedChunk(chunk=chunk, relevant=False, confidence=0.0,
                                reasoning=f"grading failed: {exc}")
                )
        return graded
