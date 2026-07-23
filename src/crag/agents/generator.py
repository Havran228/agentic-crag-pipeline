"""Final Response Generator.

Synthesizes the answer strictly from graded, relevant ZIM chunks, with
numbered citations back to the source article paths. Returns both the answer
and the exact final prompt sent to the model — the latter is required for the
Article 12 audit trail.
"""

from __future__ import annotations

from src.crag.llm.ollama_client import OllamaClient
from src.crag.models import GradedChunk

_SYSTEM_PROMPT = (
    "You are an answer synthesizer in an offline retrieval pipeline. Answer "
    "the user's question using ONLY the numbered context excerpts provided. "
    "Cite excerpts inline as [1], [2], etc. If the context is insufficient "
    "for part of the question, say so explicitly instead of guessing. Never "
    "use knowledge that is not in the excerpts. Be concise and factual."
)


class ResponseGenerator:
    def __init__(self, client: OllamaClient, model: str) -> None:
        self._client = client
        self._model = model

    @staticmethod
    def build_context(relevant_chunks: list[GradedChunk]) -> str:
        blocks = []
        for i, graded in enumerate(relevant_chunks, start=1):
            chunk = graded.chunk
            blocks.append(
                f"[{i}] (article: {chunk.article_title} | path: {chunk.article_path})\n"
                f"{chunk.text}"
            )
        return "\n\n".join(blocks)

    def generate(self, query: str, relevant_chunks: list[GradedChunk]) -> tuple[str, str]:
        """Returns ``(answer, final_prompt)``."""
        context = self.build_context(relevant_chunks)
        final_prompt = (
            f"Context excerpts from the offline Wikipedia knowledge base:\n\n"
            f"{context}\n\n"
            f"Question: {query}"
        )
        answer = self._client.chat(
            model=self._model,
            system=_SYSTEM_PROMPT,
            user=final_prompt,
            temperature=0.2,
        )
        return answer, final_prompt
