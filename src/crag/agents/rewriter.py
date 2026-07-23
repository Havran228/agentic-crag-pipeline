"""Query Rewriter — the corrective step of CRAG.

Triggered when the grader finds the retrieved context insufficient. Produces a
reformulation optimized for keyword-style full-text search over an offline
Wikipedia snapshot (the ZIM Xapian index).
"""

from __future__ import annotations

from src.crag.llm.ollama_client import OllamaClient

_REWRITE_SCHEMA = {
    "type": "object",
    "properties": {"rewritten_query": {"type": "string"}},
    "required": ["rewritten_query"],
}

_SYSTEM_PROMPT = (
    "You are a query reformulation assistant for a full-text search engine "
    "over an offline Wikipedia snapshot. The previous query retrieved "
    "irrelevant documents. Rewrite it to maximize retrieval quality: expand "
    "abbreviations, use canonical encyclopedic terminology, drop filler "
    "words, and add likely article-title keywords. Keep the original intent. "
    "Respond only with the requested JSON."
)


class QueryRewriter:
    def __init__(self, client: OllamaClient, model: str) -> None:
        self._client = client
        self._model = model

    def rewrite(self, original_query: str, failed_queries: list[str]) -> str:
        attempts = "\n".join(f"- {q}" for q in failed_queries) or "- (none)"
        user_prompt = (
            f"Original user question:\n{original_query}\n\n"
            f"Queries already tried without success:\n{attempts}\n\n"
            "Produce ONE improved search query, different from all previous attempts."
        )
        result = self._client.chat_json(
            model=self._model,
            system=_SYSTEM_PROMPT,
            user=user_prompt,
            json_schema=_REWRITE_SCHEMA,
            temperature=0.4,
        )
        return str(result["rewritten_query"]).strip()
