"""Offline Verification Agent (fact-checking / anti-hallucination).

After the generator produces an answer, this agent:
1. extracts the discrete factual claims from the answer (LLM, JSON mode),
2. searches the local ZIM archive for evidence for each claim — strictly
   offline, never the live internet,
3. judges each claim against the retrieved evidence (LLM, JSON mode),
4. returns a 0-100 correctness score (share of claims supported by the ZIM).
"""

from __future__ import annotations

import logging

from src.crag.llm.ollama_client import OllamaClient, OllamaError
from src.crag.models import VerificationReport
from src.crag.retrieval.chunker import chunk_article
from src.crag.retrieval.ranker import SemanticRanker
from src.crag.retrieval.zim_reader import ZimReader

logger = logging.getLogger(__name__)

_CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
    },
    "required": ["claims"],
}

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "required": ["supported", "reasoning"],
}

_EXTRACT_SYSTEM = (
    "Extract the discrete, checkable factual claims from the given AI-generated "
    "answer. Ignore hedges, meta-statements, and opinions. At most 6 claims, "
    "each self-contained (resolve pronouns). Respond only with the requested JSON."
)

_JUDGE_SYSTEM = (
    "You are a strict fact-checker. Decide whether the claim is supported by "
    "the evidence excerpts from an offline Wikipedia snapshot. If the evidence "
    "does not clearly support the claim, mark it unsupported. Respond only "
    "with the requested JSON."
)


class VerificationAgent:
    def __init__(
        self,
        client: OllamaClient,
        model: str,
        zim_reader: ZimReader,
        ranker: SemanticRanker,
        chunk_size: int,
        chunk_overlap: int,
        evidence_articles: int = 3,
        evidence_chunks: int = 4,
    ) -> None:
        self._client = client
        self._model = model
        self._zim = zim_reader
        self._ranker = ranker
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._evidence_articles = evidence_articles
        self._evidence_chunks = evidence_chunks

    # ------------------------------------------------------------ extraction
    def _extract_claims(self, answer: str) -> list[str]:
        result = self._client.chat_json(
            model=self._model,
            system=_EXTRACT_SYSTEM,
            user=f"AI-generated answer to fact-check:\n\n{answer}",
            json_schema=_CLAIMS_SCHEMA,
        )
        return [c.strip() for c in result.get("claims", []) if c and c.strip()]

    # -------------------------------------------------------------- evidence
    def _gather_evidence(self, claim: str) -> str:
        paths = self._zim.search(claim, limit=self._evidence_articles)
        chunks = []
        for path in paths:
            article = self._zim.get_article(path)
            if article is not None:
                chunks.extend(chunk_article(article, self._chunk_size, self._chunk_overlap))
        if not chunks:
            return ""
        best = self._ranker.rank(claim, chunks, top_n=self._evidence_chunks)
        return "\n\n".join(
            f"(from '{c.article_title}')\n{c.text}" for c in best
        )

    # ----------------------------------------------------------------- score
    def verify(self, answer: str) -> VerificationReport:
        try:
            claims = self._extract_claims(answer)
        except OllamaError as exc:
            logger.warning("Claim extraction failed: %s", exc)
            return VerificationReport(score=0, claims_checked=0, claims_supported=0,
                                      details=[{"error": str(exc)}])

        if not claims:
            # Nothing checkable (e.g. a pure refusal) — treated as neutral-safe.
            return VerificationReport(score=100, claims_checked=0, claims_supported=0)

        supported = 0
        details: list[dict] = []
        for claim in claims:
            evidence = self._gather_evidence(claim)
            if not evidence:
                details.append({"claim": claim, "supported": False,
                                "reasoning": "no evidence found in ZIM archive"})
                continue
            try:
                verdict = self._client.chat_json(
                    model=self._model,
                    system=_JUDGE_SYSTEM,
                    user=f"Claim:\n{claim}\n\nEvidence excerpts:\n{evidence}",
                    json_schema=_JUDGE_SCHEMA,
                )
                is_supported = bool(verdict["supported"])
            except (OllamaError, KeyError, TypeError) as exc:
                logger.warning("Judging failed for claim %r: %s", claim, exc)
                is_supported = False
                verdict = {"reasoning": f"judging failed: {exc}"}
            supported += int(is_supported)
            details.append({"claim": claim, "supported": is_supported,
                            "reasoning": str(verdict.get("reasoning", ""))})

        score = round(100 * supported / len(claims))
        return VerificationReport(
            score=score,
            claims_checked=len(claims),
            claims_supported=supported,
            details=details,
        )
