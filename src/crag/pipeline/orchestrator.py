"""CRAG orchestrator — the corrective state machine.

retrieve -> grade -> (generate | rewrite-and-retry | refuse) -> verify

Article 14 (human oversight) guarantee: when the grader deems the context
irrelevant and the rewrite budget is exhausted, the pipeline emits a
structured refusal. It never generates an answer from unvetted context.
"""

from __future__ import annotations

import logging

from config.settings import Settings, get_settings
from src.crag.agents.generator import ResponseGenerator
from src.crag.agents.grader import DocumentGrader
from src.crag.agents.rewriter import QueryRewriter
from src.crag.agents.verifier import VerificationAgent
from src.crag.compliance.audit_logger import AuditLogger, scrub_pii
from src.crag.compliance.disclosure import build_disclosure
from src.crag.compliance.provenance import build_provenance
from src.crag.llm.ollama_client import OllamaClient, OllamaError
from src.crag.models import Chunk, GradedChunk, PipelineOutcome, PipelineResult
from src.crag.retrieval.chunker import chunk_article
from src.crag.retrieval.ranker import SemanticRanker
from src.crag.retrieval.zim_reader import ZimReader

logger = logging.getLogger(__name__)

REFUSAL_MESSAGE = (
    "I cannot answer this question reliably. The local offline knowledge base "
    "(Wikipedia ZIM snapshot) did not yield context that my relevance grader "
    "considers sufficient, even after reformulating the query. Rather than "
    "risk a hallucinated answer, I am declining to respond. Please rephrase "
    "the question or consult a human expert."
)


class CragPipeline:
    def __init__(
        self,
        settings: Settings,
        zim_reader: ZimReader,
        ranker: SemanticRanker,
        grader: DocumentGrader,
        rewriter: QueryRewriter,
        generator: ResponseGenerator,
        verifier: VerificationAgent,
        audit: AuditLogger,
    ) -> None:
        self._settings = settings
        self._zim = zim_reader
        self._ranker = ranker
        self._grader = grader
        self._rewriter = rewriter
        self._generator = generator
        self._verifier = verifier
        self._audit = audit
        self._static_provenance = build_provenance(
            zim_reader,
            generator_model=settings.generator_model,
            grader_model=settings.grader_model,
            embedding_model=settings.embedding_model,
        )

    # -------------------------------------------------------------- retrieval
    def _retrieve(self, query: str) -> list[Chunk]:
        paths = self._zim.search(query, limit=self._settings.top_k_articles)
        chunks: list[Chunk] = []
        for path in paths:
            article = self._zim.get_article(path)
            if article is not None:
                chunks.extend(
                    chunk_article(article, self._settings.chunk_size, self._settings.chunk_overlap)
                )
        return self._ranker.rank(query, chunks, top_n=self._settings.top_n_chunks)

    # --------------------------------------------------------------- pipeline
    def run(self, query: str) -> tuple[PipelineResult, str | None]:
        """Execute the full CRAG loop. Returns ``(result, final_prompt)``.

        The result is fully compliance-stamped and audit-logged before return.
        """
        trace_id = self._audit.new_trace_id()
        result = PipelineResult(
            outcome=PipelineOutcome.ERROR, query=query, trace_id=trace_id
        )
        result.disclosure = build_disclosure(self._settings.generator_model)
        result.provenance = dict(self._static_provenance)
        final_prompt: str | None = None

        try:
            relevant = self._corrective_loop(query, trace_id, result)

            if relevant is None:
                result.outcome = PipelineOutcome.REFUSED
                result.refusal_reason = REFUSAL_MESSAGE
            else:
                answer, final_prompt = self._generator.generate(query, relevant)
                result.answer = answer
                result.outcome = PipelineOutcome.ANSWERED
                result.verification = self._verifier.verify(answer)
        except OllamaError as exc:
            logger.error("Pipeline failed (trace %s): %s", trace_id, exc)
            result.outcome = PipelineOutcome.ERROR
            result.refusal_reason = f"Local inference engine error: {exc}"

        self._audit.log_result(result, final_prompt)
        return result, final_prompt

    def _corrective_loop(
        self, original_query: str, trace_id: str, result: PipelineResult
    ) -> list[GradedChunk] | None:
        """Retrieve/grade/rewrite loop. Returns relevant chunks, or None to refuse."""
        current_query = original_query

        for attempt in range(self._settings.max_rewrites + 1):
            top_chunks = self._retrieve(current_query)
            graded = self._grader.grade(current_query, top_chunks)
            result.graded_chunks = graded

            relevant = [g for g in graded if g.relevant]
            ratio = len(relevant) / len(graded) if graded else 0.0
            self._audit.log_event(trace_id, "retrieval_graded", {
                "attempt": attempt,
                "query": scrub_pii(current_query),
                "chunks_retrieved": len(top_chunks),
                "chunks_relevant": len(relevant),
                "relevance_ratio": round(ratio, 4),
            })

            if relevant and ratio >= self._settings.relevance_threshold:
                return relevant

            if attempt >= self._settings.max_rewrites:
                return None

            current_query = self._rewriter.rewrite(
                original_query, failed_queries=[original_query, *result.rewrite_history]
            )
            result.rewrite_history.append(current_query)
            self._audit.log_event(trace_id, "query_rewritten", {
                "attempt": attempt + 1,
                "rewritten_query": scrub_pii(current_query),
            })

        return None


def build_pipeline(settings: Settings | None = None) -> CragPipeline:
    """Factory: wires every component from configuration. One call per process."""
    settings = settings or get_settings()

    zim_reader = ZimReader(settings.zim_file_path)
    ranker = SemanticRanker(settings.embedding_model, settings.embedding_cache_dir)
    client = OllamaClient(settings.ollama_base_url)

    return CragPipeline(
        settings=settings,
        zim_reader=zim_reader,
        ranker=ranker,
        grader=DocumentGrader(client, settings.grader_model),
        rewriter=QueryRewriter(client, settings.generator_model),
        generator=ResponseGenerator(client, settings.generator_model),
        verifier=VerificationAgent(
            client,
            settings.generator_model,
            zim_reader=zim_reader,
            ranker=ranker,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
        audit=AuditLogger(settings.log_dir),
    )
