"""Smoke tests — no ZIM file, no Ollama, no network required.

Covers: settings validation (incl. the loopback guard), PII scrubbing, the
chunker, and the full orchestrator refusal path using stub components.

Run from the project root:  pytest tests/ -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Settings
from src.crag.compliance.audit_logger import AuditLogger, scrub_pii
from src.crag.models import Chunk, GradedChunk, PipelineOutcome, VerificationReport
from src.crag.retrieval.chunker import chunk_article
from src.crag.retrieval.zim_reader import ZimArticle


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = dict(
        ZIM_FILE_PATH=str(tmp_path / "kb.zim"),
        HF_HOME=str(tmp_path / "hf"),
        OLLAMA_BASE_URL="http://localhost:11434",
        LOG_DIR=str(tmp_path / "logs"),
        EMBEDDING_CACHE_DIR=str(tmp_path / "cache"),
    )
    values.update(overrides)
    return Settings(_env_file=None, **values)


# --------------------------------------------------------------- settings
def test_settings_load(tmp_path):
    settings = make_settings(tmp_path)
    assert settings.generator_model
    assert settings.max_rewrites >= 0


def test_settings_reject_external_endpoint(tmp_path):
    with pytest.raises(Exception):
        make_settings(tmp_path, OLLAMA_BASE_URL="http://api.example.com:11434")


# ------------------------------------------------------------ PII scrubber
@pytest.mark.parametrize(
    "raw,expected_token",
    [
        ("contact me at jane.doe@example.com please", "[EMAIL]"),
        ("my ip is 192.168.1.42 ok", "[IPV4]"),
        ("call +49 170 1234567 now", "[PHONE]"),
        ("card 4111 1111 1111 1111 thanks", "[CARD]"),
    ],
)
def test_scrub_pii(raw, expected_token):
    scrubbed = scrub_pii(raw)
    assert expected_token in scrubbed
    assert "@" not in scrubbed if expected_token == "[EMAIL]" else True


def test_scrub_pii_leaves_normal_text_alone():
    text = "The Eiffel Tower was completed in 1889."
    assert scrub_pii(text) == text


# ---------------------------------------------------------------- chunker
def test_chunk_article_overlap_and_attribution():
    html = "<html><body><p>" + "Sentence one. " * 300 + "</p></body></html>"
    article = ZimArticle(path="A/Test", title="Test", html=html)
    chunks = chunk_article(article, chunk_size=500, chunk_overlap=100)
    assert len(chunks) > 1
    assert all(c.article_path == "A/Test" for c in chunks)
    assert all(len(c.text) <= 500 for c in chunks)
    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids)  # stable, unique ids


def test_chunk_article_strips_html_noise():
    html = "<html><body><script>evil()</script><p>Real content here.</p></body></html>"
    article = ZimArticle(path="A/X", title="X", html=html)
    chunks = chunk_article(article, chunk_size=500, chunk_overlap=50)
    assert chunks and "evil" not in chunks[0].text


# --------------------------------------------- orchestrator refusal path
class StubZim:
    provenance = {"kb_title": "stub", "kb_date": "2026-06", "kb_uuid": "0" * 32}

    def search(self, query, limit):
        return ["A/Stub"]

    def get_article(self, path):
        return ZimArticle(path=path, title="Stub", html="<p>Totally unrelated text.</p>")


class StubRanker:
    def rank(self, query, chunks, top_n):
        return chunks[:top_n]


class StubGrader:
    def grade(self, query, chunks):
        # Everything irrelevant -> forces the corrective loop to exhaust.
        return [GradedChunk(chunk=c, relevant=False, confidence=0.1) for c in chunks]


class StubRewriter:
    def __init__(self):
        self.calls = 0

    def rewrite(self, original_query, failed_queries):
        self.calls += 1
        return f"rewrite #{self.calls} of {original_query}"


class StubGenerator:
    def generate(self, query, relevant_chunks):
        raise AssertionError("generator must not run when context is irrelevant")


class StubVerifier:
    def verify(self, answer):
        return VerificationReport(score=100, claims_checked=0, claims_supported=0)


def test_refusal_after_rewrite_budget_exhausted(tmp_path):
    from src.crag.pipeline.orchestrator import CragPipeline

    settings = make_settings(tmp_path, MAX_REWRITES=2)
    rewriter = StubRewriter()
    pipeline = CragPipeline(
        settings=settings,
        zim_reader=StubZim(),
        ranker=StubRanker(),
        grader=StubGrader(),
        rewriter=rewriter,
        generator=StubGenerator(),
        verifier=StubVerifier(),
        audit=AuditLogger(settings.log_dir),
    )

    result, final_prompt = pipeline.run("what is the airspeed of an unladen swallow")

    # Article 14: structured refusal, never a hallucinated answer.
    assert result.outcome == PipelineOutcome.REFUSED
    assert result.answer is None
    assert final_prompt is None
    assert rewriter.calls == 2
    assert len(result.rewrite_history) == 2
    # Article 50 + provenance stamped even on refusals.
    assert result.disclosure["ai_generated"] is True
    assert result.provenance["kb_date"] == "2026-06"

    # Article 12: audit trail written, PII-free, with grader confidences.
    log_files = list(settings.log_dir.glob("audit_*.jsonl"))
    assert len(log_files) == 1
    records = [json.loads(line) for line in log_files[0].read_text().splitlines()]
    events = {r["event"] for r in records}
    assert {"retrieval_graded", "query_rewritten", "pipeline_result"} <= events
    final = [r for r in records if r["event"] == "pipeline_result"][0]
    assert final["outcome"] == "refused"
    assert all("grader_confidence" in c for c in final["retrieved_chunks"])
