"""Streamlit UI for the CRAG pipeline (localhost only).

Run from the project root:

    streamlit run src/crag/ui/app.py --server.address 127.0.0.1

Article 50 enforcement: the AI-disclosure banner is rendered unconditionally
on every page load and every answer is explicitly labeled as AI-generated.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit executes this file as a script; make project imports resolvable.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from config.settings import get_settings
from src.crag.compliance.disclosure import DISCLOSURE_TEXT
from src.crag.models import PipelineOutcome
from src.crag.pipeline.orchestrator import build_pipeline

st.set_page_config(page_title="Sparky CRAG — Offline Wikipedia Assistant", page_icon=None)


@st.cache_resource(show_spinner="Loading pipeline (ZIM archive + embedding model)...")
def _get_pipeline():
    return build_pipeline()


def _render_disclosure_banner() -> None:
    # Article 50: unconditional, non-dismissable disclosure.
    st.warning(f"AI SYSTEM NOTICE — {DISCLOSURE_TEXT}")


def _render_provenance(provenance: dict) -> None:
    with st.expander("Data provenance and lineage (knowledge-base version)"):
        st.markdown(
            f"- **Knowledge base:** {provenance.get('kb_title')} "
            f"({provenance.get('kb_filename')})\n"
            f"- **Snapshot date:** {provenance.get('kb_date')}\n"
            f"- **Archive UUID:** `{provenance.get('kb_uuid')}`\n"
            f"- **Entries:** {provenance.get('kb_entry_count')}\n"
            f"- **Generator model:** `{provenance.get('generator_model')}`\n"
            f"- **Grader model:** `{provenance.get('grader_model')}`\n"
            f"- **Embedding model:** `{provenance.get('embedding_model')}`\n"
            f"- **Network isolation:** {provenance.get('network_isolation')}"
        )


def _render_chunks(result) -> None:
    with st.expander(f"Retrieved context ({len(result.graded_chunks)} chunks, graded)"):
        for graded in result.graded_chunks:
            chunk = graded.chunk
            status = "relevant" if graded.relevant else "rejected"
            st.markdown(
                f"**{chunk.article_title}** — `{chunk.article_path}` | "
                f"grader: {status} ({graded.confidence:.2f}) | "
                f"semantic score: {chunk.semantic_score:.3f}"
            )
            st.caption(chunk.text[:400] + ("..." if len(chunk.text) > 400 else ""))


def main() -> None:
    _render_disclosure_banner()
    st.title("Sparky CRAG — Offline Wikipedia Assistant")
    st.caption(
        "Corrective RAG on NVIDIA DGX Spark. 100% local: ZIM knowledge base, "
        "Ollama inference, offline verification. EU AI Act limited-risk posture."
    )

    try:
        settings = get_settings()
        pipeline = _get_pipeline()
    except Exception as exc:
        st.error(
            "Pipeline could not start. Run `python scripts/preflight.py` for "
            f"diagnostics.\n\nDetails: {exc}"
        )
        st.stop()

    query = st.text_input(
        "Your question",
        placeholder="e.g. What is the Grace Blackwell architecture?",
    )
    if not st.button("Ask", type="primary") or not query.strip():
        return

    with st.spinner("Retrieving, grading, generating, verifying..."):
        result, _ = pipeline.run(query.strip())

    if result.outcome == PipelineOutcome.ANSWERED:
        st.subheader("AI-generated answer")
        st.info(result.answer)

        if result.verification is not None:
            verification = result.verification
            col1, col2 = st.columns(2)
            col1.metric("Verification score (0-100)", verification.score)
            col2.metric(
                "Claims supported",
                f"{verification.claims_supported}/{verification.claims_checked}",
            )
            if verification.details:
                with st.expander("Fact-check details (offline ZIM cross-reference)"):
                    for detail in verification.details:
                        st.markdown(
                            f"- **{'Supported' if detail.get('supported') else 'Unsupported'}:** "
                            f"{detail.get('claim', detail.get('error', ''))}"
                        )
                        if detail.get("reasoning"):
                            st.caption(detail["reasoning"])
    elif result.outcome == PipelineOutcome.REFUSED:
        st.subheader("Refused (human-oversight fallback)")
        st.error(result.refusal_reason)
        if result.rewrite_history:
            st.caption("Query reformulations attempted: " + "; ".join(result.rewrite_history))
    else:
        st.subheader("Error")
        st.error(result.refusal_reason or "Unknown pipeline error.")

    _render_chunks(result)
    _render_provenance(result.provenance)
    st.caption(
        f"Trace ID `{result.trace_id}` — full audit record in `{settings.log_dir}` "
        "(EU AI Act Article 12 record keeping)."
    )


main()
