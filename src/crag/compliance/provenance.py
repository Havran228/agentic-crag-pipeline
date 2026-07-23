"""Data provenance and lineage (EU AI Act record-keeping support).

Stamps every response with the exact version of the offline knowledge base
(the Wikipedia ZIM snapshot: name, date, UUID) plus the model tags used, so
the origin of every piece of generated information can be proven after the
fact.
"""

from __future__ import annotations

from src.crag.retrieval.zim_reader import ZimReader


def build_provenance(zim_reader: ZimReader, generator_model: str, grader_model: str,
                     embedding_model: str) -> dict:
    return {
        **zim_reader.provenance,  # kb_title, kb_date (2026-06 snapshot), kb_uuid, ...
        "generator_model": generator_model,
        "grader_model": grader_model,
        "embedding_model": embedding_model,
        "network_isolation": "localhost-only, zero external API calls",
    }
