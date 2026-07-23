# Agentic Corrective RAG Pipeline (Sparky)

Local-first Corrective RAG (CRAG) pipeline for the **NVIDIA DGX Spark** (Grace
Blackwell, 128 GB unified LPDDR5x, DGX OS / Ubuntu 24.04 arm64). Fully offline:
the knowledge base is a Wikipedia ZIM snapshot, inference runs in the local
`ai_inference_ollama` Docker container, and answers are fact-checked against
the same ZIM archive. Designed from the ground up for **EU AI Act
(limited-risk / GPAI deployer)** compliance.

## Architecture

```
User query (Streamlit, localhost)
  └─> PII scrub + audit trace
      └─> ZIM Xapian full-text search (top-K articles, hybrid on-demand)
          └─> chunk -> embed (cached) -> semantic rerank (top-N chunks)
              └─> Ollama Document Grader (per-chunk relevance + confidence)
                  ├─ enough relevant ──> Ollama Generator (cited answer)
                  │                       └─> Verification Agent
                  │                           (claims -> ZIM evidence -> 0-100 score)
                  └─ low relevance ──> Ollama Query Rewriter ──> retry
                                        └─ budget exhausted ──> structured REFUSAL
```

Retrieval is **hybrid on-demand**: nothing is pre-indexed. The ZIM file's
embedded Xapian index finds candidate articles per query; only those are
chunked and embedded (embeddings content-addressed on disk under
`data/embedding_cache/`, so recurring topics get faster).

## Layout

| Path | Purpose |
|---|---|
| `config/settings.py` | Pydantic settings; single source of truth for `.env` |
| `src/crag/retrieval/` | ZIM reader (libzim), HTML chunker, semantic ranker |
| `src/crag/llm/ollama_client.py` | Localhost-only Ollama client, JSON mode, retries |
| `src/crag/agents/` | Grader, rewriter, generator, verification agent |
| `src/crag/pipeline/orchestrator.py` | CRAG state machine + refusal fallback |
| `src/crag/compliance/` | Art. 50 disclosure, provenance, Art. 12 audit logger |
| `src/crag/ui/app.py` | Streamlit UI (disclosure banner, provenance, scores) |
| `scripts/preflight.py` | Environment diagnostics |
| `logs/` | Append-only JSONL audit records (PII-scrubbed) |

## Setup

```bash
cd /home/havran/crag_pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # already provided; adjust if paths differ

# Pull the 4-bit quantized (Q4_K_M GGUF) models into the Ollama container:
docker exec ai_inference_ollama ollama pull qwen3:14b
docker exec ai_inference_ollama ollama pull qwen3:4b

python scripts/preflight.py   # verify ZIM, Ollama, caches, permissions
```

Note: the first query downloads the embedding model (`BAAI/bge-base-en-v1.5`)
into `HF_HOME` once; after that the pipeline sets `HF_HUB_OFFLINE=1`
automatically and never touches the network again.

## Run

```bash
streamlit run src/crag/ui/app.py --server.address 127.0.0.1
```

Tests (no ZIM / no Ollama needed):

```bash
pytest tests/ -v
```

## EU AI Act compliance mapping

| Requirement | Implementation |
|---|---|
| Art. 50 transparency | `compliance/disclosure.py`; every result carries `ai_generated: true` + disclosure text; the UI renders a non-dismissable banner and labels answers as AI-generated |
| Data provenance | `compliance/provenance.py`; ZIM name/date/UUID (2026-06 snapshot) + exact model tags stamped on every response and log record |
| Art. 12 record keeping | `compliance/audit_logger.py`; per-interaction JSONL in `logs/`: scrubbed query, chunk IDs + article paths, grader confidences, rewrite history, final generator prompt, verification score. PII (emails, phones, IPs, cards, IBANs) is scrubbed before writing |
| Art. 14 human oversight | Hard rewrite budget (`MAX_REWRITES`); on exhaustion the orchestrator emits a structured refusal instead of generating from unvetted context |
| Cybersecurity sandbox | Loopback-only enforcement in both `config/settings.py` and the Ollama client (defense in depth); zero external API calls at runtime |

## Configuration

All knobs live in `.env` (documented in `.env.example`): `ZIM_FILE_PATH`,
`HF_HOME`, `OLLAMA_BASE_URL`, `GENERATOR_MODEL`, `GRADER_MODEL`,
`EMBEDDING_MODEL`, `TOP_K_ARTICLES`, `TOP_N_CHUNKS`, `RELEVANCE_THRESHOLD`,
`MAX_REWRITES`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `LOG_DIR`,
`EMBEDDING_CACHE_DIR`.
