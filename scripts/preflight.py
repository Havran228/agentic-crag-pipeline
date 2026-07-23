"""Preflight diagnostics for the CRAG pipeline on Sparky.

Run from the project root before first use:

    python scripts/preflight.py

Checks, in dependency order: configuration, ZIM archive (incl. Xapian index
and provenance metadata), Ollama endpoint and pulled models, Hugging Face
cache, and writability of the audit-log and embedding-cache directories.
Exits non-zero if any check fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PASS, FAIL, WARN = "[ OK ]", "[FAIL]", "[WARN]"
failures = 0


def report(ok: bool, label: str, detail: str = "", warn_only: bool = False) -> None:
    global failures
    tag = PASS if ok else (WARN if warn_only else FAIL)
    if not ok and not warn_only:
        failures += 1
    print(f"{tag} {label}" + (f" — {detail}" if detail else ""))


def main() -> int:
    # 1. Configuration ------------------------------------------------------
    try:
        from config.settings import get_settings

        settings = get_settings()
        report(True, "Configuration loaded from .env")
    except Exception as exc:
        report(False, "Configuration", str(exc))
        print("\nPreflight aborted: fix .env first (see .env.example).")
        return 1

    # 2. ZIM archive --------------------------------------------------------
    try:
        from src.crag.retrieval.zim_reader import ZimReader

        reader = ZimReader(settings.zim_file_path)
        meta = reader.provenance
        report(True, "ZIM archive open + Xapian full-text index present",
               f"{meta.get('kb_title')} | date {meta.get('kb_date')} | "
               f"{meta.get('kb_entry_count')} entries | uuid {meta.get('kb_uuid')}")
        sample = reader.search("computer", limit=1)
        report(bool(sample), "ZIM full-text search returns results",
               sample[0] if sample else "no hits for probe query")
    except Exception as exc:
        report(False, "ZIM archive", str(exc))

    # 3. Ollama endpoint ----------------------------------------------------
    try:
        from src.crag.llm.ollama_client import OllamaClient

        client = OllamaClient(settings.ollama_base_url, timeout_s=10.0)
        if client.is_available():
            report(True, f"Ollama reachable at {settings.ollama_base_url}")
            pulled = client.list_models()
            for needed in (settings.generator_model, settings.grader_model):
                present = any(m == needed or m.startswith(needed + ":") for m in pulled)
                report(
                    present, f"Model pulled: {needed}",
                    "" if present else
                    f"run: docker exec ai_inference_ollama ollama pull {needed}",
                )
        else:
            report(False, "Ollama endpoint",
                   f"{settings.ollama_base_url} not responding — is the "
                   "'ai_inference_ollama' container running?")
        client.close()
    except Exception as exc:
        report(False, "Ollama endpoint", str(exc))

    # 4. Hugging Face cache / embedding model -------------------------------
    hf_ok = settings.hf_home.exists() or _mkdir_ok(settings.hf_home)
    report(hf_ok, f"HF_HOME usable: {settings.hf_home}")
    cached = list((settings.hf_home / "hub").glob("models--*")) if hf_ok else []
    report(
        bool(cached), f"Embedding model cached ({settings.embedding_model})",
        "" if cached else "first run will download it once; offline afterwards",
        warn_only=True,
    )

    # 5. Writable runtime directories ---------------------------------------
    for label, path in (("Audit log dir", settings.log_dir),
                        ("Embedding cache dir", settings.embedding_cache_dir)):
        try:
            probe = path / ".write_probe"
            probe.write_text("ok")
            probe.unlink()
            report(True, f"{label} writable: {path}")
        except OSError as exc:
            report(False, f"{label}", str(exc))

    print()
    if failures:
        print(f"Preflight FAILED: {failures} check(s) need attention.")
        return 1
    print("Preflight passed: the CRAG pipeline is ready.")
    return 0


def _mkdir_ok(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        return False


if __name__ == "__main__":
    sys.exit(main())
