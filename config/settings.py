"""Central configuration for the CRAG pipeline.

Single source of truth: every module reads configuration through
``get_settings()`` — never from ``os.environ`` directly. Values come from the
project ``.env`` file (see ``.env.example`` for documentation of each key).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Hosts considered part of Sparky's local sandbox. The EU AI Act compliance
# posture of this project requires zero external traffic.
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Data sources ---
    zim_file_path: Path = Field(alias="ZIM_FILE_PATH")

    # --- Model caches ---
    hf_home: Path = Field(alias="HF_HOME")

    # --- Inference engine ---
    ollama_base_url: str = Field(alias="OLLAMA_BASE_URL")
    generator_model: str = Field(alias="GENERATOR_MODEL", default="qwen3:14b")
    grader_model: str = Field(alias="GRADER_MODEL", default="qwen3:4b")
    embedding_model: str = Field(alias="EMBEDDING_MODEL", default="BAAI/bge-base-en-v1.5")

    # --- Retrieval tuning ---
    top_k_articles: int = Field(alias="TOP_K_ARTICLES", default=8, ge=1)
    top_n_chunks: int = Field(alias="TOP_N_CHUNKS", default=6, ge=1)
    relevance_threshold: float = Field(alias="RELEVANCE_THRESHOLD", default=0.34, ge=0.0, le=1.0)
    max_rewrites: int = Field(alias="MAX_REWRITES", default=2, ge=0)
    chunk_size: int = Field(alias="CHUNK_SIZE", default=1200, ge=100)
    chunk_overlap: int = Field(alias="CHUNK_OVERLAP", default=200, ge=0)

    # --- Compliance / audit ---
    log_dir: Path = Field(alias="LOG_DIR", default=Path("logs"))
    embedding_cache_dir: Path = Field(alias="EMBEDDING_CACHE_DIR", default=Path("data/embedding_cache"))

    @field_validator("ollama_base_url")
    @classmethod
    def _enforce_loopback(cls, value: str) -> str:
        """Refuse any inference endpoint outside Sparky's local sandbox."""
        host = urlparse(value).hostname
        if host not in _LOOPBACK_HOSTS:
            raise ValueError(
                f"OLLAMA_BASE_URL host {host!r} is not loopback. "
                "This pipeline must remain 100% offline (EU AI Act sandbox)."
            )
        return value.rstrip("/")

    @field_validator("log_dir", "embedding_cache_dir")
    @classmethod
    def _anchor_to_project_root(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value


def _apply_offline_env(settings: Settings) -> None:
    """Point Hugging Face at the local cache and force offline mode once the
    embedding model is already cached, so runtime never touches the network."""
    os.environ.setdefault("HF_HOME", str(settings.hf_home))
    model_cache = settings.hf_home / "hub"
    if model_cache.exists() and any(model_cache.glob("models--*")):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    _apply_offline_env(settings)
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    settings.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
    return settings
