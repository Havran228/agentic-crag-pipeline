"""HTTP client for the local Ollama container (``ai_inference_ollama``).

Hard constraints enforced here:
- The endpoint host MUST be loopback (data-sovereignty sandbox). This is
  validated in ``config.settings`` and re-validated here, defense in depth.
- All models are expected to be 4-bit quantized GGUF (Q4_K_M — Ollama's
  default quantization), mirroring NVFP4 efficiency on Blackwell.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}


class OllamaError(RuntimeError):
    """Raised for transport failures or malformed model output."""


class OllamaClient:
    def __init__(self, base_url: str, timeout_s: float = 300.0, retries: int = 2) -> None:
        host = urlparse(base_url).hostname
        if host not in _LOOPBACK_HOSTS:
            raise OllamaError(
                f"Refusing non-loopback Ollama endpoint {base_url!r}: "
                "all inference must stay on Sparky's local sandbox."
            )
        self._base_url = base_url.rstrip("/")
        self._retries = retries
        self._client = httpx.Client(base_url=self._base_url, timeout=timeout_s)

    # ----------------------------------------------------------------- infra
    def is_available(self) -> bool:
        try:
            return self._client.get("/api/version").status_code == 200
        except httpx.HTTPError:
            return False

    def list_models(self) -> list[str]:
        try:
            response = self._client.get("/api/tags")
            response.raise_for_status()
            return [m["name"] for m in response.json().get("models", [])]
        except httpx.HTTPError as exc:
            raise OllamaError(f"Could not list Ollama models: {exc}") from exc

    # ------------------------------------------------------------------ chat
    def chat(
        self,
        model: str,
        system: str,
        user: str,
        json_schema: Optional[dict] = None,
        temperature: float = 0.1,
    ) -> str:
        """Single-turn chat completion. If ``json_schema`` is given, Ollama's
        structured-output mode constrains the response to that schema."""
        payload: dict[str, Any] = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": temperature},
        }
        if json_schema is not None:
            payload["format"] = json_schema

        last_error: Optional[Exception] = None
        for attempt in range(self._retries + 1):
            try:
                response = self._client.post("/api/chat", json=payload)
                response.raise_for_status()
                content = response.json()["message"]["content"]
                return content.strip()
            except (httpx.HTTPError, KeyError) as exc:
                last_error = exc
                logger.warning(
                    "Ollama chat attempt %d/%d failed (model=%s): %s",
                    attempt + 1, self._retries + 1, model, exc,
                )
        raise OllamaError(f"Ollama chat failed after retries (model={model}): {last_error}")

    def chat_json(
        self,
        model: str,
        system: str,
        user: str,
        json_schema: dict,
        temperature: float = 0.0,
    ) -> dict:
        """Chat completion that must return a JSON object matching the schema."""
        raw = self.chat(model, system, user, json_schema=json_schema, temperature=temperature)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OllamaError(f"Model {model} returned invalid JSON: {raw[:300]!r}") from exc

    def close(self) -> None:
        self._client.close()
