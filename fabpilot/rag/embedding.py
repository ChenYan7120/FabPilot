"""Embedding providers.

Two backends behind one interface:
  * ``OpenAICompatibleEmbedder`` — calls any OpenAI-compatible ``/embeddings`` endpoint.
  * ``HashEmbedder`` — deterministic bag-of-token hashing (no network), used for
    offline tests and development so RAG works end-to-end without a model service.
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable

import httpx
import numpy as np

from fabpilot.config import Settings, get_settings
from fabpilot.utils import ConfigurationError

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:  # shape (n, dim)
        ...


class HashEmbedder:
    """Deterministic pseudo-embedding: shared tokens yield similar vectors."""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in set(_tokenize(text)):
                h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                rng = np.random.default_rng(h)
                out[i] += rng.normal(size=self.dim).astype(np.float32)
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out


class OpenAICompatibleEmbedder:
    def __init__(self, base_url: str, api_key: str, model: str, dim: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dim = dim
        if not api_key:
            raise ConfigurationError("Embedding API key is empty; use FABPILOT_EMBEDDING_PROVIDER=hash offline.")

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = httpx.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": texts},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        data.sort(key=lambda d: d["index"])
        return np.asarray([d["embedding"] for d in data], dtype=np.float32)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def build_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    if settings.embedding_provider == "hash":
        return HashEmbedder(dim=settings.embedding_dim)
    if settings.embedding_provider == "openai_compatible":
        return OpenAICompatibleEmbedder(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
        )
    raise ConfigurationError(f"Unknown embedding provider: {settings.embedding_provider}")
