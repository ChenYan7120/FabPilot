"""Hybrid retrieval: dense (vector) + sparse (BM25) with RRF fusion and rerank.

This mirrors the retrieval design of Ragent (dense + keyword parallel recall,
RRF fusion, optional rerank) reimplemented in Python and scoped to FabPilot's
needs — a design-pattern port, not a code port.
"""
from __future__ import annotations

import math
import re
from typing import Any, Protocol, runtime_checkable

from fabpilot.rag.vectorstore import SearchHit, VectorStore
from fabpilot.utils import get_logger

logger = get_logger(__name__)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Compact Okapi BM25 over tokenized documents."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs: list[dict] = []
        self.doc_freq: dict[str, int] = {}
        self.doc_lengths: list[int] = []
        self.avgdl: float = 0.0

    def index(self, docs: list[dict]) -> None:
        self.docs = docs
        self.doc_freq = {}
        self.doc_lengths = []
        for doc in docs:
            tokens = tokenize(doc["text"])
            self.doc_lengths.append(len(tokens))
            for t in set(tokens):
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        if not self.docs:
            return []
        q_tokens = tokenize(query)
        n = len(self.docs)
        scores: dict[str, float] = {}
        for doc in self.docs:
            tokens = tokenize(doc["text"])
            dl = len(tokens)
            tf = {t: tokens.count(t) for t in q_tokens}
            score = 0.0
            for t in q_tokens:
                df = self.doc_freq.get(t, 0)
                if df == 0:
                    continue
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                ft = tf.get(t, 0)
                score += idf * (ft * (self.k1 + 1)) / (ft + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9)))
            if score > 0:
                scores[doc["id"]] = score
        return sorted(scores.items(), key=lambda x: -x[1])[:k]


def rrf_fusion(ranked_lists: list[list[tuple[str, float]]], k: int = 60) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return fused


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        ...


class NoopReranker:
    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        return hits


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedder,
        bm25: BM25Index | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25 = bm25 or BM25Index()
        self.reranker = reranker or NoopReranker()
        self._doc_map: dict[str, SearchHit] = {}

    def index(self, docs: list[dict]) -> None:
        self.bm25.index(docs)
        vectors = self.embedder.embed([d["text"] for d in docs])
        self.vector_store.add(
            ids=[d["id"] for d in docs],
            vectors=vectors,
            texts=[d["text"] for d in docs],
            metadatas=[d.get("metadata", {}) for d in docs],
        )
        self._doc_map = {
            d["id"]: SearchHit(id=d["id"], score=0.0, text=d["text"], metadata=d.get("metadata", {}))
            for d in docs
        }

    def search(self, query: str, k: int = 5, filters: dict[str, Any] | None = None) -> list[SearchHit]:
        qvec = self.embedder.embed([query])[0]
        dense_raw = self.vector_store.search(qvec, k=k * 2)
        sparse_raw = self.bm25.search(query, k=k * 2)

        dense_ranked = [(h.id, h.score) for h in dense_raw]
        fused = rrf_fusion([dense_ranked, sparse_raw])

        hits: list[SearchHit] = []
        for doc_id, score in sorted(fused.items(), key=lambda x: -x[1]):
            if doc_id not in self._doc_map:
                continue
            hit = self._doc_map[doc_id]
            if filters and not _matches(hit.metadata, filters):
                continue
            hits.append(SearchHit(id=hit.id, score=score, text=hit.text, metadata=hit.metadata))
            if len(hits) >= k:
                break

        return self.reranker.rerank(query, hits)


def _matches(metadata: dict, filters: dict) -> bool:
    for key, value in filters.items():
        if metadata.get(key) != value:
            return False
    return True
