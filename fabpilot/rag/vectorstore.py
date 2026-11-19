"""Vector store 后端：内存（默认）与 pgvector。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class SearchHit:
    id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class VectorStore(Protocol):
    def add(self, ids: list[str], vectors: np.ndarray, texts: list[str], metadatas: list[dict] | None = None) -> None:
        ...

    def search(self, query: np.ndarray, k: int = 5) -> list[SearchHit]:
        ...


class InMemoryVectorStore:
    """基于 Numpy 的余弦相似度存储。离线 / 开发的默认选项。"""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None
        self._texts: list[str] = []
        self._metadatas: list[dict] = []

    def add(self, ids: list[str], vectors: np.ndarray, texts: list[str], metadatas: list[dict] | None = None) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        self._ids.extend(ids)
        self._texts.extend(texts)
        self._metadatas.extend(metadatas or [{} for _ in ids])
        if self._matrix is None:
            self._matrix = vectors
        else:
            self._matrix = np.vstack([self._matrix, vectors])

    def search(self, query: np.ndarray, k: int = 5) -> list[SearchHit]:
        if self._matrix is None or len(self._ids) == 0:
            return []
        q = np.asarray(query, dtype=np.float32)
        qn = q / (np.linalg.norm(q) + 1e-9)
        sims = self._matrix @ qn
        order = np.argsort(-sims)[:k]
        return [
            SearchHit(
                id=self._ids[i],
                score=float(sims[i]),
                text=self._texts[i],
                metadata=self._metadatas[i],
            )
            for i in order
        ]


class PgVectorStore:
    """基于 pgvector 的存储（需要 ``pgvector`` 包以及已启用扩展的 Postgres 实例）。"""

    def __init__(self, connection: str, dim: int, table: str = "vector_documents") -> None:
        try:
            from pgvector.sqlalchemy import Vector  # noqa: F401
        except ImportError as exc:
            raise ImportError("pgvector package is required for PgVectorStore") from exc
        from sqlalchemy import JSON, Column, String, Text, create_engine
        from sqlalchemy.orm import declarative_base

        self.dim = dim
        self.engine = create_engine(connection, future=True)

        Base = declarative_base()

        class VectorDoc(Base):  # type: ignore[valid-type, misc]
            __tablename__ = table
            id = Column(String(64), primary_key=True)
            embedding = Column(Vector(dim))
            text = Column(Text)
            metadata_ = Column(JSON)

        self.table_cls = VectorDoc
        Base.metadata.create_all(self.engine)

    def add(self, ids: list[str], vectors: np.ndarray, texts: list[str], metadatas: list[dict] | None = None) -> None:
        from sqlalchemy.orm import Session

        with Session(self.engine) as s:
            for i, doc_id in enumerate(ids):
                s.merge(
                    self.table_cls(
                        id=doc_id,
                        embedding=vectors[i].tolist(),
                        text=texts[i],
                        metadata_=(metadatas or [{}])[i],
                    )
                )
            s.commit()

    def search(self, query: np.ndarray, k: int = 5) -> list[SearchHit]:
        from sqlalchemy.orm import Session

        with Session(self.engine) as s:
            rows = (
                s.query(self.table_cls)
                .order_by(self.table_cls.embedding.cosine_distance(query.tolist()))
                .limit(k)
                .all()
            )
        return [
            SearchHit(id=r.id, score=1.0 - r.embedding.cosine_distance(query.tolist()), text=r.text, metadata=r.metadata_ or {})
            for r in rows
        ]


def build_vector_store(settings=None) -> VectorStore:
    from fabpilot.config import get_settings

    settings = settings or get_settings()
    if settings.vector_store == "pgvector":
        return PgVectorStore(connection=settings.pgvector_connection, dim=settings.embedding_dim)
    return InMemoryVectorStore()
