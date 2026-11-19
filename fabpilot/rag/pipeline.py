"""RAG pipeline: document loading, chunking and the KnowledgeBase facade."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fabpilot.config import Settings, get_settings
from fabpilot.rag.embedding import build_embedder
from fabpilot.rag.retriever import HybridRetriever
from fabpilot.rag.vectorstore import SearchHit, build_vector_store
from fabpilot.utils import get_logger, new_id

logger = get_logger(__name__)


@dataclass
class RetrievedDoc:
    id: str
    text: str
    source: str
    doc_type: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    text: str
    source: str
    doc_type: str
    chunk_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def split_text(text: str, chunk_size: int = 500, overlap: int = 60) -> list[str]:
    """Sentence/paragraph-aware chunker with overlap."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            current = para
            while len(current) > chunk_size:
                cut = current.rfind(" ", 0, chunk_size)
                cut = cut if cut > 0 else chunk_size
                chunks.append(current[:cut])
                current = current[cut - overlap if cut - overlap > 0 else 0 :]
    if current:
        chunks.append(current)
    return [c for c in chunks if c]


def load_documents(kb_dir: Path) -> list[Chunk]:
    """Load markdown/text documents from a knowledge-base directory tree.

    The immediate parent directory name is used as ``doc_type``
    (e.g. sop / alarm / fault_cases) so retrieval can be scoped.
    """
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        doc_type = path.parent.name
        title = text.strip().splitlines()[0].lstrip("# ").strip() if text.strip() else path.stem
        pieces = split_text(text)
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    text=piece,
                    source=str(path.relative_to(kb_dir)),
                    doc_type=doc_type,
                    chunk_index=i,
                    metadata={"title": title, "doc_type": doc_type, "source": str(path)},
                )
            )
    return chunks


class KnowledgeBase:
    """Facade over the hybrid retriever for the Knowledge Agent."""

    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    def index_chunks(self, chunks: list[Chunk]) -> int:
        docs = [
            {
                "id": new_id("doc"),
                "text": c.text,
                "metadata": {**c.metadata, "doc_type": c.doc_type, "chunk_index": c.chunk_index, "source": c.source},
            }
            for c in chunks
        ]
        self.retriever.index(docs)
        return len(docs)

    def index_dir(self, kb_dir: Path) -> int:
        return self.index_chunks(load_documents(kb_dir))

    def retrieve(self, query: str, k: int = 5, doc_type: str | None = None) -> list[RetrievedDoc]:
        filters = {"doc_type": doc_type} if doc_type else None
        hits = self.retriever.search(query, k=k, filters=filters)
        return [_to_retrieved(h) for h in hits]

    def retrieve_similar_cases(self, query: str, k: int = 5) -> list[RetrievedDoc]:
        return self.retrieve(query, k=k, doc_type="fault_cases")


def _to_retrieved(hit: SearchHit) -> RetrievedDoc:
    return RetrievedDoc(
        id=hit.id,
        text=hit.text,
        source=hit.metadata.get("source", ""),
        doc_type=hit.metadata.get("doc_type", ""),
        score=hit.score,
        metadata=hit.metadata,
    )


def build_knowledge_base(settings: Settings | None = None) -> KnowledgeBase:
    settings = settings or get_settings()
    embedder = build_embedder(settings)
    store = build_vector_store(settings)
    retriever = HybridRetriever(vector_store=store, embedder=embedder)
    kb = KnowledgeBase(retriever)
    kb_dir = settings.knowledge_base_path
    if kb_dir.exists():
        n = kb.index_dir(kb_dir)
        logger.info("Indexed %d chunks from %s", n, kb_dir)
    else:
        logger.warning("Knowledge base dir not found: %s", kb_dir)
    return kb
