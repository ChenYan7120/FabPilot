"""RAG pipeline 测试：chunking、混合检索与 doc_type 作用域。"""
from __future__ import annotations

from fabpilot.rag.pipeline import split_text, load_documents


def test_split_text_short_text_returns_single_chunk():
    assert split_text("short") == ["short"]


def test_split_text_preserves_paragraphs():
    text = "\n\n".join(f"paragraph {i} " * 20 for i in range(10))
    chunks = split_text(text, chunk_size=200, overlap=20)
    assert len(chunks) > 1
    assert "".join(chunks).replace("\n\n", "") != ""  # 内容未整体丢失
    assert all(c for c in chunks)


def test_retrieve_returns_docs(knowledge_base):
    docs = knowledge_base.retrieve("gas flow drift", k=5)
    assert docs
    assert all(d.text for d in docs)


def test_retrieve_doc_type_scoping(knowledge_base):
    docs = knowledge_base.retrieve("gas flow drift", k=5, doc_type="sop")
    assert all(d.doc_type == "sop" for d in docs)


def test_retrieve_similar_cases_only_fault_cases(knowledge_base):
    docs = knowledge_base.retrieve_similar_cases("gas flow drift", k=5)
    assert docs
    assert all(d.doc_type == "fault_cases" for d in docs)


def test_load_documents_infers_doc_type(tmp_path):
    (tmp_path / "sop").mkdir()
    (tmp_path / "sop" / "a.md").write_text("# Title\nbody", encoding="utf-8")
    chunks = load_documents(tmp_path)
    assert chunks
    assert all(c.doc_type == "sop" for c in chunks)
    assert all(c.source == "sop/a.md" for c in chunks)
