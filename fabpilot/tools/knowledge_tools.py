"""知识工具：混合文档检索和相似案例检索。"""
from __future__ import annotations

from pydantic import BaseModel, Field

from fabpilot.tools.registry import ToolContext, tool
from fabpilot.tools.schemas import RetrievedDoc, RetrievalResult
from fabpilot.utils import ToolValidationError


class DocumentRetrievalInput(BaseModel):
    query: str = Field(description="自然语言查询或症状描述。")
    k: int = Field(default=5, description="要返回的 top chunk 数量。")
    doc_type: str | None = Field(
        default=None, description="限定到某一 doc type：'sop'、'alarm' 或 'fault_cases'。"
    )


def _map(doc) -> RetrievedDoc:
    return RetrievedDoc(
        chunk_id=doc.id,
        doc_id=doc.metadata.get("title", ""),
        doc_type=doc.doc_type,
        title=doc.metadata.get("title", ""),
        text=doc.text,
        score=round(float(doc.score), 6),
        source=doc.source,
    )


@tool(
    name="document_retrieval",
    description=(
        "使用带 RRF 融合的混合 dense+sparse 检索，检索相关的知识库文档（SOP、"
        "告警处理指南、故障案例）。返回带来源引用的排序后 chunk。"
    ),
    input_model=DocumentRetrievalInput,
    tags=["knowledge"],
)
def document_retrieval(
    ctx: ToolContext, query: str, k: int = 5, doc_type: str | None = None
) -> RetrievalResult:
    if ctx.knowledge_base is None:
        raise ToolValidationError("knowledge base is not configured")
    docs = ctx.knowledge_base.retrieve(query, k=k, doc_type=doc_type)
    return RetrievalResult(query=query, n_results=len(docs), results=[_map(d) for d in docs])


class SimilarCaseInput(BaseModel):
    query: str = Field(description="用于匹配历史故障案例的症状 / 通道模式。")
    k: int = Field(default=5, description="要返回的相似案例数量。")


@tool(
    name="similar_case_retrieval",
    description=(
        "检索与所描述症状相似的历史故障案例。由诊断 Agent 使用，"
        "用先例为假设提供依据。"
    ),
    input_model=SimilarCaseInput,
    tags=["knowledge"],
)
def similar_case_retrieval(ctx: ToolContext, query: str, k: int = 5) -> RetrievalResult:
    if ctx.knowledge_base is None:
        raise ToolValidationError("knowledge base is not configured")
    docs = ctx.knowledge_base.retrieve_similar_cases(query, k=k)
    return RetrievalResult(query=query, n_results=len(docs), results=[_map(d) for d in docs])
