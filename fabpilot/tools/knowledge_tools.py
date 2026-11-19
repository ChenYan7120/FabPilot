"""Knowledge tools: hybrid document retrieval and similar-case retrieval."""
from __future__ import annotations

from pydantic import BaseModel, Field

from fabpilot.tools.registry import ToolContext, tool
from fabpilot.tools.schemas import RetrievedDoc, RetrievalResult
from fabpilot.utils import ToolValidationError


class DocumentRetrievalInput(BaseModel):
    query: str = Field(description="Natural-language query or symptom description.")
    k: int = Field(default=5, description="Number of top chunks to return.")
    doc_type: str | None = Field(
        default=None, description="Scope to a doc type: 'sop', 'alarm' or 'fault_cases'."
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
        "Retrieve relevant knowledge-base documents (SOPs, alarm handling guides, "
        "fault cases) using hybrid dense+sparse retrieval with RRF fusion. Returns "
        "ranked chunks with source citations."
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
    query: str = Field(description="Symptom / channel pattern to match against historical fault cases.")
    k: int = Field(default=5, description="Number of similar cases to return.")


@tool(
    name="similar_case_retrieval",
    description=(
        "Retrieve historical fault cases similar to the described symptom. Used by "
        "the Diagnosis Agent to ground hypotheses in precedent."
    ),
    input_model=SimilarCaseInput,
    tags=["knowledge"],
)
def similar_case_retrieval(ctx: ToolContext, query: str, k: int = 5) -> RetrievalResult:
    if ctx.knowledge_base is None:
        raise ToolValidationError("knowledge base is not configured")
    docs = ctx.knowledge_base.retrieve_similar_cases(query, k=k)
    return RetrievalResult(query=query, n_results=len(docs), results=[_map(d) for d in docs])
