"""QUERY flow:  question > retrieve (hybrid) > rerank > generate (cited) > result."""
import time
from dataclasses import dataclass, field

from llama_index.core.schema import NodeWithScore
from pydantic import BaseModel, Field


class _SubQueries(BaseModel):
    queries: list[str] = Field(default_factory=list, description="focused search queries covering the question")

from rag_app.config import Settings, get_settings
from rag_app.embedding.embedder import build_embed_model
from rag_app.generation.generator import build_llm, generate_answer
from rag_app.generation.prompts import QA_PROMPT_VERSION
from rag_app.generation.schemas import Answer
from rag_app.logging_utils import get_logger
from rag_app.reranking.reranker import build_reranker, rerank
from rag_app.retrieval.retriever import build_retriever
from rag_app.storage.vector_store import build_vector_store

log = get_logger(__name__)

# With AWS + Azure + GCP in one index, a query naming one cloud's service can be
# swamped by another cloud's semantically-similar docs (e.g. "Cloud SQL" pulling
# Amazon RDS). When the question clearly targets one provider, scope retrieval to
# that provider's docs. AWS dominates the index, so AWS queries need no filter;
# Azure/GCP are matched on their distinct `guide` value.
_GCP_MARKERS = (
    "gcp", "google cloud", "cloud run", "bigquery", "cloud sql", "gke",
    "kubernetes engine", "compute engine", "cloud storage", "cloud cdn", "apigee",
    "looker", "cloud spanner", "cloud bigtable", "cloud logging", "cloud monitoring",
)
_AZURE_MARKERS = (
    "azure", "aks", "cosmos", "blob storage", "app service", "container apps",
    "service bus", "event hub", "event grid", "logic apps", "synapse", "data factory",
    "front door", "api management", "static web app",
)


_AZURE_GUIDE = "Azure documentation"
_GCP_GUIDE = "Google Cloud documentation"


def _detect_cloud(question: str) -> str | None:
    """Infer the target provider from the question text (used when none is chosen)."""
    q = question.lower()
    gcp = any(m in q for m in _GCP_MARKERS)
    az = any(m in q for m in _AZURE_MARKERS)
    if gcp and not az:
        return "gcp"
    if az and not gcp:
        return "azure"
    return None


def build_filters(question: str, filters: dict[str, str] | None, cloud: str | None):
    """Scope retrieval to a provider + any exact-match filters, as MetadataFilters.

    ``cloud`` is the user's explicit choice (aws/azure/gcp/all). When unset we infer
    it from the question so a stray "Cloud SQL" doesn't pull Amazon RDS. AWS is
    expressed as "not Azure and not GCP" since its docs have many guide values.
    """
    from llama_index.core.vector_stores.types import (
        FilterOperator, MetadataFilter, MetadataFilters,
    )

    cloud = (cloud or "").lower().strip()
    if cloud not in {"aws", "azure", "gcp"}:
        cloud = _detect_cloud(question) or "all"

    mfs: list = []
    if cloud == "azure":
        mfs.append(MetadataFilter(key="guide", value=_AZURE_GUIDE, operator=FilterOperator.EQ))
    elif cloud == "gcp":
        mfs.append(MetadataFilter(key="guide", value=_GCP_GUIDE, operator=FilterOperator.EQ))
    elif cloud == "aws":
        mfs.append(MetadataFilter(key="guide", value=[_AZURE_GUIDE, _GCP_GUIDE], operator=FilterOperator.NIN))
    for k, v in (filters or {}).items():
        mfs.append(MetadataFilter(key=k, value=v, operator=FilterOperator.EQ))
    return MetadataFilters(filters=mfs) if mfs else None


@dataclass
class QueryResult:
    question: str
    answer: Answer
    sources: dict[str, NodeWithScore]
    prompt_version: str
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_payload(self) -> dict:
        """Clean, UI-ready JSON: a crisp summary, cited steps, and compact sources."""
        used = {sid for c in self.answer.claims for sid in c.source_ids}
        sources = []
        for sid, n in self.sources.items():
            if sid not in used:
                continue  # only surface sources actually cited in the answer
            m = n.node.metadata
            sources.append({
                "id": sid,
                "service": m.get("service"),
                "guide": m.get("guide"),
                "file": m.get("file_name"),
                "page": m.get("page_label"),
                "url": m.get("source_url"),
                "score": round(float(n.score), 3) if n.score is not None else None,
            })
        return {
            "question": self.question,
            "answer": self.answer.summary,
            "insufficient_context": self.answer.insufficient_context,
            "steps": [{"text": c.text, "sources": c.source_ids} for c in self.answer.claims],
            "sources": sources,
            "prompt_version": self.prompt_version,
            "timings_ms": {k: round(v) for k, v in self.timings_ms.items()},
        }


class RAGService:
    """Builds every component once; reuse it for many questions."""

    def __init__(self, s: Settings | None = None):
        self.s = s or get_settings()
        self.vector_store = build_vector_store(self.s)
        self.embed_model = build_embed_model(self.s)
        self.reranker = build_reranker(self.s) if self.s.rerank_enabled else None
        self._llm = None  # created lazily, so eval can run without an LLM key

    @property
    def llm(self):
        if self._llm is None:
            self._llm = build_llm(self.s)
        return self._llm

    def _plan_queries(self, question: str) -> list[str]:
        """Break the question into focused search queries that together cover it.

        Crucial for breadth: a comparison ('DynamoDB or RDS?') becomes one query per
        option; a multi-component architecture becomes one per component. Each is
        embedded separately so retrieval surfaces ALL the relevant services, not just
        whichever one dominates the sentence. Falls back to [question] without an LLM."""
        if not self.s.query_expansion:
            return [question]
        try:
            from llama_index.core import PromptTemplate

            tmpl = PromptTemplate(
                "Break the user's question into 1-4 focused documentation search queries that TOGETHER "
                "cover everything needed to answer it. For a comparison, make one query per option. "
                "For a multi-service/architecture task, make one query per component or service. "
                "Each query should name the likely AWS service(s) and terms. "
                "Return a JSON object: {{\"queries\": [\"...\"]}}.\n\nQuestion: {q}"
            )
            model = _SubQueries
            plan = self.llm.structured_predict(model, tmpl, q=question)
            queries = [q.strip() for q in plan.queries if q.strip()]
            # Always include the original so we never lose the user's phrasing.
            return ([question] + queries) if queries else [question]
        except Exception as exc:  # noqa: BLE001 - planning is best-effort
            log.warning("query planning skipped: %s", exc)
            return [question]

    def retrieve(self, question: str, filters: dict[str, str] | None = None,
                 cloud: str | None = None) -> list[NodeWithScore]:
        mf = build_filters(question, filters, cloud)
        retriever = build_retriever(self.vector_store, self.embed_model, self.s, mf)
        # Rerank EACH sub-query's hits against the original question and keep its best
        # few. This guarantees every option/component (e.g. DynamoDB AND RDS) is
        # represented, instead of the global top-N collapsing onto one service.
        picked: dict[str, NodeWithScore] = {}
        for sq in self._plan_queries(question):
            hits = retriever.retrieve(sq)
            ranked = rerank(self.reranker, question, hits) if self.reranker else hits
            for n in ranked[: self.s.rerank_per_query]:
                nid = n.node.node_id
                if nid not in picked or (n.score or 0) > (picked[nid].score or 0):
                    picked[nid] = n
        nodes = sorted(picked.values(), key=lambda n: n.score or 0, reverse=True)
        return nodes[: self.s.rerank_top_n]

    def _source_rows(self, source_map: dict[str, NodeWithScore], used: set[str]) -> list[dict]:
        rows = []
        for sid, n in source_map.items():
            if used and sid not in used:
                continue
            m = n.node.metadata
            rows.append({
                "id": sid, "service": m.get("service"), "guide": m.get("guide"),
                "file": m.get("file_name"), "page": m.get("page_label"),
                "url": m.get("source_url"),
                "score": round(float(n.score), 3) if n.score is not None else None,
            })
        return rows

    def design(self, question: str, filters: dict[str, str] | None = None,
               cloud: str | None = None) -> dict:
        """Detailed architecture-design payload (components + rationale + data flow + diagram)."""
        from rag_app.generation.design import design_graph, design_to_mermaid, generate_design

        t0 = time.perf_counter()
        nodes = self.retrieve(question, filters, cloud)
        t1 = time.perf_counter()
        d, source_map = generate_design(self.llm, question, nodes, embed_model=self.embed_model)
        t2 = time.perf_counter()
        used = ({s for c in d.components for s in c.source_ids}
                | {s for f in d.data_flow for s in f.source_ids})
        return {
            "mode": "design",
            "question": question,
            "insufficient_context": d.insufficient_context,
            "overview": d.overview,
            "assumptions": d.assumptions,
            "clarifying_questions": d.clarifying_questions,
            "components": [c.model_dump() for c in d.components],
            "data_flow": [f.model_dump() for f in d.data_flow],
            "considerations": d.considerations,
            "architecture_mermaid": design_to_mermaid(d),
            "graph": design_graph(d),
            "sources": self._source_rows(source_map, used),
            "prompt_version": "design-v1",
            "timings_ms": {"retrieve": round((t1 - t0) * 1000), "generate": round((t2 - t1) * 1000)},
        }

    def ask(self, question: str, filters: dict[str, str] | None = None,
            cloud: str | None = None) -> QueryResult:
        t0 = time.perf_counter()
        nodes = self.retrieve(question, filters, cloud)
        t1 = time.perf_counter()
        answer, sources = generate_answer(self.llm, question, nodes, embed_model=self.embed_model)
        t2 = time.perf_counter()
        return QueryResult(
            question=question,
            answer=answer,
            sources=sources,
            prompt_version=QA_PROMPT_VERSION,
            timings_ms={"retrieve": (t1 - t0) * 1000, "generate": (t2 - t1) * 1000},
        )
