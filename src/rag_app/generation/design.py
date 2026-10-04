"""Architecture-design mode: produce a detailed, reasoned design (how AND why),
clarifying questions about requirements, and a richer diagram — for questions like
"design a data lake / DR setup / scalable web app".

This is distinct from the crisp how-to answer: design questions deserve components
with rationale, a data flow, trade-offs, and the requirements the user should
confirm (data volume, structure, latency, budget).
"""
import re

from llama_index.core import PromptTemplate
from llama_index.core.llms import LLM
from llama_index.core.schema import NodeWithScore
from pydantic import BaseModel, Field

from rag_app.generation.diagram import (
    Architecture, DiagramEdge, DiagramNode, architecture_graph, symbol_for, to_mermaid,
)
from rag_app.generation.generator import build_source_map, format_context

_DESIGN_TRIGGERS = re.compile(
    r"\b(design|architect(ure)?|data lake|data warehouse|platform|highly available|"
    r"high availability|disaster recovery|multi-region|scalab|end[- ]to[- ]end|"
    r"reference architecture|well[- ]architected)\b",
    re.I,
)


def is_design_question(q: str) -> bool:
    return bool(_DESIGN_TRIGGERS.search(q or ""))


class ArchComponent(BaseModel):
    name: str = Field(description="Layer/component name, e.g. 'Ingestion' or 'Data warehouse'.")
    service: str = Field(description="The AWS service used, e.g. 'Amazon Redshift'.")
    purpose: str = Field(description="What it does AND why this service was chosen (the rationale).")
    source_ids: list[str] = Field(default_factory=list)


class FlowStep(BaseModel):
    step: str = Field(description="What happens at this stage of the data/request flow.")
    why: str = Field(default="", description="Why it is done this way.")
    source_ids: list[str] = Field(default_factory=list)


class Connection(BaseModel):
    source: str = Field(description="name of the component the link starts FROM (match a component name).")
    target: str = Field(description="name of the component the link goes TO (match a component name).")
    label: str = Field(default="", description="relationship, e.g. 'routes to', 'reads/writes', 'monitors', 'scales', 'triggers'.")


class ArchDesign(BaseModel):
    overview: str = Field(default="", description="2-3 sentence summary of the proposed architecture.")
    assumptions: list[str] = Field(default_factory=list, description="Assumptions made about the requirements.")
    clarifying_questions: list[str] = Field(
        default_factory=list,
        description="Questions to ask the user to refine the design (data volume, structure, latency, budget, SLAs).",
    )
    components: list[ArchComponent] = Field(default_factory=list)
    connections: list[Connection] = Field(
        default_factory=list,
        description="The REAL links between components (not a straight line): routing, data, monitoring, scaling, triggers.",
    )
    data_flow: list[FlowStep] = Field(default_factory=list)
    considerations: list[str] = Field(
        default_factory=list, description="Cost, scaling, security and reliability trade-offs (the 'why')."
    )
    insufficient_context: bool = False


_DESIGN_PROMPT = PromptTemplate(
    "You are a senior AWS solutions architect. Using the sources below, propose a DETAILED, "
    "production-grade architecture for the request. Developers must be able to follow it.\n\n"
    "Produce:\n"
    "- overview: 2-3 sentences describing the architecture.\n"
    "- assumptions: the requirement assumptions you made.\n"
    "- clarifying_questions: concrete questions to tailor the design (e.g. expected data volume/size, "
    "data structure/format, read vs write patterns, latency/SLA, budget, compliance).\n"
    "- components: each layer/service with its purpose AND why it was chosen over alternatives.\n"
    "- connections: the REAL links between components (source->target with a label like 'routes to', "
    "'reads/writes', 'monitors', 'scales', 'triggers'). Reflect the true topology — fan-out, branches, "
    "monitoring and management links — NOT a single straight line.\n"
    "- data_flow: the end-to-end flow, step by step, each with a short 'why'.\n"
    "- considerations: cost, scaling, security and reliability trade-offs.\n\n"
    "Rules:\n"
    "- Ground the design in the sources and cite source ids (e.g. S1) on components and flow steps.\n"
    "- Be specific and technical (name services, patterns, formats). No vague filler.\n"
    "- If the sources are unrelated to the request, set insufficient_context=true.\n\n"
    "Sources:\n{context}\n\n"
    "Request: {question}\n"
)


def generate_design(llm: LLM, question: str, nodes: list[NodeWithScore], embed_model=None
                    ) -> tuple[ArchDesign, dict[str, NodeWithScore]]:
    source_map = build_source_map(nodes)
    if not source_map:
        return ArchDesign(insufficient_context=True), source_map
    design = llm.structured_predict(
        ArchDesign, _DESIGN_PROMPT, question=question, context=format_context(source_map)
    )
    valid = set(source_map)
    for comp in design.components:
        comp.source_ids = [s for s in comp.source_ids if s in valid]
    for step in design.data_flow:
        step.source_ids = [s for s in step.source_ids if s in valid]
    return design, source_map


def _resolve_component(design: ArchDesign, key: str) -> str | None:
    """Map a connection's source/target name to a component id (exact, then fuzzy)."""
    k = (key or "").strip().lower()
    if not k:
        return None
    comps = design.components
    for i, c in enumerate(comps):
        if c.name.strip().lower() == k or (c.service and c.service.strip().lower() == k):
            return f"c{i}"
    for i, c in enumerate(comps):
        nm, sv = c.name.lower(), (c.service or "").lower()
        if k in nm or nm in k or (sv and (k in sv or sv in k)):
            return f"c{i}"
    return None


def _design_edges(design: ArchDesign) -> list[DiagramEdge]:
    """Edges from the LLM's real connections; fall back to a sequence only if none resolve."""
    edges, seen = [], set()
    for conn in design.connections:
        s, t = _resolve_component(design, conn.source), _resolve_component(design, conn.target)
        if s and t and s != t and (s, t) not in seen:
            seen.add((s, t))
            edges.append(DiagramEdge(source=s, target=t, label=conn.label or ""))
    if not edges:  # no usable topology -> linear fallback
        edges = [DiagramEdge(source=f"c{i}", target=f"c{i+1}") for i in range(len(design.components) - 1)]
    return edges


def _design_arch(design: ArchDesign) -> Architecture:
    return Architecture(
        nodes=[DiagramNode(id=f"c{i}", label=f"{c.name}: {c.service}" if c.service else c.name,
                           service=c.service or c.name)
               for i, c in enumerate(design.components)],
        edges=_design_edges(design),
    )


def design_to_mermaid(design: ArchDesign) -> str:
    """Mermaid flow reflecting the real component connections."""
    if not design.components:
        return ""
    return to_mermaid(_design_arch(design))


def design_graph(design: ArchDesign) -> dict:
    """Node/edge graph (with per-node detail) for the interactive 3D renderer."""
    if not design.components:
        return {"nodes": [], "edges": []}
    g = architecture_graph(_design_arch(design))
    for node, comp in zip(g["nodes"], design.components):
        node["label"] = comp.name
        node["service"] = comp.service
        node["detail"] = comp.purpose
    return g
