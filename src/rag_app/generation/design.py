"""Architecture-design mode: produce a detailed, reasoned solution-architect design
(how AND why), clarifying questions about requirements, and a diagram — for
questions like "design a data lake / DR setup / scalable web app".

This is distinct from the crisp how-to answer: design questions deserve components
with rationale, a data flow, non-functional requirements, trade-offs, and the
requirements the user should confirm (data volume, structure, latency, budget).

Quality comes from three passes: draft -> critique -> refine. The critique is a
principal-architect review of the draft, and the refine pass must address it. Each
component is also assigned a TIER so the UI can lay the design out as a real
architecture blueprint instead of a blob.
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

# Canonical top-to-bottom tiers. The blueprint renders one band per tier, so this
# order IS the vertical order of the diagram.
TIERS = ["client", "edge", "application", "integration", "data", "analytics", "security", "observability"]

# Fallback classification when the model gives an unknown tier.
_TIER_PATTERNS = [
    ("client", r"\b(user|client|browser|mobile app|end[- ]user|external|device|publisher)\b"),
    ("edge", r"\b(cdn|cloudfront|media cdn|front door|waf|shield|dns|route ?53|global accelerator|"
             r"load balanc|application gateway|api gateway|api management|apigee|ingress|traffic manager)\b"),
    ("integration", r"\b(queue|topic|sqs|sns|pub/?sub|service bus|event grid|event hub|eventbridge|"
                    r"kafka|kinesis|stream|step functions|logic apps|workflow|data factory|orchestrat)\b"),
    ("data", r"\b(database|db|sql|dynamodb|cosmos|firestore|bigtable|spanner|rds|aurora|"
             r"storage|s3|blob|bucket|data lake|cache|redis|elasticache|memorystore|table)\b"),
    ("analytics", r"\b(analytic|warehouse|redshift|bigquery|synapse|snowflake|glue|etl|emr|dataflow|"
                  r"databricks|looker|quicksight|power bi|sagemaker|vertex ai|machine learning|\bml\b|feature store)\b"),
    ("security", r"\b(iam|identity|kms|key vault|secret|cognito|entra|auth|certificate|guardduty|security)\b"),
    ("observability", r"\b(monitor|logging|log analytics|cloudwatch|observab|tracing|trace|alert|metric|insights)\b"),
    ("application", r"\b(lambda|function|compute|ec2|vm|instance|container|ecs|eks|gke|aks|kubernetes|"
                    r"app service|cloud run|fargate|server|api|service|microservice|backend|web)\b"),
]


def is_design_question(q: str) -> bool:
    return bool(_DESIGN_TRIGGERS.search(q or ""))


def _normalize_tier(tier: str, name: str, service: str) -> str:
    t = (tier or "").strip().lower()
    if t in TIERS:
        return t
    blob = f"{name} {service}".lower()
    for canon, pat in _TIER_PATTERNS:
        if re.search(pat, blob):
            return canon
    return "application"


class ArchComponent(BaseModel):
    name: str = Field(description="Layer/component name, e.g. 'Ingestion' or 'Data warehouse'.")
    service: str = Field(description="The cloud service used, e.g. 'Amazon Redshift' or 'BigQuery'.")
    tier: str = Field(
        default="application",
        description="One of: client, edge, application, integration, data, analytics, security, observability.",
    )
    purpose: str = Field(description="What it does AND why this service was chosen (the rationale).")
    sizing: str = Field(default="", description="Concrete sizing/config hint, e.g. instance class, RU/s, partitions.")
    source_ids: list[str] = Field(default_factory=list)


class FlowStep(BaseModel):
    step: str = Field(description="What happens at this stage of the data/request flow.")
    why: str = Field(default="", description="Why it is done this way.")
    source_ids: list[str] = Field(default_factory=list)


class Connection(BaseModel):
    source: str = Field(description="name of the component the link starts FROM (match a component name).")
    target: str = Field(description="name of the component the link goes TO (match a component name).")
    label: str = Field(default="", description="relationship, e.g. 'routes to', 'reads/writes', 'monitors', 'scales', 'triggers'.")


class NFR(BaseModel):
    aspect: str = Field(description="One of: availability, scalability, performance, security, cost, operations, disaster recovery.")
    approach: str = Field(description="How the design satisfies it, concretely.")


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
    non_functional: list[NFR] = Field(
        default_factory=list, description="How the design meets availability, scaling, security, cost and operations."
    )
    considerations: list[str] = Field(
        default_factory=list, description="Cost, scaling, security and reliability trade-offs (the 'why')."
    )
    insufficient_context: bool = False


_RULES = (
    "Rules:\n"
    "- Use ONLY the cloud provider implied by the request and the sources. Never mix providers.\n"
    "- Ground the design in the sources and cite source ids (e.g. S1) on components and flow steps.\n"
    "- Assign every component a tier from: client, edge, application, integration, data, analytics, "
    "security, observability.\n"
    "- connections must reflect the TRUE topology — fan-out, branches, monitoring and management links — "
    "NOT a single straight line. Every component should be reachable.\n"
    "- Be specific and technical (name services, patterns, formats, sizing). No vague filler.\n"
    "- If the sources are unrelated to the request, set insufficient_context=true.\n"
)

_DESIGN_PROMPT = PromptTemplate(
    "You are a senior cloud solutions architect. Using the sources below, propose a DETAILED, "
    "production-grade architecture for the request. Engineers must be able to build from it.\n\n"
    "Produce:\n"
    "- overview: 2-3 sentences describing the architecture.\n"
    "- assumptions: the requirement assumptions you made.\n"
    "- clarifying_questions: concrete questions to tailor the design (expected data volume/size, "
    "data structure/format, read vs write patterns, latency/SLA, budget, compliance).\n"
    "- components: each layer/service with its tier, purpose, why it beat the alternatives, and a sizing hint.\n"
    "- connections: source->target with a label like 'routes to', 'reads/writes', 'monitors', 'triggers'.\n"
    "- data_flow: the end-to-end flow, step by step, each with a short 'why'.\n"
    "- non_functional: how the design meets availability, scalability, performance, security, cost, "
    "operations and disaster recovery.\n"
    "- considerations: the key trade-offs.\n\n"
    + _RULES +
    "\nSources:\n{context}\n\nRequest: {question}\n"
)

_CRITIQUE_PROMPT = PromptTemplate(
    "You are a principal cloud architect doing a design review. Critique the draft architecture below "
    "against the request and the sources. Be specific and terse — a bulleted list of concrete defects.\n\n"
    "Look for: missing components (auth, networking, caching, DR, observability); wrong or sub-optimal "
    "service choices; components that are isolated (no connections); missing or wrong data-flow steps; "
    "missing non-functional coverage; claims not supported by the sources; provider mixing; wrong tiers.\n"
    "If the draft is already sound, say so briefly instead of inventing problems.\n\n"
    "Sources:\n{context}\n\nRequest: {question}\n\nDraft design:\n{draft}\n\nReview:"
)

_REFINE_PROMPT = PromptTemplate(
    "You are a senior cloud solutions architect. Produce the FINAL architecture for the request by "
    "revising the draft to address every valid point in the review. Keep what was already correct; "
    "do not drop good components. The result must be complete and self-consistent.\n\n"
    + _RULES +
    "\nSources:\n{context}\n\nRequest: {question}\n\nDraft design:\n{draft}\n\n"
    "Review to address:\n{critique}\n"
)


def _design_text(d: ArchDesign) -> str:
    """Compact readable rendering of a design, for the critique/refine passes."""
    parts = [f"Overview: {d.overview}"]
    if d.components:
        parts.append("Components:")
        parts += [f"  - [{c.tier}] {c.name} ({c.service}): {c.purpose}"
                  + (f" | sizing: {c.sizing}" if c.sizing else "") for c in d.components]
    if d.connections:
        parts.append("Connections:")
        parts += [f"  - {c.source} -> {c.target} ({c.label})" for c in d.connections]
    if d.data_flow:
        parts.append("Data flow:")
        parts += [f"  - {f.step}" for f in d.data_flow]
    if d.non_functional:
        parts.append("Non-functional:")
        parts += [f"  - {n.aspect}: {n.approach}" for n in d.non_functional]
    if d.considerations:
        parts.append("Considerations: " + "; ".join(d.considerations))
    return "\n".join(parts)


def _clean(design: ArchDesign, source_map: dict) -> ArchDesign:
    """Keep only real citations and normalize tiers."""
    valid = set(source_map)
    for comp in design.components:
        comp.source_ids = [s for s in comp.source_ids if s in valid]
        comp.tier = _normalize_tier(comp.tier, comp.name, comp.service)
    for step in design.data_flow:
        step.source_ids = [s for s in step.source_ids if s in valid]
    return design


def generate_design(llm: LLM, question: str, nodes: list[NodeWithScore], embed_model=None,
                    refine: bool | None = None) -> tuple[ArchDesign, dict[str, NodeWithScore]]:
    """Draft -> critique -> refine. Set ``refine=False`` (or RAG_DESIGN_REFINE=false)
    for a single cheaper pass."""
    source_map = build_source_map(nodes)
    if not source_map:
        return ArchDesign(insufficient_context=True), source_map

    if refine is None:
        from rag_app.config import get_settings
        refine = get_settings().design_refine

    context = format_context(source_map)
    design = llm.structured_predict(ArchDesign, _DESIGN_PROMPT, question=question, context=context)
    design = _clean(design, source_map)

    if refine and design.components and not design.insufficient_context:
        try:
            critique = str(llm.predict(_CRITIQUE_PROMPT, question=question, context=context,
                                       draft=_design_text(design)))
            improved = llm.structured_predict(
                ArchDesign, _REFINE_PROMPT, question=question, context=context,
                draft=_design_text(design), critique=critique,
            )
            # Only accept the revision if it didn't collapse the design.
            if improved.components and len(improved.components) >= max(2, len(design.components) - 1):
                design = _clean(improved, source_map)
        except Exception as exc:  # noqa: BLE001 - a failed refine must not lose the draft
            from rag_app.logging_utils import get_logger
            get_logger(__name__).warning("design refine pass failed, keeping draft: %s", exc)

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
    """Node/edge graph (with per-node detail + tier) for the blueprint and 3D renderers."""
    if not design.components:
        return {"nodes": [], "edges": []}
    g = architecture_graph(_design_arch(design))
    for node, comp in zip(g["nodes"], design.components):
        node["label"] = comp.name
        node["service"] = comp.service
        node["detail"] = comp.purpose
        node["tier"] = comp.tier
        if comp.sizing:
            node["sizing"] = comp.sizing
    return g
