"""Turn a how-to answer into a structured architecture diagram (Mermaid).

The LLM extracts the components and their connections from the answer; we render
that deterministically as Mermaid so the diagram always has valid syntax, real
labels and AWS service symbols (unlike an image model, which garbles text).
"""
from llama_index.core import PromptTemplate
from pydantic import BaseModel, Field

from rag_app.generation.schemas import Answer

# A few recognizable symbols for common AWS building blocks (extend freely).
SERVICE_SYMBOLS = {
    "s3": "🪣", "simple storage": "🪣", "bucket": "🪣",
    "ec2": "🖥️", "compute": "🖥️", "lambda": "λ", "fargate": "📦", "ecs": "📦",
    "container": "📦", "api gateway": "🚪", "cloudfront": "🌐", "cdn": "🌐",
    "route 53": "🧭", "dns": "🧭", "vpc": "🔲", "subnet": "🔳",
    "load balancer": "⚖️", "elb": "⚖️", "alb": "⚖️", "auto scaling": "📈",
    "dynamodb": "🗂️", "rds": "🛢️", "aurora": "🛢️", "database": "🛢️",
    "iam": "🔑", "secrets manager": "🔐", "kms": "🔐", "cloudwatch": "📊",
    "sqs": "📨", "sns": "📣", "eventbridge": "🔔", "user": "👤", "client": "👤",
    "internet gateway": "🌍", "nat": "🔀", "step functions": "🧩", "bedrock": "🧠",
}

_DEFAULT_SYMBOL = "⬛"


def symbol_for(text: str | None) -> str:
    if not text:
        return _DEFAULT_SYMBOL
    low = text.lower()
    for key, sym in SERVICE_SYMBOLS.items():
        if key in low:
            return sym
    return _DEFAULT_SYMBOL


class DiagramNode(BaseModel):
    id: str = Field(description="Short id, e.g. 'n1' (letters/digits only).")
    label: str = Field(description="Short human label, e.g. 'S3 bucket'.")
    service: str = Field(default="", description="AWS service this node represents, if any.")


class DiagramEdge(BaseModel):
    source: str = Field(description="id of the source node")
    target: str = Field(description="id of the target node")
    label: str = Field(default="", description="short edge label, e.g. 'HTTPS'")


class Architecture(BaseModel):
    nodes: list[DiagramNode] = Field(default_factory=list)
    edges: list[DiagramEdge] = Field(default_factory=list)


_DIAGRAM_PROMPT = PromptTemplate(
    "From the how-to answer below, extract the ARCHITECTURE as nodes and edges.\n"
    "- nodes: each component/service involved (user, service, resource).\n"
    "- edges: how data/requests flow between them (with a short label if useful).\n"
    "- Keep it to the components actually mentioned. Ids must be simple like n1, n2.\n\n"
    "Question: {question}\n\n"
    "Answer summary: {summary}\n"
    "Steps:\n{steps}\n"
)


def build_architecture(llm, answer: Answer, question: str) -> Architecture:
    steps = "\n".join(f"- {c.text}" for c in answer.claims)
    try:
        return llm.structured_predict(
            Architecture, _DIAGRAM_PROMPT,
            question=question, summary=answer.summary, steps=steps,
        )
    except Exception:  # noqa: BLE001 - diagram is best-effort; never break the answer
        return Architecture()


def _san(label: str) -> str:
    """Escape characters that break Mermaid node labels."""
    return label.replace('"', "'").replace("[", "(").replace("]", ")")


def to_mermaid(arch: Architecture) -> str:
    """Render the architecture as a left-to-right Mermaid flowchart."""
    if not arch.nodes:
        return ""
    lines = ["flowchart LR"]
    for n in arch.nodes:
        sym = symbol_for(n.service or n.label)
        lines.append(f'    {n.id}["{sym} {_san(n.label)}"]')
    for e in arch.edges:
        if e.label:
            lines.append(f'    {e.source} -->|{_san(e.label)}| {e.target}')
        else:
            lines.append(f"    {e.source} --> {e.target}")
    return "\n".join(lines)


def architecture_graph(arch: Architecture) -> dict:
    """Plain node/edge structure for the interactive 3D renderer."""
    return {
        "nodes": [{"id": n.id, "label": n.label, "service": n.service,
                   "symbol": symbol_for(n.service or n.label)} for n in arch.nodes],
        "edges": [{"source": e.source, "target": e.target, "label": e.label} for e in arch.edges],
    }


def steps_mermaid(answer: Answer) -> str:
    """A simple top-down 'setup steps' flow, one node per step."""
    if not answer.claims:
        return ""
    lines = ["flowchart TD"]
    prev = None
    for i, c in enumerate(answer.claims, 1):
        text = _san(c.text[:70] + ("…" if len(c.text) > 70 else ""))
        nid = f"s{i}"
        lines.append(f'    {nid}["{i}. {text}"]')
        if prev:
            lines.append(f"    {prev} --> {nid}")
        prev = nid
    return "\n".join(lines)
