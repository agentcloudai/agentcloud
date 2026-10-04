"""Design-mode eval: does "design ..." produce a real, grounded architecture diagram?

For each design question it checks the whole design payload — not just prose:

  - not_insufficient : the model actually produced a design
  - n_components      : how many architecture components it proposed
  - has_diagram       : the topology graph has >=2 nodes and >=1 edge (a real diagram)
  - has_data_flow     : it described a data/request flow
  - has_mermaid       : a mermaid diagram was emitted
  - component_coverage: fraction of expected services present among the components
  - grounded          : components cite at least one source

Needs an LLM (OPENAI_API_KEY) since it generates designs.
"""
import json
from datetime import datetime
from pathlib import Path

from rag_app.config import Settings
from rag_app.evaluation.runner import load_eval_set
from rag_app.logging_utils import get_logger

log = get_logger(__name__)


def run_design_eval(service, s: Settings, path: str | Path, k: int = 5) -> dict:
    """`service` is a RAGService. Returns a summary dict and writes it to disk."""
    rows = []
    for item in load_eval_set(path):
        q = item["question"]
        d = service.design(q)

        components = d.get("components", [])
        comp_services = [c.get("service", "") for c in components]
        comp_blob = " ".join(comp_services + [c.get("name", "") for c in components]).lower()
        graph = d.get("graph", {}) or {}
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        expected = item.get("expected_component_services", [])
        matched = [e for e in expected if e.lower() in comp_blob]
        grounded = any(c.get("source_ids") for c in components)

        rows.append({
            "question": q,
            "cloud": item.get("cloud", ""),
            "insufficient": bool(d.get("insufficient_context")),
            "n_components": len(components),
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "has_diagram": len(nodes) >= 2 and len(edges) >= 1,
            "has_data_flow": len(d.get("data_flow", [])) > 0,
            "has_mermaid": bool((d.get("architecture_mermaid") or "").strip()),
            "component_coverage": round(len(matched) / len(expected), 3) if expected else None,
            "missing_components": [e for e in expected if e.lower() not in comp_blob],
            "grounded": grounded,
            "component_services": comp_services,
        })

    def _rate(key):
        vals = [r[key] for r in rows if isinstance(r.get(key), bool)]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    covs = [r["component_coverage"] for r in rows if r["component_coverage"] is not None]
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "eval_file": str(path),
        "n_questions": len(rows),
        "diagram_rate": _rate("has_diagram"),
        "data_flow_rate": _rate("has_data_flow"),
        "mermaid_rate": _rate("has_mermaid"),
        "grounded_rate": _rate("grounded"),
        "insufficient_rate": _rate("insufficient"),
        "avg_component_coverage": round(sum(covs) / len(covs), 3) if covs else None,
        "avg_components": round(sum(r["n_components"] for r in rows) / len(rows), 2) if rows else 0,
        "rows": rows,
    }

    out_dir = Path(s.eval_results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"design_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_file.write_text(json.dumps(summary, indent=2))
    log.info("Saved design-eval results to %s", out_file)
    return summary
