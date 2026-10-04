"""Run the eval set through retrieval (+ rerank) and save scores for comparison over time.

eval file format (JSONL), one per line:
  {"question": "...", "expected_files": ["agenda_march.pdf"]}
"""
import json
from datetime import datetime
from pathlib import Path

from rag_app.config import Settings
from rag_app.evaluation.metrics import hit_at_k, mean, precision_at_k, reciprocal_rank
from rag_app.logging_utils import get_logger

log = get_logger(__name__)


def load_eval_set(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_eval(service, s: Settings, k: int = 5) -> dict:
    """`service` is a RAGService (pipelines/query_pipeline.py)."""
    rows = []
    for item in load_eval_set(s.eval_file):
        nodes = service.retrieve(item["question"])
        retrieved_files = [n.node.metadata.get("file_name", "") for n in nodes]
        expected = set(item["expected_files"])
        rows.append({
            "question": item["question"],
            "retrieved": retrieved_files[:k],
            f"hit@{k}": hit_at_k(retrieved_files, expected, k),
            f"precision@{k}": precision_at_k(retrieved_files, expected, k),
            "rr": reciprocal_rank(retrieved_files, expected),
        })

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "embed_model": s.embed_model, "chunk_size": s.chunk_size, "hybrid": s.hybrid_search,
            "rerank": s.rerank_enabled, "top_k": s.retrieve_top_k,
        },
        f"recall@{k}": mean([r[f"hit@{k}"] for r in rows]),
        f"precision@{k}": mean([r[f"precision@{k}"] for r in rows]),
        "mrr": mean([r["rr"] for r in rows]),
        "n_questions": len(rows),
        "rows": rows,
    }

    out_dir = Path(s.eval_results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_file.write_text(json.dumps(summary, indent=2))
    log.info("Saved eval results to %s", out_file)
    return summary
