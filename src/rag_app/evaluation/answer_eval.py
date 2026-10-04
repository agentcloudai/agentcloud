"""Answer-quality eval for architecture / how-to questions.

Unlike ``runner.py`` (which only scores retrieval), this generates an answer and
checks whether it actually contains the correct steps. For each question it reports:

  - retrieval_hit : did an expected source guide appear in the retrieved sources?
  - service_match : did the cited sources include an expected service?
  - step_coverage : fraction of the question's ``key_terms`` present in the answer
  - insufficient  : did the model report insufficient context?

``key_terms`` are the concrete things a correct answer must mention (service names,
API/console concepts, step keywords). Coverage is a cheap, deterministic proxy for
"the steps are correct"; set ``--judge`` later to add an LLM judge if you want.

Generation requires an LLM, so this needs OPENAI_API_KEY (or a vLLM endpoint).
"""
import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from rag_app.config import Settings
from rag_app.evaluation.runner import load_eval_set
from rag_app.logging_utils import get_logger

log = get_logger(__name__)


def _coverage(answer_text: str, key_terms: list[str]) -> tuple[float, list[str]]:
    low = answer_text.lower()
    missing = [t for t in key_terms if t.lower() not in low]
    matched = len(key_terms) - len(missing)
    return (matched / len(key_terms) if key_terms else 0.0), missing


class StepJudgment(BaseModel):
    """An LLM judge's verdict on whether an answer covers the required steps."""
    covered_steps: list[str] = Field(default_factory=list, description="expected steps the answer correctly addresses")
    missing_steps: list[str] = Field(default_factory=list, description="expected steps absent or wrong")
    correct: bool = Field(description="True if the answer is a substantially correct how-to")


_JUDGE_TMPL = (
    "You are grading a technical answer against a reference list of required steps.\n"
    "Question: {question}\n\n"
    "Required steps (the reference 'correct solution'):\n{steps}\n\n"
    "Answer to grade:\n{answer}\n\n"
    "For each required step, decide if the answer correctly addresses it (wording may differ). "
    "Return the covered and missing steps, and whether the answer is substantially correct overall."
)


def judge_answer(llm, question: str, expected_steps: list[str], answer_text: str) -> StepJudgment:
    if not answer_text.strip():
        return StepJudgment(covered_steps=[], missing_steps=expected_steps, correct=False)
    from llama_index.core.prompts import PromptTemplate
    prompt = PromptTemplate(_JUDGE_TMPL)
    steps = "\n".join(f"- {s}" for s in expected_steps)
    return llm.structured_predict(StepJudgment, prompt, question=question, steps=steps, answer=answer_text)


def run_answer_eval(service, s: Settings, path: str | Path | None = None, k: int = 5,
                    judge: bool = False) -> dict:
    """`service` is a RAGService. Returns a summary dict and writes it to disk.

    With ``judge=True``, an LLM grades step-correctness (robust to wording) instead
    of relying only on the keyword-coverage proxy.
    """
    eval_path = path or s.eval_file
    rows = []
    for item in load_eval_set(eval_path):
        q = item["question"]
        result = service.ask(q)

        answer_text = " ".join([result.answer.summary, *(c.text for c in result.answer.claims)])
        retrieved_files = {n.node.metadata.get("file_name", "") for n in result.sources.values()}
        retrieved_services = {n.node.metadata.get("service", "") for n in result.sources.values()}
        # file_name + source_url text, so Azure/GCP (no PDF names) can match on a
        # service-folder substring, e.g. "container-apps" or "/run/docs/".
        retrieved_blob = " ".join(
            f"{n.node.metadata.get('file_name', '')} {n.node.metadata.get('source_url', '')}"
            for n in result.sources.values()
        ).lower()

        expected_files = set(item.get("expected_files", []))
        expected_services = set(item.get("expected_services", []))
        expected_substrs = item.get("expected_source_substr", [])
        key_terms = item.get("key_terms", [])
        coverage, missing = _coverage(answer_text, key_terms)
        retrieval_hit = bool(expected_files & retrieved_files) or any(
            ss.lower() in retrieved_blob for ss in expected_substrs
        )

        row = {
            "question": q,
            "retrieval_hit": retrieval_hit,
            "service_match": bool(expected_services & retrieved_services) if expected_services else None,
            "step_coverage": round(coverage, 3),
            "missing_terms": missing,
            "insufficient": result.answer.insufficient_context,
            "answer": answer_text,
            "cited_services": sorted(s for s in retrieved_services if s),
        }
        if judge:
            steps = item.get("expected_steps", [])
            verdict = judge_answer(service.llm, q, steps, answer_text)
            row["judge_coverage"] = round(len(verdict.covered_steps) / len(steps), 3) if steps else 0.0
            row["judge_correct"] = verdict.correct
            row["judge_missing"] = verdict.missing_steps
        rows.append(row)

    def _mean(key):
        vals = [r[key] for r in rows if isinstance(r.get(key), bool)]
        return (sum(vals) / len(vals)) if vals else 0.0

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "eval_file": str(eval_path),
        "n_questions": len(rows),
        "retrieval_hit_rate": _mean("retrieval_hit"),
        "service_match_rate": _mean("service_match"),
        "avg_step_coverage": round(sum(r["step_coverage"] for r in rows) / len(rows), 3) if rows else 0.0,
        "rows": rows,
    }
    if judge:
        summary["avg_judge_coverage"] = round(sum(r["judge_coverage"] for r in rows) / len(rows), 3) if rows else 0.0
        summary["judge_correct_rate"] = _mean("judge_correct")

    out_dir = Path(s.eval_results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"answer_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_file.write_text(json.dumps(summary, indent=2))
    log.info("Saved answer-eval results to %s", out_file)
    return summary
