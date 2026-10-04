"""Stress-test the system across a spectrum of question difficulty.

No ground truth — just surfaces behaviour: does it answer, how many steps, which
services it drew from, latency, and whether it bailed with 'insufficient context'.

    python eval/smoke.py [eval/spectrum_questions.jsonl]
"""
import json
import sys
import time

from rag_app.pipelines.query_pipeline import RAGService


def main(path: str) -> None:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    svc = RAGService()
    insufficient = 0
    print(f"{'LEVEL':<16} {'INSUF':<6} {'STEPS':<6} {'SRVCS':<6} {'ms':<6} QUESTION")
    results = []
    for item in rows:
        q = item["question"]
        t = time.time()
        p = svc.ask(q).to_payload()
        ms = int((time.time() - t) * 1000)
        services = sorted({s["service"] for s in p["sources"] if s.get("service")})
        insuf = p["insufficient_context"]
        insufficient += insuf
        print(f"{item['level']:<16} {str(insuf):<6} {len(p['steps']):<6} {len(services):<6} {ms:<6} {q[:52]}")
        results.append({**item, "insufficient": insuf, "n_steps": len(p["steps"]),
                        "services": services, "answer": p["answer"]})
    n = len(rows)
    print(f"\nanswered: {n - insufficient}/{n}   insufficient: {insufficient}/{n}")
    json.dump(results, open("eval/results/smoke.json", "w"), indent=2)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval/spectrum_questions.jsonl")
