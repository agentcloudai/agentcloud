"""Local interaction + feedback capture for RLHF / fine-tuning datasets.

Everything is append-only JSONL under ``data/feedback/`` so it's crash-safe and easy
to ship to a training pipeline:
  - interactions.jsonl : every question + the response the agent gave
  - feedback.jsonl     : human signals (👍/👎, comment, a corrected/ideal answer)

``export_rlhf`` joins them into SFT and preference-pair datasets.
"""
import json
import threading
import uuid
from datetime import datetime
from pathlib import Path

from rag_app.config import get_settings
from rag_app.logging_utils import get_logger

log = get_logger(__name__)
_LOCK = threading.Lock()


def _dir() -> Path:
    d = Path(get_settings().data_dir).resolve().parent / "feedback"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _append(name: str, record: dict) -> None:
    with _LOCK, (_dir() / name).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _answer_text(mode: str, payload: dict) -> str:
    """Flatten a response payload into plain text (the 'response' for training)."""
    if mode == "design":
        parts = [payload.get("overview", "")]
        for c in payload.get("components", []):
            parts.append(f"- {c.get('name')} ({c.get('service','')}): {c.get('purpose','')}")
        for i, s in enumerate(payload.get("data_flow", []), 1):
            parts.append(f"{i}. {s.get('step','')}")
        return "\n".join(p for p in parts if p).strip()
    parts = [payload.get("answer", "")]
    for i, s in enumerate(payload.get("steps", []), 1):
        parts.append(f"{i}. {s.get('text','')}")
    return "\n".join(p for p in parts if p).strip()


def log_interaction(question: str, mode: str, payload: dict) -> str:
    """Record one agent turn. Returns an interaction id to attach feedback to."""
    s = get_settings()
    rid = uuid.uuid4().hex
    rec = {
        "id": rid,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "question": question,
        "response_text": _answer_text(mode, payload),
        "payload": payload,
        "meta": {
            "llm_model": s.llm_model, "embed_model": s.embed_model,
            "prompt_version": payload.get("prompt_version"), "vector_backend": s.vector_backend,
        },
    }
    try:
        _append("interactions.jsonl", rec)
    except Exception as exc:  # noqa: BLE001 - never break a response over logging
        log.warning("log_interaction failed: %s", exc)
    return rid


def add_feedback(fb: dict) -> None:
    fb = dict(fb)
    fb["ts"] = datetime.now().isoformat(timespec="seconds")
    try:
        _append("feedback.jsonl", fb)
    except Exception as exc:  # noqa: BLE001
        log.warning("add_feedback failed: %s", exc)


def _load(name: str) -> list[dict]:
    p = _dir() / name
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def stats() -> dict:
    inter, fbs = _load("interactions.jsonl"), _load("feedback.jsonl")
    return {
        "interactions": len(inter),
        "feedback": len(fbs),
        "thumbs_up": sum(1 for f in fbs if f.get("rating") == 1),
        "thumbs_down": sum(1 for f in fbs if f.get("rating") == -1),
        "corrections": sum(1 for f in fbs if (f.get("correction") or "").strip()),
    }


def export_rlhf() -> dict:
    """Join interactions + feedback into SFT and preference datasets."""
    d = _dir()
    inter = {r["id"]: r for r in _load("interactions.jsonl")}
    sft, prefs = [], []
    for fb in _load("feedback.jsonl"):
        it = inter.get(fb.get("interaction_id"))
        if not it:
            continue
        q, model_ans = it["question"], it.get("response_text", "")
        correction = (fb.get("correction") or "").strip()
        rating = fb.get("rating")
        if correction:  # a human-written better answer: strongest signal
            prefs.append({"prompt": q, "chosen": correction, "rejected": model_ans})
            sft.append({"prompt": q, "response": correction})
        elif rating == 1:  # approved the model's answer
            sft.append({"prompt": q, "response": model_ans})
        # rating == -1 with no correction -> kept only as a rejected example (below)
        if rating == -1 and not correction:
            prefs.append({"prompt": q, "chosen": "", "rejected": model_ans, "note": "downvoted, no correction"})
    (d / "sft.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in sft), encoding="utf-8")
    (d / "preferences.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in prefs), encoding="utf-8")
    return {"sft_examples": len(sft), "preference_pairs": len(prefs), "dir": str(d)}
