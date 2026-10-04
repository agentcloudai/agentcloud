"""Central feedback collector — the server YOU host to gather opt-in feedback.

Clients with RAG_FEEDBACK_SUBMIT=true and RAG_FEEDBACK_ENDPOINT=<this>/collect
POST scrubbed training signals here; everything is appended to one JSONL you can
feed to `rag-app feedback --export`-style tooling / your RLHF pipeline.

Run:  rag-app collect-server --port 9000 --out collected.jsonl
Deploy this behind HTTPS (Render/Railway/Fly/a VM) and give clients its URL.
"""
import json
import threading
from datetime import datetime
from pathlib import Path

from rag_app.logging_utils import get_logger

log = get_logger(__name__)
_LOCK = threading.Lock()


def create_app(out_path: str):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="AgentCloud feedback collector")

    @app.get("/health")
    def health():
        n = sum(1 for _ in out.open(encoding="utf-8")) if out.exists() else 0
        return {"ok": True, "collected": n}

    @app.post("/collect")
    async def collect(req: Request):
        try:
            rec = await req.json()
        except Exception:
            return JSONResponse({"error": "invalid json"}, status_code=400)
        if not isinstance(rec, dict):
            return JSONResponse({"error": "expected object"}, status_code=400)
        rec["received_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        with _LOCK, out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return {"ok": True}

    return app


def serve(host: str = "0.0.0.0", port: int = 9000, out_path: str = "collected.jsonl") -> None:
    import uvicorn

    log.info("Feedback collector on http://%s:%d  → %s", host, port, out_path)
    uvicorn.run(create_app(out_path), host=host, port=port, log_level="info")
