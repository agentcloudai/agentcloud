"""FastAPI app serving the local UI and the query/diagram/art endpoints."""
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_app.config import get_settings
from rag_app.generation.diagram import architecture_graph, build_architecture, steps_mermaid, to_mermaid
from rag_app.logging_utils import get_logger

log = get_logger(__name__)
_STATIC = Path(__file__).parent / "static"


@lru_cache
def _service():
    """Build the RAG service once (loads embed + rerank models)."""
    from rag_app.pipelines.query_pipeline import RAGService

    return RAGService()


class AskRequest(BaseModel):
    question: str
    service: str | None = None


class ArtRequest(BaseModel):
    question: str
    services: list[str] = []


def create_app() -> FastAPI:
    app = FastAPI(title="rag-app", description="Ask AWS how-to questions with cited, visualized answers")

    @app.get("/")
    def index():
        return FileResponse(_STATIC / "index.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

    @app.get("/api/services")
    def services():
        """Distinct services (for the filter dropdown), from the download manifest."""
        from rag_app.ingest.manifest import load_manifest

        m = load_manifest(get_settings().data_dir)
        names = sorted({row["service"] for row in m.values() if row.get("service")})
        return {"services": names}

    @app.post("/api/ask")
    def ask(req: AskRequest):
        if not req.question.strip():
            return JSONResponse({"error": "empty question"}, status_code=400)
        from rag_app.generation.design import is_design_question

        filters = {"service": req.service} if req.service else None
        svc = _service()

        # Architecture/design questions get the detailed design payload.
        if is_design_question(req.question):
            return svc.design(req.question, filters)

        result = svc.ask(req.question, filters)
        payload = result.to_payload()
        payload["mode"] = "answer"
        if not payload["insufficient_context"]:
            arch = build_architecture(svc.llm, result.answer, req.question)
            payload["architecture_mermaid"] = to_mermaid(arch)
            payload["steps_mermaid"] = steps_mermaid(result.answer)
            payload["graph"] = architecture_graph(arch)
        else:
            payload["architecture_mermaid"] = payload["steps_mermaid"] = ""
            payload["graph"] = {"nodes": [], "edges": []}
        return payload

    @app.post("/api/art")
    def art(req: ArtRequest):
        """Optional diffusion 'artistic render'. Lazy-loads the model on first call."""
        try:
            from rag_app.generation.image_gen import generate_architecture_image

            uri = generate_architecture_image(req.question, req.services)
            return {"image": uri}
        except ImportError:
            return JSONResponse(
                {"error": "AI-art needs: pip install '.[image]'"}, status_code=501
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("art generation failed: %s", exc)
            return JSONResponse({"error": f"generation failed: {exc}"}, status_code=500)

    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")
    return app


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    app = create_app()
    # Warm up embed + rerank models at startup so the FIRST user query isn't a
    # cold start (otherwise the reranker loads on first request, adding ~10s).
    log.info("Warming up models…")
    try:
        _service()
        log.info("Models ready.")
    except Exception as exc:  # noqa: BLE001
        log.warning("Warmup skipped: %s", exc)
    log.info("AgentCloud UI at http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
