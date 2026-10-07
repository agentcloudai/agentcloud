"""FastAPI app serving the local UI and the query/diagram/export endpoints."""
import os
import re
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


def _key_file() -> Path:
    """Where a key entered in the UI is kept — in the writable data dir, which is
    the mounted volume, so it survives restarts and upgrades. (Deliberately not
    beside the index: that may be a read-only path baked into the image.)"""
    return Path(get_settings().data_dir) / ".agentcloud_key"


def load_saved_key() -> None:
    """Make a previously saved key active, unless the environment already has one."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    f = _key_file()
    try:
        if f.exists():
            key = f.read_text(encoding="utf-8").strip()
            if key:
                os.environ["OPENAI_API_KEY"] = key
                log.info("Loaded saved LLM key from %s", f)
    except OSError as exc:
        log.warning("Could not read saved key: %s", exc)


def llm_ready() -> bool:
    s = get_settings()
    return bool(s.openai_api_key or os.environ.get("OPENAI_API_KEY") or s.llm_provider.lower() == "vllm")


class AskRequest(BaseModel):
    question: str
    service: str | None = None
    cloud: str | None = None       # aws | azure | gcp | all (scopes retrieval to one provider)


class ConfigRequest(BaseModel):
    openai_api_key: str | None = None


class TerraformRequest(BaseModel):
    design: dict
    cloud: str | None = None


class ArtRequest(BaseModel):
    question: str
    services: list[str] = []


class FeedbackRequest(BaseModel):
    interaction_id: str
    rating: int | None = None      # 1 = helpful, -1 = not helpful
    comment: str = ""
    correction: str = ""           # a human-written "ideal" answer


def create_app() -> FastAPI:
    app = FastAPI(title="rag-app", description="Multi-cloud solutions architect: cited how-tos, architecture designs and exports")

    @app.get("/")
    def index():
        return FileResponse(_STATIC / "index.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

    @app.get("/api/config")
    def get_config():
        """First-run state for the UI: is an LLM key configured, and is the index loaded?"""
        indexed = 0
        try:
            indexed = _service().vector_store._collection.count()  # noqa: SLF001
        except Exception:  # noqa: BLE001 - an empty/absent store is a valid state
            pass
        return {"llm_ready": llm_ready(), "indexed_chunks": indexed,
                "index_url_configured": bool(os.environ.get("RAG_INDEX_URL"))}

    @app.post("/api/config")
    def set_config(req: ConfigRequest):
        """Accept an LLM key from the UI so users never have to edit .env."""
        key = (req.openai_api_key or "").strip()
        if not key.startswith("sk-") or len(key) < 20:
            return JSONResponse({"error": "That doesn't look like an OpenAI key (expected sk-…)."},
                                status_code=400)
        os.environ["OPENAI_API_KEY"] = key
        try:
            f = _key_file()
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(key, encoding="utf-8")
            try:
                f.chmod(0o600)
            except OSError:
                pass            # best effort; not supported on every filesystem
        except OSError as exc:
            log.warning("Could not persist key (it is active for this run): %s", exc)
        get_settings.cache_clear()  # rebuild Settings so the key is picked up
        _service.cache_clear()      # and rebuild the LLM
        return {"ok": True, "llm_ready": llm_ready()}

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

        from rag_app.feedback import log_interaction

        filters = {"service": req.service} if req.service else None
        svc = _service()

        # Architecture/design questions get the detailed design payload.
        if is_design_question(req.question):
            payload = svc.design(req.question, filters, cloud=req.cloud)
            payload["interaction_id"] = log_interaction(req.question, "design", payload)
            return payload

        result = svc.ask(req.question, filters, cloud=req.cloud)
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
        payload["interaction_id"] = log_interaction(req.question, "answer", payload)
        return payload

    @app.post("/api/export/terraform")
    def export_terraform(req: TerraformRequest):
        """Turn a design payload into a Terraform skeleton (uses the user's own LLM key)."""
        comps = req.design.get("components") or []
        if not comps:
            return JSONResponse({"error": "no components to export"}, status_code=400)
        from llama_index.core import PromptTemplate

        lines = [f"- [{c.get('tier','')}] {c.get('name','')} using {c.get('service','')}"
                 f"{(' (' + c['sizing'] + ')') if c.get('sizing') else ''}" for c in comps]
        conns = [f"- {c.get('source','')} -> {c.get('target','')} ({c.get('label','')})"
                 for c in (req.design.get("connections") or [])]
        prompt = PromptTemplate(
            "Write Terraform for the architecture below. Output ONLY HCL, no prose and no code fences.\n"
            "Requirements:\n"
            "- Use the correct provider for the services named ({cloud}).\n"
            "- Include a terraform block with required_providers, the provider block, and variables "
            "for region/project/names with sensible defaults.\n"
            "- One resource per component, wired together via references where the connections imply it.\n"
            "- Add a short comment above each resource saying which component it implements.\n"
            "- Where a resource needs details the design does not specify, add a TODO comment rather "
            "than inventing values.\n\n"
            "Architecture: {overview}\n\nComponents:\n{components}\n\nConnections:\n{connections}\n"
        )
        try:
            hcl = str(_service().llm.predict(
                prompt, cloud=req.cloud or "infer from the service names",
                overview=req.design.get("overview", ""),
                components="\n".join(lines), connections="\n".join(conns) or "(none given)",
            ))
        except Exception as exc:  # noqa: BLE001
            log.warning("terraform export failed: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500)
        hcl = re.sub(r"^\s*```[a-zA-Z]*\n|\n```\s*$", "", hcl).strip()
        return {"terraform": hcl}

    @app.post("/api/feedback")
    def feedback(req: FeedbackRequest):
        from rag_app.feedback import add_feedback

        add_feedback(req.model_dump())
        return {"ok": True}

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

    load_saved_key()   # a key entered in the UI on a previous run stays active
    app = create_app()
    # Warm the LOCAL models so the first query isn't a cold start (the reranker
    # alone costs ~10s). Deliberately not the whole service: that builds the LLM
    # client too, which has nothing to preload and just stalls when the machine
    # is offline or has no key yet.
    log.info("Warming up models…")
    try:
        from rag_app.config import get_settings as _gs
        from rag_app.embedding.embedder import build_embed_model
        from rag_app.reranking.reranker import build_reranker

        _s = _gs()
        build_embed_model(_s)
        if _s.rerank_enabled:
            build_reranker(_s)
        log.info("Models ready.")
    except Exception as exc:  # noqa: BLE001
        log.warning("Warmup skipped: %s", exc)
    log.info("AgentCloud UI at http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
