"""Ingest Azure documentation (Markdown from MicrosoftDocs/azure-docs) into the
vector store, tagged per Azure service — the Azure counterpart of the AWS docs.

Point it at a (sparse) clone of MicrosoftDocs/azure-docs; it reads articles/<service>/**.md,
tags each chunk with the friendly service name + the real learn.microsoft.com URL, and
adds them to the existing Chroma collection (resumable, alongside the AWS chunks).
"""
import re
from pathlib import Path

from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline

from rag_app.chunking.splitter import build_splitter
from rag_app.config import Settings, get_settings
from rag_app.embedding.embedder import build_embed_model
from rag_app.ingest.loaders import _finalize, deduplicate
from rag_app.logging_utils import get_logger
from rag_app.pipelines.index_pipeline import _append_done, _load_done, _progress_path
from rag_app.storage.vector_store import add_nodes_batched, build_vector_store

log = get_logger(__name__)

AZURE_SERVICE_NAMES = {
    "app-service": "Azure App Service", "azure-functions": "Azure Functions",
    "container-apps": "Azure Container Apps", "static-web-apps": "Azure Static Web Apps",
    "storage": "Azure Storage", "virtual-network": "Azure Virtual Network",
    "load-balancer": "Azure Load Balancer", "application-gateway": "Azure Application Gateway",
    "frontdoor": "Azure Front Door", "api-management": "Azure API Management",
    "event-hubs": "Azure Event Hubs", "service-bus-messaging": "Azure Service Bus",
    "event-grid": "Azure Event Grid", "logic-apps": "Azure Logic Apps",
    "data-factory": "Azure Data Factory", "synapse-analytics": "Azure Synapse Analytics",
    "stream-analytics": "Azure Stream Analytics", "iot-hub": "Azure IoT Hub",
    "backup": "Azure Backup", "site-recovery": "Azure Site Recovery",
}

_FRONTMATTER = re.compile(r"^---\n.*?\n---\n", re.S)
_MIN_CHARS = 250  # skip stubs / include fragments


def _friendly(folder: str) -> str:
    return AZURE_SERVICE_NAMES.get(folder, "Azure " + folder.replace("-", " ").title())


def load_azure_docs(src_dir: str | Path) -> list[Document]:
    root = Path(src_dir) / "articles"
    if not root.exists():
        raise FileNotFoundError(f"No articles/ under {src_dir} — clone MicrosoftDocs/azure-docs first")
    docs: list[Document] = []
    for md in root.rglob("*.md"):
        rel = md.relative_to(root)
        folder = rel.parts[0]
        text = _FRONTMATTER.sub("", md.read_text(encoding="utf-8", errors="ignore")).strip()
        if len(text) < _MIN_CHARS or rel.name.lower().startswith("includes"):
            continue
        slug = "/".join(rel.parts[1:])[:-3]  # drop ".md"
        url = f"https://learn.microsoft.com/en-us/azure/{folder}/{slug}"
        d = Document(text=text, id_=f"azure/{rel.as_posix()}")
        d.metadata["file_name"] = f"azure/{rel.as_posix()}"
        d.metadata["service"] = _friendly(folder)
        d.metadata["guide"] = "Azure documentation"
        d.metadata["source_url"] = url
        docs.append(_finalize(d))
    log.info("Loaded %d Azure markdown docs from %s", len(docs), root)
    return docs


def run_azure_index(src_dir: str | Path, s: Settings | None = None) -> int:
    s = s or get_settings()
    docs = deduplicate(load_azure_docs(src_dir))
    vector_store = build_vector_store(s)
    progress = _progress_path(s)
    done = _load_done(progress)
    todo = [d for d in docs if d.metadata.get("content_hash") not in done]
    skipped = len(docs) - len(todo)
    if skipped:
        log.info("Resuming Azure: %d already indexed, %d to go", skipped, len(todo))
    if not todo:
        log.info("Nothing new to index (%d Azure docs already stored).", len(docs))
        return 0

    splitter = build_splitter(s.chunk_size, s.chunk_overlap)
    embed = build_embed_model(s)
    batch = max(1, s.index_batch_docs)
    total = 0
    for start in range(0, len(todo), batch):
        group = todo[start:start + batch]
        nodes = IngestionPipeline(transformations=[splitter, embed]).run(documents=group)
        add_nodes_batched(vector_store, nodes, batch_size=s.vector_add_batch_size)
        _append_done(progress, [d.metadata["content_hash"] for d in group])
        total += len(nodes)
        log.info("Azure indexed: %d docs, %d chunks (%d/%d docs done)",
                 len(group), len(nodes), min(start + batch, len(todo)), len(todo))
    log.info("Indexed %d Azure chunks from %d docs", total, len(todo))
    return total
