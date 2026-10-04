"""INDEX flow:  load > dedupe > chunk > embed > store.

Indexing is RESUMABLE: documents are processed in batches, each batch is written to
the vector store immediately, and its content hashes are recorded in a progress file
next to the store. If indexing is interrupted (e.g. the PC is shut down), re-running
`rag-app index` skips the documents already stored and continues from where it left off.
"""
from pathlib import Path

from llama_index.core.ingestion import IngestionPipeline

from rag_app.chunking.splitter import build_splitter
from rag_app.config import Settings, get_settings
from rag_app.embedding.embedder import build_embed_model
from rag_app.ingest.loaders import deduplicate, load_directory, load_urls
from rag_app.logging_utils import get_logger
from rag_app.storage.vector_store import add_nodes_batched, build_vector_store

log = get_logger(__name__)

_PROGRESS_NAME = "_indexed_hashes.txt"


def _progress_path(s: Settings) -> Path:
    base = Path(s.chroma_path) if s.vector_backend.lower() == "chroma" else Path(s.data_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base / _PROGRESS_NAME


def _load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _append_done(path: Path, hashes: list[str]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for h in hashes:
            fh.write(h + "\n")


def run_indexing(s: Settings | None = None, urls: list[str] | None = None) -> int:
    s = s or get_settings()

    docs = load_urls(urls) if urls else load_directory(s.data_dir, num_workers=s.ingest_workers)
    docs = deduplicate(docs)

    vector_store = build_vector_store(s)
    progress = _progress_path(s)
    done = _load_done(progress)

    todo = [d for d in docs if d.metadata.get("content_hash") not in done]
    skipped = len(docs) - len(todo)
    if skipped:
        log.info("Resuming: %d documents already indexed, %d to go", skipped, len(todo))
    if not todo:
        log.info("Nothing new to index (%d documents already stored).", len(docs))
        return 0

    splitter = build_splitter(s.chunk_size, s.chunk_overlap)
    embed = build_embed_model(s)

    batch = max(1, s.index_batch_docs)
    total_nodes = 0
    for start in range(0, len(todo), batch):
        group = todo[start:start + batch]
        pipeline = IngestionPipeline(transformations=[splitter, embed])
        nodes = pipeline.run(documents=group, show_progress=True)
        add_nodes_batched(vector_store, nodes, batch_size=s.vector_add_batch_size)
        _append_done(progress, [d.metadata["content_hash"] for d in group])
        total_nodes += len(nodes)
        log.info("Indexed batch: %d docs, %d chunks (%d/%d docs done)",
                 len(group), len(nodes), min(start + batch, len(todo)), len(todo))

    log.info("Indexed %d chunks from %d documents into '%s'", total_nodes, len(todo), s.pg_table)
    return total_nodes
