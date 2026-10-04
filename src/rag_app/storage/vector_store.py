"""Vector store factory. Two backends, chosen by ``settings.vector_backend``:

  - "chroma"  : embedded, file-based (Chroma). No server to run, so the app works
                on a plain install / git clone. This is the default.
  - "pgvector": PostgreSQL + pgvector. Better for large/multi-tenant/SaaS setups;
                supports hybrid (vector + full-text) search. Needs a running
                Postgres with the pgvector extension (see docker-compose.yml).

Both store text, metadata and embedding together and are created on first use.
"""
from pathlib import Path

from llama_index.core.vector_stores.types import BasePydanticVectorStore

from rag_app.config import Settings
from rag_app.logging_utils import get_logger

log = get_logger(__name__)


def add_nodes_batched(vector_store, nodes, batch_size: int = 5000) -> int:
    """Insert nodes in batches. Chroma rejects a single ``add`` larger than ~5461,
    so large corpora must be chunked into sub-batches. Returns the count added.
    """
    total = len(nodes)
    for i in range(0, total, batch_size):
        batch = nodes[i:i + batch_size]
        vector_store.add(batch)
        log.info("Stored %d/%d chunks", min(i + batch_size, total), total)
    return total


def build_vector_store(s: Settings) -> BasePydanticVectorStore:
    backend = s.vector_backend.lower()
    if backend == "chroma":
        return _build_chroma(s)
    if backend == "pgvector":
        return _build_pgvector(s)
    raise ValueError(f"Unknown vector_backend: {s.vector_backend!r} (use 'chroma' or 'pgvector')")


def _build_chroma(s: Settings) -> BasePydanticVectorStore:
    """Embedded Chroma. Needs: pip install '.[dev]' (chroma is a core dependency)."""
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore

    path = Path(s.chroma_path)
    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_or_create_collection(name=s.pg_table)
    log.info("Using embedded Chroma vector store at %s (collection '%s')", path, s.pg_table)
    return ChromaVectorStore(chroma_collection=collection)


def _build_pgvector(s: Settings) -> BasePydanticVectorStore:
    """PostgreSQL + pgvector. Needs: pip install '.[postgres]'."""
    from llama_index.vector_stores.postgres import PGVectorStore

    log.info("Using pgvector store on %s:%s/%s", s.pg_host, s.pg_port, s.pg_database)
    return PGVectorStore.from_params(
        host=s.pg_host,
        port=str(s.pg_port),
        user=s.pg_user,
        password=s.pg_password,
        database=s.pg_database,
        table_name=s.pg_table,
        embed_dim=s.embed_dim,
        hybrid_search=s.hybrid_search,
        text_search_config="english",
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": s.hnsw_ef_search,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )
