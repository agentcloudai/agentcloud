"""Builds a retriever over the existing pgvector table (no re-indexing)."""
from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.core.vector_stores.types import BasePydanticVectorStore

from rag_app.config import Settings


def build_retriever(
    vector_store: BasePydanticVectorStore,
    embed_model: BaseEmbedding,
    s: Settings,
    filters: dict[str, str] | None = None,
) -> BaseRetriever:
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    kwargs: dict = {"similarity_top_k": s.retrieve_top_k}
    # Hybrid (vector + full-text) search is a pgvector feature; the embedded
    # Chroma backend does vector-only retrieval (the reranker still refines it).
    if s.hybrid_search and s.vector_backend.lower() == "pgvector":
        kwargs.update(vector_store_query_mode="hybrid", sparse_top_k=s.retrieve_top_k)
    if filters:
        # e.g. {"file_name": "agenda_march.pdf"} -> SQL WHERE on metadata
        kwargs["filters"] = MetadataFilters(
            filters=[ExactMatchFilter(key=k, value=v) for k, v in filters.items()]
        )
    return index.as_retriever(**kwargs)
