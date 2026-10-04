"""Embedding model factory. The SAME model must be used for indexing and querying."""
from llama_index.core.base.embeddings.base import BaseEmbedding

from rag_app.config import Settings


def build_embed_model(s: Settings) -> BaseEmbedding:
    if s.embed_provider == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding

        return OpenAIEmbedding(model=s.embed_model)
    if s.embed_provider == "huggingface":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        from rag_app.compute import resolve_device

        return HuggingFaceEmbedding(
            model_name=s.embed_model,
            device=resolve_device(s),          # run on GPU when available
            embed_batch_size=s.embed_batch_size,
        )
    raise ValueError(f"Unknown embed_provider: {s.embed_provider}")
