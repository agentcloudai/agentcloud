"""Cross-encoder reranking: reads (question, chunk) together -> more accurate than embeddings."""
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore

from rag_app.config import Settings


def build_reranker(s: Settings) -> SentenceTransformerRerank:
    from rag_app.compute import resolve_device

    return SentenceTransformerRerank(
        model=s.rerank_model, top_n=s.rerank_pool, device=resolve_device(s)
    )


def rerank(reranker: SentenceTransformerRerank, question: str, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
    if not nodes:
        return []
    return reranker.postprocess_nodes(nodes, query_str=question)
