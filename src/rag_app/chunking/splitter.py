"""Chunking strategies. Swap or add splitters here without touching other modules."""
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode


def build_splitter(chunk_size: int, chunk_overlap: int) -> SentenceSplitter:
    """Token-sized chunks that try not to cut sentences in half."""
    return SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def chunk_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[BaseNode]:
    """Standalone helper to inspect chunks before indexing (useful for debugging)."""
    return build_splitter(chunk_size, chunk_overlap).get_nodes_from_documents(docs)
