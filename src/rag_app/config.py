"""Central configuration. Values come from environment variables (prefix RAG_) or a .env file."""
from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    # OpenAI key: read from the bare OPENAI_API_KEY (no RAG_ prefix), env or .env.
    openai_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY", "RAG_OPENAI_API_KEY")
    )
    # HuggingFace token for model downloads (AI-art feature).
    hf_token: str | None = Field(
        default=None, validation_alias=AliasChoices("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "RAG_HF_TOKEN")
    )

    # --- PostgreSQL / pgvector ---
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_user: str = "rag"
    pg_password: str = "rag"
    pg_database: str = "rag"
    pg_table: str = "chunks"  # LlamaIndex stores it as "data_chunks"

    # --- Ingestion ---
    data_dir: str = "data/raw"
    ingest_workers: int = 0                     # parallel PDF-parsing processes (0 = auto / CPU count)
    index_batch_docs: int = 50                  # docs embedded+written per batch (resumable checkpoints)

    # --- Vector store ---
    # "chroma" = embedded, file-based, no server (default; works on a plain pip install).
    # "pgvector" = PostgreSQL + pgvector (for production / multi-tenant / SaaS).
    vector_backend: str = "chroma"
    chroma_path: str = "data/chroma"           # where the embedded index is stored
    vector_add_batch_size: int = 5000          # Chroma caps a single insert at 5461

    # --- Chunking ---
    chunk_size: int = 512      # tokens per chunk
    chunk_overlap: int = 64    # tokens shared between neighbouring chunks

    # --- Compute ---
    # "auto" uses CUDA (GPU) when available, else CPU. Force with "cuda" / "cpu".
    device: str = "auto"

    # --- Embedding ---
    embed_provider: str = "huggingface"        # "huggingface" | "openai"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384                       # must match the model's output size
    embed_batch_size: int = 64                 # larger batches use the GPU far better

    # --- Retrieval ---
    retrieve_top_k: int = 60   # candidates fed to the reranker (larger = better recall)
    query_expansion: bool = True   # LLM rewrites the query to name likely AWS services/terms
    hybrid_search: bool = True                 # vector + Postgres full-text search
    hnsw_ef_search: int = 40

    # --- Reranking ---
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_pool: int = 25        # chunks the cross-encoder scores per sub-query
    rerank_per_query: int = 4    # best chunks kept from EACH sub-query (ensures breadth)
    rerank_top_n: int = 12       # final chunks passed to the LLM as context (more = fuller answers)

    # --- Generation ---
    llm_provider: str = "openai"               # "openai" | "vllm"
    llm_model: str = "gpt-4o-mini"
    llm_api_base: str = "http://localhost:8000/v1"  # used only for vllm
    llm_temperature: float = 0.0

    # --- Evaluation ---
    eval_file: str = "eval/questions.jsonl"
    eval_results_dir: str = "eval/results"


@lru_cache
def get_settings() -> Settings:
    return Settings()
