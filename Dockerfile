# AgentCloud AI — single application image.
#
# Users don't build this: they pull the published image and `docker compose up`
# (Airflow-style). The container starts the web UI on :8000. The vector index and
# any downloaded docs live on a mounted volume, so they survive restarts/upgrades.
#
# Build locally only if you're hacking on it:  docker build -t agentcloud .
FROM python:3.12-slim

# System deps: git is handy for the Azure ingest path; curl for healthchecks.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Models are baked into the image at /opt/models so the app works on first run
# with no download. This path is deliberately NOT a mounted volume (a volume would
# shadow the baked files), unlike the data dir below.
ENV HF_HOME=/opt/models \
    SENTENCE_TRANSFORMERS_HOME=/opt/models \
    RAG_DATA_DIR=/app/data \
    RAG_CHROMA_PATH=/app/data/chroma \
    PYTHONUNBUFFERED=1

# CPU-only torch FIRST: the default PyPI wheel drags in ~3GB of CUDA libraries this
# container can never use. Installing it up front means sentence-transformers sees
# torch as satisfied and won't pull the GPU build. (Run natively via pip for GPU.)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install the package (cached across code edits). Include the postgres extra so
# RAG_VECTOR_BACKEND=pgvector also works; the default embedded Chroma needs no DB.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[postgres]"

# Pre-download the embedding + reranker models into the baked model dir.
RUN python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("BAAI/bge-small-en-v1.5", "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    snapshot_download(repo)
    print("cached", repo)
PY

# Ship the AWS document lists + eval set with the image.
COPY aws_service_guides.csv aws_all_doc_pdfs.csv aws_services.csv ./
COPY eval ./eval
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

RUN mkdir -p /app/data

EXPOSE 8000

# A simple liveness check for compose.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=5 \
    CMD curl -fsS http://localhost:8000/api/services || exit 1

# The entrypoint auto-fetches a prebuilt index (RAG_INDEX_URL) on first serve,
# then runs the command. Default command serves the web UI; override for one-offs:
#   docker compose run --rm app rag-app ingest-gcp
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["rag-app", "serve", "--host", "0.0.0.0", "--port", "8000"]
