# AgentCloud AI — the whole product in one image: app, models and the prebuilt
# multi-cloud vector index. Users pull it and `docker compose up`; nothing is
# downloaded at run time, so it works on an air-gapped machine.
#
# Slim variant (index fetched at run time from RAG_INDEX_URL):
#   docker build --build-arg WITH_INDEX=false -t agentcloud:slim .
FROM python:3.12-slim

# git is used by the Azure ingest path; curl by the healthcheck and index fetch.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV RAG_DATA_DIR=/app/data \
    RAG_CHROMA_PATH=/opt/index \
    PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# Heavy, rarely-changing layers first, so editing the app does not invalidate
# them. The index in particular is ~9GB and must stay cached across code edits.
# ---------------------------------------------------------------------------

# The prebuilt index (AWS + Azure + GCP). It lives at /opt/index, deliberately
# NOT under /app/data, which is a mount point a volume would shadow — that also
# means it is stored once, instead of being copied into the volume on boot.
ARG WITH_INDEX=true
ARG INDEX_URL=https://huggingface.co/datasets/rdprojects/agentcloud/resolve/main/agentcloud-index.tar.gz
ARG INDEX_SHA256=fce9bcfa63d5318f00491e63ffeb619016e7aa01debf4c29aa3de001a7d06222
RUN if [ "$WITH_INDEX" = "true" ]; then \
      set -eux; \
      curl -fL --retry 5 --retry-delay 5 -o /tmp/index.tar.gz "$INDEX_URL"; \
      echo "$INDEX_SHA256  /tmp/index.tar.gz" | sha256sum -c -; \
      mkdir -p /opt/index; \
      tar -xzf /tmp/index.tar.gz -C /opt/index --strip-components=1; \
      rm -f /tmp/index.tar.gz; \
      test -f /opt/index/chroma.sqlite3; \
      touch /opt/index/.index_complete; \
      du -sh /opt/index; \
    fi

# CPU-only torch: the default PyPI wheel drags in ~3GB of CUDA libraries this
# container can never use. Installing it first also stops sentence-transformers
# pulling the GPU build. (Run natively via pip for GPU.)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir sentence-transformers

# Save the embedding + reranker models as plain directories and point the app at
# those paths. Loading by path never consults the HuggingFace cache or network,
# which is what makes offline start-up reliable.
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer("BAAI/bge-small-en-v1.5").save("/opt/models/embed")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2").save("/opt/models/rerank")
print("models saved")
PY
ENV RAG_EMBED_MODEL=/opt/models/embed \
    RAG_RERANK_MODEL=/opt/models/rerank \
    HF_HOME=/opt/models \
    HF_HUB_OFFLINE=1

# ---------------------------------------------------------------------------
# Application code last: edits here rebuild only these cheap layers.
# ---------------------------------------------------------------------------
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[postgres]"

COPY aws_service_guides.csv aws_all_doc_pdfs.csv aws_services.csv ./
COPY eval ./eval
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh && mkdir -p /app/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=5 \
    CMD curl -fsS http://localhost:8000/api/services || exit 1

# The entrypoint only fetches an index for slim builds; with the index baked in
# it is a no-op. Override the command for one-off work, e.g.
#   docker compose run --rm app rag-app ingest-gcp
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["rag-app", "serve", "--host", "0.0.0.0", "--port", "8000"]
