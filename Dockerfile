# Application image for rag-app. The database (pgvector) is a separate service;
# see docker-compose.yml. Keeping them separate means `docker compose up` brings
# the whole stack online with one command.
FROM python:3.12-slim

WORKDIR /app

# Install the package and its dependencies. Copying only what pip needs first
# keeps this layer cached across code edits.
COPY pyproject.toml README.md ./
COPY src ./src
# Include the optional pgvector backend so either RAG_VECTOR_BACKEND works in the
# container; the default (embedded Chroma) needs no database at all.
RUN pip install --no-cache-dir ".[postgres]"

# Ship the generated AWS document lists and the eval set with the image.
COPY aws_service_guides.csv aws_all_doc_pdfs.csv aws_services.csv ./
COPY eval ./eval

# Data (downloaded PDFs + manifest) lives on a mounted volume, not in the image.
ENV RAG_DATA_DIR=/app/data
RUN mkdir -p /app/data

# No ENTRYPOINT: compose keeps this container alive (sleep infinity) and you run
# `docker compose exec app rag-app ...`, or one-off `docker compose run --rm app rag-app ...`.
CMD ["rag-app", "--help"]
