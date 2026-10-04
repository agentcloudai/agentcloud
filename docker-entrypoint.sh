#!/bin/sh
# On first boot, if the data volume has no index yet and RAG_INDEX_URL is set,
# download the prebuilt index into it. No-op when an index already exists, when no
# URL is configured, or for one-off commands (ingest/download/etc.).
set -e

if [ "$1" = "rag-app" ] && [ "$2" = "serve" ]; then
    rag-app fetch-index || echo "fetch-index skipped; serving with whatever index is present."
fi

exec "$@"
