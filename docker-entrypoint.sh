#!/bin/sh
# Startup for the web UI.
#
# The image normally ships the vector index at /opt/index, so there is nothing to
# download. For a slim build (--build-arg WITH_INDEX=false) there is no baked
# index: point Chroma at the mounted volume instead and fetch a bundle into it if
# RAG_INDEX_URL is set. One-off commands (ingest/download/...) skip all of this.
set -e

if [ "$1" = "rag-app" ] && [ "$2" = "serve" ]; then
    if [ ! -f "/opt/index/chroma.sqlite3" ]; then
        export RAG_CHROMA_PATH="${RAG_CHROMA_PATH_OVERRIDE:-/app/data/chroma}"
        rag-app fetch-index || echo "fetch-index skipped; serving with whatever index is present."
    fi
fi

exec "$@"
