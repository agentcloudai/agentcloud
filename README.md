# rag-app: modular RAG with LlamaIndex

Ask questions about your own documents and get answers where **every sentence cites its source**.
Each RAG step is its own Python package, so you can build, test and swap one step at a time.

**Runs with no database by default** — an embedded, file-based vector store (Chroma) ships in
the box, so a plain `pip install` just works. Switch to PostgreSQL/pgvector when you need scale
or multi-tenancy (see [Vector store backends](#vector-store-backends)).

## Flow

```
INDEX:  data/raw > ingest > chunking > embedding > storage (Chroma or pgvector)
QUERY:  question > retrieval > reranking > generation (cited JSON) > answer
EVAL:   eval/questions.jsonl > retrieval + reranking > recall@k, precision@k, MRR > eval/results/*.json
```

## Project layout

```
src/rag_app/
  config.py              all settings (env vars with RAG_ prefix or .env)
  logging_utils.py       shared logging
  ingest/loaders.py      STEP 1  load files / URLs, content hash, dedupe
  chunking/splitter.py   STEP 2  SentenceSplitter (chunk size + overlap)
  embedding/embedder.py  STEP 3  HuggingFace (local) or OpenAI embeddings
  storage/vector_store.py STEP 4 PGVectorStore: HNSW index + full-text column
  retrieval/retriever.py STEP 5  hybrid search + metadata filters
  reranking/reranker.py  STEP 6  cross-encoder reranker
  generation/            STEP 7  schemas.py (Pydantic), prompts.py (versioned), generator.py (LLM + citation check)
  evaluation/            STEP 8  metrics.py (pure Python), runner.py (runs eval set, saves results)
  pipelines/             wires steps: index_pipeline.py, query_pipeline.py (RAGService)
  cli.py                 rag-app index | ask | eval
tests/                   unit tests for metrics and citation validation
```

## AgentCloud AI

Your private AWS solutions agent — ask how-tos (cited steps) or "design …" (a 3D
architecture you can orbit), grounded in AWS docs & whitepapers.

```bash
pip install agtcld            # (from PyPI once published; or `pip install -e .` from this repo)
agtcld                        # shows the logo, asks: work in the terminal, or open the web app?
```

- **Terminal mode** → an interactive agent right in your shell (`(~) agentcloud >`).
- **Web app** → the immersive localhost UI with the live 3D architecture view.

## Quick start (no database, no Docker)

Only Python 3.10+ is required. The vector store is embedded, so there is nothing to run.

```bash
git clone <this-repo> && cd rag_app
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[ui,dev]"          # installs the `agtcld` command; add ,browser for `crawl`

# Add your own PDFs/TXT/MD to ./data/raw, then:
rag-app index                       # chunks + embeds into an embedded Chroma store under data/
agtcld                              # launch the agent (terminal or web)
rag-app ask "What was decided about the solar project?"   # one-off query
```

`rag-app ask` needs an LLM: set `OPENAI_API_KEY=sk-...` (in `.env` or your shell), or point
`RAG_LLM_PROVIDER=vllm` at your own server. **Indexing needs no API key** (local embedding model).

Pull AWS's service docs (373 services) and query just one of them — every chunk is tagged with its service:

```bash
rag-app download --csv aws_service_guides.csv
rag-app index
rag-app ask "How do I enable S3 versioning?" --filter service="Amazon Simple Storage Service"
```

### Run it in Docker instead (optional)

Only needs [Docker](https://www.docker.com/products/docker-desktop/); still no database.

```bash
docker compose up -d --build        # builds + starts the app (embedded store)
docker compose exec app rag-app index
docker compose exec app rag-app ask "..."
```

Shortcuts with `make`: `make up`, `make download`, `make index`, `make ask Q="..."`, `make test`.

## Web UI (`rag-app serve`)

A local web app to ask AWS how-to questions and get a **crisp cited answer + an
architecture diagram** — plus an optional AI-art render.

```bash
pip install -e ".[ui]"          # fastapi + uvicorn;  add ,image for AI-art (diffusers)
rag-app serve                   # http://127.0.0.1:8000
```

What it shows per question:
- a one-line **answer** and numbered **steps**, each with clickable source citations
- a **structured architecture diagram** (Mermaid, with AWS service symbols) and a setup-steps flow — accurate, instant
- an optional **"AI art view"** (HuggingFace SD-Turbo) — a stylized render; the structured diagram is the accurate one. Needs `.[image]`; the model (~2.5 GB) downloads on first use. For gated models set `HF_TOKEN` in `.env`.
- a **service filter** so you can scope a question to one of the 363 services

## Vector store backends

Set `RAG_VECTOR_BACKEND` (default `chroma`):

| Backend | Needs | Good for |
| --- | --- | --- |
| `chroma` (default) | nothing — embedded, stored under `data/chroma` | laptops, demos, open-source, small/medium corpora |
| `pgvector` | a Postgres with the pgvector extension | large corpora, concurrency, multi-tenant / SaaS; adds hybrid (vector + full-text) search |

To use pgvector: `pip install -e ".[postgres]"`, set `RAG_VECTOR_BACKEND=pgvector`, and start a
database — e.g. `RAG_VECTOR_BACKEND=pgvector docker compose --profile pgvector up -d` (exposes it on
host port 5544). The table/extension are created automatically on first index.

## Crawling JavaScript sites

Some document listings (e.g. the [AWS whitepapers](https://aws.amazon.com/whitepapers/))
are rendered with JavaScript and paginated, so plain HTTP fetching finds no links.
`rag-app crawl` drives a real Chromium browser (Playwright) to render and page
through the listing, then downloads the PDFs into `data/raw/` ready for `rag-app index`.

```bash
# one-time setup (editable install already covers the code; just add the browser)
pip install playwright
playwright install chromium

# download up to 5 AWS whitepapers, watching the browser
rag-app crawl https://aws.amazon.com/whitepapers/ --limit 5 --show-browser

# then index them
rag-app index
```

Options: `--max-pages N` (listing pages to visit), `--limit N` (max files to
download), `--show-browser` (run Chromium with a visible window instead of headless).

The crawler picks a **site adapter** (`src/rag_app/ingest/sites/`) based on the URL:
`aws_whitepapers` knows how to turn whitepaper guide pages into their PDF URLs, and
`generic_pdf` is the fallback that simply keeps any link ending in `.pdf`. To support a
new site, add an adapter module next to those. If the crawl finds zero links, it saves
`crawl_debug.png` so you can inspect the rendered page and adjust the selectors.

## Downloading AWS service docs, chunked per service

`rag-app download` fetches PDFs listed in a CSV (columns `title, guide, pdf_url`) and
writes a `_sources.csv` manifest alongside them. On `rag-app index`, every chunk is then
tagged with its `service`, so you can group and filter retrieval per AWS service.

```bash
# all 373 services' how-to guides (large: many GB) ...
rag-app download --csv aws_service_guides.csv

# ... or just the services you care about
rag-app download --csv aws_service_guides.csv --service "Amazon Simple Storage Service" --service "AWS Lambda"

rag-app index
rag-app ask "How do I enable S3 versioning?" --filter service="Amazon Simple Storage Service"
```

`aws_service_guides.csv` (373 services / 439 guides) and `aws_all_doc_pdfs.csv` (1,541 incl.
whitepapers + architecture diagrams) were generated from AWS's documentation sitemap; each
guide's exact PDF URL comes from its `meta-inf/guide-info.json`. `aws_services.csv` is the
per-service index.

## What each step stores

| Where | What |
| --- | --- |
| `data/raw/` | your original files |
| `data/chroma/` (or Postgres `data_chunks` with pgvector) | chunk text, metadata (file name, page, hash, service), embedding vector |
| `eval/results/*.json` | scores per eval run, with the config used |

## Things to try (one change at a time, run `rag-app eval` after each)

1. Change `RAG_CHUNK_SIZE` (256 / 512 / 1024). Re-index first.
2. Turn hybrid search off: `RAG_HYBRID_SEARCH=false`.
3. Turn reranking off: `RAG_RERANK_ENABLED=false`.
4. Swap embedding model, e.g. `RAG_EMBED_MODEL=BAAI/bge-base-en-v1.5`, `RAG_EMBED_DIM=768`.
   Use a new `RAG_PG_TABLE` because the vector size changes.
5. Serve an open model with vLLM and set `RAG_LLM_PROVIDER=vllm`.

## Where to extend

- New source (e.g. scraped county agendas): add a function in `ingest/loaders.py`.
- Structure-aware chunking (one chunk per agenda item): add a splitter in `chunking/splitter.py`.
- Claim-level support check (LLM judge): extend `validate_citations` in `generation/generator.py`.
- Answer-quality metrics (faithfulness): add to `evaluation/`.
- Signal classifiers / entity linking: new packages next to the others, called from a pipeline.
