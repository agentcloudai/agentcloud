<div align="center">

<img src="https://raw.githubusercontent.com/agentcloudai/agentcloud/main/docs/logo.png" alt="AgentCloud AI" width="320" />

# AgentCloud AI

**Your private cloud solutions agent — ask how-tos with cited steps, then watch it design the architecture in live 3D. Grounded in official cloud docs. Runs on your machine.**

[![PyPI](https://img.shields.io/pypi/v/agtcld.svg)](https://pypi.org/project/agtcld/)
[![Python](https://img.shields.io/pypi/pyversions/agtcld.svg)](https://pypi.org/project/agtcld/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Live demo](https://img.shields.io/badge/live-demo-ff9900.svg)](https://agentcloudai.github.io/)

[**▶ Try the live 3D demo**](https://agentcloudai.github.io/) · [Install](#install) · [What makes it different](#what-makes-it-different) · [Cloud support](#cloud-support)

</div>

---

AgentCloud answers questions about building on the cloud **from the official documentation** — every step cites the exact source page, so you get real APIs and real steps, not hallucinations. Ask a *how-to* and get cited steps; say *"design …"* and it proposes a full architecture (components, data flow, trade-offs, the questions it needs answered) and renders it as an **interactive 3D diagram** you can orbit and zoom. It runs **locally** with an embedded vector store — no database, no Docker, your questions stay yours.

> **AWS, Azure and Google Cloud are all available today** — one agent, the same 3D designs, grounded in each provider's docs.

## Install — Docker (recommended)

Two commands — no Python, no pip, no database, and nothing to index:

```bash
curl -LfO https://raw.githubusercontent.com/agentcloudai/agentcloud/main/docker-compose.yml
docker compose up -d
```

Then open **http://localhost:8000** and paste your OpenAI key when it asks. That's the whole setup.

On first boot the container downloads the **prebuilt index — 440k chunks covering AWS, Azure and Google Cloud** (~3.6 GB, once) into its volume, so the agent already knows all three clouds. The embedding and reranker models are baked into the image, so nothing else is fetched at query time.

| | |
|---|---|
| **Your key** | Entered once in the UI and kept in the container's volume (`0600`), never in the image. Prefer a file? Put `OPENAI_API_KEY=sk-...` in `.env` and it skips the prompt. |
| **Persistence** | Index, key and feedback live in a Docker volume — they survive restarts and upgrades. |
| **Upgrading** | `docker compose pull && docker compose up -d` |
| **Start empty instead** | Set `RAG_INDEX_URL=` (blank) to skip the download and build your own with the ingest commands below. |

<details>
<summary><b>Build your own index instead of using the prebuilt one</b></summary>

```bash
docker compose run --rm app rag-app ingest-gcp                         # Google Cloud
docker compose run --rm app rag-app download --csv aws_service_guides.csv \
  && docker compose run --rm app rag-app index                         # AWS
```
Then `rag-app export-index --out my-index.tar.gz`, host it, and point `RAG_INDEX_URL` at it to share with your team.
</details>

<details>
<summary><b>Prefer pip? (Python 3.10+)</b></summary>

```bash
pip install agtcld
agtcld            # shows the logo, asks: work in the terminal, or open the web app?
```

- **Terminal mode** → an interactive agent in your shell (`(~) agentcloud >`).
- **Web app** → `http://127.0.0.1:8000` with the live 3D architecture view (`rag-app serve`, or choose "web").
</details>

## What makes it different

| | |
|---|---|
| 📑 **Cited, not guessed** | Answers are built from the cloud's official docs & whitepapers; every step links to its source. |
| 🧊 **Live 3D architecture** | "design …" → real service icons as 3D tiles, arrowed data flow, real topology — orbit, zoom, hover for the *why*. |
| 🔒 **Private & local** | Embedded vector store (Chroma). No database, no Docker, nothing leaves your machine. |
| 💬 **Terminal or web** | Use `agtcld` in your shell or the immersive web UI. Answer the agent's follow-ups to refine a design in place. |
| ♻️ **Improves over time** | Built-in 👍/👎 + "better answer" capture → export SFT / preference datasets for fine-tuning (RLHF). |

## Cloud support

| Cloud | Status |
|---|---|
| **AWS** | ✅ Available — 373 services |
| **Azure** | ✅ Available — 20 core services |
| **Google Cloud** | ✅ Available — 13 core services |

## Quickstart with your own docs

```bash
pip install agtcld
# put your PDFs / TXT / MD in ./data/raw, then:
rag-app index                       # chunk + embed locally (GPU-accelerated if available)
rag-app ask "What does our return policy say about refunds?"
rag-app serve                       # or launch the web UI
```

`ask`/`design` need an LLM — set `OPENAI_API_KEY` (in `.env` or your shell), or point `RAG_LLM_PROVIDER=vllm` at your own server. **Indexing needs no API key** (local embedding model).

### Or load AWS's docs (373 services)

```bash
rag-app download --csv aws_service_guides.csv        # or --service "AWS Lambda" for a subset
rag-app index
rag-app ask "How do I enable S3 versioning?" --filter service="Amazon Simple Storage Service"
```

Every chunk is tagged with its service, so you can scope any question to one service. The CSVs (service guides, whitepapers, Well-Architected, prescriptive guidance) were generated from AWS's documentation sitemap.

### Or load Azure's docs (20 core services)

```bash
# sparse-clone the Markdown docs, then index (tagged per Azure service)
git clone --depth 1 --filter=blob:none --sparse https://github.com/MicrosoftDocs/azure-docs _azure_src
rag-app ingest-azure --src _azure_src
rag-app ask "How do I deploy a container to Azure Container Apps?"
```

Azure docs are sourced from [MicrosoftDocs/azure-docs](https://github.com/MicrosoftDocs/azure-docs); each chunk links back to its `learn.microsoft.com` page. They index into the **same** store as AWS, so you can mix providers in one deployment.

### Or load Google Cloud's docs (13 core services)

```bash
rag-app ingest-gcp                         # crawls docs.cloud.google.com, tagged per service
rag-app ingest-gcp --service run --service bigquery   # or a subset of services
rag-app ask "How do I deploy a container to Cloud Run?"
```

Google Cloud docs aren't open-sourced as Markdown, so AgentCloud crawls the (server-rendered) pages at `docs.cloud.google.com`, extracts the article body, and tags each chunk with its service + source URL — again into the same store, so AWS, Azure and GCP live side by side.

## How it works

```
INDEX:   your docs ──▶ parse (PyMuPDF) ──▶ chunk ──▶ embed ──▶ store (Chroma / pgvector)
ANSWER:  question ──▶ multi-query retrieval ──▶ rerank ──▶ cited answer  (+ grounding repair)
DESIGN:  request  ──▶ retrieve across services ──▶ components + flow + trade-offs ──▶ 3D diagram
```

## Feedback & RLHF

Interactions and feedback are captured locally for continuous improvement:

```bash
rag-app feedback            # stats (👍/👎, corrections)
rag-app feedback --export   # → sft.jsonl (prompt→response) + preferences.jsonl (chosen/rejected)
```

Opt-in automated collection: set `RAG_FEEDBACK_SUBMIT=true` and `RAG_FEEDBACK_ENDPOINT=<your collector>` (PII-scrubbed before upload). Host the collector with `rag-app collect-server`.

## Configuration (selected)

All settings are env vars with the `RAG_` prefix (or a `.env` file):

| Setting | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | – | needed for `ask`/`design` |
| `RAG_VECTOR_BACKEND` | `chroma` | `chroma` (embedded) or `pgvector` |
| `RAG_LLM_MODEL` | `gpt-4o-mini` | any OpenAI model; or `RAG_LLM_PROVIDER=vllm` |
| `RAG_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | local embeddings |
| `RAG_DEVICE` | `auto` | `cuda` when a GPU is present, else `cpu` |

## Advanced

<details>
<summary><b>Scale: PostgreSQL / pgvector</b></summary>

For large / multi-tenant deployments, `pip install "agtcld[postgres]"`, set `RAG_VECTOR_BACKEND=pgvector`, and start Postgres (e.g. `RAG_VECTOR_BACKEND=pgvector docker compose --profile pgvector up -d`). The table + extension are created on first index; pgvector also enables hybrid (vector + full-text) search.
</details>

<details>
<summary><b>Crawl JavaScript-rendered doc sites</b></summary>

```bash
pip install "agtcld[browser]" && playwright install chromium
rag-app crawl https://aws.amazon.com/whitepapers/ --limit 5
```
Site adapters live in `src/rag_app/ingest/sites/`; the fallback keeps any `.pdf` link.
</details>

<details>
<summary><b>Build the image yourself</b></summary>

The published image (`ghcr.io/agentcloudai/agentcloud`) is what `docker compose up` pulls. To build from source instead, edit `docker-compose.yml` to comment `image:` and uncomment `build: .`, then:

```bash
docker compose up -d --build
```
The embedded store lives on a mounted volume — still no database required.
</details>

<details>
<summary><b>Evaluate quality</b></summary>

```bash
rag-app eval --file eval/architecture_questions.jsonl     # recall@k / precision@k / MRR
rag-app eval-answers --judge                               # LLM-judged step coverage
```
</details>

## Contributing

Issues and PRs welcome — new cloud providers, site adapters, chunking/retrieval strategies, and eval sets especially. `pip install -e ".[dev]"` then `pytest`.

## License

MIT © Ruturaj Dixit
