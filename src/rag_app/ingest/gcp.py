"""Ingest Google Cloud documentation into the vector store, tagged per service —
the GCP counterpart of the AWS/Azure ingesters.

Unlike Azure (whose docs are an open Markdown repo), Google Cloud docs are not
open-sourced, so we crawl each service's docs section over HTTP. The pages at
docs.cloud.google.com are server-rendered: every page embeds the service's full
left-nav, so a small BFS over same-service ``/<slug>/docs/`` links enumerates the
section. We parse each page once, extract the ``<article class="devsite-article">``
body, strip devsite chrome, and tag each chunk with the friendly service name +
the real doc URL. Crawl + embed run per service, so progress is resumable.
"""
import re
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline

from rag_app.chunking.splitter import build_splitter
from rag_app.config import Settings, get_settings
from rag_app.embedding.embedder import build_embed_model
from rag_app.ingest.loaders import _finalize, deduplicate
from rag_app.logging_utils import get_logger
from rag_app.pipelines.index_pipeline import _append_done, _load_done, _progress_path
from rag_app.storage.vector_store import add_nodes_batched, build_vector_store

log = get_logger(__name__)

_BASE = "https://docs.cloud.google.com"
_UA = "Mozilla/5.0 (compatible; AgentCloud docs indexer)"
_MIN_CHARS = 250
# Skip generated API reference, changelogs, sample dumps and version-pinned mirrors
# (e.g. /hybrid/v1.4/...) — huge, low value for how-tos, and heavily duplicated.
_SKIP_RE = re.compile(
    r"/(reference|release-notes|samples|quotas|iam-roles|rpc|resources)(/|$)"
    r"|/v\d+(\.\d+)?(/|$)"
)
_BOILER = ("Stay organized with collections", "Save and categorize content based on your preferences.")

GCP_SERVICE_NAMES = {  # docs slug -> friendly service name
    "apigee": "Apigee API Management", "bigquery": "BigQuery",
    "compute": "Compute Engine", "cdn": "Cloud CDN", "run": "Cloud Run",
    "storage": "Cloud Storage", "sql": "Cloud SQL",
    "kubernetes-engine": "Google Kubernetes Engine", "looker": "Looker",
    "vpc": "Virtual Private Cloud", "iam": "Identity and Access Management",
    "monitoring": "Cloud Monitoring", "logging": "Cloud Logging",
}


def _extract(soup: BeautifulSoup) -> tuple[str, str] | None:
    """Return (title, clean_text) from an already-parsed devsite page, or None."""
    art = soup.find("article", class_="devsite-article")
    if art is None:
        return None
    for sel in ("script", "style", "nav", "aside", "button", "devsite-feedback",
                "devsite-thumb-rating", "devsite-code-buttons-container",
                ".devsite-article-meta", ".nocontent", "[class*=breadcrumb]",
                ".devsite-banner", ".devsite-page-rating"):
        for t in art.select(sel):
            t.decompose()
    text = re.sub(r"\n{3,}", "\n\n", art.get_text("\n", strip=True)).strip()
    for b in _BOILER:
        text = text.replace(b, "")
    text = text.strip()
    h1 = soup.find("h1")
    title = h1.get_text(strip=True).split("Stay organized")[0].strip() if h1 else ""
    return title, text


def crawl_service(slug: str, client: httpx.Client, max_pages: int = 150, delay: float = 0.05) -> list[Document]:
    """BFS over a service's /docs/ nav, parsing each page once for links + content."""
    name = GCP_SERVICE_NAMES[slug]
    pref = f"/{slug}/docs/"
    start = f"{_BASE}/{slug}/docs"
    seen = {start}
    queue = [start]
    docs: list[Document] = []
    fetched = 0
    while queue and len(docs) < max_pages:
        url = queue.pop(0)
        try:
            r = client.get(url, timeout=30.0)
        except httpx.HTTPError as exc:
            log.debug("GCP fetch failed %s: %s", url, exc)
            continue
        if r.status_code != 200:
            continue
        fetched += 1
        soup = BeautifulSoup(r.text, "html.parser")
        # Discover same-service links first (extraction mutates the tree).
        for a in soup.select("a[href]"):
            href = a.get("href", "").split("#")[0].split("?")[0]
            if href.startswith(_BASE):
                href = href[len(_BASE):]
            if href.startswith(pref) and not _SKIP_RE.search(href):
                full = _BASE + href
                if full not in seen:
                    seen.add(full)
                    queue.append(full)
        parsed = _extract(soup)
        if not parsed:
            continue
        title, text = parsed
        if len(text) < _MIN_CHARS:
            continue
        rel = url[len(_BASE):].lstrip("/")
        d = Document(text=text, id_=f"gcp/{rel}")
        d.metadata["file_name"] = f"gcp/{rel}"
        d.metadata["service"] = name
        d.metadata["guide"] = "Google Cloud documentation"
        d.metadata["source_url"] = url
        if title:
            d.metadata["title"] = title
        docs.append(_finalize(d))
        time.sleep(delay)
    log.info("Crawled %s: fetched %d pages, kept %d docs", name, fetched, len(docs))
    return docs


def load_gcp_docs(services: list[str] | None = None, max_pages: int = 150) -> list[Document]:
    """Crawl every requested service (used for inspection; indexing crawls per service)."""
    slugs = _resolve_slugs(services)
    docs: list[Document] = []
    with httpx.Client(headers={"User-Agent": _UA}, follow_redirects=True) as client:
        for slug in slugs:
            docs.extend(crawl_service(slug, client, max_pages=max_pages))
    log.info("Loaded %d GCP docs across %d services", len(docs), len(slugs))
    return docs


def _resolve_slugs(services: list[str] | None) -> list[str]:
    slugs = services or list(GCP_SERVICE_NAMES)
    unknown = [s for s in slugs if s not in GCP_SERVICE_NAMES]
    if unknown:
        raise ValueError(f"Unknown GCP service slug(s): {unknown}. Known: {sorted(GCP_SERVICE_NAMES)}")
    return slugs


def run_gcp_index(services: list[str] | None = None, s: Settings | None = None, max_pages: int = 150) -> int:
    """Crawl + embed one service at a time so completed services persist (resumable)."""
    s = s or get_settings()
    slugs = _resolve_slugs(services)
    vector_store = build_vector_store(s)
    progress = _progress_path(s)
    splitter = build_splitter(s.chunk_size, s.chunk_overlap)
    embed = build_embed_model(s)
    batch = max(1, s.index_batch_docs)
    total = 0
    with httpx.Client(headers={"User-Agent": _UA}, follow_redirects=True) as client:
        for slug in slugs:
            docs = deduplicate(crawl_service(slug, client, max_pages=max_pages))
            done = _load_done(progress)
            todo = [d for d in docs if d.metadata.get("content_hash") not in done]
            if not todo:
                log.info("%s: nothing new (%d docs already stored)", GCP_SERVICE_NAMES[slug], len(docs))
                continue
            for start in range(0, len(todo), batch):
                group = todo[start:start + batch]
                nodes = IngestionPipeline(transformations=[splitter, embed]).run(documents=group)
                add_nodes_batched(vector_store, nodes, batch_size=s.vector_add_batch_size)
                _append_done(progress, [d.metadata["content_hash"] for d in group])
                total += len(nodes)
                log.info("GCP indexed [%s]: %d docs, %d chunks (%d/%d docs)",
                         GCP_SERVICE_NAMES[slug], len(group), len(nodes),
                         min(start + batch, len(todo)), len(todo))
    log.info("Indexed %d GCP chunks total", total)
    return total
