"""Load documents from a folder (PDF, TXT, MD, DOCX, HTML...) or from URLs.

Every Document gets:
  - a stable id (its file path) so re-indexing replaces instead of duplicating
  - a content_hash so identical documents can be skipped
"""
import hashlib
import os
from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader

from rag_app.logging_utils import get_logger

log = get_logger(__name__)

# Metadata that is useful to store but should NOT influence embeddings or the LLM prompt.
_HIDDEN_KEYS = ["content_hash", "file_path", "file_size", "creation_date", "last_modified_date",
                "source", "total_pages", "file_type"]


def _finalize(doc: Document) -> Document:
    doc.metadata["content_hash"] = hashlib.sha256(doc.text.encode("utf-8")).hexdigest()
    doc.excluded_embed_metadata_keys = _HIDDEN_KEYS
    doc.excluded_llm_metadata_keys = _HIDDEN_KEYS
    return doc


def _auto_workers(num_workers: int) -> int:
    """Resolve 0 to a sensible parallelism (CPU count, capped); clamp to >=1."""
    if num_workers and num_workers > 0:
        return num_workers
    return max(1, min(os.cpu_count() or 1, 16))


def load_directory(data_dir: str | Path, num_workers: int = 0) -> list[Document]:
    """Read every supported file under data_dir (recursively).

    PDF parsing is CPU-bound and single-threaded per file, so files are parsed in
    parallel across ``num_workers`` processes (0 = auto, based on CPU count).

    If a ``_sources.csv`` manifest is present (written by ``rag-app download``),
    each document is tagged with its ``service`` so chunks can be grouped and
    filtered per AWS service.
    """
    from rag_app.ingest.manifest import MANIFEST_NAME, load_manifest

    path = Path(data_dir)
    if not path.exists() or not any(p.is_file() and p.name != ".gitkeep" for p in path.rglob("*")):
        raise FileNotFoundError(f"No files found in {path.resolve()}")

    # PyMuPDF extracts PDF text ~20x faster than the default pypdf reader.
    file_extractor = None
    try:
        from llama_index.readers.file import PyMuPDFReader

        file_extractor = {".pdf": PyMuPDFReader()}
    except Exception:  # noqa: BLE001 - fall back to the default reader if pymupdf is absent
        log.warning("PyMuPDF unavailable; using the default (slower) PDF reader")

    reader = SimpleDirectoryReader(
        input_dir=str(path), recursive=True, filename_as_id=True,
        exclude=[".gitkeep", MANIFEST_NAME], file_extractor=file_extractor,
    )

    workers = _auto_workers(num_workers)
    # A single worker avoids multiprocessing overhead for tiny corpora.
    file_count = sum(1 for p in path.rglob("*") if p.is_file() and p.name not in (".gitkeep", MANIFEST_NAME))
    workers = min(workers, max(1, file_count))
    log.info("Parsing %d files from %s with %d worker(s)", file_count, path, workers)
    raw_docs = reader.load_data(num_workers=workers) if workers > 1 else reader.load_data()

    manifest = load_manifest(path)
    docs = []
    tagged = 0
    for d in raw_docs:
        # PyMuPDF stores the page number under "source"; expose it as page_label
        # so citations show pages like the default reader does.
        if "page_label" not in d.metadata and d.metadata.get("source"):
            d.metadata["page_label"] = str(d.metadata["source"])
        meta = manifest.get(d.metadata.get("file_name", ""))
        if meta:
            d.metadata["service"] = meta["service"]
            d.metadata["guide"] = meta["guide"]
            d.metadata["source_url"] = meta["source_url"]
            tagged += 1
        docs.append(_finalize(d))
    log.info("Loaded %d documents from %s (%d tagged with a service)", len(docs), path, tagged)
    return docs


def load_urls(urls: list[str]) -> list[Document]:
    """Fetch web pages and convert HTML to text. Needs: pip install -e '.[web]'."""
    from llama_index.readers.web import SimpleWebPageReader

    docs = SimpleWebPageReader(html_to_text=True).load_data(urls)
    for d, url in zip(docs, urls):
        d.id_ = url
        d.metadata["source_url"] = url
    docs = [_finalize(d) for d in docs]
    log.info("Loaded %d web pages", len(docs))
    return docs


def deduplicate(docs: list[Document]) -> list[Document]:
    """Drop documents whose text is identical to one already seen."""
    seen: set[str] = set()
    unique = []
    for d in docs:
        h = d.metadata["content_hash"]
        if h not in seen:
            seen.add(h)
            unique.append(d)
    if len(unique) < len(docs):
        log.info("Removed %d duplicate documents", len(docs) - len(unique))
    return unique
