"""Export and fetch a prebuilt vector index as a single compressed bundle.

The full multi-cloud index is large (several GB), so instead of baking it into the
Docker image we ship it once as a ``.tar.gz`` that the container downloads into its
data volume on first boot. ``export_index`` makes the bundle; ``fetch_index``
downloads + extracts it. Any HTTPS host works (HuggingFace Hub resolve URL, an
object store, a GitHub release asset) — set ``RAG_INDEX_URL`` to point at it.
"""
import os
import tarfile
import tempfile
from pathlib import Path

import httpx

from rag_app.config import Settings, get_settings
from rag_app.logging_utils import get_logger

log = get_logger(__name__)


def _chroma_dir(s: Settings) -> Path:
    return Path(s.chroma_path)


def index_present(s: Settings) -> bool:
    """True if a non-empty Chroma index already exists on disk."""
    db = _chroma_dir(s) / "chroma.sqlite3"
    return db.exists() and db.stat().st_size > 0


def export_index(out_path: str | Path, s: Settings | None = None) -> Path:
    """Tar+gzip the Chroma index directory into ``out_path`` for hosting."""
    s = s or get_settings()
    src = _chroma_dir(s)
    if not index_present(s):
        raise FileNotFoundError(f"No index found at {src} — build one before exporting.")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    log.info("Exporting index %s -> %s (this can take a while)…", src, out)
    with tarfile.open(out, "w:gz") as tar:
        # Store files under a top-level "chroma/" prefix so extraction is predictable.
        tar.add(src, arcname="chroma")
    log.info("Wrote %s (%.1f GB)", out, out.stat().st_size / 1e9)
    return out


def fetch_index(url: str | None = None, s: Settings | None = None, force: bool = False) -> bool:
    """Download + extract a prebuilt index bundle into the Chroma path.

    Returns True if it installed an index, False if it skipped (already present, or
    no URL given). Safe to call on every container start.
    """
    s = s or get_settings()
    url = url or os.environ.get("RAG_INDEX_URL")
    dst = _chroma_dir(s)
    if index_present(s) and not force:
        log.info("Index already present at %s — skipping fetch.", dst)
        return False
    if not url:
        log.info("No RAG_INDEX_URL set — starting with an empty index.")
        return False

    dst.mkdir(parents=True, exist_ok=True)
    log.info("Fetching prebuilt index from %s …", url)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=None) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                got = 0
                for chunk in r.iter_bytes(chunk_size=1 << 20):
                    tmp.write(chunk)
                    got += len(chunk)
                    if total and got % (200 << 20) < (1 << 20):
                        log.info("  downloaded %.1f/%.1f GB", got / 1e9, total / 1e9)
            tmp.flush()
            log.info("Extracting index into %s …", dst)
            with tarfile.open(tmp_path, "r:gz") as tar:
                members = tar.getmembers()
                # Bundles are created with a top-level "chroma/" prefix; strip it so
                # files land directly in the configured chroma_path.
                for m in members:
                    m.name = m.name.split("chroma/", 1)[-1] if "chroma/" in m.name else m.name
                tar.extractall(dst, members=[m for m in members if m.name])
        finally:
            tmp_path.unlink(missing_ok=True)
    log.info("Index ready at %s", dst)
    return True
