"""Export and fetch a prebuilt vector index as a single compressed bundle.

The full multi-cloud index is large (several GB), so instead of baking it into the
Docker image we ship it once as a ``.tar.gz`` that the container downloads into its
data volume on first boot. ``export_index`` makes the bundle; ``fetch_index``
downloads + extracts it. Any HTTPS host works (HuggingFace Hub resolve URL, an
object store, a GitHub release asset) — set ``RAG_INDEX_URL`` to point at it.
"""
import hashlib
import os
import shutil
import tarfile
import time
from pathlib import Path

import httpx

from rag_app.config import Settings, get_settings
from rag_app.logging_utils import get_logger

log = get_logger(__name__)

# The published multi-cloud bundle (AWS + Azure + GCP). The Docker image bakes the
# index in and never needs this; it is the default for pip users, who otherwise
# have nothing to search. Override with RAG_INDEX_URL / --url.
DEFAULT_INDEX_URL = (
    "https://huggingface.co/datasets/rdprojects/agentcloud/resolve/main/agentcloud-index.tar.gz"
)
DEFAULT_INDEX_SHA256 = "fce9bcfa63d5318f00491e63ffeb619016e7aa01debf4c29aa3de001a7d06222"


def _chroma_dir(s: Settings) -> Path:
    return Path(s.chroma_path)


def _complete_marker(s: Settings) -> Path:
    return _chroma_dir(s) / ".index_complete"


def index_present(s: Settings) -> bool:
    """True if a COMPLETE Chroma index exists on disk.

    The marker matters: a half-extracted bundle still leaves a chroma.sqlite3
    behind, and without this check we would treat that corrupt index as valid
    forever and never re-fetch it.
    """
    db = _chroma_dir(s) / "chroma.sqlite3"
    if not (db.exists() and db.stat().st_size > 0):
        return False
    # Indexes built locally by `rag-app index` have no marker, so accept those —
    # unless a download was clearly interrupted mid-extract.
    return _complete_marker(s).exists() or not (_chroma_dir(s).parent / ".chroma_incoming").exists()


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract with the 'data' filter, which blocks absolute paths, .. traversal and links."""
    members = []
    for m in tar.getmembers():
        name = m.name.split("chroma/", 1)[-1] if "chroma/" in m.name else m.name
        if not name or name.startswith("/") or ".." in Path(name).parts:
            log.warning("Skipping unsafe path in bundle: %s", m.name)
            continue
        m.name = name
        members.append(m)
    try:
        tar.extractall(dest, members=members, filter="data")
    except TypeError:                      # Python < 3.12 has no filter argument
        tar.extractall(dest, members=members)


def _download(url: str, dest: Path, expect_sha256: str | None = None) -> None:
    """Stream to `dest`, resuming with a Range request and retrying transient failures."""
    attempts = 4
    for attempt in range(1, attempts + 1):
        got = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={got}-"} if got else {}
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=60.0, headers=headers) as r:
                if got and r.status_code == 200:   # server ignored Range: restart
                    got = 0
                    dest.unlink(missing_ok=True)
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0)) + got
                with open(dest, "ab" if got else "wb") as fh:
                    last = got
                    for chunk in r.iter_bytes(chunk_size=1 << 20):
                        fh.write(chunk)
                        got += len(chunk)
                        if got - last >= (250 << 20):
                            last = got
                            log.info("  downloaded %.1f/%.1f GB", got / 1e9, total / 1e9)
            break
        except (httpx.HTTPError, OSError) as exc:
            if attempt == attempts:
                raise
            log.warning("Download failed (%s); retrying %d/%d…", exc, attempt + 1, attempts)
            time.sleep(2 * attempt)

    if expect_sha256:
        log.info("Verifying checksum…")
        h = hashlib.sha256()
        with open(dest, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest().lower() != expect_sha256.strip().lower():
            raise ValueError(f"Index checksum mismatch: got {h.hexdigest()}, expected {expect_sha256}")
        log.info("Checksum OK.")


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
    no URL given). Safe to call on every container start: the index is swapped into
    place only after it has fully extracted, so an interrupted run never leaves a
    half-written index behind. Set RAG_INDEX_SHA256 to verify the download.
    """
    s = s or get_settings()
    env_url = os.environ.get("RAG_INDEX_URL")
    if url:
        pass                                   # caller was explicit
    elif env_url is not None:
        # Set-but-empty (or "none"/"off") is a deliberate "don't fetch anything".
        if env_url.strip().lower() in ("", "none", "off", "false"):
            log.info("RAG_INDEX_URL is empty — starting with an empty index.")
            return False
        url = env_url
    else:
        url = DEFAULT_INDEX_URL                # pip users get the published bundle

    dst = _chroma_dir(s)
    if index_present(s) and not force:
        log.info("Index already present at %s — skipping fetch.", dst)
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    staging = dst.parent / ".chroma_incoming"
    tmp_path = dst.parent / ".index_download.tar.gz"
    log.info("Fetching prebuilt index from %s …", url)
    try:
        _download(url, tmp_path, os.environ.get("RAG_INDEX_SHA256")
                  or (DEFAULT_INDEX_SHA256 if url == DEFAULT_INDEX_URL else None))

        free = shutil.disk_usage(dst.parent).free
        need = tmp_path.stat().st_size * 3      # archive + expanded copy, with headroom
        if free < need:
            raise OSError(f"Not enough disk space to unpack the index: "
                          f"{free/1e9:.1f} GB free, about {need/1e9:.1f} GB needed")

        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        log.info("Extracting index …")
        with tarfile.open(tmp_path, "r:gz") as tar:
            _safe_extract(tar, staging)
        if not (staging / "chroma.sqlite3").exists():
            raise ValueError("Bundle did not contain chroma.sqlite3 — wrong archive?")

        # Swap last, so the live index is replaced only by a complete one.
        if dst.exists():
            old = dst.parent / ".chroma_old"
            shutil.rmtree(old, ignore_errors=True)
            dst.rename(old)
            shutil.rmtree(old, ignore_errors=True)
        staging.rename(dst)
        (dst / ".index_complete").write_text("ok", encoding="utf-8")
    finally:
        tmp_path.unlink(missing_ok=True)
        shutil.rmtree(staging, ignore_errors=True)
    log.info("Index ready at %s", dst)
    return True
