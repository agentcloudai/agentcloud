"""Download remote files (e.g. PDFs) into the local data folder.

Used by ``rag-app crawl``: once a site adapter has turned listing links into
direct download URLs, this module fetches them so ``rag-app index`` can read
them from disk like any other local file.
"""
import re
import time
from pathlib import Path
from urllib.parse import unquote, urlsplit

import httpx

from rag_app.logging_utils import get_logger

log = get_logger(__name__)

# Characters that are illegal in Windows file names (and best avoided elsewhere).
_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def filename_for(url: str) -> str:
    """Derive a safe, cross-platform local file name from a URL's path.

    Strips the query/fragment and replaces characters that are illegal on
    Windows (e.g. ``*`` sometimes appears in malformed upstream metadata).
    """
    name = Path(unquote(urlsplit(url).path)).name
    name = _UNSAFE_CHARS.sub("_", name).strip(" .")
    return name or "download"


def download_files(
    urls: list[str],
    dest_dir: str | Path,
    limit: int | None = None,
    delay: float = 1.0,
) -> list[Path]:
    """Download each URL into ``dest_dir``.

    Skips files that already exist, waits ``delay`` seconds between downloads to
    be polite, and skips (does not crash on) URLs that fail to download.

    Returns the paths that are present on disk afterwards (existing + newly saved).
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    selected = urls[:limit] if limit is not None else urls
    total = len(selected)
    saved: list[Path] = []

    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        for i, url in enumerate(selected, start=1):
            target = dest / filename_for(url)
            if target.exists():
                log.info("Skipping %s (already exists) (%d/%d)", target.name, i, total)
                saved.append(target)
                continue

            if saved or i > 1:
                time.sleep(delay)  # polite pause between network requests

            try:
                resp = client.get(url)
                resp.raise_for_status()
                target.write_bytes(resp.content)
            except (httpx.HTTPError, OSError) as exc:
                # One bad URL or file name must not abort the whole batch.
                log.warning("Failed to download %s: %s", url, exc)
                continue

            saved.append(target)
            log.info("Saved %s (%d/%d)", target.name, i, total)

    return saved
