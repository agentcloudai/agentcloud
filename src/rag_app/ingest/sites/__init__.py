"""Site adapters: per-site rules for which links matter and how to turn them
into direct download URLs.

Each adapter module exposes:
  - ``NAME``: a short identifier
  - ``matches(start_url) -> bool``: whether the adapter handles this URL
  - ``is_relevant(link) -> bool``: keep this link found on a listing page?
  - ``to_download_url(link) -> str | None``: direct download URL for a link
  - ``dedupe_key(download_url) -> str``: key for de-duplicating downloads
"""
from types import ModuleType

from rag_app.ingest.sites import aws_whitepapers, generic_pdf

# Specific adapters first; the generic fallback matches everything, so it is last.
_ADAPTERS: list[ModuleType] = [aws_whitepapers, generic_pdf]


def pick_adapter(start_url: str) -> ModuleType:
    """Return the first adapter whose ``matches`` accepts ``start_url``."""
    for adapter in _ADAPTERS:
        if adapter.matches(start_url):
            return adapter
    return generic_pdf  # unreachable (generic_pdf.matches is always True), kept for safety
