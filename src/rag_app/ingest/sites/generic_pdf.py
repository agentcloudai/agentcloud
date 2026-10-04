"""Fallback site adapter: keep any link that points at a PDF.

Used when no site-specific adapter matches the start URL.
"""
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

NAME = "generic_pdf"


def matches(start_url: str) -> bool:
    """The fallback adapter matches anything (it is chosen last)."""
    return True


def is_relevant(link: str) -> bool:
    """Keep links whose path ends in ``.pdf``."""
    return urlsplit(link).path.lower().endswith(".pdf")


def to_download_url(link: str) -> str | None:
    """Return the PDF URL (minus query/fragment), or None if it is not a PDF."""
    parts = urlsplit(link)
    if not parts.path.lower().endswith(".pdf"):
        return None
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def dedupe_key(download_url: str) -> str:
    """Deduplicate by the file's stem."""
    return Path(urlsplit(download_url).path).stem
