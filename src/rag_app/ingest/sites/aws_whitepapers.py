"""Site adapter for the AWS whitepapers listing.

The listing at https://aws.amazon.com/whitepapers/ links to HTML guides such as

    https://docs.aws.amazon.com/whitepapers/latest/aws-overview/introduction.html

The matching PDF lives at (verified against the live site)

    https://docs.aws.amazon.com/pdfs/whitepapers/latest/aws-overview/aws-overview.pdf

i.e. the pattern ``/pdfs/whitepapers/latest/{slug}/{slug}.pdf``. Links that
already point at a ``.pdf`` are kept as-is (minus query string and fragment).
"""
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

NAME = "aws_whitepapers"

# A whitepaper guide URL: .../whitepapers/latest/<slug>/<something>.html
_GUIDE_RE = re.compile(r"/whitepapers/latest/(?P<slug>[^/]+)/")


def matches(start_url: str) -> bool:
    """True when this adapter should handle ``start_url``."""
    host = urlsplit(start_url).netloc.lower()
    path = urlsplit(start_url).path.lower()
    return host.endswith("aws.amazon.com") and "whitepaper" in path


def _slug(link: str) -> str | None:
    m = _GUIDE_RE.search(urlsplit(link).path)
    return m.group("slug") if m else None


def is_relevant(link: str) -> bool:
    """Keep whitepaper guide links and direct PDF links."""
    path = urlsplit(link).path.lower()
    return path.endswith(".pdf") or _slug(link) is not None


def _strip(link: str) -> str:
    """Drop query string and fragment."""
    parts = urlsplit(link)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def to_download_url(link: str) -> str | None:
    """Convert a listing link to a direct PDF URL, or None if it is not one."""
    clean = _strip(link)
    if clean.lower().endswith(".pdf"):
        return clean

    slug = _slug(clean)
    if slug is None:
        return None
    return f"https://docs.aws.amazon.com/pdfs/whitepapers/latest/{slug}/{slug}.pdf"


def dedupe_key(download_url: str) -> str:
    """Deduplicate whitepapers by slug so guide pages of the same paper collapse."""
    slug = _slug(download_url)
    if slug is not None:
        return slug
    return Path(urlsplit(download_url).path).stem
