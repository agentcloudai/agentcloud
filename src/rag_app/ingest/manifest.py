"""A sidecar manifest that records where each downloaded file came from.

When ``rag-app download`` saves PDFs it also writes/updates ``_sources.csv`` in
the data folder, mapping each local file name to its service, guide type and
source URL. The loader reads this so every chunk can be tagged with its
``service`` (e.g. "Amazon Elastic Compute Cloud"), letting you group or filter
retrieval by service.
"""
import csv
from pathlib import Path

MANIFEST_NAME = "_sources.csv"
_FIELDS = ["file_name", "service", "guide", "source_url"]


def manifest_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / MANIFEST_NAME


def load_manifest(data_dir: str | Path) -> dict[str, dict[str, str]]:
    """Return ``{file_name: {service, guide, source_url}}`` (empty if none)."""
    path = manifest_path(data_dir)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as fh:
        return {row["file_name"]: row for row in csv.DictReader(fh)}


def update_manifest(data_dir: str | Path, records: list[dict[str, str]]) -> None:
    """Merge ``records`` (each with the manifest fields) into ``_sources.csv``.

    Keyed by ``file_name`` so re-downloading updates the existing row.
    """
    merged = load_manifest(data_dir)
    for rec in records:
        merged[rec["file_name"]] = {k: rec.get(k, "") for k in _FIELDS}
    path = manifest_path(data_dir)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDS)
        writer.writeheader()
        for row in sorted(merged.values(), key=lambda r: r["file_name"]):
            writer.writerow(row)
