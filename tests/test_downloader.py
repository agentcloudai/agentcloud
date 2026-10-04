"""Unit tests for the downloader's file-name handling (no network)."""
from rag_app.ingest.downloader import filename_for


def test_basic_name_from_url():
    assert filename_for("https://x/a/b/report.pdf") == "report.pdf"


def test_strips_query_and_fragment():
    assert filename_for("https://x/a/report.pdf?v=2#top") == "report.pdf"


def test_sanitizes_windows_illegal_chars():
    # AWS metadata sometimes contains a literal '*' in the path.
    out = filename_for("https://x/y/migration-hub-journeys-user-guide.*.pdf")
    assert "*" not in out
    assert out == "migration-hub-journeys-user-guide._.pdf"


def test_decodes_percent_encoding():
    assert filename_for("https://x/My%20Guide.pdf") == "My Guide.pdf"


def test_falls_back_when_no_name():
    assert filename_for("https://example.com/") == "download"
