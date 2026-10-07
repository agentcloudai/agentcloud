"""Tests for downloading and installing a prebuilt index bundle."""
import hashlib
import http.server
import socketserver
import tarfile
import threading
from pathlib import Path

import pytest

from rag_app.config import Settings
from rag_app.storage import index_bundle as ib


def _make_bundle(tmp: Path, extra_member: str | None = None) -> Path:
    """A minimal bundle shaped like a real one: chroma/chroma.sqlite3."""
    src = tmp / "chroma"
    src.mkdir()
    (src / "chroma.sqlite3").write_bytes(b"SQLite format 3\x00" + b"x" * 512)
    (src / "data_level0.bin").write_bytes(b"y" * 128)
    out = tmp / "bundle.tar.gz"
    with tarfile.open(out, "w:gz") as tar:
        tar.add(src, arcname="chroma")
        if extra_member:                       # simulate a hostile path
            evil = tmp / "evil.txt"
            evil.write_text("pwned")
            tar.add(evil, arcname=extra_member)
    return out


@pytest.fixture()
def served(tmp_path):
    """Serve tmp_path over HTTP and yield the base URL."""
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=str(tmp_path), **k)  # noqa: E731
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as httpd:
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
        httpd.shutdown()


def _settings(tmp_path) -> Settings:
    return Settings(chroma_path=str(tmp_path / "store" / "chroma"))


def test_fetch_installs_and_is_idempotent(tmp_path, served):
    _make_bundle(tmp_path)
    s = _settings(tmp_path)
    assert ib.fetch_index(f"{served}/bundle.tar.gz", s) is True
    assert (Path(s.chroma_path) / "chroma.sqlite3").exists()
    assert (Path(s.chroma_path) / ".index_complete").exists()
    # second call is a no-op
    assert ib.fetch_index(f"{served}/bundle.tar.gz", s) is False


def test_interrupted_extract_is_refetched(tmp_path, served):
    """A half-written index must NOT be treated as valid (it used to be)."""
    _make_bundle(tmp_path)
    s = _settings(tmp_path)
    chroma = Path(s.chroma_path)
    chroma.mkdir(parents=True)
    chroma.joinpath("chroma.sqlite3").write_bytes(b"truncated")   # no completion marker
    chroma.parent.joinpath(".chroma_incoming").mkdir()            # evidence of a crashed run
    assert ib.index_present(s) is False
    assert ib.fetch_index(f"{served}/bundle.tar.gz", s) is True
    assert chroma.joinpath(".index_complete").exists()


def test_empty_env_url_means_do_not_fetch(tmp_path, monkeypatch):
    """Set-but-empty is a deliberate opt-out, not 'use the default'."""
    monkeypatch.setenv("RAG_INDEX_URL", "")
    assert ib.fetch_index(None, _settings(tmp_path)) is False


def test_unset_url_falls_back_to_published_bundle(tmp_path, monkeypatch):
    """pip users with no configuration still get the published index."""
    monkeypatch.delenv("RAG_INDEX_URL", raising=False)
    used = {}
    monkeypatch.setattr(ib, "_download", lambda u, d, sha=None: used.update(url=u, sha=sha))
    with pytest.raises(Exception):      # no real archive lands, which is fine here
        ib.fetch_index(None, _settings(tmp_path))
    assert used["url"] == ib.DEFAULT_INDEX_URL
    assert used["sha"] == ib.DEFAULT_INDEX_SHA256


def test_checksum_mismatch_rejected(tmp_path, served, monkeypatch):
    _make_bundle(tmp_path)
    monkeypatch.setenv("RAG_INDEX_SHA256", hashlib.sha256(b"not it").hexdigest())
    s = _settings(tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        ib.fetch_index(f"{served}/bundle.tar.gz", s)
    assert not Path(s.chroma_path).exists()        # nothing installed


def test_path_traversal_blocked(tmp_path, served):
    _make_bundle(tmp_path, extra_member="../escaped.txt")
    s = _settings(tmp_path)
    ib.fetch_index(f"{served}/bundle.tar.gz", s)
    assert not (Path(s.chroma_path).parent.parent / "escaped.txt").exists()
