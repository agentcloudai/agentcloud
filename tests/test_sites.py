"""Unit tests for site adapters (no network)."""
from rag_app.ingest.sites import aws_whitepapers, generic_pdf, pick_adapter


def test_pick_adapter_aws():
    for url in [
        "https://aws.amazon.com/whitepapers/",
        "https://aws.amazon.com/whitepapers/?foo=bar",
    ]:
        assert pick_adapter(url) is aws_whitepapers


def test_pick_adapter_falls_back_to_generic():
    assert pick_adapter("https://example.com/docs/") is generic_pdf


def test_aws_guide_link_to_pdf():
    link = "https://docs.aws.amazon.com/whitepapers/latest/aws-overview/introduction.html"
    assert (
        aws_whitepapers.to_download_url(link)
        == "https://docs.aws.amazon.com/pdfs/whitepapers/latest/aws-overview/aws-overview.pdf"
    )


def test_aws_strips_query_and_fragment():
    link = "https://docs.aws.amazon.com/whitepapers/latest/aws-overview/introduction.html?x=1#top"
    assert (
        aws_whitepapers.to_download_url(link)
        == "https://docs.aws.amazon.com/pdfs/whitepapers/latest/aws-overview/aws-overview.pdf"
    )


def test_aws_keeps_direct_pdf_links():
    link = "https://docs.aws.amazon.com/pdfs/whitepapers/latest/foo/foo.pdf?versionId=9#p"
    assert (
        aws_whitepapers.to_download_url(link)
        == "https://docs.aws.amazon.com/pdfs/whitepapers/latest/foo/foo.pdf"
    )


def test_aws_ignores_unrelated_links():
    assert aws_whitepapers.to_download_url("https://aws.amazon.com/about-aws/") is None
    assert aws_whitepapers.is_relevant("https://aws.amazon.com/about-aws/") is False


def test_aws_is_relevant_for_guides_and_pdfs():
    assert aws_whitepapers.is_relevant(
        "https://docs.aws.amazon.com/whitepapers/latest/aws-overview/introduction.html"
    )
    assert aws_whitepapers.is_relevant("https://x/doc.pdf")


def test_aws_dedupe_by_slug():
    a = "https://docs.aws.amazon.com/whitepapers/latest/aws-overview/introduction.html"
    b = "https://docs.aws.amazon.com/whitepapers/latest/aws-overview/types-of-cloud-computing.html"
    ka = aws_whitepapers.dedupe_key(aws_whitepapers.to_download_url(a))
    kb = aws_whitepapers.dedupe_key(aws_whitepapers.to_download_url(b))
    assert ka == kb == "aws-overview"


def test_generic_keeps_only_pdfs():
    assert generic_pdf.is_relevant("https://x/report.pdf") is True
    assert generic_pdf.is_relevant("https://x/report.html") is False
    assert generic_pdf.to_download_url("https://x/a/report.pdf?z=1#f") == "https://x/a/report.pdf"
    assert generic_pdf.to_download_url("https://x/page.html") is None
