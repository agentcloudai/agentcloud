from rag_app.generation.schemas import Answer, Claim
from rag_app.generation.generator import validate_citations


def test_drops_claims_with_unknown_sources():
    ans = Answer(claims=[Claim(text="Real.", source_ids=["S1"]), Claim(text="Made up.", source_ids=["S9"])])
    out = validate_citations(ans, {"S1", "S2"})
    assert [c.text for c in out.claims] == ["Real."]
    assert out.insufficient_context is False


def test_all_dropped_marks_insufficient():
    ans = Answer(claims=[Claim(text="Made up.", source_ids=["S9"])])
    assert validate_citations(ans, {"S1"}).insufficient_context is True
