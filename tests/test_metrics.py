from rag_app.evaluation.metrics import hit_at_k, mean, precision_at_k, reciprocal_rank


def test_hit_at_k():
    assert hit_at_k(["a", "b", "c"], {"c"}, k=3) == 1.0
    assert hit_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a"], {"z"}) == 0.0


def test_precision_and_mean():
    assert precision_at_k(["a", "b", "c", "d"], {"a", "c"}, k=4) == 0.5
    assert mean([1.0, 0.0]) == 0.5
