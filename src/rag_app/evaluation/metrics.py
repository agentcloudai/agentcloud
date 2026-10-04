"""Pure-Python retrieval metrics (no dependencies, easy to unit test)."""


def hit_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    """1.0 if any expected item appears in the top k results, else 0.0. Averaged = recall@k / hit rate."""
    return 1.0 if any(r in expected for r in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: list[str], expected: set[str]) -> float:
    """1 / rank of the first correct result (0 if none). Averaged = MRR."""
    for rank, r in enumerate(retrieved, start=1):
        if r in expected:
            return 1.0 / rank
    return 0.0


def precision_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    top = retrieved[:k]
    return sum(r in expected for r in top) / k if k else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
