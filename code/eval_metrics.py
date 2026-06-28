"""
eval_metrics.py
---------------
Pure-Python retrieval metrics used across the lab. No external libraries.

We treat retrieval as: for a question, the system returns an ORDERED list of
document ids. We compare against the set of 'relevant' document ids from the
golden set.

Metrics:
  recall@k : of all the relevant docs, how many appeared in the top k?
  MRR@k    : 1 / (rank of the FIRST relevant doc), averaged over questions.
  nDCG@k   : rank-aware quality, rewards putting relevant docs higher.

Run this file directly to execute built-in self-tests:
    python eval_metrics.py
"""
from math import log2
from typing import List, Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of relevant docs found within the top k retrieved."""
    rel = set(relevant)
    if not rel:
        return float("nan")  # undefined for negative questions
    topk = list(retrieved[:k])
    hits = sum(1 for d in rel if d in topk)
    return hits / len(rel)


def reciprocal_rank_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """1 / rank of the first relevant doc within top k, else 0."""
    rel = set(relevant)
    if not rel:
        return float("nan")
    for i, d in enumerate(retrieved[:k], start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """
    Binary-relevance nDCG@k.
    DCG = sum over ranks of rel_i / log2(rank + 1).
    IDCG = the best achievable DCG (all relevant docs at the top).
    """
    rel = set(relevant)
    if not rel:
        return float("nan")
    dcg = 0.0
    for i, d in enumerate(retrieved[:k], start=1):
        if d in rel:
            dcg += 1.0 / log2(i + 1)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def mean_ignore_nan(values: List[float]) -> float:
    vals = [v for v in values if v == v]  # NaN != NaN
    return sum(vals) / len(vals) if vals else float("nan")


def _selftest():
    # Perfect retrieval
    assert recall_at_k(["a", "b", "c"], ["a", "b"], 3) == 1.0
    # Half found
    assert recall_at_k(["a", "x", "y"], ["a", "b"], 3) == 0.5
    # MRR: first relevant at rank 2 -> 0.5
    assert reciprocal_rank_at_k(["x", "a", "b"], ["a"], 5) == 0.5
    # MRR: none found -> 0
    assert reciprocal_rank_at_k(["x", "y"], ["a"], 5) == 0.0
    # nDCG: single relevant at rank 1 -> 1.0
    assert abs(ndcg_at_k(["a", "x"], ["a"], 5) - 1.0) < 1e-9
    # nDCG: single relevant at rank 2 -> 1/log2(3) / 1.0
    expected = (1.0 / log2(3))
    assert abs(ndcg_at_k(["x", "a"], ["a"], 5) - expected) < 1e-9
    # mean ignores NaN
    assert mean_ignore_nan([1.0, float("nan"), 0.0]) == 0.5
    print("eval_metrics self-test: ALL PASSED")


if __name__ == "__main__":
    _selftest()
