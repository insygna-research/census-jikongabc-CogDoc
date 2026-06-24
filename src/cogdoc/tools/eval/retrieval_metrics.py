from statistics import mean
from typing import Dict, List, Sequence


def recall_at_k(
    retrieved_sources: Sequence[str], expected_sources: Sequence[str], k: int
) -> float:
    expected = set(expected_sources)
    if not expected:
        return 0.0
    top = set(retrieved_sources[:k])
    return len(expected & top) / len(expected)


def hit_at_k(
    retrieved_sources: Sequence[str], expected_sources: Sequence[str], k: int
) -> float:
    expected = set(expected_sources)
    if not expected:
        return 0.0
    return 1.0 if expected & set(retrieved_sources[:k]) else 0.0


def reciprocal_rank(
    retrieved_sources: Sequence[str], expected_sources: Sequence[str]
) -> float:
    expected = set(expected_sources)
    for rank, source in enumerate(retrieved_sources, start=1):
        if source in expected:
            return 1.0 / rank
    return 0.0


def evaluate_query(
    retrieved_sources: Sequence[str],
    expected_sources: Sequence[str],
    k_values: Sequence[int],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "mrr": reciprocal_rank(retrieved_sources, expected_sources)
    }
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(retrieved_sources, expected_sources, k)
        metrics[f"hit@{k}"] = hit_at_k(retrieved_sources, expected_sources, k)
    return metrics


def aggregate(per_query_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_query_metrics:
        return {}
    keys = per_query_metrics[0].keys()
    return {key: mean(metrics[key] for metrics in per_query_metrics) for key in keys}
