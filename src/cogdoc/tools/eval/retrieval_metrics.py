from statistics import mean
from typing import Dict, List, Sequence

RECOMMENDED_RETRIEVAL_LAYERS = ("single-source", "multi-source", "no-answer")


# 计算atk。
def recall_at_k(
    retrieved_sources: Sequence[str], expected_sources: Sequence[str], k: int
) -> float:
    expected = set(expected_sources)
    if not expected:
        return 0.0
    top = set(retrieved_sources[:k])
    return len(expected & top) / len(expected)


# 计算atk。
def hit_at_k(
    retrieved_sources: Sequence[str], expected_sources: Sequence[str], k: int
) -> float:
    expected = set(expected_sources)
    if not expected:
        return 0.0
    return 1.0 if expected & set(retrieved_sources[:k]) else 0.0


# 计算排序。
def reciprocal_rank(
    retrieved_sources: Sequence[str], expected_sources: Sequence[str]
) -> float:
    expected = set(expected_sources)
    for rank, source in enumerate(retrieved_sources, start=1):
        if source in expected:
            return 1.0 / rank
    return 0.0


# 评估问题。
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


# 聚合结果。
def aggregate(per_query_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_query_metrics:
        return {}
    keys = per_query_metrics[0].keys()
    return {key: mean(metrics[key] for metrics in per_query_metrics) for key in keys}


# 推断检索评测样本层级。
def infer_retrieval_layer(item: dict) -> str:
    expected = item.get("expected_sources", [])
    if not expected:
        return "no-answer"
    if len(set(expected)) > 1:
        return "multi-source"
    return "single-source"


# 审计检索评测集覆盖面。
def audit_coverage(items: List[dict]) -> dict:
    layers = {
        str(item.get("layer") or infer_retrieval_layer(item)) for item in items
    }
    missing_layers = [
        layer for layer in RECOMMENDED_RETRIEVAL_LAYERS if layer not in layers
    ]
    return {
        "layers": sorted(layers),
        "missing_layers": missing_layers,
        "is_coverage_complete": not missing_layers,
    }
