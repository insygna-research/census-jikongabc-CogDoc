import math
from statistics import mean
from typing import Dict, List, Mapping, Sequence

RECOMMENDED_RETRIEVAL_LAYERS = (
    "single-source",
    "multi-source",
    "hard",
    "no-answer",
)
RETRIEVAL_COVERAGE_PROFILES = {
    "smoke": {layer: 1 for layer in RECOMMENDED_RETRIEVAL_LAYERS},
    "baseline": {
        "single-source": 40,
        "multi-source": 20,
        "hard": 20,
        "no-answer": 20,
    },
}
LOWER_IS_BETTER_PREFIXES = ("latency_", "no_answer_false_positive@")


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
    if not expected_sources:
        return {
            f"no_answer_false_positive@{k}": (
                1.0 if retrieved_sources[:k] else 0.0
            )
            for k in k_values
        }

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
    keys = sorted({key for metrics in per_query_metrics for key in metrics})
    return {
        key: mean(metrics[key] for metrics in per_query_metrics if key in metrics)
        for key in keys
    }


# 使用 nearest-rank 定义计算百分位，避免引入额外数值依赖。
def percentile(values: Sequence[float], percent: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 < percent <= 100:
        raise ValueError("percent must be in (0, 100]")
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil((percent / 100.0) * len(ordered)))
    return ordered[rank - 1]


# 返回指标优化方向，供基线比较正确处理延迟和误命中率。
def metric_direction(metric: str) -> str:
    if metric.startswith(LOWER_IS_BETTER_PREFIXES):
        return "lower"
    return "higher"


# 获取覆盖配置对应的每层最小样本数。
def coverage_minimums(profile: str) -> Dict[str, int]:
    if profile not in RETRIEVAL_COVERAGE_PROFILES:
        raise ValueError(f"未知检索覆盖配置: {profile}")
    return dict(RETRIEVAL_COVERAGE_PROFILES[profile])


# 推断检索评测样本层级。
def infer_retrieval_layer(item: dict) -> str:
    expected = item.get("expected_sources", [])
    if not expected:
        return "no-answer"
    if len(set(expected)) > 1:
        return "multi-source"
    return "single-source"


# 审计检索评测集覆盖面。
def audit_coverage(
    items: List[dict], minimum_counts: Mapping[str, int] | None = None
) -> dict:
    minimums = dict(minimum_counts or coverage_minimums("smoke"))
    layer_counts: Dict[str, int] = {}
    for item in items:
        layer = str(item.get("layer") or infer_retrieval_layer(item))
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
    missing_layers = [layer for layer in minimums if layer_counts.get(layer, 0) == 0]
    insufficient_layers = {
        layer: {
            "actual": layer_counts.get(layer, 0),
            "required": required,
        }
        for layer, required in minimums.items()
        if layer_counts.get(layer, 0) < required
    }
    return {
        "layers": sorted(layer_counts),
        "layer_counts": dict(sorted(layer_counts.items())),
        "minimum_layer_counts": minimums,
        "missing_layers": missing_layers,
        "insufficient_layers": insufficient_layers,
        "total_count": len(items),
        "is_coverage_complete": not insufficient_layers,
    }


# 根据绝对阈值生成门禁结果。minimum 指标越大越好，maximum 指标越小越好。
def evaluate_thresholds(aggregate_metrics: Mapping[str, float], config: dict) -> dict:
    rows = []
    bounds = (
        ("minimum", lambda current, limit: current >= limit),
        ("maximum", lambda current, limit: current <= limit),
    )
    for bound_name, comparator in bounds:
        for metric, raw_limit in sorted(config.get(bound_name, {}).items()):
            current = aggregate_metrics.get(metric)
            limit = float(raw_limit)
            passed = current is not None and comparator(float(current), limit)
            rows.append(
                {
                    "metric": metric,
                    "bound": bound_name,
                    "limit": limit,
                    "current": current,
                    "passed": passed,
                }
            )
    return {"passed": bool(rows) and all(row["passed"] for row in rows), "rows": rows}
