import math
from statistics import mean
from typing import Any, Dict, List, Mapping, Sequence

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
            f"no_answer_false_positive@{k}": (1.0 if retrieved_sources[:k] else 0.0)
            for k in k_values
        }

    metrics: Dict[str, float] = {
        "mrr": reciprocal_rank(retrieved_sources, expected_sources)
    }
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(retrieved_sources, expected_sources, k)
        metrics[f"hit@{k}"] = hit_at_k(retrieved_sources, expected_sources, k)
    return metrics


def _identity(item: Mapping[str, Any]) -> tuple[str, str]:
    return str(item.get("chunk_id") or ""), str(item.get("source") or "")


def _requirement_is_covered(
    top_items: Sequence[Mapping[str, Any]], requirement: Mapping[str, Any]
) -> bool:
    expected_chunks = {
        str(value)
        for value in requirement.get("acceptable_chunk_ids", [])
        if str(value)
    }
    expected_sources = {
        str(value) for value in requirement.get("acceptable_sources", []) if str(value)
    }
    for item in top_items:
        chunk_id, source = _identity(item)
        if (chunk_id and chunk_id in expected_chunks) or (
            source and source in expected_sources
        ):
            return True
    return False


# 计算一个有界上下文对全部 gold requirements 的覆盖比例。
def requirement_coverage_rate(
    retrieved: Sequence[Mapping[str, Any]],
    requirements: Sequence[Mapping[str, Any]],
) -> float:
    if not requirements:
        return 0.0
    covered = sum(
        _requirement_is_covered(retrieved, requirement) for requirement in requirements
    )
    return covered / len(requirements)


def _requirement_mask(
    item: Mapping[str, Any], requirements: Sequence[Mapping[str, Any]]
) -> int:
    chunk_id, source = _identity(item)
    mask = 0
    for index, requirement in enumerate(requirements):
        expected_chunks = {
            str(value)
            for value in requirement.get("acceptable_chunk_ids", [])
            if str(value)
        }
        expected_sources = {
            str(value)
            for value in requirement.get("acceptable_sources", [])
            if str(value)
        }
        if (chunk_id and chunk_id in expected_chunks) or (
            source and source in expected_sources
        ):
            mask |= 1 << index
    return mask


def _ideal_requirement_masks(
    requirements: Sequence[Mapping[str, Any]], actual_masks: Sequence[int]
) -> List[int]:
    # 同一 gold identity 可同时覆盖多个需求；按 identity 合并 mask，才能正确表示
    # “一个 chunk 覆盖两个需求”的理想排序。
    by_identity: Dict[tuple[str, str], int] = {}
    for index, requirement in enumerate(requirements):
        bit = 1 << index
        for chunk_id in requirement.get("acceptable_chunk_ids", []):
            identity = ("chunk", str(chunk_id))
            if identity[1]:
                by_identity[identity] = by_identity.get(identity, 0) | bit
        for source in requirement.get("acceptable_sources", []):
            identity = ("source", str(source))
            if identity[1]:
                by_identity[identity] = by_identity.get(identity, 0) | bit

    # 实际 item 可能同时用 chunk 和 source 命中不同需求；将其可实现 mask 纳入
    # 理想候选，保证 IDCG 不会低于当前排序本身可实现的增益。
    return sorted({mask for mask in (*by_identity.values(), *actual_masks) if mask})


def _requirement_ndcg(
    actual_masks: Sequence[int], ideal_masks: Sequence[int], k: int
) -> float:
    covered = 0
    dcg = 0.0
    for rank, mask in enumerate(actual_masks[:k], start=1):
        new_requirements = (mask & ~covered).bit_count()
        dcg += new_requirements / math.log2(rank + 1)
        covered |= mask

    if not ideal_masks or k <= 0:
        return 0.0

    # requirement 数通常不超过 3；按覆盖 mask 做精确 DP，得到允许一块覆盖多个
    # 需求时真正可实现的 IDCG，而不是假定每个 rank 只能贡献 1。
    best_by_covered = {0: 0.0}
    for rank in range(1, k + 1):
        discount = math.log2(rank + 1)
        next_best = dict(best_by_covered)
        for current_mask, score in best_by_covered.items():
            for candidate_mask in ideal_masks:
                combined = current_mask | candidate_mask
                gain = (candidate_mask & ~current_mask).bit_count() / discount
                next_best[combined] = max(next_best.get(combined, 0.0), score + gain)
        best_by_covered = next_best
    ideal = max(best_by_covered.values(), default=0.0)
    return min(dcg / ideal, 1.0) if ideal else 0.0


# 以原子证据需求和 chunk 级标注评估真实证据覆盖，避免“命中正确 PDF 的错误块”被算作成功。
def evaluate_requirement_coverage(
    retrieved_items: Sequence[Mapping[str, Any]],
    gold_requirements: Sequence[Mapping[str, Any]],
    k_values: Sequence[int],
    *,
    hard_negative_chunk_ids: Sequence[str] = (),
) -> Dict[str, float]:
    requirements = [
        requirement
        for requirement in gold_requirements
        if isinstance(requirement, Mapping)
        and (
            requirement.get("acceptable_chunk_ids")
            or requirement.get("acceptable_sources")
        )
    ]
    if not requirements:
        return {}

    expected_chunk_ids = {
        str(chunk_id)
        for requirement in requirements
        for chunk_id in requirement.get("acceptable_chunk_ids", [])
        if str(chunk_id)
    }
    expected_sources = {
        str(source)
        for requirement in requirements
        for source in requirement.get("acceptable_sources", [])
        if str(source)
    }
    hard_negatives = {str(value) for value in hard_negative_chunk_ids if str(value)}
    metrics: Dict[str, float] = {}
    for k in k_values:
        top_items = list(retrieved_items[:k])
        covered = sum(
            _requirement_is_covered(top_items, requirement)
            for requirement in requirements
        )
        metrics[f"requirement_recall@{k}"] = covered / len(requirements)
        metrics[f"all_requirements_covered@{k}"] = float(covered == len(requirements))

        relevances = []
        for item in top_items:
            chunk_id, source = _identity(item)
            relevances.append(
                float(
                    bool(
                        (chunk_id and chunk_id in expected_chunk_ids)
                        or (source and source in expected_sources)
                    )
                )
            )
        actual_masks = [
            _requirement_mask(item, requirements) for item in retrieved_items
        ]
        metrics[f"evidence_ndcg@{k}"] = _requirement_ndcg(
            actual_masks,
            _ideal_requirement_masks(requirements, actual_masks),
            k,
        )
        if expected_chunk_ids:
            metrics[f"chunk_precision@{k}"] = (
                sum(relevances) / len(top_items) if top_items else 0.0
            )
        if hard_negatives:
            hard_negative_hit = any(
                _identity(item)[0] in hard_negatives for item in top_items
            )
            metrics[f"hard_negative_rejection@{k}"] = float(not hard_negative_hit)
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
