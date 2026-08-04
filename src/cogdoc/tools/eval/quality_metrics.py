import json
import math
from collections.abc import Mapping
from pathlib import Path
from statistics import mean
from typing import Any
from typing import Dict, List
from cogdoc.agents.citation_validator import CitationValidatorAgent
from cogdoc.agents.router import classify_intent_by_rule


BASELINE_GATED_METRICS = ("router_rule_accuracy", "citation_accuracy")
REQUIRED_CASE_TYPES = ("router", "citation", "faithfulness")
RECOMMENDED_LAYERS = (
    "easy",
    "hard",
    "no-answer",
    "summary",
    "compare",
    "multi-turn",
    "feedback",
)
CLAIM_VERDICTS = {"supported", "unsupported", "insufficient", "not_factual"}
CLAIM_AUDIT_OBSERVABLE_STATUSES = {
    "passed",
    "repaired",
    "failed",
    "rejected",
    "error",
}


# 计算结果。
def accuracy(correct: bool) -> float:
    return 1.0 if correct else 0.0


# 评估路由器用例。
def evaluate_router_case(item: dict) -> dict:
    decision = classify_intent_by_rule(item["query"])
    expected = item["expected_task_type"]
    return {
        "case_type": "router",
        "layer": item.get("layer", "unspecified"),
        "query": item["query"],
        "expected": expected,
        "actual": decision.task_type,
        "metrics": {"router_rule_accuracy": accuracy(decision.task_type == expected)},
    }


# 评估引用用例。
def evaluate_citation_case(item: dict) -> dict:
    result = CitationValidatorAgent.validate_citations(
        item.get("answer", ""), item.get("docs", [])
    )
    actual = bool(result["is_valid"])
    expected = bool(item["expected_valid"])
    return {
        "case_type": "citation",
        "layer": item.get("layer", "unspecified"),
        "expected": expected,
        "actual": actual,
        "metrics": {"citation_accuracy": accuracy(actual == expected)},
    }


# 从质量用例中读取运行时声明审计；兼容直接、output 和 trace.output 三种载荷。
def _claim_audit(item: Mapping[str, Any]) -> Mapping[str, Any] | None:
    candidates = (
        item.get("claim_audit"),
        (item.get("output") or {}).get("claim_audit")
        if isinstance(item.get("output"), Mapping)
        else None,
        ((item.get("trace") or {}).get("output") or {}).get("claim_audit")
        if isinstance(item.get("trace"), Mapping)
        and isinstance((item.get("trace") or {}).get("output"), Mapping)
        else None,
    )
    return next((value for value in candidates if isinstance(value, Mapping)), None)


# 只依据声明明细重新计算诊断计数，不信任运行时写入的 counts / metrics 汇总。
def _claim_audit_counts(audit: Mapping[str, Any]) -> dict[str, Any] | None:
    # `not_run` 只说明运行时写入了占位结构，不能把“门禁未执行”统计成可观测。
    if str(audit.get("status") or "") not in CLAIM_AUDIT_OBSERVABLE_STATUSES:
        return None
    claims = audit.get("claims")
    if not isinstance(claims, list):
        return None

    counts = {
        "claim_count": 0,
        "supported": 0,
        "unsupported": 0,
        "insufficient": 0,
        "cited": 0,
        "not_factual": 0,
        "repair_attempted": 0,
        "repair_succeeded": 0,
        "duration_ms": None,
    }
    for claim in claims:
        if not isinstance(claim, Mapping):
            return None
        verdict = str(claim.get("verdict") or "")
        if verdict not in CLAIM_VERDICTS:
            return None
        if verdict == "not_factual":
            counts["not_factual"] += 1
            continue
        counts["claim_count"] += 1
        counts[verdict] += 1
        if bool(claim.get("cited_chunk_ids")):
            counts["cited"] += 1

    repair = audit.get("repair")
    if isinstance(repair, Mapping) and bool(repair.get("attempted")):
        counts["repair_attempted"] = 1
        counts["repair_succeeded"] = int(bool(repair.get("succeeded")))

    verifier = audit.get("verifier")
    if isinstance(verifier, Mapping) and verifier.get("duration_ms") is not None:
        try:
            duration_ms = float(verifier["duration_ms"])
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(duration_ms):
            return None
        counts["duration_ms"] = max(0.0, duration_ms)
    return counts


# 安全计算比率；没有分母时保持不可用，而不是把它伪装成零分或满分。
def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


# 构建单条 faithfulness 用例的声明审计诊断。
def _claim_row_metrics(counts: Mapping[str, Any]) -> dict[str, float | None]:
    claim_count = int(counts["claim_count"])
    metrics: dict[str, float | None] = {
        "claim_audit_observable": 1.0,
        "claim_support_rate": _ratio(int(counts["supported"]), claim_count),
        "citation_coverage": _ratio(int(counts["cited"]), claim_count),
        "unsupported_claim_rate": _ratio(
            int(counts["unsupported"]), claim_count
        ),
        "insufficient_claim_rate": _ratio(
            int(counts["insufficient"]), claim_count
        ),
    }
    if counts["repair_attempted"]:
        metrics["repair_success"] = float(bool(counts["repair_succeeded"]))
    return metrics


# 评估faithfulness用例。
def evaluate_faithfulness_case(item: dict) -> dict:
    has_manual_label = "is_faithful" in item
    is_faithful = bool(item.get("is_faithful")) if has_manual_label else None
    metrics: dict[str, float | None] = {}
    if has_manual_label:
        metrics["faithfulness_manual_support_rate"] = accuracy(bool(is_faithful))

    audit = _claim_audit(item)
    claim_counts = _claim_audit_counts(audit) if audit is not None else None
    if claim_counts is None:
        metrics["claim_audit_observable"] = 0.0
    else:
        metrics.update(_claim_row_metrics(claim_counts))

    row = {
        "case_type": "faithfulness",
        "layer": item.get("layer", "unspecified"),
        "expected": True,
        "actual": is_faithful,
        "metrics": metrics,
        "manual_only": claim_counts is None,
    }
    if claim_counts is not None:
        row["claim_audit_status"] = str(audit.get("status") or "")
        row["claim_audit_counts"] = claim_counts
    return row


# 评估用例。
def evaluate_case(item: dict) -> dict:
    case_type = item.get("case_type")
    if case_type == "router":
        return evaluate_router_case(item)
    if case_type == "citation":
        return evaluate_citation_case(item)
    if case_type == "faithfulness":
        return evaluate_faithfulness_case(item)
    raise ValueError(f"不支持的评测 case_type: {case_type}")


# 计算指标。
def mean_metric(rows: List[dict], metric: str) -> float | None:
    values = [row["metrics"][metric] for row in rows if metric in row["metrics"]]
    return mean(values) if values else None


# 分组行列表。
def group_rows(rows: List[dict], key: str) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "unspecified")), []).append(row)
    return grouped


# 按 claim 数做微平均，并单独报告审计结果的可观测率。
def claim_audit_diagnostics(rows: List[dict]) -> dict[str, float | None]:
    faithfulness_rows = [row for row in rows if row.get("case_type") == "faithfulness"]
    observed = [
        row["claim_audit_counts"]
        for row in faithfulness_rows
        if isinstance(row.get("claim_audit_counts"), Mapping)
    ]
    total_claims = sum(int(item["claim_count"]) for item in observed)
    repair_attempts = sum(int(item["repair_attempted"]) for item in observed)
    durations = [
        float(item["duration_ms"])
        for item in observed
        if item.get("duration_ms") is not None
    ]
    return {
        "claim_audit_observable_rate": (
            len(observed) / len(faithfulness_rows) if faithfulness_rows else None
        ),
        "claim_support_rate": _ratio(
            sum(int(item["supported"]) for item in observed), total_claims
        ),
        "citation_coverage": _ratio(
            sum(int(item["cited"]) for item in observed), total_claims
        ),
        "unsupported_claim_rate": _ratio(
            sum(int(item["unsupported"]) for item in observed), total_claims
        ),
        "insufficient_claim_rate": _ratio(
            sum(int(item["insufficient"]) for item in observed), total_claims
        ),
        "repair_success_rate": _ratio(
            sum(int(item["repair_succeeded"]) for item in observed), repair_attempts
        ),
        "claim_verifier_mean_duration_ms": mean(durations) if durations else None,
    }


# 汇总一个类型或分层中的全部质量指标。
def _group_metrics(items: List[dict]) -> dict[str, Any]:
    metrics = {
        "count": len(items),
        "router_rule_accuracy": mean_metric(items, "router_rule_accuracy"),
        "citation_accuracy": mean_metric(items, "citation_accuracy"),
        "faithfulness_manual_support_rate": mean_metric(
            items, "faithfulness_manual_support_rate"
        ),
    }
    metrics.update(claim_audit_diagnostics(items))
    return metrics


# 生成摘要。
def summarize(rows: List[dict]) -> dict:
    by_type = group_rows(rows, "case_type")
    by_layer = group_rows(rows, "layer")

    aggregate = {
        "router_rule_accuracy": mean_metric(
            by_type.get("router", []), "router_rule_accuracy"
        ),
        "citation_accuracy": mean_metric(
            by_type.get("citation", []), "citation_accuracy"
        ),
        "faithfulness_manual_support_rate": mean_metric(
            by_type.get("faithfulness", []), "faithfulness_manual_support_rate"
        ),
    }
    aggregate.update(claim_audit_diagnostics(by_type.get("faithfulness", [])))

    return {
        "aggregate": aggregate,
        "baseline_gated_metrics": list(BASELINE_GATED_METRICS),
        "by_case_type": {
            case_type: _group_metrics(items)
            for case_type, items in sorted(by_type.items())
        },
        "by_layer": {
            layer: _group_metrics(items)
            for layer, items in sorted(by_layer.items())
        },
    }


# 运行 eval。
def run_eval(items: List[dict]) -> dict:
    rows = [evaluate_case(item) for item in items]
    summary = summarize(rows)
    return {
        "config": {"num_cases": len(items)},
        "aggregate": summary["aggregate"],
        "baseline_gated_metrics": summary["baseline_gated_metrics"],
        "by_case_type": summary["by_case_type"],
        "by_layer": summary["by_layer"],
        "rows": rows,
    }


# 审计评测集覆盖面。
def audit_coverage(items: List[dict]) -> dict:
    case_types = {str(item.get("case_type", "")) for item in items}
    layers = {str(item.get("layer", "unspecified")) for item in items}
    missing_case_types = [
        case_type for case_type in REQUIRED_CASE_TYPES if case_type not in case_types
    ]
    missing_layers = [layer for layer in RECOMMENDED_LAYERS if layer not in layers]
    return {
        "case_types": sorted(case_types),
        "layers": sorted(layers),
        "missing_case_types": missing_case_types,
        "missing_layers": missing_layers,
        "is_coverage_complete": not missing_case_types and not missing_layers,
    }


# 生成对比 baseline。
def compare_baseline(report: dict, baseline_path: Path) -> int:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    base_agg = baseline.get("aggregate", {})
    cur_agg = report["aggregate"]
    gated_metrics = tuple(report.get("baseline_gated_metrics", BASELINE_GATED_METRICS))
    print(f"\n对比基线 {baseline_path}:")
    regressed = False
    for key in sorted(cur_agg):
        cur = cur_agg[key]
        gated = key in gated_metrics
        if cur is None:
            print(f"  {key:<34} -  (当前报告缺该指标)")
            if gated:
                regressed = True
            continue
        base = base_agg.get(key)
        if base is None:
            print(f"  {key:<34} {cur:.4f}  (基线缺该指标)")
            continue
        delta = cur - base
        flag = ""
        if gated and delta < -1e-9:
            flag = "  回退"
            regressed = True
        elif delta > 1e-9:
            flag = "  提升"
        elif not gated:
            flag = "  人工台账"
        print(f"  {key:<34} {cur:.4f}  (基线 {base:.4f}, Δ{delta:+.4f}){flag}")
    print()
    return 1 if regressed else 0
