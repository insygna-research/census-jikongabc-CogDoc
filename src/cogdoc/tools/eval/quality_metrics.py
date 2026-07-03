import json
from pathlib import Path
from statistics import mean
from typing import Dict, List
from cogdoc.agents.citation_validator import CitationValidatorAgent
from cogdoc.agents.router import classify_intent_by_rule


BASELINE_GATED_METRICS = ("router_rule_accuracy", "citation_accuracy")


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


# 评估faithfulness用例。
def evaluate_faithfulness_case(item: dict) -> dict:
    is_faithful = bool(item["is_faithful"])
    return {
        "case_type": "faithfulness",
        "layer": item.get("layer", "unspecified"),
        "expected": True,
        "actual": is_faithful,
        "metrics": {"faithfulness_manual_support_rate": accuracy(is_faithful)},
        "manual_only": True,
    }


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

    return {
        "aggregate": aggregate,
        "baseline_gated_metrics": list(BASELINE_GATED_METRICS),
        "by_case_type": {
            case_type: {
                "count": len(items),
                "router_rule_accuracy": mean_metric(items, "router_rule_accuracy"),
                "citation_accuracy": mean_metric(items, "citation_accuracy"),
                "faithfulness_manual_support_rate": mean_metric(
                    items, "faithfulness_manual_support_rate"
                ),
            }
            for case_type, items in sorted(by_type.items())
        },
        "by_layer": {
            layer: {
                "count": len(items),
                "router_rule_accuracy": mean_metric(items, "router_rule_accuracy"),
                "citation_accuracy": mean_metric(items, "citation_accuracy"),
                "faithfulness_manual_support_rate": mean_metric(
                    items, "faithfulness_manual_support_rate"
                ),
            }
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
