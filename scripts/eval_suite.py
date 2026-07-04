import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # 项目根目录用于导入脚本目录下的同级模块。
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    # 包源码在源码目录下，项目根目录用于解析数据文件相对路径。
    sys.path.insert(0, str(ROOT / "src"))

from cogdoc.tools.eval.quality_metrics import (
    audit_coverage as audit_quality_coverage,
    run_eval as run_quality_eval,
)
from cogdoc.tools.eval.retrieval_metrics import (
    audit_coverage as audit_retrieval_coverage,
)
from scripts import eval_quality, eval_retrieval

DEFAULT_REPORT_PATH = ROOT / "eval" / "eval_suite_report.json"
DEFAULT_BASELINE_PATH = ROOT / "eval" / "eval_suite_baseline.json"
QUALITY_CASE_TYPE_ORDER = ("router", "citation", "faithfulness")
QUALITY_LAYER_ORDER = ("easy", "hard", "no-answer", "compare", "multi-turn")


# 构建覆盖门禁结果。
def build_gate(
    quality_coverage: Dict[str, Any], retrieval_coverage: Dict[str, Any]
) -> Dict[str, Any]:
    checks = {
        "quality_coverage": bool(quality_coverage["is_coverage_complete"]),
        "retrieval_coverage": bool(retrieval_coverage["is_coverage_complete"]),
    }
    return {"checks": checks, "passed": all(checks.values())}


# 输出组合报告摘要。
def print_summary(report: Dict[str, Any]) -> None:
    gate = report["gate"]
    print("\n评测门禁:")
    for name, passed in gate["checks"].items():
        status = "通过" if passed else "失败"
        print(f"  {name:<20} {status}")
    print(f"\n整体结果: {'通过' if gate['passed'] else '失败'}")
    quality = report.get("quality_report")
    if quality:
        print("\n质量指标:")
        for key, value in quality["aggregate"].items():
            shown = "-" if value is None else f"{value:.4f}"
            print(f"  {key:<34} {shown}")
        print_quality_groups(
            "质量类型", report.get("quality_case_types", []), "case_type"
        )
        print_quality_groups("质量分层", report.get("quality_layers", []), "layer")
    retrieval = report.get("retrieval_report")
    if retrieval and not retrieval.get("skipped"):
        print("\n检索指标:")
        for key, value in retrieval["aggregate"].items():
            print(f"  {key:<12} {value:.4f}")
    elif retrieval:
        print(f"\n检索指标: 已跳过（{retrieval['reason']}）")
    print()


# 输出质量分组摘要。
def print_quality_groups(
    title: str, groups: List[Dict[str, Any]], name_key: str
) -> None:
    if not groups:
        return
    print(f"\n{title}:")
    for group in groups:
        metrics = "  ".join(
            f"{metric}={format_metric(group['metrics'].get(metric))}"
            for metric in group["gated_metrics"]
        )
        print(f"  {group[name_key]:<14} count={group['count']}  {metrics}")


# 格式化指标值。
def format_metric(value: Any) -> str:
    return "-" if value is None else f"{value:.4f}"


# 写入组合评测报告。
def write_report(report: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(tmp_path, path)


# 比较单组指标。
def compare_metric_group(
    current: Dict[str, Any],
    baseline: Dict[str, Any],
    metric_names: List[str],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    regressed = False
    for name in metric_names:
        cur = current.get(name)
        base = baseline.get(name)
        if cur is None or base is None:
            rows.append(
                {
                    "metric": name,
                    "current": cur,
                    "baseline": base,
                    "delta": None,
                    "status": "missing",
                }
            )
            regressed = True
            continue
        delta = cur - base
        status = "same"
        if delta < -1e-9:
            status = "regressed"
            regressed = True
        elif delta > 1e-9:
            status = "improved"
        rows.append(
            {
                "metric": name,
                "current": cur,
                "baseline": base,
                "delta": delta,
                "status": status,
            }
        )
    return {"regressed": regressed, "rows": rows}


# 解析可对比门禁指标。
def comparable_gated_metrics(
    current_quality: Dict[str, Any], baseline_quality: Dict[str, Any]
) -> List[str]:
    current_metrics = list(current_quality.get("baseline_gated_metrics", []))
    baseline_metrics = baseline_quality.get("baseline_gated_metrics")
    baseline_aggregate = baseline_quality.get("aggregate", {})
    if baseline_metrics is None:
        return [metric for metric in current_metrics if metric in baseline_aggregate]
    baseline_set = set(baseline_metrics)
    return [metric for metric in current_metrics if metric in baseline_set]


# 构建质量类型摘要。
def build_quality_case_type_summary(
    quality_report: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return build_quality_group_summary(
        quality_report, "by_case_type", "case_type", QUALITY_CASE_TYPE_ORDER
    )


# 构建质量分层摘要。
def build_quality_layer_summary(
    quality_report: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return build_quality_group_summary(
        quality_report, "by_layer", "layer", QUALITY_LAYER_ORDER
    )


# 构建质量分组摘要。
def build_quality_group_summary(
    quality_report: Dict[str, Any],
    source_key: str,
    name_key: str,
    preferred_order: tuple[str, ...],
) -> List[Dict[str, Any]]:
    groups = quality_report.get(source_key, {})
    gated_metrics = list(quality_report.get("baseline_gated_metrics", []))
    ordered_names = [
        name for name in preferred_order if name in groups
    ] + sorted(name for name in groups if name not in preferred_order)
    return [
        {
            name_key: name,
            "count": groups[name]["count"],
            "gated_metrics": gated_metrics,
            "metrics": {
                metric: groups[name].get(metric) for metric in gated_metrics
            },
        }
        for name in ordered_names
    ]


# 比较质量分组基线。
def compare_quality_groups(
    current_groups: List[Dict[str, Any]],
    baseline_groups: List[Dict[str, Any]],
    name_key: str,
) -> Dict[str, Any]:
    baseline_by_name = {row[name_key]: row for row in baseline_groups}
    rows: List[Dict[str, Any]] = []
    regressed = False
    for current in current_groups:
        baseline = baseline_by_name.get(current[name_key])
        if baseline is None:
            continue
        metrics = comparable_quality_group_metrics(current, baseline)
        if not metrics:
            continue
        result = compare_metric_group(
            current.get("metrics", {}),
            baseline.get("metrics", {}),
            metrics,
        )
        if result["regressed"]:
            regressed = True
        rows.append({name_key: current[name_key], **result})
    return {"regressed": regressed, "rows": rows}


# 比较质量分层基线。
def compare_quality_layers(
    current_layers: List[Dict[str, Any]], baseline_layers: List[Dict[str, Any]]
) -> Dict[str, Any]:
    return compare_quality_groups(current_layers, baseline_layers, "layer")


# 比较质量类型基线。
def compare_quality_case_types(
    current_case_types: List[Dict[str, Any]],
    baseline_case_types: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return compare_quality_groups(current_case_types, baseline_case_types, "case_type")


# 解析可对比质量分组指标。
def comparable_quality_group_metrics(
    current_group: Dict[str, Any], baseline_group: Dict[str, Any]
) -> List[str]:
    current_metrics = list(current_group.get("gated_metrics", []))
    baseline_metrics = baseline_group.get("gated_metrics")
    current_values = current_group.get("metrics", {})
    baseline_values = baseline_group.get("metrics", {})
    if baseline_metrics is None:
        return [
            metric
            for metric in current_metrics
            if current_values.get(metric) is not None
            and baseline_values.get(metric) is not None
        ]
    baseline_set = set(baseline_metrics)
    return [
        metric
        for metric in current_metrics
        if metric in baseline_set
        and current_values.get(metric) is not None
        and baseline_values.get(metric) is not None
    ]


# 比较组合评测基线。
def compare_baseline(report: Dict[str, Any], baseline_path: Path) -> Dict[str, Any]:
    if not baseline_path.exists():
        raise ValueError(f"组合基线不存在: {baseline_path}")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    quality = report["quality_report"]
    base_quality = baseline.get("quality_report", {})
    quality_metrics = comparable_gated_metrics(quality, base_quality)
    quality_result = compare_metric_group(
        quality.get("aggregate", {}),
        base_quality.get("aggregate", {}),
        quality_metrics,
    )
    case_type_result = compare_quality_case_types(
        report.get("quality_case_types", []), baseline.get("quality_case_types", [])
    )
    layer_result = compare_quality_layers(
        report.get("quality_layers", []), baseline.get("quality_layers", [])
    )

    retrieval = report["retrieval_report"]
    retrieval_result: Dict[str, Any]
    if retrieval.get("skipped"):
        retrieval_result = {"skipped": True, "regressed": False, "rows": []}
    else:
        base_retrieval = baseline.get("retrieval_report", {})
        retrieval_metrics = sorted(retrieval.get("aggregate", {}).keys())
        retrieval_result = compare_metric_group(
            retrieval.get("aggregate", {}),
            base_retrieval.get("aggregate", {}),
            retrieval_metrics,
        )
        retrieval_result["skipped"] = False

    regressed = bool(
        quality_result["regressed"]
        or case_type_result["regressed"]
        or layer_result["regressed"]
        or retrieval_result.get("regressed", False)
    )
    return {
        "baseline_path": str(baseline_path),
        "regressed": regressed,
        "quality": quality_result,
        "quality_case_types": case_type_result,
        "quality_layers": layer_result,
        "retrieval": retrieval_result,
    }


# 输出基线对比摘要。
def print_baseline(result: Dict[str, Any]) -> None:
    print(f"\n对比组合基线 {result['baseline_path']}:")
    for section_name in ("quality", "retrieval"):
        section = result[section_name]
        title = "质量指标" if section_name == "quality" else "检索指标"
        if section.get("skipped"):
            print(f"  {title}: 已跳过")
            continue
        print(f"  {title}:")
        for row in section["rows"]:
            print_baseline_row(row, "    ")
    case_type_section = result.get("quality_case_types", {})
    if case_type_section.get("rows"):
        print("  质量类型:")
        for case_type in case_type_section["rows"]:
            print(f"    {case_type['case_type']}:")
            for row in case_type["rows"]:
                print_baseline_row(row, "      ")
    layer_section = result.get("quality_layers", {})
    if layer_section.get("rows"):
        print("  质量分层:")
        for layer in layer_section["rows"]:
            print(f"    {layer['layer']}:")
            for row in layer["rows"]:
                print_baseline_row(row, "      ")
    print(f"  结果: {'回退' if result['regressed'] else '通过'}")
    print()


# 输出基线对比行。
def print_baseline_row(row: Dict[str, Any], prefix: str) -> None:
    if row["delta"] is None:
        print(f"{prefix}{row['metric']:<34} 缺少当前值或基线值")
        return
    print(
        f"{prefix}{row['metric']:<34} "
        f"{row['current']:.4f} "
        f"(基线 {row['baseline']:.4f}, Δ{row['delta']:+.4f}) "
        f"{row['status']}"
    )


# 构建组合评测报告。
def build_report(
    quality_eval_set: Path,
    retrieval_eval_set: Path,
    run_retrieval: bool,
    k_values: List[int],
    rerank: bool,
) -> Dict[str, Any]:
    quality_items = eval_quality.load_eval_set(quality_eval_set)
    retrieval_items = eval_retrieval.load_eval_set(retrieval_eval_set)
    if not quality_items:
        raise ValueError(f"质量评测集为空: {quality_eval_set}")
    if not retrieval_items:
        raise ValueError(f"检索评测集为空: {retrieval_eval_set}")

    quality_coverage = audit_quality_coverage(quality_items)
    retrieval_coverage = audit_retrieval_coverage(retrieval_items)
    quality_report = run_quality_eval(quality_items)
    quality_report["coverage"] = quality_coverage
    quality_case_types = build_quality_case_type_summary(quality_report)
    quality_layers = build_quality_layer_summary(quality_report)

    retrieval_report: Dict[str, Any]
    if run_retrieval:
        retrieval_report = eval_retrieval.run_eval(
            retrieval_items, sorted(k_values), rerank
        )
        retrieval_report["coverage"] = retrieval_coverage
    else:
        retrieval_report = {
            "skipped": True,
            "reason": "默认只跑覆盖门禁；需要真实检索指标时加 --run-retrieval",
            "coverage": retrieval_coverage,
        }

    return {
        "config": {
            "quality_eval_set": str(quality_eval_set),
            "retrieval_eval_set": str(retrieval_eval_set),
            "run_retrieval": run_retrieval,
            "k_values": sorted(k_values),
            "rerank": rerank,
        },
        "gate": build_gate(quality_coverage, retrieval_coverage),
        "quality_report": quality_report,
        "quality_case_types": quality_case_types,
        "quality_layers": quality_layers,
        "retrieval_report": retrieval_report,
    }


# 启动入口。
def main() -> int:
    parser = argparse.ArgumentParser(description="组合离线评测门禁")
    parser.add_argument("--quality-eval-set", type=Path, default=None)
    parser.add_argument("--retrieval-eval-set", type=Path, default=None)
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help=f"把组合报告写入指定路径，默认不写；推荐路径 {DEFAULT_REPORT_PATH}",
    )
    parser.add_argument("--baseline", type=Path, default=None, help="与组合基线对比")
    parser.add_argument(
        "--update-baseline",
        type=Path,
        nargs="?",
        const=DEFAULT_BASELINE_PATH,
        default=None,
        help=f"用当前通过的组合报告更新基线，默认路径 {DEFAULT_BASELINE_PATH}",
    )
    parser.add_argument(
        "--run-retrieval",
        action="store_true",
        help="同时执行真实检索评测；默认只检查检索评测集覆盖",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=eval_retrieval.DEFAULT_K_VALUES,
        help="真实检索评测的 k 截断值",
    )
    parser.add_argument("--rerank", action="store_true", help="真实检索评测时启用精排")
    args = parser.parse_args()
    if args.baseline and args.update_baseline:
        parser.error("--baseline 不能与 --update-baseline 同时使用")

    quality_eval_set = args.quality_eval_set or eval_quality.resolve_default_eval_set()
    retrieval_eval_set = (
        args.retrieval_eval_set or eval_retrieval.resolve_default_eval_set()
    )
    report = build_report(
        quality_eval_set=quality_eval_set,
        retrieval_eval_set=retrieval_eval_set,
        run_retrieval=args.run_retrieval,
        k_values=args.k,
        rerank=args.rerank,
    )
    print_summary(report)
    baseline_result = None
    if args.baseline:
        try:
            baseline_result = compare_baseline(report, args.baseline)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        report["baseline"] = baseline_result
        print_baseline(baseline_result)

    if args.json:
        write_report(report, args.json)
        print(f"组合报告已写入 {args.json}")

    if not report["gate"]["passed"]:
        return 1
    if baseline_result and baseline_result["regressed"]:
        return 1
    if args.update_baseline:
        write_report(report, args.update_baseline)
        print(f"组合基线已更新 {args.update_baseline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
