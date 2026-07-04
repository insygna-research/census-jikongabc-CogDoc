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
    retrieval = report.get("retrieval_report")
    if retrieval and not retrieval.get("skipped"):
        print("\n检索指标:")
        for key, value in retrieval["aggregate"].items():
            print(f"  {key:<12} {value:.4f}")
    elif retrieval:
        print(f"\n检索指标: 已跳过（{retrieval['reason']}）")
    print()


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
        quality_result["regressed"] or retrieval_result.get("regressed", False)
    )
    return {
        "baseline_path": str(baseline_path),
        "regressed": regressed,
        "quality": quality_result,
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
            if row["delta"] is None:
                print(f"    {row['metric']:<34} 缺少当前值或基线值")
                continue
            print(
                f"    {row['metric']:<34} "
                f"{row['current']:.4f} "
                f"(基线 {row['baseline']:.4f}, Δ{row['delta']:+.4f}) "
                f"{row['status']}"
            )
    print(f"  结果: {'回退' if result['regressed'] else '通过'}")
    print()


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
