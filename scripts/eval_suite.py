import argparse
import json
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
    parser.add_argument("--json", type=Path, default=None, help="把组合报告写入 JSON")
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

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"组合报告已写入 {args.json}")

    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
