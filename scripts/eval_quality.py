import argparse
import json
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    # 包源码在源码目录下，项目根目录用于解析数据文件相对路径。
    sys.path.insert(0, str(ROOT / "src"))

from cogdoc.config.settings import get_settings
from cogdoc.tools.eval.quality_metrics import audit_coverage, compare_baseline, run_eval


# 返回项目根目录路径。
def _project_path(path: str) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else ROOT / resolved


_settings = get_settings()
DEFAULT_EVAL_SET = _project_path(_settings.quality_eval_set_path)
EXAMPLE_EVAL_SET = _project_path(_settings.quality_eval_example_set_path)


# 解析默认评测集。
def resolve_default_eval_set() -> Path:
    if DEFAULT_EVAL_SET.exists():
        return DEFAULT_EVAL_SET
    print(
        f"未找到本地质量评测集 {DEFAULT_EVAL_SET}，回退到示例 {EXAMPLE_EVAL_SET.name}。\n"
        f"提示：复制为 {DEFAULT_EVAL_SET.name} 并填入真实标注样本后再跑，结果才有意义。"
    )
    return EXAMPLE_EVAL_SET


# 加载评测集。
def load_eval_set(path: Path) -> List[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


# 格式化评测指标数值。
def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


# 输出报告。
def print_report(report: dict) -> None:
    print(f"\n质量评测  |  cases={report['config']['num_cases']}\n")
    print("聚合:")
    for key, value in report["aggregate"].items():
        print(f"  {key:<34} {_fmt(value)}")
    print("\n按类型:")
    for key, value in report["by_case_type"].items():
        print(
            f"  {key:<14} count={value['count']} "
            f"router_rule={_fmt(value['router_rule_accuracy'])} "
            f"citation={_fmt(value['citation_accuracy'])} "
            f"manual_support={_fmt(value['faithfulness_manual_support_rate'])}"
        )
    print("\n按层级:")
    for key, value in report["by_layer"].items():
        print(
            f"  {key:<14} count={value['count']} "
            f"router_rule={_fmt(value['router_rule_accuracy'])} "
            f"citation={_fmt(value['citation_accuracy'])} "
            f"manual_support={_fmt(value['faithfulness_manual_support_rate'])}"
        )
    print(
        "\n说明：faithfulness_manual_support_rate 是人工抽检台账，不作为自动回归门禁。"
    )
    print()


# 输出覆盖审计结果。
def print_coverage(coverage: dict) -> None:
    print("\n覆盖审计:")
    print(f"  case_types={coverage['case_types']}")
    print(f"  layers={coverage['layers']}")
    if coverage["missing_case_types"]:
        print(f"  缺少 case_type: {coverage['missing_case_types']}")
    if coverage["missing_layers"]:
        print(f"  缺少 layer: {coverage['missing_layers']}")
    if coverage["is_coverage_complete"]:
        print("  覆盖完整")
    print()


# 启动入口。
def main() -> int:
    parser = argparse.ArgumentParser(description="离线质量评测 harness")
    parser.add_argument("--eval-set", type=Path, default=None, help="质量评测 JSONL")
    parser.add_argument("--json", type=Path, default=None, help="把报告写入 JSON 文件")
    parser.add_argument("--baseline", type=Path, default=None, help="与基线 JSON 对比")
    parser.add_argument(
        "--check-coverage",
        action="store_true",
        help="检查评测集是否覆盖必要 case_type 和推荐 layer",
    )
    parser.add_argument(
        "--coverage-only",
        action="store_true",
        help="只检查评测集覆盖面，不运行质量评测",
    )
    args = parser.parse_args()
    if args.coverage_only and (args.check_coverage or args.json or args.baseline):
        parser.error(
            "--coverage-only 不能与 --check-coverage、--json 或 --baseline 同时使用"
        )

    eval_set = args.eval_set or resolve_default_eval_set()
    items = load_eval_set(eval_set)
    if not items:
        print(f"评测集为空: {eval_set}")
        return 1

    coverage = audit_coverage(items)
    if args.coverage_only:
        print_coverage(coverage)
        return 0 if coverage["is_coverage_complete"] else 1

    report = run_eval(items)
    print_report(report)
    if args.check_coverage:
        report["coverage"] = coverage
        print_coverage(coverage)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"报告已写入 {args.json}")

    if args.baseline:
        baseline_status = compare_baseline(report, args.baseline)
        if args.check_coverage and not coverage["is_coverage_complete"]:
            return 1
        return baseline_status
    if args.check_coverage and not coverage["is_coverage_complete"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
