import argparse
import json
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from tools.eval.quality_metrics import compare_baseline, run_eval


def _project_path(path: str) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else ROOT / resolved


_settings = get_settings()
DEFAULT_EVAL_SET = _project_path(_settings.quality_eval_set_path)
EXAMPLE_EVAL_SET = _project_path(_settings.quality_eval_example_set_path)


def resolve_default_eval_set() -> Path:
    if DEFAULT_EVAL_SET.exists():
        return DEFAULT_EVAL_SET
    print(
        f"未找到本地质量评测集 {DEFAULT_EVAL_SET}，回退到示例 {EXAMPLE_EVAL_SET.name}。\n"
        f"提示：复制为 {DEFAULT_EVAL_SET.name} 并填入真实标注样本后再跑，结果才有意义。"
    )
    return EXAMPLE_EVAL_SET


def load_eval_set(path: Path) -> List[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


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
    print("\n说明：faithfulness_manual_support_rate 是人工抽检台账，不作为自动回归门禁。")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="离线质量评测 harness")
    parser.add_argument("--eval-set", type=Path, default=None, help="质量评测 JSONL")
    parser.add_argument("--json", type=Path, default=None, help="把报告写入 JSON 文件")
    parser.add_argument("--baseline", type=Path, default=None, help="与基线 JSON 对比")
    args = parser.parse_args()

    eval_set = args.eval_set or resolve_default_eval_set()
    items = load_eval_set(eval_set)
    if not items:
        print(f"评测集为空: {eval_set}")
        return 1

    report = run_eval(items)
    print_report(report)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"报告已写入 {args.json}")

    if args.baseline:
        return compare_baseline(report, args.baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
