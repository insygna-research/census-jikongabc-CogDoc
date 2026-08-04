"""运行通用 Agent 评测集。

JSONL 每行一个 Case：
{"case_id":"qa-1", "evaluators":[...], "trials":[{...},{...}]}
Trial 至少包含 execution_status、agent_output；RAGAS/LLM Judge 还应提供
case_input、expected、retrieved_context、citations 和 tool_trace 等证据。
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cogdoc.tools.eval.scoring import (
    aggregate_case,
    aggregate_run,
    evaluate_trial,
    judge_from_settings,
)


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="通用 Agent Trial/Case/Run 评测")
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--offline", action="store_true", help="不调用云端 LLM Judge")
    args = parser.parse_args()

    judge = None if args.offline else judge_from_settings()
    cases = []
    for raw in load_cases(args.eval_set):
        evaluators = raw.get("evaluators", [])
        trial_reports = [
            evaluate_trial(trial, evaluators, judge=judge)
            for trial in raw.get("trials", [])
        ]
        case_report = aggregate_case(trial_reports)
        case_report.update(
            {
                "case_id": raw.get("case_id", ""),
                "agent_type": raw.get("agent_type", "general"),
                "trial_reports": trial_reports,
            }
        )
        cases.append(case_report)

    report = {"cases": cases, "run": aggregate_run(cases), "judge_enabled": judge is not None}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["run"], ensure_ascii=False, indent=2))
    return 0 if report["run"].get("decision") != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
