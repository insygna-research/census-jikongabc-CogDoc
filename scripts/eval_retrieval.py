import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    # src-layout：包源码在 src/ 下；ROOT 仍用于解析数据文件相对路径。
    sys.path.insert(0, str(ROOT / "src"))

from cogdoc.config.settings import get_settings
from cogdoc.graph.subgraphs.qa import RetrieverFactory
from cogdoc.tools.eval.retrieval_metrics import aggregate, evaluate_query


# 返回项目根目录路径。
def _project_path(path: str) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else ROOT / resolved


_settings = get_settings()
DEFAULT_EVAL_SET = _project_path(_settings.eval_set_path)
# 真实评测集不入库；clean checkout 回退到 example，保证零参数命令可运行。
EXAMPLE_EVAL_SET = _project_path(_settings.eval_example_set_path)
DEFAULT_K_VALUES = [1, 3, 5, 9]


# 解析 default eval set。
def resolve_default_eval_set() -> Path:
    if DEFAULT_EVAL_SET.exists():
        return DEFAULT_EVAL_SET
    print(
        f"未找到本地评测集 {DEFAULT_EVAL_SET}，回退到示例 {EXAMPLE_EVAL_SET.name}。\n"
        f"提示：复制为 {DEFAULT_EVAL_SET.name} 并按你的真实语料填写后再跑，结果才有意义。"
    )
    return EXAMPLE_EVAL_SET


# 加载 eval set。
def load_eval_set(path: Path) -> List[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


# 检索 sources。
def retrieve_sources(query: str, doc_id: str, top_k: int, rerank: bool) -> List[str]:
    engine = RetrieverFactory.get_engine(doc_id)
    docs = engine.search(query=query, top_k=top_k)
    if rerank and docs:
        from cogdoc.tools.reranker import BGEReranker

        docs = BGEReranker.rerank(query=query, docs=docs, top_n=len(docs))
    return [doc["meta"]["source"] for doc in docs]


# 运行 eval。
def run_eval(items: List[dict], k_values: List[int], rerank: bool) -> dict:
    top_k = max(k_values)
    rows: List[dict] = []
    per_query_metrics: List[Dict[str, float]] = []

    for item in items:
        retrieved = retrieve_sources(
            item["query"], item.get("doc_id", "default"), top_k, rerank
        )
        metrics = evaluate_query(retrieved, item["expected_sources"], k_values)
        per_query_metrics.append(metrics)
        rows.append(
            {
                "query": item["query"],
                "expected_sources": item["expected_sources"],
                "retrieved_sources": retrieved,
                "metrics": metrics,
            }
        )

    return {
        "config": {"k_values": k_values, "rerank": rerank, "num_queries": len(items)},
        "aggregate": aggregate(per_query_metrics),
        "rows": rows,
    }


# 输出 report。
def print_report(report: dict) -> None:
    cfg = report["config"]
    print(
        f"\n检索评测  |  queries={cfg['num_queries']}  rerank={cfg['rerank']}  k={cfg['k_values']}\n"
    )

    for row in report["rows"]:
        recalls = "  ".join(
            f"r@{k}={row['metrics'][f'recall@{k}']:.2f}" for k in cfg["k_values"]
        )
        print(f"  [{row['metrics']['mrr']:.2f} MRR] {recalls}  | {row['query']}")
        print(f"        expected={row['expected_sources']}")
        print(f"        top={row['retrieved_sources'][: max(cfg['k_values'])]}")

    print("\n聚合:")
    for key, value in report["aggregate"].items():
        print(f"  {key:<12} {value:.4f}")
    print()


# 生成对比 baseline。
def compare_baseline(report: dict, baseline_path: Path) -> int:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    base_agg = baseline.get("aggregate", {})
    cur_agg = report["aggregate"]
    print(f"\n对比基线 {baseline_path}:")
    regressed = False
    for key in sorted(cur_agg):
        cur = cur_agg[key]
        base = base_agg.get(key)
        if base is None:
            print(f"  {key:<12} {cur:.4f}  (基线缺该指标)")
            continue
        delta = cur - base
        flag = ""
        if delta < -1e-9:
            flag = "  ⚠ 回退"
            regressed = True
        elif delta > 1e-9:
            flag = "  ✅ 提升"
        print(f"  {key:<12} {cur:.4f}  (基线 {base:.4f}, Δ{delta:+.4f}){flag}")
    print()
    return 1 if regressed else 0


# 启动入口。
def main() -> int:
    parser = argparse.ArgumentParser(description="离线检索评测 harness")
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=None,
        help="评测集 JSONL；缺省用本地 retrieval_eval.jsonl，没有则回退 example",
    )
    parser.add_argument("--rerank", action="store_true", help="在检索后加 BGE 精排")
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=DEFAULT_K_VALUES,
        help="recall/hit 的 k 截断值",
    )
    parser.add_argument("--json", type=Path, default=None, help="把报告写入 JSON 文件")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="与基线 JSON 报告对比，回退则退出码非零",
    )
    args = parser.parse_args()

    eval_set = args.eval_set or resolve_default_eval_set()
    items = load_eval_set(eval_set)
    if not items:
        print(f"评测集为空: {eval_set}")
        return 1

    report = run_eval(items, sorted(args.k), args.rerank)
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
