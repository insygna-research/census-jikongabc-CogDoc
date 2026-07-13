import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    # 包源码在源码目录下，项目根目录用于解析数据文件相对路径。
    sys.path.insert(0, str(ROOT / "src"))

from cogdoc.config.settings import get_settings
from cogdoc.tools.eval.retrieval_metrics import (
    aggregate,
    audit_coverage,
    coverage_minimums,
    evaluate_query,
    evaluate_thresholds,
    infer_retrieval_layer,
    metric_direction,
    percentile,
)


# 返回项目根目录路径。
def _project_path(path: str) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else ROOT / resolved


_settings = get_settings()
DEFAULT_EVAL_SET = _project_path(_settings.eval_set_path)
# 真实评测集不入库，干净检出时回退到示例评测集。
EXAMPLE_EVAL_SET = _project_path(_settings.eval_example_set_path)
DEFAULT_K_VALUES = [1, 3, 5, 9]


# 解析默认评测集。
def resolve_default_eval_set() -> Path:
    if DEFAULT_EVAL_SET.exists():
        return DEFAULT_EVAL_SET
    print(
        f"未找到本地评测集 {DEFAULT_EVAL_SET}，回退到示例 {EXAMPLE_EVAL_SET.name}。\n"
        f"提示：复制为 {DEFAULT_EVAL_SET.name} 并按你的真实语料填写后再跑，结果才有意义。"
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


# 检索来源。
def retrieve_sources(query: str, doc_id: str, top_k: int, rerank: bool) -> List[str]:
    return retrieve_result(query, doc_id, top_k, rerank)["sources"]


# 执行检索并用线上同一套规则判断是否有足够证据。
def retrieve_result(
    query: str,
    doc_id: str,
    top_k: int,
    rerank: bool,
    *,
    verify_evidence: bool = False,
    is_local_verifier: bool = False,
) -> dict:
    from cogdoc.graph.subgraphs.qa import RetrieverFactory
    from cogdoc.tools.retriever.confidence import assess_retrieval_support

    engine = RetrieverFactory.get_engine(doc_id)
    docs = engine.search(query=query, top_k=top_k)
    if rerank and docs:
        from cogdoc.tools.reranker import BGEReranker

        docs = BGEReranker.rerank(query=query, docs=docs, top_n=len(docs))
    support = assess_retrieval_support(docs)
    result = {
        "sources": [doc["meta"]["source"] for doc in docs],
        "supported": support.supported,
        "first_stage_supported": support.supported,
        "confidence": support.score,
        "reason": support.reason,
        "signals": support.signals,
        "evidence_verification_required": False,
        "evidence_supported": support.supported,
        "evidence_verification_reason": "not_requested",
        "evidence_verified_chunk_ids": [],
    }
    if not verify_evidence:
        return result

    from cogdoc.agents.evidence_verifier import (
        EvidenceVerifierAgent,
        select_verification_docs,
        should_verify_evidence,
    )

    settings = get_settings()
    verify_state = {
        "query": query,
        "is_local": is_local_verifier,
        "retrieval_first_stage_supported": support.supported,
        "retrieval_abstained": not support.supported,
        "retrieval_abstain_reason": support.reason,
        "retrieval_confidence": support.score,
        "verification_docs": select_verification_docs(
            docs, settings.qa_evidence_verify_max_docs
        ),
    }
    if not should_verify_evidence(verify_state, settings):
        result["evidence_verification_reason"] = "not_required"
        return result

    verification = EvidenceVerifierAgent.verify(verify_state)
    result.update(
        {
            "supported": bool(verification.get("evidence_supported")),
            "reason": str(
                verification.get("retrieval_abstain_reason") or support.reason
            ),
            "evidence_verification_required": True,
            "evidence_supported": bool(verification.get("evidence_supported")),
            "evidence_verification_reason": str(
                verification.get("evidence_verification_reason") or ""
            ),
            "evidence_verified_chunk_ids": list(
                verification.get("evidence_verified_chunk_ids") or []
            ),
            "evidence_verifier_error": str(
                verification.get("evidence_verifier_error") or ""
            ),
        }
    )
    return result


# 运行评测。
def run_eval(
    items: List[dict],
    k_values: List[int],
    rerank: bool,
    verify_evidence: bool = False,
    is_local_verifier: bool = False,
) -> dict:
    top_k = max(k_values)
    rows: List[dict] = []

    # 模型加载、设备选择和首轮内核初始化单独计时，不污染稳态请求 P95。
    warmup_item = items[0]
    if verify_evidence:
        from cogdoc.agents.evidence_verifier import requires_evidence_verification

        warmup_item = next(
            (
                item
                for item in items
                if requires_evidence_verification(str(item.get("query") or ""))
            ),
            items[0],
        )
    warmup_started = time.perf_counter()
    retrieve_result(
        warmup_item["query"],
        warmup_item.get("doc_id", "default"),
        top_k,
        rerank,
        verify_evidence=verify_evidence,
        is_local_verifier=is_local_verifier,
    )
    warmup_latency_ms = (time.perf_counter() - warmup_started) * 1000.0

    for item in items:
        started = time.perf_counter()
        retrieval_result = retrieve_result(
            item["query"],
            item.get("doc_id", "default"),
            top_k,
            rerank,
            verify_evidence=verify_evidence,
            is_local_verifier=is_local_verifier,
        )
        retrieved = retrieval_result["sources"]
        latency_ms = (time.perf_counter() - started) * 1000.0
        metrics = evaluate_query(retrieved, item["expected_sources"], k_values)
        if item["expected_sources"]:
            metrics["answerable_acceptance_rate"] = (
                1.0 if retrieval_result["supported"] else 0.0
            )
            if verify_evidence:
                metrics["answerable_first_stage_acceptance_rate"] = (
                    1.0 if retrieval_result["first_stage_supported"] else 0.0
                )
        else:
            metrics["no_answer_abstention_rate"] = (
                0.0 if retrieval_result["supported"] else 1.0
            )
            if verify_evidence:
                metrics["no_answer_first_stage_abstention_rate"] = (
                    0.0 if retrieval_result["first_stage_supported"] else 1.0
                )
        rows.append(
            {
                "id": item.get("id"),
                "layer": str(item.get("layer") or infer_retrieval_layer(item)),
                "query": item["query"],
                "expected_sources": item["expected_sources"],
                "retrieved_sources": retrieved,
                "retrieval_supported": retrieval_result["supported"],
                "retrieval_confidence": retrieval_result["confidence"],
                "retrieval_abstain_reason": retrieval_result["reason"],
                "retrieval_signals": retrieval_result["signals"],
                "retrieval_first_stage_supported": retrieval_result[
                    "first_stage_supported"
                ],
                "evidence_verification_required": retrieval_result[
                    "evidence_verification_required"
                ],
                "evidence_supported": retrieval_result["evidence_supported"],
                "evidence_verification_reason": retrieval_result[
                    "evidence_verification_reason"
                ],
                "evidence_verified_chunk_ids": retrieval_result[
                    "evidence_verified_chunk_ids"
                ],
                "evidence_verifier_error": retrieval_result.get(
                    "evidence_verifier_error", ""
                ),
                "latency_ms": latency_ms,
                "metrics": metrics,
            }
        )

    aggregate_metrics = _aggregate_rows(rows)
    by_layer = {
        layer: {
            "count": len(layer_rows),
            "aggregate": _aggregate_rows(layer_rows),
        }
        for layer, layer_rows in _group_rows(rows, "layer").items()
    }
    return {
        "config": {
            "k_values": k_values,
            "rerank": rerank,
            "verify_evidence": verify_evidence,
            "is_local_verifier": is_local_verifier,
            "num_queries": len(items),
            "answerable_queries": sum(bool(item["expected_sources"]) for item in items),
            "no_answer_queries": sum(not item["expected_sources"] for item in items),
            "warmup_latency_ms": warmup_latency_ms,
        },
        "aggregate": aggregate_metrics,
        "baseline_gated_metrics": sorted(
            metric
            for metric in aggregate_metrics
            if metric_direction(metric) == "higher"
        ),
        "metric_directions": {
            metric: metric_direction(metric) for metric in aggregate_metrics
        },
        "by_layer": by_layer,
        "rows": rows,
    }


# 按报告行聚合质量和性能指标。
def _aggregate_rows(rows: List[dict]) -> Dict[str, float]:
    if not rows:
        return {}
    result = aggregate([row["metrics"] for row in rows])
    latencies = [float(row["latency_ms"]) for row in rows]
    result["latency_mean_ms"] = statistics.mean(latencies)
    result["latency_p95_ms"] = percentile(latencies, 95)
    return result


# 按字段分组报告行。
def _group_rows(rows: List[dict], key: str) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return dict(sorted(grouped.items()))


# 输出报告。
def print_report(report: dict) -> None:
    cfg = report["config"]
    print(
        f"\n检索评测  |  queries={cfg['num_queries']}  rerank={cfg['rerank']}  k={cfg['k_values']}\n"
    )
    print(f"  warmup={cfg['warmup_latency_ms']:.1f}ms（不计入稳态延迟）\n")

    for row in report["rows"]:
        if row["expected_sources"]:
            recalls = "  ".join(
                f"r@{k}={row['metrics'][f'recall@{k}']:.2f}"
                for k in cfg["k_values"]
            )
            score = f"{row['metrics']['mrr']:.2f} MRR  {recalls}"
            score += (
                "  accepted"
                if row["retrieval_supported"]
                else "  false-abstain"
            )
        else:
            flags = "  ".join(
                f"fp@{k}={row['metrics'][f'no_answer_false_positive@{k}']:.0f}"
                for k in cfg["k_values"]
            )
            decision = "accepted" if row["retrieval_supported"] else "abstained"
            score = f"{flags}  {decision}"
        print(
            f"  [{row['layer']}] [{score}] {row['latency_ms']:.1f}ms"
            f"  | {row['query']}"
        )
        print(f"        expected={row['expected_sources']}")
        print(f"        top={row['retrieved_sources'][: max(cfg['k_values'])]}")
        print(
            f"        confidence={row['retrieval_confidence']:.4f} "
            f"reason={row['retrieval_abstain_reason']} "
            f"signals={row['retrieval_signals']}"
        )
        if cfg.get("verify_evidence"):
            print(
                "        evidence_verify="
                f"{row['evidence_verification_required']} "
                f"supported={row['evidence_supported']} "
                f"chunks={row['evidence_verified_chunk_ids']} "
                f"reason={row['evidence_verification_reason']}"
            )

    print("\n聚合:")
    for key, value in report["aggregate"].items():
        print(f"  {key:<34} {value:.4f}")
    print("\n按检索层:")
    for layer, summary in report["by_layer"].items():
        metrics = "  ".join(
            f"{key}={value:.4f}"
            for key, value in summary["aggregate"].items()
        )
        print(f"  {layer:<14} count={summary['count']}  {metrics}")
    print()


# 输出覆盖审计结果。
def print_coverage(coverage: dict) -> None:
    print("\n覆盖审计:")
    print(f"  layer_counts={coverage['layer_counts']}")
    print(f"  minimums={coverage['minimum_layer_counts']}")
    if coverage["missing_layers"]:
        print(f"  缺少 layer: {coverage['missing_layers']}")
    if coverage["insufficient_layers"]:
        print(f"  数量不足: {coverage['insufficient_layers']}")
    if coverage["is_coverage_complete"]:
        print("  覆盖完整")
    print()


# 生成基线对比。
def compare_baseline(report: dict, baseline_path: Path) -> int:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    base_agg = baseline.get("aggregate", {})
    cur_agg = report["aggregate"]
    print(f"\n对比基线 {baseline_path}:")
    regressed = False
    gated_metrics = report.get("baseline_gated_metrics")
    metric_names = (
        sorted(gated_metrics) if gated_metrics is not None else sorted(cur_agg)
    )
    for key in metric_names:
        cur = cur_agg[key]
        base = base_agg.get(key)
        if base is None:
            print(f"  {key:<12} {cur:.4f}  (基线缺该指标)")
            continue
        delta = cur - base
        direction = report.get("metric_directions", {}).get(
            key, metric_direction(key)
        )
        flag = ""
        regressed_metric = delta < -1e-9 if direction == "higher" else delta > 1e-9
        improved_metric = delta > 1e-9 if direction == "higher" else delta < -1e-9
        if regressed_metric:
            flag = "  ⚠ 回退"
            regressed = True
        elif improved_metric:
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
        "--verify-evidence",
        action="store_true",
        help="对精确事实问题执行二阶段证据充分性模型校验",
    )
    parser.add_argument(
        "--local-verifier",
        action="store_true",
        help="二阶段证据校验使用本地 Ollama；必须同时指定 --verify-evidence",
    )
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
    parser.add_argument(
        "--check-coverage",
        action="store_true",
        help="检查评测集是否覆盖单源、多源和无答案层级",
    )
    parser.add_argument(
        "--coverage-only",
        action="store_true",
        help="只检查评测集覆盖面，不执行真实检索",
    )
    parser.add_argument(
        "--coverage-profile",
        choices=("smoke", "baseline"),
        default="smoke",
        help="smoke 每层至少 1 条；baseline 要求 40/20/20/20 共 100 条",
    )
    parser.add_argument(
        "--gate",
        type=Path,
        default=None,
        help="绝对指标门禁 JSON，包含 minimum/maximum 两组阈值",
    )
    args = parser.parse_args()
    if args.local_verifier and not args.verify_evidence:
        parser.error("--local-verifier 必须与 --verify-evidence 同时使用")
    if args.coverage_only and (
        args.check_coverage or args.json or args.baseline or args.gate
    ):
        parser.error(
            "--coverage-only 不能与 --check-coverage、--json、--baseline 或 --gate 同时使用"
        )

    eval_set = args.eval_set or resolve_default_eval_set()
    items = load_eval_set(eval_set)
    if not items:
        print(f"评测集为空: {eval_set}")
        return 1

    coverage = audit_coverage(items, coverage_minimums(args.coverage_profile))
    if args.coverage_only:
        print_coverage(coverage)
        return 0 if coverage["is_coverage_complete"] else 1

    report = run_eval(
        items,
        sorted(args.k),
        args.rerank,
        verify_evidence=args.verify_evidence,
        is_local_verifier=args.local_verifier,
    )
    threshold_gate = None
    if args.gate:
        threshold_config = json.loads(args.gate.read_text(encoding="utf-8"))
        threshold_gate = evaluate_thresholds(report["aggregate"], threshold_config)
        report["threshold_gate"] = threshold_gate
    print_report(report)
    if threshold_gate:
        print("绝对指标门禁:")
        for row in threshold_gate["rows"]:
            current = "-" if row["current"] is None else f"{row['current']:.4f}"
            status = "通过" if row["passed"] else "失败"
            print(
                f"  {row['metric']:<34} {current} "
                f"{row['bound']}={row['limit']:.4f}  {status}"
            )
        print()
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
        if threshold_gate and not threshold_gate["passed"]:
            return 1
        return baseline_status
    if threshold_gate and not threshold_gate["passed"]:
        return 1
    if args.check_coverage and not coverage["is_coverage_complete"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
