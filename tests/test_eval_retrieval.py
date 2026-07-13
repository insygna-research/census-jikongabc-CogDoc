import json
import sys

import pytest
import scripts.eval_retrieval as eval_retrieval
from cogdoc.tools.eval.retrieval_metrics import (
    aggregate,
    audit_coverage,
    coverage_minimums,
    evaluate_query,
    evaluate_thresholds,
    hit_at_k,
    infer_retrieval_layer,
    metric_direction,
    percentile,
    recall_at_k,
    reciprocal_rank,
)


# 验证召回率按截断范围内的不同期望来源计算。
def test_recall_at_k_counts_distinct_expected_within_cutoff():
    retrieved = ["a.pdf", "a.pdf", "b.pdf", "c.pdf"]
    assert recall_at_k(retrieved, ["a.pdf", "b.pdf"], k=3) == 1.0
    assert recall_at_k(retrieved, ["a.pdf", "b.pdf"], k=2) == 0.5
    assert recall_at_k(retrieved, ["x.pdf"], k=4) == 0.0


# 验证空期望来源的召回率为零。
def test_recall_at_k_empty_expected_is_zero():
    assert recall_at_k(["a.pdf"], [], k=3) == 0.0


# 验证命中率是二值指标。
def test_hit_at_k_is_binary():
    retrieved = ["a.pdf", "b.pdf", "c.pdf"]
    assert hit_at_k(retrieved, ["c.pdf"], k=3) == 1.0
    assert hit_at_k(retrieved, ["c.pdf"], k=2) == 0.0


# 验证倒数排名使用首个命中位置。
def test_reciprocal_rank_uses_first_hit_position():
    retrieved = ["x.pdf", "a.pdf", "b.pdf"]
    assert reciprocal_rank(retrieved, ["a.pdf"]) == 0.5
    assert reciprocal_rank(retrieved, ["x.pdf"]) == 1.0
    assert reciprocal_rank(retrieved, ["none.pdf"]) == 0.0


# 验证单问题评测输出所有请求的截断指标。
def test_evaluate_query_emits_all_requested_cutoffs():
    metrics = evaluate_query(["a.pdf", "b.pdf"], ["b.pdf"], k_values=[1, 3])
    assert set(metrics) == {"mrr", "recall@1", "hit@1", "recall@3", "hit@3"}
    assert metrics["recall@1"] == 0.0
    assert metrics["recall@3"] == 1.0
    assert metrics["mrr"] == 0.5


# 验证无答案问题与可回答问题分开统计误命中。
def test_evaluate_query_emits_no_answer_false_positive_metrics():
    metrics = evaluate_query(["a.pdf"], [], k_values=[1, 5])

    assert metrics == {
        "no_answer_false_positive@1": 1.0,
        "no_answer_false_positive@5": 1.0,
    }
    assert evaluate_query([], [], k_values=[5]) == {
        "no_answer_false_positive@5": 0.0
    }


# 验证聚合逻辑对每个指标取均值。
def test_aggregate_means_each_metric():
    agg = aggregate(
        [
            {"recall@1": 1.0, "mrr": 1.0},
            {"recall@1": 0.0, "mrr": 0.5},
        ]
    )
    assert agg["recall@1"] == 0.5
    assert agg["mrr"] == 0.75


# 验证聚合允许可回答与无答案行携带不同指标。
def test_aggregate_uses_only_rows_that_define_metric():
    agg = aggregate(
        [
            {"mrr": 1.0, "recall@5": 1.0},
            {"no_answer_false_positive@5": 1.0},
        ]
    )

    assert agg == {
        "mrr": 1.0,
        "no_answer_false_positive@5": 1.0,
        "recall@5": 1.0,
    }


# 验证 nearest-rank P95 与指标方向。
def test_percentile_and_metric_direction():
    assert percentile([1, 2, 3, 4, 100], 95) == 100
    assert metric_direction("mrr") == "higher"
    assert metric_direction("latency_p95_ms") == "lower"
    assert metric_direction("no_answer_false_positive@5") == "lower"


# 验证空聚合结果为空字典。
def test_aggregate_empty_is_empty():
    assert aggregate([]) == {}


# 验证检索层级可从期望来源推断。
def test_infer_retrieval_layer_from_expected_sources():
    assert infer_retrieval_layer({"expected_sources": ["a.pdf"]}) == "single-source"
    assert (
        infer_retrieval_layer({"expected_sources": ["a.pdf", "b.pdf"]})
        == "multi-source"
    )
    assert infer_retrieval_layer({"expected_sources": []}) == "no-answer"


# 验证检索覆盖审计能报告缺失层级。
def test_retrieval_coverage_audit_reports_missing_layers():
    coverage = audit_coverage(
        [
            {"expected_sources": ["a.pdf"]},
            {"expected_sources": ["a.pdf", "b.pdf"]},
        ]
    )

    assert coverage["missing_layers"] == ["hard", "no-answer"]
    assert coverage["layer_counts"] == {"multi-source": 1, "single-source": 1}
    assert coverage["is_coverage_complete"] is False


# 验证真实基线配置执行 40/20/20/20 数量门禁。
def test_retrieval_baseline_coverage_requires_layer_quotas():
    items = (
        [{"layer": "single-source"}] * 40
        + [{"layer": "multi-source"}] * 20
        + [{"layer": "hard"}] * 19
        + [{"layer": "no-answer"}] * 20
    )

    coverage = audit_coverage(items, coverage_minimums("baseline"))

    assert coverage["total_count"] == 99
    assert coverage["insufficient_layers"] == {
        "hard": {"actual": 19, "required": 20}
    }
    assert coverage["is_coverage_complete"] is False


# 验证绝对门禁同时支持下限和上限指标。
def test_retrieval_threshold_gate_handles_minimum_and_maximum():
    gate = evaluate_thresholds(
        {"mrr": 0.8, "latency_p95_ms": 900.0},
        {
            "minimum": {"mrr": 0.75},
            "maximum": {"latency_p95_ms": 1000.0},
        },
    )

    assert gate["passed"] is True
    assert all(row["passed"] for row in gate["rows"])


# 写入覆盖完整的检索评测集。
def _write_complete_retrieval_eval(path):
    rows = [
        {
            "query": "单文档问题",
            "expected_sources": ["a.pdf"],
            "doc_id": "demo",
            "layer": "single-source",
        },
        {
            "query": "跨文档问题",
            "expected_sources": ["a.pdf", "b.pdf"],
            "doc_id": "demo",
            "layer": "multi-source",
        },
        {
            "query": "无答案问题",
            "expected_sources": [],
            "doc_id": "demo",
            "layer": "no-answer",
        },
        {
            "query": "细粒度困难问题",
            "expected_sources": ["a.pdf"],
            "doc_id": "demo",
            "layer": "hard",
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )


# 验证检索覆盖快速模式跳过真实检索。
def test_retrieval_cli_coverage_only_skips_eval(tmp_path, monkeypatch, capsys):
    eval_set = tmp_path / "retrieval.jsonl"
    _write_complete_retrieval_eval(eval_set)

    # 阻止覆盖快速模式误入真实检索。
    def fail_run_eval(_items, _k_values, _rerank):
        raise AssertionError("run_eval should not be called")

    monkeypatch.setattr(eval_retrieval, "run_eval", fail_run_eval)
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_retrieval.py", "--eval-set", str(eval_set), "--coverage-only"],
    )

    assert eval_retrieval.main() == 0
    out = capsys.readouterr().out
    assert "覆盖完整" in out
    assert "检索评测" not in out


# 验证检索覆盖快速模式拒绝写报告参数。
def test_retrieval_cli_coverage_only_rejects_json(tmp_path, monkeypatch):
    eval_set = tmp_path / "retrieval.jsonl"
    _write_complete_retrieval_eval(eval_set)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_retrieval.py",
            "--eval-set",
            str(eval_set),
            "--coverage-only",
            "--json",
            str(tmp_path / "report.json"),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        eval_retrieval.main()
    assert exc.value.code == 2


# 验证检索覆盖快速模式拒绝重复覆盖参数。
def test_retrieval_cli_coverage_only_rejects_check_coverage(tmp_path, monkeypatch):
    eval_set = tmp_path / "retrieval.jsonl"
    _write_complete_retrieval_eval(eval_set)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_retrieval.py",
            "--eval-set",
            str(eval_set),
            "--coverage-only",
            "--check-coverage",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        eval_retrieval.main()
    assert exc.value.code == 2


# 验证真实检索报告包含分层指标和延迟。
def test_retrieval_run_eval_reports_layers_and_latency(monkeypatch):
    def fake_retrieve(query, doc_id, top_k, rerank):
        supported = query != "none"
        return {
            "sources": ["a.pdf"] if supported else [],
            "supported": supported,
            "confidence": 1.0 if supported else 0.0,
            "reason": "supported" if supported else "no_candidates",
            "signals": {},
        }

    monkeypatch.setattr(
        eval_retrieval,
        "retrieve_result",
        fake_retrieve,
    )

    report = eval_retrieval.run_eval(
        [
            {
                "query": "answerable",
                "expected_sources": ["a.pdf"],
                "layer": "single-source",
            },
            {"query": "none", "expected_sources": [], "layer": "no-answer"},
        ],
        [1, 5],
        False,
    )

    assert report["aggregate"]["mrr"] == 1.0
    assert report["aggregate"]["no_answer_false_positive@5"] == 0.0
    assert report["aggregate"]["answerable_acceptance_rate"] == 1.0
    assert report["aggregate"]["no_answer_abstention_rate"] == 1.0
    assert report["aggregate"]["latency_p95_ms"] >= 0.0
    assert report["by_layer"]["single-source"]["count"] == 1
    assert report["by_layer"]["no-answer"]["count"] == 1
