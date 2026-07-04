import json
import sys

import pytest
import scripts.eval_retrieval as eval_retrieval
from cogdoc.tools.eval.retrieval_metrics import (
    aggregate,
    audit_coverage,
    evaluate_query,
    hit_at_k,
    infer_retrieval_layer,
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

    assert coverage["missing_layers"] == ["no-answer"]
    assert coverage["is_coverage_complete"] is False


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
