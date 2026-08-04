import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import scripts.eval_retrieval as eval_retrieval
from cogdoc.tools.eval.retrieval_metrics import (
    aggregate,
    audit_coverage,
    coverage_minimums,
    evaluate_query,
    evaluate_requirement_coverage,
    evaluate_thresholds,
    hit_at_k,
    infer_retrieval_layer,
    metric_direction,
    percentile,
    recall_at_k,
    reciprocal_rank,
    requirement_coverage_rate,
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
    assert evaluate_query([], [], k_values=[5]) == {"no_answer_false_positive@5": 0.0}


# 验证 chunk 级需求覆盖不会把同一 PDF 内的错误文本块算成成功。
def test_requirement_coverage_uses_chunk_level_gold_and_full_coverage():
    retrieved = [
        {"chunk_id": "a-wrong", "source": "a.pdf"},
        {"chunk_id": "b-right", "source": "b.pdf"},
        {"chunk_id": "a-right", "source": "a.pdf"},
    ]
    requirements = [
        {"requirement_id": "r1", "acceptable_chunk_ids": ["a-right"]},
        {"requirement_id": "r2", "acceptable_chunk_ids": ["b-right"]},
    ]

    metrics = evaluate_requirement_coverage(retrieved, requirements, [2, 3])

    assert metrics["requirement_recall@2"] == 0.5
    assert metrics["all_requirements_covered@2"] == 0.0
    assert metrics["requirement_recall@3"] == 1.0
    assert metrics["all_requirements_covered@3"] == 1.0
    assert metrics["chunk_precision@2"] == 0.5
    assert 0.0 < metrics["evidence_ndcg@3"] < 1.0


# 验证来源级迁移标注和 hard-negative 拒绝率可同时评估。
def test_requirement_coverage_supports_source_gold_and_hard_negatives():
    retrieved = [
        {"chunk_id": "distractor", "source": "noise.pdf"},
        {"chunk_id": "right", "source": "policy.pdf"},
    ]
    requirements = [{"requirement_id": "r1", "acceptable_sources": ["policy.pdf"]}]

    metrics = evaluate_requirement_coverage(
        retrieved,
        requirements,
        [1, 2],
        hard_negative_chunk_ids=["distractor"],
    )

    assert metrics["requirement_recall@1"] == 0.0
    assert metrics["requirement_recall@2"] == 1.0
    assert metrics["hard_negative_rejection@1"] == 0.0
    assert metrics["hard_negative_rejection@2"] == 0.0


def test_requirement_coverage_rate_measures_bounded_generation_context():
    assert (
        requirement_coverage_rate(
            [{"chunk_id": "answer-sibling", "source": "policy.pdf"}],
            [
                {"requirement_id": "r1", "acceptable_chunk_ids": ["answer-sibling"]},
                {"requirement_id": "r2", "acceptable_chunk_ids": ["missing"]},
            ],
        )
        == 0.5
    )


# 验证同一需求的多个可接受块不会重复抬高 nDCG。
def test_requirement_ndcg_deduplicates_multiple_hits_for_one_requirement():
    metrics = evaluate_requirement_coverage(
        [{"chunk_id": "a"}, {"chunk_id": "b"}],
        [{"requirement_id": "r1", "acceptable_chunk_ids": ["a", "b"]}],
        [2],
    )

    assert metrics["evidence_ndcg@2"] == pytest.approx(1.0)
    assert metrics["evidence_ndcg@2"] <= 1.0


# 验证一块同时覆盖两个需求时，实际增益和理想增益使用相同多需求口径。
def test_requirement_ndcg_handles_one_chunk_covering_two_requirements():
    metrics = evaluate_requirement_coverage(
        [{"chunk_id": "shared"}],
        [
            {"requirement_id": "r1", "acceptable_chunk_ids": ["shared"]},
            {"requirement_id": "r2", "acceptable_chunk_ids": ["shared"]},
        ],
        [1, 2],
    )

    assert metrics["evidence_ndcg@1"] == pytest.approx(1.0)
    assert metrics["evidence_ndcg@2"] == pytest.approx(1.0)
    assert metrics["evidence_ndcg@1"] <= 1.0
    assert metrics["evidence_ndcg@2"] <= 1.0


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
    assert coverage["insufficient_layers"] == {"hard": {"actual": 19, "required": 20}}
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
    def fake_retrieve(
        query,
        doc_id,
        top_k,
        rerank,
        *,
        verify_evidence=False,
        is_local_verifier=False,
        rewritten_queries=None,
        evidence_requirements=None,
    ):
        supported = query != "none"
        return {
            "sources": ["a.pdf"] if supported else [],
            "supported": supported,
            "first_stage_supported": supported,
            "confidence": 1.0 if supported else 0.0,
            "reason": "supported" if supported else "no_candidates",
            "signals": {},
            "evidence_verification_required": False,
            "evidence_supported": supported,
            "evidence_verification_reason": "not_required",
            "evidence_verified_chunk_ids": [],
            "generation_context_items": (
                [{"chunk_id": "a-child", "source": "a.pdf"}] if supported else []
            ),
            "parent_context_expanded_count": 1 if supported else 0,
            "neighbor_context_expanded_count": 0,
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
                "gold_requirements": [
                    {
                        "requirement_id": "r1",
                        "acceptable_chunk_ids": ["a-child"],
                    }
                ],
            },
            {"query": "none", "expected_sources": [], "layer": "no-answer"},
        ],
        [1, 5],
        False,
        verify_evidence=True,
    )

    assert report["aggregate"]["mrr"] == 1.0
    assert report["aggregate"]["no_answer_false_positive@5"] == 0.0
    assert report["aggregate"]["answerable_acceptance_rate"] == 1.0
    assert report["aggregate"]["no_answer_abstention_rate"] == 1.0
    assert report["aggregate"]["answerable_first_stage_acceptance_rate"] == 1.0
    assert report["aggregate"]["no_answer_first_stage_abstention_rate"] == 1.0
    assert report["config"]["verify_evidence"] is True
    assert report["aggregate"]["latency_p95_ms"] >= 0.0
    assert report["rows"][0]["metrics"]["generation_requirement_coverage"] == 1.0
    assert report["rows"][0]["parent_context_expanded_count"] == 1
    assert report["aggregate"]["parent_context_trigger_rate"] == 0.5
    assert report["by_layer"]["single-source"]["count"] == 1
    assert report["by_layer"]["no-answer"]["count"] == 1


# 验证关闭模型证据校验时，多需求首轮失败仍按线上语义执行一次定向补检索。
def test_retrieve_result_adaptive_second_round_without_model_verifier(monkeypatch):
    from cogdoc import state_runtime
    from cogdoc.graph.subgraphs import qa
    from cogdoc.tools.retriever import confidence

    settings = SimpleNamespace(
        qa_adaptive_retrieval_enabled=True,
        qa_adaptive_retrieval_max_retries=1,
        qa_adaptive_retrieval_top_k_multiplier=2.0,
        qa_adaptive_retrieval_max_top_k=12,
        qa_retrieval_max_queries=7,
        qa_rerank_top_n=1,
        qa_evidence_pack_max_docs=8,
        qa_evidence_pack_max_chars=7200,
        hybrid_rrf_k=60,
    )
    requirements = [
        {
            "requirement_id": "r1",
            "retrieval_query": "A 主检索",
            "recovery_query": "A 恢复检索",
        },
        {
            "requirement_id": "r2",
            "retrieval_query": "B 主检索",
            "recovery_query": "B 恢复检索",
        },
    ]

    class Engine:
        def __init__(self):
            self.calls = []

        def search(self, query, top_k):
            self.calls.append((query, top_k))
            if query == "B 恢复检索":
                return [
                    {
                        "text": f"B 的直接证据 {index}",
                        "meta": {
                            "chunk_id": f"c{index}",
                            "source": f"b{index}.pdf",
                        },
                    }
                    for index in range(3)
                ]
            return []

    class Knowledge:
        def search(self, kb_id, query, top_k):
            return []

    class Feedback:
        def boosts_for_query(self, kb_id, query):
            return {}

    engine = Engine()

    @contextmanager
    def lease(_kb_id):
        yield

    monkeypatch.setattr(eval_retrieval, "get_settings", lambda: settings)
    monkeypatch.setattr(qa.RetrieverFactory, "get_engine", lambda _kb_id: engine)
    monkeypatch.setattr("cogdoc.service.kb_readers.kb_read_lease", lease)
    monkeypatch.setattr(
        state_runtime,
        "default_state_runtime",
        lambda: SimpleNamespace(
            derived_knowledge_retriever=Knowledge(),
            retrieval_feedback_store=Feedback(),
        ),
    )
    decision_candidate_counts = []

    def assess_support(docs, _settings):
        decision_candidate_counts.append(len(docs))
        return SimpleNamespace(
            supported=bool(docs),
            score=1.0 if docs else 0.0,
            reason="supported" if docs else "no_candidates",
            signals={},
        )

    monkeypatch.setattr(confidence, "assess_retrieval_support", assess_support)

    result = eval_retrieval.retrieve_result(
        "比较 A 和 B",
        "kb",
        3,
        False,
        verify_evidence=False,
        evidence_requirements=requirements,
    )

    assert result["supported"] is True
    assert result["retrieval_retry_count"] == 1
    assert result["adaptive_retrieval_rescued"] is True
    assert result["sources"] == ["b0.pdf", "b1.pdf", "b2.pdf"]
    assert decision_candidate_counts == [0, 1]
    assert result["retrieval_query_count"] == 6
    assert result["retrieval_ranking_count"] == 1
    assert result["retrieval_channel_counts"] == {
        "hybrid": 3,
        "derived_knowledge": 0,
    }
    assert ("B 主检索", 3) in engine.calls
    assert ("B 恢复检索", 6) in engine.calls


# 验证补检索只召回新缺口时，上一轮已验证证据仍进入终轮闭集和评测排名。
def test_retrieve_result_carries_verified_docs_into_second_round(monkeypatch):
    from cogdoc import state_runtime
    from cogdoc.agents import evidence_verifier
    from cogdoc.graph.subgraphs import qa
    from cogdoc.tools.retriever import confidence

    settings = SimpleNamespace(
        qa_adaptive_retrieval_enabled=True,
        qa_adaptive_retrieval_max_retries=1,
        qa_adaptive_retrieval_top_k_multiplier=2.0,
        qa_adaptive_retrieval_max_top_k=12,
        qa_retrieval_max_queries=7,
        qa_rerank_top_n=2,
        qa_evidence_pack_max_docs=8,
        qa_evidence_pack_max_chars=7200,
        qa_evidence_verify_max_docs=4,
        hybrid_rrf_k=60,
    )
    requirements = [
        {
            "requirement_id": "r1",
            "retrieval_query": "A 主检索",
            "recovery_query": "A 恢复检索",
        },
        {
            "requirement_id": "r2",
            "retrieval_query": "B 主检索",
            "recovery_query": "B 恢复检索",
        },
    ]

    class Engine:
        def search(self, query, top_k):
            if query == "A 主检索" and top_k == 3:
                return [
                    {
                        "text": "A 的直接证据",
                        "meta": {"chunk_id": "c1", "source": "a.pdf"},
                    }
                ]
            if query == "B 恢复检索" and top_k == 6:
                return [
                    {
                        "text": "B 的直接证据",
                        "meta": {"chunk_id": "c2", "source": "b.pdf"},
                    }
                ]
            return []

    class Knowledge:
        def search(self, kb_id, query, top_k):
            return []

    class Feedback:
        def boosts_for_query(self, kb_id, query):
            return {}

    @contextmanager
    def lease(_kb_id):
        yield

    verifier_calls = []

    def verify(state):
        chunk_ids = [doc["meta"]["chunk_id"] for doc in state["verification_docs"]]
        verifier_calls.append(chunk_ids)
        if "c2" not in chunk_ids:
            return {
                "evidence_verification_required": True,
                "evidence_supported": False,
                "evidence_verification_reason": "缺少 B",
                "evidence_verified_chunk_ids": ["c1"],
                "evidence_requirement_assessments": [
                    {
                        "requirement_id": "r1",
                        "verdict": "supported",
                        "evidence_chunk_ids": ["c1"],
                        "reason": "A 已覆盖",
                    },
                    {
                        "requirement_id": "r2",
                        "verdict": "missing",
                        "evidence_chunk_ids": [],
                        "reason": "缺少 B",
                    },
                ],
                "missing_evidence_requirement_ids": ["r2"],
                "retrieval_abstained": True,
                "retrieval_abstain_reason": "evidence_not_supported",
            }
        return {
            "evidence_verification_required": True,
            "evidence_supported": True,
            "evidence_verification_reason": "全部覆盖",
            "evidence_verified_chunk_ids": ["c1", "c2"],
            "evidence_requirement_assessments": [
                {
                    "requirement_id": requirement_id,
                    "verdict": "supported",
                    "evidence_chunk_ids": [chunk_id],
                    "reason": "已覆盖",
                }
                for requirement_id, chunk_id in (("r1", "c1"), ("r2", "c2"))
            ],
            "missing_evidence_requirement_ids": [],
            "retrieval_abstained": False,
            "retrieval_abstain_reason": "evidence_supported",
        }

    monkeypatch.setattr(eval_retrieval, "get_settings", lambda: settings)
    monkeypatch.setattr(qa.RetrieverFactory, "get_engine", lambda _kb_id: Engine())
    monkeypatch.setattr("cogdoc.service.kb_readers.kb_read_lease", lease)
    monkeypatch.setattr(
        state_runtime,
        "default_state_runtime",
        lambda: SimpleNamespace(
            derived_knowledge_retriever=Knowledge(),
            retrieval_feedback_store=Feedback(),
        ),
    )
    monkeypatch.setattr(
        confidence,
        "assess_retrieval_support",
        lambda docs, _settings: SimpleNamespace(
            supported=bool(docs),
            score=1.0 if docs else 0.0,
            reason="supported" if docs else "no_candidates",
            signals={},
        ),
    )
    monkeypatch.setattr(
        evidence_verifier, "should_verify_evidence", lambda state, settings: True
    )
    monkeypatch.setattr(
        evidence_verifier,
        "select_verification_docs",
        lambda docs, max_docs, requirement_ids=None, pinned_chunk_ids=None: list(docs),
    )
    monkeypatch.setattr(evidence_verifier.EvidenceVerifierAgent, "verify", verify)

    result = eval_retrieval.retrieve_result(
        "比较 A 和 B",
        "kb",
        3,
        False,
        verify_evidence=True,
        evidence_requirements=requirements,
    )

    assert verifier_calls == [["c1"], ["c1", "c2"]]
    assert result["supported"] is True
    assert result["retrieval_retry_count"] == 1
    assert result["retrieval_carryover_count"] == 1
    assert result["sources"] == ["a.pdf", "b.pdf"]
    assert [item["chunk_id"] for item in result["items"]] == ["c1", "c2"]
    assert result["evidence_verified_chunk_ids"] == ["c1", "c2"]


# Parent sibling 必须在离线 verifier 调用前进入闭集，顺序与线上 QA 一致。
def test_retrieve_result_hydrates_parent_context_before_verifier(monkeypatch):
    from cogdoc import state_runtime
    from cogdoc.agents import evidence_verifier
    from cogdoc.config.settings import Settings
    from cogdoc.graph.subgraphs import qa
    from cogdoc.tools.retriever import confidence

    settings = Settings(
        _env_file=None,
        qa_adaptive_retrieval_enabled=False,
        qa_retrieval_max_queries=1,
        qa_rerank_top_n=1,
        qa_evidence_verify_max_docs=3,
        qa_parent_context_max_chunks=3,
        qa_parent_context_max_chars=1000,
    )

    def child(index):
        return {
            "text": f"child-{index}",
            "meta": {
                "chunk_id": f"c{index}",
                "source": "paper.pdf",
                "page": 1,
                "page_start": 1,
                "page_end": 1,
                "chunk_index": index,
                "parent_chunk_id": "parent-1",
                "section_title": "Methods",
                "section_path": "Methods",
                "section_level": 1,
                "child_index_in_parent": index,
            },
        }

    chunks = [child(index) for index in range(3)]

    class Engine:
        def search(self, query, top_k):
            return [child(1)]

        def load_source_chunks(self, source):
            assert source == "paper.pdf"
            return chunks

    class Knowledge:
        def search(self, kb_id, query, top_k):
            return []

    class Feedback:
        def boosts_for_query(self, kb_id, query):
            return {}

    @contextmanager
    def lease(_kb_id):
        yield

    verifier_calls = []

    def verify(state):
        chunk_ids = [doc["meta"]["chunk_id"] for doc in state["verification_docs"]]
        verifier_calls.append(chunk_ids)
        return {
            "evidence_verification_required": True,
            "evidence_supported": True,
            "evidence_verification_reason": "右侧 sibling 包含答案",
            "evidence_verified_chunk_ids": ["c2"],
            "evidence_requirement_assessments": [
                {
                    "requirement_id": "r1",
                    "verdict": "supported",
                    "evidence_chunk_ids": ["c2"],
                    "reason": "直接证据",
                }
            ],
            "missing_evidence_requirement_ids": [],
            "retrieval_abstained": False,
            "retrieval_abstain_reason": "evidence_supported",
        }

    engine = Engine()
    monkeypatch.setattr(eval_retrieval, "get_settings", lambda: settings)
    monkeypatch.setattr(qa, "get_settings", lambda: settings)
    monkeypatch.setattr(qa, "kb_read_lease", lease)
    monkeypatch.setattr(qa.RetrieverFactory, "get_engine", lambda _kb_id: engine)
    monkeypatch.setattr("cogdoc.service.kb_readers.kb_read_lease", lease)
    monkeypatch.setattr(
        state_runtime,
        "default_state_runtime",
        lambda: SimpleNamespace(
            derived_knowledge_retriever=Knowledge(),
            retrieval_feedback_store=Feedback(),
        ),
    )
    monkeypatch.setattr(
        confidence,
        "assess_retrieval_support",
        lambda docs, _settings: SimpleNamespace(
            supported=True,
            score=1.0,
            reason="supported",
            signals={},
        ),
    )
    monkeypatch.setattr(
        evidence_verifier, "should_verify_evidence", lambda state, settings: True
    )
    monkeypatch.setattr(evidence_verifier.EvidenceVerifierAgent, "verify", verify)

    result = eval_retrieval.retrieve_result(
        "训练阶段是什么？",
        "kb",
        3,
        False,
        verify_evidence=True,
        evidence_requirements=[
            {
                "requirement_id": "r1",
                "question": "训练阶段是什么？",
                "retrieval_query": "训练阶段",
                "recovery_query": "训练阶段细节",
            }
        ],
    )

    assert verifier_calls == [["c1", "c0", "c2"]]
    assert [item["chunk_id"] for item in result["generation_context_items"]] == [
        "c0",
        "c1",
        "c2",
    ]
    assert result["evidence_pack_input_count"] == 3
    assert result["evidence_pack_kept_count"] == 3
    assert result["evidence_pack_dropped_count"] == 0
    packed_ids = {item["chunk_id"] for item in result["evidence_pack_context_items"]}
    assert set(verifier_calls[0]) <= packed_ids
    assert result["evidence_supported"] is True


# Anchor 硬约束本身超出 pack 预算时必须在模型 verifier 前安全拒答。
def test_retrieve_result_fails_closed_before_verifier_when_pack_is_over_budget(
    monkeypatch,
):
    from cogdoc import state_runtime
    from cogdoc.agents import evidence_verifier
    from cogdoc.config.settings import Settings
    from cogdoc.graph.subgraphs import qa
    from cogdoc.tools.retriever import confidence

    settings = Settings(
        _env_file=None,
        qa_adaptive_retrieval_enabled=True,
        qa_adaptive_retrieval_max_retries=1,
        qa_retrieval_max_queries=1,
        qa_rerank_top_n=2,
        qa_evidence_pack_max_docs=1,
        qa_evidence_pack_max_chars=1000,
    )

    class Engine:
        def search(self, query, top_k):
            return [
                {
                    "text": f"hard evidence {index}",
                    "meta": {
                        "chunk_id": f"c{index}",
                        "source": f"source-{index}.pdf",
                    },
                }
                for index in range(2)
            ]

    class Knowledge:
        def search(self, kb_id, query, top_k):
            return []

    class Feedback:
        def boosts_for_query(self, kb_id, query):
            return {}

    @contextmanager
    def lease(_kb_id):
        yield

    verifier_calls = []

    def fail_verify(state):
        verifier_calls.append(state)
        raise AssertionError("hard-over-budget pack must not reach verifier")

    monkeypatch.setattr(eval_retrieval, "get_settings", lambda: settings)
    monkeypatch.setattr(qa.RetrieverFactory, "get_engine", lambda _kb_id: Engine())
    monkeypatch.setattr("cogdoc.service.kb_readers.kb_read_lease", lease)
    monkeypatch.setattr(
        state_runtime,
        "default_state_runtime",
        lambda: SimpleNamespace(
            derived_knowledge_retriever=Knowledge(),
            retrieval_feedback_store=Feedback(),
        ),
    )
    monkeypatch.setattr(
        confidence,
        "assess_retrieval_support",
        lambda docs, _settings: SimpleNamespace(
            supported=True,
            score=1.0,
            reason="supported",
            signals={},
        ),
    )
    monkeypatch.setattr(
        evidence_verifier, "should_verify_evidence", lambda state, settings: True
    )
    monkeypatch.setattr(evidence_verifier.EvidenceVerifierAgent, "verify", fail_verify)

    result = eval_retrieval.retrieve_result(
        "需要精确核验的事实是什么？",
        "kb",
        2,
        False,
        verify_evidence=True,
    )

    assert verifier_calls == []
    assert result["supported"] is False
    assert result["reason"] == "evidence_pack_hard_budget_exceeded"
    assert result["evidence_verification_required"] is False
    assert result["retrieval_retry_count"] == 0
    assert result["evidence_pack_input_count"] == 2
    assert result["evidence_pack_kept_count"] == 2
    assert result["evidence_pack_over_budget"] is True
    assert result["generation_context_items"] == []
    assert len(result["evidence_pack_context_items"]) == 2


# Pack 容量与覆盖变化是诊断数据，报告应保留但不得自动进入历史 baseline gate。
def test_run_eval_reports_pack_diagnostics_without_baseline_gating(monkeypatch):
    def fake_retrieve(
        query,
        doc_id,
        top_k,
        rerank,
        *,
        verify_evidence=False,
        is_local_verifier=False,
        rewritten_queries=None,
        evidence_requirements=None,
    ):
        return {
            "sources": ["policy.pdf", "appendix.pdf"],
            "items": [
                {"chunk_id": "kept", "source": "policy.pdf"},
                {"chunk_id": "dropped", "source": "appendix.pdf"},
            ],
            "supported": True,
            "first_stage_supported": True,
            "confidence": 1.0,
            "reason": "supported",
            "signals": {},
            "evidence_verification_required": False,
            "evidence_supported": True,
            "evidence_verification_reason": "not_required",
            "evidence_verified_chunk_ids": [],
            "generation_context_items": [{"chunk_id": "kept", "source": "policy.pdf"}],
            "evidence_pack_input_items": [
                {"chunk_id": "kept", "source": "policy.pdf"},
                {"chunk_id": "dropped", "source": "appendix.pdf"},
            ],
            "evidence_pack_context_items": [
                {"chunk_id": "kept", "source": "policy.pdf"}
            ],
            "evidence_pack_input_count": 2,
            "evidence_pack_kept_count": 1,
            "evidence_pack_dropped_count": 1,
            "evidence_pack_input_chars": 120,
            "evidence_pack_kept_chars": 60,
            "evidence_pack_overlap_removed_chars": 12,
            "evidence_pack_anchor_count": 1,
            "evidence_pack_pinned_count": 0,
            "evidence_pack_drop_reason_counts": {"max_docs": 1},
            "evidence_pack_over_budget": False,
        }

    monkeypatch.setattr(eval_retrieval, "retrieve_result", fake_retrieve)
    report = eval_retrieval.run_eval(
        [
            {
                "query": "比较两项要求",
                "expected_sources": ["policy.pdf"],
                "gold_requirements": [
                    {
                        "requirement_id": "r1",
                        "acceptable_chunk_ids": ["kept"],
                    },
                    {
                        "requirement_id": "r2",
                        "acceptable_chunk_ids": ["dropped"],
                    },
                ],
            }
        ],
        [1, 2],
        False,
    )

    row = report["rows"][0]
    assert row["evidence_pack_input_count"] == 2
    assert row["evidence_pack_kept_count"] == 1
    assert row["evidence_pack_input_chars"] == 120
    assert row["evidence_pack_kept_chars"] == 60
    assert row["evidence_pack_overlap_removed_chars"] == 12
    assert row["evidence_pack_drop_reason_counts"] == {"max_docs": 1}
    assert row["evidence_pack_requirement_coverage_pre"] == 1.0
    assert row["evidence_pack_requirement_coverage_post"] == 0.5
    assert report["aggregate"]["evidence_pack_dropped_count"] == 1.0
    assert not any(
        metric.startswith("evidence_pack_")
        for metric in report["baseline_gated_metrics"]
    )
