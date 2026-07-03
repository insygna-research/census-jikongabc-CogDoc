from cogdoc.tools.eval.retrieval_metrics import (
    aggregate,
    audit_coverage,
    evaluate_query,
    hit_at_k,
    infer_retrieval_layer,
    recall_at_k,
    reciprocal_rank,
)


# 验证 recall at k counts distinct expected within cutoff 场景。
def test_recall_at_k_counts_distinct_expected_within_cutoff():
    retrieved = ["a.pdf", "a.pdf", "b.pdf", "c.pdf"]
    assert recall_at_k(retrieved, ["a.pdf", "b.pdf"], k=3) == 1.0
    assert recall_at_k(retrieved, ["a.pdf", "b.pdf"], k=2) == 0.5
    assert recall_at_k(retrieved, ["x.pdf"], k=4) == 0.0


# 验证 recall at k empty expected is zero 场景。
def test_recall_at_k_empty_expected_is_zero():
    assert recall_at_k(["a.pdf"], [], k=3) == 0.0


# 验证 hit at k is binary 场景。
def test_hit_at_k_is_binary():
    retrieved = ["a.pdf", "b.pdf", "c.pdf"]
    assert hit_at_k(retrieved, ["c.pdf"], k=3) == 1.0
    assert hit_at_k(retrieved, ["c.pdf"], k=2) == 0.0


# 验证 reciprocal rank uses first hit position 场景。
def test_reciprocal_rank_uses_first_hit_position():
    retrieved = ["x.pdf", "a.pdf", "b.pdf"]
    assert reciprocal_rank(retrieved, ["a.pdf"]) == 0.5
    assert reciprocal_rank(retrieved, ["x.pdf"]) == 1.0
    assert reciprocal_rank(retrieved, ["none.pdf"]) == 0.0


# 验证 evaluate query emits all requested cutoffs 场景。
def test_evaluate_query_emits_all_requested_cutoffs():
    metrics = evaluate_query(["a.pdf", "b.pdf"], ["b.pdf"], k_values=[1, 3])
    assert set(metrics) == {"mrr", "recall@1", "hit@1", "recall@3", "hit@3"}
    assert metrics["recall@1"] == 0.0
    assert metrics["recall@3"] == 1.0
    assert metrics["mrr"] == 0.5


# 验证 aggregate means each metric 场景。
def test_aggregate_means_each_metric():
    agg = aggregate(
        [
            {"recall@1": 1.0, "mrr": 1.0},
            {"recall@1": 0.0, "mrr": 0.5},
        ]
    )
    assert agg["recall@1"] == 0.5
    assert agg["mrr"] == 0.75


# 验证 aggregate empty is empty 场景。
def test_aggregate_empty_is_empty():
    assert aggregate([]) == {}


# 验证 infer retrieval layer from expected sources 场景。
def test_infer_retrieval_layer_from_expected_sources():
    assert infer_retrieval_layer({"expected_sources": ["a.pdf"]}) == "single-source"
    assert (
        infer_retrieval_layer({"expected_sources": ["a.pdf", "b.pdf"]})
        == "multi-source"
    )
    assert infer_retrieval_layer({"expected_sources": []}) == "no-answer"


# 验证 retrieval coverage audit reports missing layers 场景。
def test_retrieval_coverage_audit_reports_missing_layers():
    coverage = audit_coverage(
        [
            {"expected_sources": ["a.pdf"]},
            {"expected_sources": ["a.pdf", "b.pdf"]},
        ]
    )

    assert coverage["missing_layers"] == ["no-answer"]
    assert coverage["is_coverage_complete"] is False
