import pytest
from tests._native import require_rust_core

rust_core = require_rust_core("rrf_fusion_native")

def _doc(source: str, chunk_index: int) -> dict:
    return {
        "text": f"{source}#{chunk_index}",
        "meta": {"source": source, "chunk_index": chunk_index},
    }

def _keys(docs):
    return [(d["meta"]["source"], d["meta"]["chunk_index"]) for d in docs]

def test_tie_break_is_deterministic_by_doc_key():
    # 验证融合得分相同时按文档键稳定排序。
    vector = [_doc("a.pdf", 0), _doc("b.pdf", 1)]
    bm25 = [_doc("b.pdf", 1), _doc("a.pdf", 0)]

    result = rust_core.rrf_fusion_native(vector, bm25, 60.0, 2)

    assert _keys(result) == [("a.pdf", 0), ("b.pdf", 1)]

def test_tie_break_top_k_boundary_is_stable():
    # 验证 top_k 截断不会破坏平分排序的稳定性。
    vector = [_doc("a.pdf", 0), _doc("b.pdf", 1)]
    bm25 = [_doc("b.pdf", 1), _doc("a.pdf", 0)]

    result = rust_core.rrf_fusion_native(vector, bm25, 60.0, 1)
    assert _keys(result) == [("a.pdf", 0)]

def test_repeated_runs_produce_identical_order():
    # 验证相同输入多次融合得到相同顺序。
    vector = [_doc("a.pdf", i) for i in range(5)]
    bm25 = [_doc("a.pdf", i) for i in range(4, -1, -1)]

    first = _keys(rust_core.rrf_fusion_native(vector, bm25, 60.0, 5))
    for _ in range(10):
        again = _keys(rust_core.rrf_fusion_native(
            [_doc("a.pdf", i) for i in range(5)],
            [_doc("a.pdf", i) for i in range(4, -1, -1)],
            60.0,
            5,
        ))
        assert again == first

def test_same_doc_in_both_channels_is_merged():
    # 验证两路召回中的同一文档会去重并累加得分。
    vector = [_doc("a.pdf", 0)]
    bm25 = [_doc("a.pdf", 0)]

    result = rust_core.rrf_fusion_native(vector, bm25, 60.0, 5)
    assert len(result) == 1

    expected = 1.0 / (60.0 + 1) + 1.0 / (60.0 + 1)
    assert result[0]["retrieval"]["rrf_score"] == pytest.approx(expected)
    assert result[0]["retrieval"]["search_channel"] == "hybrid"

def test_higher_rank_wins_when_scores_differ():
    # 验证融合得分不同时高分文档排在前面。
    vector = [_doc("b.pdf", 1), _doc("a.pdf", 0)]
    bm25 = [_doc("b.pdf", 1), _doc("a.pdf", 0)]

    result = rust_core.rrf_fusion_native(vector, bm25, 60.0, 2)
    assert _keys(result) == [("b.pdf", 1), ("a.pdf", 0)]
