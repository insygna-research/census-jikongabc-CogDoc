import pytest
from tests._native import require_rust_core

rust_core = require_rust_core("rrf_fusion_native")


def _doc(source: str, chunk_index: int) -> dict:
    # 构造带稳定 chunk_id 的最小检索结果。
    return {
        "text": f"{source}#{chunk_index}",
        "meta": {
            "chunk_id": f"chunk:{source}:{chunk_index}",
            "source": source,
            "chunk_index": chunk_index,
        },
    }


def _keys(docs):
    # 测试只关心排序身份。
    return [(d["meta"]["source"], d["meta"]["chunk_index"]) for d in docs]


def test_tie_break_is_deterministic_by_doc_key():
    # 平分时按 chunk_id 稳定排序。
    vector = [_doc("a.pdf", 0), _doc("b.pdf", 1)]
    bm25 = [_doc("b.pdf", 1), _doc("a.pdf", 0)]

    result = rust_core.rrf_fusion_native(vector, bm25, 60.0, 2)

    assert _keys(result) == [("a.pdf", 0), ("b.pdf", 1)]


def test_tie_break_top_k_boundary_is_stable():
    # top_k 截断不能破坏平分排序。
    vector = [_doc("a.pdf", 0), _doc("b.pdf", 1)]
    bm25 = [_doc("b.pdf", 1), _doc("a.pdf", 0)]

    result = rust_core.rrf_fusion_native(vector, bm25, 60.0, 1)
    assert _keys(result) == [("a.pdf", 0)]


def test_repeated_runs_produce_identical_order():
    # 相同输入多次融合必须顺序一致。
    vector = [_doc("a.pdf", i) for i in range(5)]
    bm25 = [_doc("a.pdf", i) for i in range(4, -1, -1)]

    first = _keys(rust_core.rrf_fusion_native(vector, bm25, 60.0, 5))
    for _ in range(10):
        again = _keys(
            rust_core.rrf_fusion_native(
                [_doc("a.pdf", i) for i in range(5)],
                [_doc("a.pdf", i) for i in range(4, -1, -1)],
                60.0,
                5,
            )
        )
        assert again == first


def test_same_doc_in_both_channels_is_merged():
    # 相同 chunk_id 的两路结果必须合并。
    vector = [_doc("a.pdf", 0)]
    bm25 = [_doc("a.pdf", 0)]

    result = rust_core.rrf_fusion_native(vector, bm25, 60.0, 5)
    assert len(result) == 1

    expected = 1.0 / (60.0 + 1) + 1.0 / (60.0 + 1)
    assert result[0]["retrieval"]["rrf_score"] == pytest.approx(expected)
    assert result[0]["retrieval"]["search_channel"] == "hybrid"


def test_chunk_id_is_the_rrf_identity_key():
    # source/chunk_index 相同但 chunk_id 不同时不能合并。
    vector = [_doc("same.pdf", 0)]
    bm25 = [_doc("same.pdf", 0)]
    bm25[0]["meta"]["chunk_id"] = "chunk:same.pdf:different"

    result = rust_core.rrf_fusion_native(vector, bm25, 60.0, 5)

    assert len(result) == 2
    assert {doc["meta"]["chunk_id"] for doc in result} == {
        "chunk:same.pdf:0",
        "chunk:same.pdf:different",
    }


def test_higher_rank_wins_when_scores_differ():
    # 融合得分更高的文档必须排在前面。
    vector = [_doc("b.pdf", 1), _doc("a.pdf", 0)]
    bm25 = [_doc("b.pdf", 1), _doc("a.pdf", 0)]

    result = rust_core.rrf_fusion_native(vector, bm25, 60.0, 2)
    assert _keys(result) == [("b.pdf", 1), ("a.pdf", 0)]
