import random
import pytest
from rank_bm25 import BM25Okapi
from cogdoc.tools.rust_core_loader import ensure_rust_core

_rust_core = ensure_rust_core("Bm25Index")


# 构造或驱动 pythontopk 测试场景。
def _python_topk(corpus, query, top_k):
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(i, float(scores[i])) for i in order]


# 构建 corpus。
def _build_corpus(seed, size):
    rng = random.Random(seed)
    vocab = [
        "模型",
        "方法",
        "架构",
        "检索",
        "向量",
        "bm25",
        "rrf",
        "chunk",
        "page",
        "摘要",
        "对比",
        "实验",
        "结论",
        "数据",
        "指标",
    ]
    return [[rng.choice(vocab) for _ in range(rng.randint(3, 25))] for _ in range(size)]


# 验证 native bm25 matches rank bm25 ranking 场景。
@pytest.mark.parametrize(
    "query",
    [
        ["模型", "方法"],
        ["bm25", "rrf", "向量"],
        ["检索"],
        ["摘要", "摘要", "对比"],
        ["不存在词", "模型"],
        ["数据", "指标", "结论", "实验"],
    ],
)
def test_native_bm25_matches_rank_bm25_ranking(query):
    corpus = _build_corpus(seed=7, size=120)
    index = _rust_core.Bm25Index(corpus)

    native = list(index.score_topk(query, 10))
    reference = _python_topk(corpus, query, 10)

    assert [doc_id for doc_id, _ in native] == [doc_id for doc_id, _ in reference]
    for (_, native_score), (_, ref_score) in zip(native, reference):
        assert native_score == pytest.approx(ref_score, abs=1e-9)


# 验证 native bm25 handles empty corpus 场景。
def test_native_bm25_handles_empty_corpus():
    index = _rust_core.Bm25Index([])
    assert list(index.score_topk(["模型"], 5)) == []


# 验证 native bm25 serialization roundtrip preserves scores 场景。
def test_native_bm25_serialization_roundtrip_preserves_scores():
    corpus = _build_corpus(seed=11, size=80)
    index = _rust_core.Bm25Index(corpus)
    restored = _rust_core.Bm25Index.from_bytes(index.to_bytes())

    query = ["模型", "检索", "向量", "不存在词"]
    assert list(index.score_topk(query, 10)) == list(restored.score_topk(query, 10))


# 验证 native bm25 from bytes rejects garbage 场景。
def test_native_bm25_from_bytes_rejects_garbage():
    with pytest.raises(ValueError):
        _rust_core.Bm25Index.from_bytes(b"not a valid index")


# 验证 native bm25 rebuild from kept matches full rebuild 场景。
def test_native_bm25_rebuild_from_kept_matches_full_rebuild():
    corpus = _build_corpus(seed=3, size=60)
    index = _rust_core.Bm25Index(corpus)

    # 丢弃偶数下标文档，再追加两篇新文档：等价于按相同顺序的全量重建。
    keep_indices = [i for i in range(len(corpus)) if i % 2 == 1]
    new_tokens = [["模型", "新增", "检索"], ["摘要", "对比"]]
    incremental = index.rebuild_from_kept(keep_indices, new_tokens)

    expected_corpus = [corpus[i] for i in keep_indices] + new_tokens
    full = _rust_core.Bm25Index(expected_corpus)

    query = ["模型", "检索", "摘要"]
    assert list(incremental.score_topk(query, 10)) == list(full.score_topk(query, 10))


# 验证 native bm25 rebuild rejects out of range index 场景。
def test_native_bm25_rebuild_rejects_out_of_range_index():
    index = _rust_core.Bm25Index(_build_corpus(seed=5, size=10))
    with pytest.raises(ValueError):
        index.rebuild_from_kept([99], [])
