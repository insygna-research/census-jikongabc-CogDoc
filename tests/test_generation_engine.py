import pytest
from collections import OrderedDict
from unittest.mock import MagicMock, patch
from cogdoc.config.settings import get_settings
from cogdoc.service.kb_epoch import EpochStore
from cogdoc.service.kb_state import KBState
from cogdoc.graph.subgraphs.qa import RetrieverFactory
from cogdoc.tools.retriever.base_retriever import NullRetriever, NullWriteError
from cogdoc.tools.retriever.hybrid import HybridRetriever, IndexCorruptError
from cogdoc.tools.retriever.vector_retriever import EmbeddingModelMismatchError


# 构造 make state 相关逻辑。
def _make_state(tmp_path, kb_id="kb"):
    epochs = EpochStore(path=str(tmp_path / "epochs.json"))
    return KBState(kb_id, path=str(tmp_path / kb_id / "state.json"), epochs=epochs)


# 处理 fresh factory 相关逻辑。
def _fresh_factory():
    # 每个测试独立清空缓存，conftest autouse fixture 在 yield 后也会清，两者不冲突。
    RetrieverFactory._engines = OrderedDict()


# ---- NullRetriever 契约 ----


# 验证 null retriever read safe。
def test_null_retriever_read_safe():
    nr = NullRetriever()
    assert nr.exists() is False
    assert nr.count() == 0
    assert nr.chunk_ids() == set()
    assert nr.max_chunk_index() == -1
    assert nr.search("q") == []
    assert nr.list_sources() == []
    assert nr.load_source_chunks("x") == []
    nr.clear()
    nr.delete_by_source(["x"])


# 验证 null retriever write raises。
def test_null_retriever_write_raises():
    # 写方法必须显式报错，不能静默 no-op，以便在 Phase 3 之前及早暴露误用。
    nr = NullRetriever()
    with pytest.raises(NullWriteError):
        nr.index([])
    with pytest.raises(NullWriteError):
        nr.add_documents([])
    with pytest.raises(NullWriteError):
        nr.upsert_documents([], set())


# 验证 null hybrid engine not corrupt。
def test_null_hybrid_engine_not_corrupt():
    # 两路 NullRetriever count 相等（0==0），is_corrupt=False，检索返回空列表。
    engine = HybridRetriever(NullRetriever(), NullRetriever())
    assert engine.is_corrupt() is False
    assert engine.count() == 0
    assert engine.search("q") == []


# ---- collection_id hash 命名 ----


# 验证 collection id deterministic。
def test_collection_id_deterministic():
    s = get_settings()
    assert s.kb_collection_id("kb1", "g123") == s.kb_collection_id("kb1", "g123")
    assert s.kb_collection_id("kb1", "g123") != s.kb_collection_id("kb2", "g123")


# 验证 collection id within chroma limit。
def test_collection_id_within_chroma_limit():
    # col-{8hex}-{gen_id(13)} = 4+8+1+13 = 26 字符，远低于 Chroma 60 字符上限。
    kb_id = "a" * 56  # 最长允许的 kb_id
    gen_id = "g" + "f" * 12
    collection_id = get_settings().kb_collection_id(kb_id, gen_id)
    assert len(f"col-{collection_id}") <= 60


# ---- _resolve_gen_id ----


# 验证 resolve gen id no active。
def test_resolve_gen_id_no_active(tmp_path):
    state = _make_state(tmp_path)
    with patch("cogdoc.graph.subgraphs.qa.KBState", return_value=state):
        assert RetrieverFactory._resolve_gen_id("kb") is None


# 验证 resolve gen id expected count zero。
def test_resolve_gen_id_expected_count_zero(tmp_path):
    state = _make_state(tmp_path)
    gen_id = state.begin_generation("m", "v")
    state.mark_ready(gen_id, expected_count=0, documents=[])
    state.switch_active(gen_id)
    with patch("cogdoc.graph.subgraphs.qa.KBState", return_value=state):
        assert RetrieverFactory._resolve_gen_id("kb") is None


# 验证 resolve gen id returns active id。
def test_resolve_gen_id_returns_active_id(tmp_path):
    state = _make_state(tmp_path)
    gen_id = state.begin_generation("m", "v")
    state.mark_ready(gen_id, expected_count=3, documents=[])
    state.switch_active(gen_id)
    with patch("cogdoc.graph.subgraphs.qa.KBState", return_value=state):
        assert RetrieverFactory._resolve_gen_id("kb") == gen_id


# ---- _build_engine ----


# 验证 build engine null when gen id none。
def test_build_engine_null_when_gen_id_none():
    engine = RetrieverFactory._build_engine("kb", None)
    assert engine.count() == 0
    assert engine.is_corrupt() is False


# 验证 build engine uses settings collection id。
def test_build_engine_uses_settings_collection_id(tmp_path):
    # 验证传入 VectorRetriever/BM25Retriever 的 collection_id 来自 settings.kb_collection_id。
    kb_id = "kb1"
    state = _make_state(tmp_path, kb_id)
    gen_id = state.begin_generation("m", "v")
    state.mark_ready(gen_id, expected_count=2, documents=[])
    state.switch_active(gen_id)

    expected_cid = get_settings().kb_collection_id(kb_id, gen_id)

    with (
        patch("cogdoc.graph.subgraphs.qa.KBState", return_value=state),
        patch("cogdoc.graph.subgraphs.qa.VectorRetriever") as MockVec,
        patch("cogdoc.graph.subgraphs.qa.BM25Retriever") as MockBm25,
    ):
        mock_engine = MagicMock(spec=HybridRetriever)
        mock_engine.count.return_value = 2
        mock_engine.is_consistent.return_value = True
        with patch("cogdoc.graph.subgraphs.qa.HybridRetriever", return_value=mock_engine):
            RetrieverFactory._build_engine(kb_id, gen_id)

    MockVec.assert_called_once_with(collection_id=expected_cid)
    MockBm25.assert_called_once_with(collection_id=expected_cid)


# 验证 build engine model mismatch returns null。
def test_build_engine_model_mismatch_returns_null(tmp_path):
    state = _make_state(tmp_path)
    gen_id = state.begin_generation("old", "v")
    state.mark_ready(gen_id, expected_count=3, documents=[])
    state.switch_active(gen_id)

    with (
        patch("cogdoc.graph.subgraphs.qa.KBState", return_value=state),
        patch(
            "cogdoc.graph.subgraphs.qa.VectorRetriever",
            side_effect=EmbeddingModelMismatchError("mismatch"),
        ),
    ):
        engine = RetrieverFactory._build_engine("kb", gen_id)

    assert engine.count() == 0
    assert engine.is_corrupt() is False


# 验证 build engine count mismatch raises。
def test_build_engine_count_mismatch_raises(tmp_path):
    # 磁盘数据丢失导致 count != expected_count：必须 raise IndexCorruptError，不能静默返回空库。
    state = _make_state(tmp_path)
    gen_id = state.begin_generation("m", "v")
    state.mark_ready(gen_id, expected_count=5, documents=[])
    state.switch_active(gen_id)

    with (
        patch("cogdoc.graph.subgraphs.qa.KBState", return_value=state),
        patch("cogdoc.graph.subgraphs.qa.VectorRetriever"),
        patch("cogdoc.graph.subgraphs.qa.BM25Retriever"),
    ):
        mock_engine = MagicMock(spec=HybridRetriever)
        mock_engine.count.return_value = 0  # 磁盘数据丢失，count 为 0
        mock_engine.is_consistent.return_value = True
        with patch("cogdoc.graph.subgraphs.qa.HybridRetriever", return_value=mock_engine):
            with pytest.raises(IndexCorruptError):
                RetrieverFactory._build_engine("kb", gen_id)


# 验证 build engine gen gcod returns null。
def test_build_engine_gen_gcod_returns_null():
    # 该代在构造期间已被 GC 回收（get() 返回 None）：返回空引擎，不报错。 get_engine 在插入前重解析 gen_id 不符，不会把这个空引擎写回缓存。
    with (
        patch("cogdoc.graph.subgraphs.qa.KBState") as MockKBState,
        patch("cogdoc.graph.subgraphs.qa.VectorRetriever"),
        patch("cogdoc.graph.subgraphs.qa.BM25Retriever"),
    ):
        mock_state = MagicMock()
        mock_state.get.return_value = None  # 已被 GC 回收
        MockKBState.return_value = mock_state
        engine = RetrieverFactory._build_engine("kb", "g_gone_0000000")

    # gen 已回收 → 真实空引擎（两路 NullRetriever）：count 为 0、不判损坏。
    assert engine.count() == 0
    assert engine.is_corrupt() is False


# 验证 build engine inconsistent raises。
def test_build_engine_inconsistent_raises(tmp_path):
    # count 正确但两路 chunk_id 集合不等（向量索引等量错块）：必须 raise IndexCorruptError。
    state = _make_state(tmp_path)
    gen_id = state.begin_generation("m", "v")
    state.mark_ready(gen_id, expected_count=3, documents=[])
    state.switch_active(gen_id)

    with (
        patch("cogdoc.graph.subgraphs.qa.KBState", return_value=state),
        patch("cogdoc.graph.subgraphs.qa.VectorRetriever"),
        patch("cogdoc.graph.subgraphs.qa.BM25Retriever"),
    ):
        mock_engine = MagicMock(spec=HybridRetriever)
        mock_engine.count.return_value = 3  # count 正确
        mock_engine.is_consistent.return_value = False  # 但 chunk_id 集合不等
        with patch("cogdoc.graph.subgraphs.qa.HybridRetriever", return_value=mock_engine):
            with pytest.raises(IndexCorruptError):
                RetrieverFactory._build_engine("kb", gen_id)


# ---- 缓存：(kb_id, gen_id) 键、invalidate、竞态保护 ----


# 验证 cache hit returns same instance。
def test_cache_hit_returns_same_instance(tmp_path):
    _fresh_factory()
    kb_id = "kb-hit"
    gen_id = "g123456789abc"
    sentinel = MagicMock(spec=HybridRetriever)
    with RetrieverFactory._lock:
        RetrieverFactory._engines[(kb_id, gen_id)] = sentinel

    # get_engine 解析到同一 gen_id 命中缓存。
    with patch.object(RetrieverFactory, "_resolve_gen_id", return_value=gen_id):
        engine = RetrieverFactory.get_engine(kb_id)
    assert engine is sentinel


# 验证 invalidate clears all entries for kb。
def test_invalidate_clears_all_entries_for_kb():
    _fresh_factory()
    kb_id = "kb-inv"
    sentinel_a = MagicMock(spec=HybridRetriever)
    sentinel_b = MagicMock(spec=HybridRetriever)
    # 同一 kb_id 的两个 generation 条目都要被清掉。
    with RetrieverFactory._lock:
        RetrieverFactory._engines[(kb_id, "g001")] = sentinel_a
        RetrieverFactory._engines[(kb_id, "g002")] = sentinel_b
        RetrieverFactory._engines[("other-kb", "g003")] = MagicMock()

    RetrieverFactory.invalidate(kb_id)
    with RetrieverFactory._lock:
        keys = list(RetrieverFactory._engines.keys())
    assert (kb_id, "g001") not in keys
    assert (kb_id, "g002") not in keys
    assert ("other-kb", "g003") in keys  # 其他 kb 不受影响


# 验证 race stale engine not cached。
def test_race_stale_engine_not_cached(tmp_path):
    # 模拟构造期间 gen_id 已切换（switch_active + invalidate）：旧引擎不写回缓存。
    _fresh_factory()
    kb_id = "kb-race"
    old_gen = "g_old_0000000"
    new_gen = "g_new_1111111"

    call_count = [0]

    # 解析 resolve side effect 相关逻辑。
    def resolve_side_effect(kb):
        call_count[0] += 1
        # 第一次（锁外解析）返回旧代；第二次（锁内插入前重解析）返回新代。
        return old_gen if call_count[0] == 1 else new_gen

    old_engine = MagicMock(spec=HybridRetriever)
    with (
        patch.object(
            RetrieverFactory, "_resolve_gen_id", side_effect=resolve_side_effect
        ),
        patch.object(RetrieverFactory, "_build_engine", return_value=old_engine),
    ):
        result = RetrieverFactory.get_engine(kb_id)

    # 旧引擎被返回（这次请求），但没有写入缓存。
    assert result is old_engine
    with RetrieverFactory._lock:
        assert (kb_id, old_gen) not in RetrieverFactory._engines
        assert (kb_id, new_gen) not in RetrieverFactory._engines
