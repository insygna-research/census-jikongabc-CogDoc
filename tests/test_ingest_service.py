import types
from collections import OrderedDict
from service import ingest_service


# 封装 FakeEngine 的状态与行为。
class FakeEngine:
    # 初始化实例状态。
    def __init__(self):
        self.cleared = False
        self.indexed = None

    # 清理 clear 相关逻辑。
    def clear(self):
        self.cleared = True

    # 写入索引 index 相关逻辑。
    def index(self, chunks):
        self.indexed = chunks

    # 处理 is consistent 相关逻辑。
    def is_consistent(self):
        # 全量重建后写后校验：假引擎视为一致。
        return True


# 配置测试替身 patch common 相关逻辑。
def _patch_common(monkeypatch, engine, pages, chunks):
    monkeypatch.setattr(
        ingest_service.RetrieverFactory, "get_engine", lambda kb: engine
    )
    monkeypatch.setattr(ingest_service, "_invalidate_engine_cache", lambda kb: None)
    monkeypatch.setattr(
        ingest_service,
        "ensure_rust_core",
        lambda *s: types.SimpleNamespace(
            scan_pdf_manifest_native=lambda kb, d: {
                "documents": [{"name": "a.pdf", "sha256": "sha"}]
            }
        ),
    )
    monkeypatch.setattr(ingest_service, "stamp_chunk_identity_contract", lambda m: m)
    monkeypatch.setattr(ingest_service, "smart_parse", lambda p: pages)
    monkeypatch.setattr(
        ingest_service, "chunk_paper", lambda pages, source_sha256: chunks
    )
    saved = {}
    monkeypatch.setattr(
        ingest_service, "save_index_manifest", lambda m: saved.update(manifest=m)
    )
    return saved


# 验证 build kb index clears stale index when no chunks。
def test_build_kb_index_clears_stale_index_when_no_chunks(tmp_path, monkeypatch):
    src = tmp_path / "sources"
    src.mkdir()
    (src / "a.pdf").write_bytes(b"%PDF-1.4")
    engine = FakeEngine()
    saved = _patch_common(monkeypatch, engine, pages=[], chunks=[])

    result = ingest_service.build_kb_index("kb", str(src))

    assert engine.cleared is True
    assert engine.indexed is None
    assert result.document_count == 1 and result.chunk_count == 0
    # manifest 仍落盘，避免每次都判 stale 反复重建。
    assert "manifest" in saved


# 验证 invalidate only evicts target kb。
def test_invalidate_only_evicts_target_kb(monkeypatch):
    from graph.subgraphs.qa import RetrieverFactory

    sentinel_a, sentinel_b = object(), object()
    monkeypatch.setattr(RetrieverFactory, "_engines", OrderedDict())
    # 缓存键格式为 (kb_id, gen_id)，使用 None 代表「当前无活跃代」的条目。
    with RetrieverFactory._lock:
        RetrieverFactory._engines[("kb-a", None)] = sentinel_a
        RetrieverFactory._engines[("kb-b", None)] = sentinel_b

    RetrieverFactory.invalidate("kb-a")

    assert ("kb-a", None) not in RetrieverFactory._engines
    # 重建 kb-a 不应波及 kb-b 的已缓存引擎。
    assert RetrieverFactory._engines[("kb-b", None)] is sentinel_b


# 验证 build kb index indexes when chunks present。
def test_build_kb_index_indexes_when_chunks_present(tmp_path, monkeypatch):
    src = tmp_path / "sources"
    src.mkdir()
    (src / "a.pdf").write_bytes(b"%PDF-1.4")
    engine = FakeEngine()
    chunks = [{"meta": {}, "text": "正文"}]
    _patch_common(monkeypatch, engine, pages=[object()], chunks=chunks)

    result = ingest_service.build_kb_index("kb", str(src))

    assert engine.cleared is False
    assert engine.indexed == chunks
    assert result.chunk_count == 1
