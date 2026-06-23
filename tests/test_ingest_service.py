import types
from collections import OrderedDict
from service import ingest_service


class FakeEngine:
    def __init__(self):
        self.cleared = False
        self.indexed = None

    def clear(self):
        self.cleared = True

    def index(self, chunks):
        self.indexed = chunks


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


def test_invalidate_only_evicts_target_kb(monkeypatch):
    from graph.subgraphs.qa import RetrieverFactory

    sentinel_a, sentinel_b = object(), object()
    monkeypatch.setattr(RetrieverFactory, "_engines", OrderedDict())
    with RetrieverFactory._lock:
        RetrieverFactory._engines["kb-a"] = sentinel_a
        RetrieverFactory._engines["kb-b"] = sentinel_b

    RetrieverFactory.invalidate("kb-a")

    assert "kb-a" not in RetrieverFactory._engines
    # 重建 kb-a 不应波及 kb-b 的已缓存引擎。
    assert RetrieverFactory._engines["kb-b"] is sentinel_b


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
