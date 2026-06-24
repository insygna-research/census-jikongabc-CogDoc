import pytest
from unittest.mock import MagicMock
from cogdoc.service.kb_epoch import EpochStore
from cogdoc.service.kb_state import KBState
from cogdoc.service import ingest_service
from cogdoc.service.ingest_service import (
    INDEX_BUILD_VERSION,
    IncrementalPlan,
    IndexInconsistencyError,
    _populate_staging,
    _fill_staging_incremental,
)
from cogdoc.tools.chunk_identity import build_chunk_id
from cogdoc.tools.embedder import Embedder


# 构造 make state 相关逻辑。
def _make_state(tmp_path, kb_id="kb"):
    epochs = EpochStore(path=str(tmp_path / "epochs.json"))
    return KBState(kb_id, path=str(tmp_path / kb_id / "state.json"), epochs=epochs)


# 处理 reg doc 相关逻辑。
def _reg_doc(source, sha, local_idx, chunk_index, page_start=1, page_end=1):
    # BM25 registry 形态的自洽复用 chunk：chunk_id 由 hash/name/页跨度/局部序号真实构建。
    chunk_id = build_chunk_id(sha, source, page_start, page_end, local_idx)
    return {
        "text": f"text-{chunk_id}",
        "meta": {
            "chunk_id": chunk_id,
            "source": source,
            "source_sha256": sha,
            "local_chunk_index": local_idx,
            "chunk_index": chunk_index,
            "page": page_start,
            "page_start": page_start,
            "page_end": page_end,
            "origin": "file",
        },
    }


# 处理 docs 相关逻辑。
def _docs(*pairs):
    return [{"name": n, "sha256": h} for n, h in pairs]


# 准备 seed active 相关逻辑。
def _seed_active(state, documents, registry):
    gen_id = state.begin_generation(Embedder.MODEL_NAME, INDEX_BUILD_VERSION)
    state.mark_ready(gen_id, expected_count=len(registry), documents=documents)
    state.switch_active(gen_id)
    return gen_id


# 配置测试替身 patch prev stores 相关逻辑。
def _patch_prev_stores(monkeypatch, registry, embeddings):
    fake_vec = MagicMock()
    fake_vec.embeddings_by_chunk_id.return_value = dict(embeddings)
    fake_bm25 = MagicMock()
    fake_bm25.export_registry.return_value = [dict(d) for d in registry]
    monkeypatch.setattr(
        ingest_service, "VectorRetriever", lambda collection_id: fake_vec
    )
    monkeypatch.setattr(
        ingest_service, "BM25Retriever", lambda collection_id: fake_bm25
    )
    return fake_vec, fake_bm25


# 处理 emb for 相关逻辑。
def _emb_for(registry):
    return {d["meta"]["chunk_id"]: [0.1] for d in registry}


# 处理 manifest 相关逻辑。
def _manifest(kb_id, documents, build_version=INDEX_BUILD_VERSION):
    return {
        "doc_id": kb_id,
        "index_build_version": build_version,
        "documents": documents,
    }


# ---- 未修改文档：复用向量，绝不重算 embedding ----


# 验证 unchanged docs reuse without embedding。
def test_unchanged_docs_reuse_without_embedding(tmp_path, monkeypatch):
    kb_id = "kb-reuse"
    state = _make_state(tmp_path, kb_id)
    documents = _docs(("a.pdf", "H1"))
    registry = [_reg_doc("a.pdf", "H1", 0, 0)]
    _seed_active(state, documents, registry)
    _patch_prev_stores(monkeypatch, registry, _emb_for(registry))

    parsed_names = []
    monkeypatch.setattr(
        ingest_service,
        "_parse_and_chunk",
        lambda gdir, names, hmap, **kw: parsed_names.append(list(names)) or ([], []),
    )

    staging = MagicMock()
    all_chunks, _ = _populate_staging(
        kb_id, state, "/gen", ["a.pdf"], _manifest(kb_id, documents), {}, staging
    )

    assert parsed_names == [[]]
    staging.vector_retriever.add_with_embeddings.assert_called_once()
    staging.vector_retriever.add_documents.assert_not_called()
    assert {c["meta"]["chunk_id"] for c in all_chunks} == {
        registry[0]["meta"]["chunk_id"]
    }


# ---- 仅新增/修改文档调用 embedding ----


# 验证 new doc goes through embedding。
def test_new_doc_goes_through_embedding(tmp_path, monkeypatch):
    kb_id = "kb-add"
    state = _make_state(tmp_path, kb_id)
    prev_docs = _docs(("a.pdf", "H1"))
    registry = [_reg_doc("a.pdf", "H1", 0, 0)]
    _seed_active(state, prev_docs, registry)
    _patch_prev_stores(monkeypatch, registry, _emb_for(registry))

    new_chunk = {
        "text": "nt",
        "meta": {"chunk_id": "c-new", "source": "b.pdf", "chunk_index": 1},
    }
    parsed_names = []
    monkeypatch.setattr(
        ingest_service,
        "_parse_and_chunk",
        lambda gdir, names, hmap, **kw: (
            parsed_names.append(list(names)) or ([new_chunk], [])
        ),
    )

    cur_docs = prev_docs + _docs(("b.pdf", "H2"))
    staging = MagicMock()
    all_chunks, _ = _populate_staging(
        kb_id,
        state,
        "/gen",
        ["a.pdf", "b.pdf"],
        _manifest(kb_id, cur_docs),
        {},
        staging,
    )

    assert parsed_names == [["b.pdf"]]
    reused = staging.vector_retriever.add_with_embeddings.call_args.args[0]
    added = staging.vector_retriever.add_documents.call_args.args[0]
    assert {c["meta"]["chunk_id"] for c in reused} == {registry[0]["meta"]["chunk_id"]}
    assert {c["meta"]["chunk_id"] for c in added} == {"c-new"}


# ---- 纯删除：零 embedding ----


# 验证 pure delete zero embedding。
def test_pure_delete_zero_embedding(tmp_path, monkeypatch):
    kb_id = "kb-del"
    state = _make_state(tmp_path, kb_id)
    prev_docs = _docs(("a.pdf", "H1"), ("b.pdf", "H2"))
    registry = [_reg_doc("a.pdf", "H1", 0, 0), _reg_doc("b.pdf", "H2", 0, 1)]
    _seed_active(state, prev_docs, registry)
    _patch_prev_stores(monkeypatch, registry, _emb_for(registry))

    parsed_names = []
    monkeypatch.setattr(
        ingest_service,
        "_parse_and_chunk",
        lambda gdir, names, hmap, **kw: parsed_names.append(list(names)) or ([], []),
    )

    cur_docs = _docs(("a.pdf", "H1"))
    staging = MagicMock()
    all_chunks, _ = _populate_staging(
        kb_id, state, "/gen", ["a.pdf"], _manifest(kb_id, cur_docs), {}, staging
    )

    assert parsed_names == [[]]
    staging.vector_retriever.add_documents.assert_not_called()
    assert {c["meta"]["chunk_id"] for c in all_chunks} == {
        registry[0]["meta"]["chunk_id"]
    }


# ---- 旧 Vector/BM25 同数量但 ID 不同：识破并回退全量 ----


# 验证 diverged stores fallback to full。
def test_diverged_stores_fallback_to_full(tmp_path, monkeypatch):
    kb_id = "kb-diverge"
    state = _make_state(tmp_path, kb_id)
    documents = _docs(("a.pdf", "H1"))
    registry = [_reg_doc("a.pdf", "H1", 0, 0)]
    _seed_active(state, documents, registry)
    _patch_prev_stores(
        monkeypatch, registry, {"cX": [0.1]}
    )  # 向量 ID 与 BM25 不同，同为 1 个

    full = [_reg_doc("a.pdf", "H1", 0, 0)]
    monkeypatch.setattr(ingest_service, "_parse_and_chunk", lambda *a, **k: (full, []))
    monkeypatch.setattr(ingest_service, "log_event", lambda *a, **k: None)

    staging = MagicMock()
    all_chunks, _ = _populate_staging(
        kb_id, state, "/gen", ["a.pdf"], _manifest(kb_id, documents), {}, staging
    )

    staging.clear.assert_called_once()
    staging.index.assert_called_once_with(full)
    assert all_chunks == full


# 验证 diverged stores raise in fill。
def test_diverged_stores_raise_in_fill(tmp_path, monkeypatch):
    kb_id = "kb-diverge2"
    state = _make_state(tmp_path, kb_id)
    documents = _docs(("a.pdf", "H1"))
    registry = [_reg_doc("a.pdf", "H1", 0, 0)]
    _seed_active(state, documents, registry)
    _patch_prev_stores(monkeypatch, registry, {"cX": [0.1]})

    prev_active = state.active()
    with pytest.raises(IndexInconsistencyError, match="diverge"):
        _fill_staging_incremental(
            kb_id, MagicMock(), prev_active, IncrementalPlan([], set()), "/gen", {}
        )


# ---- BM25 同 ID 同数量但内容损坏：source/hash/metadata 自洽校验识破 ----


# 验证 reuse rejects source not in active。
def test_reuse_rejects_source_not_in_active(tmp_path, monkeypatch):
    kb_id = "kb-badsrc"
    state = _make_state(tmp_path, kb_id)
    documents = _docs(("a.pdf", "H1"))
    # registry 的 source 不在 active documents 中（损坏）。
    registry = [_reg_doc("ghost.pdf", "H1", 0, 0)]
    _seed_active(state, documents, registry)
    _patch_prev_stores(monkeypatch, registry, _emb_for(registry))

    prev_active = state.active()
    with pytest.raises(IndexInconsistencyError, match="source/hash"):
        _fill_staging_incremental(
            kb_id, MagicMock(), prev_active, IncrementalPlan([], set()), "/gen", {}
        )


# 验证 reuse rejects sha mismatch。
def test_reuse_rejects_sha_mismatch(tmp_path, monkeypatch):
    kb_id = "kb-badsha"
    state = _make_state(tmp_path, kb_id)
    documents = _docs(("a.pdf", "H1"))
    registry = [_reg_doc("a.pdf", "H1", 0, 0)]
    _seed_active(state, documents, registry)
    # 篡改 registry 内 source_sha256，与 active documents 不一致。
    registry[0]["meta"]["source_sha256"] = "TAMPERED"
    _patch_prev_stores(monkeypatch, registry, _emb_for(registry))

    prev_active = state.active()
    with pytest.raises(IndexInconsistencyError, match="source/hash"):
        _fill_staging_incremental(
            kb_id, MagicMock(), prev_active, IncrementalPlan([], set()), "/gen", {}
        )


# 验证 reuse rejects chunk id metadata mismatch。
def test_reuse_rejects_chunk_id_metadata_mismatch(tmp_path, monkeypatch):
    kb_id = "kb-badid"
    state = _make_state(tmp_path, kb_id)
    documents = _docs(("a.pdf", "H1"))
    registry = [_reg_doc("a.pdf", "H1", 0, 0)]
    _seed_active(state, documents, registry)
    # chunk_id 与 metadata 的页跨度不再自洽（chunk_id 编码了 page span/local index）。
    registry[0]["meta"]["page_end"] = 99
    _patch_prev_stores(monkeypatch, registry, _emb_for(registry))

    prev_active = state.active()
    with pytest.raises(IndexInconsistencyError, match="chunk_id/metadata"):
        _fill_staging_incremental(
            kb_id, MagicMock(), prev_active, IncrementalPlan([], set()), "/gen", {}
        )


# ---- 复用部分写入失败：清空并全量回退 ----


# 验证 partial write clears and full rebuild。
def test_partial_write_clears_and_full_rebuild(tmp_path, monkeypatch):
    kb_id = "kb-partial"
    state = _make_state(tmp_path, kb_id)
    documents = _docs(("a.pdf", "H1"))
    registry = [_reg_doc("a.pdf", "H1", 0, 0)]
    _seed_active(state, documents, registry)
    _patch_prev_stores(monkeypatch, registry, _emb_for(registry))

    full = [_reg_doc("a.pdf", "H1", 0, 0)]
    monkeypatch.setattr(ingest_service, "_parse_and_chunk", lambda *a, **k: (full, []))
    monkeypatch.setattr(ingest_service, "log_event", lambda *a, **k: None)

    staging = MagicMock()
    staging.vector_retriever.add_with_embeddings.side_effect = RuntimeError("disk full")

    all_chunks, _ = _populate_staging(
        kb_id, state, "/gen", ["a.pdf"], _manifest(kb_id, documents), {}, staging
    )

    staging.clear.assert_called_once()
    staging.index.assert_called_once_with(full)
    assert all_chunks == full


# ---- 模型契约变化强制全量构建 ----


# 验证 contract change forces full build。
def test_contract_change_forces_full_build(tmp_path, monkeypatch):
    kb_id = "kb-contract"
    state = _make_state(tmp_path, kb_id)
    documents = _docs(("a.pdf", "H1"))
    registry = [_reg_doc("a.pdf", "H1", 0, 0)]
    _seed_active(state, documents, registry)
    fake_vec, _ = _patch_prev_stores(monkeypatch, registry, _emb_for(registry))

    full = [_reg_doc("a.pdf", "H1", 0, 0)]
    monkeypatch.setattr(ingest_service, "_parse_and_chunk", lambda *a, **k: (full, []))

    changed = _manifest(kb_id, documents, build_version="DIFFERENT-VERSION")
    staging = MagicMock()
    _populate_staging(kb_id, state, "/gen", ["a.pdf"], changed, {}, staging)

    fake_vec.embeddings_by_chunk_id.assert_not_called()
    staging.index.assert_called_once_with(full)
    staging.vector_retriever.add_with_embeddings.assert_not_called()


# ---- 嵌入契约版本进入构建门控 ----


# 验证 embedding contract in build version。
def test_embedding_contract_in_build_version():
    assert Embedder.EMBEDDING_CONTRACT_VERSION in INDEX_BUILD_VERSION
    assert Embedder.MODEL_NAME in Embedder.EMBEDDING_CONTRACT_VERSION
    assert "dim=" in Embedder.EMBEDDING_CONTRACT_VERSION
