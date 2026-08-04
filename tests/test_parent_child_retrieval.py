from contextlib import nullcontext

from cogdoc.config.settings import Settings
from cogdoc.graph.subgraphs import qa


def _child(index: int, *, parent: str | None = "section:1") -> dict:
    meta = {
        "chunk_id": f"c{index}",
        "source": "paper.pdf",
        "page": index + 1,
        "chunk_index": index,
    }
    if parent is not None:
        meta.update(
            {
                "parent_chunk_id": parent,
                "section_title": "Methods",
                "section_path": "Paper > Methods",
                "section_level": 1,
                "child_index_in_parent": index,
            }
        )
    return {"text": f"child-{index}", "meta": meta}


class _Engine:
    def __init__(self, chunks):
        self.chunks = chunks

    def load_source_chunks(self, source):
        assert source == "paper.pdf"
        return self.chunks


def _install(monkeypatch, chunks, **settings_overrides):
    values = {
        "qa_parent_context_enabled": True,
        "qa_parent_context_max_chunks": 3,
        "qa_parent_context_max_chars": 1000,
        **settings_overrides,
    }
    settings = Settings(_env_file=None, **values)
    monkeypatch.setattr(qa, "get_settings", lambda: settings)
    monkeypatch.setattr(qa, "kb_read_lease", lambda _kb_id: nullcontext())
    monkeypatch.setattr(
        qa.RetrieverFactory, "get_engine", lambda _kb_id: _Engine(chunks)
    )


# 同 section 的上下文以独立 child 进入生成闭集，anchor 的重排指标不被覆盖。
def test_parent_context_expansion_preserves_child_identities(monkeypatch):
    chunks = [_child(index) for index in range(5)]
    _install(monkeypatch, chunks)
    anchor = _child(2)
    anchor["retrieval"] = {"rerank_score": 0.93, "search_channel": "hybrid"}

    expanded = qa._expand_with_neighbor_chunks("kb", [anchor])

    assert [doc["meta"]["chunk_id"] for doc in expanded] == ["c1", "c2", "c3"]
    assert [doc["meta"]["page"] for doc in expanded] == [2, 3, 4]
    assert expanded[1]["retrieval"] == {
        "rerank_score": 0.93,
        "search_channel": "hybrid",
    }
    assert expanded[0]["retrieval"] == {
        "search_channel": "parent_context",
        "context_anchor_chunk_id": "c2",
        "context_expansion": "section",
    }


# 旧索引无 parent 元数据时继续使用原来的前后一个邻块兼容路径。
def test_parent_context_expansion_falls_back_to_legacy_neighbors(monkeypatch):
    chunks = [_child(index, parent=None) for index in range(4)]
    _install(monkeypatch, chunks)
    anchor = _child(1, parent=None)

    expanded = qa._expand_with_neighbor_chunks("kb", [anchor])

    assert [doc["meta"]["chunk_id"] for doc in expanded] == ["c0", "c1", "c2"]
    assert expanded[0]["retrieval"] == {
        "search_channel": "neighbor",
        "context_anchor_chunk_id": "c1",
        "context_expansion": "neighbor",
    }


# 关闭结构扩展只关闭新能力，不改变旧邻块上下文行为。
def test_disabling_parent_context_keeps_neighbor_compatibility(monkeypatch):
    chunks = [_child(index) for index in range(4)]
    _install(monkeypatch, chunks, qa_parent_context_enabled=False)

    expanded = qa._expand_with_neighbor_chunks("kb", [_child(1)])

    assert [doc["meta"]["chunk_id"] for doc in expanded] == ["c0", "c1", "c2"]
    assert expanded[2]["retrieval"]["context_expansion"] == "neighbor"


# 生成前 verifier 先保留 anchor，再在原有文档数预算内看到同章节 sibling。
def test_rerank_exposes_parent_context_to_bounded_verification(monkeypatch):
    chunks = [_child(index) for index in range(5)]
    _install(monkeypatch, chunks)
    monkeypatch.setattr(qa.BGEReranker, "default_device", lambda: "cpu")
    monkeypatch.setattr(qa, "log_event", lambda *args, **kwargs: None)
    anchor = _child(2)
    anchor["retrieval"] = {"bm25_score": 12.0}

    output = qa.rerank_node(
        {"query": "训练阶段是什么？", "doc_id": "kb", "retrieved_docs": [anchor]}
    )

    assert [doc["meta"]["chunk_id"] for doc in output["verification_docs"]] == [
        "c2",
        "c1",
        "c3",
    ]
    assert {doc["meta"]["chunk_id"] for doc in output["verification_docs"]} <= {
        doc["meta"]["chunk_id"] for doc in output["reranked_docs"]
    }
    assert output["parent_context_expanded_count"] == 2
    assert output["neighbor_context_expanded_count"] == 0
    assert output["evidence_pack_kept_count"] == 3
    assert output["evidence_pack_over_budget"] is False
    assert output["evidence_pack_kept_chars"] == len(
        qa.Generator._build_context_string(output["reranked_docs"])
    )


# 默认五块生成窗口被三块 verifier 预算裁剪时仍保留 anchor 两侧最近证据。
def test_verification_budget_keeps_nearest_siblings_on_both_sides(monkeypatch):
    chunks = [_child(index) for index in range(5)]
    _install(monkeypatch, chunks, qa_parent_context_max_chunks=5)
    monkeypatch.setattr(qa.BGEReranker, "default_device", lambda: "cpu")
    monkeypatch.setattr(qa, "log_event", lambda *args, **kwargs: None)
    anchor = _child(2)
    anchor["retrieval"] = {"bm25_score": 12.0}

    output = qa.rerank_node(
        {"query": "训练阶段是什么？", "doc_id": "kb", "retrieved_docs": [anchor]}
    )

    assert [doc["meta"]["chunk_id"] for doc in output["verification_docs"]] == [
        "c2",
        "c1",
        "c3",
    ]


# 同一 parent 多个 anchor 的重叠窗口合并后保持章节顺序，不随命中顺序旋转。
def test_multiple_anchors_keep_parent_children_in_document_order(monkeypatch):
    chunks = [_child(index) for index in range(5)]
    _install(monkeypatch, chunks)

    expanded = qa._expand_with_neighbor_chunks("kb", [_child(3), _child(1)])

    assert [doc["meta"]["chunk_id"] for doc in expanded] == [
        "c0",
        "c1",
        "c2",
        "c3",
        "c4",
    ]


def test_evidence_pack_keeps_multi_anchor_generation_in_document_order(monkeypatch):
    chunks = [_child(index) for index in range(5)]
    _install(
        monkeypatch,
        chunks,
        qa_rerank_top_n=2,
        qa_evidence_pack_max_docs=5,
    )
    monkeypatch.setattr(qa.BGEReranker, "default_device", lambda: "cpu")
    monkeypatch.setattr(qa, "log_event", lambda *args, **kwargs: None)
    anchors = [_child(3), _child(1)]
    for anchor in anchors:
        anchor["retrieval"] = {"bm25_score": 12.0}

    output = qa.rerank_node(
        {"query": "训练阶段是什么？", "doc_id": "kb", "retrieved_docs": anchors}
    )

    assert [doc["meta"]["chunk_id"] for doc in output["reranked_docs"]] == [
        "c0",
        "c1",
        "c2",
        "c3",
        "c4",
    ]


def test_evidence_pack_hard_budget_excess_fails_closed(monkeypatch):
    anchor = _child(0)
    anchor["text"] = "x" * 600
    anchor["retrieval"] = {"bm25_score": 12.0}
    _install(
        monkeypatch,
        [anchor],
        qa_evidence_pack_max_docs=1,
        qa_evidence_pack_max_chars=500,
        qa_adaptive_retrieval_enabled=True,
        qa_adaptive_retrieval_max_retries=1,
    )
    monkeypatch.setattr(qa.BGEReranker, "default_device", lambda: "cpu")
    monkeypatch.setattr(qa, "log_event", lambda *args, **kwargs: None)

    state = {
        "query": "A 和 B 的日期分别是什么？",
        "doc_id": "kb",
        "retrieved_docs": [anchor],
        "evidence_requirements": [
            {"requirement_id": "r1", "question": "A 的日期"},
            {"requirement_id": "r2", "question": "B 的日期"},
        ],
    }
    output = qa.rerank_node(state)

    assert output["evidence_pack_over_budget"] is True
    assert output["retrieval_abstained"] is True
    assert output["retrieval_abstain_reason"] == "evidence_pack_hard_budget_exceeded"
    assert output["verification_docs"] == []
    assert output["evidence_verification_pending"] is False
    assert output["adaptive_retrieval_retry_pending"] is False
    assert qa.retrieval_check({**state, **output}) == "abstain_node"


def test_rerank_selects_one_traceable_span_for_the_shared_evidence_set(monkeypatch):
    target = "报名截止日期为 2026 年 8 月 30 日。"
    original_text = f"{'无关背景。' * 40}{target}{'附录说明。' * 40}"
    anchor = _child(0)
    anchor["text"] = original_text
    anchor["meta"]["context"] = "只存在于旧定位上下文的秘密事实"
    anchor["retrieval"] = {
        "bm25_score": 12.0,
        "matched_requirement_ids": ["r1"],
    }
    _install(
        monkeypatch,
        [anchor],
        qa_evidence_span_max_chars_per_doc=120,
        qa_evidence_span_context_sentences=0,
    )
    monkeypatch.setattr(qa.BGEReranker, "default_device", lambda: "cpu")
    monkeypatch.setattr(qa, "log_event", lambda *args, **kwargs: None)

    output = qa.rerank_node(
        {
            "query": "报名截止日期是什么？",
            "doc_id": "kb",
            "retrieved_docs": [anchor],
            "evidence_requirements": [
                {
                    "requirement_id": "r1",
                    "question": "报名截止日期是什么？",
                    "retrieval_query": "报名 截止 日期",
                    "recovery_query": "报名日期",
                }
            ],
        }
    )

    packed = output["reranked_docs"][0]
    assert packed["text"] == target
    assert packed["meta"].get("context") is None
    assert output["verification_docs"][0]["text"] == target
    assert output["evidence_span_input_count"] == 1
    assert output["evidence_span_output_count"] == 1
    assert output["evidence_span_compressed_count"] == 1
    assert output["evidence_span_fallback_count"] == 0
    assert output["evidence_span_selected_chars"] == len(target)
    assert output["evidence_span_input_chars"] == len(original_text)
    assert output["evidence_span_reason_counts"] == {"query_span": 1}
    retrieval = packed["retrieval"]
    start = original_text.index(target)
    assert retrieval["evidence_span_start"] == start
    assert retrieval["evidence_span_end"] == start + len(target)
    assert retrieval["evidence_text_start"] == start
    assert retrieval["evidence_text_end"] == start + len(target)
    assert retrieval["matched_requirement_ids"] == ["r1"]
    public_evidence = qa._generation_evidence(packed)
    assert public_evidence["retrieval"]["evidence_span_input_start"] == 0
    assert public_evidence["retrieval"]["evidence_span_start"] == start
    assert public_evidence["retrieval"]["evidence_span_matched_requirement_ids"] == [
        "r1"
    ]
    assert not any(key.startswith("_evidence_") for key in public_evidence)
    assert anchor["text"] == original_text
    assert anchor["meta"]["context"] == "只存在于旧定位上下文的秘密事实"


def test_disabling_evidence_spans_keeps_the_full_pack_view(monkeypatch):
    anchor = _child(0)
    anchor["text"] = "完整正文。" * 40
    anchor["meta"]["context"] = "定位上下文"
    anchor["retrieval"] = {"bm25_score": 12.0}
    _install(monkeypatch, [anchor], qa_evidence_span_enabled=False)
    monkeypatch.setattr(qa.BGEReranker, "default_device", lambda: "cpu")
    monkeypatch.setattr(qa, "log_event", lambda *args, **kwargs: None)

    output = qa.rerank_node(
        {"query": "完整正文", "doc_id": "kb", "retrieved_docs": [anchor]}
    )

    assert output["reranked_docs"][0]["text"] == anchor["text"]
    assert output["reranked_docs"][0]["meta"]["context"] == "定位上下文"
    assert output["evidence_span_input_count"] == 0
    assert output["evidence_span_output_count"] == 0


def test_generation_evidence_omits_fake_structure_but_keeps_real_zero_indexes():
    legacy = qa._generation_evidence(_child(0, parent=None))
    structured = qa._generation_evidence(_child(0))

    assert "parent_chunk_id" not in legacy
    assert "section_level" not in legacy
    assert "child_index_in_parent" not in legacy
    assert structured["parent_chunk_id"] == "section:1"
    assert structured["child_index_in_parent"] == 0
