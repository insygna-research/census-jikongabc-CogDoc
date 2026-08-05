from cogdoc.api.retrieval_feedback_store import (
    RetrievalFeedbackStore,
    normalize_query_text,
    query_hash,
)


# 验证查询归一化和哈希稳定。
def test_query_hash_normalizes_width_space_and_ascii_case():
    assert normalize_query_text("  ＡBC   问题  ") == "abc 问题"
    assert query_hash("ABC 问题") == query_hash("  Ａbc   问题 ")


# 验证反馈生成检索调权并支持禁用。
def test_record_from_feedback_and_disable(tmp_path):
    store = RetrievalFeedbackStore(path=str(tmp_path / "retrieval_feedback.jsonl"))
    records = store.record_from_feedback(
        "fb1",
        {
            "kb_id": "kb",
            "query": "报名要求",
            "feedback": "thumbs_down",
            "feedback_type": "bad_retrieval",
            "trace_id": "t1",
            "citations": [{"chunk_id": "c1", "source_type": "document"}],
            "evidence": [{"chunk_id": "c1"}, {"chunk_id": "c2"}],
        },
    )

    assert len(records) == 1
    assert records[0]["chunk_id"] == "c1"
    assert [row["chunk_id"] for row in records[0]["target_chunks"]] == ["c1", "c2"]
    boosts = store.boosts_for_query("kb", "报名要求")
    assert boosts == {"c1": -0.35, "c2": -0.35}
    assert store.counts(kb_id="kb") == {"total": 1, "enabled": 1, "disabled": 0}
    assert store.list(kb_id="kb")[0]["chunk_count"] == 2

    store.set_enabled(records[0]["retrieval_feedback_id"], False, actor="admin")

    assert store.boosts_for_query("kb", "报名要求") == {}


def _ledger_entry(chunk_id, answer, *, evidence_id="E001", index=0):
    citation = "[a.pdf:P1]"
    start = answer.index(citation)
    return {
        "evidence_id": evidence_id,
        "chunk_id": chunk_id,
        "source_type": "document",
        "source": "a.pdf",
        "page": 1,
        "span_start": 0,
        "span_end": 20,
        "occurrences": [
            {
                "index": index,
                "answer_start": start,
                "answer_end": start + len(citation),
            }
        ],
    }


# 有精确引用账本时，只调权最终答案实际引用的 chunk。
def test_retrieval_feedback_prefers_citation_ledger_targets(tmp_path):
    store = RetrievalFeedbackStore(path=str(tmp_path / "retrieval_feedback.jsonl"))
    answer = "结论[a.pdf:P1]。"

    records = store.record_from_feedback(
        "fb-ledger",
        {
            "kb_id": "kb",
            "query": "问题",
            "feedback": "thumbs_down",
            "feedback_type": "bad_retrieval",
            "answer": answer,
            "citations": [{"chunk_id": "c1"}, {"chunk_id": "c2"}],
            "evidence": [
                {
                    "chunk_id": "c1",
                    "source_type": "document",
                    "source": "a.pdf",
                    "page": 1,
                },
                {
                    "chunk_id": "c2",
                    "source_type": "document",
                    "source": "a.pdf",
                    "page": 1,
                },
            ],
            "citation_ledger": [_ledger_entry("c2", answer)],
        },
    )

    assert [item["chunk_id"] for item in records[0]["target_chunks"]] == ["c2"]
    assert store.boosts_for_query("kb", "问题") == {"c2": -0.35}


# 混入闭集外 chunk 的 ledger 整体失败关闭，不能回退惩罚全部候选证据。
def test_retrieval_feedback_rejects_mixed_forged_ledger(tmp_path):
    store = RetrievalFeedbackStore(path=str(tmp_path / "retrieval_feedback.jsonl"))
    citation = "[a.pdf:P1]"
    answer = f"结论{citation}。"
    second_start = answer.index(citation)
    forged = _ledger_entry("forged", answer, evidence_id="E002", index=1)
    forged["occurrences"][0]["answer_start"] = second_start
    forged["occurrences"][0]["answer_end"] = second_start + len(citation)

    records = store.record_from_feedback(
        "fb-forged",
        {
            "kb_id": "kb",
            "query": "问题",
            "feedback": "thumbs_down",
            "feedback_type": "bad_retrieval",
            "answer": answer,
            "citations": [{"chunk_id": "c1"}],
            "evidence": [
                {
                    "chunk_id": "c1",
                    "source_type": "document",
                    "source": "a.pdf",
                    "page": 1,
                }
            ],
            "citation_ledger": [
                _ledger_entry("c1", answer),
                forged,
            ],
        },
    )

    assert records == []
    assert store.boosts_for_query("kb", "问题") == {}


# 验证旧版按分块展开的调权会按同一次反馈折叠。
def test_legacy_expanded_feedback_groups_by_feedback_id(tmp_path):
    path = tmp_path / "retrieval_feedback.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"retrieval_feedback_id":"rf1","feedback_id":"fb1","kb_id":"kb",'
                '"query_hash":"h","query_text":"报名要求","chunk_id":"c1",'
                '"source_type":"document","weight_delta":-0.35,"confidence":1.0,'
                '"enabled":true,"created_at":"2026-01-01T00:00:00Z"}',
                '{"retrieval_feedback_id":"rf2","feedback_id":"fb1","kb_id":"kb",'
                '"query_hash":"h","query_text":"报名要求","chunk_id":"c2",'
                '"source_type":"document","weight_delta":-0.35,"confidence":1.0,'
                '"enabled":true,"created_at":"2026-01-01T00:00:01Z"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    store = RetrievalFeedbackStore(path=str(path))

    rows = store.list(kb_id="kb")

    assert len(rows) == 1
    assert rows[0]["chunk_count"] == 2
    assert [row["chunk_id"] for row in rows[0]["target_chunks"]] == ["c1", "c2"]
    assert store.counts(kb_id="kb") == {"total": 1, "enabled": 1, "disabled": 0}

    store.set_enabled("rf1", False, actor="admin")

    assert store.counts(kb_id="kb") == {"total": 1, "enabled": 0, "disabled": 1}


# 验证缺少必填文本不会生成调权。
def test_record_from_feedback_requires_string_kb_and_query(tmp_path):
    store = RetrievalFeedbackStore(path=str(tmp_path / "retrieval_feedback.jsonl"))

    assert (
        store.record_from_feedback(
            "fb1",
            {
                "kb_id": None,
                "query": None,
                "feedback": "thumbs_down",
                "citations": [{"chunk_id": "c1"}],
            },
        )
        == []
    )
    assert not (tmp_path / "retrieval_feedback.jsonl").exists()


# 验证缓存会在追加后刷新。
def test_boost_cache_refreshes_after_append(tmp_path):
    store = RetrievalFeedbackStore(path=str(tmp_path / "retrieval_feedback.jsonl"))
    first = {
        "kb_id": "kb",
        "query": "问题",
        "feedback": "thumbs_down",
        "feedback_type": "bad_retrieval",
        "citations": [{"chunk_id": "c1"}],
    }
    second = {
        "kb_id": "kb",
        "query": "问题",
        "feedback": "thumbs_up",
        "citations": [{"chunk_id": "c2"}],
    }

    store.record_from_feedback("fb1", first)
    assert store.boosts_for_query("kb", "问题") == {"c1": -0.35}
    store.record_from_feedback("fb2", second)

    assert store.boosts_for_query("kb", "问题") == {"c1": -0.35, "c2": 0.2}


# 验证普通点踩、回答错误和纠错不会误惩罚所有引用分块。
def test_negative_feedback_requires_explicit_bad_retrieval_attribution(tmp_path):
    store = RetrievalFeedbackStore(path=str(tmp_path / "retrieval_feedback.jsonl"))
    base = {
        "kb_id": "kb",
        "query": "问题",
        "citations": [{"chunk_id": "c1"}],
    }

    cases = [
        {"feedback": "thumbs_down"},
        {"feedback": "thumbs_down", "feedback_type": "wrong_answer"},
        {"feedback": "correction", "feedback_type": "correction"},
        {"feedback": "thumbs_up", "rating": 1},
    ]

    for index, case in enumerate(cases):
        assert store.record_from_feedback(f"fb{index}", {**base, **case}) == []
    assert store.boosts_for_query("kb", "问题") == {}


# 验证正反向检索调权的边界与显式跳过开关。
def test_attributed_negative_and_positive_feedback_weights(tmp_path):
    store = RetrievalFeedbackStore(path=str(tmp_path / "retrieval_feedback.jsonl"))
    base = {
        "kb_id": "kb",
        "citations": [{"chunk_id": "c1"}],
    }

    assert (
        store.record_from_feedback(
            "negative",
            {
                **base,
                "query": "负反馈",
                "feedback": "thumbs_down",
                "feedback_type": "bad_retrieval",
            },
        )[0]["weight_delta"]
        == -0.35
    )
    assert (
        store.record_from_feedback(
            "positive",
            {**base, "query": "正反馈", "feedback": "thumbs_up"},
        )[0]["weight_delta"]
        == 0.2
    )
    assert (
        store.record_from_feedback(
            "high-rating",
            {
                **base,
                "query": "高评分",
                "feedback": "thumbs_up",
                "rating": 5,
            },
        )[0]["weight_delta"]
        == 0.24
    )
    assert (
        store.record_from_feedback(
            "skipped",
            {
                **base,
                "query": "跳过",
                "feedback": "thumbs_up",
                "skip_retrieval_feedback": True,
            },
        )
        == []
    )
