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
            "trace_id": "t1",
            "citations": [{"chunk_id": "c1", "source_type": "document"}],
            "evidence": [{"chunk_id": "c1"}, {"chunk_id": "c2"}],
        },
    )

    assert [row["chunk_id"] for row in records] == ["c1", "c2"]
    boosts = store.boosts_for_query("kb", "报名要求")
    assert boosts == {"c1": -0.35, "c2": -0.35}

    store.set_enabled(records[0]["retrieval_feedback_id"], False, actor="admin")

    assert store.boosts_for_query("kb", "报名要求") == {"c2": -0.35}


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
