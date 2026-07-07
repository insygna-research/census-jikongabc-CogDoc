from cogdoc.api.feedback_analysis_store import FeedbackAnalysisStore


# 验证裸文件名路径可正常写入。
def test_feedback_analysis_store_accepts_bare_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = FeedbackAnalysisStore(path="feedback_analysis.jsonl")

    row = store.record(
        "fb1",
        {"kb_id": "kb", "trace_id": "t1", "query": "问题"},
        {
            "feedback_type": "other",
            "sentiment": "neutral",
            "target": {"chunk_ids": [], "sources": [], "source_type": "none"},
            "extracted_claim": "",
            "recommended_action": "record_only",
            "weight_delta": 0.0,
            "confidence": 0.55,
            "needs_review": True,
        },
    )

    assert row["feedback_analysis_id"]
    assert store.list(kb_id="kb")[0]["feedback_id"] == "fb1"


# 验证追加后缓存会刷新。
def test_feedback_analysis_store_refreshes_cache_after_append(tmp_path):
    store = FeedbackAnalysisStore(path=str(tmp_path / "feedback_analysis.jsonl"))
    first_analysis = {
        "feedback_type": "other",
        "sentiment": "neutral",
        "target": {"chunk_ids": [], "sources": [], "source_type": "none"},
        "extracted_claim": "",
        "recommended_action": "record_only",
        "weight_delta": 0.0,
        "confidence": 0.55,
        "needs_review": True,
    }
    second_analysis = {**first_analysis, "recommended_action": "adjust_retrieval"}

    store.record("fb1", {"kb_id": "kb", "trace_id": "t1"}, first_analysis)
    assert len(store.list(kb_id="kb")) == 1
    store.record("fb2", {"kb_id": "kb", "trace_id": "t2"}, second_analysis)

    rows = store.list(kb_id="kb")
    assert [row["feedback_id"] for row in rows] == ["fb2", "fb1"]
