import json

from cogdoc.api.feedback_store import SqliteFeedbackStore


# 读取逐行对象文件。
def _read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# 验证数据库反馈存储保留查询统计和导出副本。
def test_sqlite_feedback_store_records_lists_counts_and_exports(tmp_path):
    store = SqliteFeedbackStore(
        db_path=str(tmp_path / "feedback.db"),
        feedback_path=str(tmp_path / "feedback.jsonl"),
        bad_cases_path=str(tmp_path / "bad_cases.jsonl"),
    )

    up = store.record(
        {
            "kb_id": "kb",
            "trace_id": "t1",
            "session_id": "s1",
            "feedback": "thumbs_up",
            "feedback_type": "other",
        }
    )
    down = store.record(
        {
            "kb_id": "kb",
            "trace_id": "t2",
            "session_id": "s1",
            "feedback": "thumbs_down",
            "feedback_type": "bad_retrieval",
            "query": "问题",
            "answer": "答案",
        }
    )

    assert up["is_bad_case"] is False
    assert down["is_bad_case"] is True
    assert [row["trace_id"] for row in store.list(kb_id="kb", session_id="s1")] == [
        "t2",
        "t1",
    ]
    assert store.list(kb_id="kb", is_bad_case=True)[0]["trace_id"] == "t2"
    assert store.counts(kb_id="kb") == {
        "total": 2,
        "bad_cases": 1,
        "by_feedback": {"thumbs_up": 1, "thumbs_down": 1},
        "by_type": {"other": 1, "bad_retrieval": 1},
    }
    assert len(_read_jsonl(tmp_path / "feedback.jsonl")) == 2
    assert _read_jsonl(tmp_path / "bad_cases.jsonl")[0]["trace_id"] == "t2"


# 验证数据库反馈存储可导入旧逐行对象文件。
def test_sqlite_feedback_store_bootstraps_from_jsonl(tmp_path):
    feedback_path = tmp_path / "feedback.jsonl"
    feedback_path.write_text(
        json.dumps(
            {
                "feedback_id": "f1",
                "created_at": "2026-01-01T00:00:00+00:00",
                "kb_id": "kb",
                "trace_id": "t1",
                "feedback": "correction",
                "feedback_type": "correction",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    store = SqliteFeedbackStore(
        db_path=str(tmp_path / "feedback.db"),
        feedback_path=str(feedback_path),
        bad_cases_path=str(tmp_path / "bad_cases.jsonl"),
    )

    assert store.list(kb_id="kb")[0]["feedback_id"] == "f1"
    assert store.counts(kb_id="kb")["bad_cases"] == 1
