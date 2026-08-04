import pytest

from cogdoc.api.derived_knowledge_store import (
    AUTO_REBIND_REVIEW_NOTE,
    DerivedKnowledgeStore,
    SqliteDerivedKnowledgeStore,
)


@pytest.fixture(params=["jsonl", "sqlite"])
def knowledge_store(request, tmp_path):
    if request.param == "jsonl":
        return DerivedKnowledgeStore(path=str(tmp_path / "knowledge.jsonl"))
    return SqliteDerivedKnowledgeStore(db_path=str(tmp_path / "state.db"))


# 两种持久化实现复用同一行为场景，覆盖冲突、筛选、修订、stale 和审核事件。
def test_derived_knowledge_store_persistence_parity(knowledge_store):
    seed, duplicate = knowledge_store.create(
        {
            "kb_id": "kb",
            "text": "差旅报销必须在七天内提交。",
            "status": "approved",
            "origin": "feedback_correction",
            "created_by": "alice",
            "related_document_id": "doc-1",
            "related_source": "policy.pdf",
            "related_source_sha256": "old-sha",
            "related_chunk_ids": ["chunk-1", "chunk-2"],
            "related_page_start": 2,
            "related_page_end": 3,
        }
    )
    same, is_duplicate = knowledge_store.create(
        {"kb_id": "kb", "text": "  差旅报销必须在七天内提交。  "}
    )

    assert duplicate is False
    assert is_duplicate is True
    assert same["knowledge_id"] == seed["knowledge_id"]
    assert knowledge_store.get(seed["knowledge_id"])["related_chunk_ids"] == [
        "chunk-1",
        "chunk-2",
    ]

    similar, _ = knowledge_store.create(
        {"kb_id": "kb", "text": "差旅报销必须在七天内完成提交。"}
    )
    seed = knowledge_store.get(seed["knowledge_id"])
    assert seed["conflict_group_id"]
    assert similar["conflict_group_id"] == seed["conflict_group_id"]
    assert similar["status"] == "pending"
    assert [row["knowledge_id"] for row in knowledge_store.conflicts_for(seed)] == [
        similar["knowledge_id"]
    ]

    created_date = seed["created_at"][:10]
    filtered = knowledge_store.list(
        kb_id="kb",
        document_id="doc-1",
        origin="feedback_correction",
        created_by="alice",
        has_conflict=True,
        created_after=created_date,
        created_before=created_date,
    )
    assert [row["knowledge_id"] for row in filtered] == [seed["knowledge_id"]]
    assert knowledge_store.conflict_counts(kb_id="kb")["groups"] == 1

    knowledge_store.set_status(similar["knowledge_id"], "rejected", actor="reviewer")
    revised = knowledge_store.revise(
        seed["knowledge_id"],
        {
            "text": "发票必须在五个工作日内提交财务。",
            "status": "approved",
            "created_by": "reviewer",
        },
    )
    assert revised["version"] == seed["version"] + 1
    assert revised["previous_version_id"] == seed["knowledge_id"]
    assert knowledge_store.get(seed["knowledge_id"])["status"] == "archived"

    bound, _ = knowledge_store.create(
        {
            "kb_id": "kb",
            "text": "合同模板由法务部门维护。",
            "status": "approved",
            "related_source": "contract.pdf",
            "related_source_sha256": "sha-v1",
        }
    )
    stale = knowledge_store.mark_stale_for_source("kb", "contract.pdf", "sha-v1")
    assert [row["knowledge_id"] for row in stale] == [bound["knowledge_id"]]
    assert knowledge_store.mark_stale_for_source("kb", "contract.pdf", "sha-v1") == []
    assert knowledge_store.stale_review_counts(kb_id="kb") == {
        "total": 1,
        "reviewed": 0,
    }
    rebound = knowledge_store.set_status(
        bound["knowledge_id"],
        "approved",
        actor="system",
        note=AUTO_REBIND_REVIEW_NOTE,
        binding_updates={"related_source_sha256": "sha-v2"},
    )
    assert rebound["related_source_sha256"] == "sha-v2"
    assert knowledge_store.auto_review_counts(kb_id="kb") == {"auto_rebound": 1}
    assert knowledge_store.auto_review_events(kb_id="kb")[0][
        "knowledge_id"
    ] == bound["knowledge_id"]
    assert knowledge_store.stale_review_counts(kb_id="kb") == {
        "total": 1,
        "reviewed": 1,
    }

    first, _ = knowledge_store.create({"kb_id": "kb", "text": "批量知识甲。"})
    second, _ = knowledge_store.create({"kb_id": "kb", "text": "批量知识乙。"})
    updated, missing = knowledge_store.batch_set_status(
        [first["knowledge_id"], "missing", second["knowledge_id"]],
        "approved",
        actor="admin",
    )
    assert [row["knowledge_id"] for row in updated] == [
        first["knowledge_id"],
        second["knowledge_id"],
    ]
    assert missing == ["missing"]
    assert knowledge_store.counts(kb_id="kb")["by_status"]["approved"] == 4


# JSONL 历史可幂等迁移到 state.db，未知嵌套元数据也按原 JSON 保真。
def test_sqlite_store_persists_and_imports_records_idempotently(tmp_path):
    source = DerivedKnowledgeStore(path=str(tmp_path / "knowledge.jsonl"))
    created, _ = source.create(
        {
            "kb_id": "kb",
            "text": "迁移记录。",
            "status": "approved",
            "related_chunk_ids": ["c1"],
        }
    )
    records = source.export_records()
    records[0]["custom_metadata"] = {"labels": ["甲", "乙"], "score": 0.75}

    db_path = tmp_path / "state.db"
    store = SqliteDerivedKnowledgeStore(db_path=str(db_path))
    before = store.revision_token()
    first = store.import_records(records)
    after = store.revision_token()
    second = store.import_records(records)

    assert db_path.name == "state.db"
    assert first == {"imported": 1, "skipped": 0}
    assert second == {"imported": 0, "skipped": 1}
    assert before != after
    assert store.revision_token() == after
    assert store.get(created["knowledge_id"])["custom_metadata"] == {
        "labels": ["甲", "乙"],
        "score": 0.75,
    }
    conflicting = [{**records[0], "text": "同一事件键的冲突载荷。"}]
    with pytest.raises(ValueError, match="import event key conflict"):
        store.import_records(conflicting)
    assert store.export_records() == records
    store.close()

    reopened = SqliteDerivedKnowledgeStore(db_path=str(db_path))
    assert reopened.export_records() == records
    reopened.clear_kb("kb")
    assert reopened.list(kb_id="kb") == []
    reopened.close()


# 无显式路径时与其他 SQLite store 共用配置的 state.db。
def test_sqlite_store_defaults_to_configured_state_db(tmp_path, monkeypatch):
    state_db_path = str(tmp_path / "configured-state.db")

    class StubSettings:
        pass

    settings = StubSettings()
    settings.state_db_path = state_db_path
    monkeypatch.setattr(
        "cogdoc.api.derived_knowledge_store.get_settings", lambda: settings
    )

    store = SqliteDerivedKnowledgeStore()
    assert store._path == state_db_path
    store.close()
