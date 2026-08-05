import pytest
from pydantic import ValidationError

from cogdoc.api.retrieval_eval_draft_store import (
    DraftRevisionConflictError,
    RetrievalEvalDraftStore,
    SqliteRetrievalEvalDraftStore,
)
from cogdoc.tools.eval.retrieval_eval_drafts import (
    DatasetPartition,
    EvidenceUnitDraft,
    EvidenceUnitTask,
    RetrievalEvalDraft,
    create_pending_draft,
    draft_dedupe_key,
    draft_id_from_key,
)


def _store(tmp_path, backend):
    path = tmp_path / (
        "retrieval-eval-drafts.jsonl" if backend == "jsonl" else "state.db"
    )
    if backend == "jsonl":
        return RetrievalEvalDraftStore(str(path)), path
    return SqliteRetrievalEvalDraftStore(str(path)), path


def _draft(
    *,
    partition=DatasetPartition.TRAINING,
    feedback_id="feedback-1",
    generation="generation-7",
    source_sha="sha-a",
):
    return create_pending_draft(
        kb_id="kb-1",
        query="报名条件是什么？",
        units=[
            EvidenceUnitDraft(
                unit_id="r1",
                task_kind=EvidenceUnitTask.QA_REQUIREMENT,
                label="年龄条件是什么？",
                retrieval_query="报名 年龄 条件",
                recovery_query="参赛者 年龄限制",
            )
        ],
        dataset_partition=partition,
        index_generation=generation,
        index_build_version="hybrid-v2",
        chunk_identity_version="chunk-v5",
        source_versions=[{"source": "a.pdf", "sha256": source_sha}],
        origin_feedback_id=feedback_id,
        now="2026-01-01T00:00:00+00:00",
    )


def _legacy_draft(draft):
    payload = draft.model_dump(mode="json")
    legacy_key = draft_dedupe_key(
        kb_id=draft.kb_id,
        query=draft.query,
        dataset_partition=draft.dataset_partition,
        units=draft.units,
    )
    payload.pop("identity_snapshot")
    payload["dedupe_key"] = legacy_key
    payload["draft_id"] = draft_id_from_key(legacy_key)
    return RetrievalEvalDraft.model_validate(payload)


ANNOTATIONS = {
    "units": [
        {
            "unit_id": "r1",
            "acceptable_evidence": [
                {
                    "chunk_id": "chunk-1",
                    "source": "a.pdf",
                    "source_sha256": "sha-a",
                    "start": 4,
                    "end": 18,
                }
            ],
        }
    ],
    "hard_negative_chunks": [
        {
            "chunk_id": "chunk-bad",
            "source": "a.pdf",
            "source_sha256": "sha-a",
        }
    ],
}


@pytest.mark.parametrize("backend", ["jsonl", "sqlite"])
def test_store_ensure_review_list_and_partitioned_export(tmp_path, backend):
    store, _ = _store(tmp_path, backend)
    training = _draft()
    duplicate = _draft(feedback_id="feedback-2")
    release = _draft(partition=DatasetPartition.RELEASE_GATE)

    assert store.ensure(training)["origin_feedback_id"] == "feedback-1"
    # Stable dedupe never overwrites an existing pending or reviewed record.
    assert store.ensure(duplicate)["origin_feedback_id"] == "feedback-1"
    store.ensure(release)
    approved_training = store.review(
        training.draft_id,
        decision="approved",
        reviewer="reviewer",
        annotations=ANNOTATIONS,
        now="2026-01-02T00:00:00+00:00",
    )
    store.approve(
        release.draft_id,
        reviewer="gate-reviewer",
        annotations=ANNOTATIONS,
        now="2026-01-03T00:00:00+00:00",
    )

    assert approved_training["status"] == "approved"
    assert store.get(training.draft_id) == approved_training
    assert len(store.list(kb_id="kb-1", status="approved")) == 2
    training_cases = store.export_eval_cases(
        dataset_partition=DatasetPartition.TRAINING
    )
    gate_cases = store.export_eval_cases(
        dataset_partition=DatasetPartition.RELEASE_GATE
    )
    assert [row["id"] for row in training_cases] == [training.draft_id]
    assert [row["id"] for row in gate_cases] == [release.draft_id]
    assert training_cases[0]["gold_requirements"][0]["acceptable_chunk_ids"] == [
        "chunk-1"
    ]
    store.close()


@pytest.mark.parametrize("backend", ["jsonl", "sqlite"])
def test_store_reject_requires_reason_and_cannot_export_pending_or_rejected(
    tmp_path, backend
):
    store, _ = _store(tmp_path, backend)
    draft = _draft()
    store.ensure(draft)

    with pytest.raises(ValueError, match="reviewer and rejection reason"):
        store.reject(draft.draft_id, reviewer="reviewer", reason="")
    rejected = store.review(
        draft.draft_id,
        decision="rejected",
        reviewer="reviewer",
        reason="反馈归因错误",
    )

    assert rejected["status"] == "rejected"
    assert store.export_eval_cases(dataset_partition="training") == []
    with pytest.raises(ValueError, match="only pending"):
        store.approve(
            draft.draft_id,
            reviewer="reviewer",
            annotations=ANNOTATIONS,
        )
    store.close()


@pytest.mark.parametrize("backend", ["jsonl", "sqlite"])
def test_store_import_is_atomic_idempotent_and_persistent(tmp_path, backend):
    source, _ = _store(tmp_path / "source", backend)
    source.ensure(_draft())
    source.ensure(_draft(partition=DatasetPartition.RELEASE_GATE))
    records = source.export_records()
    source.close()

    target, path = _store(tmp_path / "target", backend)
    invalid = {**records[1], "draft_id": "forged"}
    with pytest.raises((ValueError, ValidationError)):
        target.import_records([records[0], invalid])
    assert target.export_records() == []
    assert target.import_records(records) == {"imported": 2, "skipped": 0}
    assert target.import_records(records) == {"imported": 0, "skipped": 2}
    target.close()
    target.close()

    reopened = (
        RetrievalEvalDraftStore(str(path))
        if backend == "jsonl"
        else SqliteRetrievalEvalDraftStore(str(path))
    )
    assert len(reopened.list(kb_id="kb-1")) == 2
    reopened.clear_kb("kb-1")
    assert reopened.list(kb_id="kb-1") == []
    reopened.close()


@pytest.mark.parametrize("backend", ["jsonl", "sqlite"])
def test_store_close_is_idempotent_and_blocks_further_use(tmp_path, backend):
    store, _ = _store(tmp_path, backend)
    store.close()
    store.close()

    with pytest.raises(RuntimeError, match="store is closed"):
        store.list()


@pytest.mark.parametrize("backend", ["jsonl", "sqlite"])
def test_store_review_rejects_stale_revision_atomically(tmp_path, backend):
    store, _ = _store(tmp_path, backend)
    draft = _draft()
    store.ensure(draft)

    with pytest.raises(DraftRevisionConflictError, match="expected 2, found 1"):
        store.review(
            draft.draft_id,
            decision="approved",
            reviewer="stale-reviewer",
            annotations=ANNOTATIONS,
            expected_revision=2,
        )
    unchanged = store.get(draft.draft_id)
    assert unchanged is not None
    assert unchanged["status"] == "pending"
    assert unchanged["revision"] == 1

    approved = store.approve(
        draft.draft_id,
        reviewer="reviewer",
        annotations=ANNOTATIONS,
        expected_revision=1,
    )
    assert approved["revision"] == 2
    with pytest.raises(DraftRevisionConflictError, match="expected 1, found 2"):
        store.reject(
            draft.draft_id,
            reviewer="stale-reviewer",
            reason="陈旧页面",
            expected_revision=1,
        )
    store.close()


@pytest.mark.parametrize("backend", ["jsonl", "sqlite"])
def test_store_ensure_is_legacy_compatible_but_snapshot_scoped(tmp_path, backend):
    store, _ = _store(tmp_path, backend)
    current = _draft()
    legacy = _legacy_draft(current)
    rebuilt = _draft(generation="generation-8")
    rehashed = _draft(source_sha="sha-a-new")

    store.import_records([legacy.model_dump(mode="json")])
    assert store.ensure(current)["draft_id"] == legacy.draft_id
    assert store.ensure(rebuilt)["draft_id"] == rebuilt.draft_id
    assert store.ensure(rehashed)["draft_id"] == rehashed.draft_id
    assert {row["draft_id"] for row in store.list(kb_id="kb-1")} == {
        legacy.draft_id,
        rebuilt.draft_id,
        rehashed.draft_id,
    }
    store.close()
