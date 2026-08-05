import hashlib
import json

import pytest
from httpx import ASGITransport, AsyncClient

from cogdoc.api.app import create_app
from cogdoc.api.derived_knowledge_store import DerivedKnowledgeStore
from cogdoc.api.feedback_analysis_store import FeedbackAnalysisStore
from cogdoc.api.feedback_store import FeedbackStore
from cogdoc.api.retrieval_eval_draft_store import RetrievalEvalDraftStore
from cogdoc.api.retrieval_feedback_store import RetrievalFeedbackStore
from cogdoc.api.session_store import SessionStore
from cogdoc.state_runtime import StateRuntime
from cogdoc.tools.eval.retrieval_eval_drafts import (
    EvidenceUnitTask,
    create_pending_draft,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _stores(tmp_path):
    return {
        "feedback_store": FeedbackStore(
            str(tmp_path / "feedback.jsonl"),
            str(tmp_path / "bad_cases.jsonl"),
        ),
        "feedback_analysis_store": FeedbackAnalysisStore(
            str(tmp_path / "feedback_analysis.jsonl")
        ),
        "knowledge_store": DerivedKnowledgeStore(str(tmp_path / "knowledge.jsonl")),
        "retrieval_feedback_store": RetrievalFeedbackStore(
            str(tmp_path / "retrieval_feedback.jsonl")
        ),
        "retrieval_eval_draft_store": RetrievalEvalDraftStore(
            str(tmp_path / "retrieval_eval_drafts.jsonl")
        ),
    }


def _make_app(tmp_path, *, review_keys={"review-key"}, api_keys={"normal-key"}):
    stores = _stores(tmp_path)
    app = create_app(
        **stores,
        session_store=SessionStore(),
        api_keys=set(api_keys),
        eval_review_api_keys=set(review_keys),
    )
    return app, stores["retrieval_eval_draft_store"]


def _close_app(app):
    app.state.offload_executor.shutdown(wait=True)
    app.state.index_jobs.shutdown(wait=True)
    app.state.state_runtime.close()


def _snapshot(generation="gen-1"):
    return {
        "index_generation": generation,
        "index_build_version": "build-v1",
        "chunk_identity_version": "chunk-v1",
        "source_versions": [{"source": "a.pdf", "sha256": "sha-a"}],
    }


def _pending(
    *,
    task=EvidenceUnitTask.QA_REQUIREMENT,
    generation="gen-1",
    query="问题",
):
    unit = {
        "unit_id": "r1" if task is EvidenceUnitTask.QA_REQUIREMENT else "section-1",
        "task_kind": task,
        "label": "核心结论",
        "retrieval_query": "核心结论",
        "recovery_query": "a.pdf 核心结论",
    }
    if task is EvidenceUnitTask.SUMMARY_SECTION:
        unit.update({"source": "a.pdf", "dimension_id": "section-1"})
    return create_pending_draft(
        kb_id="kb",
        query=query,
        units=[unit],
        index_generation=generation,
        index_build_version="build-v1",
        chunk_identity_version="chunk-v1",
        source_versions=[{"source": "a.pdf", "sha256": "sha-a"}],
        origin_trace_id=f"trace-{task.value}-{generation}",
        now="2026-08-05T00:00:00+00:00",
    )


def _gold_annotations():
    return {
        "units": [
            {
                "unit_id": "r1",
                "acceptable_evidence": [
                    {
                        "chunk_id": "c1",
                        "source": "a.pdf",
                        "source_sha256": "sha-a",
                        "parent_chunk_id": "p1",
                        "start": 3,
                        "end": 12,
                    }
                ],
            }
        ]
    }


@pytest.mark.anyio
async def test_review_routes_require_independent_review_key(tmp_path, monkeypatch):
    import cogdoc.api.routes.retrieval_eval_drafts as route

    monkeypatch.setattr(route, "current_index_provenance", lambda kb_id: _snapshot())
    app, store = _make_app(tmp_path)
    store.ensure(_pending())
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            ordinary = await client.get(
                "/v1/retrieval-eval-drafts",
                headers={"Authorization": "Bearer normal-key"},
            )
            reviewer = await client.get(
                "/v1/retrieval-eval-drafts",
                headers={"Authorization": "Bearer review-key"},
            )
        assert ordinary.status_code == 403
        assert reviewer.status_code == 200
        assert len(reviewer.json()["drafts"]) == 1
    finally:
        _close_app(app)

    disabled, _ = _make_app(
        tmp_path / "disabled", review_keys=set(), api_keys={"normal-key"}
    )
    try:
        async with AsyncClient(
            transport=ASGITransport(app=disabled), base_url="http://test"
        ) as client:
            response = await client.get(
                "/v1/retrieval-eval-drafts",
                headers={"Authorization": "Bearer normal-key"},
            )
        assert response.status_code == 403
        assert "未启用" in response.json()["detail"]
    finally:
        _close_app(disabled)


@pytest.mark.anyio
async def test_review_approves_gold_with_server_derived_actor_and_revision_guard(
    tmp_path, monkeypatch
):
    import cogdoc.api.routes.retrieval_eval_drafts as route

    monkeypatch.setattr(route, "current_index_provenance", lambda kb_id: _snapshot())
    app, store = _make_app(tmp_path)
    pending = store.ensure(_pending())
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            approved = await client.post(
                f"/v1/retrieval-eval-drafts/{pending['draft_id']}/review",
                headers={"X-API-Key": "review-key"},
                json={
                    "decision": "approved",
                    "expected_revision": 1,
                    "annotations": _gold_annotations(),
                },
            )
            stale_submit = await client.post(
                f"/v1/retrieval-eval-drafts/{pending['draft_id']}/review",
                headers={"X-API-Key": "review-key"},
                json={
                    "decision": "rejected",
                    "expected_revision": 1,
                    "reason": "stale browser tab",
                },
            )
        assert approved.status_code == 200
        row = approved.json()["draft"]
        assert row["status"] == "approved"
        assert row["revision"] == 2
        expected_actor = hashlib.sha256(b"review-key").hexdigest()[:16]
        assert row["reviewed_by"] == f"eval-review:{expected_actor}"
        assert "review-key" not in json.dumps(row)
        assert stale_submit.status_code == 409
    finally:
        _close_app(app)


@pytest.mark.anyio
async def test_review_and_export_fail_closed_for_stale_or_non_qa_drafts(
    tmp_path, monkeypatch
):
    import cogdoc.api.routes.retrieval_eval_drafts as route

    monkeypatch.setattr(route, "current_index_provenance", lambda kb_id: _snapshot())
    app, store = _make_app(tmp_path)
    stale = store.ensure(_pending(generation="old-gen", query="旧索引问题"))
    qa = store.ensure(_pending())
    store.approve(qa["draft_id"], reviewer="seed", annotations=_gold_annotations())
    summary = store.ensure(_pending(task=EvidenceUnitTask.SUMMARY_SECTION))
    store.approve(
        summary["draft_id"],
        reviewer="seed",
        annotations={
            "units": [
                {
                    "unit_id": "section-1",
                    "acceptable_evidence": [
                        {
                            "chunk_id": "c-summary",
                            "source": "a.pdf",
                            "source_sha256": "sha-a",
                        }
                    ],
                }
            ]
        },
    )
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            stale_review = await client.post(
                f"/v1/retrieval-eval-drafts/{stale['draft_id']}/review",
                headers={"X-API-Key": "review-key"},
                json={
                    "decision": "approved",
                    "expected_revision": 1,
                    "annotations": _gold_annotations(),
                },
            )
            qa_export = await client.get(
                "/v1/retrieval-eval-drafts/export",
                params={"format": "retrieval_eval_v1"},
                headers={"X-API-Key": "review-key"},
            )
            generic_export = await client.get(
                "/v1/retrieval-eval-drafts/export",
                params={"format": "generic_v1"},
                headers={"X-API-Key": "review-key"},
            )
        assert stale_review.status_code == 409
        assert "index_generation_changed" in stale_review.json()["detail"]["reasons"]
        assert qa_export.status_code == 200
        assert qa_export.json()["exported_count"] == 1
        assert qa_export.json()["items"][0]["id"] == qa["draft_id"]
        assert qa_export.json()["excluded_incompatible"] == [summary["draft_id"]]
        assert generic_export.json()["exported_count"] == 2
    finally:
        _close_app(app)


@pytest.mark.anyio
async def test_missing_optional_store_is_explicitly_unavailable(tmp_path):
    stores = _stores(tmp_path)
    runtime = StateRuntime(
        feedback_store=stores["feedback_store"],
        feedback_analysis_store=stores["feedback_analysis_store"],
        knowledge_store=stores["knowledge_store"],
        retrieval_feedback_store=stores["retrieval_feedback_store"],
        derived_knowledge_index_persist_directory=str(tmp_path / "index"),
        derived_knowledge_index_state_directory=str(tmp_path / "state"),
    )
    app = create_app(
        state_runtime=runtime,
        session_store=SessionStore(),
        api_keys=set(),
        eval_review_api_keys={"review-key"},
    )
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/v1/retrieval-eval-drafts",
                headers={"X-API-Key": "review-key"},
            )
        assert response.status_code == 503
    finally:
        _close_app(app)


@pytest.mark.anyio
async def test_trusted_negative_feedback_creates_one_unlabelled_generic_draft(
    tmp_path, monkeypatch
):
    import cogdoc.api.routes.feedback as feedback_route

    app, store = _make_app(tmp_path)
    trace_file = tmp_path / "trace.json"
    trace_file.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "trace_id": "trace-qa",
                "task_type": "qa",
                "execution_status": "SUCCESS",
                "evidence_completeness": 1.0,
                "config": {"doc_id": "kb", **_snapshot()},
                "input": {"query": "真实问题", "doc_id": "kb"},
                "output": {
                    "answer": "错误回答",
                    "evidence_requirements": [
                        {
                            "requirement_id": "r1",
                            "question": "核心条件是什么",
                            "retrieval_query": "核心条件",
                            "recovery_query": "核心条件 原文",
                        }
                    ],
                    # Observed evidence is deliberately not promoted to gold.
                    "evidence": [
                        {"chunk_id": "observed", "source": "a.pdf", "page": 1}
                    ],
                    "sources": [],
                },
                "steps": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(feedback_route, "trace_path", lambda trace_id: trace_file)
    payload = {
        "trace_id": "trace-qa",
        "feedback": "thumbs_down",
        "feedback_type": "bad_retrieval",
        "kb_id": "kb",
        "query": "客户端问题",
    }
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            first = await client.post(
                "/v1/feedback",
                json=payload,
                headers={"X-API-Key": "normal-key"},
            )
            duplicate = await client.post(
                "/v1/feedback",
                json=payload,
                headers={"X-API-Key": "normal-key"},
            )
            incomplete_trace = json.loads(trace_file.read_text(encoding="utf-8"))
            incomplete_trace["trace_id"] = "trace-incomplete"
            incomplete_trace["evidence_completeness"] = 0.5
            trace_file.write_text(
                json.dumps(incomplete_trace, ensure_ascii=False), encoding="utf-8"
            )
            incomplete = await client.post(
                "/v1/feedback",
                json={**payload, "trace_id": "trace-incomplete"},
                headers={"X-API-Key": "normal-key"},
            )
        assert first.status_code == 201
        assert first.json()["retrieval_eval_draft_id"]
        assert duplicate.json()["status"] == "duplicate_ignored"
        assert (
            duplicate.json()["retrieval_eval_draft_id"]
            == first.json()["retrieval_eval_draft_id"]
        )
        assert incomplete.json()["retrieval_eval_draft_id"] is None
        rows = store.export_records()
        assert len(rows) == 1
        assert rows[0]["query"] == "真实问题"
        assert rows[0]["units"][0]["acceptable_evidence"] == []
        assert rows[0]["hard_negative_chunks"] == []
        assert rows[0]["origin_feedback_id"] == first.json()["feedback_id"]
    finally:
        _close_app(app)
