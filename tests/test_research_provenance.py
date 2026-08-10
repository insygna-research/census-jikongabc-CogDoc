import copy
import hashlib
from types import SimpleNamespace

import pytest

from cogdoc.service import research_provenance
from cogdoc.service.research_provenance import (
    RESEARCH_ARTIFACT_VERSION,
    RESEARCH_CONTRACT_VERSION,
    RESEARCH_PROVENANCE_VERSION,
    build_research_verification_snapshot,
    capture_research_provenance,
    freeze_research_execution_nodes,
    is_trackable_research_provenance,
    research_artifact_integrity_status,
    research_artifact_sha256,
    research_provenance_stale_reasons,
    research_provenance_status,
)


def _snapshot(
    *,
    generation: str = "generation-1",
    source_sha256: str = "source-sha-1",
    derived_revision: str = "derived-1",
    tuning_revision: str = "tuning-1",
    contract_revision: str = "contract-1",
):
    return {
        "schema_version": RESEARCH_PROVENANCE_VERSION,
        "kb_id": "kb",
        "index_generation": generation,
        "index_build_version": "index-build-v1",
        "chunk_identity_version": "chunk-identity-v1",
        "source_versions": [{"source": "rules.pdf", "sha256": source_sha256}],
        "derived_knowledge_revision": derived_revision,
        "retrieval_tuning_revision": tuning_revision,
        "research_contract_version": RESEARCH_CONTRACT_VERSION,
        "research_contract_revision": contract_revision,
        "captured_at": "2026-08-09T00:00:00+00:00",
    }


def _verified_artifact():
    verification_metrics = {
        "supported_count": 1,
        "claim_audit": {"passed_section_count": 1},
    }
    verification = build_research_verification_snapshot(
        job={
            "job_id": "rj_verified",
            "kb_id": "kb",
            "execution_id": "evidence-execution-1",
            "report_execution_id": "report-execution-1",
            "title": "Verified research report",
            "objective": "Verify every planned obligation",
            "is_local": True,
            "report_execution_nodes": freeze_research_execution_nodes(is_local=True),
        },
        verification_metrics=verification_metrics,
        sections=[
            {
                "section_id": "s1",
                "position": 1,
                "title": "Eligibility",
                "research_question": "What are the eligibility rules?",
                "success_criteria": "Every eligibility rule is supported.",
                "revision_instruction": "Include the applicable date.",
                "evidence_requirements": [
                    {
                        "requirement_id": "s1:r1",
                        "question": "Who is eligible?",
                        "retrieval_query": "eligibility primary",
                        "recovery_query": "eligible participants fallback",
                    }
                ],
                "evidence_requirement_ids": ["s1:r1"],
                "status": "generated",
                "verification_status": "supported",
                "verification_reason_code": "supported",
                "evidence_requirement_results": [
                    {
                        "requirement_id": "s1:r1",
                        "status": "supported",
                        "reason_code": "direct_support",
                        "evidence_count": 1,
                    }
                ],
                "claim_audit": {
                    "status": "passed",
                    "counts": {"claim_count": 1, "supported": 1},
                    "claims": [{"text": "must never enter the artifact"}],
                },
                "coverage_audit": {
                    "status": "passed",
                    "requirement_count": 1,
                    "covered_count": 1,
                    "assessments": [{"model_reason": "must never be persisted"}],
                },
                "evidence": [
                    {
                        "chunk_id": "chunk-1",
                        "source": "rules.pdf",
                        "source_sha256": "source-sha-1",
                        "text_hash": "text-sha-1",
                        "page": 3,
                        "text_preview": "must never enter verification.json",
                    }
                ],
            }
        ],
    )
    report = {
        "artifact_schema_version": RESEARCH_ARTIFACT_VERSION,
        "format": "markdown",
        "content": "# Verified research report\n",
        "citation_ledger": [],
        "verification_metrics": verification_metrics,
        "verification": verification,
        "provenance": _snapshot(),
        "version": 1,
        "generated_at": "2026-08-10T00:00:00+00:00",
    }
    report["sha256"] = research_artifact_sha256(
        content=report["content"],
        citation_ledger=report["citation_ledger"],
        provenance=report["provenance"],
        verification=report["verification"],
        metadata={
            "version": report["version"],
            "generated_at": report["generated_at"],
        },
    )
    return report


class _ListStore:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def list(self, **kwargs):
        self.calls.append(kwargs)
        return list(self.rows)


class _ExportStore(_ListStore):
    def __init__(self, rows):
        super().__init__(rows)
        self.export_calls = 0

    def export_records(self):
        self.export_calls += 1
        return list(self.rows)


def _patch_index_provenance(monkeypatch):
    monkeypatch.setattr(
        research_provenance,
        "current_index_provenance",
        lambda kb_id: {
            "index_generation": f"{kb_id}-generation",
            "index_build_version": "index-build-v1",
            "chunk_identity_version": "chunk-identity-v1",
            "source_versions": [{"source": "rules.pdf", "sha256": "source-sha-1"}],
        },
    )


def test_capture_research_provenance_freezes_all_research_inputs(monkeypatch):
    _patch_index_provenance(monkeypatch)
    knowledge_store = _ListStore(
        [
            {
                "knowledge_id": "dk-2",
                "kb_id": "kb",
                "text": "第二条知识",
                "version": 2,
                "normalized_hash": "normalized-2",
                "status": "approved",
                "related_source": "rules.pdf",
                "related_source_sha256": "source-sha-1",
            },
            {
                "knowledge_id": "dk-1",
                "kb_id": "kb",
                "text": "第一条知识",
                "version": 1,
                "normalized_hash": "normalized-1",
                "status": "approved",
                "related_source": "rules.pdf",
                "related_source_sha256": "source-sha-1",
            },
        ]
    )
    feedback_store = _ExportStore(
        [
            {
                "retrieval_feedback_id": "rf-1",
                "kb_id": "kb",
                "enabled": True,
                "query_hash": "query-1",
                "weight_delta": 0.25,
                "confidence": 0.8,
                "target_chunks": [
                    {"chunk_id": "chunk-2", "source_type": "document"},
                    {"chunk_id": "chunk-1", "source_type": "document"},
                ],
            }
        ]
    )
    runtime = SimpleNamespace(
        knowledge_store=knowledge_store,
        retrieval_feedback_store=feedback_store,
    )

    captured = capture_research_provenance("kb", state_runtime=runtime)
    knowledge_store.rows.reverse()
    feedback_store.rows[0]["target_chunks"].reverse()
    reordered = capture_research_provenance("kb", state_runtime=runtime)
    feedback_store.rows[0]["target_chunks"][0]["chunk_id"] = "chunk-3"
    retuned = capture_research_provenance("kb", state_runtime=runtime)

    assert captured["schema_version"] == RESEARCH_PROVENANCE_VERSION
    assert captured["kb_id"] == "kb"
    assert captured["index_generation"] == "kb-generation"
    assert captured["source_versions"] == [
        {"source": "rules.pdf", "sha256": "source-sha-1"}
    ]
    assert len(captured["derived_knowledge_revision"]) == 64
    assert len(captured["retrieval_tuning_revision"]) == 64
    assert captured["research_contract_version"] == RESEARCH_CONTRACT_VERSION
    assert len(captured["research_contract_revision"]) == 64
    assert is_trackable_research_provenance(captured)
    assert (
        reordered["derived_knowledge_revision"]
        == captured["derived_knowledge_revision"]
    )
    assert (
        reordered["retrieval_tuning_revision"] == captured["retrieval_tuning_revision"]
    )
    assert retuned["retrieval_tuning_revision"] != captured["retrieval_tuning_revision"]
    assert knowledge_store.calls == [
        {"kb_id": "kb", "status": "approved"},
        {"kb_id": "kb", "status": "approved"},
        {"kb_id": "kb", "status": "approved"},
    ]
    assert feedback_store.export_calls == 3
    assert feedback_store.calls == []


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("text", "更新后的知识正文"),
        ("normalized_hash", "normalized-2"),
        ("version", 2),
        ("origin", "answer_feedback"),
        ("status", "stale"),
        ("certainty", "low"),
        ("source_note", "更新后的来源说明"),
        ("related_document_id", "doc-2"),
        ("related_source", "new-rules.pdf"),
        ("related_source_sha256", "source-sha-2"),
        ("related_chunk_ids", ["chunk-2"]),
        ("related_page_start", 3),
        ("related_page_end", 4),
        ("related_chunk_text_hash", "chunk-sha-2"),
        ("related_anchor_text", "更新后的锚点"),
    ],
)
def test_derived_knowledge_retrieval_fields_change_revision(
    monkeypatch, field, changed_value
):
    _patch_index_provenance(monkeypatch)
    row = {
        "knowledge_id": "dk-1",
        "kb_id": "kb",
        "text": "知识正文",
        "normalized_hash": "normalized-1",
        "version": 1,
        "origin": "manual_entry",
        "status": "approved",
        "certainty": "high",
        "source_note": "来源说明",
        "related_document_id": "doc-1",
        "related_source": "rules.pdf",
        "related_source_sha256": "source-sha-1",
        "related_chunk_ids": ["chunk-1"],
        "related_page_start": 1,
        "related_page_end": 2,
        "related_chunk_text_hash": "chunk-sha-1",
        "related_anchor_text": "锚点",
    }
    knowledge_store = _ListStore([row])
    runtime = SimpleNamespace(
        knowledge_store=knowledge_store,
        retrieval_feedback_store=_ExportStore([]),
    )
    captured = capture_research_provenance("kb", state_runtime=runtime)
    row[field] = changed_value
    current = capture_research_provenance("kb", state_runtime=runtime)

    assert (
        current["derived_knowledge_revision"] != captured["derived_knowledge_revision"]
    )
    assert research_provenance_stale_reasons(captured, current) == (
        "derived_knowledge_revision_changed",
    )


def test_derived_binding_set_order_does_not_change_revision(monkeypatch):
    _patch_index_provenance(monkeypatch)
    row = {
        "knowledge_id": "dk-1",
        "kb_id": "kb",
        "text": "知识正文",
        "normalized_hash": "normalized-1",
        "version": 1,
        "status": "approved",
        "related_chunk_ids": ["chunk-2", "chunk-1"],
    }
    runtime = SimpleNamespace(
        knowledge_store=_ListStore([row]),
        retrieval_feedback_store=_ExportStore([]),
    )
    captured = capture_research_provenance("kb", state_runtime=runtime)
    row["related_chunk_ids"].reverse()
    reordered = capture_research_provenance("kb", state_runtime=runtime)

    assert (
        reordered["derived_knowledge_revision"]
        == captured["derived_knowledge_revision"]
    )


def test_retrieval_tuning_uses_raw_records_and_ignores_unconsumed_fields(
    monkeypatch,
):
    _patch_index_provenance(monkeypatch)
    rows = [
        {
            "retrieval_feedback_id": "rf-2",
            "kb_id": "kb",
            "enabled": True,
            "query_hash": "query-1",
            "query_text": "原始查询不会被 boost 消费",
            "weight_delta": 0.25,
            "confidence": 0.8,
            "target_chunks": [
                {"chunk_id": "chunk-2", "source_type": "document"},
                {"chunk_id": "chunk-1", "source_type": "document"},
            ],
            "created_at": "2026-08-09T00:00:00Z",
        },
        {
            "retrieval_feedback_id": "rf-disabled",
            "kb_id": "kb",
            "enabled": False,
            "query_hash": "query-disabled",
            "weight_delta": 9,
            "target_chunks": [{"chunk_id": "disabled"}],
        },
        {
            "retrieval_feedback_id": "rf-other-kb",
            "kb_id": "other",
            "enabled": True,
            "query_hash": "query-other",
            "weight_delta": 9,
            "target_chunks": [{"chunk_id": "other"}],
        },
    ]
    feedback_store = _ExportStore(rows)
    runtime = SimpleNamespace(
        knowledge_store=_ListStore([]),
        retrieval_feedback_store=feedback_store,
    )
    captured = capture_research_provenance("kb", state_runtime=runtime)
    rows.reverse()
    rows[-1]["target_chunks"].reverse()
    rows[-1]["query_text"] = "改变未消费的显示文本"
    rows[-1]["created_at"] = "2026-08-10T00:00:00Z"
    reordered = capture_research_provenance("kb", state_runtime=runtime)

    assert (
        reordered["retrieval_tuning_revision"] == captured["retrieval_tuning_revision"]
    )
    assert feedback_store.calls == []


def test_retrieval_tuning_falls_back_to_aggregated_list(monkeypatch):
    _patch_index_provenance(monkeypatch)
    feedback_store = _ListStore(
        [
            {
                "retrieval_feedback_id": "rf-1",
                "query_hash": "query-1",
                "weight_delta": 0.25,
                "confidence": 0.8,
                "target_chunks": [{"chunk_id": "chunk-1"}],
            }
        ]
    )
    runtime = SimpleNamespace(
        knowledge_store=_ListStore([]),
        retrieval_feedback_store=feedback_store,
    )

    captured = capture_research_provenance("kb", state_runtime=runtime)

    assert len(captured["retrieval_tuning_revision"]) == 64
    assert feedback_store.calls == [
        {"kb_id": "kb", "enabled": True, "limit": 2**31 - 1}
    ]


def test_research_contract_setting_change_is_stale(monkeypatch):
    _patch_index_provenance(monkeypatch)
    settings = SimpleNamespace(cogdoc_research_retrieval_top_k=8)
    monkeypatch.setattr(research_provenance, "get_settings", lambda: settings)
    runtime = SimpleNamespace(
        knowledge_store=_ListStore([]),
        retrieval_feedback_store=_ExportStore([]),
    )
    captured = capture_research_provenance("kb", state_runtime=runtime)
    settings.cogdoc_research_retrieval_top_k = 12
    current = capture_research_provenance("kb", state_runtime=runtime)

    assert captured["research_contract_version"] == RESEARCH_CONTRACT_VERSION
    assert (
        current["research_contract_revision"] != captured["research_contract_revision"]
    )
    assert research_provenance_stale_reasons(captured, current) == (
        "research_contract_revision_changed",
    )


def test_research_contract_handles_missing_reranker_metadata(monkeypatch):
    from cogdoc.tools.reranker import BGEReranker

    _patch_index_provenance(monkeypatch)
    monkeypatch.delattr(BGEReranker, "MODEL_NAME")
    monkeypatch.delattr(BGEReranker, "MAX_LENGTH")
    runtime = SimpleNamespace(
        knowledge_store=_ListStore([]),
        retrieval_feedback_store=_ExportStore([]),
    )

    captured = capture_research_provenance("kb", state_runtime=runtime)

    assert len(captured["research_contract_revision"]) == 64


def test_research_provenance_status_is_current_for_the_same_snapshot():
    captured = _snapshot()
    current = {**captured, "captured_at": "2026-08-10T00:00:00+00:00"}

    assert research_provenance_stale_reasons(captured, current) == ()
    assert research_provenance_status(captured, current)["status"] == "current"


def test_research_provenance_without_contract_identity_is_untracked():
    captured = _snapshot()
    captured.pop("research_contract_revision")

    assert not is_trackable_research_provenance(captured)
    assert research_provenance_stale_reasons(captured, _snapshot()) == (
        "evidence_provenance_untracked",
    )


@pytest.mark.parametrize(
    ("changes", "expected_reason"),
    [
        ({"index_generation": "generation-2"}, "index_generation_changed"),
        (
            {"source_versions": [{"source": "rules.pdf", "sha256": "source-sha-2"}]},
            "source_sha256_changed:rules.pdf",
        ),
        (
            {"derived_knowledge_revision": "derived-2"},
            "derived_knowledge_revision_changed",
        ),
        (
            {"retrieval_tuning_revision": "tuning-2"},
            "retrieval_tuning_revision_changed",
        ),
        (
            {"research_contract_version": "research-contract-v2"},
            "research_contract_version_changed",
        ),
        (
            {"research_contract_revision": "contract-2"},
            "research_contract_revision_changed",
        ),
    ],
)
def test_research_provenance_reports_each_stale_input(changes, expected_reason):
    captured = _snapshot()
    current = {**captured, **changes}

    assert research_provenance_stale_reasons(captured, current) == (expected_reason,)
    status = research_provenance_status(captured, current)
    assert status["status"] == "stale"
    assert status["stale_reasons"] == [expected_reason]


def test_research_verification_snapshot_is_bounded_and_body_free():
    report = _verified_artifact()
    serialized = research_provenance._canonical_json(report["verification"])
    commitment = report["verification"]["sections"][0]["evidence_commitments"][0]

    assert research_artifact_integrity_status(report) == "verified"
    assert report["verification"]["execution"]["is_local"] is True
    assert (
        report["verification"]["sections"][0]["requirements"][0]["retrieval_query"]
        == "eligibility primary"
    )
    assert commitment["source_sha256"] == "source-sha-1"
    assert commitment["text_hash"] == "text-sha-1"
    assert "text_preview" not in commitment
    assert "must never" not in serialized
    assert "claims" not in serialized
    assert "assessments" not in serialized


@pytest.mark.parametrize(
    "tamper",
    [
        "claim_audit",
        "coverage_audit",
        "aggregate_metrics",
        "public_metrics",
        "evidence_commitment",
        "incomplete_evidence_span",
        "execution_mode",
        "requirement_plan",
        "version",
        "generated_at",
        "malformed_ledger",
    ],
)
def test_research_artifact_rejects_every_committed_field_tamper(tamper):
    report = copy.deepcopy(_verified_artifact())
    section = report["verification"]["sections"][0]
    if tamper == "claim_audit":
        section["claim_audit"]["status"] = "failed"
    elif tamper == "coverage_audit":
        section["coverage_audit"]["covered_count"] = 0
    elif tamper == "aggregate_metrics":
        report["verification"]["aggregate"]["supported_count"] = 0
    elif tamper == "public_metrics":
        report["verification_metrics"]["supported_count"] = 0
    elif tamper == "evidence_commitment":
        section["evidence_commitments"][0]["source_sha256"] = "tampered"
    elif tamper == "incomplete_evidence_span":
        section["evidence_commitments"][0]["span_start"] = 999
    elif tamper == "execution_mode":
        report["verification"]["execution"]["is_local"] = False
    elif tamper == "requirement_plan":
        section["requirements"][0]["question"] = "tampered question"
    elif tamper == "version":
        report["version"] = 2
    elif tamper == "generated_at":
        report["generated_at"] = "2026-08-11T00:00:00+00:00"
    elif tamper == "malformed_ledger":
        report["citation_ledger"].append("not-an-object")

    assert research_artifact_integrity_status(report) == "invalid"


def test_research_artifact_hash_rejects_coercible_input_types():
    report = _verified_artifact()

    with pytest.raises(TypeError, match="citation_ledger"):
        research_artifact_sha256(
            content=report["content"],
            citation_ledger=({},),
            provenance=report["provenance"],
            verification=report["verification"],
            metadata={
                "version": report["version"],
                "generated_at": report["generated_at"],
            },
        )


def test_research_artifact_requires_trackable_provenance_and_marks_legacy():
    report = _verified_artifact()
    report["provenance"] = {}
    assert research_artifact_integrity_status(report) == "invalid"

    legacy = {"format": "markdown", "content": "# Legacy\n"}
    assert research_artifact_integrity_status(legacy) == "legacy-unverified"


def test_research_artifact_v2_verification_v1_migrates_as_legacy_unverified():
    report = copy.deepcopy(_verified_artifact())
    section = report["verification"]["sections"][0]
    legacy_evidence = []
    for commitment in section["evidence_commitments"]:
        projected = dict(commitment)
        projected.pop("span_start", None)
        projected.pop("span_end", None)
        legacy_evidence.append(projected)
    report["verification"] = {
        "schema_version": "research-verification-v1",
        "aggregate": copy.deepcopy(report["verification_metrics"]),
        "sections": [
            {
                "section_id": section["section_id"],
                "generation_status": section["generation_status"],
                "verification_status": section["verification_status"],
                "verification_reason_code": section["verification_reason_code"],
                "requirement_results": copy.deepcopy(section["requirement_results"]),
                "claim_audit": copy.deepcopy(section["claim_audit"]),
                "coverage_audit": copy.deepcopy(section["coverage_audit"]),
                "evidence_commitments": legacy_evidence,
            }
        ],
    }
    payload = {
        "artifact_schema_version": RESEARCH_ARTIFACT_VERSION,
        "content": report["content"],
        "citation_ledger": report["citation_ledger"],
        "provenance": report["provenance"],
        "verification": report["verification"],
        "metadata": {
            "version": report["version"],
            "generated_at": report["generated_at"],
        },
    }
    report["sha256"] = hashlib.sha256(
        research_provenance._canonical_json(payload).encode("utf-8")
    ).hexdigest()

    assert research_artifact_integrity_status(report) == "legacy-unverified"

    report["verification"]["sections"][0]["verification_reason_code"] = "tampered"
    assert research_artifact_integrity_status(report) == "invalid"
