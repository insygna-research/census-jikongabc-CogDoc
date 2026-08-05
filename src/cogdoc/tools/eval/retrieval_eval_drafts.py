from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = 1
_SPACE_PATTERN = re.compile(r"\s+")


class EvidenceUnitTask(str, Enum):
    """Task-independent atomic evidence unit shapes."""

    QA_REQUIREMENT = "qa_requirement"
    SUMMARY_SECTION = "summary_section"
    COMPARE_SOURCE_DIMENSION = "compare_source_dimension"


class DraftStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class DatasetPartition(str, Enum):
    """Training feedback and immutable release gates must never mix implicitly."""

    TRAINING = "training"
    RELEASE_GATE = "release_gate"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value: Any) -> str:
    return _SPACE_PATTERN.sub(" ", str(value or "")).strip()


def _canonical_text(value: Any) -> str:
    return unicodedata.normalize("NFKC", _text(value)).casefold()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceVersion(_StrictModel):
    source: str
    sha256: str

    @field_validator("source", "sha256", mode="before")
    @classmethod
    def _normalize_text(cls, value: Any) -> str:
        return _text(value)


class RetrievalIdentitySnapshot(_StrictModel):
    """Immutable index provenance captured when a proposal is created."""

    index_generation: str = ""
    index_build_version: str = ""
    chunk_identity_version: str = ""
    source_versions: list[SourceVersion] = Field(default_factory=list)

    @field_validator(
        "index_generation",
        "index_build_version",
        "chunk_identity_version",
        mode="before",
    )
    @classmethod
    def _normalize_text(cls, value: Any) -> str:
        return _text(value)


class ChunkIdentity(_StrictModel):
    """Stable child identity and the source version it was created from."""

    chunk_id: str
    source: str
    source_sha256: str
    parent_chunk_id: str = ""

    @field_validator(
        "chunk_id", "source", "source_sha256", "parent_chunk_id", mode="before"
    )
    @classmethod
    def _normalize_text(cls, value: Any) -> str:
        return _text(value)


class AcceptableEvidence(ChunkIdentity):
    """An acceptable chunk, optionally narrowed to a half-open character span."""

    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)

    @field_validator("start", "end", mode="before")
    @classmethod
    def _reject_boolean_offset(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("span offsets must be integers, not booleans")
        return value

    @model_validator(mode="after")
    def _validate_span(self) -> AcceptableEvidence:
        if (self.start is None) != (self.end is None):
            raise ValueError("span start and end must be provided together")
        if self.start is not None and self.end is not None and self.end <= self.start:
            raise ValueError("span end must be greater than start")
        return self


class EvidenceUnitDraft(_StrictModel):
    """One atomic requirement shared by QA, Summary, and Compare."""

    unit_id: str
    task_kind: EvidenceUnitTask
    label: str
    retrieval_query: str
    recovery_query: str = ""
    source: str = ""
    dimension_id: str = ""
    expected_status: Literal["supported", "no_evidence"] | None = None
    acceptable_evidence: list[AcceptableEvidence] = Field(default_factory=list)
    hard_negative_chunks: list[ChunkIdentity] = Field(default_factory=list)

    @field_validator(
        "unit_id",
        "label",
        "retrieval_query",
        "recovery_query",
        "source",
        "dimension_id",
        mode="before",
    )
    @classmethod
    def _normalize_text(cls, value: Any) -> str:
        return _text(value)

    @model_validator(mode="after")
    def _validate_task_scope(self) -> EvidenceUnitDraft:
        if not self.unit_id or not self.label:
            raise ValueError("unit_id and label are required")
        if self.task_kind is EvidenceUnitTask.SUMMARY_SECTION and not self.dimension_id:
            raise ValueError("summary units require dimension_id")
        if self.task_kind is EvidenceUnitTask.COMPARE_SOURCE_DIMENSION and (
            not self.source or not self.dimension_id
        ):
            raise ValueError("compare units require source and dimension_id")
        return self


class RetrievalIndexSnapshot(RetrievalIdentitySnapshot):
    kb_id: str

    @field_validator("kb_id", mode="before")
    @classmethod
    def _normalize_text(cls, value: Any) -> str:
        return _text(value)


class RetrievalEvalDraft(_StrictModel):
    schema_version: int = SCHEMA_VERSION
    draft_id: str
    dedupe_key: str
    status: DraftStatus = DraftStatus.PENDING
    dataset_partition: DatasetPartition = DatasetPartition.TRAINING
    kb_id: str
    query: str
    layer: str = ""
    no_answer: bool = False
    units: list[EvidenceUnitDraft]
    hard_negative_chunks: list[ChunkIdentity] = Field(default_factory=list)
    index_generation: str = ""
    index_build_version: str = ""
    chunk_identity_version: str = ""
    source_versions: list[SourceVersion] = Field(default_factory=list)
    # None identifies a legacy record whose key omitted index provenance.
    identity_snapshot: RetrievalIdentitySnapshot | None = None
    origin_trace_id: str = ""
    origin_feedback_id: str = ""
    created_at: str
    updated_at: str
    reviewed_at: str = ""
    reviewed_by: str = ""
    rejection_reason: str = ""
    revision: int = Field(default=1, ge=1)

    @field_validator(
        "draft_id",
        "dedupe_key",
        "kb_id",
        "query",
        "layer",
        "index_generation",
        "index_build_version",
        "chunk_identity_version",
        "origin_trace_id",
        "origin_feedback_id",
        "created_at",
        "updated_at",
        "reviewed_at",
        "reviewed_by",
        "rejection_reason",
        mode="before",
    )
    @classmethod
    def _normalize_text(cls, value: Any) -> str:
        return _text(value)

    @model_validator(mode="after")
    def _validate_identity_and_state(self) -> RetrievalEvalDraft:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported draft schema version: {self.schema_version}")
        if not self.kb_id or not self.query or not self.units:
            raise ValueError(
                "kb_id, query, and at least one evidence unit are required"
            )
        unit_ids = [unit.unit_id for unit in self.units]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("unit_id values must be unique")
        expected_key = draft_dedupe_key(
            kb_id=self.kb_id,
            query=self.query,
            dataset_partition=self.dataset_partition,
            units=self.units,
            identity_snapshot=self.identity_snapshot,
        )
        if self.dedupe_key != expected_key:
            raise ValueError("dedupe_key does not match the draft identity")
        if self.draft_id != draft_id_from_key(expected_key):
            raise ValueError("draft_id does not match dedupe_key")
        if self.status is DraftStatus.PENDING and (
            self.reviewed_at or self.reviewed_by or self.rejection_reason
        ):
            raise ValueError("pending drafts cannot contain review metadata")
        if self.status is DraftStatus.APPROVED and (
            not self.reviewed_at or not self.reviewed_by or self.rejection_reason
        ):
            raise ValueError("approved drafts require reviewer metadata")
        if self.status is DraftStatus.REJECTED and (
            not self.reviewed_at or not self.reviewed_by or not self.rejection_reason
        ):
            raise ValueError("rejected drafts require reviewer and reason")
        return self


def draft_dedupe_key(
    *,
    kb_id: str,
    query: str,
    dataset_partition: DatasetPartition | str,
    units: Sequence[EvidenceUnitDraft | Mapping[str, Any]],
    identity_snapshot: RetrievalIdentitySnapshot | Mapping[str, Any] | None = None,
) -> str:
    """Hash immutable sample intent and its optional creation-time snapshot."""

    unit_rows: list[dict[str, str]] = []
    for raw_unit in units:
        unit = (
            raw_unit
            if isinstance(raw_unit, EvidenceUnitDraft)
            else EvidenceUnitDraft.model_validate(raw_unit)
        )
        unit_rows.append(
            {
                "unit_id": _canonical_text(unit.unit_id),
                "task_kind": unit.task_kind.value,
                "label": _canonical_text(unit.label),
                "source": _canonical_text(unit.source),
                "dimension_id": _canonical_text(unit.dimension_id),
            }
        )
    partition = DatasetPartition(dataset_partition).value
    canonical: dict[str, Any] = {
        "kb_id": _canonical_text(kb_id),
        "query": _canonical_text(query),
        "dataset_partition": partition,
        "units": unit_rows,
    }
    if identity_snapshot is not None:
        snapshot = (
            identity_snapshot
            if isinstance(identity_snapshot, RetrievalIdentitySnapshot)
            else RetrievalIdentitySnapshot.model_validate(identity_snapshot)
        )
        canonical["identity_snapshot"] = {
            "index_generation": _canonical_text(snapshot.index_generation),
            "index_build_version": _canonical_text(snapshot.index_build_version),
            "chunk_identity_version": _canonical_text(snapshot.chunk_identity_version),
            "source_versions": sorted(
                (
                    _canonical_text(item.source),
                    _canonical_text(item.sha256),
                )
                for item in snapshot.source_versions
            ),
        }
    encoded = json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def draft_id_from_key(dedupe_key: str) -> str:
    return f"retrieval-eval-{dedupe_key[:32]}"


def draft_snapshot_identity_key(
    draft: RetrievalEvalDraft | Mapping[str, Any],
) -> str:
    """Compare legacy and new drafts by their effective index snapshot."""

    candidate = (
        draft
        if isinstance(draft, RetrievalEvalDraft)
        else RetrievalEvalDraft.model_validate(draft)
    )
    snapshot = candidate.identity_snapshot or RetrievalIdentitySnapshot(
        index_generation=candidate.index_generation,
        index_build_version=candidate.index_build_version,
        chunk_identity_version=candidate.chunk_identity_version,
        source_versions=candidate.source_versions,
    )
    return draft_dedupe_key(
        kb_id=candidate.kb_id,
        query=candidate.query,
        dataset_partition=candidate.dataset_partition,
        units=candidate.units,
        identity_snapshot=snapshot,
    )


def create_pending_draft(
    *,
    kb_id: str,
    query: str,
    units: Sequence[EvidenceUnitDraft | Mapping[str, Any]],
    dataset_partition: DatasetPartition | str = DatasetPartition.TRAINING,
    no_answer: bool = False,
    layer: str = "",
    hard_negative_chunks: Sequence[ChunkIdentity | Mapping[str, Any]] = (),
    index_generation: str = "",
    index_build_version: str = "",
    chunk_identity_version: str = "",
    source_versions: Sequence[SourceVersion | Mapping[str, Any]] = (),
    origin_trace_id: str = "",
    origin_feedback_id: str = "",
    now: str | None = None,
) -> RetrievalEvalDraft:
    normalized_units = [
        unit
        if isinstance(unit, EvidenceUnitDraft)
        else EvidenceUnitDraft.model_validate(unit)
        for unit in units
    ]
    normalized_hard_negatives = [
        item if isinstance(item, ChunkIdentity) else ChunkIdentity.model_validate(item)
        for item in hard_negative_chunks
    ]
    normalized_source_versions = [
        item if isinstance(item, SourceVersion) else SourceVersion.model_validate(item)
        for item in source_versions
    ]
    identity_snapshot = RetrievalIdentitySnapshot(
        index_generation=index_generation,
        index_build_version=index_build_version,
        chunk_identity_version=chunk_identity_version,
        source_versions=normalized_source_versions,
    )
    partition = DatasetPartition(dataset_partition)
    dedupe_key = draft_dedupe_key(
        kb_id=kb_id,
        query=query,
        dataset_partition=partition,
        units=normalized_units,
        identity_snapshot=identity_snapshot,
    )
    timestamp = _text(now) or _now_iso()
    return RetrievalEvalDraft(
        draft_id=draft_id_from_key(dedupe_key),
        dedupe_key=dedupe_key,
        dataset_partition=partition,
        kb_id=kb_id,
        query=query,
        layer=layer,
        no_answer=no_answer,
        units=normalized_units,
        hard_negative_chunks=normalized_hard_negatives,
        index_generation=index_generation,
        index_build_version=index_build_version,
        chunk_identity_version=chunk_identity_version,
        source_versions=normalized_source_versions,
        identity_snapshot=identity_snapshot,
        origin_trace_id=origin_trace_id,
        origin_feedback_id=origin_feedback_id,
        created_at=timestamp,
        updated_at=timestamp,
    )


def _trace_output(trace: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for raw_step in _sequence(trace.get("steps")):
        snapshot = _mapping(_mapping(raw_step).get("output_snapshot"))
        merged.update(snapshot)
    merged.update(_mapping(trace.get("output_payload")))
    merged.update(_mapping(trace.get("output")))
    return merged


def _summary_units(output: Mapping[str, Any], query: str) -> list[EvidenceUnitDraft]:
    plans = _sequence(output.get("summary_section_plans")) or _sequence(
        output.get("summary_section_results")
    )
    source = _text(output.get("summary_source"))
    units = []
    for position, raw_plan in enumerate(plans):
        plan = _mapping(raw_plan)
        section_id = _text(plan.get("section_id")) or f"section-{position + 1}"
        title = _text(plan.get("title")) or section_id
        instruction = _text(plan.get("instruction"))
        retrieval_query = _text(" ".join((query, source, title, instruction)))
        recovery_query = (
            _text(" ".join((source, title, instruction))) or retrieval_query
        )
        units.append(
            EvidenceUnitDraft(
                unit_id=section_id,
                task_kind=EvidenceUnitTask.SUMMARY_SECTION,
                label=title,
                retrieval_query=retrieval_query,
                recovery_query=recovery_query,
                source=source,
                dimension_id=section_id,
            )
        )
    return units


def _runtime_evidence_units(output: Mapping[str, Any]) -> list[EvidenceUnitDraft]:
    """Prefer the exact online plan so evaluation never rebuilds query text."""

    units: list[EvidenceUnitDraft] = []
    for position, raw_unit in enumerate(_sequence(output.get("evidence_units"))):
        unit = _mapping(raw_unit)
        task_kind = _text(unit.get("task_kind"))
        if task_kind not in {item.value for item in EvidenceUnitTask}:
            continue
        binding = _mapping(unit.get("binding"))
        source = _text(binding.get("source"))
        dimension_id = _text(binding.get("dimension_id") or binding.get("section_id"))
        if task_kind == EvidenceUnitTask.QA_REQUIREMENT.value:
            unit_id = _text(binding.get("requirement_id"))
        elif task_kind == EvidenceUnitTask.SUMMARY_SECTION.value:
            unit_id = _text(binding.get("section_id"))
        else:
            unit_id = _text(unit.get("unit_id"))
        unit_id = unit_id or _text(unit.get("unit_id")) or f"unit-{position + 1}"
        units.append(
            EvidenceUnitDraft(
                unit_id=unit_id,
                task_kind=EvidenceUnitTask(task_kind),
                label=_text(unit.get("label")) or unit_id,
                retrieval_query=_text(unit.get("retrieval_query")),
                recovery_query=_text(unit.get("recovery_query")),
                source=source,
                dimension_id=dimension_id,
            )
        )
    return units


def _compare_units(output: Mapping[str, Any], query: str) -> list[EvidenceUnitDraft]:
    sources = [_text(value) for value in _sequence(output.get("compare_sources"))]
    dimensions = list(_sequence(output.get("compare_dimensions")))
    if not sources:
        sources = [
            _text(_mapping(profile).get("source"))
            for profile in _sequence(output.get("document_profiles"))
        ]
    if not dimensions:
        seen_dimensions: dict[str, dict[str, str]] = {}
        for raw_profile in _sequence(output.get("document_profiles")):
            for raw_cell in _sequence(_mapping(raw_profile).get("cells")):
                cell = _mapping(raw_cell)
                dimension_id = _text(cell.get("dimension_id"))
                if dimension_id:
                    seen_dimensions.setdefault(
                        dimension_id,
                        {"dimension_id": dimension_id, "title": dimension_id},
                    )
        dimensions = list(seen_dimensions.values())
    units = []
    for source in dict.fromkeys(value for value in sources if value):
        for position, raw_dimension in enumerate(dimensions):
            dimension = _mapping(raw_dimension)
            dimension_id = _text(dimension.get("dimension_id")) or (
                f"dimension-{position + 1}"
            )
            title = _text(dimension.get("title")) or dimension_id
            instruction = _text(dimension.get("instruction"))
            retrieval_query = _text(" ".join((query, source, title, instruction)))
            recovery_query = _text(" ".join((source, title, instruction))) or (
                retrieval_query
            )
            units.append(
                EvidenceUnitDraft(
                    unit_id=f"{source}::{dimension_id}",
                    task_kind=EvidenceUnitTask.COMPARE_SOURCE_DIMENSION,
                    label=title,
                    retrieval_query=retrieval_query,
                    recovery_query=recovery_query,
                    source=source,
                    dimension_id=dimension_id,
                )
            )
    return units


def _qa_units(output: Mapping[str, Any], query: str) -> list[EvidenceUnitDraft]:
    units = []
    for position, raw_requirement in enumerate(
        _sequence(output.get("evidence_requirements"))
    ):
        requirement = _mapping(raw_requirement)
        unit_id = _text(requirement.get("requirement_id")) or f"r{position + 1}"
        label = _text(requirement.get("question")) or query
        retrieval_query = _text(requirement.get("retrieval_query")) or label
        recovery_query = _text(requirement.get("recovery_query"))
        units.append(
            EvidenceUnitDraft(
                unit_id=unit_id,
                task_kind=EvidenceUnitTask.QA_REQUIREMENT,
                label=label,
                retrieval_query=retrieval_query,
                recovery_query=recovery_query,
            )
        )
    return units or [
        EvidenceUnitDraft(
            unit_id="r1",
            task_kind=EvidenceUnitTask.QA_REQUIREMENT,
            label=query,
            retrieval_query=query,
            recovery_query="",
        )
    ]


def build_retrieval_eval_draft(
    feedback: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    dataset_partition: DatasetPartition | str = DatasetPartition.TRAINING,
    now: str | None = None,
) -> RetrievalEvalDraft:
    """Build an unlabelled pending proposal from trusted feedback and trace data.

    Runtime-retrieved chunks are intentionally ignored: observed evidence is not
    gold evidence, nor is a missed candidate automatically a hard negative.
    """

    output = _trace_output(trace)
    trace_input = _mapping(trace.get("input")) or _mapping(trace.get("input_payload"))
    config = _mapping(trace.get("config"))
    query = _text(feedback.get("query")) or _text(trace_input.get("query"))
    kb_id = (
        _text(feedback.get("kb_id"))
        or _text(trace_input.get("doc_id"))
        or _text(config.get("doc_id"))
    )
    task_type = _text(trace.get("task_type")) or _text(feedback.get("task_type"))
    if task_type not in {"qa", "summary", "compare"}:
        raise ValueError(f"unsupported retrieval eval task_type: {task_type or '-'}")
    units = _runtime_evidence_units(output)
    if not units and task_type == "summary":
        units = _summary_units(output, query)
    elif not units and task_type == "compare":
        units = _compare_units(output, query)
    elif not units and task_type == "qa":
        units = _qa_units(output, query)
    if not units:
        raise ValueError(f"trace does not contain {task_type} evidence units")

    raw_source_versions = config.get("source_versions") or trace.get("source_versions")
    source_versions = [
        SourceVersion.model_validate(item)
        for item in _sequence(raw_source_versions)
        if isinstance(item, Mapping)
    ]
    return create_pending_draft(
        kb_id=kb_id,
        query=query,
        units=units,
        dataset_partition=dataset_partition,
        no_answer=(
            feedback.get("no_answer") is True
            or feedback.get("expected_no_answer") is True
        ),
        index_generation=_text(
            config.get("index_generation") or trace.get("index_generation")
        ),
        index_build_version=_text(
            config.get("index_build_version") or trace.get("index_build_version")
        ),
        chunk_identity_version=_text(
            config.get("chunk_identity_version") or trace.get("chunk_identity_version")
        ),
        source_versions=source_versions,
        origin_trace_id=_text(trace.get("trace_id")),
        origin_feedback_id=_text(feedback.get("feedback_id") or feedback.get("id")),
        now=now,
    )


def _resolved_expected_status(
    unit: EvidenceUnitDraft, *, legacy_no_answer: bool
) -> Literal["supported", "no_evidence"]:
    """Resolve new unit labels while keeping legacy draft-level semantics."""

    if unit.expected_status is not None:
        return unit.expected_status
    return "no_evidence" if legacy_no_answer else "supported"


def _normalized_units(candidate: RetrievalEvalDraft) -> list[dict[str, Any]]:
    """Freeze unit statuses and copy legacy global negatives into every unit."""

    normalized: list[dict[str, Any]] = []
    for unit in candidate.units:
        payload = unit.model_dump(mode="json")
        payload["expected_status"] = _resolved_expected_status(
            unit, legacy_no_answer=candidate.no_answer
        )
        hard_negatives = list(unit.hard_negative_chunks)
        seen_chunk_ids = {item.chunk_id for item in hard_negatives}
        for legacy_negative in candidate.hard_negative_chunks:
            if legacy_negative.chunk_id in seen_chunk_ids:
                continue
            hard_negatives.append(legacy_negative)
            seen_chunk_ids.add(legacy_negative.chunk_id)
        payload["hard_negative_chunks"] = [
            item.model_dump(mode="json") for item in hard_negatives
        ]
        normalized.append(payload)
    return normalized


def approval_errors(draft: RetrievalEvalDraft | Mapping[str, Any]) -> list[str]:
    candidate = (
        draft
        if isinstance(draft, RetrievalEvalDraft)
        else RetrievalEvalDraft.model_validate(draft)
    )
    errors: list[str] = []
    for field in (
        "index_generation",
        "index_build_version",
        "chunk_identity_version",
    ):
        if not getattr(candidate, field):
            errors.append(f"{field}_required")

    source_hashes: dict[str, str] = {}
    for source_version in candidate.source_versions:
        if not source_version.source or not source_version.sha256:
            errors.append("source_version_incomplete")
            continue
        existing = source_hashes.get(source_version.source)
        if existing is not None and existing != source_version.sha256:
            errors.append(f"source_version_conflict:{source_version.source}")
        source_hashes[source_version.source] = source_version.sha256

    all_acceptable_ids: set[str] = set()
    for unit in candidate.units:
        expected_status = _resolved_expected_status(
            unit, legacy_no_answer=candidate.no_answer
        )
        if not unit.retrieval_query:
            errors.append(f"retrieval_query_required:{unit.unit_id}")
        if not unit.recovery_query:
            errors.append(f"recovery_query_required:{unit.unit_id}")
        if expected_status == "no_evidence" and unit.acceptable_evidence:
            error_prefix = (
                "no_answer_has_acceptable_evidence"
                if unit.expected_status is None and candidate.no_answer
                else "no_evidence_has_acceptable_evidence"
            )
            errors.append(f"{error_prefix}:{unit.unit_id}")
        if expected_status == "supported" and not unit.acceptable_evidence:
            errors.append(f"acceptable_evidence_required:{unit.unit_id}")
        acceptable_ids: set[str] = set()
        seen_targets: set[tuple[str, int | None, int | None]] = set()
        for target in unit.acceptable_evidence:
            if not target.chunk_id or not target.source or not target.source_sha256:
                errors.append(f"acceptable_chunk_identity_incomplete:{unit.unit_id}")
                continue
            target_key = (target.chunk_id, target.start, target.end)
            if target_key in seen_targets:
                errors.append(f"duplicate_acceptable_evidence:{unit.unit_id}")
            seen_targets.add(target_key)
            acceptable_ids.add(target.chunk_id)
            all_acceptable_ids.add(target.chunk_id)
            if source_hashes.get(target.source) != target.source_sha256:
                errors.append(f"source_version_mismatch:{unit.unit_id}:{target.source}")

        unit_hard_negative_ids: set[str] = set()
        for hard_negative in unit.hard_negative_chunks:
            if (
                not hard_negative.chunk_id
                or not hard_negative.source
                or not hard_negative.source_sha256
            ):
                errors.append(f"hard_negative_chunk_identity_incomplete:{unit.unit_id}")
                continue
            if hard_negative.chunk_id in unit_hard_negative_ids:
                errors.append(
                    f"duplicate_hard_negative:{unit.unit_id}:{hard_negative.chunk_id}"
                )
            unit_hard_negative_ids.add(hard_negative.chunk_id)
            if source_hashes.get(hard_negative.source) != hard_negative.source_sha256:
                errors.append(
                    "hard_negative_source_version_mismatch:"
                    f"{unit.unit_id}:{hard_negative.source}"
                )
        for chunk_id in sorted(acceptable_ids & unit_hard_negative_ids):
            errors.append(f"acceptable_hard_negative_overlap:{unit.unit_id}:{chunk_id}")

    # Draft-level negatives retain their original global meaning for legacy rows.
    hard_negative_ids: set[str] = set()
    for hard_negative in candidate.hard_negative_chunks:
        if (
            not hard_negative.chunk_id
            or not hard_negative.source
            or not hard_negative.source_sha256
        ):
            errors.append("hard_negative_chunk_identity_incomplete")
            continue
        if hard_negative.chunk_id in hard_negative_ids:
            errors.append(f"duplicate_hard_negative:{hard_negative.chunk_id}")
        hard_negative_ids.add(hard_negative.chunk_id)
        if source_hashes.get(hard_negative.source) != hard_negative.source_sha256:
            errors.append(
                f"hard_negative_source_version_mismatch:{hard_negative.source}"
            )
    for chunk_id in sorted(all_acceptable_ids & hard_negative_ids):
        errors.append(f"acceptable_hard_negative_overlap:{chunk_id}")
    return list(dict.fromkeys(errors))


def approve_draft(
    draft: RetrievalEvalDraft | Mapping[str, Any],
    *,
    reviewer: str,
    now: str | None = None,
) -> RetrievalEvalDraft:
    candidate = (
        draft
        if isinstance(draft, RetrievalEvalDraft)
        else RetrievalEvalDraft.model_validate(draft)
    )
    if candidate.status is not DraftStatus.PENDING:
        raise ValueError("only pending drafts can be approved")
    reviewer = _text(reviewer)
    if not reviewer:
        raise ValueError("reviewer is required")
    errors = approval_errors(candidate)
    if errors:
        raise ValueError("draft is not approvable: " + ", ".join(errors))
    timestamp = _text(now) or _now_iso()
    normalized_units = _normalized_units(candidate)
    derived_no_answer = all(
        unit["expected_status"] == "no_evidence" for unit in normalized_units
    )
    return RetrievalEvalDraft.model_validate(
        {
            **candidate.model_dump(mode="json"),
            "no_answer": derived_no_answer,
            "units": normalized_units,
            "status": DraftStatus.APPROVED.value,
            "reviewed_at": timestamp,
            "reviewed_by": reviewer,
            "rejection_reason": "",
            "updated_at": timestamp,
            "revision": candidate.revision + 1,
        }
    )


def apply_review_annotations(
    draft: RetrievalEvalDraft | Mapping[str, Any],
    annotations: Mapping[str, Any],
) -> RetrievalEvalDraft:
    """Apply only reviewer-authored gold labels to a pending proposal."""

    candidate = (
        draft
        if isinstance(draft, RetrievalEvalDraft)
        else RetrievalEvalDraft.model_validate(draft)
    )
    if candidate.status is not DraftStatus.PENDING:
        raise ValueError("only pending drafts can receive review annotations")
    allowed_fields = {
        "no_answer",
        "layer",
        "units",
        "hard_negative_chunks",
        "index_generation",
        "index_build_version",
        "chunk_identity_version",
        "source_versions",
    }
    unknown_fields = sorted(set(annotations) - allowed_fields)
    if unknown_fields:
        raise ValueError(
            "unsupported review annotation fields: " + ", ".join(unknown_fields)
        )

    payload = candidate.model_dump(mode="json")
    for field in allowed_fields - {"units"}:
        if field in annotations:
            payload[field] = annotations[field]

    if "units" in annotations:
        raw_unit_annotations = annotations.get("units")
        if not isinstance(raw_unit_annotations, Sequence) or isinstance(
            raw_unit_annotations, (str, bytes, bytearray)
        ):
            raise ValueError("review units must be a list")
        unit_patches: dict[str, Mapping[str, Any]] = {}
        for raw_patch in raw_unit_annotations:
            if not isinstance(raw_patch, Mapping):
                raise ValueError("review unit annotations must be objects")
            patch = _mapping(raw_patch)
            unit_id = _text(patch.get("unit_id"))
            if not unit_id:
                raise ValueError("review unit annotation requires unit_id")
            if unit_id in unit_patches:
                raise ValueError(f"duplicate review unit annotation: {unit_id}")
            unknown_unit_fields = sorted(
                set(patch)
                - {
                    "unit_id",
                    "retrieval_query",
                    "recovery_query",
                    "expected_status",
                    "acceptable_evidence",
                    "hard_negative_chunks",
                }
            )
            if unknown_unit_fields:
                raise ValueError(
                    f"unsupported review fields for {unit_id}: "
                    + ", ".join(unknown_unit_fields)
                )
            unit_patches[unit_id] = patch

        known_ids = {unit.unit_id for unit in candidate.units}
        unknown_ids = sorted(set(unit_patches) - known_ids)
        if unknown_ids:
            raise ValueError("unknown review unit_id: " + ", ".join(unknown_ids))
        updated_units = []
        for unit in candidate.units:
            unit_payload = unit.model_dump(mode="json")
            unit_payload.update(unit_patches.get(unit.unit_id, {}))
            updated_units.append(unit_payload)
        payload["units"] = updated_units
    return RetrievalEvalDraft.model_validate(payload)


def reject_draft(
    draft: RetrievalEvalDraft | Mapping[str, Any],
    *,
    reviewer: str,
    reason: str,
    now: str | None = None,
) -> RetrievalEvalDraft:
    candidate = (
        draft
        if isinstance(draft, RetrievalEvalDraft)
        else RetrievalEvalDraft.model_validate(draft)
    )
    if candidate.status is not DraftStatus.PENDING:
        raise ValueError("only pending drafts can be rejected")
    reviewer = _text(reviewer)
    reason = _text(reason)
    if not reviewer or not reason:
        raise ValueError("reviewer and rejection reason are required")
    timestamp = _text(now) or _now_iso()
    return RetrievalEvalDraft.model_validate(
        {
            **candidate.model_dump(mode="json"),
            "status": DraftStatus.REJECTED.value,
            "reviewed_at": timestamp,
            "reviewed_by": reviewer,
            "rejection_reason": reason,
            "updated_at": timestamp,
            "revision": candidate.revision + 1,
        }
    )


def detect_stale_reasons(
    draft: RetrievalEvalDraft | Mapping[str, Any],
    current: RetrievalIndexSnapshot | Mapping[str, Any],
) -> tuple[str, ...]:
    candidate = (
        draft
        if isinstance(draft, RetrievalEvalDraft)
        else RetrievalEvalDraft.model_validate(draft)
    )
    snapshot = (
        current
        if isinstance(current, RetrievalIndexSnapshot)
        else RetrievalIndexSnapshot.model_validate(current)
    )
    reasons = []
    if candidate.kb_id != snapshot.kb_id:
        reasons.append("kb_id_changed")
    for field in (
        "index_generation",
        "index_build_version",
        "chunk_identity_version",
    ):
        if getattr(candidate, field) != getattr(snapshot, field):
            reasons.append(f"{field}_changed")
    annotated = {item.source: item.sha256 for item in candidate.source_versions}
    active = {item.source: item.sha256 for item in snapshot.source_versions}
    for source in sorted(annotated.keys() - active.keys()):
        reasons.append(f"source_removed:{source}")
    for source in sorted(active.keys() - annotated.keys()):
        reasons.append(f"source_added:{source}")
    for source in sorted(annotated.keys() & active.keys()):
        if annotated[source] != active[source]:
            reasons.append(f"source_sha256_changed:{source}")
    return tuple(reasons)


def is_stale(
    draft: RetrievalEvalDraft | Mapping[str, Any],
    current: RetrievalIndexSnapshot | Mapping[str, Any],
) -> bool:
    return bool(detect_stale_reasons(draft, current))


def export_retrieval_eval_case(
    draft: RetrievalEvalDraft | Mapping[str, Any],
) -> dict[str, Any]:
    """Return the existing retrieval_eval JSONL row shape without writing a file."""

    candidate = (
        draft
        if isinstance(draft, RetrievalEvalDraft)
        else RetrievalEvalDraft.model_validate(draft)
    )
    if candidate.status is not DraftStatus.APPROVED:
        raise ValueError("only approved drafts can be exported")
    errors = approval_errors(candidate)
    if errors:
        raise ValueError("approved draft is invalid: " + ", ".join(errors))

    units = [
        EvidenceUnitDraft.model_validate(unit) for unit in _normalized_units(candidate)
    ]
    expected_unit_statuses = {unit.unit_id: unit.expected_status for unit in units}
    derived_no_answer = all(
        status == "no_evidence" for status in expected_unit_statuses.values()
    )
    expected_sources = sorted(
        {
            target.source
            for unit in units
            if unit.expected_status == "supported"
            for target in unit.acceptable_evidence
        }
    )
    evidence_requirements = [
        {
            "requirement_id": unit.unit_id,
            "question": unit.label,
            "retrieval_query": unit.retrieval_query,
            "recovery_query": unit.recovery_query,
            "expected_status": unit.expected_status,
            "hard_negative_chunk_ids": sorted(
                {target.chunk_id for target in unit.hard_negative_chunks}
            ),
        }
        for unit in units
    ]
    gold_requirements = []
    for unit in units:
        if unit.expected_status == "supported":
            identities = [
                {
                    "chunk_id": target.chunk_id,
                    "parent_chunk_id": target.parent_chunk_id,
                    "source": target.source,
                    "source_sha256": target.source_sha256,
                }
                for target in unit.acceptable_evidence
            ]
            spans = [
                {
                    "chunk_id": target.chunk_id,
                    "start": target.start,
                    "end": target.end,
                }
                for target in unit.acceptable_evidence
                if target.start is not None and target.end is not None
            ]
            gold_requirements.append(
                {
                    "requirement_id": unit.unit_id,
                    "acceptable_chunk_ids": sorted(
                        {target.chunk_id for target in unit.acceptable_evidence}
                    ),
                    "acceptable_sources": sorted(
                        {target.source for target in unit.acceptable_evidence}
                    ),
                    "acceptable_spans": spans,
                    "acceptable_chunk_identities": identities,
                    "task_kind": unit.task_kind.value,
                    "source": unit.source,
                    "dimension_id": unit.dimension_id,
                    "hard_negative_chunk_ids": sorted(
                        {target.chunk_id for target in unit.hard_negative_chunks}
                    ),
                }
            )

    positive_chunk_ids = {
        target.chunk_id
        for unit in units
        if unit.expected_status == "supported"
        for target in unit.acceptable_evidence
    }
    hard_negative_chunk_ids_by_unit = {
        unit.unit_id: sorted({target.chunk_id for target in unit.hard_negative_chunks})
        for unit in units
    }
    # The legacy root field can only represent globally negative chunks.  Keep
    # safe values for old runners, but never flatten a chunk that is positive
    # for a different unit.
    unit_negative_sets = [
        set(chunk_ids) for chunk_ids in hard_negative_chunk_ids_by_unit.values()
    ]
    globally_safe_hard_negative_ids = sorted(
        (set.intersection(*unit_negative_sets) if unit_negative_sets else set())
        - positive_chunk_ids
    )
    layer = candidate.layer
    if not layer:
        if derived_no_answer:
            layer = "no-answer"
        elif len(expected_sources) > 1:
            layer = "multi-source"
        else:
            layer = "single-source"
    return {
        "id": candidate.draft_id,
        "query": candidate.query,
        "doc_id": candidate.kb_id,
        "layer": layer,
        "expected_sources": [] if derived_no_answer else expected_sources,
        "rewritten_queries": list(
            dict.fromkeys(unit.retrieval_query for unit in units)
        ),
        "evidence_requirements": evidence_requirements,
        "gold_requirements": gold_requirements,
        "expected_unit_statuses": expected_unit_statuses,
        "hard_negative_chunk_ids_by_unit": hard_negative_chunk_ids_by_unit,
        "hard_negative_chunk_ids": globally_safe_hard_negative_ids,
        "dataset_partition": candidate.dataset_partition.value,
        "index_generation": candidate.index_generation,
        "index_build_version": candidate.index_build_version,
        "chunk_identity_version": candidate.chunk_identity_version,
        "source_versions": [
            item.model_dump(mode="json") for item in candidate.source_versions
        ],
        "annotation_provenance": {
            "draft_id": candidate.draft_id,
            "origin_trace_id": candidate.origin_trace_id,
            "origin_feedback_id": candidate.origin_feedback_id,
            "reviewed_at": candidate.reviewed_at,
            "reviewed_by": candidate.reviewed_by,
        },
    }
