from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


_SPACE_PATTERN = re.compile(r"\s+")
_UNIT_ID_PATTERN = re.compile(r"^eu_[0-9a-f]{24}$")
_GROUP_ID_PATTERN = re.compile(r"^eg_[0-9a-f]{24}$")


class EvidenceUnitKind(str, Enum):
    """Agent-independent shapes of one atomic evidence obligation."""

    QA_REQUIREMENT = "qa_requirement"
    SUMMARY_SECTION = "summary_section"
    COMPARE_SOURCE_DIMENSION = "compare_source_dimension"


class EvidenceClosureStatus(str, Enum):
    """Semantic and operational outcomes must never be conflated."""

    SUPPORTED = "supported"
    NO_EVIDENCE = "no_evidence"
    CONTRADICTORY = "contradictory"
    RETRIEVAL_ERROR = "retrieval_error"
    VERIFICATION_ERROR = "verification_error"
    BUDGET_EXHAUSTED = "budget_exhausted"


class EvidenceSourceType(str, Enum):
    DOCUMENT = "document"
    DERIVED_KNOWLEDGE = "derived_knowledge"


def _text(value: Any) -> str:
    return _SPACE_PATTERN.sub(
        " ", unicodedata.normalize("NFKC", str(value or ""))
    ).strip()


def _require_text(value: Any, name: str) -> str:
    normalized = _text(value)
    if not normalized:
        raise ValueError(f"{name} is required")
    return normalized


def _strict_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _strict_positive_int(value: Any, name: str) -> int:
    normalized = _strict_nonnegative_int(value, name)
    if normalized == 0:
        raise ValueError(f"{name} must be greater than zero")
    return normalized


def _opaque_id(prefix: str, payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def _normalized_unique(values: Sequence[Any], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence of strings")
    normalized = tuple(_require_text(value, name) for value in values)
    keys = [value.casefold() for value in normalized]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{name} must not contain duplicates")
    return normalized


def _without_source_names(text: Any, sources: Sequence[str]) -> str:
    """Remove routing constraints from semantic retrieval text.

    Source names belong in ``EvidenceSourceScope``.  Leaving them in semantic
    query text makes filename tokens dominate short Summary/Compare queries and
    accidentally turns a hard source constraint into a ranking hint.
    """

    result = _text(text)
    for source in sorted(sources, key=len, reverse=True):
        result = re.sub(re.escape(source), " ", result, flags=re.IGNORECASE)
    return _text(result)


def _compose_query(*values: Any, excluded_sources: Sequence[str] = ()) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for value in values:
        part = _without_source_names(value, excluded_sources)
        key = part.casefold()
        if part and key not in seen:
            seen.add(key)
            parts.append(part)
    return _require_text(" ".join(parts), "semantic query")


@dataclass(frozen=True, slots=True)
class EvidenceSourceScope:
    """Hard source boundary applied before ranking and after every transform.

    An empty ``allowed_sources`` tuple means all document sources in the KB.
    Derived knowledge is separately opt-in.  When a scope is source-restricted,
    derived knowledge must carry an explicit matching ``related_source``.
    """

    allowed_sources: tuple[str, ...] = ()
    allow_derived_knowledge: bool = True

    def __post_init__(self) -> None:
        normalized = _normalized_unique(self.allowed_sources, "allowed_sources")
        object.__setattr__(self, "allowed_sources", normalized)
        if not isinstance(self.allow_derived_knowledge, bool):
            raise ValueError("allow_derived_knowledge must be a boolean")

    @property
    def is_restricted(self) -> bool:
        return bool(self.allowed_sources)

    def contains(
        self,
        *,
        source: str,
        source_type: EvidenceSourceType = EvidenceSourceType.DOCUMENT,
        related_source: str = "",
    ) -> bool:
        normalized_source = _text(source)
        normalized_related = _text(related_source)
        if source_type is EvidenceSourceType.DERIVED_KNOWLEDGE:
            if not self.allow_derived_knowledge:
                return False
            candidate_source = normalized_related
        elif source_type is EvidenceSourceType.DOCUMENT:
            candidate_source = normalized_source
        else:  # pragma: no cover - the enum prevents this for typed callers.
            return False
        if not candidate_source:
            return False
        if not self.allowed_sources:
            return True
        allowed = {value.casefold() for value in self.allowed_sources}
        return candidate_source.casefold() in allowed


@dataclass(frozen=True, slots=True)
class EvidenceTaskBinding:
    """Task coordinates retained separately from opaque runtime identifiers."""

    task_kind: EvidenceUnitKind
    requirement_id: str = ""
    section_id: str = ""
    source: str = ""
    dimension_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.task_kind, EvidenceUnitKind):
            raise ValueError("task_kind must be an EvidenceUnitKind")
        for name in ("requirement_id", "section_id", "source", "dimension_id"):
            object.__setattr__(self, name, _text(getattr(self, name)))

        if self.task_kind is EvidenceUnitKind.QA_REQUIREMENT:
            if not self.requirement_id:
                raise ValueError("QA binding requires requirement_id")
            if self.section_id or self.source or self.dimension_id:
                raise ValueError("QA binding contains non-QA coordinates")
        elif self.task_kind is EvidenceUnitKind.SUMMARY_SECTION:
            if not self.section_id or not self.source:
                raise ValueError("Summary binding requires section_id and source")
            if self.requirement_id or self.dimension_id:
                raise ValueError("Summary binding contains non-Summary coordinates")
        elif self.task_kind is EvidenceUnitKind.COMPARE_SOURCE_DIMENSION:
            if not self.source or not self.dimension_id:
                raise ValueError("Compare binding requires source and dimension_id")
            if self.requirement_id or self.section_id:
                raise ValueError("Compare binding contains non-Compare coordinates")

    def as_metadata(self) -> dict[str, str]:
        return {
            key: value
            for key, value in {
                "task_kind": self.task_kind.value,
                "requirement_id": self.requirement_id,
                "section_id": self.section_id,
                "source": self.source,
                "dimension_id": self.dimension_id,
            }.items()
            if value
        }


@dataclass(frozen=True, slots=True)
class EvidenceUnitPolicy:
    """Scheduling policy; admission groups are atomic fairness boundaries."""

    admission_group: str
    required: bool = True
    priority: int = 0
    max_retrieval_retries: int = 1

    def __post_init__(self) -> None:
        group = _text(self.admission_group)
        if not _GROUP_ID_PATTERN.fullmatch(group):
            raise ValueError("admission_group must be an opaque eg_<digest> identifier")
        object.__setattr__(self, "admission_group", group)
        if not isinstance(self.required, bool):
            raise ValueError("required must be a boolean")
        _strict_nonnegative_int(self.priority, "priority")
        _strict_nonnegative_int(self.max_retrieval_retries, "max_retrieval_retries")


@dataclass(frozen=True, slots=True)
class EvidenceUnit:
    """One immutable, agent-independent retrieval and verification plan."""

    unit_id: str
    binding: EvidenceTaskBinding
    label: str
    instruction: str
    retrieval_query: str
    recovery_query: str
    scope: EvidenceSourceScope
    policy: EvidenceUnitPolicy

    def __post_init__(self) -> None:
        unit_id = _text(self.unit_id)
        if not _UNIT_ID_PATTERN.fullmatch(unit_id):
            raise ValueError("unit_id must be an opaque eu_<digest> identifier")
        object.__setattr__(self, "unit_id", unit_id)
        if not isinstance(self.binding, EvidenceTaskBinding):
            raise ValueError("binding must be an EvidenceTaskBinding")
        if not isinstance(self.scope, EvidenceSourceScope):
            raise ValueError("scope must be an EvidenceSourceScope")
        if not isinstance(self.policy, EvidenceUnitPolicy):
            raise ValueError("policy must be an EvidenceUnitPolicy")
        for name in ("label", "instruction", "retrieval_query", "recovery_query"):
            object.__setattr__(self, name, _require_text(getattr(self, name), name))

        if self.binding.task_kind in {
            EvidenceUnitKind.SUMMARY_SECTION,
            EvidenceUnitKind.COMPARE_SOURCE_DIMENSION,
        } and self.scope.allowed_sources != (self.binding.source,):
            raise ValueError("source-bound units require an exact one-source scope")

    @property
    def task_kind(self) -> EvidenceUnitKind:
        return self.binding.task_kind

    @property
    def binding_metadata(self) -> dict[str, str]:
        return self.binding.as_metadata()


@dataclass(frozen=True, slots=True)
class EvidenceUnitBudget:
    """Global and per-unit model-visible evidence budgets."""

    max_total_docs: int
    max_total_chars: int
    max_docs_per_unit: int
    max_chars_per_unit: int
    min_docs_per_required_unit: int = 1
    min_chars_per_required_unit: int = 1

    def __post_init__(self) -> None:
        for name in (
            "max_total_docs",
            "max_total_chars",
            "max_docs_per_unit",
            "max_chars_per_unit",
            "min_docs_per_required_unit",
            "min_chars_per_required_unit",
        ):
            _strict_positive_int(getattr(self, name), name)
        if self.max_docs_per_unit > self.max_total_docs:
            raise ValueError("max_docs_per_unit cannot exceed max_total_docs")
        if self.max_chars_per_unit > self.max_total_chars:
            raise ValueError("max_chars_per_unit cannot exceed max_total_chars")
        if self.min_docs_per_required_unit > self.max_docs_per_unit:
            raise ValueError(
                "min_docs_per_required_unit cannot exceed max_docs_per_unit"
            )
        if self.min_chars_per_required_unit > self.max_chars_per_unit:
            raise ValueError(
                "min_chars_per_required_unit cannot exceed max_chars_per_unit"
            )

    def validate_plan_capacity(self, units: Sequence[EvidenceUnit]) -> None:
        _validate_unit_sequence(units)
        required_count = sum(unit.policy.required for unit in units)
        if required_count * self.min_docs_per_required_unit > self.max_total_docs:
            raise ValueError(
                "global document budget cannot reserve every required unit"
            )
        if required_count * self.min_chars_per_required_unit > self.max_total_chars:
            raise ValueError(
                "global character budget cannot reserve every required unit"
            )

    def reserve_plan_capacity(
        self, units: Sequence[EvidenceUnit]
    ) -> EvidenceUnitBudget:
        """Expand aggregate minima just enough to admit every required unit.

        Per-unit limits remain hard caps.  The configured aggregate limits act as
        the normal workload target, but a valid plan with more required units
        must still reserve each unit's declared minimum instead of failing only
        after retrieval starts.
        """

        normalized = _validate_unit_sequence(units)
        required_count = sum(unit.policy.required for unit in normalized)
        reserved = EvidenceUnitBudget(
            max_total_docs=max(
                self.max_total_docs,
                required_count * self.min_docs_per_required_unit,
            ),
            max_total_chars=max(
                self.max_total_chars,
                required_count * self.min_chars_per_required_unit,
            ),
            max_docs_per_unit=self.max_docs_per_unit,
            max_chars_per_unit=self.max_chars_per_unit,
            min_docs_per_required_unit=self.min_docs_per_required_unit,
            min_chars_per_required_unit=self.min_chars_per_required_unit,
        )
        reserved.validate_plan_capacity(normalized)
        return reserved


@dataclass(frozen=True, slots=True)
class EvidenceView:
    """Stable evidence identity used by a generation closure."""

    chunk_id: str
    source: str
    estimated_chars: int
    source_type: EvidenceSourceType = EvidenceSourceType.DOCUMENT
    related_source: str = ""
    parent_chunk_id: str = ""
    span_start: int | None = None
    span_end: int | None = None

    def __post_init__(self) -> None:
        for name in ("chunk_id", "source"):
            object.__setattr__(self, name, _require_text(getattr(self, name), name))
        for name in ("related_source", "parent_chunk_id"):
            object.__setattr__(self, name, _text(getattr(self, name)))
        if not isinstance(self.source_type, EvidenceSourceType):
            raise ValueError("source_type must be an EvidenceSourceType")
        _strict_positive_int(self.estimated_chars, "estimated_chars")
        if (self.span_start is None) != (self.span_end is None):
            raise ValueError("span_start and span_end must be provided together")
        if self.span_start is not None and self.span_end is not None:
            _strict_nonnegative_int(self.span_start, "span_start")
            _strict_nonnegative_int(self.span_end, "span_end")
            if self.span_end <= self.span_start:
                raise ValueError("span_end must be greater than span_start")
        if (
            self.source_type is EvidenceSourceType.DERIVED_KNOWLEDGE
            and not self.related_source
        ):
            raise ValueError("derived knowledge requires related_source")

    @property
    def identity(self) -> tuple[str, int | None, int | None]:
        return self.chunk_id, self.span_start, self.span_end


@dataclass(frozen=True, slots=True)
class EvidenceUnitClosure:
    """The exact, immutable evidence set visible to one downstream generator."""

    unit_id: str
    status: EvidenceClosureStatus
    evidence: tuple[EvidenceView, ...] = ()
    grounding_chunk_ids: tuple[str, ...] = ()
    retrieval_round: int = 0
    reason_code: str = ""

    def __post_init__(self) -> None:
        unit_id = _text(self.unit_id)
        if not _UNIT_ID_PATTERN.fullmatch(unit_id):
            raise ValueError("unit_id must be an opaque eu_<digest> identifier")
        object.__setattr__(self, "unit_id", unit_id)
        if not isinstance(self.status, EvidenceClosureStatus):
            raise ValueError("status must be an EvidenceClosureStatus")
        if not isinstance(self.evidence, tuple) or not all(
            isinstance(item, EvidenceView) for item in self.evidence
        ):
            raise ValueError("evidence must be a tuple of EvidenceView values")
        grounding = _normalized_unique(self.grounding_chunk_ids, "grounding_chunk_ids")
        object.__setattr__(self, "grounding_chunk_ids", grounding)
        object.__setattr__(self, "reason_code", _text(self.reason_code))
        _strict_nonnegative_int(self.retrieval_round, "retrieval_round")

        identities = [item.identity for item in self.evidence]
        if len(identities) != len(set(identities)):
            raise ValueError("evidence closure contains duplicate evidence views")
        available_ids = {item.chunk_id for item in self.evidence}
        if not set(grounding).issubset(available_ids):
            raise ValueError("grounding_chunk_ids must be contained in the closure")

        grounded_statuses = {
            EvidenceClosureStatus.SUPPORTED,
            EvidenceClosureStatus.CONTRADICTORY,
        }
        if self.status in grounded_statuses:
            if not self.evidence or not grounding:
                raise ValueError(
                    f"{self.status.value} closure requires grounded evidence"
                )
        else:
            if self.evidence or grounding:
                raise ValueError(
                    f"{self.status.value} closure must fail closed without evidence"
                )
            if not self.reason_code:
                raise ValueError(f"{self.status.value} closure requires reason_code")

    @property
    def estimated_chars(self) -> int:
        return sum(item.estimated_chars for item in self.evidence)

    def validate_for(self, unit: EvidenceUnit, budget: EvidenceUnitBudget) -> None:
        if self.unit_id != unit.unit_id:
            raise ValueError("closure unit_id does not match its plan")
        if len(self.evidence) > budget.max_docs_per_unit:
            raise ValueError("closure exceeds per-unit document budget")
        if self.estimated_chars > budget.max_chars_per_unit:
            raise ValueError("closure exceeds per-unit character budget")
        for item in self.evidence:
            if not unit.scope.contains(
                source=item.source,
                source_type=item.source_type,
                related_source=item.related_source,
            ):
                raise ValueError(
                    "closure contains evidence outside the unit source scope"
                )


def _validate_unit_sequence(units: Sequence[EvidenceUnit]) -> tuple[EvidenceUnit, ...]:
    if isinstance(units, (str, bytes, bytearray)):
        raise ValueError("units must be a sequence of EvidenceUnit values")
    normalized = tuple(units)
    if not normalized:
        raise ValueError("at least one evidence unit is required")
    if not all(isinstance(unit, EvidenceUnit) for unit in normalized):
        raise ValueError("units must contain only EvidenceUnit values")
    ids = [unit.unit_id for unit in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError("unit_id values must be unique")
    return normalized


def validate_evidence_unit_closures(
    units: Sequence[EvidenceUnit],
    closures: Sequence[EvidenceUnitClosure],
    budget: EvidenceUnitBudget,
) -> None:
    """Validate one complete batch and count model-visible reuse each time."""

    normalized_units = _validate_unit_sequence(units)
    if isinstance(closures, (str, bytes, bytearray)):
        raise ValueError("closures must be a sequence of EvidenceUnitClosure values")
    normalized_closures = tuple(closures)
    if not all(isinstance(item, EvidenceUnitClosure) for item in normalized_closures):
        raise ValueError("closures must contain only EvidenceUnitClosure values")
    closure_by_id = {item.unit_id: item for item in normalized_closures}
    if len(closure_by_id) != len(normalized_closures):
        raise ValueError("closures must not contain duplicate unit_id values")
    unit_by_id = {item.unit_id: item for item in normalized_units}
    if set(closure_by_id) != set(unit_by_id):
        raise ValueError("closures must contain exactly one result for every unit")

    budget.validate_plan_capacity(normalized_units)
    for unit_id, unit in unit_by_id.items():
        closure_by_id[unit_id].validate_for(unit, budget)
    total_docs = sum(len(item.evidence) for item in normalized_closures)
    total_chars = sum(item.estimated_chars for item in normalized_closures)
    if total_docs > budget.max_total_docs:
        raise ValueError("closures exceed global document budget")
    if total_chars > budget.max_total_chars:
        raise ValueError("closures exceed global character budget")


def _unit_id(
    *,
    query: str,
    binding: EvidenceTaskBinding,
) -> str:
    return _opaque_id(
        "eu",
        {
            "query": query,
            "binding": binding.as_metadata(),
        },
    )


def _group_id(*, query: str, task_kind: EvidenceUnitKind, key: str) -> str:
    return _opaque_id(
        "eg",
        {"query": query, "task_kind": task_kind.value, "key": key},
    )


def _mapping_rows(
    rows: Sequence[Mapping[str, Any]], name: str
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(rows, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence of mappings")
    normalized = tuple(rows)
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if not all(isinstance(row, Mapping) for row in normalized):
        raise ValueError(f"{name} must contain only mappings")
    return normalized


def build_qa_evidence_units(
    query: str,
    requirements: Sequence[Mapping[str, Any]],
    *,
    max_retrieval_retries: int = 1,
) -> tuple[EvidenceUnit, ...]:
    """Adapt atomic QA requirements without importing any QA agent."""

    normalized_query = _require_text(query, "query")
    rows = _mapping_rows(requirements, "requirements")
    _strict_nonnegative_int(max_retrieval_retries, "max_retrieval_retries")
    requirement_ids = [
        _require_text(row.get("requirement_id"), "requirement_id") for row in rows
    ]
    if len(requirement_ids) != len({value.casefold() for value in requirement_ids}):
        raise ValueError("requirement_id values must be unique")
    group = _group_id(
        query=normalized_query,
        task_kind=EvidenceUnitKind.QA_REQUIREMENT,
        key="all-requirements",
    )
    units: list[EvidenceUnit] = []
    for position, (row, requirement_id) in enumerate(zip(rows, requirement_ids)):
        question = _require_text(row.get("question") or normalized_query, "question")
        retrieval_query = _require_text(
            row.get("retrieval_query") or question, "retrieval_query"
        )
        recovery_query = _require_text(
            row.get("recovery_query") or retrieval_query, "recovery_query"
        )
        binding = EvidenceTaskBinding(
            task_kind=EvidenceUnitKind.QA_REQUIREMENT,
            requirement_id=requirement_id,
        )
        units.append(
            EvidenceUnit(
                unit_id=_unit_id(query=normalized_query, binding=binding),
                binding=binding,
                label=question,
                instruction=question,
                retrieval_query=retrieval_query,
                recovery_query=recovery_query,
                scope=EvidenceSourceScope(),
                policy=EvidenceUnitPolicy(
                    admission_group=group,
                    priority=position,
                    max_retrieval_retries=max_retrieval_retries,
                ),
            )
        )
    return tuple(units)


def build_summary_evidence_units(
    query: str,
    source: str,
    sections: Sequence[Mapping[str, Any]],
    *,
    max_retrieval_retries: int = 1,
) -> tuple[EvidenceUnit, ...]:
    """Adapt Summary sections into source-scoped semantic obligations."""

    normalized_source = _require_text(source, "source")
    semantic_query = _without_source_names(query, (normalized_source,))
    rows = _mapping_rows(sections, "sections")
    _strict_nonnegative_int(max_retrieval_retries, "max_retrieval_retries")
    section_ids = [_require_text(row.get("section_id"), "section_id") for row in rows]
    if len(section_ids) != len({value.casefold() for value in section_ids}):
        raise ValueError("section_id values must be unique")
    units: list[EvidenceUnit] = []
    for position, (row, section_id) in enumerate(zip(rows, section_ids)):
        title = _require_text(row.get("title") or section_id, "title")
        instruction = _require_text(row.get("instruction") or title, "instruction")
        retrieval_query = _compose_query(
            row.get("retrieval_query") or semantic_query,
            title,
            instruction,
            excluded_sources=(normalized_source,),
        )
        recovery_query = _compose_query(
            row.get("recovery_query") or title,
            instruction,
            excluded_sources=(normalized_source,),
        )
        binding = EvidenceTaskBinding(
            task_kind=EvidenceUnitKind.SUMMARY_SECTION,
            section_id=section_id,
            source=normalized_source,
        )
        units.append(
            EvidenceUnit(
                unit_id=_unit_id(query=semantic_query, binding=binding),
                binding=binding,
                label=title,
                instruction=instruction,
                retrieval_query=retrieval_query,
                recovery_query=recovery_query,
                scope=EvidenceSourceScope(
                    allowed_sources=(normalized_source,),
                    allow_derived_knowledge=False,
                ),
                policy=EvidenceUnitPolicy(
                    admission_group=_group_id(
                        query=semantic_query,
                        task_kind=EvidenceUnitKind.SUMMARY_SECTION,
                        key=section_id,
                    ),
                    priority=position,
                    max_retrieval_retries=max_retrieval_retries,
                ),
            )
        )
    return tuple(units)


def build_compare_evidence_units(
    query: str,
    sources: Sequence[str],
    dimensions: Sequence[Mapping[str, Any]],
    *,
    max_retrieval_retries: int = 1,
) -> tuple[EvidenceUnit, ...]:
    """Build a fair, dimension-major source-by-dimension evidence matrix."""

    normalized_sources = _normalized_unique(sources, "sources")
    if len(normalized_sources) < 2:
        raise ValueError("Compare planning requires at least two sources")
    semantic_query = _without_source_names(query, normalized_sources)
    rows = _mapping_rows(dimensions, "dimensions")
    _strict_nonnegative_int(max_retrieval_retries, "max_retrieval_retries")
    dimension_ids = [
        _require_text(row.get("dimension_id"), "dimension_id") for row in rows
    ]
    if len(dimension_ids) != len({value.casefold() for value in dimension_ids}):
        raise ValueError("dimension_id values must be unique")

    units: list[EvidenceUnit] = []
    for dimension_position, (row, dimension_id) in enumerate(zip(rows, dimension_ids)):
        title = _require_text(row.get("title") or dimension_id, "title")
        instruction = _require_text(row.get("instruction") or title, "instruction")
        retrieval_query = _compose_query(
            row.get("retrieval_query") or semantic_query,
            title,
            instruction,
            excluded_sources=normalized_sources,
        )
        recovery_query = _compose_query(
            row.get("recovery_query") or title,
            instruction,
            excluded_sources=normalized_sources,
        )
        group = _group_id(
            query=semantic_query,
            task_kind=EvidenceUnitKind.COMPARE_SOURCE_DIMENSION,
            key=dimension_id,
        )
        # Dimension-major order lets a scheduler admit or reject the complete
        # cross-source comparison group instead of starving later sources.
        for source in normalized_sources:
            binding = EvidenceTaskBinding(
                task_kind=EvidenceUnitKind.COMPARE_SOURCE_DIMENSION,
                source=source,
                dimension_id=dimension_id,
            )
            units.append(
                EvidenceUnit(
                    unit_id=_unit_id(query=semantic_query, binding=binding),
                    binding=binding,
                    label=title,
                    instruction=instruction,
                    retrieval_query=retrieval_query,
                    recovery_query=recovery_query,
                    scope=EvidenceSourceScope(
                        allowed_sources=(source,),
                        allow_derived_knowledge=False,
                    ),
                    policy=EvidenceUnitPolicy(
                        admission_group=group,
                        priority=dimension_position,
                        max_retrieval_retries=max_retrieval_retries,
                    ),
                )
            )
    return tuple(units)
