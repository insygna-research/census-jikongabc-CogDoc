from __future__ import annotations

import copy
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from cogdoc.graph.state import RetrievedDoc


ANCHOR_PRIORITY = 0
PINNED_PRIORITY = 10
REQUIREMENT_PRIORITY = 15
CONTEXT_PRIORITY = 20
VERIFICATION_PRIORITY = 30
DEFAULT_PRIORITY = 100

DROP_DUPLICATE_CHUNK_ID = "duplicate_chunk_id"
DROP_MAX_DOCS = "max_docs"
DROP_MAX_CHARS = "max_chars"

DEFAULT_MIN_OVERLAP_CHARS = 16

_SOURCE_TEXT_KEY = "_evidence_source_text"
_SOURCE_START_KEY = "_evidence_source_start"
_SOURCE_END_KEY = "_evidence_source_end"
_SOURCE_OVERLAP_KEY = "_evidence_source_overlap_chars"


@dataclass(frozen=True, slots=True)
class EvidencePackCandidate:
    """One possible member of an evidence pack.

    ``priority`` is ordered ascending.  ``anchor_rank`` and ``distance`` are
    secondary stable ordering hints.  Only ``hard_required`` candidates may
    exceed the configured global budgets.
    """

    doc: RetrievedDoc
    priority: int = DEFAULT_PRIORITY
    provenance: str = "candidate"
    hard_required: bool = False
    anchor_rank: int | None = None
    distance: int | None = None
    presentation_order: int | None = None
    matched_requirement_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    chunk_id: str
    source: str
    page: int | None
    parent_chunk_id: str
    input_order: int
    provenance: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PackedEvidence:
    """An immutable selection record with an isolated document snapshot."""

    doc: RetrievedDoc
    ref: EvidenceRef
    display_text: str
    estimated_chars: int
    priority: int
    provenance: tuple[str, ...]
    hard_required: bool
    anchor_rank: int | None
    distance: int | None
    evidence_text_start: int
    evidence_text_end: int
    evidence_trimmed_overlap_chars: int
    matched_requirement_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DroppedEvidence:
    ref: EvidenceRef
    reason: str


@dataclass(frozen=True, slots=True)
class EvidencePack:
    """A deterministic evidence closure shared by all downstream consumers."""

    kept: tuple[PackedEvidence, ...]
    dropped: tuple[DroppedEvidence, ...]
    # Identity-normalized candidates; missing chunk IDs remain distinct.
    input_count: int
    input_estimated_chars: int
    estimated_chars: int
    overlap_removed_chars: int
    over_budget_hard_constraints: bool

    @property
    def kept_docs(self) -> tuple[RetrievedDoc, ...]:
        return tuple(item.doc for item in self.kept)

    @property
    def dropped_refs(self) -> tuple[EvidenceRef, ...]:
        return tuple(item.ref for item in self.dropped)

    def require_within_budget(self) -> EvidencePack:
        """Fail closed when hard evidence alone cannot satisfy the budgets."""

        if self.over_budget_hard_constraints:
            raise EvidencePackBudgetExceeded(
                "hard-required evidence exceeds the global evidence-pack budget"
            )
        return self


class EvidencePackBudgetExceeded(RuntimeError):
    pass


@dataclass(slots=True)
class _NormalizedCandidate:
    doc: RetrievedDoc
    priority: int
    provenance: tuple[str, ...]
    hard_required: bool
    anchor_rank: int | None
    distance: int | None
    presentation_order: int | None
    matched_requirement_ids: tuple[str, ...]
    input_order: int


def _meta(doc: Mapping[str, Any]) -> Mapping[str, Any]:
    value = doc.get("meta")
    return value if isinstance(value, Mapping) else {}


def _retrieval(doc: Mapping[str, Any]) -> Mapping[str, Any]:
    value = doc.get("retrieval")
    return value if isinstance(value, Mapping) else {}


def _string_meta(doc: Mapping[str, Any], key: str) -> str:
    return str(_meta(doc).get(key) or "").strip()


def _chunk_id(doc: Mapping[str, Any]) -> str:
    return _string_meta(doc, "chunk_id")


def _non_negative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return None
    return normalized if normalized >= 0 else None


def _order_value(doc: Mapping[str, Any]) -> int | None:
    meta = _meta(doc)
    child_order = _non_negative_int(meta.get("child_index_in_parent"))
    if child_order is not None:
        return child_order
    return _non_negative_int(meta.get("chunk_index"))


def _page(doc: Mapping[str, Any]) -> int | None:
    return _non_negative_int(_meta(doc).get("page"))


def _validate_order_hint(name: str, value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or None")


def _validate_candidate(candidate: EvidencePackCandidate) -> None:
    if isinstance(candidate.priority, bool) or not isinstance(candidate.priority, int):
        raise ValueError("priority must be an integer")
    _validate_order_hint("anchor_rank", candidate.anchor_rank)
    _validate_order_hint("distance", candidate.distance)
    _validate_order_hint("presentation_order", candidate.presentation_order)
    if not isinstance(candidate.provenance, str):
        raise ValueError("provenance must be a string")
    if isinstance(candidate.matched_requirement_ids, (str, bytes)) or not isinstance(
        candidate.matched_requirement_ids, Sequence
    ):
        raise ValueError("matched_requirement_ids must be a sequence of strings")


def _ref(
    doc: Mapping[str, Any], input_order: int, provenance: tuple[str, ...]
) -> EvidenceRef:
    return EvidenceRef(
        chunk_id=_chunk_id(doc),
        source=_string_meta(doc, "source"),
        page=_page(doc),
        parent_chunk_id=_string_meta(doc, "parent_chunk_id"),
        input_order=input_order,
        provenance=provenance,
    )


def _stable_provenance(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = value.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return tuple(ordered)


def _stable_requirement_ids(values: Sequence[Any]) -> tuple[str, ...]:
    return _stable_provenance([str(value) for value in values])


def _selection_key(candidate: _NormalizedCandidate) -> tuple[int, int, int, int, int]:
    sentinel = 2**63 - 1
    return (
        0 if candidate.hard_required else 1,
        candidate.priority,
        candidate.anchor_rank if candidate.anchor_rank is not None else sentinel,
        candidate.distance if candidate.distance is not None else sentinel,
        candidate.input_order,
    )


def _presentation_key(
    candidate: _NormalizedCandidate,
) -> tuple[int, int, int, int, int, int]:
    sentinel = 2**63 - 1
    if candidate.presentation_order is not None:
        return (0, candidate.presentation_order, 0, 0, 0, candidate.input_order)
    if candidate.anchor_rank is not None:
        order = _order_value(candidate.doc)
        return (
            1,
            candidate.anchor_rank,
            order if order is not None else sentinel,
            candidate.priority,
            candidate.distance if candidate.distance is not None else sentinel,
            candidate.input_order,
        )
    selection = _selection_key(candidate)
    return (2, selection[1], selection[2], selection[3], selection[4], 0)


def _normalize_candidates(
    candidates: Sequence[EvidencePackCandidate],
) -> tuple[list[_NormalizedCandidate], list[DroppedEvidence]]:
    grouped: dict[str, list[tuple[int, EvidencePackCandidate]]] = {}
    missing_id: list[tuple[int, EvidencePackCandidate]] = []
    for input_order, candidate in enumerate(candidates):
        _validate_candidate(candidate)
        chunk_id = _chunk_id(candidate.doc)
        if chunk_id:
            grouped.setdefault(chunk_id, []).append((input_order, candidate))
        else:
            # Identity-less rows are deliberately never content/meta deduplicated.
            missing_id.append((input_order, candidate))

    normalized: list[_NormalizedCandidate] = []
    dropped: list[DroppedEvidence] = []
    for members in grouped.values():
        provisional = [
            _NormalizedCandidate(
                doc=candidate.doc,
                priority=candidate.priority,
                provenance=(candidate.provenance,),
                hard_required=candidate.hard_required,
                anchor_rank=candidate.anchor_rank,
                distance=candidate.distance,
                presentation_order=candidate.presentation_order,
                matched_requirement_ids=_stable_requirement_ids(
                    candidate.matched_requirement_ids
                ),
                input_order=input_order,
            )
            for input_order, candidate in members
        ]
        representative = min(provisional, key=_selection_key)
        provenance = _stable_provenance(
            [candidate.provenance for _, candidate in members]
        )
        normalized.append(
            _NormalizedCandidate(
                doc=representative.doc,
                priority=representative.priority,
                provenance=provenance,
                hard_required=any(candidate.hard_required for _, candidate in members),
                anchor_rank=representative.anchor_rank,
                distance=representative.distance,
                presentation_order=next(
                    (
                        candidate.presentation_order
                        for candidate in provisional
                        if candidate.presentation_order is not None
                    ),
                    None,
                ),
                matched_requirement_ids=_stable_requirement_ids(
                    [
                        requirement_id
                        for candidate in provisional
                        for requirement_id in candidate.matched_requirement_ids
                    ]
                ),
                input_order=representative.input_order,
            )
        )
        for input_order, candidate in members:
            if input_order == representative.input_order:
                continue
            dropped.append(
                DroppedEvidence(
                    ref=_ref(candidate.doc, input_order, (candidate.provenance,)),
                    reason=DROP_DUPLICATE_CHUNK_ID,
                )
            )

    for input_order, candidate in missing_id:
        normalized.append(
            _NormalizedCandidate(
                doc=candidate.doc,
                priority=candidate.priority,
                provenance=_stable_provenance((candidate.provenance,)),
                hard_required=candidate.hard_required,
                anchor_rank=candidate.anchor_rank,
                distance=candidate.distance,
                presentation_order=candidate.presentation_order,
                matched_requirement_ids=_stable_requirement_ids(
                    candidate.matched_requirement_ids
                ),
                input_order=input_order,
            )
        )
    return normalized, dropped


def _overlap_group(doc: Mapping[str, Any]) -> tuple[str, str] | None:
    source = _string_meta(doc, "source")
    parent_chunk_id = _string_meta(doc, "parent_chunk_id")
    if not source or not parent_chunk_id:
        return None
    return source, parent_chunk_id


def _longest_suffix_prefix_overlap(previous: str, current: str, *, minimum: int) -> int:
    upper = min(len(previous), len(current))
    for size in range(upper, minimum - 1, -1):
        if previous.endswith(current[:size]):
            return size
    return 0


def _source_text_range(doc: Mapping[str, Any]) -> tuple[str, int, int, int]:
    """Return the fullest locally available text and its ultimate source range."""

    hidden_text = doc.get(_SOURCE_TEXT_KEY)
    if isinstance(hidden_text, str):
        has_hidden_source = True
        text = hidden_text
    else:
        has_hidden_source = False
        text = str(doc.get("text") or "")
    retrieval = _retrieval(doc)
    start = _non_negative_int(
        doc.get(_SOURCE_START_KEY)
        if has_hidden_source
        else retrieval.get("evidence_text_start")
    )
    start = start or 0
    end = _non_negative_int(
        doc.get(_SOURCE_END_KEY)
        if has_hidden_source
        else retrieval.get("evidence_text_end")
    )
    if end is None or end < start + len(text):
        end = start + len(text)
    overlap = _non_negative_int(
        doc.get(_SOURCE_OVERLAP_KEY)
        if has_hidden_source
        else retrieval.get("evidence_trimmed_overlap_chars")
    )
    if overlap is None:
        overlap = start
    return text, start, end, overlap


def _display_texts(
    candidates: Sequence[_NormalizedCandidate], *, min_overlap_chars: int
) -> tuple[list[str], list[int]]:
    texts = [_source_text_range(candidate.doc)[0] for candidate in candidates]
    children: dict[tuple[str, str], dict[int, int] | None] = {}
    for index, candidate in enumerate(candidates):
        group = _overlap_group(candidate.doc)
        order = _order_value(candidate.doc)
        if group is None or order is None:
            continue
        existing = children.setdefault(group, {})
        if existing is None:
            continue
        if order in existing:
            # Ambiguous structural order is not safe to trim.
            children[group] = None
            continue
        existing[order] = index

    display = list(texts)
    starts = [0] * len(texts)
    for group, order_to_index in children.items():
        if order_to_index is None:
            continue
        for order, current_index in order_to_index.items():
            previous_index = order_to_index.get(order - 1)
            if previous_index is None:
                continue
            overlap = _longest_suffix_prefix_overlap(
                texts[previous_index],
                texts[current_index],
                minimum=min_overlap_chars,
            )
            if overlap:
                display[current_index] = texts[current_index][overlap:]
                starts[current_index] = overlap
    return display, starts


def _locator_chars(doc: Mapping[str, Any]) -> int:
    meta = _meta(doc)
    context = str(meta.get("context") or "").strip()
    section_path = str(meta.get("section_path") or "").strip()
    total = 0
    if context:
        total += len(context) + len("定位上下文：\n\n正文：\n")
    if section_path:
        total += len(section_path) + len("章节路径：\n\n")
    return total


def _estimated_chars(
    candidates: Sequence[_NormalizedCandidate],
    *,
    min_overlap_chars: int,
    document_char_cost: Callable[[Mapping[str, Any], str], int] | None,
    separator_chars: int,
) -> tuple[int, list[str], list[int], list[int]]:
    display, starts = _display_texts(candidates, min_overlap_chars=min_overlap_chars)
    per_item: list[int] = []
    for index, (candidate, text) in enumerate(zip(candidates, display, strict=True)):
        item_chars = (
            document_char_cost(candidate.doc, text)
            if document_char_cost is not None
            else len(text) + _locator_chars(candidate.doc)
        )
        if index:
            item_chars += separator_chars
        per_item.append(item_chars)
    return sum(per_item), display, starts, per_item


def _materialize_doc(
    doc: RetrievedDoc,
    display_text: str,
    *,
    evidence_text_start: int,
    matched_requirement_ids: tuple[str, ...],
) -> tuple[RetrievedDoc, int, int, int]:
    snapshot = cast(dict[str, Any], copy.deepcopy(doc))
    source_text, source_start, source_end, source_overlap = _source_text_range(doc)
    snapshot["text"] = display_text
    snapshot[_SOURCE_TEXT_KEY] = source_text
    snapshot[_SOURCE_START_KEY] = source_start
    snapshot[_SOURCE_END_KEY] = source_end
    snapshot[_SOURCE_OVERLAP_KEY] = source_overlap
    raw_retrieval = snapshot.get("retrieval")
    retrieval = dict(raw_retrieval) if isinstance(raw_retrieval, Mapping) else {}
    composed_start = source_start + evidence_text_start
    composed_overlap = source_overlap + evidence_text_start
    retrieval.update(
        {
            "evidence_text_start": composed_start,
            "evidence_text_end": source_end,
            "evidence_trimmed_overlap_chars": composed_overlap,
        }
    )
    if matched_requirement_ids:
        retrieval["matched_requirement_ids"] = list(matched_requirement_ids)
    snapshot["retrieval"] = retrieval
    return cast(RetrievedDoc, snapshot), composed_start, source_end, composed_overlap


def build_evidence_pack(
    candidates: Sequence[EvidencePackCandidate],
    *,
    max_docs: int,
    max_chars: int,
    requirement_ids: Sequence[str] = (),
    min_overlap_chars: int = DEFAULT_MIN_OVERLAP_CHARS,
    document_char_cost: Callable[[Mapping[str, Any], str], int] | None = None,
    separator_chars: int = 0,
    document_transform: (
        Callable[[RetrievedDoc, tuple[str, ...]], RetrievedDoc] | None
    ) = None,
) -> EvidencePack:
    """Build a deterministic, globally budgeted evidence closure.

    Anchor/pinned callers mark rows as ``hard_required``.  Such rows are never
    silently discarded, even when their count or text alone exceeds a budget.
    Normal rows are greedily selected by priority, anchor rank, distance, and
    original input order.  Exact child overlap is removed only from an isolated
    packed snapshot: input text is untouched, while packed ``doc["text"]`` is
    the exact display text charged to the global character budget.
    """

    if isinstance(max_docs, bool) or not isinstance(max_docs, int) or max_docs < 0:
        raise ValueError("max_docs must be a non-negative integer")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 0:
        raise ValueError("max_chars must be a non-negative integer")
    if (
        isinstance(min_overlap_chars, bool)
        or not isinstance(min_overlap_chars, int)
        or min_overlap_chars < 1
    ):
        raise ValueError("min_overlap_chars must be a positive integer")
    if isinstance(requirement_ids, (str, bytes)):
        raise ValueError("requirement_ids must be a sequence of strings")
    if (
        isinstance(separator_chars, bool)
        or not isinstance(separator_chars, int)
        or separator_chars < 0
    ):
        raise ValueError("separator_chars must be a non-negative integer")

    normalized, dropped = _normalize_candidates(candidates)
    if document_transform is not None:
        for candidate in normalized:
            original_chunk_id = _chunk_id(candidate.doc)
            canonical_doc = cast(dict[str, Any], copy.deepcopy(candidate.doc))
            raw_retrieval = canonical_doc.get("retrieval")
            canonical_retrieval = (
                dict(raw_retrieval) if isinstance(raw_retrieval, Mapping) else {}
            )
            if candidate.matched_requirement_ids:
                canonical_retrieval["matched_requirement_ids"] = list(
                    candidate.matched_requirement_ids
                )
            else:
                canonical_retrieval.pop("matched_requirement_ids", None)
            canonical_doc["retrieval"] = canonical_retrieval
            transformed = document_transform(
                cast(RetrievedDoc, canonical_doc), candidate.matched_requirement_ids
            )
            if _chunk_id(transformed) != original_chunk_id:
                raise ValueError("document_transform must preserve chunk_id")
            candidate.doc = transformed
            candidate.matched_requirement_ids = _document_requirement_ids(transformed)
    input_presentation = sorted(normalized, key=_presentation_key)
    input_estimated_chars, _, _, _ = _estimated_chars(
        input_presentation,
        min_overlap_chars=min_overlap_chars,
        document_char_cost=document_char_cost,
        separator_chars=separator_chars,
    )
    ordered = sorted(normalized, key=_selection_key)
    selected = [candidate for candidate in ordered if candidate.hard_required]
    remaining = [candidate for candidate in ordered if not candidate.hard_required]
    target_requirement_ids = _stable_requirement_ids(requirement_ids)
    covered_requirement_ids = {
        requirement_id
        for candidate in selected
        for requirement_id in candidate.matched_requirement_ids
    }

    while remaining:
        candidate_index = 0
        uncovered_requirement_ids = (
            set(target_requirement_ids) - covered_requirement_ids
        )
        coverage_counts = [
            len(
                uncovered_requirement_ids.intersection(
                    candidate.matched_requirement_ids
                )
            )
            for candidate in remaining
        ]
        if coverage_counts and max(coverage_counts) > 0:
            # Spend a tight document/character slot on the candidate that closes
            # the most remaining requirements; stable selection order breaks ties.
            candidate_index = max(
                range(len(remaining)),
                key=lambda index: (coverage_counts[index], -index),
            )
        candidate = remaining.pop(candidate_index)
        if len(selected) >= max_docs:
            dropped.append(
                DroppedEvidence(
                    ref=_ref(
                        candidate.doc, candidate.input_order, candidate.provenance
                    ),
                    reason=DROP_MAX_DOCS,
                )
            )
            continue
        prospective_chars, _, _, _ = _estimated_chars(
            [*selected, candidate],
            min_overlap_chars=min_overlap_chars,
            document_char_cost=document_char_cost,
            separator_chars=separator_chars,
        )
        if prospective_chars > max_chars:
            dropped.append(
                DroppedEvidence(
                    ref=_ref(
                        candidate.doc, candidate.input_order, candidate.provenance
                    ),
                    reason=DROP_MAX_CHARS,
                )
            )
            continue
        selected.append(candidate)
        covered_requirement_ids.update(candidate.matched_requirement_ids)

    presented = sorted(selected, key=_presentation_key)
    estimated_chars, display_texts, text_starts, per_item_chars = _estimated_chars(
        presented,
        min_overlap_chars=min_overlap_chars,
        document_char_cost=document_char_cost,
        separator_chars=separator_chars,
    )
    packed_items: list[PackedEvidence] = []
    for candidate, display_text, text_start, item_chars in zip(
        presented, display_texts, text_starts, per_item_chars, strict=True
    ):
        snapshot, composed_start, composed_end, composed_overlap = _materialize_doc(
            candidate.doc,
            display_text,
            evidence_text_start=text_start,
            matched_requirement_ids=candidate.matched_requirement_ids,
        )
        packed_items.append(
            PackedEvidence(
                doc=snapshot,
                ref=_ref(candidate.doc, candidate.input_order, candidate.provenance),
                display_text=display_text,
                estimated_chars=item_chars,
                priority=candidate.priority,
                provenance=candidate.provenance,
                hard_required=candidate.hard_required,
                anchor_rank=candidate.anchor_rank,
                distance=candidate.distance,
                evidence_text_start=composed_start,
                evidence_text_end=composed_end,
                evidence_trimmed_overlap_chars=composed_overlap,
                matched_requirement_ids=candidate.matched_requirement_ids,
            )
        )
    kept = tuple(packed_items)
    dropped.sort(key=lambda item: item.ref.input_order)
    return EvidencePack(
        kept=kept,
        dropped=tuple(dropped),
        input_count=len(normalized),
        input_estimated_chars=input_estimated_chars,
        estimated_chars=estimated_chars,
        overlap_removed_chars=sum(text_starts),
        over_budget_hard_constraints=(
            len(presented) > max_docs or estimated_chars > max_chars
        ),
    )


def _anchor_association(
    doc: Mapping[str, Any],
    anchors: Sequence[RetrievedDoc],
    anchor_by_id: Mapping[str, tuple[int, RetrievedDoc]],
) -> tuple[int | None, int | None]:
    doc_chunk_id = _chunk_id(doc)
    association = anchor_by_id.get(doc_chunk_id)
    if association is None:
        context_anchor_id = str(
            _retrieval(doc).get("context_anchor_chunk_id") or ""
        ).strip()
        association = anchor_by_id.get(context_anchor_id)
    if association is None:
        source = _string_meta(doc, "source")
        parent = _string_meta(doc, "parent_chunk_id")
        if source and parent:
            association = next(
                (
                    (rank, anchor)
                    for rank, anchor in enumerate(anchors)
                    if _string_meta(anchor, "source") == source
                    and _string_meta(anchor, "parent_chunk_id") == parent
                ),
                None,
            )
    if association is None:
        return None, None
    rank, anchor = association
    doc_order = _order_value(doc)
    anchor_order = _order_value(anchor)
    distance = (
        abs(doc_order - anchor_order)
        if doc_order is not None and anchor_order is not None
        else None
    )
    return rank, distance


def _document_requirement_ids(doc: Mapping[str, Any]) -> tuple[str, ...]:
    raw = _retrieval(doc).get("matched_requirement_ids")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return _stable_requirement_ids(raw)
    return ()


def evidence_candidates_from_sources(
    *,
    anchors: Sequence[RetrievedDoc],
    expanded_docs: Sequence[RetrievedDoc] = (),
    verification_candidates: Sequence[RetrievedDoc] = (),
    pinned_chunk_ids: Collection[str] = (),
) -> tuple[EvidencePackCandidate, ...]:
    """Translate current retrieval stages into the generic candidate model."""

    pinned_ids = {
        str(value).strip() for value in pinned_chunk_ids if str(value).strip()
    }
    expanded_position_by_id: dict[str, int] = {}
    for stage_order, doc in enumerate(expanded_docs):
        if chunk_id := _chunk_id(doc):
            expanded_position_by_id.setdefault(chunk_id, stage_order)
    anchor_by_id = {
        chunk_id: (rank, doc)
        for rank, doc in enumerate(anchors)
        if (chunk_id := _chunk_id(doc))
    }
    output: list[EvidencePackCandidate] = []

    for rank, doc in enumerate(anchors):
        matched_requirement_ids = _document_requirement_ids(doc)
        output.append(
            EvidencePackCandidate(
                doc=doc,
                priority=ANCHOR_PRIORITY,
                provenance="anchor",
                hard_required=True,
                anchor_rank=rank,
                distance=0,
                presentation_order=expanded_position_by_id.get(_chunk_id(doc)),
                matched_requirement_ids=matched_requirement_ids,
            )
        )

    def append_docs(
        docs: Sequence[RetrievedDoc],
        *,
        priority: int,
        provenance: str,
        presentation_offset: int,
    ) -> None:
        for stage_order, doc in enumerate(docs):
            chunk_id = _chunk_id(doc)
            anchor_rank, distance = _anchor_association(doc, anchors, anchor_by_id)
            is_pinned = bool(chunk_id and chunk_id in pinned_ids)
            matched_requirement_ids = _document_requirement_ids(doc)
            candidate_priority = priority
            if matched_requirement_ids:
                candidate_priority = min(candidate_priority, REQUIREMENT_PRIORITY)
            output.append(
                EvidencePackCandidate(
                    doc=doc,
                    priority=PINNED_PRIORITY if is_pinned else candidate_priority,
                    provenance=provenance,
                    hard_required=is_pinned,
                    anchor_rank=anchor_rank,
                    distance=distance,
                    presentation_order=presentation_offset + stage_order,
                    matched_requirement_ids=matched_requirement_ids,
                )
            )

    append_docs(
        expanded_docs,
        priority=CONTEXT_PRIORITY,
        provenance="expanded",
        presentation_offset=0,
    )
    append_docs(
        verification_candidates,
        priority=VERIFICATION_PRIORITY,
        provenance="verification",
        presentation_offset=len(expanded_docs),
    )
    return tuple(output)


def build_evidence_pack_from_sources(
    *,
    anchors: Sequence[RetrievedDoc],
    expanded_docs: Sequence[RetrievedDoc] = (),
    verification_candidates: Sequence[RetrievedDoc] = (),
    pinned_chunk_ids: Collection[str] = (),
    max_docs: int,
    max_chars: int,
    requirement_ids: Sequence[str] = (),
    min_overlap_chars: int = DEFAULT_MIN_OVERLAP_CHARS,
    document_char_cost: Callable[[Mapping[str, Any], str], int] | None = None,
    separator_chars: int = 0,
    document_transform: (
        Callable[[RetrievedDoc, tuple[str, ...]], RetrievedDoc] | None
    ) = None,
) -> EvidencePack:
    """Convenience entry point for retrieval pipelines and offline evaluation."""

    return build_evidence_pack(
        evidence_candidates_from_sources(
            anchors=anchors,
            expanded_docs=expanded_docs,
            verification_candidates=verification_candidates,
            pinned_chunk_ids=pinned_chunk_ids,
        ),
        max_docs=max_docs,
        max_chars=max_chars,
        requirement_ids=requirement_ids,
        min_overlap_chars=min_overlap_chars,
        document_char_cost=document_char_cost,
        separator_chars=separator_chars,
        document_transform=document_transform,
    )
