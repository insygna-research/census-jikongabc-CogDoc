from __future__ import annotations

import copy
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from cogdoc.graph.state import RetrievedDoc
from cogdoc.tools.tokenizer import tokenize_mixed_text


logger = logging.getLogger(__name__)

REASON_WITHIN_BUDGET = "within_budget"
REASON_QUERY_SPAN = "query_span"
REASON_LONG_SENTENCE_WINDOW = "long_sentence_window"
REASON_FALLBACK_NO_TERMS = "fallback_no_terms"
REASON_FALLBACK_NO_MATCH = "fallback_no_match"

_PACK_SOURCE_TEXT_KEY = "_evidence_source_text"
_PACK_SOURCE_START_KEY = "_evidence_source_start"
_PACK_SOURCE_END_KEY = "_evidence_source_end"
_PACK_SOURCE_OVERLAP_KEY = "_evidence_source_overlap_chars"
_PACK_SOURCE_KEYS = (
    _PACK_SOURCE_TEXT_KEY,
    _PACK_SOURCE_START_KEY,
    _PACK_SOURCE_END_KEY,
    _PACK_SOURCE_OVERLAP_KEY,
)

# Evidence Pack deliberately does not know these keys.  They let a later
# adaptive round select a different span from the locally available source,
# without allowing a pack/repack operation to restore text outside the span.
_SPAN_SOURCE_TEXT_KEY = "_evidence_span_source_text"
_SPAN_SOURCE_START_KEY = "_evidence_span_source_start"
_SPAN_SOURCE_END_KEY = "_evidence_span_source_end"
_SPAN_SOURCE_OVERLAP_KEY = "_evidence_span_source_overlap_chars"

_REQUIREMENT_TEXT_KEYS = (
    "question",
    "retrieval_query",
    "recovery_query",
    "text",
)
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[_.-][A-Za-z0-9]+)*")
_CJK_CHAR_RE = re.compile(r"[\u3400-\u9fff]")
_CLOSING_PUNCTUATION = frozenset("\"'\u2019\u201d)\uff09]\u3011}\u3009\u300d\u300f")
_UNCONDITIONAL_SENTENCE_END = frozenset("\u3002\uff01\uff1f!?\uff1b;")


@dataclass(frozen=True, slots=True)
class EvidenceSpanBatch:
    """Isolated, query-aware document views plus aggregate compression data."""

    docs: tuple[RetrievedDoc, ...]
    input_count: int
    compressed_count: int
    fallback_count: int
    input_chars: int
    selected_chars: int
    reason_counts: dict[str, int]

    @property
    def output_count(self) -> int:
        return len(self.docs)


@dataclass(frozen=True, slots=True)
class _SourceView:
    text: str
    start: int
    end: int
    overlap_chars: int


@dataclass(frozen=True, slots=True)
class _TextSpan:
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _RequirementTerms:
    requirement_id: str
    terms: tuple[str, ...]
    match_terms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Selection:
    start: int
    end: int
    score: float
    matched_terms: tuple[str, ...]
    reason: str
    fallback: bool = False


def _non_negative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return None
    return normalized if normalized >= 0 else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _source_view(doc: Mapping[str, Any]) -> _SourceView:
    """Return the fullest locally available view and its ultimate offsets.

    Span-private text wins across adaptive rounds.  Pack-private text is only
    promoted when receiving an older packed document for the first time.  It
    is removed from every output snapshot, so Evidence Pack cannot restore the
    full text after this selector has established the evidence boundary.
    """

    span_text = doc.get(_SPAN_SOURCE_TEXT_KEY)
    pack_text = doc.get(_PACK_SOURCE_TEXT_KEY)
    retrieval = _mapping(doc.get("retrieval"))
    if isinstance(span_text, str):
        text = span_text
        start = _non_negative_int(doc.get(_SPAN_SOURCE_START_KEY)) or 0
        stored_end = _non_negative_int(doc.get(_SPAN_SOURCE_END_KEY))
        overlap = _non_negative_int(doc.get(_SPAN_SOURCE_OVERLAP_KEY)) or 0
    elif isinstance(pack_text, str):
        text = pack_text
        start = _non_negative_int(doc.get(_PACK_SOURCE_START_KEY)) or 0
        stored_end = _non_negative_int(doc.get(_PACK_SOURCE_END_KEY))
        overlap = _non_negative_int(doc.get(_PACK_SOURCE_OVERLAP_KEY)) or 0
    else:
        text = str(doc.get("text") or "")
        start = _non_negative_int(retrieval.get("evidence_text_start")) or 0
        stored_end = _non_negative_int(retrieval.get("evidence_text_end"))
        overlap = (
            _non_negative_int(retrieval.get("evidence_trimmed_overlap_chars")) or 0
        )
    minimum_end = start + len(text)
    if stored_end is not None and stored_end != minimum_end:
        meta = _mapping(doc.get("meta"))
        logger.warning(
            "evidence_span_source_end_mismatch chunk_id=%s stored_end=%s "
            "text_derived_end=%s",
            str(meta.get("chunk_id") or ""),
            stored_end,
            minimum_end,
        )
    end = stored_end if stored_end == minimum_end else minimum_end
    return _SourceView(text=text, start=start, end=end, overlap_chars=overlap)


def _stable_tokens(text: str) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for token in tokenize_mixed_text(text):
        normalized = str(token).strip().casefold()
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return tuple(ordered)


def _requirement_text(requirement: Mapping[str, Any] | str) -> str:
    if isinstance(requirement, str):
        return requirement
    return " ".join(
        str(requirement.get(key) or "").strip()
        for key in _REQUIREMENT_TEXT_KEYS
        if str(requirement.get(key) or "").strip()
    )


def _fallback_entity_terms(text: str) -> tuple[str, ...]:
    """Keep exact single-character entities discarded by the corpus tokenizer.

    These terms are only used when two or more requirements have no distinctive
    word-level token.  Cross-requirement uniqueness removes shared function
    characters, while retaining labels such as ``A``/``B`` or ``甲``/``乙``.
    """

    seen: set[str] = set()
    ordered: list[str] = []
    for match in _WORD_RE.finditer(text):
        value = match.group(0).casefold()
        if len(value) == 1 and value not in seen:
            seen.add(value)
            ordered.append(value)
    for value in _CJK_CHAR_RE.findall(text):
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)


def _term_plan(
    query: str,
    evidence_requirements: Sequence[Mapping[str, Any] | str],
) -> tuple[tuple[str, ...], tuple[_RequirementTerms, ...]]:
    query_terms = _stable_tokens(query)
    raw_requirements: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
    for requirement in evidence_requirements:
        text = _requirement_text(requirement)
        terms = _stable_tokens(text)
        entity_terms = _fallback_entity_terms(text)
        if not terms and not entity_terms:
            continue
        requirement_id = (
            str(requirement.get("requirement_id") or "").strip()
            if isinstance(requirement, Mapping)
            else ""
        )
        raw_requirements.append((requirement_id, terms, entity_terms))
    term_requirement_counts: dict[str, int] = {}
    entity_requirement_counts: dict[str, int] = {}
    for _, terms, entity_terms in raw_requirements:
        for term in set(terms):
            term_requirement_counts[term] = term_requirement_counts.get(term, 0) + 1
        for term in set(entity_terms):
            entity_requirement_counts[term] = entity_requirement_counts.get(term, 0) + 1
    requirements = tuple(
        _RequirementTerms(
            requirement_id=requirement_id,
            terms=terms,
            match_terms=(
                distinctive
                if (
                    distinctive := tuple(
                        term
                        for term in terms
                        if term_requirement_counts.get(term, 0) == 1
                    )
                )
                else tuple(
                    term
                    for term in entity_terms
                    if entity_requirement_counts.get(term, 0) == 1
                )
            ),
        )
        for requirement_id, terms, entity_terms in raw_requirements
    )
    return query_terms, requirements


def _target_terms(
    query_terms: tuple[str, ...], requirements: Sequence[_RequirementTerms]
) -> tuple[str, ...]:
    ordered = list(query_terms)
    seen = set(ordered)
    for requirement in requirements:
        for term in (*requirement.match_terms, *requirement.terms):
            if term not in seen:
                seen.add(term)
                ordered.append(term)
    return tuple(ordered)


def _stable_ids(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return ()
    seen: set[str] = set()
    ordered: list[str] = []
    for item in value:
        normalized = str(item).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return tuple(ordered)


def _active_requirements(
    doc: Mapping[str, Any],
    requirements: tuple[_RequirementTerms, ...],
    matched_requirement_ids: Sequence[str] | None,
) -> tuple[tuple[_RequirementTerms, ...], tuple[str, ...] | None]:
    if matched_requirement_ids is not None:
        selected_ids = _stable_ids(matched_requirement_ids)
        has_attribution = True
    else:
        retrieval = _mapping(doc.get("retrieval"))
        has_attribution = "matched_requirement_ids" in retrieval
        selected_ids = _stable_ids(retrieval.get("matched_requirement_ids"))
    if not has_attribution:
        return requirements, None
    selected = set(selected_ids)
    # Requirements without a stable ID cannot participate in ID attribution,
    # but remain useful for callers that supplied free-form requirement strings.
    return (
        tuple(
            requirement
            for requirement in requirements
            if not requirement.requirement_id or requirement.requirement_id in selected
        ),
        selected_ids,
    )


def _is_sentence_period(text: str, index: int) -> bool:
    if text[index] != ".":
        return False
    previous = text[index - 1] if index else ""
    following = text[index + 1] if index + 1 < len(text) else ""
    if previous.isdigit() and following.isdigit():
        return False
    return not following or following.isspace() or following in _CLOSING_PUNCTUATION


def _trimmed_span(text: str, start: int, end: int) -> _TextSpan | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return _TextSpan(start, end) if start < end else None


def _sentence_spans(text: str) -> tuple[_TextSpan, ...]:
    spans: list[_TextSpan] = []
    start = 0
    index = 0
    while index < len(text):
        char = text[index]
        boundary = (
            char == "\n"
            or char in _UNCONDITIONAL_SENTENCE_END
            or _is_sentence_period(text, index)
        )
        if not boundary:
            index += 1
            continue
        end = index if char == "\n" else index + 1
        if char != "\n":
            while end < len(text) and text[end] in _CLOSING_PUNCTUATION:
                end += 1
        span = _trimmed_span(text, start, end)
        if span is not None:
            spans.append(span)
        index = max(index + 1, end)
        start = index
    span = _trimmed_span(text, start, len(text))
    if span is not None:
        spans.append(span)
    return tuple(spans)


def _quality(
    tokens: set[str],
    *,
    query_terms: tuple[str, ...],
    requirement_terms: tuple[tuple[str, ...], ...],
    all_terms: tuple[str, ...],
) -> tuple[int, int, int]:
    requirement_hits = sum(
        bool(tokens.intersection(terms)) for terms in requirement_terms
    )
    query_hits = len(tokens.intersection(query_terms))
    distinct_hits = len(tokens.intersection(all_terms))
    return requirement_hits, query_hits, distinct_hits


def _score(quality: tuple[int, int, int]) -> float:
    requirement_hits, query_hits, distinct_hits = quality
    return float(requirement_hits * 100 + query_hits * 10 + distinct_hits)


def _matched_terms(tokens: set[str], all_terms: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(term for term in all_terms if term in tokens)


def _matching_intervals(text: str, terms: set[str]) -> tuple[_TextSpan, ...]:
    intervals: set[tuple[int, int]] = set()
    lowered = text.casefold()
    for term in terms:
        # Chinese terms and identifiers generally survive tokenization verbatim.
        # Pure Latin terms may be stems (for example ``retriev``), so matching
        # them as arbitrary substrings would turn ``art`` into a hit on
        # ``partial``.  Lexical units below handle those terms safely.
        if term.isascii() and term.isalpha():
            continue
        search_from = 0
        while term and (found := lowered.find(term, search_from)) >= 0:
            intervals.add((found, found + len(term)))
            search_from = found + max(1, len(term))
    # English terms may be stemmed.  Map the normalized token back to the exact
    # lexical unit instead of attempting to synthesize a substring.
    for match in _WORD_RE.finditer(text):
        lexical_terms = set(_stable_tokens(match.group(0)))
        lexical_terms.add(match.group(0).casefold())
        if terms.intersection(lexical_terms):
            intervals.add((match.start(), match.end()))
    return tuple(_TextSpan(start, end) for start, end in sorted(intervals))


def _scoring_tokens(text: str, target_terms: tuple[str, ...]) -> set[str]:
    """Include exact lexical hits that the corpus tokenizer punctuates."""

    tokens = set(_stable_tokens(text))
    lowered = text.casefold()
    target = set(target_terms)
    tokens.update(
        term
        for term in target_terms
        if term and term in lowered and (not term.isascii() or not term.isalpha())
    )
    for match in _WORD_RE.finditer(text):
        word_tokens = set(_stable_tokens(match.group(0)))
        word_tokens.add(match.group(0).casefold())
        tokens.update(target.intersection(word_tokens))
    return tokens


def _window_around_interval(
    *,
    text_chars: int,
    interval: _TextSpan,
    max_chars: int,
) -> _TextSpan:
    interval_chars = interval.end - interval.start
    if interval_chars >= max_chars:
        center = (interval.start + interval.end) // 2
        start = max(0, center - max_chars // 2)
    else:
        left_context = (max_chars - interval_chars) // 2
        start = max(0, interval.start - left_context)
    start = min(start, max(0, text_chars - max_chars))
    return _TextSpan(start, min(text_chars, start + max_chars))


def _select_long_sentence_window(
    text: str,
    *,
    target_terms: tuple[str, ...],
    query_terms: tuple[str, ...],
    requirement_terms: tuple[tuple[str, ...], ...],
    max_chars: int,
) -> _Selection | None:
    intervals = _matching_intervals(text, set(target_terms))
    if not intervals:
        return None
    choices: list[tuple[tuple[int, int, int], _TextSpan, set[str]]] = []
    for interval in intervals:
        span = _window_around_interval(
            text_chars=len(text), interval=interval, max_chars=max_chars
        )
        tokens = _scoring_tokens(text[span.start : span.end], target_terms)
        quality = _quality(
            tokens,
            query_terms=query_terms,
            requirement_terms=requirement_terms,
            all_terms=target_terms,
        )
        choices.append((quality, span, tokens))
    quality, span, tokens = max(
        choices,
        key=lambda choice: (*choice[0], -choice[1].start),
    )
    return _Selection(
        start=span.start,
        end=span.end,
        score=_score(quality),
        matched_terms=_matched_terms(tokens, target_terms),
        reason=REASON_LONG_SENTENCE_WINDOW,
    )


def _select_span(
    text: str,
    *,
    query_terms: tuple[str, ...],
    requirement_terms: tuple[tuple[str, ...], ...],
    target_terms: tuple[str, ...],
    max_chars: int,
    context_sentences: int,
) -> _Selection:
    full_tokens = _scoring_tokens(text, target_terms)
    full_quality = _quality(
        full_tokens,
        query_terms=query_terms,
        requirement_terms=requirement_terms,
        all_terms=target_terms,
    )
    full_matches = _matched_terms(full_tokens, target_terms)
    if len(text) <= max_chars:
        return _Selection(
            0,
            len(text),
            _score(full_quality),
            full_matches,
            REASON_WITHIN_BUDGET,
        )
    if not target_terms:
        return _Selection(
            0,
            len(text),
            0.0,
            (),
            REASON_FALLBACK_NO_TERMS,
            fallback=True,
        )
    if not full_matches:
        return _Selection(
            0,
            len(text),
            0.0,
            (),
            REASON_FALLBACK_NO_MATCH,
            fallback=True,
        )

    sentences = _sentence_spans(text)
    sentence_tokens = [
        _scoring_tokens(text[span.start : span.end], target_terms) for span in sentences
    ]
    matching_sentence_indexes = [
        index
        for index, tokens in enumerate(sentence_tokens)
        if tokens.intersection(target_terms)
    ]
    if not matching_sentence_indexes:
        # This is uncommon (for example, a tokenizer boundary disagreement),
        # but a raw lexical hit can still provide a trustworthy exact window.
        window = _select_long_sentence_window(
            text,
            target_terms=target_terms,
            query_terms=query_terms,
            requirement_terms=requirement_terms,
            max_chars=max_chars,
        )
        if window is not None:
            return window
        return _Selection(
            0,
            len(text),
            0.0,
            (),
            REASON_FALLBACK_NO_MATCH,
            fallback=True,
        )

    focal_index = max(
        matching_sentence_indexes,
        key=lambda index: (
            *_quality(
                sentence_tokens[index],
                query_terms=query_terms,
                requirement_terms=requirement_terms,
                all_terms=target_terms,
            ),
            -index,
        ),
    )
    focal = sentences[focal_index]
    if focal.end - focal.start > max_chars:
        focal_text = text[focal.start : focal.end]
        local_window = _select_long_sentence_window(
            focal_text,
            target_terms=target_terms,
            query_terms=query_terms,
            requirement_terms=requirement_terms,
            max_chars=max_chars,
        )
        if local_window is not None:
            return _Selection(
                start=focal.start + local_window.start,
                end=focal.start + local_window.end,
                score=local_window.score,
                matched_terms=local_window.matched_terms,
                reason=local_window.reason,
            )
        return _Selection(
            0,
            len(text),
            0.0,
            (),
            REASON_FALLBACK_NO_MATCH,
            fallback=True,
        )

    lower = max(0, focal_index - context_sentences)
    upper = min(len(sentences) - 1, focal_index + context_sentences)
    choices: list[tuple[tuple[int, int, int], int, int, _TextSpan, set[str]]] = []
    for start_index in range(lower, focal_index + 1):
        for end_index in range(focal_index, upper + 1):
            span = _TextSpan(sentences[start_index].start, sentences[end_index].end)
            if span.end - span.start > max_chars:
                continue
            tokens = _scoring_tokens(text[span.start : span.end], target_terms)
            quality = _quality(
                tokens,
                query_terms=query_terms,
                requirement_terms=requirement_terms,
                all_terms=target_terms,
            )
            choices.append((quality, start_index, end_index, span, tokens))
    quality, start_index, end_index, span, tokens = max(
        choices,
        key=lambda choice: (
            *choice[0],
            choice[2] - choice[1] + 1,
            -abs((focal_index - choice[1]) - (choice[2] - focal_index)),
            -choice[1],
        ),
    )
    return _Selection(
        start=span.start,
        end=span.end,
        score=_score(quality),
        matched_terms=_matched_terms(tokens, target_terms),
        reason=REASON_QUERY_SPAN,
    )


def _materialize_doc(
    doc: RetrievedDoc,
    source: _SourceView,
    selection: _Selection,
    requirements: Sequence[_RequirementTerms],
    attributed_requirement_ids: tuple[str, ...] | None,
) -> RetrievedDoc:
    snapshot = cast(dict[str, Any], copy.deepcopy(doc))
    for key in _PACK_SOURCE_KEYS:
        snapshot.pop(key, None)
    raw_meta = snapshot.get("meta")
    if isinstance(raw_meta, Mapping):
        meta = dict(raw_meta)
        # ``context`` may contain facts outside the selected body span, while
        # section_path is only a locator.  Keeping context would make the
        # generator's rendered evidence broader than the verifier's closure.
        meta.pop("context", None)
        snapshot["meta"] = meta
    snapshot["text"] = source.text[selection.start : selection.end]
    snapshot[_SPAN_SOURCE_TEXT_KEY] = source.text
    snapshot[_SPAN_SOURCE_START_KEY] = source.start
    snapshot[_SPAN_SOURCE_END_KEY] = source.end
    snapshot[_SPAN_SOURCE_OVERLAP_KEY] = source.overlap_chars
    retrieval = dict(_mapping(snapshot.get("retrieval")))
    ultimate_start = source.start + selection.start
    ultimate_end = source.start + selection.end
    selected_tokens = _scoring_tokens(snapshot["text"], _target_terms((), requirements))
    detected_requirement_ids = [
        requirement.requirement_id
        for requirement in requirements
        if requirement.requirement_id
        and selected_tokens.intersection(requirement.match_terms)
    ]
    compressed = selection.end - selection.start < len(source.text)
    matched_requirement_ids = (
        detected_requirement_ids
        if compressed or attributed_requirement_ids is None
        else list(attributed_requirement_ids)
    )
    retrieval.update(
        {
            "evidence_span_selected": compressed,
            "evidence_span_input_start": source.start,
            "evidence_span_input_end": source.start + len(source.text),
            "evidence_span_start": ultimate_start,
            "evidence_span_end": ultimate_end,
            "evidence_span_original_chars": len(source.text),
            "evidence_span_selected_chars": selection.end - selection.start,
            "evidence_span_score": selection.score,
            "evidence_span_matched_terms": list(selection.matched_terms),
            "evidence_span_matched_requirement_ids": matched_requirement_ids,
            "evidence_span_reason": selection.reason,
            "evidence_text_start": ultimate_start,
            "evidence_text_end": ultimate_end,
            "evidence_trimmed_overlap_chars": 0,
        }
    )
    if compressed or attributed_requirement_ids is not None:
        # A compressed view must never retain requirement attribution supported
        # only by text outside the selected span.
        retrieval["matched_requirement_ids"] = matched_requirement_ids
    snapshot["retrieval"] = retrieval
    return cast(RetrievedDoc, snapshot)


class EvidenceSpanSelector:
    """Reusable selector with one tokenized query/requirement plan."""

    def __init__(
        self,
        *,
        query: str,
        evidence_requirements: Sequence[Mapping[str, Any] | str] = (),
        max_chars_per_doc: int,
        context_sentences: int = 1,
    ) -> None:
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if isinstance(evidence_requirements, (str, bytes)) or not isinstance(
            evidence_requirements, Sequence
        ):
            raise ValueError("evidence_requirements must be a sequence")
        if (
            isinstance(max_chars_per_doc, bool)
            or not isinstance(max_chars_per_doc, int)
            or max_chars_per_doc < 1
        ):
            raise ValueError("max_chars_per_doc must be a positive integer")
        if (
            isinstance(context_sentences, bool)
            or not isinstance(context_sentences, int)
            or context_sentences < 0
        ):
            raise ValueError("context_sentences must be a non-negative integer")
        for requirement in evidence_requirements:
            if not isinstance(requirement, (str, Mapping)):
                raise ValueError(
                    "each evidence requirement must be a mapping or string"
                )

        self.query = query
        self.max_chars_per_doc = max_chars_per_doc
        self.context_sentences = context_sentences
        self._query_terms, self._requirements = _term_plan(query, evidence_requirements)

    def _select_with_details(
        self,
        doc: RetrievedDoc,
        *,
        matched_requirement_ids: Sequence[str] | None = None,
    ) -> tuple[RetrievedDoc, _SourceView, _Selection]:
        if isinstance(matched_requirement_ids, (str, bytes)):
            raise ValueError("matched_requirement_ids must be a sequence of strings")
        requirements, attributed_requirement_ids = _active_requirements(
            doc, self._requirements, matched_requirement_ids
        )
        requirement_terms = tuple(
            requirement.match_terms for requirement in requirements
        )
        target_terms = _target_terms(self._query_terms, requirements)
        source = _source_view(doc)
        selection = _select_span(
            source.text,
            query_terms=self._query_terms,
            requirement_terms=requirement_terms,
            target_terms=target_terms,
            max_chars=self.max_chars_per_doc,
            context_sentences=self.context_sentences,
        )
        return (
            _materialize_doc(
                doc,
                source,
                selection,
                requirements,
                attributed_requirement_ids,
            ),
            source,
            selection,
        )

    def select(
        self,
        doc: RetrievedDoc,
        *,
        matched_requirement_ids: Sequence[str] | None = None,
    ) -> RetrievedDoc:
        """Return an isolated verbatim span for one canonical document."""

        snapshot, _, _ = self._select_with_details(
            doc, matched_requirement_ids=matched_requirement_ids
        )
        return snapshot

    def select_many(self, docs: Sequence[RetrievedDoc]) -> EvidenceSpanBatch:
        """Select documents and aggregate body-character compression metrics."""

        output: list[RetrievedDoc] = []
        compressed_count = 0
        fallback_count = 0
        input_chars = 0
        selected_chars = 0
        reason_counts: dict[str, int] = {}
        for doc in docs:
            snapshot, source, selection = self._select_with_details(doc)
            output.append(snapshot)
            input_chars += len(source.text)
            selected_chars += selection.end - selection.start
            compressed_count += selection.end - selection.start < len(source.text)
            fallback_count += selection.fallback
            reason_counts[selection.reason] = reason_counts.get(selection.reason, 0) + 1
        return EvidenceSpanBatch(
            docs=tuple(output),
            input_count=len(docs),
            compressed_count=compressed_count,
            fallback_count=fallback_count,
            input_chars=input_chars,
            selected_chars=selected_chars,
            reason_counts=reason_counts,
        )


def select_evidence_span(
    doc: RetrievedDoc,
    *,
    query: str,
    evidence_requirements: Sequence[Mapping[str, Any] | str] = (),
    matched_requirement_ids: Sequence[str] | None = None,
    max_chars_per_doc: int,
    context_sentences: int = 1,
) -> RetrievedDoc:
    """Convenience wrapper for selecting one canonical document."""

    selector = EvidenceSpanSelector(
        query=query,
        evidence_requirements=evidence_requirements,
        max_chars_per_doc=max_chars_per_doc,
        context_sentences=context_sentences,
    )
    return selector.select(doc, matched_requirement_ids=matched_requirement_ids)


def select_evidence_spans(
    docs: Sequence[RetrievedDoc],
    *,
    query: str,
    evidence_requirements: Sequence[Mapping[str, Any] | str] = (),
    max_chars_per_doc: int,
    context_sentences: int = 1,
) -> EvidenceSpanBatch:
    """Select exact spans, failing open to full text when matching is unsafe."""

    selector = EvidenceSpanSelector(
        query=query,
        evidence_requirements=evidence_requirements,
        max_chars_per_doc=max_chars_per_doc,
        context_sentences=context_sentences,
    )
    return selector.select_many(docs)
