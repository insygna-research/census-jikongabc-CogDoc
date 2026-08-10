from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

from cogdoc.tools.citation_ledger import format_evidence_id
from cogdoc.tools.public_citation_ledger import validate_public_citation_ledger


RESEARCH_CLAIM_REJECTED_CONTENT = (
    "本章节声明审计未通过，已阻止未经证据支持的内容进入报告。"
)
RESEARCH_GENERATION_ERROR_CONTENT = "本章节生成失败，已阻止未经校验的内容进入报告。"
RESEARCH_BLOCKED_CONTENT = {
    "no_evidence": "本章节未找到足以支持结论的证据。",
    "contradictory": "本章节候选证据存在直接冲突，已阻止自动生成。",
    "retrieval_error": "本章节检索失败，已阻止自动生成。",
    "verification_error": "本章节证据校验失败，已阻止自动生成。",
    "budget_exhausted": "本章节证据预算不足，已阻止自动生成。",
}
RESEARCH_DEFAULT_BLOCKED_CONTENT = "本章节证据状态不允许自动生成。"

_RAW_HTML_RE = re.compile(
    r"<!--|<![A-Za-z\[]|<\?|"
    r"</?[A-Za-z][^<>]*>|"
    r"<[A-Za-z][A-Za-z0-9+.-]{1,31}:[^<>\s]+>|"
    r"<[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}>",
    re.IGNORECASE | re.DOTALL,
)
_MARKDOWN_REFERENCE_DEFINITION_RE = re.compile(
    r"^[ \t]{0,3}\[[^\]\r\n]+\]:[ \t]*\S",
    re.MULTILINE,
)


class UnsafeResearchContentError(ValueError):
    """Raised when publishable research prose contains active markup."""


def _has_unescaped_token(value: str, token: str) -> bool:
    start = 0
    while (index := value.find(token, start)) >= 0:
        backslashes = 0
        cursor = index - 1
        while cursor >= 0 and value[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        if backslashes % 2 == 0:
            return True
        start = index + len(token)
    return False


def ensure_passive_research_markdown(content: str) -> None:
    """Reject markup that can navigate, fetch, or execute when rendered."""

    if type(content) is not str:
        raise TypeError("research section content must be a string")
    if (
        _RAW_HTML_RE.search(content)
        or _MARKDOWN_REFERENCE_DEFINITION_RE.search(content)
        or _has_unescaped_token(content, "![")
        or _has_unescaped_token(content, "](")
    ):
        raise UnsafeResearchContentError(
            "research section content contains active HTML or Markdown"
        )


def canonical_research_gap_content(
    generation_status: str, verification_status: str
) -> str:
    """Return the only publishable prose for a non-generated section."""

    if generation_status == "claim_rejected":
        return RESEARCH_CLAIM_REJECTED_CONTENT
    if generation_status == "generation_error":
        return RESEARCH_GENERATION_ERROR_CONTENT
    return RESEARCH_BLOCKED_CONTENT.get(
        generation_status,
        RESEARCH_BLOCKED_CONTENT.get(
            verification_status, RESEARCH_DEFAULT_BLOCKED_CONTENT
        ),
    )


def _escape_markdown_inline(value: str) -> str:
    escaped = value.replace("\\", "\\\\")
    for character in "`*_{}[]<>()#+-.!|":
        escaped = escaped.replace(character, f"\\{character}")
    return escaped


def _rebuild_global_citation_ledger(
    markdown: str,
    located_entries: Sequence[tuple[int, Mapping[str, Any]]],
) -> tuple[dict[str, Any], ...]:
    """Rebase strict section-local citation occurrences into the whole report."""

    by_identity: OrderedDict[tuple[str, int, int], dict[str, Any]] = OrderedDict()
    occurrences: list[tuple[int, int, tuple[str, int, int]]] = []
    for offset, raw_entry in located_entries:
        if type(raw_entry) is not dict:
            raise TypeError("section citation ledger entries must be plain objects")
        chunk_id = raw_entry.get("chunk_id")
        span_start = raw_entry.get("span_start")
        span_end = raw_entry.get("span_end")
        if (
            type(chunk_id) is not str
            or not chunk_id
            or type(span_start) is not int
            or type(span_end) is not int
        ):
            raise ValueError("section citation ledger has an invalid evidence identity")
        identity = (chunk_id, span_start, span_end)
        if identity not in by_identity:
            entry = {
                key: value
                for key, value in raw_entry.items()
                if key not in {"evidence_id", "occurrences"}
            }
            entry["occurrences"] = []
            by_identity[identity] = entry
        raw_occurrences = raw_entry.get("occurrences")
        if type(raw_occurrences) is not list or not raw_occurrences:
            raise ValueError("section citation ledger entry must have occurrences")
        for raw_occurrence in raw_occurrences:
            if type(raw_occurrence) is not dict:
                raise TypeError("section citation occurrences must be plain objects")
            start = raw_occurrence.get("answer_start")
            end = raw_occurrence.get("answer_end")
            if type(start) is not int or type(end) is not int:
                raise ValueError("section citation ledger has an invalid occurrence")
            occurrences.append((offset + start, offset + end, identity))

    ordered_occurrences = sorted(occurrences, key=lambda item: (item[0], item[1]))
    identity_order: list[tuple[str, int, int]] = []
    for index, (start, end, identity) in enumerate(ordered_occurrences):
        if identity not in identity_order:
            identity_order.append(identity)
        by_identity[identity]["occurrences"].append(
            {"index": index, "answer_start": start, "answer_end": end}
        )
    ledger: list[dict[str, Any]] = []
    for position, identity in enumerate(identity_order, start=1):
        entry = by_identity[identity]
        entry["evidence_id"] = format_evidence_id(position)
        ledger.append(entry)
    validation = validate_public_citation_ledger(markdown, ledger)
    if not validation.is_valid:
        raise ValueError(
            f"research report failed public citation validation: {validation.reason}"
        )
    return tuple(ledger)


def compose_research_markdown(
    job: Mapping[str, Any],
    sections: Sequence[Mapping[str, Any]],
) -> tuple[str, tuple[dict[str, Any], ...]]:
    """Compose the only publishable Markdown/ledger representation."""

    title = job.get("title") or job.get("objective")
    objective = job.get("objective")
    if type(title) is not str or type(objective) is not str:
        raise TypeError("research title and objective must be strings")
    if not isinstance(sections, Sequence) or isinstance(
        sections, (str, bytes, bytearray)
    ):
        raise TypeError("research report sections must be a sequence")
    # User/model plan labels define scope; they are not audited factual claims.
    # Render them as escaped, explicitly-labelled metadata and keep all report
    # headings deterministic so a factual or Markdown-injection title cannot
    # masquerade as a verified conclusion.
    header = (
        "# 研究报告\n\n"
        "**研究任务（用户提供，仅定义范围，不代表研究结论）：** "
        f"{_escape_markdown_inline(objective)}"
    )
    markdown_parts = [header]
    markdown_length = len(header)
    located_entries: list[tuple[int, Mapping[str, Any]]] = []
    seen_section_ids: set[str] = set()
    for section_position, raw_section in enumerate(sections, start=1):
        if not isinstance(raw_section, Mapping):
            raise TypeError("research report sections must contain objects")
        section_id = raw_section.get("section_id")
        section_title = raw_section.get("title")
        content = raw_section.get("content", "")
        local_ledger = raw_section.get("citation_ledger", [])
        evidence = raw_section.get("evidence", [])
        generation_status = str(
            raw_section.get("generation_status")
            if "generation_status" in raw_section
            else raw_section.get("status") or ""
        )
        if (
            type(section_id) is not str
            or not section_id
            or section_id in seen_section_ids
        ):
            raise ValueError("research report section IDs must be non-empty and unique")
        seen_section_ids.add(section_id)
        if type(section_title) is not str or type(content) is not str:
            raise TypeError("research report section title/content must be strings")
        if type(local_ledger) is not list or any(
            type(entry) is not dict for entry in local_ledger
        ):
            raise TypeError("section citation ledger must be a strict list of objects")
        if generation_status == "generated":
            ensure_passive_research_markdown(content)
            if (
                type(evidence) is not list
                or not evidence
                or any(type(item) is not dict for item in evidence)
            ):
                raise TypeError(
                    "generated section evidence must be a non-empty list of objects"
                )
            if any(
                type(item.get("span_start")) is not int
                or type(item.get("span_end")) is not int
                or item["span_start"] < 0
                or item["span_end"] <= item["span_start"]
                for item in evidence
            ):
                raise ValueError(
                    "generated section evidence must bind an exact text span"
                )
            local_validation = validate_public_citation_ledger(
                content,
                local_ledger,
                evidence=evidence,
                require_evidence=True,
            )
        else:
            if local_ledger:
                raise ValueError("non-generated research sections cannot cite evidence")
            expected_content = canonical_research_gap_content(
                generation_status,
                str(raw_section.get("verification_status") or ""),
            )
            if content != expected_content:
                raise ValueError(
                    "non-generated research section content is not canonical"
                )
            local_validation = validate_public_citation_ledger(content, local_ledger)
        if not local_validation.is_valid:
            raise ValueError(
                "research section failed public citation validation: "
                f"{section_id}:{local_validation.reason}"
            )
        prefix = (
            f"\n\n## 章节 {section_position}\n\n"
            "**计划标签（非事实结论）：** "
            f"{_escape_markdown_inline(section_title)}\n\n"
        )
        markdown_parts.append(prefix)
        markdown_length += len(prefix)
        content_start = markdown_length
        markdown_parts.append(content)
        markdown_length += len(content)
        located_entries.extend((content_start, entry) for entry in local_ledger)
    markdown_parts.append("\n")
    markdown = "".join(markdown_parts)
    return markdown, _rebuild_global_citation_ledger(markdown, located_entries)
