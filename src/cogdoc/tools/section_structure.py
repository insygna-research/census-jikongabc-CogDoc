"""Conservative, dependency-free section structure detection.

The detector works on the full document text and returns character spans.  A
section span includes its heading line and all content up to the next detected
heading.  Text before the first heading is represented as a level-zero
preamble so callers can retain it instead of silently dropping it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SectionSpan:
    """A detected section represented as a half-open character span."""

    start: int
    end: int
    title: str
    path: tuple[str, ...]
    level: int
    ordinal: int


@dataclass(frozen=True, slots=True)
class _Line:
    start: int
    text: str


@dataclass(frozen=True, slots=True)
class _Heading:
    start: int
    title: str
    level: int


_LINE_RE = re.compile(r"[^\r\n]*(?:\r\n|\r|\n|$)")
_WHITESPACE_RE = re.compile(r"\s+")
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*|\d+")

_MARKDOWN_HEADING_RE = re.compile(
    r"^(?P<marks>#{1,6})[ \t]+(?P<title>.+?)(?:[ \t]+#+)?$"
)
_CHINESE_ORDINAL_RE = re.compile(
    r"^第\s*(?P<number>[〇零一二三四五六七八九十百千万两\d]+)\s*"
    r"(?P<kind>小节|章|节|篇|部|卷)"
    r"(?:\s*[：:、.．\-—]\s*)?(?P<title>.*)$"
)
_CHINESE_ENUM_RE = re.compile(
    r"^(?P<number>[〇零一二三四五六七八九十百]+)[、.．]\s*(?P<title>.+)$"
)
_PAREN_ENUM_RE = re.compile(
    r"^[（(](?P<number>[〇零一二三四五六七八九十百\d]{1,4})[）)]"
    r"\s*(?P<title>.+)$"
)
_ARABIC_PAREN_ENUM_RE = re.compile(r"^(?P<number>\d{1,3})[)）]\s+(?P<title>.+)$")
_DECIMAL_HEADING_RE = re.compile(
    r"^(?P<number>\d{1,3}(?:\.\d{1,3}){0,5})"
    r"(?P<separator>[.．、]?)(?P<space>[ \t]*)(?P<title>.+)$"
)
_ROMAN_HEADING_RE = re.compile(
    r"^(?P<number>[IVXLCDM]{1,7})[.．、)]\s+(?P<title>.+)$",
    re.IGNORECASE,
)
_ENGLISH_STRUCTURAL_RE = re.compile(
    r"^(?P<kind>chapter|part|section|subsection|appendix)"
    r"(?:\s+(?P<number>\d{1,3}(?:\.\d{1,3})*|[IVXLCDM]+|[A-Z])"
    r"(?=\s|[：:.\-—]|$))?"
    r"(?:\s*[：:.\-—]\s*|\s+)?(?P<title>.*)$",
    re.IGNORECASE,
)

_ENGLISH_COMMON_TITLES = frozenset(
    {
        "abstract",
        "acknowledgement",
        "acknowledgements",
        "acknowledgment",
        "acknowledgments",
        "appendices",
        "appendix",
        "background",
        "bibliography",
        "conclusion",
        "conclusions",
        "discussion",
        "executive summary",
        "experimental setup",
        "experiments",
        "future work",
        "introduction",
        "limitations",
        "literature review",
        "materials and methods",
        "methodology",
        "methods",
        "preface",
        "references",
        "related work",
        "results",
        "results and discussion",
        "summary",
    }
)

_CHINESE_COMMON_TITLES = frozenset(
    {
        "摘要",
        "绪论",
        "前言",
        "引言",
        "背景",
        "研究背景",
        "相关工作",
        "文献综述",
        "方法",
        "研究方法",
        "材料与方法",
        "实验",
        "实验设置",
        "结果",
        "讨论",
        "结果与讨论",
        "结论",
        "总结",
        "总结与展望",
        "未来工作",
        "局限性",
        "参考文献",
        "致谢",
        "附录",
    }
)

# These words deliberately describe section-like concepts rather than general
# vocabulary.  Generic short headings are accepted only at an isolated block
# boundary and when one of these signals (or English title case) is present.
_ENGLISH_HEADING_SIGNALS = frozenset(
    {
        "analysis",
        "architecture",
        "configuration",
        "contribution",
        "contributions",
        "data",
        "deployment",
        "design",
        "evaluation",
        "implementation",
        "model",
        "objective",
        "objectives",
        "overview",
        "performance",
        "pipeline",
        "scope",
        "security",
        "setup",
        "system",
        "workflow",
    }
)
_CHINESE_HEADING_SIGNALS = (
    "概述",
    "背景",
    "目标",
    "范围",
    "贡献",
    "创新",
    "架构",
    "设计",
    "原理",
    "流程",
    "方案",
    "模型",
    "数据",
    "实现",
    "配置",
    "部署",
    "实验",
    "评估",
    "分析",
    "性能",
    "安全",
    "限制",
    "展望",
    "案例",
    "问题",
    "注意事项",
)
_ENGLISH_SENTENCE_STARTS = frozenset(
    {
        "a",
        "an",
        "he",
        "i",
        "it",
        "our",
        "she",
        "that",
        "the",
        "their",
        "there",
        "these",
        "they",
        "this",
        "those",
        "we",
        "you",
    }
)
_CHINESE_SENTENCE_STARTS = (
    "请",
    "我们",
    "本文将",
    "本研究将",
    "这",
    "那",
    "点击",
    "选择",
    "输入",
    "打开",
    "关闭",
    "安装",
    "运行",
    "确保",
)
_TERMINAL_SENTENCE_PUNCTUATION = ("。", "！", "？", "!", "?", ";", "；", ".")


def detect_section_spans(text: str) -> list[SectionSpan]:
    """Detect an ordered, conservative section hierarchy in ``text``.

    ``path`` contains the current heading and its detected ancestors.  Explicit
    decimal/Markdown numbering determines hierarchy; common and generic
    unnumbered headings are top-level.  If no heading is found, non-whitespace
    input is returned as one preamble span.
    """

    if not text or not text.strip():
        return []

    lines = _split_lines(text)
    headings: list[_Heading] = []
    for index, line in enumerate(lines):
        stripped = line.text.strip()
        if not stripped:
            continue

        previous_is_blank = index == 0 or not lines[index - 1].text.strip()
        next_is_blank = index == len(lines) - 1 or not lines[index + 1].text.strip()
        heading = _classify_heading(
            stripped,
            start=line.start,
            previous_is_blank=previous_is_blank,
            next_is_blank=next_is_blank,
        )
        if heading is not None:
            headings.append(heading)

    if not headings:
        return [
            SectionSpan(
                start=0,
                end=len(text),
                title="",
                path=(),
                level=0,
                ordinal=0,
            )
        ]

    spans: list[SectionSpan] = []
    first_heading_start = headings[0].start
    if text[:first_heading_start].strip():
        spans.append(
            SectionSpan(
                start=0,
                end=first_heading_start,
                title="",
                path=(),
                level=0,
                ordinal=0,
            )
        )

    ancestors: list[tuple[int, str]] = []
    for index, heading in enumerate(headings):
        while ancestors and ancestors[-1][0] >= heading.level:
            ancestors.pop()
        path = tuple(title for _, title in ancestors) + (heading.title,)
        end = headings[index + 1].start if index + 1 < len(headings) else len(text)
        spans.append(
            SectionSpan(
                start=heading.start,
                end=end,
                title=heading.title,
                path=path,
                level=heading.level,
                ordinal=len(spans),
            )
        )
        ancestors.append((heading.level, heading.title))

    return spans


def _split_lines(text: str) -> list[_Line]:
    lines: list[_Line] = []
    for match in _LINE_RE.finditer(text):
        raw = match.group(0)
        if not raw:
            break
        lines.append(_Line(start=match.start(), text=raw.rstrip("\r\n")))
    return lines


def _classify_heading(
    text: str,
    *,
    start: int,
    previous_is_blank: bool,
    next_is_blank: bool,
) -> _Heading | None:
    markdown = _MARKDOWN_HEADING_RE.fullmatch(text)
    if markdown:
        title = _clean_title(markdown.group("title"))
        if title:
            return _Heading(
                start=start, title=title, level=len(markdown.group("marks"))
            )

    chinese_ordinal = _CHINESE_ORDINAL_RE.fullmatch(text)
    if chinese_ordinal:
        kind = chinese_ordinal.group("kind")
        marker = f"第{chinese_ordinal.group('number')}{kind}"
        title = _clean_title(chinese_ordinal.group("title")) or marker
        level = 3 if kind == "小节" else 2 if kind == "节" else 1
        return _Heading(start=start, title=title, level=level)

    english_structural = _ENGLISH_STRUCTURAL_RE.fullmatch(text)
    if english_structural:
        kind = english_structural.group("kind").casefold()
        number = english_structural.group("number") or ""
        raw_title = english_structural.group("title")
        fallback = " ".join(part for part in (kind.title(), number) if part)
        title = _clean_title(raw_title) or fallback
        has_safe_title = bool(number) and (
            not raw_title
            or _numbered_title_is_safe(_clean_title(raw_title), previous_is_blank)
        )
        has_safe_unnumbered_title = not number and (
            not raw_title
            or _is_common_title(raw_title)
            or (previous_is_blank and _looks_like_short_heading(raw_title))
        )
        if title and (has_safe_title or has_safe_unnumbered_title):
            level = _english_structural_level(kind, number)
            return _Heading(start=start, title=title, level=level)

    numbered = _classify_numbered_heading(text, start, previous_is_blank)
    if numbered is not None:
        return numbered

    if _is_common_title(text):
        return _Heading(start=start, title=_clean_title(text), level=1)

    if previous_is_blank and next_is_blank and _looks_like_short_heading(text):
        return _Heading(start=start, title=_clean_title(text), level=1)

    return None


def _classify_numbered_heading(
    text: str, start: int, previous_is_blank: bool
) -> _Heading | None:
    for pattern, level in (
        (_CHINESE_ENUM_RE, 1),
        (_PAREN_ENUM_RE, 2),
        (_ARABIC_PAREN_ENUM_RE, 2),
        (_ROMAN_HEADING_RE, 1),
    ):
        match = pattern.fullmatch(text)
        if match:
            title = _clean_title(match.group("title"))
            if _numbered_title_is_safe(title, previous_is_blank):
                return _Heading(start=start, title=title, level=level)
            return None

    decimal = _DECIMAL_HEADING_RE.fullmatch(text)
    if not decimal:
        return None
    if not decimal.group("separator") and not decimal.group("space"):
        return None

    number = decimal.group("number")
    components = number.split(".")
    if len(components) == 1 and int(components[0]) > 99:
        return None

    title = _clean_title(decimal.group("title"))
    if not _numbered_title_is_safe(title, previous_is_blank):
        return None
    return _Heading(start=start, title=title, level=len(components))


def _english_structural_level(kind: str, number: str) -> int:
    if kind == "subsection":
        return 2
    if kind == "section" and "." in number:
        return len(number.split("."))
    return 1


def _numbered_title_is_safe(title: str, previous_is_blank: bool) -> bool:
    if not title or len(title) > 100 or title.endswith(_TERMINAL_SENTENCE_PUNCTUATION):
        return False
    if _is_common_title(title) or _looks_like_short_heading(title):
        return True
    if not previous_is_blank:
        return False

    if _CJK_RE.search(title):
        compact = _WHITESPACE_RE.sub("", title)
        if len(compact) > 24 or compact.startswith(_CHINESE_SENTENCE_STARTS):
            return False
        return not any(mark in compact for mark in ("，", "。", "！", "？", "；"))

    words = _WORD_RE.findall(title)
    if not 1 <= len(words) <= 10:
        return False
    return words[0].casefold() not in _ENGLISH_SENTENCE_STARTS


def _is_common_title(title: str) -> bool:
    cleaned = _clean_title(title)
    english_key = re.sub(r"[\s_-]+", " ", cleaned).casefold()
    if english_key in _ENGLISH_COMMON_TITLES:
        return True
    chinese_key = _WHITESPACE_RE.sub("", cleaned)
    return chinese_key in _CHINESE_COMMON_TITLES


def _looks_like_short_heading(title: str) -> bool:
    cleaned = _clean_title(title)
    if not cleaned or len(cleaned) > 80:
        return False
    if cleaned.endswith(_TERMINAL_SENTENCE_PUNCTUATION):
        return False
    if any(mark in cleaned for mark in ("，", ",", "；", ";", "。", "！", "？")):
        return False
    if cleaned.startswith(("- ", "* ", "+ ", "• ", "> ")):
        return False

    if _CJK_RE.search(cleaned):
        compact = _WHITESPACE_RE.sub("", cleaned)
        if not 2 <= len(compact) <= 18:
            return False
        if compact.startswith(_CHINESE_SENTENCE_STARTS) or compact.endswith(
            ("了", "吗", "呢", "吧")
        ):
            return False
        return any(signal in compact for signal in _CHINESE_HEADING_SIGNALS)

    words = _WORD_RE.findall(cleaned)
    if not 1 <= len(words) <= 8:
        return False
    lowered = [word.casefold() for word in words]
    if lowered[0] in _ENGLISH_SENTENCE_STARTS:
        return False
    return any(word in _ENGLISH_HEADING_SIGNALS for word in lowered)


def _clean_title(title: str) -> str:
    return _WHITESPACE_RE.sub(" ", title.strip()).rstrip("：:").rstrip()
