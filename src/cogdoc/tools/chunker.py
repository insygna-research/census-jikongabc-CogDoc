import bisect
import re
from dataclasses import dataclass
from typing import List
from cogdoc.graph.state import ParsedPage, RetrievedDoc
from cogdoc.tools.chunk_identity import (
    DEFAULT_CHUNK_CONTEXT_CHARS,
    DEFAULT_CHUNK_CHAR_OVERLAP,
    DEFAULT_CHUNK_CHAR_SIZE,
    MIN_CHUNK_CHARS,
    build_chunk_id,
)


_BLANK_LINE_RE = re.compile(r"\n\s*\n")
_SENTENCE_END_RE = re.compile(r"[。！？!?；;]+[\"'”’）】》」』]*|[.!?;]+(?=\s|$)")
_SOFT_BREAK_RE = re.compile(r"\n+")


# 文本片段用全局字符下标表示闭开区间。
@dataclass(frozen=True)
class TextSpan:
    start: int
    end: int


# 修剪 span 两端空白并丢弃空片段。
def _trim_span(text: str, start: int, end: int) -> TextSpan | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return TextSpan(start, end)


# 按空行提取段落级文本片段。
def _paragraph_spans(text: str) -> List[TextSpan]:
    spans: List[TextSpan] = []
    pos = 0
    while pos < len(text):
        match = _BLANK_LINE_RE.search(text, pos)
        raw_end = match.start() if match else len(text)
        span = _trim_span(text, pos, raw_end)
        if span:
            spans.append(span)
        if not match:
            break
        pos = match.end()
    return spans


# 把超长片段继续拆到最大长度以内。
def _split_long_span(text: str, span: TextSpan, max_chars: int) -> List[TextSpan]:
    pieces: List[TextSpan] = []
    start = span.start
    while start < span.end:
        hard_end = min(start + max_chars, span.end)
        if hard_end >= span.end:
            piece = _trim_span(text, start, span.end)
            if piece:
                pieces.append(piece)
            break

        search_start = start + max(max_chars // 2, 1)
        boundary = -1
        for pattern in (_SENTENCE_END_RE, _SOFT_BREAK_RE, re.compile(r"\s+")):
            for match in pattern.finditer(text, search_start, hard_end):
                boundary = match.end()
            if boundary != -1:
                break
        if boundary <= start:
            boundary = hard_end

        piece = _trim_span(text, start, boundary)
        if piece:
            pieces.append(piece)
        start = boundary
    return pieces


# 按句末标点或软换行拆分段落。
def _sentence_spans(text: str, paragraph: TextSpan, max_chars: int) -> List[TextSpan]:
    spans: List[TextSpan] = []
    start = paragraph.start
    matches = sorted(
        list(_SENTENCE_END_RE.finditer(text, paragraph.start, paragraph.end))
        + list(_SOFT_BREAK_RE.finditer(text, paragraph.start, paragraph.end)),
        key=lambda match: match.end(),
    )
    for match in matches:
        end = match.end()
        span = _trim_span(text, start, end)
        if span:
            spans.extend(_split_long_span(text, span, max_chars))
        start = end

    tail = _trim_span(text, start, paragraph.end)
    if tail:
        spans.extend(_split_long_span(text, tail, max_chars))
    return spans


# 构造语义优先的最小组合单元。
def _semantic_spans(text: str, max_chars: int) -> List[TextSpan]:
    spans: List[TextSpan] = []
    for paragraph in _paragraph_spans(text):
        if paragraph.end - paragraph.start <= max_chars:
            spans.append(paragraph)
        else:
            spans.extend(_sentence_spans(text, paragraph, max_chars))
    return spans


# 查找下一个 chunk 的完整语义重叠起点。
def _find_overlap_start(
    units: List[TextSpan], start_idx: int, end_idx: int, overlap_chars: int
) -> int:
    # 从当前 chunk 末尾向左找 overlap 起点，优先复用完整语义单元。
    if overlap_chars <= 0 or end_idx <= start_idx + 1:
        return end_idx

    target = units[end_idx - 1].end - overlap_chars
    next_start = end_idx - 1
    while next_start > start_idx and units[next_start - 1].end > target:
        next_start -= 1
    if next_start <= start_idx:
        next_start = end_idx - 1
    return next_start


# 构造 chunk 前后的定位上下文。
def _build_context(text: str, start: int, end: int, context_chars: int) -> str:
    if context_chars <= 0:
        return ""

    before = _context_before(text, start, context_chars)
    after = _context_after(text, end, context_chars)
    parts = []
    if before:
        parts.append(f"前文：{before}")
    if after:
        parts.append(f"后文：{after}")
    return "\n".join(parts)


# 截取 chunk 前方的上下文片段。
def _context_before(text: str, start: int, context_chars: int) -> str:
    snippet = text[max(0, start - context_chars) : start].strip()
    for match in _SENTENCE_END_RE.finditer(snippet):
        candidate = snippet[match.end() :].strip()
        if candidate:
            return candidate
    return snippet


# 截取 chunk 后方的上下文片段。
def _context_after(text: str, end: int, context_chars: int) -> str:
    snippet = text[end : min(len(text), end + context_chars)].strip()
    boundary = -1
    for match in _SENTENCE_END_RE.finditer(snippet):
        boundary = match.end()
    if boundary > 0:
        return snippet[:boundary].strip()
    return snippet


# 切分 paper。
def chunk_paper(
    parsed_pages: List[ParsedPage],
    source_sha256: str = "",
    chunk_char_size: int = DEFAULT_CHUNK_CHAR_SIZE,
    chunk_char_overlap: int = DEFAULT_CHUNK_CHAR_OVERLAP,
    chunk_context_chars: int = DEFAULT_CHUNK_CONTEXT_CHARS,
) -> List[RetrievedDoc]:

    if not parsed_pages:
        return []

    source_name = parsed_pages[0]["source"]
    if not source_sha256:
        raise ValueError("source_sha256 is required for stable chunk identity")

    global_text = ""

    page_starts: List[int] = []

    page_nums: List[int] = []

    current_idx = 0

    for page in parsed_pages:
        p_text = page["text"]

        if global_text and p_text:
            global_text += "\n\n"
            current_idx += 2

        page_starts.append(current_idx)
        page_nums.append(page["page"])

        global_text += p_text
        current_idx += len(p_text)

    chunks: List[RetrievedDoc] = []

    total_len = len(global_text)

    # local_chunk_index 只在单个 PDF 内递增，参与稳定 chunk_id。
    local_chunk_index = 0

    # 完成 find页码bypos 处理。
    def find_page_by_pos(pos: int) -> int:
        idx = bisect.bisect_right(page_starts, pos) - 1
        return page_nums[max(0, idx)]

    if total_len == 0:
        return []

    max_chars = max(1, chunk_char_size)
    semantic_units = _semantic_spans(global_text, max_chars)
    unit_idx = 0

    while unit_idx < len(semantic_units):
        chunk_start = semantic_units[unit_idx].start
        next_idx = unit_idx
        while next_idx < len(semantic_units):
            candidate_end = semantic_units[next_idx].end
            if next_idx > unit_idx and candidate_end - chunk_start > max_chars:
                break
            next_idx += 1

        if next_idx == unit_idx:
            next_idx += 1

        while (
            next_idx > unit_idx + 1
            and semantic_units[next_idx - 1].end - chunk_start > max_chars
        ):
            next_idx -= 1

        chunk_end = semantic_units[next_idx - 1].end
        chunk_text = global_text[chunk_start:chunk_end].strip()

        if len(chunk_text) > MIN_CHUNK_CHARS:
            p_start = find_page_by_pos(chunk_start)
            p_end = find_page_by_pos(chunk_end - 1)
            chunk_id = build_chunk_id(
                source_sha256, source_name, p_start, p_end, local_chunk_index
            )
            context = _build_context(
                global_text, chunk_start, chunk_end, chunk_context_chars
            )

            meta = {
                "chunk_id": chunk_id,
                "source_sha256": source_sha256,
                "local_chunk_index": local_chunk_index,
                "chunk_index": local_chunk_index,
                "source": source_name,
                "page": p_start,
                "page_start": p_start,
                "page_end": p_end,
                "origin": "vector",
            }
            covered_pages = [
                page
                for page in parsed_pages
                if p_start <= int(page["page"]) <= p_end
            ]
            extraction_methods = {
                str(page.get("extraction_method", "native"))
                for page in covered_pages
            }
            meta["extraction_method"] = (
                next(iter(extraction_methods))
                if len(extraction_methods) == 1
                else "mixed"
            )
            ocr_providers = {
                str(page.get("ocr_provider"))
                for page in covered_pages
                if page.get("ocr_provider")
                and page.get("extraction_method") == "ocr"
            }
            if ocr_providers:
                meta["ocr_provider"] = (
                    next(iter(ocr_providers))
                    if len(ocr_providers) == 1
                    else "mixed"
                )
            if context:
                meta["context"] = context

            chunks.append({"text": chunk_text, "meta": meta})

            local_chunk_index += 1

        if next_idx >= len(semantic_units):
            break

        if chunk_char_overlap >= chunk_char_size:
            break
        unit_idx = _find_overlap_start(
            semantic_units, unit_idx, next_idx, chunk_char_overlap
        )

    return chunks
