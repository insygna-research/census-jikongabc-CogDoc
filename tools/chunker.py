import bisect
from typing import List
from graph.state import ParsedPage, RetrievedDoc
from tools.chunk_identity import (
    DEFAULT_CHUNK_CHAR_OVERLAP,
    DEFAULT_CHUNK_CHAR_SIZE,
    MIN_CHUNK_CHARS,
    build_chunk_id,
)


def chunk_paper(
    parsed_pages: List[ParsedPage],
    source_sha256: str = "",
    chunk_char_size: int = DEFAULT_CHUNK_CHAR_SIZE,
    chunk_char_overlap: int = DEFAULT_CHUNK_CHAR_OVERLAP,
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

    start_idx = 0

    # local_chunk_index 只在单个 PDF 内递增，参与稳定 chunk_id。
    local_chunk_index = 0

    def find_page_by_pos(pos: int) -> int:
        idx = bisect.bisect_right(page_starts, pos) - 1
        return page_nums[max(0, idx)]

    if total_len == 0:
        return []

    while start_idx < total_len:
        end_idx = min(start_idx + chunk_char_size, total_len)

        # 在目标边界附近优先按段落或行切分。
        if end_idx < total_len:
            search_start = max(start_idx + 1, end_idx - 60)
            search_end = min(end_idx + 120, total_len)

            para_break = global_text.rfind("\n\n", search_start, search_end)
            if para_break != -1:
                end_idx = para_break + 2
            else:
                line_break = global_text.rfind("\n", search_start, search_end)
                if line_break != -1:
                    end_idx = line_break + 1

        chunk_text = global_text[start_idx:end_idx].strip()

        if len(chunk_text) > MIN_CHUNK_CHARS:
            p_start = find_page_by_pos(start_idx)
            p_end = find_page_by_pos(end_idx - 1)
            chunk_id = build_chunk_id(source_sha256, p_start, p_end, local_chunk_index)

            chunks.append(
                {
                    "text": chunk_text,
                    "meta": {
                        "chunk_id": chunk_id,
                        "source_sha256": source_sha256,
                        "local_chunk_index": local_chunk_index,
                        "chunk_index": local_chunk_index,
                        "source": source_name,
                        "page": p_start,
                        "page_start": p_start,
                        "page_end": p_end,
                        "origin": "vector",
                    },
                }
            )

            local_chunk_index += 1

        if end_idx >= total_len:
            break

        start_idx = end_idx - chunk_char_overlap

        if chunk_char_overlap >= chunk_char_size:
            break

    return chunks
