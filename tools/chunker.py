import bisect
from typing import List
from graph.state import ParsedPage, RetrievedDoc

def chunk_paper(
    parsed_pages: List[ParsedPage],
    chunk_char_size: int = 600,
    chunk_char_overlap: int = 60
) -> List[RetrievedDoc]:

    # 空文档直接返回
    if not parsed_pages:
        return []

    source_name = parsed_pages[0]["source"]

    # 拼接后的全文
    global_text = ""

    # 记录每页在全文中的起始位置
    page_starts: List[int] = []

    # 对应页码
    page_nums: List[int] = []

    # 当前字符位置
    current_idx = 0

    for page in parsed_pages:
        p_text = page["text"]

        # 页面之间插入两个换行
        if global_text and p_text:
            global_text += "\n\n"
            current_idx += 2

        # 记录当前页起始位置
        page_starts.append(current_idx)
        page_nums.append(page["page"])

        global_text += p_text
        current_idx += len(p_text)

    chunks: List[RetrievedDoc] = []

    # 全文长度
    total_len = len(global_text)

    # 当前切片起点
    start_idx = 0

    # Chunk编号
    global_chunk_count = 0

    # 根据字符位置反查所在页码
    def find_page_by_pos(pos: int) -> int:
        idx = bisect.bisect_right(page_starts, pos) - 1
        return page_nums[max(0, idx)]

    # 全文为空直接返回
    if total_len == 0:
        return []

    while start_idx < total_len:

        # 当前Chunk终点
        end_idx = min(start_idx + chunk_char_size, total_len)

        # 按字符数切片
        chunk_text = global_text[start_idx:end_idx].strip()

        # 过滤过短文本
        if len(chunk_text) > 30:

            # 计算Chunk起止页码
            p_start = find_page_by_pos(start_idx)
            p_end = find_page_by_pos(end_idx - 1)

            chunks.append({
                "text": chunk_text,
                "meta": {
                    "chunk_index": global_chunk_count,
                    "source": source_name,
                    "page": p_start,
                    "page_start": p_start,
                    "page_end": p_end,
                    "origin": "vector"
                }
            })

            global_chunk_count += 1

        # 已到末尾
        if end_idx >= total_len:
            break

        # 重叠切分
        start_idx = end_idx - chunk_char_overlap

        # 防止死循环
        if chunk_char_overlap >= chunk_char_size:
            break

    return chunks