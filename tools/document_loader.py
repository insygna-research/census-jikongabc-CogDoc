import os
import re
from typing import Iterable, List, Optional
from graph.state import RetrievedDoc


def sort_document_chunks(chunks: Iterable[RetrievedDoc]) -> List[RetrievedDoc]:
    # 单文档摘要必须按原文顺序读取 chunk。
    return sorted(
        chunks,
        key = lambda doc: (
            str(doc.get("meta", {}).get("source", "")),
            int(doc.get("meta", {}).get("page_start", doc.get("meta", {}).get("page", 0))),
            int(doc.get("meta", {}).get("page_end", doc.get("meta", {}).get("page", 0))),
            int(doc.get("meta", {}).get("local_chunk_index", doc.get("meta", {}).get("chunk_index", 0))),
            str(doc.get("meta", {}).get("chunk_id", "")),
        ),
    )


def list_sources(chunks: Iterable[RetrievedDoc]) -> List[str]:
    # source 是用户选择单文档摘要目标的最小稳定标识。
    sources = {
        str(doc.get("meta", {}).get("source", ""))
        for doc in chunks
        if doc.get("meta", {}).get("source")
    }
    return sorted(sources)


def load_source_chunks(chunks: Iterable[RetrievedDoc], source: str) -> List[RetrievedDoc]:
    # 只加载指定 source 的 chunk，避免跨文档摘要混写。
    return sort_document_chunks(
        doc for doc in chunks if str(doc.get("meta", {}).get("source", "")) == source
    )


def select_source_for_summary(query: str, sources: List[str]) -> Optional[str]:
    # 单文档摘要优先从用户问题中匹配完整文件名或文件名主干。
    if not sources:
        return None
    if len(sources) == 1:
        return sources[0]

    query_lower = query.lower()
    for source in sources:
        if source.lower() in query_lower:
            return source

    for source in sources:
        stem = os.path.splitext(source)[0].lower()
        if len(stem) >= 3 and re.search(rf"(?<![a-z0-9_\-]){re.escape(stem)}(?![a-z0-9_\-])", query_lower):
            return source

    return None
