from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


EVIDENCE_BLOCK_SEPARATOR = "\n\n"
EMPTY_EVIDENCE_CONTEXT = "（未检索到任何相关的参考本地知识库内容。）"


def _meta(doc: Mapping[str, Any]) -> Mapping[str, Any]:
    value = doc.get("meta")
    return value if isinstance(value, Mapping) else {}


def render_evidence_block(
    doc: Mapping[str, Any], *, text_override: str | None = None
) -> str:
    """Render one evidence block exactly as the QA generator sees it."""

    meta = _meta(doc)
    body = (
        str(doc.get("text") or "") if text_override is None else text_override
    ).strip()
    if meta.get("source_type") == "derived_knowledge":
        knowledge_id = meta.get("knowledge_id") or str(
            meta.get("chunk_id", "")
        ).replace("knowledge:", "")
        certainty = meta.get("certainty", "")
        related_source = meta.get("related_source", "")
        chunk_context = str(meta.get("context", "") or "").strip()
        if chunk_context:
            body = f"来源说明：\n{chunk_context}\n\n内容：\n{body}"
        return (
            f'<Knowledge knowledge_id="{knowledge_id}" certainty="{certainty}" '
            f'related_source="{related_source}">\n'
            f"{body}\n"
            "</Knowledge>"
        )

    source = meta.get("source", "未知文件")
    page = meta.get("page", 1)
    chunk_id = meta.get("chunk_id", meta.get("chunk_index", 0))
    section_path = str(meta.get("section_path", "") or "").strip()
    chunk_context = str(meta.get("context", "") or "").strip()
    if section_path:
        body = f"章节路径：{section_path}\n\n{body}"
    if chunk_context:
        body = f"定位上下文：\n{chunk_context}\n\n正文：\n{body}"
    return (
        f'<Document source="{source}" page="{page}" chunk_id="{chunk_id}">\n'
        f"{body}\n"
        "</Document>"
    )


def evidence_block_char_count(doc: Mapping[str, Any], text: str) -> int:
    return len(render_evidence_block(doc, text_override=text))


def render_evidence_context(docs: Sequence[Mapping[str, Any]]) -> str:
    if not docs:
        return EMPTY_EVIDENCE_CONTEXT
    return EVIDENCE_BLOCK_SEPARATOR.join(render_evidence_block(doc) for doc in docs)
