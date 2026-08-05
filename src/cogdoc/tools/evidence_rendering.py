from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from cogdoc.tools.citation_ledger import EVIDENCE_ID_PLACEHOLDER


EVIDENCE_BLOCK_SEPARATOR = "\n\n"
EMPTY_EVIDENCE_CONTEXT = "（未检索到任何相关的参考本地知识库内容。）"


def _meta(doc: Mapping[str, Any]) -> Mapping[str, Any]:
    value = doc.get("meta")
    return value if isinstance(value, Mapping) else {}


def render_evidence_block(
    doc: Mapping[str, Any],
    *,
    text_override: str | None = None,
    evidence_id_override: str | None = None,
) -> str:
    """Render one evidence block exactly as the QA generator sees it."""

    meta = _meta(doc)
    retrieval = doc.get("retrieval")
    retrieval = retrieval if isinstance(retrieval, Mapping) else {}
    evidence_id = (
        evidence_id_override
        if evidence_id_override is not None
        else str(retrieval.get("evidence_id") or "").strip()
    )
    evidence_id_attribute = f' evidence_id="{evidence_id}"' if evidence_id else ""
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
            f'related_source="{related_source}"{evidence_id_attribute}>\n'
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
        f'<Document source="{source}" page="{page}" chunk_id="{chunk_id}"'
        f"{evidence_id_attribute}>\n"
        f"{body}\n"
        "</Document>"
    )


def evidence_block_char_count(doc: Mapping[str, Any], text: str) -> int:
    # Evidence Pack 在最终展示顺序确定前尚不知道具体编号。定长占位符
    # 与 E001..E999 等长，因此选择阶段的字符预算仍与最终 prompt 精确一致。
    return len(
        render_evidence_block(
            doc,
            text_override=text,
            evidence_id_override=EVIDENCE_ID_PLACEHOLDER,
        )
    )


def render_evidence_context(docs: Sequence[Mapping[str, Any]]) -> str:
    if not docs:
        return EMPTY_EVIDENCE_CONTEXT
    return EVIDENCE_BLOCK_SEPARATOR.join(render_evidence_block(doc) for doc in docs)
