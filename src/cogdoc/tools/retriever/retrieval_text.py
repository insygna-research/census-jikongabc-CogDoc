from cogdoc.graph.state import RetrievedDoc


# 构造检索索引用文本。
def retrieval_text(doc: RetrievedDoc) -> str:
    # 来源、章节路径和定位上下文只参与召回/向量化，原始正文仍作为返回内容。
    meta = doc.get("meta", {})
    source = str(meta.get("source", "") or "").strip()
    source_type = str(meta.get("source_type", "document") or "document")
    section_path = str(meta.get("section_path", "") or "").strip()
    context = str(meta.get("context", "") or "").strip()
    text = str(doc.get("text", "") or "").strip()
    location = []
    if source and source_type != "derived_knowledge":
        location.append(f"来源：{source}")
    if section_path:
        location.append(f"章节：{section_path}")
    return "\n\n".join(part for part in ("\n".join(location), context, text) if part)
