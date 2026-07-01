from cogdoc.graph.state import RetrievedDoc


# 构造检索索引用文本。
def retrieval_text(doc: RetrievedDoc) -> str:
    # 定位上下文只参与召回和向量化，原始正文仍作为返回内容。
    context = str(doc.get("meta", {}).get("context", "") or "").strip()
    text = str(doc.get("text", "") or "").strip()
    if context and text:
        return f"{context}\n\n{text}"
    return context or text
