from agents.citation_validator import CitationValidatorAgent

def _doc(source: str, page: int) -> dict:
    return {"text": "", "meta": {"source": source, "page": page}}

VALID_DOCS = [_doc("a.pdf", 5), _doc("a.pdf", 6), _doc("b.pdf", 2)]

def validate(answer, docs=VALID_DOCS):
    return CitationValidatorAgent.validate_citations(answer, docs)

def test_empty_answer_passes():
    # 验证空答案直接通过校验。
    result = validate("")
    assert result["is_valid"] is True
    assert result["critique"] == ""

def test_no_valid_docs_passes_without_citation():
    # 验证无可用文档时不强制要求引用。
    result = validate("随便一句没有引用的话", docs=[])
    assert result["is_valid"] is True

def test_fallback_marker_passes():
    # 验证未找到依据的兜底回答直接通过。
    answer = "在所提供的参考资料中未找到与该问题相关的内容。"
    result = validate(answer)
    assert result["is_valid"] is True
    assert result["critique"] == ""

def test_doc_without_page_is_ignored_in_registry():
    # 验证缺少页码的文档不会进入可引用页码表。
    docs = [{"text": "", "meta": {"source": "a.pdf"}}]
    result = validate("一句没有引用的事实陈述", docs=docs)
    assert result["is_valid"] is True

def test_valid_half_width_citation():
    # 验证半角引用标签可通过校验。
    result = validate("模型在该任务上表现最好[a.pdf:P5]。")
    assert result["is_valid"] is True
    assert result["critique"] == ""

def test_valid_full_width_citation():
    # 验证全角引用标签可通过校验。
    result = validate("方法见原文［a.pdf：P5］。")
    assert result["is_valid"] is True

def test_valid_lowercase_p_and_spaces():
    # 验证小写页码标记和标签内空格可通过校验。
    result = validate("结论如此 [ a.pdf : p6 ] 。")
    assert result["is_valid"] is True

def test_missing_citation_is_rejected():
    # 验证有事实陈述但无引用时会被拒绝。
    result = validate("模型在该任务上达到了最优效果，但没有任何引用。")
    assert result["is_valid"] is False
    assert "未包含任何引用标签" in result["critique"]

def test_wrong_source_is_rejected():
    # 验证引用不存在的文件名会被拒绝。
    result = validate("依据见[c.pdf:P5]。")
    assert result["is_valid"] is False
    assert "文件名错误" in result["critique"]
    assert "c.pdf" in result["critique"]

def test_wrong_page_is_rejected():
    # 验证引用未召回页码会被拒绝。
    result = validate("依据见[a.pdf:P99]。")
    assert result["is_valid"] is False
    assert "页码错误" in result["critique"]
    assert "P5" in result["critique"] and "P6" in result["critique"]

def test_mixed_valid_and_invalid_still_rejected():
    # 验证答案中任一非法引用都会导致整体拒绝。
    result = validate("正确依据[a.pdf:P5]，错误依据[a.pdf:P99]。")
    assert result["is_valid"] is False
    assert "页码错误" in result["critique"]
