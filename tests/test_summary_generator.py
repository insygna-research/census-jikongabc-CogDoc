from cogdoc.agents.summary_generator import (
    attach_section_citations,
    build_section_citations,
    is_no_evidence_summary,
    select_section_docs,
    tokenize_for_section,
)


# 构造测试用文档。
def _doc(text: str, idx: int) -> dict:
    return {
        "text": text,
        "meta": {
            "chunk_id": f"chunk:{idx}",
            "source": "a.pdf",
            "page": idx + 1,
            "page_start": idx + 1,
            "page_end": idx + 1,
            "local_chunk_index": idx,
            "chunk_index": idx,
        },
    }


# 验证 select section docs keeps relevant chunks in original order 场景。
def test_select_section_docs_keeps_relevant_chunks_in_original_order():
    # 相关 chunk 选择后保持原文顺序。
    docs = [
        _doc("报名 时间 入口", 0),
        _doc("模型 方法 架构", 1),
        _doc("实验 指标 结果", 2),
        _doc("方法 部署 流程", 3),
    ]

    selected = select_section_docs(
        docs,
        {"title": "方法与方案", "instruction": "概括方法和系统架构"},
        "总结文档",
        max_chunks=2,
    )

    assert [doc["text"] for doc in selected] == ["模型 方法 架构", "方法 部署 流程"]


# 验证 tokenize for section uses word level chinese tokens 场景。
def test_tokenize_for_section_uses_word_level_chinese_tokens():
    # 中文分词保持词级 token。
    tokens = tokenize_for_section("模型 方法 架构")

    assert "模型" in tokens
    assert "方法" in tokens
    assert "架构" in tokens
    assert "模" not in tokens


# 验证 select section docs accepts precomputed doc tokens 场景。
def test_select_section_docs_accepts_precomputed_doc_tokens():
    # 章节循环复用预计算 token。
    docs = [_doc("模型 方法 架构", 0), _doc("实验 指标 结果", 1)]
    precomputed = [(docs[0], {"方法", "架构"}), (docs[1], {"实验"})]

    selected = select_section_docs(
        docs,
        {"title": "方法与方案", "instruction": "概括方法和系统架构"},
        "总结文档",
        doc_tokens=precomputed,
        max_chunks=1,
    )

    assert [doc["text"] for doc in selected] == ["模型 方法 架构"]


# 验证 select section docs falls back to leading chunks without overlap 场景。
def test_select_section_docs_falls_back_to_leading_chunks_without_overlap():
    # 无关键词重叠时回退前几个 chunk。
    docs = [_doc("aaa", 0), _doc("bbb", 1), _doc("ccc", 2)]

    selected = select_section_docs(
        docs,
        {"title": "局限", "instruction": "概括限制"},
        "总结文档",
        max_chunks=2,
    )

    assert [doc["text"] for doc in selected] == ["aaa", "bbb"]


# 验证 build section citations uses unique doc metadata in order 场景。
def test_build_section_citations_uses_unique_doc_metadata_in_order():
    # 章节引用按 chunk 元数据顺序去重。
    docs = [_doc("上下文", 0), _doc("上下文", 0), _doc("上下文", 2)]

    result = build_section_citations(docs)

    assert result == "[a.pdf:P1][a.pdf:P3]"


# 验证 attach section citations appends valid doc citations 场景。
def test_attach_section_citations_appends_valid_doc_citations():
    # 漏写引用时绑定章节上下文引用。
    result = attach_section_citations("这是摘要内容。", [_doc("上下文", 0)])

    assert result == "这是摘要内容。[a.pdf:P1]"


# 验证 attach section citations replaces model generated citations 场景。
def test_attach_section_citations_replaces_model_generated_citations():
    # 模型自写引用统一替换为程序引用。
    result = attach_section_citations(
        "这是摘要内容。[fake.pdf:P99]", [_doc("上下文", 0)]
    )

    assert result == "这是摘要内容。[a.pdf:P1]"


# 验证 attach section citations keeps no evidence marker plain 场景。
def test_attach_section_citations_keeps_no_evidence_marker_plain():
    # 无依据标记不补引用。
    result = attach_section_citations("文档中未明确说明", [_doc("上下文", 0)])

    assert result == "文档中未明确说明"


# 验证 attach section citations keeps no evidence variants plain 场景。
def test_attach_section_citations_keeps_no_evidence_variants_plain():
    # 无依据变体不补引用。
    docs = [_doc("上下文", 0)]

    assert attach_section_citations("文档中未明确说明。", docs) == "文档中未明确说明。"
    assert attach_section_citations("文档中未明确说明！", docs) == "文档中未明确说明！"
    assert (
        attach_section_citations("文档中未明确说明相关内容。", docs)
        == "文档中未明确说明相关内容。"
    )


# 验证 is no evidence summary only matches no evidence prefix 场景。
def test_is_no_evidence_summary_only_matches_no_evidence_prefix():
    # 非前缀无依据表述仍可绑定引用。
    assert is_no_evidence_summary("文档中未明确说明。")
    assert not is_no_evidence_summary("该文档未明确说明报名费用。")


# 验证 hedge summary with real fact after marker is not no evidence 场景。
def test_hedge_summary_with_real_fact_after_marker_is_not_no_evidence():
    # 无依据前缀后有转折事实时必须绑定引用。
    hedge = "文档中未明确说明评测指标，但要求6月1日前提交作品。"

    assert not is_no_evidence_summary(hedge)
    assert attach_section_citations(hedge, [_doc("上下文", 0)]) == f"{hedge}[a.pdf:P1]"


# 验证 no evidence marker then separate factual sentence is cited 场景。
def test_no_evidence_marker_then_separate_factual_sentence_is_cited():
    # 无依据前缀后另起事实句时必须绑定引用。
    mixed = "文档中未明确说明评测指标。提交作品需在6月1日前完成。"

    assert not is_no_evidence_summary(mixed)
    assert attach_section_citations(mixed, [_doc("上下文", 0)]) == f"{mixed}[a.pdf:P1]"


# 验证 pure no evidence expansion stays plain 场景。
def test_pure_no_evidence_expansion_stays_plain():
    # 纯无依据扩写不补引用。
    assert is_no_evidence_summary("文档中未明确说明相关内容。")
    assert (
        attach_section_citations("文档中未明确说明相关内容。", [_doc("上下文", 0)])
        == "文档中未明确说明相关内容。"
    )
