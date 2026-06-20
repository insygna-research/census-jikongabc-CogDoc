from agents.summary_generator import select_section_docs, tokenize_for_section


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


def test_select_section_docs_keeps_relevant_chunks_in_original_order():
    # 章节摘要只选择相关 chunk，并保持原文顺序。
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
        max_chunks = 2,
    )

    assert [doc["text"] for doc in selected] == ["模型 方法 架构", "方法 部署 流程"]


def test_tokenize_for_section_uses_word_level_chinese_tokens():
    # 中文分词必须产出词级 token，不能退回单字集合。
    tokens = tokenize_for_section("模型 方法 架构")

    assert "模型" in tokens
    assert "方法" in tokens
    assert "架构" in tokens
    assert "模" not in tokens


def test_select_section_docs_accepts_precomputed_doc_tokens():
    # 章节循环可复用预计算 token，避免重复分词所有 chunk。
    docs = [_doc("模型 方法 架构", 0), _doc("实验 指标 结果", 1)]
    precomputed = [(docs[0], {"方法", "架构"}), (docs[1], {"实验"})]

    selected = select_section_docs(
        docs,
        {"title": "方法与方案", "instruction": "概括方法和系统架构"},
        "总结文档",
        doc_tokens = precomputed,
        max_chunks = 1,
    )

    assert [doc["text"] for doc in selected] == ["模型 方法 架构"]


def test_select_section_docs_falls_back_to_leading_chunks_without_overlap():
    # 没有关键词重叠时保留前几个 chunk 作为稳定兜底。
    docs = [_doc("aaa", 0), _doc("bbb", 1), _doc("ccc", 2)]

    selected = select_section_docs(
        docs,
        {"title": "局限", "instruction": "概括限制"},
        "总结文档",
        max_chunks = 2,
    )

    assert [doc["text"] for doc in selected] == ["aaa", "bbb"]
