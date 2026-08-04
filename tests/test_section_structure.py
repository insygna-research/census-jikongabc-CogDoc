from cogdoc.tools.section_structure import SectionSpan, detect_section_spans


def test_empty_and_whitespace_input_have_no_spans():
    assert detect_section_spans("") == []
    assert detect_section_spans(" \n\r\n\t") == []


def test_unstructured_text_is_retained_as_preamble():
    text = "This document has no explicit section headings.\nIt remains one span."

    assert detect_section_spans(text) == [
        SectionSpan(start=0, end=len(text), title="", path=(), level=0, ordinal=0)
    ]


def test_detects_preamble_and_common_english_titles_with_exact_offsets():
    text = (
        "A Study of Reliable Retrieval\nAlice and Bob\n\n"
        "Abstract\nA concise summary.\n\n"
        "Introduction\nThe problem context.\n\n"
        "Results and Discussion\nThe measured result."
    )

    spans = detect_section_spans(text)

    assert [(span.title, span.level, span.path) for span in spans] == [
        ("", 0, ()),
        ("Abstract", 1, ("Abstract",)),
        ("Introduction", 1, ("Introduction",)),
        ("Results and Discussion", 1, ("Results and Discussion",)),
    ]
    assert spans[0].start == 0
    assert spans[0].end == text.index("Abstract")
    assert spans[1].start == text.index("Abstract")
    assert spans[1].end == text.index("Introduction")
    assert spans[-1].end == len(text)
    assert [span.ordinal for span in spans] == list(range(len(spans)))


def test_decimal_numbering_builds_parent_and_sibling_paths():
    text = (
        "1 Introduction\nOpening text.\n\n"
        "1.1 Motivation\nMotivation text.\n\n"
        "1.2 System Design\nDesign text.\n\n"
        "2 Methods\nMethod text."
    )

    spans = detect_section_spans(text)

    assert [(span.title, span.level, span.path) for span in spans] == [
        ("Introduction", 1, ("Introduction",)),
        ("Motivation", 2, ("Introduction", "Motivation")),
        ("System Design", 2, ("Introduction", "System Design")),
        ("Methods", 1, ("Methods",)),
    ]
    assert all(text[span.start : span.end].strip() for span in spans)


def test_detects_chinese_ordinal_and_enumerated_headings():
    text = (
        "文档说明\n\n"
        "第一章 总则\n这里是总则。\n\n"
        "第一节 适用范围\n这里是范围。\n\n"
        "（一）模型设计\n这里是设计。\n\n"
        "第二章 实施方案\n这里是方案。"
    )

    spans = detect_section_spans(text)

    assert [(span.title, span.level, span.path) for span in spans] == [
        ("", 0, ()),
        ("总则", 1, ("总则",)),
        ("适用范围", 2, ("总则", "适用范围")),
        ("模型设计", 2, ("总则", "模型设计")),
        ("实施方案", 1, ("实施方案",)),
    ]


def test_detects_common_chinese_and_isolated_short_titles():
    text = (
        "摘要\n简要说明。\n\n"
        "系统架构\n\n"
        "架构正文。\n\n"
        "Model Evaluation\n\n"
        "Evaluation body.\n\n"
        "结论\n结论正文。"
    )

    spans = detect_section_spans(text)

    assert [span.title for span in spans] == [
        "摘要",
        "系统架构",
        "Model Evaluation",
        "结论",
    ]
    assert all(span.level == 1 for span in spans)


def test_ordinary_sentences_and_list_items_are_not_headings():
    text = (
        "This is an ordinary sentence without terminal punctuation\n\n"
        "这是一个普通句子\n\n"
        "- Bullet Item\n\n"
        "1. Install the package.\n"
        "2. Restart the service.\n\n"
        "Section content continues on the next line\n\n"
        "Chapter discusses ordinary prose\n\n"
        "The introduction appears inside this sentence."
    )

    spans = detect_section_spans(text)

    assert spans == [
        SectionSpan(start=0, end=len(text), title="", path=(), level=0, ordinal=0)
    ]


def test_markdown_level_controls_hierarchy():
    text = (
        "# Overview\nBody.\n\n"
        "## Data Pipeline\nBody.\n\n"
        "### Evaluation Setup\nBody.\n\n"
        "# Conclusion\nBody."
    )

    spans = detect_section_spans(text)

    assert [(span.title, span.level, span.path) for span in spans] == [
        ("Overview", 1, ("Overview",)),
        ("Data Pipeline", 2, ("Overview", "Data Pipeline")),
        (
            "Evaluation Setup",
            3,
            ("Overview", "Data Pipeline", "Evaluation Setup"),
        ),
        ("Conclusion", 1, ("Conclusion",)),
    ]


def test_crlf_input_preserves_source_character_offsets():
    text = "Cover\r\n\r\nAbstract\r\nSummary.\r\n\r\nMethods\r\nDetails."

    spans = detect_section_spans(text)

    assert spans[1].start == text.index("Abstract")
    assert spans[1].end == text.index("Methods")
    assert text[spans[2].start :].startswith("Methods")


def test_number_like_year_is_not_treated_as_a_section_number():
    text = "2024 Annual Report\n\nThis line is cover material."

    spans = detect_section_spans(text)

    assert len(spans) == 1
    assert spans[0].level == 0


def test_title_case_author_line_remains_in_preamble():
    text = "A Reliable Retrieval Study\n\nAlice and Bob\n\nAbstract\nA concise summary."

    spans = detect_section_spans(text)

    assert [span.title for span in spans] == ["", "Abstract"]
    assert spans[0].end == text.index("Abstract")
    assert "Alice and Bob" in text[spans[0].start : spans[0].end]
