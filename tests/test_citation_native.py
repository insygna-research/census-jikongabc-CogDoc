from tests._native import require_rust_core

rust_core = require_rust_core("validate_citations_native")


# 构造测试用文档。
def _doc(source: str, page) -> dict:
    # 构造 native checker 所需的最小文档结构。
    return {"text": "", "meta": {"source": source, "page": page}}


VALID_DOCS = [_doc("a.pdf", 5), _doc("a.pdf", 6), _doc("b.pdf", 2)]


# 验证 native returns missing citations flag 场景。
def test_native_returns_missing_citations_flag():
    # 无引用时 native 应返回结构化 missing_citations。
    result = rust_core.validate_citations_native("没有引用的事实。", VALID_DOCS)

    assert result["is_valid"] is False
    assert result["missing_citations"] is True
    assert result["invalid_sources"] == []
    assert result["invalid_pages"] == []


# 验证 native returns invalid source items 场景。
def test_native_returns_invalid_source_items():
    # 错误文件名应进入 invalid_sources。
    result = rust_core.validate_citations_native("依据见[c.pdf:P5]。", VALID_DOCS)

    assert result["is_valid"] is False
    assert result["missing_citations"] is False
    assert result["invalid_sources"] == [{"source": "c.pdf", "page": 5}]
    assert result["invalid_pages"] == []


# 验证 native returns invalid page items with valid pages 场景。
def test_native_returns_invalid_page_items_with_valid_pages():
    # 错误页码应携带该文件本轮允许引用的页码。
    result = rust_core.validate_citations_native("依据见[a.pdf:P99]。", VALID_DOCS)

    assert result["is_valid"] is False
    assert result["invalid_sources"] == []
    assert result["invalid_pages"] == [
        {"source": "a.pdf", "page": 99, "valid_pages": [5, 6]}
    ]


# 验证 native accepts full width citation 场景。
def test_native_accepts_full_width_citation():
    # 全角括号和冒号保持与旧 Python 正则等价。
    result = rust_core.validate_citations_native(
        "方法见原文［a.pdf：P5］。", VALID_DOCS
    )

    assert result["is_valid"] is True
    assert result["missing_citations"] is False
    assert result["invalid_sources"] == []
    assert result["invalid_pages"] == []


# 验证 native accepts full width digit citation 场景。
def test_native_accepts_full_width_digit_citation():
    # 全角数字页码应归一为普通整数。
    result = rust_core.validate_citations_native(
        "方法见原文［a.pdf：P５］。", VALID_DOCS
    )

    assert result["is_valid"] is True
    assert result["missing_citations"] is False
    assert result["invalid_sources"] == []
    assert result["invalid_pages"] == []


# 验证 native rejects invalid full width digit page as page error 场景。
def test_native_rejects_invalid_full_width_digit_page_as_page_error():
    # 全角数字非法页码不能被误判为缺少引用。
    result = rust_core.validate_citations_native("依据见［a.pdf：P９９］。", VALID_DOCS)

    assert result["is_valid"] is False
    assert result["missing_citations"] is False
    assert result["invalid_sources"] == []
    assert result["invalid_pages"] == [
        {"source": "a.pdf", "page": 99, "valid_pages": [5, 6]}
    ]


# 验证 native accepts float page metadata 场景。
def test_native_accepts_float_page_metadata():
    # 原生 float 页码按旧 int(page) 语义处理。
    result = rust_core.validate_citations_native(
        "依据见[a.pdf:P5]。", [_doc("a.pdf", 5.0)]
    )

    assert result["is_valid"] is True
    assert result["invalid_sources"] == []
    assert result["invalid_pages"] == []


# 验证 native accepts float like page metadata 场景。
def test_native_accepts_float_like_page_metadata():
    # 支持实现 __float__ 的页码对象。
    class FloatLikePage:
        # 解析结果。
        def __float__(self):
            return 5.0

    result = rust_core.validate_citations_native(
        "依据见[a.pdf:P5]。",
        [_doc("a.pdf", FloatLikePage())],
    )

    assert result["is_valid"] is True
    assert result["invalid_sources"] == []
    assert result["invalid_pages"] == []
