import pytest
from cogdoc.tools.chunk_identity import build_chunk_id
from cogdoc.tools.chunk_identity import (
    DEFAULT_CHUNK_CONTEXT_CHARS,
    CHUNK_IDENTITY_VERSION,
    DEFAULT_CHUNK_CHAR_OVERLAP,
    DEFAULT_CHUNK_CHAR_SIZE,
    MIN_CHUNK_CHARS,
)
from cogdoc.tools.chunker import chunk_paper


SOURCE_SHA = "a" * 64


# 构造测试页。
def _page(page: int, text: str) -> dict:
    # 构造最小 ParsedPage 输入。
    return {
        "page": page,
        "source": "paper.pdf",
        "text": text,
        "is_ocr_fallback": False,
    }


# 验证 chunk id contract uses source hash name page span and local index。
def test_chunk_id_contract_uses_source_hash_name_page_span_and_local_index():
    # chunk_id 必须包含文件哈希、文件名、页跨度和局部序号。
    assert (
        build_chunk_id(SOURCE_SHA, "paper.pdf", 2, 3, 4)
        == f"sha256:{SOURCE_SHA}:src:paper.pdf:p2-p3:c4"
    )


# 验证 chunk id distinguishes same content different name。
def test_chunk_id_distinguishes_same_content_different_name():
    # 同内容不同名文档必须得到不同 chunk_id，否则删一个会误伤另一个。
    a = build_chunk_id(SOURCE_SHA, "a.pdf", 1, 1, 0)
    b = build_chunk_id(SOURCE_SHA, "b.pdf", 1, 1, 0)
    assert a != b


# 验证 chunk identity version includes chunking parameters。
def test_chunk_identity_version_includes_chunking_parameters():
    # 切块参数变化必须让 manifest 失效。
    assert f"cs{DEFAULT_CHUNK_CHAR_SIZE}" in CHUNK_IDENTITY_VERSION
    assert f"ov{DEFAULT_CHUNK_CHAR_OVERLAP}" in CHUNK_IDENTITY_VERSION
    assert f"min{MIN_CHUNK_CHARS}" in CHUNK_IDENTITY_VERSION
    assert f"ctx{DEFAULT_CHUNK_CONTEXT_CHARS}" in CHUNK_IDENTITY_VERSION


# 验证 chunk id requires source sha256。
def test_chunk_id_requires_source_sha256():
    # 没有文件哈希不能生成稳定身份。
    with pytest.raises(ValueError):
        build_chunk_id("", "paper.pdf", 1, 1, 0)


# 验证 chunk id requires source name。
def test_chunk_id_requires_source_name():
    # 没有文件名不能生成稳定身份。
    with pytest.raises(ValueError):
        build_chunk_id(SOURCE_SHA, "", 1, 1, 0)


# 验证 chunker requires source sha256。
def test_chunker_requires_source_sha256():
    # chunker 必须显式接收文件哈希。
    with pytest.raises(ValueError):
        chunk_paper([_page(1, "正文足够长，可以触发稳定身份契约校验。" * 4)])


# 验证 chunker writes stable chunk identity fields。
def test_chunker_writes_stable_chunk_identity_fields():
    # chunker 输出必须携带完整身份字段。
    text = "第一段用于测试稳定切块身份。" * 12
    chunks = chunk_paper(
        [_page(1, text)],
        source_sha256=SOURCE_SHA,
        chunk_char_size=120,
        chunk_char_overlap=0,
    )

    assert chunks
    first_meta = chunks[0]["meta"]
    assert first_meta["source_sha256"] == SOURCE_SHA
    assert first_meta["local_chunk_index"] == 0
    assert first_meta["chunk_id"] == build_chunk_id(
        SOURCE_SHA,
        first_meta["source"],
        first_meta["page_start"],
        first_meta["page_end"],
        first_meta["local_chunk_index"],
    )


# 验证 semantic chunker keeps sentence boundaries and overlap。
def test_semantic_chunker_keeps_sentence_boundaries_and_overlap():
    # 优先按句子组成 chunk，overlap 复用完整语义单元，避免从句中间开头。
    sentences = [
        "第一句说明背景和范围。",
        "第二句说明目标和对象。",
        "第三句说明流程和方法。",
        "第四句说明结果和产出。",
        "第五句说明限制和注意。",
    ]
    chunks = chunk_paper(
        [_page(1, "".join(sentences))],
        source_sha256=SOURCE_SHA,
        chunk_char_size=40,
        chunk_char_overlap=12,
        chunk_context_chars=20,
    )

    assert len(chunks) >= 2
    assert all(len(chunk["text"]) <= 40 for chunk in chunks)
    assert chunks[0]["text"].endswith("。")
    assert chunks[1]["text"].startswith(sentences[1])
    assert sentences[1] in chunks[0]["text"]
    assert "前文：" in chunks[1]["meta"]["context"]
    assert sentences[0] in chunks[1]["meta"]["context"]
    assert sentences[3] in chunks[0]["meta"]["context"]
    assert sentences[4] not in chunks[0]["meta"]["context"]


# 验证 semantic chunker caps combined semantic units。
def test_semantic_chunker_caps_combined_semantic_units():
    # 多个短语义单元组合时，最终 chunk 仍必须遵守 chunk_char_size 硬上限。
    text = "一二三四五六七八九十。一二三四五六七八九十。一二三四五六七八九十。甲乙丙丁戊己庚辛壬癸子丑申酉戌亥。"
    chunks = chunk_paper(
        [_page(1, text)],
        source_sha256=SOURCE_SHA,
        chunk_char_size=40,
        chunk_char_overlap=0,
    )

    assert chunks
    assert all(len(chunk["text"]) <= 40 for chunk in chunks)


# 验证 long text without punctuation still respects max size。
def test_long_text_without_punctuation_still_respects_max_size():
    # 没有段落/标点边界时退回固定窗口，但每块仍不超过 chunk_char_size。
    chunks = chunk_paper(
        [_page(1, "x" * 180)],
        source_sha256=SOURCE_SHA,
        chunk_char_size=50,
        chunk_char_overlap=0,
    )

    assert chunks
    assert all(len(chunk["text"]) <= 50 for chunk in chunks)
