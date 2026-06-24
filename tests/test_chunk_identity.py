import pytest
from cogdoc.tools.chunk_identity import build_chunk_id
from cogdoc.tools.chunk_identity import (
    CHUNK_IDENTITY_VERSION,
    DEFAULT_CHUNK_CHAR_OVERLAP,
    DEFAULT_CHUNK_CHAR_SIZE,
    MIN_CHUNK_CHARS,
)
from cogdoc.tools.chunker import chunk_paper


SOURCE_SHA = "a" * 64


# 处理 page 相关逻辑。
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
