import pytest

from tools.chunk_identity import build_chunk_id
from tools.chunk_identity import (
    CHUNK_IDENTITY_VERSION,
    DEFAULT_CHUNK_CHAR_OVERLAP,
    DEFAULT_CHUNK_CHAR_SIZE,
    MIN_CHUNK_CHARS,
)
from tools.chunker import chunk_paper


SOURCE_SHA = "a" * 64


def _page(page: int, text: str) -> dict:
    # 构造最小 ParsedPage 输入。
    return {
        "page": page,
        "source": "paper.pdf",
        "text": text,
        "is_ocr_fallback": False,
    }


def test_chunk_id_contract_uses_source_hash_page_span_and_local_index():
    # chunk_id 必须包含文件哈希、页跨度和局部序号。
    assert build_chunk_id(SOURCE_SHA, 2, 3, 4) == f"sha256:{SOURCE_SHA}:p2-p3:c4"


def test_chunk_identity_version_includes_chunking_parameters():
    # 切块参数变化必须让 manifest 失效。
    assert f"cs{DEFAULT_CHUNK_CHAR_SIZE}" in CHUNK_IDENTITY_VERSION
    assert f"ov{DEFAULT_CHUNK_CHAR_OVERLAP}" in CHUNK_IDENTITY_VERSION
    assert f"min{MIN_CHUNK_CHARS}" in CHUNK_IDENTITY_VERSION


def test_chunk_id_requires_source_sha256():
    # 没有文件哈希不能生成稳定身份。
    with pytest.raises(ValueError):
        build_chunk_id("", 1, 1, 0)


def test_chunker_requires_source_sha256():
    # chunker 必须显式接收文件哈希。
    with pytest.raises(ValueError):
        chunk_paper([_page(1, "正文足够长，可以触发稳定身份契约校验。" * 4)])


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
        first_meta["page_start"],
        first_meta["page_end"],
        first_meta["local_chunk_index"],
    )
