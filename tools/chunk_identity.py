DEFAULT_CHUNK_CHAR_SIZE = 600
DEFAULT_CHUNK_CHAR_OVERLAP = 60
MIN_CHUNK_CHARS = 30

# 切块边界逻辑变化时必须同步更新版本。
CHUNK_IDENTITY_BASE_VERSION = "source_sha256_page_span_local_v1"
CHUNK_IDENTITY_VERSION = (
    f"{CHUNK_IDENTITY_BASE_VERSION}"
    f"_cs{DEFAULT_CHUNK_CHAR_SIZE}"
    f"_ov{DEFAULT_CHUNK_CHAR_OVERLAP}"
    f"_min{MIN_CHUNK_CHARS}"
)


def build_chunk_id(
    source_sha256: str,
    page_start: int,
    page_end: int,
    local_chunk_index: int,
) -> str:
    # chunk_id 由文件内容和 chunk 局部位置共同决定。
    if not source_sha256:
        raise ValueError("source_sha256 is required to build a stable chunk_id")

    return f"sha256:{source_sha256}:p{int(page_start)}-p{int(page_end)}:c{int(local_chunk_index)}"
