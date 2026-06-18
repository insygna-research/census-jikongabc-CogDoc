from tools.chunk_identity import CHUNK_IDENTITY_VERSION
from tools.manifest import manifests_match

DOCS = [{"name": "a.pdf", "size": 10, "sha256": "abc123"}]

def _manifest(
    doc_id,
    documents,
    doc_dir="/abs/machine/path",
    chunk_identity_version=CHUNK_IDENTITY_VERSION,
):
    # 构造只包含匹配逻辑所需字段的 manifest。
    return {
        "doc_id": doc_id,
        "doc_dir": doc_dir,
        "chunk_identity_version": chunk_identity_version,
        "documents": documents,
    }

def test_match_ignores_machine_specific_doc_dir():
    # doc_dir 是机器相关路径，不参与匹配。
    current = _manifest("kb", DOCS, doc_dir="/home/alice/CogDoc/测试论文")
    saved = _manifest("kb", DOCS, doc_dir="/home/bob/data/papers")
    assert manifests_match(current, saved) is True

def test_content_change_breaks_match():
    # 文件指纹变化必须触发重建。
    current = _manifest("kb", DOCS)
    saved = _manifest("kb", [{"name": "a.pdf", "size": 10, "sha256": "DIFFERENT"}])
    assert manifests_match(current, saved) is False

def test_new_file_breaks_match():
    # 文件集合变化必须触发重建。
    saved = _manifest("kb", DOCS)
    current = _manifest("kb", DOCS + [{"name": "b.pdf", "size": 20, "sha256": "def456"}])
    assert manifests_match(current, saved) is False

def test_different_doc_id_does_not_match():
    # 不同知识库不能复用 manifest。
    assert manifests_match(_manifest("kb", DOCS), _manifest("other", DOCS)) is False

def test_different_chunk_identity_version_does_not_match():
    # chunk 身份契约变化必须触发重建。
    assert manifests_match(_manifest("kb", DOCS), _manifest("kb", DOCS, chunk_identity_version="old")) is False

def test_missing_saved_manifest_does_not_match():
    # 缺失旧 manifest 时必须重建。
    assert manifests_match(_manifest("kb", DOCS), {}) is False
