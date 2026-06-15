from tools.manifest import manifests_match

DOCS = [{"name": "a.pdf", "size": 10, "sha256": "abc123"}]

def _manifest(doc_id, documents, doc_dir="/abs/machine/path"):
    return {"doc_id": doc_id, "doc_dir": doc_dir, "documents": documents}

def test_match_ignores_machine_specific_doc_dir():
    # 验证 doc_dir 路径变化不会导致索引误判过期。
    current = _manifest("kb", DOCS, doc_dir="/home/alice/CogDoc/测试论文")
    saved = _manifest("kb", DOCS, doc_dir="/home/bob/data/papers")
    assert manifests_match(current, saved) is True

def test_content_change_breaks_match():
    # 验证文件内容变化会触发 manifest 不匹配。
    current = _manifest("kb", DOCS)
    saved = _manifest("kb", [{"name": "a.pdf", "size": 10, "sha256": "DIFFERENT"}])
    assert manifests_match(current, saved) is False

def test_new_file_breaks_match():
    # 验证新增文件会触发 manifest 不匹配。
    saved = _manifest("kb", DOCS)
    current = _manifest("kb", DOCS + [{"name": "b.pdf", "size": 20, "sha256": "def456"}])
    assert manifests_match(current, saved) is False

def test_different_doc_id_does_not_match():
    # 验证不同知识库 ID 不会匹配。
    assert manifests_match(_manifest("kb", DOCS), _manifest("other", DOCS)) is False

def test_missing_saved_manifest_does_not_match():
    # 验证缺失或损坏的旧 manifest 会触发重建。
    assert manifests_match(_manifest("kb", DOCS), {}) is False
