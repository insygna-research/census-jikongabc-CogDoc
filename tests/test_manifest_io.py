import pytest
from cogdoc.tools import manifest


@pytest.fixture(autouse=True)
def manifest_dir(tmp_path, monkeypatch):
    # 避免读写真实 data/manifests。
    monkeypatch.setattr(manifest, "MANIFEST_DIR", str(tmp_path))
    return tmp_path


def test_missing_manifest_returns_empty():
    # 验证清单文件不存在时返回空 dict。
    assert manifest.load_index_manifest("nonexistent") == {}


def test_corrupt_manifest_returns_empty(manifest_dir):
    # 验证清单文件损坏时返回空 dict。
    (manifest_dir / "kb.json").write_text("{ not valid json", encoding="utf-8")
    assert manifest.load_index_manifest("kb") == {}


def test_save_then_load_roundtrip(manifest_dir):
    # 验证 manifest 保存后可以完整读回。
    data = {
        "doc_id": "kb",
        "doc_dir": "/abs/machine/path",
        "documents": [{"name": "a.pdf", "size": 10, "sha256": "abc123"}],
    }
    manifest.save_index_manifest(data)
    assert manifest.load_index_manifest("kb") == data


def test_save_creates_missing_manifest_dir(tmp_path, monkeypatch):
    # 验证保存时会自动创建缺失的 manifest 目录。
    target = tmp_path / "nested" / "manifests"
    monkeypatch.setattr(manifest, "MANIFEST_DIR", str(target))
    manifest.save_index_manifest({"doc_id": "kb", "documents": []})
    assert (target / "kb.json").exists()


def test_save_preserves_unicode_filenames(manifest_dir):
    # 验证中文文件名会原样落盘且可读回。
    data = {
        "doc_id": "kb",
        "documents": [{"name": "测试论文.pdf", "size": 1, "sha256": "x"}],
    }
    manifest.save_index_manifest(data)
    raw = (manifest_dir / "kb.json").read_text(encoding="utf-8")
    assert "测试论文.pdf" in raw
    assert manifest.load_index_manifest("kb") == data
