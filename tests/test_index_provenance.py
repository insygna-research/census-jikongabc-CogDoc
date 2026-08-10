import json

from cogdoc.service import index_provenance


def test_current_index_provenance_prefers_committed_generation(monkeypatch):
    class State:
        def __init__(self, kb_id):
            assert kb_id == "kb"

        def active(self):
            return {
                "id": "g-active",
                "index_build_version": "build-active",
                "chunk_identity_version": "chunk-active",
                "documents": [
                    {"name": "b.pdf", "sha256": "sha-b"},
                    {"name": "a.pdf", "sha256": "sha-a"},
                ],
            }

    monkeypatch.setattr(index_provenance, "KBState", State)
    monkeypatch.setattr(
        index_provenance,
        "load_index_manifest",
        lambda _kb_id: {
            "index_build_version": "build-manifest",
            "chunk_identity_version": "chunk-manifest",
            "documents": [{"name": "old.pdf", "sha256": "sha-old"}],
        },
    )

    assert index_provenance.current_index_provenance("kb") == {
        "index_generation": "g-active",
        "index_build_version": "build-active",
        "chunk_identity_version": "chunk-active",
        "source_versions": [
            {"source": "a.pdf", "sha256": "sha-a"},
            {"source": "b.pdf", "sha256": "sha-b"},
        ],
    }


def test_current_index_provenance_uses_manifest_for_legacy_generation(monkeypatch):
    class LegacyState:
        def __init__(self, _kb_id):
            pass

        def active(self):
            return {
                "id": "g-legacy",
                "index_build_version": "legacy-build",
                "documents": [{"name": "legacy.pdf", "sha256": "sha-legacy"}],
            }

    monkeypatch.setattr(index_provenance, "KBState", LegacyState)
    monkeypatch.setattr(
        index_provenance,
        "load_index_manifest",
        lambda _kb_id: {"chunk_identity_version": "legacy-chunks"},
    )

    assert index_provenance.current_index_provenance("legacy") == {
        "index_generation": "g-legacy",
        "index_build_version": "legacy-build",
        "chunk_identity_version": "legacy-chunks",
        "source_versions": [
            {"source": "legacy.pdf", "sha256": "sha-legacy"},
        ],
    }


def test_current_index_provenance_falls_back_and_never_breaks_chat(monkeypatch):
    class BrokenState:
        def __init__(self, _kb_id):
            pass

        def active(self):
            raise json.JSONDecodeError("bad", "", 0)

    monkeypatch.setattr(index_provenance, "KBState", BrokenState)
    monkeypatch.setattr(
        index_provenance,
        "load_index_manifest",
        lambda _kb_id: {
            "index_build_version": "legacy-build",
            "chunk_identity_version": "legacy-chunks",
            "documents": [{"name": "legacy.pdf", "sha256": "sha-legacy"}],
        },
    )

    assert index_provenance.current_index_provenance("legacy") == {
        "index_generation": "",
        "index_build_version": "legacy-build",
        "chunk_identity_version": "legacy-chunks",
        "source_versions": [
            {"source": "legacy.pdf", "sha256": "sha-legacy"},
        ],
    }
