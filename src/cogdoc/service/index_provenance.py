from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cogdoc.service.kb_state import KBState
from cogdoc.tools.manifest import load_index_manifest


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _source_versions(*document_sets: Any) -> list[dict[str, str]]:
    """Return a deterministic source/SHA snapshot from the first usable set."""

    for raw_documents in document_sets:
        if not isinstance(raw_documents, list):
            continue
        versions: dict[str, str] = {}
        for raw in raw_documents:
            document = _mapping(raw)
            source = str(document.get("name") or document.get("source") or "").strip()
            sha256 = str(
                document.get("sha256") or document.get("source_sha256") or ""
            ).strip()
            if source and sha256:
                versions[source] = sha256
        if versions:
            return [
                {"source": source, "sha256": versions[source]}
                for source in sorted(versions)
            ]
    return []


def current_index_provenance(kb_id: str) -> dict[str, Any]:
    """Capture the committed index identity used by an evaluation observation.

    Generation state is the transactional authority.  Current generations persist
    the chunk-identity contract beside the build version; the manifest remains a
    compatibility fallback for legacy indexes.  Observability must never make chat
    unavailable, so corrupt or absent provenance degrades to explicit empty fields.
    """

    active: Mapping[str, Any] = {}
    manifest: Mapping[str, Any] = {}
    try:
        active = _mapping(KBState(kb_id).active())
    except Exception:
        active = {}
    try:
        manifest = _mapping(load_index_manifest(kb_id))
    except Exception:
        manifest = {}
    return {
        "index_generation": str(active.get("id") or ""),
        "index_build_version": str(
            active.get("index_build_version")
            or manifest.get("index_build_version")
            or ""
        ),
        "chunk_identity_version": str(
            active.get("chunk_identity_version")
            or manifest.get("chunk_identity_version")
            or ""
        ),
        "source_versions": _source_versions(
            active.get("documents"), manifest.get("documents")
        ),
    }
