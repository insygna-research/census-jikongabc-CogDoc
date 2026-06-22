import json
import os
from tools.chunk_identity import CHUNK_IDENTITY_VERSION

MANIFEST_DIR = os.path.join("data", "manifests")


def manifest_path(doc_id: str) -> str:
    return os.path.join(MANIFEST_DIR, f"{doc_id}.json")


def load_index_manifest(doc_id: str) -> dict:
    # 读取失败按无 manifest 处理，让上层重建索引。
    path = manifest_path(doc_id)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_index_manifest(manifest: dict) -> None:
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    with open(manifest_path(manifest["doc_id"]), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def manifests_match(current_manifest: dict, saved_manifest: dict) -> bool:
    # chunk_identity_version 变化必须触发重建。
    return (
        current_manifest.get("doc_id") == saved_manifest.get("doc_id")
        and current_manifest.get("chunk_identity_version")
        == saved_manifest.get("chunk_identity_version")
        and current_manifest.get("documents", []) == saved_manifest.get("documents", [])
    )


def stamp_chunk_identity_contract(manifest: dict) -> dict:
    # 保存 manifest 前写入当前 chunk 身份契约版本。
    manifest["chunk_identity_version"] = CHUNK_IDENTITY_VERSION
    return manifest
