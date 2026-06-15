import json
import os

MANIFEST_DIR = os.path.join("data", "manifests")

def manifest_path(doc_id: str) -> str:
    return os.path.join(MANIFEST_DIR, f"{doc_id}.json")

def load_index_manifest(doc_id: str) -> dict:
    path = manifest_path(doc_id)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_index_manifest(manifest: dict) -> None:
    os.makedirs(MANIFEST_DIR, exist_ok = True)
    with open(manifest_path(manifest["doc_id"]), "w", encoding = "utf-8") as f:
        json.dump(manifest, f, ensure_ascii = False, indent = 2)

def manifests_match(current_manifest: dict, saved_manifest: dict) -> bool:
    return (
        current_manifest.get("doc_id") == saved_manifest.get("doc_id")
        and current_manifest.get("documents", []) == saved_manifest.get("documents", [])
    )
