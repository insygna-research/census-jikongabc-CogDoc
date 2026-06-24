import json
import os
from cogdoc.config.settings import get_settings
from cogdoc.tools.chunk_identity import CHUNK_IDENTITY_VERSION

# 测试和本地工具可覆盖该路径；默认从 COGDOC_DATA_DIR 派生。
MANIFEST_DIR = None


# 处理 manifest dir 相关逻辑。
def manifest_dir() -> str:
    return MANIFEST_DIR or get_settings().manifest_dir


# 处理 manifest path 相关逻辑。
def manifest_path(doc_id: str) -> str:
    return os.path.join(manifest_dir(), f"{doc_id}.json")


# 加载 load index manifest 相关逻辑。
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


# 保存 save index manifest 相关逻辑。
def save_index_manifest(manifest: dict) -> None:
    # 原子写：先写临时文件再 os.replace，进程中断不会留下半截 JSON。
    os.makedirs(manifest_dir(), exist_ok=True)
    path = manifest_path(manifest["doc_id"])
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


# 处理 manifests match 相关逻辑。
def manifests_match(current_manifest: dict, saved_manifest: dict) -> bool:
    # 分块身份版本或构建版本（解析器/分词器/嵌入模型）变化都必须触发重建。
    return (
        current_manifest.get("doc_id") == saved_manifest.get("doc_id")
        and current_manifest.get("chunk_identity_version")
        == saved_manifest.get("chunk_identity_version")
        and current_manifest.get("index_build_version")
        == saved_manifest.get("index_build_version")
        and current_manifest.get("documents", []) == saved_manifest.get("documents", [])
    )


# 处理 stamp chunk identity contract 相关逻辑。
def stamp_chunk_identity_contract(manifest: dict) -> dict:
    # 保存 manifest 前写入当前 chunk 身份契约版本。
    manifest["chunk_identity_version"] = CHUNK_IDENTITY_VERSION
    return manifest
