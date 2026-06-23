import os
from dataclasses import dataclass, field
from graph.subgraphs.qa import RetrieverFactory
from tools.chunker import chunk_paper
from tools.manifest import (
    manifest_path,
    save_index_manifest,
    stamp_chunk_identity_contract,
)
from tools.parser import smart_parse
from tools.rust_core_loader import ensure_rust_core


@dataclass(frozen=True)
class IngestDocResult:
    name: str
    chunk_count: int


@dataclass(frozen=True)
class IngestResult:
    kb_id: str
    document_count: int
    chunk_count: int
    documents: list[IngestDocResult] = field(default_factory=list)


def list_pdf_files(source_dir: str) -> list[str]:
    if not os.path.isdir(source_dir):
        return []
    return sorted(f for f in os.listdir(source_dir) if f.lower().endswith(".pdf"))


def _invalidate_engine_cache(kb_id: str) -> None:
    # 只失效本库引擎，否则 /chat 命中旧引擎读旧索引；不波及其他 kb。
    RetrieverFactory.invalidate(kb_id)


def delete_kb_index(kb_id: str) -> None:
    # 删库索引：清向量/BM25 + 删 manifest + 失效引擎缓存。
    RetrieverFactory.get_engine(kb_id).clear()
    manifest_file = manifest_path(kb_id)
    if os.path.exists(manifest_file):
        os.remove(manifest_file)
    _invalidate_engine_cache(kb_id)


def build_kb_index(kb_id: str, source_dir: str) -> IngestResult:
    engine = RetrieverFactory.get_engine(kb_id)
    pdf_files = list_pdf_files(source_dir)
    if not pdf_files:
        # 空库：HybridRetriever.index 对空 chunk 直接返回，必须显式清索引。
        engine.clear()
        _invalidate_engine_cache(kb_id)
        return IngestResult(kb_id, 0, 0, [])

    rust_core = ensure_rust_core("scan_pdf_manifest_native")
    abs_dir = os.path.abspath(source_dir)
    manifest = stamp_chunk_identity_contract(
        rust_core.scan_pdf_manifest_native(kb_id, abs_dir)
    )
    source_hash_by_name = {
        doc["name"]: doc["sha256"] for doc in manifest.get("documents", [])
    }

    all_chunks = []
    next_chunk_index = 0
    doc_results = []
    for pdf in pdf_files:
        pages = smart_parse(os.path.join(source_dir, pdf))
        chunks = chunk_paper(pages, source_sha256=source_hash_by_name[pdf])
        for chunk in chunks:
            # chunk_index 仅用于展示，chunk_id 才是身份键。
            chunk["meta"]["chunk_index"] = next_chunk_index
            next_chunk_index += 1
        all_chunks.extend(chunks)
        doc_results.append(IngestDocResult(pdf, len(chunks)))

    if all_chunks:
        engine.index(all_chunks)
    else:
        # 有 PDF 但没抽出任何 chunk（扫描件/空 PDF）：index([]) 会早退不清，必须显式清旧索引。
        engine.clear()
    save_index_manifest(manifest)
    _invalidate_engine_cache(kb_id)
    return IngestResult(kb_id, len(pdf_files), len(all_chunks), doc_results)
