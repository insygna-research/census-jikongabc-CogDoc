from tools.document_loader import (
    list_sources,
    load_source_chunks,
    select_source_for_summary,
    sort_document_chunks,
)
from tools.retriever.bm25_retriever import BM25Retriever


def _doc(source: str, page: int, local_chunk_index: int, chunk_id: str = None) -> dict:
    return {
        "text": f"{source} p{page} c{local_chunk_index}",
        "meta": {
            "chunk_id": chunk_id or f"chunk:{source}:{local_chunk_index}",
            "source_sha256": f"sha:{source}",
            "local_chunk_index": local_chunk_index,
            "chunk_index": local_chunk_index,
            "source": source,
            "page": page,
            "page_start": page,
            "page_end": page,
            "origin": "file",
        },
    }


def test_sort_document_chunks_uses_source_page_and_local_index():
    # 文档 chunk 加载后必须恢复原文顺序。
    docs = [
        _doc("b.pdf", 1, 0),
        _doc("a.pdf", 2, 1),
        _doc("a.pdf", 1, 0),
    ]

    assert [doc["text"] for doc in sort_document_chunks(docs)] == [
        "a.pdf p1 c0",
        "a.pdf p2 c1",
        "b.pdf p1 c0",
    ]


def test_list_sources_is_stable_and_unique():
    # source 列表按文件名稳定排序并去重。
    docs = [_doc("b.pdf", 1, 0), _doc("a.pdf", 1, 0), _doc("a.pdf", 2, 1)]

    assert list_sources(docs) == ["a.pdf", "b.pdf"]


def test_load_source_chunks_filters_single_document():
    # 单文档加载不得混入其它 source。
    docs = [_doc("b.pdf", 1, 0), _doc("a.pdf", 2, 1), _doc("a.pdf", 1, 0)]

    loaded = load_source_chunks(docs, "a.pdf")

    assert [doc["text"] for doc in loaded] == ["a.pdf p1 c0", "a.pdf p2 c1"]


def test_select_source_for_summary_matches_full_name_or_stem():
    # 用户问题可按完整文件名或文件名主干定位目标文档。
    sources = ["paper-a.pdf", "paper-b.pdf"]

    assert select_source_for_summary("总结 paper-a.pdf", sources) == "paper-a.pdf"
    assert select_source_for_summary("总结 paper-b 的方法", sources) == "paper-b.pdf"


def test_select_source_for_summary_does_not_match_short_stem():
    # 短文件名主干不能用宽松子串匹配，避免多文档时静默选错。
    assert select_source_for_summary("总结 data-arch 的方法", ["a.pdf", "data-arch.pdf"]) == "data-arch.pdf"
    assert select_source_for_summary("总结一个 ai 比赛", ["ai.pdf", "paper.pdf"]) is None


def test_select_source_for_summary_uses_single_source_fallback():
    # 知识库只有一个 source 时无需用户显式点名。
    assert select_source_for_summary("总结这篇文档", ["only.pdf"]) == "only.pdf"


def test_select_source_for_summary_returns_none_when_ambiguous():
    # 多文档且无法匹配时交给 Summary 子图生成可操作错误。
    assert select_source_for_summary("总结这篇文档", ["a.pdf", "b.pdf"]) is None


def test_bm25_retriever_exposes_indexed_document_loader(tmp_path):
    # BM25 registry 可直接服务 Summary DocumentLoader。
    retriever = BM25Retriever("summary_loader_test", persist_directory = str(tmp_path))
    retriever.index([_doc("b.pdf", 1, 0), _doc("a.pdf", 2, 1), _doc("a.pdf", 1, 0)])

    assert retriever.list_sources() == ["a.pdf", "b.pdf"]
    assert [doc["text"] for doc in retriever.load_source_chunks("a.pdf")] == [
        "a.pdf p1 c0",
        "a.pdf p2 c1",
    ]
