import pytest

from tools.embedder import Embedder
from tools.retriever.bm25_retriever import BM25Retriever
from tools.retriever.vector_retriever import VectorRetriever


class _DummyBM25:
    # 只模拟 BM25 返回一个命中。
    def get_scores(self, tokenized_query):
        return [1.0]


class _DummyVectorCollection:
    # 只模拟缺少 chunk_id 的旧 Chroma 结果。
    def query(self, query_embeddings, n_results):
        return {
            "documents": [["legacy text"]],
            "ids": [["legacy-id"]],
            "metadatas": [[{
                "chunk_index": 0,
                "source": "legacy.pdf",
                "page": 1,
                "page_start": 1,
                "page_end": 1,
                "origin": "file",
            }]],
            "distances": [[0.1]],
        }


def test_bm25_search_rejects_legacy_docs_without_chunk_id():
    # BM25 旧索引缺 chunk_id 时必须显式失败。
    retriever = BM25Retriever.__new__(BM25Retriever)
    retriever.bm25 = _DummyBM25()
    retriever.doc_registry = [{
        "text": "legacy text",
        "meta": {
            "chunk_index": 0,
            "source": "legacy.pdf",
            "page": 1,
            "page_start": 1,
            "page_end": 1,
            "origin": "file",
        },
    }]

    with pytest.raises(RuntimeError, match="missing stable chunk_id"):
        retriever.search("legacy", top_k=1)


def test_vector_search_rejects_legacy_docs_without_chunk_id(monkeypatch):
    # Vector 旧索引缺 chunk_id 时必须显式失败。
    retriever = VectorRetriever.__new__(VectorRetriever)
    retriever.collection = _DummyVectorCollection()
    monkeypatch.setattr(Embedder, "embed_query", lambda query: [0.0])

    with pytest.raises(RuntimeError, match="missing stable chunk_id"):
        retriever.search("legacy", top_k=1)
