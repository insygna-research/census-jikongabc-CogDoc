import pytest
from cogdoc.tools.embedder import Embedder
from cogdoc.tools.retriever.bm25_retriever import BM25Retriever
from cogdoc.tools.retriever.vector_retriever import VectorRetriever, _meta_from_stored


# BM25 新 native 接口返回 (doc_id, score)，测试只需要一个命中。
class _DummyBM25:
    # BM25 新 native 接口返回 (doc_id, score)，测试只需要一个命中。
    def score_topk(self, tokenized_query, top_n):
        return [(0, 1.0)]


# 模拟旧 Chroma 索引结果缺少稳定 chunk_id。
class _DummyVectorCollection:
    # 模拟旧 Chroma 索引结果缺少稳定 chunk_id。
    def query(self, query_embeddings, n_results):
        return {
            "documents": [["legacy text"]],
            "ids": [["legacy-id"]],
            "metadatas": [
                [
                    {
                        "chunk_index": 0,
                        "source": "legacy.pdf",
                        "page": 1,
                        "page_start": 1,
                        "page_end": 1,
                        "origin": "file",
                    }
                ]
            ],
            "distances": [[0.1]],
        }


# 验证 bm25 search rejects legacy docs without chunk id。
def test_bm25_search_rejects_legacy_docs_without_chunk_id():
    # 旧 BM25 索引不能现场补 chunk_id，必须提示重建。
    retriever = BM25Retriever.__new__(BM25Retriever)
    from threading import RLock

    retriever._lock = RLock()
    retriever.bm25 = _DummyBM25()
    retriever.doc_registry = [
        {
            "text": "legacy text",
            "meta": {
                "chunk_index": 0,
                "source": "legacy.pdf",
                "page": 1,
                "page_start": 1,
                "page_end": 1,
                "origin": "file",
            },
        }
    ]

    with pytest.raises(RuntimeError, match="missing stable chunk_id"):
        retriever.search("legacy", top_k=1)


# 验证 vector search rejects legacy docs without chunk id。
def test_vector_search_rejects_legacy_docs_without_chunk_id(monkeypatch):
    # 旧向量索引同样不能绕过稳定 chunk identity 契约。
    retriever = VectorRetriever.__new__(VectorRetriever)
    retriever.collection = _DummyVectorCollection()
    monkeypatch.setattr(Embedder, "embed_query", lambda query: [0.0])

    with pytest.raises(RuntimeError, match="missing stable chunk_id"):
        retriever.search("legacy", top_k=1)


# 验证 retriever metadata preserves chunk context。
def test_retriever_metadata_preserves_chunk_context():
    # 定位上下文属于 chunk 契约的一部分，BM25 registry 与向量元数据都必须保留。
    meta = {
        "chunk_id": "chunk:1",
        "source_sha256": "sha",
        "local_chunk_index": 0,
        "chunk_index": 3,
        "source": "paper.pdf",
        "page": 1,
        "page_start": 1,
        "page_end": 2,
        "origin": "vector",
        "context": "前文：背景\n后文：结论",
    }
    doc = {"text": "正文", "meta": meta}

    assert BM25Retriever._clean_doc(doc)["meta"]["context"] == meta["context"]
    assert _meta_from_stored(meta)["context"] == meta["context"]
