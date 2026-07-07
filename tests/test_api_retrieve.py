from contextlib import contextmanager

from cogdoc.api.routes import agent
from cogdoc.api.schemas import RetrieveRequest


# 验证调试检索合并派生知识命中。
def test_run_retrieve_merges_derived_knowledge(monkeypatch):
    raw_doc = {
        "text": "原文内容",
        "meta": {"chunk_id": "chunk-1", "source": "a.pdf"},
    }
    knowledge_doc = {
        "text": "补充知识",
        "meta": {
            "chunk_id": "knowledge:K1",
            "source_type": "derived_knowledge",
            "knowledge_id": "K1",
            "source": "knowledge:K1",
        },
        "retrieval": {"search_channel": "derived_knowledge"},
    }

    class Engine:
        def search(self, query, top_k):
            return [raw_doc]

    class KnowledgeRetriever:
        def search(self, kb_id, query, top_k):
            return [knowledge_doc]

    @contextmanager
    def lease(kb_id):
        yield

    monkeypatch.setattr(
        "cogdoc.graph.subgraphs.qa.RetrieverFactory.get_engine",
        lambda kb_id: Engine(),
    )
    monkeypatch.setattr("cogdoc.service.kb_readers.kb_read_lease", lease)
    monkeypatch.setattr(
        "cogdoc.tools.retriever.derived_knowledge.DerivedKnowledgeRetriever",
        lambda: KnowledgeRetriever(),
    )

    docs = agent._run_retrieve(
        RetrieveRequest(query="问题", doc_id="kb", top_k=3, rerank=False)
    )

    assert [doc["meta"]["chunk_id"] for doc in docs] == [
        "chunk-1",
        "knowledge:K1",
    ]


# 验证调试检索命中项保留派生知识字段。
def test_retrieve_hit_preserves_derived_knowledge_metadata():
    hit = agent._retrieve_hit(
        1,
        {
            "text": "补充知识",
            "meta": {
                "chunk_id": "knowledge:K1",
                "source_type": "derived_knowledge",
                "knowledge_id": "K1",
                "source": "knowledge:K1",
            },
            "retrieval": {
                "search_channel": "derived_knowledge",
                "matched_terms": ["问题"],
            },
        },
    )

    assert hit.source_type == "derived_knowledge"
    assert hit.knowledge_id == "K1"
    assert hit.retrieval["matched_terms"] == ["问题"]
