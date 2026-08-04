from contextlib import contextmanager
from types import SimpleNamespace

from cogdoc.api.routes import agent
from cogdoc.api.schemas import RetrieveRequest


class _NoRetrievalFeedback:
    def boosts_for_query(self, kb_id, query):
        return {}


def _runtime(knowledge_retriever, retrieval_feedback_store=None):
    return SimpleNamespace(
        derived_knowledge_retriever=knowledge_retriever,
        retrieval_feedback_store=(retrieval_feedback_store or _NoRetrievalFeedback()),
    )


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
    runtime = _runtime(KnowledgeRetriever())

    docs = agent._run_retrieve(
        RetrieveRequest(query="问题", doc_id="kb", top_k=3, rerank=False),
        state_runtime=runtime,
    )

    assert [doc["meta"]["chunk_id"] for doc in docs] == [
        "chunk-1",
        "knowledge:K1",
    ]


# 验证调试检索不会在多个应用运行时之间串用派生知识。
def test_run_retrieve_isolates_injected_state_runtimes(monkeypatch):
    class Engine:
        def search(self, query, top_k):
            return []

    class KnowledgeRetriever:
        def __init__(self, knowledge_id):
            self.knowledge_id = knowledge_id

        def search(self, kb_id, query, top_k):
            return [
                {
                    "text": self.knowledge_id,
                    "meta": {
                        "chunk_id": f"knowledge:{self.knowledge_id}",
                        "source_type": "derived_knowledge",
                        "knowledge_id": self.knowledge_id,
                        "source": f"knowledge:{self.knowledge_id}",
                    },
                }
            ]

    @contextmanager
    def lease(kb_id):
        yield

    monkeypatch.setattr(
        "cogdoc.graph.subgraphs.qa.RetrieverFactory.get_engine",
        lambda kb_id: Engine(),
    )
    monkeypatch.setattr("cogdoc.service.kb_readers.kb_read_lease", lease)
    body = RetrieveRequest(query="问题", doc_id="kb", top_k=3, rerank=False)

    docs_a = agent._run_retrieve(
        body,
        state_runtime=_runtime(KnowledgeRetriever("A")),
    )
    docs_b = agent._run_retrieve(
        body,
        state_runtime=_runtime(KnowledgeRetriever("B")),
    )

    assert [doc["meta"]["knowledge_id"] for doc in docs_a] == ["A"]
    assert [doc["meta"]["knowledge_id"] for doc in docs_b] == ["B"]


# 验证线上检索 helper 使用同一 runtime 的反馈存储调权。
def test_run_retrieve_applies_runtime_retrieval_feedback(monkeypatch):
    class Engine:
        def search(self, query, top_k):
            return [
                {"text": "A", "meta": {"chunk_id": "c1", "source": "a.pdf"}},
                {"text": "B", "meta": {"chunk_id": "c2", "source": "b.pdf"}},
            ]

    class KnowledgeRetriever:
        def search(self, kb_id, query, top_k):
            return []

    class FeedbackStore:
        def boosts_for_query(self, kb_id, query):
            return {"c2": 0.75}

    @contextmanager
    def lease(kb_id):
        yield

    monkeypatch.setattr(
        "cogdoc.graph.subgraphs.qa.RetrieverFactory.get_engine",
        lambda kb_id: Engine(),
    )
    monkeypatch.setattr("cogdoc.service.kb_readers.kb_read_lease", lease)
    runtime = _runtime(KnowledgeRetriever(), FeedbackStore())

    docs = agent._run_retrieve(
        RetrieveRequest(query="问题", doc_id="kb", top_k=3, rerank=False),
        state_runtime=runtime,
    )

    assert [doc["meta"]["chunk_id"] for doc in docs] == ["c2", "c1"]
    assert docs[0]["retrieval"]["feedback_boost"] == 0.75


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
