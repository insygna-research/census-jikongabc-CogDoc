from cogdoc.api.derived_knowledge_store import DerivedKnowledgeStore
from cogdoc.tools.retriever.derived_knowledge import DerivedKnowledgeRetriever


# 验证只召回已审核派生知识。
def test_derived_knowledge_retriever_returns_approved_only(tmp_path):
    store = DerivedKnowledgeStore(path=str(tmp_path / "knowledge.jsonl"))
    approved, _ = store.create(
        {
            "kb_id": "kb",
            "text": "差旅报销需要在七天内提交。",
            "status": "approved",
            "certainty": "high",
            "related_source": "policy.pdf",
            "related_source_sha256": "sha",
            "related_chunk_ids": ["chunk-1"],
        }
    )
    store.create(
        {
            "kb_id": "kb",
            "text": "差旅报销可以一个月后提交。",
            "status": "pending",
        }
    )

    docs = DerivedKnowledgeRetriever(store).search("kb", "差旅报销提交", top_k=5)

    assert len(docs) == 1
    assert docs[0]["meta"]["knowledge_id"] == approved["knowledge_id"]
    assert docs[0]["meta"]["source_type"] == "derived_knowledge"
    assert docs[0]["meta"]["source"] == f"knowledge:{approved['knowledge_id']}"
    assert docs[0]["meta"]["related_chunk_ids"] == ["chunk-1"]
