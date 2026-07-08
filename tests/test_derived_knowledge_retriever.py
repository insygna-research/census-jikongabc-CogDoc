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
            "related_page_start": 2,
            "related_page_end": 3,
            "related_chunk_text_hash": "hash",
            "related_anchor_text": "报销需要七天内提交",
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
    assert docs[0]["meta"]["page"] == 2
    assert docs[0]["meta"]["page_start"] == 2
    assert docs[0]["meta"]["page_end"] == 3
    assert docs[0]["meta"]["related_chunk_text_hash"] == "hash"
    assert docs[0]["meta"]["related_anchor_text"] == "报销需要七天内提交"
    assert docs[0]["retrieval"]["search_channel"] == "derived_knowledge"
    assert docs[0]["retrieval"]["status_filter"] == "approved"
    assert docs[0]["retrieval"]["match_coverage"] > 0
    assert docs[0]["retrieval"]["query_term_count"] > 0
    assert docs[0]["retrieval"]["knowledge_term_count"] > 0
    assert "差旅" in docs[0]["retrieval"]["matched_terms"]
