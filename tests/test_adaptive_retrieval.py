from types import SimpleNamespace

from langchain_core.messages import AIMessage

from cogdoc.config.settings import Settings
from cogdoc.graph.subgraphs import qa


def _doc(chunk_id: str, source: str) -> dict:
    return {
        "text": f"{source} 的直接证据",
        "meta": {
            "chunk_id": chunk_id,
            "source": source,
            "page": 1,
            "page_start": 1,
            "page_end": 1,
            "chunk_index": 0,
        },
        "retrieval": {"bm25_score": 12.0},
    }


class _Engine:
    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int):
        self.calls.append((query, top_k))
        # 第二轮不再重新召回 A，必须由已验证证据携带进新候选池。
        if query in {"比较 A 和 B", "A 主检索"} and top_k == 9:
            return [_doc("c1", "a.pdf")]
        if query == "B 恢复检索":
            return [_doc("c2", "b.pdf")]
        return []


class _Knowledge:
    def search(self, kb_id: str, query: str, top_k: int):
        return []


class _Feedback:
    def boosts_for_query(self, kb_id: str, query: str):
        return {}


class _AnswerLLM:
    def invoke(self, messages):
        return AIMessage(content="A 与 B 均有直接证据。[a.pdf:P1] [b.pdf:P1]")


def test_verified_carryover_retains_requirement_attribution():
    carried, count = qa._carry_verified_docs(
        {
            "retrieval_retry_count": 1,
            "evidence_verified_chunk_ids": ["c1"],
            "evidence_requirement_assessments": [
                {
                    "requirement_id": "r1",
                    "verdict": "supported",
                    "evidence_chunk_ids": ["c1"],
                    "reason": "A 已覆盖",
                }
            ],
            "verification_docs": [_doc("c1", "a.pdf")],
        },
        [_doc("c2", "b.pdf")],
    )

    assert count == 1
    assert carried[0]["retrieval"]["matched_requirement_ids"] == ["r1"]


def test_adaptive_retrieval_recovers_missing_requirement_once(monkeypatch):
    settings = Settings(
        _env_file=None,
        qa_rerank_on_cpu=False,
        qa_evidence_verify_enabled=True,
        qa_adaptive_retrieval_enabled=True,
        qa_adaptive_retrieval_max_retries=1,
        qa_retrieval_top_k=9,
    )
    requirements = [
        {
            "requirement_id": "r1",
            "question": "A 的规则是什么",
            "retrieval_query": "A 主检索",
            "recovery_query": "A 恢复检索",
        },
        {
            "requirement_id": "r2",
            "question": "B 的规则是什么",
            "retrieval_query": "B 主检索",
            "recovery_query": "B 恢复检索",
        },
    ]
    engine = _Engine()
    verifier_calls: list[list[str]] = []

    monkeypatch.setattr(qa, "get_settings", lambda: settings)
    monkeypatch.setattr(
        qa.QueryRewriteAgent,
        "rewrite_query",
        lambda state: {
            "rewritten_queries": [],
            "evidence_requirements": requirements,
        },
    )
    monkeypatch.setattr(
        qa.RewriteVerifyAgent,
        "verify_rewrites",
        lambda state: {
            "rewritten_queries": state["rewritten_queries"],
            "evidence_requirements": state["evidence_requirements"],
        },
    )
    monkeypatch.setattr(qa.RetrieverFactory, "get_engine", lambda _kb_id: engine)
    monkeypatch.setattr(qa.BGEReranker, "default_device", lambda: "cpu")
    monkeypatch.setattr(
        qa, "_expand_with_neighbor_chunks", lambda _kb_id, docs, _state: docs
    )
    monkeypatch.setattr(qa, "log_event", lambda *args, **kwargs: None)

    def verify(state):
        chunk_ids = [doc["meta"]["chunk_id"] for doc in state["verification_docs"]]
        verifier_calls.append(chunk_ids)
        if "c2" not in chunk_ids:
            return {
                "evidence_verification_required": True,
                "evidence_supported": False,
                "evidence_verification_reason": "缺少 B",
                "evidence_verified_chunk_ids": ["c1"],
                "evidence_requirement_assessments": [
                    {
                        "requirement_id": "r1",
                        "verdict": "supported",
                        "evidence_chunk_ids": ["c1"],
                        "reason": "A 已覆盖",
                    },
                    {
                        "requirement_id": "r2",
                        "verdict": "missing",
                        "evidence_chunk_ids": [],
                        "reason": "缺少 B",
                    },
                ],
                "missing_evidence_requirement_ids": ["r2"],
                "retrieval_abstained": True,
                "retrieval_abstain_reason": "evidence_not_supported",
            }
        return {
            "evidence_verification_required": True,
            "evidence_supported": True,
            "evidence_verification_reason": "全部覆盖",
            "evidence_verified_chunk_ids": ["c1", "c2"],
            "evidence_requirement_assessments": [
                {
                    "requirement_id": requirement_id,
                    "verdict": "supported",
                    "evidence_chunk_ids": [chunk_id],
                    "reason": "已覆盖",
                }
                for requirement_id, chunk_id in (("r1", "c1"), ("r2", "c2"))
            ],
            "missing_evidence_requirement_ids": [],
            "retrieval_abstained": False,
            "retrieval_abstain_reason": "evidence_supported",
        }

    monkeypatch.setattr(qa.EvidenceVerifierAgent, "verify", verify)
    monkeypatch.setattr(
        qa.Generator, "_get_client_for_node", lambda *args, **kwargs: _AnswerLLM()
    )
    monkeypatch.setattr(
        qa.CitationValidatorAgent,
        "validate_citations",
        lambda answer, docs: {"is_valid": True, "critique": ""},
    )
    runtime = SimpleNamespace(
        derived_knowledge_retriever=_Knowledge(),
        retrieval_feedback_store=_Feedback(),
    )

    result = qa.qa_subgraph_node.invoke(
        {"messages": [], "query": "比较 A 和 B", "doc_id": "kb"},
        {"configurable": {"state_runtime": runtime}},
    )

    assert result["retrieval_retry_count"] == 1
    assert result["retrieval_carryover_count"] == 1
    assert result["retrieval_abstained"] is False
    assert result["evidence_supported"] is True
    assert result["missing_evidence_requirement_ids"] == []
    assert result["answer"].startswith("A 与 B 均有直接证据")
    assert len(verifier_calls) == 2
    assert "c1" in verifier_calls[1]
    assert "c2" in verifier_calls[1]
    assert ("B 主检索", 9) in engine.calls
    assert ("B 恢复检索", 18) in engine.calls
    assert sum(query == "B 恢复检索" for query, _ in engine.calls) == 1
