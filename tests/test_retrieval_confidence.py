from cogdoc.agents.answer_markers import NO_RELEVANT_CONTENT_ANSWER
from cogdoc.config.settings import Settings
from cogdoc.graph.subgraphs import qa
from cogdoc.graph.subgraphs.qa import abstain_node, rerank_node, retrieval_check
from cogdoc.tools.retriever.confidence import assess_retrieval_support


def _settings(**overrides):
    return Settings(_env_file=None, **overrides)


def _doc(*, distance=None, bm25_score=None, source_type="document"):
    retrieval = {}
    if distance is not None:
        retrieval["distance"] = distance
    if bm25_score is not None:
        retrieval["bm25_score"] = bm25_score
    return {
        "text": "evidence",
        "meta": {"source_type": source_type},
        "retrieval": retrieval,
    }


# 验证空候选和双低分候选会触发拒答。
def test_support_rejects_empty_and_low_confidence_candidates():
    settings = _settings()

    assert assess_retrieval_support([], settings).reason == "no_candidates"
    result = assess_retrieval_support(
        [_doc(distance=0.95, bm25_score=5.0)], settings
    )

    assert result.supported is False
    assert result.reason == "below_threshold"
    assert result.signals == {"distance": 0.95, "bm25_score": 5.0}


# 验证向量或 BM25 任一达到阈值即可保留证据。
def test_support_accepts_semantic_or_lexical_signal():
    settings = _settings()

    semantic = assess_retrieval_support(
        [_doc(distance=0.8, bm25_score=1.0)], settings
    )
    lexical = assess_retrieval_support(
        [_doc(distance=1.1, bm25_score=12.0)], settings
    )

    assert semantic.supported is True
    assert lexical.supported is True


# 验证缺少旧索引评分元数据时兼容放行。
def test_support_fails_open_when_confidence_signals_are_unavailable():
    result = assess_retrieval_support([_doc()], _settings())

    assert result.supported is True
    assert result.reason == "signals_unavailable"


# 验证派生知识使用独立支持度阈值。
def test_support_uses_derived_knowledge_score():
    doc = _doc(source_type="derived_knowledge")
    doc["retrieval"]["retrieval_score"] = 0.4

    result = assess_retrieval_support([doc], _settings())

    assert result.supported is False
    assert result.signals == {"knowledge_score": 0.4}


# 验证重排节点把低置信度判断写回图状态，供条件边直接拒答。
def test_rerank_node_marks_low_confidence_retrieval_for_abstention(monkeypatch):
    settings = _settings(qa_rerank_on_cpu=False)
    monkeypatch.setattr(qa, "get_settings", lambda: settings)
    monkeypatch.setattr(qa.BGEReranker, "default_device", lambda: "cpu")
    monkeypatch.setattr(
        qa, "_expand_with_neighbor_chunks", lambda _doc_id, docs, _state: docs
    )
    monkeypatch.setattr(qa, "log_event", lambda *args, **kwargs: None)

    output = rerank_node(
        {
            "query": "unrelated question",
            "doc_id": "kb",
            "retrieved_docs": [_doc(distance=0.95, bm25_score=5.0)],
        }
    )

    assert output["retrieval_abstained"] is True
    assert output["retrieval_abstain_reason"] == "below_threshold"
    assert output["retrieval_signals"] == {"distance": 0.95, "bm25_score": 5.0}


# 验证拒答节点不携带候选证据并直接结束 QA。
def test_abstain_node_returns_stable_answer_without_evidence():
    output = abstain_node(
        {
            "retrieval_abstained": True,
            "retrieval_confidence": 0.5,
            "retrieval_abstain_reason": "below_threshold",
        }
    )

    assert output["answer"] == NO_RELEVANT_CONTENT_ANSWER
    assert output["evidence"] == []
    assert output["sources"] == []
    assert output["reranked_docs"] == []
    assert retrieval_check(output) == "abstain_node"
    assert retrieval_check({"retrieval_abstained": False}) == "generate_node"
