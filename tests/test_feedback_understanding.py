from cogdoc.agents.feedback_understanding import analyze_feedback


# 验证明确纠错会建议生成知识草稿。
def test_analyze_feedback_recommends_pending_knowledge_for_correction():
    analysis = analyze_feedback(
        {
            "trace_id": "t1",
            "feedback": "correction",
            "kb_id": "kb",
            "query": "问题",
            "correction_text": "正确说法",
            "citations": [{"chunk_id": "c1", "source": "a.pdf"}],
        }
    )

    assert analysis["feedback_type"] == "correction"
    assert analysis["recommended_action"] == "create_pending_knowledge"
    assert analysis["extracted_claim"] == "正确说法"
    assert analysis["confidence"] >= 0.8
    assert analysis["target"]["chunk_ids"] == ["c1"]


# 验证坏检索反馈会建议调权。
def test_analyze_feedback_recommends_retrieval_adjustment():
    analysis = analyze_feedback(
        {
            "trace_id": "t2",
            "feedback": "thumbs_down",
            "kb_id": "kb",
            "query": "问题",
            "feedback_text": "引用不相关",
            "evidence": [{"chunk_id": "c2", "source": "b.pdf"}],
        }
    )

    assert analysis["feedback_type"] == "bad_retrieval"
    assert analysis["recommended_action"] == "adjust_retrieval"
    assert analysis["weight_delta"] < 0
    assert analysis["target"]["sources"] == ["b.pdf"]


# 验证缺少目标的普通反馈只记录。
def test_analyze_feedback_records_low_context_feedback_only():
    analysis = analyze_feedback({"trace_id": "t3", "feedback": "thumbs_down"})

    assert analysis["recommended_action"] == "record_only"
    assert analysis["needs_review"] is True
