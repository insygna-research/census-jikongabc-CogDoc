from agents.qa_generator import Generator
from agents.router import RouteDecision, RouterAgent, classify_intent_by_rule


class RaisingStructuredLLM:
    def with_structured_output(self, schema, **kwargs):
        assert kwargs["method"] == "json_mode"
        return self

    def invoke(self, messages):
        raise RuntimeError("This response_format type is unavailable now")


def test_rule_router_classifies_explicit_summary():
    decision = classify_intent_by_rule("总结大模型开发应用赛")

    assert decision.task_type == "summary"


def test_rule_router_avoids_common_false_positive_phrases():
    assert classify_intent_by_rule("比较好用的检索方法是什么").task_type == "qa"
    assert classify_intent_by_rule("请比较详细地介绍这个算法").task_type == "qa"
    assert classify_intent_by_rule("这篇论文的总结部分讲了什么").task_type == "qa"


def test_router_falls_back_to_summary_when_structured_llm_fails(monkeypatch):
    monkeypatch.setattr(Generator, "_get_client", lambda is_local = False: RaisingStructuredLLM())

    result = RouterAgent.route_intent(
        {},
        {
            "configurable": {
                "query": "总结大模型开发应用赛",
                "doc_id": "kb",
                "is_local": False,
            }
        },
    )

    assert result["task_type"] == "summary"
    assert "规则原因：命中摘要关键词" in result["router_reason"]


def test_router_uses_llm_before_keyword_fallback(monkeypatch):
    class LLM:
        def with_structured_output(self, schema, **kwargs):
            return self

        def invoke(self, messages):
            return RouteDecision(task_type = "qa", reason = "用户询问方法")

    monkeypatch.setattr(Generator, "_get_client", lambda is_local = False: LLM())

    result = RouterAgent.route_intent(
        {},
        {
            "configurable": {
                "query": "比较好用的检索方法是什么",
                "doc_id": "kb",
                "is_local": False,
            }
        },
    )

    assert result["task_type"] == "qa"
    assert result["router_reason"] == "用户询问方法"


def test_router_uses_json_mode_for_llm_structured_output(monkeypatch):
    class JsonModeLLM:
        def with_structured_output(self, schema, **kwargs):
            assert kwargs["method"] == "json_mode"
            return self

        def invoke(self, messages):
            return schema_response("qa", "普通信息诉求")

    def schema_response(task_type, reason):
        return RouteDecision(task_type = task_type, reason = reason)

    monkeypatch.setattr(Generator, "_get_client", lambda is_local = False: JsonModeLLM())

    result = RouterAgent.route_intent(
        {},
        {
            "configurable": {
                "query": "大模型开发应用赛需要准备什么",
                "doc_id": "kb",
                "is_local": False,
            }
        },
    )

    assert result["task_type"] == "qa"
    assert result["router_reason"] == "普通信息诉求"
