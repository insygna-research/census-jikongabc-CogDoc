from cogdoc.agents.qa_generator import Generator
from cogdoc.agents.router import RouteDecision, RouterAgent, classify_intent_by_rule


# 模拟结构化路由不可用，验证规则路由兜底仍可工作。
class RaisingStructuredLLM:
    # 模拟结构化路由不可用，验证规则路由兜底仍可工作。
    def with_structured_output(self, schema, **kwargs):
        assert kwargs["method"] == "json_mode"
        return self

    # 调用测试替身并返回预设结果。
    def invoke(self, messages):
        raise RuntimeError("This response_format type is unavailable now")


# 验证 rule router classifies explicit summary 场景。
def test_rule_router_classifies_explicit_summary():
    decision = classify_intent_by_rule("总结大模型开发应用赛")

    assert decision.task_type == "summary"


# 验证 rule router avoids common false positive phrases 场景。
def test_rule_router_avoids_common_false_positive_phrases():
    # “比较好用”“比较详细”“总结部分”不是 compare/summary 任务触发词。
    assert classify_intent_by_rule("比较好用的检索方法是什么").task_type == "qa"
    assert classify_intent_by_rule("请比较详细地介绍这个算法").task_type == "qa"
    assert classify_intent_by_rule("这篇论文的总结部分讲了什么").task_type == "qa"


# 验证 router falls back to summary when structured llm fails 场景。
def test_router_falls_back_to_summary_when_structured_llm_fails(monkeypatch):
    monkeypatch.setattr(
        Generator, "_get_client", lambda is_local=False: RaisingStructuredLLM()
    )

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


# 验证 router uses llm before keyword fallback 场景。
def test_router_uses_llm_before_keyword_fallback(monkeypatch):
    # LLM 明确判为 QA 时，不应被关键词规则改写成 compare。
    class LLM:
        # 返回支持结构化输出的测试替身。
        def with_structured_output(self, schema, **kwargs):
            return self

        # 调用测试替身并返回预设结果。
        def invoke(self, messages):
            return RouteDecision(task_type="qa", reason="用户询问方法")

    monkeypatch.setattr(Generator, "_get_client", lambda is_local=False: LLM())

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


# 验证 router uses json mode for llm structured output 场景。
def test_router_uses_json_mode_for_llm_structured_output(monkeypatch):
    # Router 统一使用 json_mode，兼容本地和云端的结构化输出路径。
    class JsonModeLLM:
        # 返回支持结构化输出的测试替身。
        def with_structured_output(self, schema, **kwargs):
            assert kwargs["method"] == "json_mode"
            return self

        # 调用测试替身并返回预设结果。
        def invoke(self, messages):
            return schema_response("qa", "普通信息诉求")

    # 构造响应。
    def schema_response(task_type, reason):
        return RouteDecision(task_type=task_type, reason=reason)

    monkeypatch.setattr(Generator, "_get_client", lambda is_local=False: JsonModeLLM())

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


# 验证 router forced task short circuits llm 场景。
def test_router_forced_task_short_circuits_llm(monkeypatch):
    # 抛出if调用。
    def raise_if_called(is_local=False):
        raise AssertionError("forced_task should bypass LLM routing")

    monkeypatch.setattr(Generator, "_get_client", raise_if_called)

    result = RouterAgent.route_intent(
        {"chat_history": [{"role": "user", "content": "总结 a.pdf"}]},
        {
            "configurable": {
                "query": "对比 a.pdf 和 b.pdf",
                "doc_id": "kb",
                "is_local": False,
                "forced_task": "compare",
            }
        },
    )

    assert result["task_type"] == "compare"
    assert result["router_reason"] == "用户显式指定"


# 验证 router injects recent history for llm routing 场景。
def test_router_injects_recent_history_for_llm_routing(monkeypatch):
    captured = {}

    # 定义 CapturingLLM 数据结构。
    class CapturingLLM:
        # 返回支持结构化输出的测试替身。
        def with_structured_output(self, schema, **kwargs):
            return self

        # 调用测试替身并返回预设结果。
        def invoke(self, messages):
            captured["messages"] = messages
            return RouteDecision(task_type="summary", reason="结合历史判断摘要")

    monkeypatch.setattr(Generator, "_get_client", lambda is_local=False: CapturingLLM())

    result = RouterAgent.route_intent(
        {
            "chat_history": [
                {"role": "user", "content": "介绍 a.pdf 的核心方法"},
                {"role": "assistant", "content": "a.pdf 主要提出方法 A。"},
            ]
        },
        {
            "configurable": {
                "query": "那总结一下呢",
                "doc_id": "kb",
                "is_local": False,
            }
        },
    )

    user_message = captured["messages"][1]["content"]
    assert result["task_type"] == "summary"
    assert "【近期对话】" in user_message
    assert "用户: 介绍 a.pdf 的核心方法" in user_message
    assert "【当前提问】\n那总结一下呢" in user_message
