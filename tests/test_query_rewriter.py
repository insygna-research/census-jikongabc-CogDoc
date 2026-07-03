from cogdoc.agents.query_rewriter import QueryRewriteAgent, QueryRewriteOutput
from cogdoc.agents.qa_generator import Generator


# 模拟本地模型结构化输出不可用，验证重写链路必须降级到原始问题。
class _RaisingLLM:
    # 模拟本地模型结构化输出不可用，验证重写链路必须降级到原始问题。
    def with_structured_output(self, schema, **kwargs):
        assert kwargs["method"] == "json_mode"
        return self

    # 调用测试替身并返回预设结果。
    def invoke(self, messages):
        raise RuntimeError("LLM 暂不可用")


# 模拟结构化输出成功，避免测试依赖真实 LLM。
class _OkLLM:
    # 模拟结构化输出成功，避免测试依赖真实 LLM。
    def __init__(self, queries):
        self._queries = queries

    # 返回支持结构化输出的测试替身。
    def with_structured_output(self, schema, **kwargs):
        assert kwargs["method"] == "json_mode"
        return self

    # 调用测试替身并返回预设结果。
    def invoke(self, messages):
        return QueryRewriteOutput(queries=self._queries)


# 定义 _CapturingLLM 数据结构。
class _CapturingLLM:
    # 初始化 _CapturingLLM 实例。
    def __init__(self):
        self.messages = None

    # 返回支持结构化输出的测试替身。
    def with_structured_output(self, schema, **kwargs):
        assert kwargs["method"] == "json_mode"
        return self

    # 调用测试替身并返回预设结果。
    def invoke(self, messages):
        self.messages = messages
        return QueryRewriteOutput(queries=["Transformer 作者"])


# 验证 empty query returns empty list 场景。
def test_empty_query_returns_empty_list():
    assert QueryRewriteAgent.rewrite_query({"query": ""}) == {"rewritten_queries": []}


# 验证 missing query key returns empty list 场景。
def test_missing_query_key_returns_empty_list():
    assert QueryRewriteAgent.rewrite_query({}) == {"rewritten_queries": []}


# 验证 llm failure falls back to original query 场景。
def test_llm_failure_falls_back_to_original_query(monkeypatch):
    monkeypatch.setattr(Generator, "_get_client", lambda **kwargs: _RaisingLLM())
    query = "大模型如何做检索增强"
    result = QueryRewriteAgent.rewrite_query({"query": query})
    assert result == {"rewritten_queries": [query]}


# 验证 successful rewrite passes through 场景。
def test_successful_rewrite_passes_through(monkeypatch):
    monkeypatch.setattr(Generator, "_get_client", lambda **kwargs: _OkLLM(["q1", "q2"]))
    result = QueryRewriteAgent.rewrite_query({"query": "原始问题"})
    assert result == {"rewritten_queries": ["q1", "q2"]}


# 验证 rewrite prompt includes recent chat history 场景。
def test_rewrite_prompt_includes_recent_chat_history(monkeypatch):
    llm = _CapturingLLM()
    monkeypatch.setattr(Generator, "_get_client", lambda **kwargs: llm)

    result = QueryRewriteAgent.rewrite_query(
        {
            "query": "它的作者是谁？",
            "chat_history": [
                {
                    "role": "user",
                    "content": "Transformer 这篇论文讲了什么？",
                    "timestamp": None,
                },
                {
                    "role": "assistant",
                    "content": "它提出了自注意力架构。",
                    "timestamp": None,
                },
            ],
        }
    )

    assert result == {"rewritten_queries": ["Transformer 作者"]}
    user_prompt = llm.messages[1]["content"]
    assert "用户: Transformer 这篇论文讲了什么？" in user_prompt
    assert "助手: 它提出了自注意力架构。" in user_prompt
    assert "【当前提问】\n它的作者是谁？" in user_prompt
