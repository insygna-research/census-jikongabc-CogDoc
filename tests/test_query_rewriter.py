from agents.query_rewriter import QueryRewriteAgent, QueryRewriteOutput
from agents.qa_generator import Generator

class _RaisingLLM:
    # 模拟本地模型结构化输出不可用，验证重写链路必须降级到原始问题。
    def with_structured_output(self, schema, **kwargs):
        assert kwargs["method"] == "json_mode"
        return self

    def invoke(self, messages):
        raise RuntimeError("LLM 暂不可用")

class _OkLLM:
    # 模拟结构化输出成功，避免测试依赖真实 LLM。
    def __init__(self, queries):
        self._queries = queries

    def with_structured_output(self, schema, **kwargs):
        assert kwargs["method"] == "json_mode"
        return self

    def invoke(self, messages):
        return QueryRewriteOutput(queries=self._queries)

def test_empty_query_returns_empty_list():
    assert QueryRewriteAgent.rewrite_query({"query": ""}) == {"rewritten_queries": []}

def test_missing_query_key_returns_empty_list():
    assert QueryRewriteAgent.rewrite_query({}) == {"rewritten_queries": []}

def test_llm_failure_falls_back_to_original_query(monkeypatch):
    monkeypatch.setattr(Generator, "_get_client", lambda **kwargs: _RaisingLLM())
    query = "大模型如何做检索增强"
    result = QueryRewriteAgent.rewrite_query({"query": query})
    assert result == {"rewritten_queries": [query]}

def test_successful_rewrite_passes_through(monkeypatch):
    monkeypatch.setattr(Generator, "_get_client", lambda **kwargs: _OkLLM(["q1", "q2"]))
    result = QueryRewriteAgent.rewrite_query({"query": "原始问题"})
    assert result == {"rewritten_queries": ["q1", "q2"]}
