from agents.query_rewriter import QueryRewriteAgent, QueryRewriteOutput
from agents.generator import Generator

class _RaisingLLM:
    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        raise RuntimeError("LLM 暂不可用")

class _OkLLM:
    def __init__(self, queries):
        self._queries = queries

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        return QueryRewriteOutput(queries=self._queries)

def test_empty_query_returns_empty_list():
    # 验证空 query 直接返回空改写。
    assert QueryRewriteAgent.rewrite_query({"query": ""}) == {"rewritten_queries": []}

def test_missing_query_key_returns_empty_list():
    # 验证缺少 query 字段时直接返回空改写。
    assert QueryRewriteAgent.rewrite_query({}) == {"rewritten_queries": []}

def test_llm_failure_falls_back_to_original_query(monkeypatch):
    # 验证改写失败时回退为原始 query。
    monkeypatch.setattr(Generator, "_get_client", lambda **kwargs: _RaisingLLM())
    query = "大模型如何做检索增强"
    result = QueryRewriteAgent.rewrite_query({"query": query})
    assert result == {"rewritten_queries": [query]}

def test_successful_rewrite_passes_through(monkeypatch):
    # 验证改写成功时透传结构化输出。
    monkeypatch.setattr(Generator, "_get_client", lambda **kwargs: _OkLLM(["q1", "q2"]))
    result = QueryRewriteAgent.rewrite_query({"query": "原始问题"})
    assert result == {"rewritten_queries": ["q1", "q2"]}
