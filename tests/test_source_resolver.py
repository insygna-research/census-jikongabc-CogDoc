from cogdoc.agents.source_resolver import (
    SourceResolution,
    resolve_compare_sources,
    resolve_summary_source,
)
from cogdoc.agents.qa_generator import Generator


# 定义 FakeResolverLLM 数据结构。
class FakeResolverLLM:
    # 初始化 FakeResolverLLM 实例。
    def __init__(self, names):
        self.names = names

    # 返回支持结构化输出的测试替身。
    def with_structured_output(self, schema, **kwargs):
        return self

    # 调用测试替身并返回预设结果。
    def invoke(self, messages):
        return SourceResolution(sources=self.names)


_HISTORY = [
    {"role": "user", "content": "讲讲 a.pdf 这篇", "timestamp": None},
    {"role": "assistant", "content": "a.pdf 介绍了方法 A。", "timestamp": None},
]


# 验证 resolves referential sources 场景。
def test_resolves_referential_sources(monkeypatch):
    monkeypatch.setattr(
        Generator, "_get_client", lambda **kw: FakeResolverLLM(["a.pdf", "b.pdf"])
    )
    out = resolve_compare_sources(
        "对比这个文件和 b.pdf", ["a.pdf", "b.pdf", "c.pdf"], _HISTORY
    )
    assert out == ["a.pdf", "b.pdf"]


# 验证 filters hallucinated names 场景。
def test_filters_hallucinated_names(monkeypatch):
    # 集合外文件名被过滤，剩 1 个有效 < 2 → 回落空。
    monkeypatch.setattr(
        Generator, "_get_client", lambda **kw: FakeResolverLLM(["a.pdf", "ghost.pdf"])
    )
    assert resolve_compare_sources("对比这个和那个", ["a.pdf", "b.pdf"], _HISTORY) == []


# 验证 orders by source index 场景。
def test_orders_by_source_index(monkeypatch):
    # LLM 乱序返回，输出按 sources 下标排序，保证 compare 列序确定。
    monkeypatch.setattr(
        Generator, "_get_client", lambda **kw: FakeResolverLLM(["c.pdf", "a.pdf"])
    )
    out = resolve_compare_sources("对比上面两篇", ["a.pdf", "b.pdf", "c.pdf"], _HISTORY)
    assert out == ["a.pdf", "c.pdf"]


# 验证 dedups repeated names 场景。
def test_dedups_repeated_names(monkeypatch):
    monkeypatch.setattr(
        Generator,
        "_get_client",
        lambda **kw: FakeResolverLLM(["a.pdf", "a.pdf", "b.pdf"]),
    )
    out = resolve_compare_sources("对比这两个", ["a.pdf", "b.pdf"], _HISTORY)
    assert out == ["a.pdf", "b.pdf"]


# 验证 no history skips llm 场景。
def test_no_history_skips_llm(monkeypatch):
    # 模拟失败结果。
    def boom(**kw):
        raise AssertionError("无历史时不应调用 LLM")

    monkeypatch.setattr(Generator, "_get_client", boom)
    assert resolve_compare_sources("对比这个文件", ["a.pdf", "b.pdf"], []) == []


# 验证 fewer than two sources skips llm 场景。
def test_fewer_than_two_sources_skips_llm(monkeypatch):
    # 模拟失败结果。
    def boom(**kw):
        raise AssertionError("可用文件不足 2 篇时不应调用 LLM")

    monkeypatch.setattr(Generator, "_get_client", boom)
    assert resolve_compare_sources("对比这个文件", ["a.pdf"], _HISTORY) == []


# 验证 llm error returns empty 场景。
def test_llm_error_returns_empty(monkeypatch):
    # 模拟失败结果。
    def boom(**kw):
        raise RuntimeError("api down")

    monkeypatch.setattr(Generator, "_get_client", boom)
    assert (
        resolve_compare_sources("对比这个文件和那个", ["a.pdf", "b.pdf"], _HISTORY)
        == []
    )


# 验证 summary resolves single referential source 场景。
def test_summary_resolves_single_referential_source(monkeypatch):
    monkeypatch.setattr(
        Generator, "_get_client", lambda **kw: FakeResolverLLM(["a.pdf"])
    )
    assert (
        resolve_summary_source("总结这个文件", ["a.pdf", "b.pdf"], _HISTORY) == "a.pdf"
    )


# 验证 summary takes first valid when model returns many 场景。
def test_summary_takes_first_valid_when_model_returns_many(monkeypatch):
    monkeypatch.setattr(
        Generator, "_get_client", lambda **kw: FakeResolverLLM(["c.pdf", "a.pdf"])
    )
    # 按 sources 下标排序后取第一篇。
    assert (
        resolve_summary_source("总结上面那篇", ["a.pdf", "b.pdf", "c.pdf"], _HISTORY)
        == "a.pdf"
    )


# 验证 summary returns none when unresolved 场景。
def test_summary_returns_none_when_unresolved(monkeypatch):
    monkeypatch.setattr(
        Generator, "_get_client", lambda **kw: FakeResolverLLM(["ghost.pdf"])
    )
    assert resolve_summary_source("总结那个", ["a.pdf", "b.pdf"], _HISTORY) is None


# 验证 summary no history skips llm 场景。
def test_summary_no_history_skips_llm(monkeypatch):
    # 模拟失败结果。
    def boom(**kw):
        raise AssertionError("无历史时不应调用 LLM")

    monkeypatch.setattr(Generator, "_get_client", boom)
    assert resolve_summary_source("总结这个文件", ["a.pdf", "b.pdf"], []) is None
