from agents.source_resolver import (
    SourceResolution,
    resolve_compare_sources,
    resolve_summary_source,
)
from agents.qa_generator import Generator


class FakeResolverLLM:
    def __init__(self, names):
        self.names = names

    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, messages):
        return SourceResolution(sources=self.names)


_HISTORY = [
    {"role": "user", "content": "讲讲 a.pdf 这篇", "timestamp": None},
    {"role": "assistant", "content": "a.pdf 介绍了方法 A。", "timestamp": None},
]


def test_resolves_referential_sources(monkeypatch):
    monkeypatch.setattr(
        Generator, "_get_client", lambda **kw: FakeResolverLLM(["a.pdf", "b.pdf"])
    )
    out = resolve_compare_sources(
        "对比这个文件和 b.pdf", ["a.pdf", "b.pdf", "c.pdf"], _HISTORY
    )
    assert out == ["a.pdf", "b.pdf"]


def test_filters_hallucinated_names(monkeypatch):
    # 集合外文件名被过滤，剩 1 个有效 < 2 → 回落空。
    monkeypatch.setattr(
        Generator, "_get_client", lambda **kw: FakeResolverLLM(["a.pdf", "ghost.pdf"])
    )
    assert resolve_compare_sources("对比这个和那个", ["a.pdf", "b.pdf"], _HISTORY) == []


def test_orders_by_source_index(monkeypatch):
    # LLM 乱序返回，输出按 sources 下标排序，保证 compare 列序确定。
    monkeypatch.setattr(
        Generator, "_get_client", lambda **kw: FakeResolverLLM(["c.pdf", "a.pdf"])
    )
    out = resolve_compare_sources("对比上面两篇", ["a.pdf", "b.pdf", "c.pdf"], _HISTORY)
    assert out == ["a.pdf", "c.pdf"]


def test_dedups_repeated_names(monkeypatch):
    monkeypatch.setattr(
        Generator,
        "_get_client",
        lambda **kw: FakeResolverLLM(["a.pdf", "a.pdf", "b.pdf"]),
    )
    out = resolve_compare_sources("对比这两个", ["a.pdf", "b.pdf"], _HISTORY)
    assert out == ["a.pdf", "b.pdf"]


def test_no_history_skips_llm(monkeypatch):
    def boom(**kw):
        raise AssertionError("无历史时不应调用 LLM")

    monkeypatch.setattr(Generator, "_get_client", boom)
    assert resolve_compare_sources("对比这个文件", ["a.pdf", "b.pdf"], []) == []


def test_fewer_than_two_sources_skips_llm(monkeypatch):
    def boom(**kw):
        raise AssertionError("可用文件不足 2 篇时不应调用 LLM")

    monkeypatch.setattr(Generator, "_get_client", boom)
    assert resolve_compare_sources("对比这个文件", ["a.pdf"], _HISTORY) == []


def test_llm_error_returns_empty(monkeypatch):
    def boom(**kw):
        raise RuntimeError("api down")

    monkeypatch.setattr(Generator, "_get_client", boom)
    assert (
        resolve_compare_sources("对比这个文件和那个", ["a.pdf", "b.pdf"], _HISTORY)
        == []
    )


def test_summary_resolves_single_referential_source(monkeypatch):
    monkeypatch.setattr(
        Generator, "_get_client", lambda **kw: FakeResolverLLM(["a.pdf"])
    )
    assert (
        resolve_summary_source("总结这个文件", ["a.pdf", "b.pdf"], _HISTORY) == "a.pdf"
    )


def test_summary_takes_first_valid_when_model_returns_many(monkeypatch):
    monkeypatch.setattr(
        Generator, "_get_client", lambda **kw: FakeResolverLLM(["c.pdf", "a.pdf"])
    )
    # 按 sources 下标排序后取第一篇。
    assert (
        resolve_summary_source("总结上面那篇", ["a.pdf", "b.pdf", "c.pdf"], _HISTORY)
        == "a.pdf"
    )


def test_summary_returns_none_when_unresolved(monkeypatch):
    monkeypatch.setattr(
        Generator, "_get_client", lambda **kw: FakeResolverLLM(["ghost.pdf"])
    )
    assert resolve_summary_source("总结那个", ["a.pdf", "b.pdf"], _HISTORY) is None


def test_summary_no_history_skips_llm(monkeypatch):
    def boom(**kw):
        raise AssertionError("无历史时不应调用 LLM")

    monkeypatch.setattr(Generator, "_get_client", boom)
    assert resolve_summary_source("总结这个文件", ["a.pdf", "b.pdf"], []) is None
