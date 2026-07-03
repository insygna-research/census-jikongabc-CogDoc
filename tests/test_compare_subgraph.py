import time
from cogdoc.agents import compare_generator, compare_profile
from cogdoc.graph import workflow
from cogdoc.graph.subgraphs import compare
from cogdoc.graph.subgraphs.compare import (
    citation_node,
    compare_table_node,
    document_loader_check,
    document_loader_node,
    document_profile_check,
    document_profile_node,
)


# 定义 FakeMessage 数据结构。
class FakeMessage:
    # 初始化 FakeMessage 实例。
    def __init__(self, content):
        self.content = content


# 定义 FakeCompareLLM 数据结构。
class FakeCompareLLM:
    # 返回支持结构化输出的测试替身。
    def with_structured_output(self, schema):
        return FakeStructuredRouter(schema)

    # 调用测试替身并返回预设结果。
    def invoke(self, messages):
        return FakeMessage("该维度内容来自文档。")


# 定义 FakeNoEvidenceLLM 数据结构。
class FakeNoEvidenceLLM:
    # 调用测试替身并返回预设结果。
    def invoke(self, messages):
        return FakeMessage("文档中未明确说明。")


# 定义 FakeStructuredRouter 数据结构。
class FakeStructuredRouter:
    # 初始化 FakeStructuredRouter 实例。
    def __init__(self, schema):
        self.schema = schema

    # 调用测试替身并返回预设结果。
    def invoke(self, messages):
        return self.schema(task_type="compare", reason="用户要求对比文档")


# 构造测试用文档。
def _doc(source: str, page: int, local_chunk_index: int) -> dict:
    return {
        "text": f"{source} p{page} c{local_chunk_index}",
        "meta": {
            "chunk_id": f"chunk:{source}:{local_chunk_index}",
            "source_sha256": f"sha:{source}",
            "local_chunk_index": local_chunk_index,
            "chunk_index": local_chunk_index,
            "source": source,
            "page": page,
            "page_start": page,
            "page_end": page,
            "origin": "file",
        },
    }


# 定义 FakeEngine 数据结构。
class FakeEngine:
    # 初始化 FakeEngine 实例。
    def __init__(self, docs):
        self.docs = docs

    # 列出 sources。
    def list_sources(self):
        return sorted({doc["meta"]["source"] for doc in self.docs})

    # 加载 source chunks。
    def load_source_chunks(self, source):
        return [doc for doc in self.docs if doc["meta"]["source"] == source]


# 构造测试用对比维度。
def _dimensions():
    return [
        {"dimension_id": "method", "title": "方法", "instruction": "概括方法"},
        {"dimension_id": "metrics", "title": "指标", "instruction": "概括指标"},
    ]


# 验证 document loader selects named compare sources 场景。
def test_document_loader_selects_named_compare_sources(monkeypatch):
    engine = FakeEngine([_doc("a.pdf", 1, 0), _doc("b.pdf", 1, 0), _doc("c.pdf", 1, 0)])
    monkeypatch.setattr(compare.RetrieverFactory, "get_engine", lambda doc_id: engine)

    result = document_loader_node({"query": "请对比 a.pdf 和 b.pdf", "doc_id": "kb"})

    assert result["compare_sources"] == ["a.pdf", "b.pdf"]
    assert list(result["compare_docs_by_source"].keys()) == ["a.pdf", "b.pdf"]
    assert result["steps_trace"][0]["step_name"] == "compare_document_loader"


# 验证 document loader resolves referential sources 场景。
def test_document_loader_resolves_referential_sources(monkeypatch):
    engine = FakeEngine([_doc("a.pdf", 1, 0), _doc("b.pdf", 1, 0)])
    monkeypatch.setattr(compare.RetrieverFactory, "get_engine", lambda doc_id: engine)
    # 「这个文件」字面匹配不到，靠近期对话消解出 a.pdf。
    monkeypatch.setattr(
        compare, "resolve_compare_sources", lambda *args, **kwargs: ["a.pdf", "b.pdf"]
    )

    result = document_loader_node(
        {
            "query": "对比这个文件和 b.pdf",
            "doc_id": "kb",
            "chat_history": [
                {"role": "user", "content": "讲讲 a.pdf", "timestamp": None}
            ],
        }
    )

    assert result["compare_sources"] == ["a.pdf", "b.pdf"]
    assert result["steps_trace"][0]["step_name"] == "compare_source_resolution"


# 验证 document loader skips resolution without history 场景。
def test_document_loader_skips_resolution_without_history(monkeypatch):
    engine = FakeEngine([_doc("a.pdf", 1, 0), _doc("b.pdf", 1, 0)])
    monkeypatch.setattr(compare.RetrieverFactory, "get_engine", lambda doc_id: engine)

    # 记录失败结果。
    def fail(*args, **kwargs):
        raise AssertionError("无历史时不应触发指代消解")

    monkeypatch.setattr(compare, "resolve_compare_sources", fail)

    result = document_loader_node({"query": "对比这个文件", "doc_id": "kb"})

    assert result["compare_docs_by_source"] == {}
    assert "可直接说出文件名" in result["answer"]
    assert document_loader_check(result) == "__end__"


# 验证 document loader explicit match skips resolution 场景。
def test_document_loader_explicit_match_skips_resolution(monkeypatch):
    engine = FakeEngine([_doc("a.pdf", 1, 0), _doc("b.pdf", 1, 0)])
    monkeypatch.setattr(compare.RetrieverFactory, "get_engine", lambda doc_id: engine)

    # 记录失败结果。
    def fail(*args, **kwargs):
        raise AssertionError("显式点名已足够时不应触发指代消解")

    monkeypatch.setattr(compare, "resolve_compare_sources", fail)

    result = document_loader_node(
        {
            "query": "请对比 a.pdf 和 b.pdf",
            "doc_id": "kb",
            "chat_history": [
                {"role": "user", "content": "随便聊聊", "timestamp": None}
            ],
        }
    )

    assert result["compare_sources"] == ["a.pdf", "b.pdf"]


# 验证 document loader requires at least two named sources 场景。
def test_document_loader_requires_at_least_two_named_sources(monkeypatch):
    engine = FakeEngine([_doc("a.pdf", 1, 0), _doc("b.pdf", 1, 0)])
    monkeypatch.setattr(compare.RetrieverFactory, "get_engine", lambda doc_id: engine)

    result = document_loader_node({"query": "请对比这些文档", "doc_id": "kb"})

    assert result["compare_docs_by_source"] == {}
    assert "点名至少 2 篇" in result["answer"]
    assert "a.pdf" in result["answer"] and "b.pdf" in result["answer"]
    assert document_loader_check(result) == "__end__"


# 验证 document loader limits local compare sources 场景。
def test_document_loader_limits_local_compare_sources(monkeypatch):
    engine = FakeEngine([_doc("a.pdf", 1, 0), _doc("b.pdf", 1, 0), _doc("c.pdf", 1, 0)])
    monkeypatch.setattr(compare.RetrieverFactory, "get_engine", lambda doc_id: engine)

    result = document_loader_node(
        {"query": "请对比 a.pdf、b.pdf 和 c.pdf", "doc_id": "kb", "is_local": True}
    )

    assert result["compare_docs_by_source"] == {}
    assert "本地 Ollama 模式最多支持同时对比 2 篇文档" in result["answer"]
    assert document_loader_check(result) == "__end__"


# 验证 document loader uses limited local dimensions 场景。
def test_document_loader_uses_limited_local_dimensions(monkeypatch):
    engine = FakeEngine([_doc("a.pdf", 1, 0), _doc("b.pdf", 2, 0)])
    monkeypatch.setattr(compare.RetrieverFactory, "get_engine", lambda doc_id: engine)

    result = document_loader_node(
        {"query": "请对比 a.pdf 和 b.pdf", "doc_id": "kb", "is_local": True}
    )

    assert [dimension["title"] for dimension in result["compare_dimensions"]] == [
        "方法",
        "数据",
        "指标",
        "限制",
    ]


# 验证 document profile node generates cells with evidence 场景。
def test_document_profile_node_generates_cells_with_evidence(monkeypatch):
    monkeypatch.setattr(
        compare_profile.Generator,
        "_get_client",
        lambda is_local=False: FakeCompareLLM(),
    )

    result = document_profile_node(
        {
            "query": "对比 a.pdf 和 b.pdf",
            "compare_sources": ["a.pdf", "b.pdf"],
            "compare_docs_by_source": {
                "a.pdf": [_doc("a.pdf", 1, 0)],
                "b.pdf": [_doc("b.pdf", 2, 0)],
            },
            "compare_dimensions": _dimensions(),
        }
    )

    assert [profile["source"] for profile in result["document_profiles"]] == [
        "a.pdf",
        "b.pdf",
    ]
    assert [
        cell["dimension_id"] for cell in result["document_profiles"][0]["cells"]
    ] == ["method", "metrics"]
    assert (
        result["document_profiles"][0]["cells"][0]["content"]
        == "该维度内容来自文档。[a.pdf:P1]"
    )
    assert (
        result["document_profiles"][1]["cells"][0]["content"]
        == "该维度内容来自文档。[b.pdf:P2]"
    )
    assert (
        result["document_profiles"][0]["cells"][0]["evidence"][0]["chunk_id"]
        == "chunk:a.pdf:0"
    )


# 验证 document profile node keeps no evidence cells empty 场景。
def test_document_profile_node_keeps_no_evidence_cells_empty(monkeypatch):
    monkeypatch.setattr(
        compare_profile.Generator,
        "_get_client",
        lambda is_local=False: FakeNoEvidenceLLM(),
    )

    result = document_profile_node(
        {
            "query": "对比 a.pdf 和 b.pdf",
            "compare_sources": ["a.pdf", "b.pdf"],
            "compare_docs_by_source": {
                "a.pdf": [_doc("a.pdf", 1, 0)],
                "b.pdf": [_doc("b.pdf", 2, 0)],
            },
            "compare_dimensions": _dimensions()[:1],
        }
    )

    assert result["document_profiles"][0]["cells"][0]["content"] == "文档中未明确说明。"
    assert result["document_profiles"][0]["cells"][0]["evidence"] == []


# 验证 document profile node returns actionable message on llm error 场景。
def test_document_profile_node_returns_actionable_message_on_llm_error(monkeypatch):
    # 抛出内存错误。
    def raise_memory_error(is_local=False):
        raise RuntimeError("model requires more system memory")

    monkeypatch.setattr(compare_profile.Generator, "_get_client", raise_memory_error)

    result = document_profile_node(
        {
            "query": "对比 a.pdf 和 b.pdf",
            "compare_sources": ["a.pdf", "b.pdf"],
            "compare_docs_by_source": {
                "a.pdf": [_doc("a.pdf", 1, 0)],
                "b.pdf": [_doc("b.pdf", 2, 0)],
            },
            "compare_dimensions": _dimensions()[:1],
            "is_local": True,
        }
    )

    assert result["document_profiles"] == []
    assert "模型生成对比画像失败" in result["answer"]
    assert "ollama stop" in result["answer"]
    assert document_profile_check(result) == "__end__"


# 验证 compare table node builds dimension blocks and conclusion 场景。
def test_compare_table_node_builds_dimension_blocks_and_conclusion(monkeypatch):
    monkeypatch.setattr(
        compare_generator.CompareGeneratorAgent,
        "_generate_conclusion",
        lambda state, table_answer: (
            "a.pdf 强调方法，b.pdf 强调指标。[a.pdf:P1][b.pdf:P2]"
        ),
    )
    state = {
        "compare_sources": ["a.pdf", "b.pdf"],
        "compare_docs_by_source": {
            "a.pdf": [_doc("a.pdf", 1, 0)],
            "b.pdf": [_doc("b.pdf", 2, 0)],
        },
        "compare_dimensions": _dimensions(),
        "document_profiles": [
            {
                "source": "a.pdf",
                "cells": [
                    {
                        "dimension_id": "method",
                        "source": "a.pdf",
                        "content": "方法 A。[a.pdf:P1]",
                        "evidence": [],
                    },
                    {
                        "dimension_id": "metrics",
                        "source": "a.pdf",
                        "content": "指标 A。[a.pdf:P1]",
                        "evidence": [],
                    },
                ],
            },
            {
                "source": "b.pdf",
                "cells": [
                    {
                        "dimension_id": "method",
                        "source": "b.pdf",
                        "content": "方法 B。[b.pdf:P2]",
                        "evidence": [],
                    },
                    {
                        "dimension_id": "metrics",
                        "source": "b.pdf",
                        "content": "指标 B。[b.pdf:P2]",
                        "evidence": [],
                    },
                ],
            },
        ],
    }
    table_result = compare_table_node(state)
    result = citation_node({**state, **table_result})

    assert "## 方法" in result["answer"]
    assert "- **a.pdf**：方法 A。[a.pdf:P1]" in result["answer"]
    assert "- **b.pdf**：方法 B。[b.pdf:P2]" in result["answer"]
    assert "## 简短结论" in result["answer"]
    assert "引用校验警告" not in result["answer"]


# 验证 compare table node records conclusion failure 场景。
def test_compare_table_node_records_conclusion_failure(monkeypatch):
    # 抛出超时。
    def raise_timeout(state, table_answer):
        raise TimeoutError("api timeout")

    monkeypatch.setattr(
        compare_generator.CompareGeneratorAgent, "_generate_conclusion", raise_timeout
    )
    state = {
        "compare_sources": ["a.pdf", "b.pdf"],
        "compare_docs_by_source": {
            "a.pdf": [_doc("a.pdf", 1, 0)],
            "b.pdf": [_doc("b.pdf", 2, 0)],
        },
        "compare_dimensions": _dimensions()[:1],
        "document_profiles": [
            {
                "source": "a.pdf",
                "cells": [
                    {
                        "dimension_id": "method",
                        "source": "a.pdf",
                        "content": "方法 A。[a.pdf:P1]",
                        "evidence": [],
                    },
                ],
            },
            {
                "source": "b.pdf",
                "cells": [
                    {
                        "dimension_id": "method",
                        "source": "b.pdf",
                        "content": "方法 B。[b.pdf:P2]",
                        "evidence": [],
                    },
                ],
            },
        ],
    }

    result = compare_table_node(state)

    assert "## 简短结论" not in result["answer"]
    assert "TimeoutError" in result["compare_conclusion_warning"]
    assert result["steps_trace"][-1]["step_name"] == "compare_conclusion_warning"


# 验证 compare table node skips conclusion in local mode 场景。
def test_compare_table_node_skips_conclusion_in_local_mode(monkeypatch):
    # 记录失败if调用。
    def fail_if_called(state, table_answer):
        raise AssertionError("local mode should not generate conclusion")

    monkeypatch.setattr(
        compare_generator.CompareGeneratorAgent, "_generate_conclusion", fail_if_called
    )
    state = {
        "compare_sources": ["a.pdf", "b.pdf"],
        "compare_docs_by_source": {
            "a.pdf": [_doc("a.pdf", 1, 0)],
            "b.pdf": [_doc("b.pdf", 2, 0)],
        },
        "compare_dimensions": _dimensions()[:1],
        "document_profiles": [
            {
                "source": "a.pdf",
                "cells": [
                    {
                        "dimension_id": "method",
                        "source": "a.pdf",
                        "content": "方法 A。[a.pdf:P1]",
                        "evidence": [],
                    },
                ],
            },
            {
                "source": "b.pdf",
                "cells": [
                    {
                        "dimension_id": "method",
                        "source": "b.pdf",
                        "content": "方法 B。[b.pdf:P2]",
                        "evidence": [],
                    },
                ],
            },
        ],
        "is_local": True,
    }

    result = compare_table_node(state)

    assert result["compare_conclusion"] == ""
    assert "本地 Ollama 模式已跳过简短结论生成" in result["compare_conclusion_warning"]
    assert "## 简短结论" not in result["answer"]


# 验证 citation node downgrades invalid compare citation 场景。
def test_citation_node_downgrades_invalid_compare_citation(monkeypatch):
    monkeypatch.setattr(
        compare_generator.CompareGeneratorAgent,
        "_generate_conclusion",
        lambda state, table_answer: "",
    )
    state = {
        "compare_sources": ["a.pdf", "b.pdf"],
        "compare_docs_by_source": {
            "a.pdf": [_doc("a.pdf", 1, 0)],
            "b.pdf": [_doc("b.pdf", 2, 0)],
        },
        "compare_dimensions": _dimensions()[:1],
        "document_profiles": [
            {
                "source": "a.pdf",
                "cells": [
                    {
                        "dimension_id": "method",
                        "source": "a.pdf",
                        "content": "方法 A。[a.pdf:P99]",
                        "evidence": [],
                    },
                ],
            },
            {
                "source": "b.pdf",
                "cells": [
                    {
                        "dimension_id": "method",
                        "source": "b.pdf",
                        "content": "方法 B。[b.pdf:P2]",
                        "evidence": [],
                    },
                ],
            },
        ],
    }
    table_result = compare_table_node(state)
    result = citation_node({**state, **table_result})

    assert "- **a.pdf**：方法 A。[a.pdf:P99]" in result["answer"]
    assert "- **b.pdf**：方法 B。[b.pdf:P2]" in result["answer"]
    assert "引用校验警告" in result["answer"]
    assert "页码错误" in result["critique"]


# 验证 citation node downgrades uncited conclusion 场景。
def test_citation_node_downgrades_uncited_conclusion(monkeypatch):
    monkeypatch.setattr(
        compare_generator.CompareGeneratorAgent,
        "_generate_conclusion",
        lambda state, table_answer: "综合来看 a.pdf 更适合生产环境。",
    )
    state = {
        "compare_sources": ["a.pdf", "b.pdf"],
        "compare_docs_by_source": {
            "a.pdf": [_doc("a.pdf", 1, 0)],
            "b.pdf": [_doc("b.pdf", 2, 0)],
        },
        "compare_dimensions": _dimensions()[:1],
        "document_profiles": [
            {
                "source": "a.pdf",
                "cells": [
                    {
                        "dimension_id": "method",
                        "source": "a.pdf",
                        "content": "方法 A。[a.pdf:P1]",
                        "evidence": [],
                    },
                ],
            },
            {
                "source": "b.pdf",
                "cells": [
                    {
                        "dimension_id": "method",
                        "source": "b.pdf",
                        "content": "方法 B。[b.pdf:P2]",
                        "evidence": [],
                    },
                ],
            },
        ],
    }
    table_result = compare_table_node(state)
    result = citation_node({**state, **table_result})
    assert "综合来看 a.pdf 更适合生产环境。" not in result["answer"]
    assert "## 简短结论" not in result["answer"]
    assert "- **a.pdf**：方法 A。[a.pdf:P1]" in result["answer"]
    assert "- **b.pdf**：方法 B。[b.pdf:P2]" in result["answer"]
    assert "引用校验警告" in result["answer"]
    assert result["critique"]


# 验证 citation node checks table after bad conclusion 场景。
def test_citation_node_checks_table_after_bad_conclusion(monkeypatch):
    monkeypatch.setattr(
        compare_generator.CompareGeneratorAgent,
        "_generate_conclusion",
        lambda state, table_answer: "综合来看 a.pdf 更适合生产环境。",
    )
    state = {
        "compare_sources": ["a.pdf", "b.pdf"],
        "compare_docs_by_source": {
            "a.pdf": [_doc("a.pdf", 1, 0)],
            "b.pdf": [_doc("b.pdf", 2, 0)],
        },
        "compare_dimensions": _dimensions()[:1],
        "document_profiles": [
            {
                "source": "a.pdf",
                "cells": [
                    {
                        "dimension_id": "method",
                        "source": "a.pdf",
                        "content": "方法 A。[a.pdf:P99]",
                        "evidence": [],
                    },
                ],
            },
            {
                "source": "b.pdf",
                "cells": [
                    {
                        "dimension_id": "method",
                        "source": "b.pdf",
                        "content": "方法 B。[b.pdf:P2]",
                        "evidence": [],
                    },
                ],
            },
        ],
    }
    table_result = compare_table_node(state)
    result = citation_node({**state, **table_result})

    assert "综合来看 a.pdf 更适合生产环境。" not in result["answer"]
    assert "引用校验警告" in result["answer"]
    assert "未包含任何引用标签" in result["critique"]
    assert "页码错误" in result["critique"]


# 验证 citation node flags substantive uncited table 场景。
def test_citation_node_flags_substantive_uncited_table(monkeypatch):
    monkeypatch.setattr(
        compare_generator.CompareGeneratorAgent,
        "_generate_conclusion",
        lambda state, table_answer: "",
    )
    state = {
        "compare_sources": ["a.pdf", "b.pdf"],
        "compare_docs_by_source": {
            "a.pdf": [_doc("a.pdf", 1, 0)],
            "b.pdf": [_doc("b.pdf", 2, 0)],
        },
        "compare_dimensions": _dimensions()[:1],
        "document_profiles": [
            {
                "source": "a.pdf",
                "cells": [
                    {
                        "dimension_id": "method",
                        "source": "a.pdf",
                        "content": "采用了方法 A。",
                        "evidence": [],
                    },
                ],
            },
            {
                "source": "b.pdf",
                "cells": [
                    {
                        "dimension_id": "method",
                        "source": "b.pdf",
                        "content": "采用了方法 B。",
                        "evidence": [],
                    },
                ],
            },
        ],
    }
    table_result = compare_table_node(state)
    result = citation_node({**state, **table_result})

    assert "引用校验警告" in result["answer"]
    assert result["critique"]


# 验证 citation node accepts all no evidence table 场景。
def test_citation_node_accepts_all_no_evidence_table(monkeypatch):
    monkeypatch.setattr(
        compare_generator.CompareGeneratorAgent,
        "_generate_conclusion",
        lambda state, table_answer: "",
    )
    state = {
        "compare_sources": ["a.pdf", "b.pdf"],
        "compare_docs_by_source": {
            "a.pdf": [_doc("a.pdf", 1, 0)],
            "b.pdf": [_doc("b.pdf", 2, 0)],
        },
        "compare_dimensions": _dimensions()[:1],
        "document_profiles": [
            {
                "source": "a.pdf",
                "cells": [
                    {
                        "dimension_id": "method",
                        "source": "a.pdf",
                        "content": "文档中未明确说明。",
                        "evidence": [],
                    },
                ],
            },
            {
                "source": "b.pdf",
                "cells": [
                    {
                        "dimension_id": "method",
                        "source": "b.pdf",
                        "content": "文档中未明确说明。",
                        "evidence": [],
                    },
                ],
            },
        ],
    }
    table_result = compare_table_node(state)
    result = citation_node({**state, **table_result})

    assert "- **a.pdf**：文档中未明确说明。" in result["answer"]
    assert "- **b.pdf**：文档中未明确说明。" in result["answer"]
    assert "引用校验警告" not in result["answer"]
    assert result["critique"] == ""


# 每次调用睡随机抖动时长，放大并发完成乱序，用于验证回填保序。
class FakeJitterLLM:
    # 每次调用睡随机抖动时长，放大并发完成乱序，用于验证回填保序。
    def __init__(self):
        self._counter = 0

    # 调用测试替身并返回预设结果。
    def invoke(self, messages):
        self._counter += 1
        time.sleep(0.04 if self._counter % 2 else 0.01)
        return FakeMessage("该维度内容来自文档。")


# 验证 document profile node preserves order under concurrency 场景。
def test_document_profile_node_preserves_order_under_concurrency(monkeypatch):
    monkeypatch.setattr(
        compare_profile.Generator, "_get_client", lambda is_local=False: FakeJitterLLM()
    )

    sources = ["a.pdf", "b.pdf", "c.pdf"]
    dimensions = [
        {"dimension_id": "method", "title": "方法", "instruction": "概括方法"},
        {"dimension_id": "data", "title": "数据", "instruction": "概括数据"},
        {"dimension_id": "metrics", "title": "指标", "instruction": "概括指标"},
    ]
    result = document_profile_node(
        {
            "query": "对比这些文档",
            "compare_sources": sources,
            "compare_docs_by_source": {
                s: [_doc(s, i + 1, 0)] for i, s in enumerate(sources)
            },
            "compare_dimensions": dimensions,
            "is_local": False,
        }
    )

    # 列序 == sources 顺序，每列行序 == dimensions 顺序，不受并发完成顺序影响。
    assert [profile["source"] for profile in result["document_profiles"]] == sources
    for profile in result["document_profiles"]:
        assert [cell["dimension_id"] for cell in profile["cells"]] == [
            "method",
            "data",
            "metrics",
        ]
        assert all(cell["source"] == profile["source"] for cell in profile["cells"])


# 验证 workflow routes to compare subgraph smoke 场景。
def test_workflow_routes_to_compare_subgraph_smoke(monkeypatch):
    engine = FakeEngine([_doc("a.pdf", 1, 0), _doc("b.pdf", 2, 0)])
    monkeypatch.setattr(compare.RetrieverFactory, "get_engine", lambda doc_id: engine)
    monkeypatch.setattr(
        compare_profile.Generator,
        "_get_client",
        lambda is_local=False: FakeCompareLLM(),
    )
    monkeypatch.setattr(
        compare_generator.CompareGeneratorAgent,
        "_generate_conclusion",
        lambda state, table_answer: "两篇文档都提供了可对比事实。[a.pdf:P1][b.pdf:P2]",
    )

    result = workflow.app.invoke(
        {"messages": []},
        config={
            "configurable": {
                "query": "请对比 a.pdf 和 b.pdf 的方法和指标",
                "doc_id": "kb",
                "is_local": False,
            }
        },
    )

    assert result["task_type"] == "compare"
    assert result["compare_sources"] == ["a.pdf", "b.pdf"]
    assert "# 多文档对比" in result["answer"]
    assert "## 方法" in result["answer"]
    assert "- **a.pdf**：" in result["answer"]
    assert (
        result["document_profiles"][0]["cells"][0]["evidence"][0]["source"] == "a.pdf"
    )
    assert result["critique"] == ""
