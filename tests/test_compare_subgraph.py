import json
import time
from types import SimpleNamespace

from cogdoc.agents import (
    claim_evidence_verifier,
    compare_generator,
    compare_profile,
)
from cogdoc.agents.claim_evidence_verifier import (
    CLAIM_AUDIT_EXEMPTION_GUIDANCE,
    CLAIM_AUDIT_EXEMPTION_UPSTREAM_ERROR,
    documents_for_state,
)
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


def _support_all_evidence_units(schema, messages):
    payload = json.loads(messages[1]["content"].split("\n", 1)[1].rsplit("\n\n", 1)[0])
    return schema(
        assessments=[
            {
                "unit_id": row["unit_id"],
                "status": "supported",
                "evidence_ids": [row["candidate_evidence_ids"][0]],
                "reason": "测试证据充分",
            }
            for row in payload
        ]
    )


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
    assert result["claim_audit_exemption"] == {
        "reason_code": CLAIM_AUDIT_EXEMPTION_GUIDANCE,
        "answer": result["answer"],
    }
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
        == "该维度内容来自文档[E001]。"
    )
    assert (
        result["document_profiles"][1]["cells"][0]["content"]
        == "该维度内容来自文档[E002]。"
    )
    assert (
        result["document_profiles"][0]["cells"][0]["evidence"][0]["chunk_id"]
        == "chunk:a.pdf:0"
    )
    assert (
        result["document_profiles"][0]["cells"][0]["evidence"][0]["evidence_id"]
        == "E001"
    )


def test_document_profile_node_does_not_generate_terminal_gate_cell(monkeypatch):
    monkeypatch.setattr(
        compare_profile.Generator,
        "_get_client",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("terminal gate must not call the generator")
        ),
    )

    result = document_profile_node(
        {
            "query": "对比 a.pdf 和 b.pdf",
            "compare_sources": ["a.pdf", "b.pdf"],
            "compare_docs_by_source": {},
            "compare_dimensions": _dimensions()[:1],
            "evidence_unit_batch_can_generate": False,
            "evidence_unit_results": [
                {
                    "status": "supported",
                    "gate_action": "terminal",
                    "binding": {"source": source, "dimension_id": "method"},
                    "selected_docs": [_doc(source, index + 1, 0)],
                }
                for index, source in enumerate(["a.pdf", "b.pdf"])
            ],
        }
    )

    cells = [
        cell for profile in result["document_profiles"] for cell in profile["cells"]
    ]
    assert len(cells) == 2
    assert all(cell["failure_stage"] == "verification" for cell in cells)
    assert all(cell["content"] == "本单元证据处理未完成，请重试。" for cell in cells)


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


# 验证 document profile node isolates llm error per cell 场景。
def test_document_profile_node_isolates_llm_error_per_cell(monkeypatch):
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

    cells = [
        cell for profile in result["document_profiles"] for cell in profile["cells"]
    ]
    assert len(cells) == 2
    assert all(cell["content"] == "本单元证据处理未完成，请重试。" for cell in cells)
    assert all(cell["status"] == "generation_error" for cell in cells)
    assert all(cell["error_class"] == "RuntimeError" for cell in cells)
    assert document_profile_check(result) == "compare_table_node"


def test_compare_citation_rejects_cross_source_cell_evidence():
    state = {
        "compare_sources": ["a.pdf", "b.pdf"],
        "document_profiles": [
            {
                "source": "a.pdf",
                "cells": [
                    {
                        "dimension_id": "method",
                        "source": "a.pdf",
                        "content": "A 的方法来自另一篇文档[E002]。",
                    }
                ],
            },
            {
                "source": "b.pdf",
                "cells": [
                    {
                        "dimension_id": "method",
                        "source": "b.pdf",
                        "content": "B 的方法有本地证据[E002]。",
                    }
                ],
            },
        ],
        "compare_table_answer": (
            "# 多文档对比\n- **a.pdf**：A 的方法来自另一篇文档[E002]。\n"
            "- **b.pdf**：B 的方法有本地证据[E002]。"
        ),
        "answer": (
            "# 多文档对比\n- **a.pdf**：A 的方法来自另一篇文档[E002]。\n"
            "- **b.pdf**：B 的方法有本地证据[E002]。"
        ),
        "evidence_ledger": [
            {
                "evidence_id": "E002",
                "chunk_id": "b-method",
                "source": "b.pdf",
                "page": 1,
                "span_start": 0,
                "span_end": 10,
                "display_citation": "[b.pdf:P1]",
            }
        ],
    }

    result = citation_node(state)

    assert "【单元格证据越界】" in result["critique"]
    assert "a.pdf/method:E002" in result["critique"]
    assert "证据引用未通过校验" in result["answer"]


def test_compare_skips_conclusion_when_any_cell_is_operationally_incomplete(
    monkeypatch,
):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("不完整矩阵不得调用结论模型")

    monkeypatch.setattr(
        compare_generator.CompareGeneratorAgent,
        "_generate_conclusion",
        fail_if_called,
    )
    state = {
        "compare_sources": ["a.pdf", "b.pdf"],
        "compare_dimensions": _dimensions()[:1],
        "compare_docs_by_source": {},
        "document_profiles": [
            {
                "source": "a.pdf",
                "cells": [
                    {
                        "dimension_id": "method",
                        "source": "a.pdf",
                        "content": "本单元证据处理未完成，请重试。",
                        "status": "retrieval_error",
                        "evidence": [],
                    }
                ],
            },
            {
                "source": "b.pdf",
                "cells": [
                    {
                        "dimension_id": "method",
                        "source": "b.pdf",
                        "content": "文档中未明确说明。",
                        "status": "no_evidence",
                        "evidence": [],
                    }
                ],
            },
        ],
    }

    result = compare_table_node(state)

    assert result["compare_conclusion"] == ""
    assert "部分对比单元处理未完成" in result["compare_conclusion_warning"]
    assert "本单元证据处理未完成，请重试。" in result["answer"]


def test_compare_all_operational_cells_bypass_citations_and_claim_audit(monkeypatch):
    monkeypatch.setattr(
        compare_generator.CompareGeneratorAgent,
        "_generate_conclusion",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("operational terminal cells must not generate a conclusion")
        ),
    )
    monkeypatch.setattr(
        claim_evidence_verifier,
        "get_settings",
        lambda: SimpleNamespace(claim_verification_enabled=True),
    )
    state = {
        "task_type": "compare",
        "compare_sources": ["a.pdf", "b.pdf"],
        "compare_dimensions": _dimensions()[:1],
        "compare_docs_by_source": {},
        "evidence_ledger": [],
        "document_profiles": [
            {
                "source": source,
                "cells": [
                    {
                        "dimension_id": "method",
                        "source": source,
                        "content": "本单元证据处理未完成，请重试。",
                        "status": "verification_error",
                        "failure_stage": "verification",
                        "evidence": [],
                    }
                ],
            }
            for source in ("a.pdf", "b.pdf")
        ],
    }

    table_result = compare_table_node(state)
    citation_result = citation_node({**state, **table_result})
    candidate = {**state, **table_result, **citation_result}
    audit_result = workflow.claim_audit_node(candidate)
    finalized = workflow.citation_finalize_node({**candidate, **audit_result})

    assert citation_result["critique"] == ""
    assert "引用校验警告" not in citation_result["answer"]
    assert table_result["error"] == "compare_evidence_units_incomplete"
    assert table_result["claim_audit_exemption"] == {
        "reason_code": CLAIM_AUDIT_EXEMPTION_UPSTREAM_ERROR,
        "answer": table_result["answer"],
    }
    assert audit_result["claim_audit"]["status"] == "not_run"
    assert audit_result["claim_audit"]["reason_code"] == (
        CLAIM_AUDIT_EXEMPTION_UPSTREAM_ERROR
    )
    assert workflow.claim_audit_check(audit_result) == "citation_finalize_node"
    assert finalized.get("answer", candidate["answer"]) == table_result["answer"]
    assert finalized["citation_ledger"] == []


def test_compare_citation_validates_only_generated_cells():
    state = {
        "compare_sources": ["a.pdf", "b.pdf"],
        "compare_dimensions": _dimensions()[:1],
        "compare_docs_by_source": {},
        "evidence_ledger": [
            {
                "evidence_id": "E001",
                "chunk_id": "a-method",
                "source": "a.pdf",
                "page": 1,
                "span_start": 0,
                "span_end": 10,
                "display_citation": "[a.pdf:P1]",
            }
        ],
        "document_profiles": [
            {
                "source": "a.pdf",
                "cells": [
                    {
                        "dimension_id": "method",
                        "source": "a.pdf",
                        "content": "A 使用分层检索[E001]。",
                        "status": "generated",
                        "evidence": [],
                    }
                ],
            },
            {
                "source": "b.pdf",
                "cells": [
                    {
                        "dimension_id": "method",
                        "source": "b.pdf",
                        "content": "本单元证据处理未完成，请重试。",
                        "status": "retrieval_error",
                        "evidence": [],
                    }
                ],
            },
        ],
    }

    table_result = compare_table_node(state)
    result = citation_node({**state, **table_result})

    assert result["critique"] == ""
    assert "引用校验警告" not in result["answer"]
    assert "claim_audit_exemption" not in table_result


# 验证 compare table node builds dimension blocks and conclusion 场景。
def test_compare_table_node_builds_dimension_blocks_and_conclusion(monkeypatch):
    monkeypatch.setattr(
        compare_generator.CompareGeneratorAgent,
        "_generate_conclusion",
        lambda state, table_answer: "a.pdf 强调方法，b.pdf 强调指标[E001][E002]。",
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
                        "content": "方法 A[E001]。",
                        "evidence": [],
                    },
                    {
                        "dimension_id": "metrics",
                        "source": "a.pdf",
                        "content": "指标 A[E001]。",
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
                        "content": "方法 B[E002]。",
                        "evidence": [],
                    },
                    {
                        "dimension_id": "metrics",
                        "source": "b.pdf",
                        "content": "指标 B[E002]。",
                        "evidence": [],
                    },
                ],
            },
        ],
    }
    table_result = compare_table_node(state)
    result = citation_node({**state, **table_result})

    assert "## 方法" in result["answer"]
    assert "- **a.pdf**：方法 A[E001]。" in result["answer"]
    assert "- **b.pdf**：方法 B[E002]。" in result["answer"]
    assert "## 简短结论" in result["answer"]
    assert "引用校验警告" not in result["answer"]
    assert [entry["evidence_id"] for entry in result["evidence_ledger"]] == [
        "E001",
        "E002",
    ]
    assert [
        result["compare_docs_by_source"][source][0]["retrieval"]["evidence_id"]
        for source in state["compare_sources"]
    ] == ["E001", "E002"]

    finalized = workflow.citation_finalize_node(
        {"task_type": "compare", **state, **table_result, **result}
    )
    assert "[E001]" not in finalized["answer"]
    assert "[E002]" not in finalized["answer"]
    assert [entry["chunk_id"] for entry in finalized["citation_ledger"]] == [
        "chunk:a.pdf:0",
        "chunk:b.pdf:0",
    ]


def test_compare_conclusion_normalizes_e1000_before_terminator():
    result = compare_generator._normalize_evidence_citation_placement(
        "第一句。[E1000] 第二句！[E001][E1001]"
    )

    assert result == "第一句[E1000]。 第二句[E001][E1001]！"


def test_compare_claim_documents_exclude_unseen_same_page_child():
    seen = _doc("a.pdf", 1, 0)
    unseen = _doc("a.pdf", 1, 1)
    other = _doc("b.pdf", 2, 0)
    state = {
        "task_type": "compare",
        "compare_sources": ["a.pdf", "b.pdf"],
        "compare_docs_by_source": {
            "a.pdf": [seen, unseen],
            "b.pdf": [other],
        },
        "compare_dimensions": _dimensions()[:1],
        "document_profiles": [
            {
                "source": "a.pdf",
                "cells": [
                    {
                        "dimension_id": "method",
                        "source": "a.pdf",
                        "content": "方法 A[E001]。",
                        "evidence": [{"chunk_id": "chunk:a.pdf:0"}],
                    }
                ],
            },
            {
                "source": "b.pdf",
                "cells": [
                    {
                        "dimension_id": "method",
                        "source": "b.pdf",
                        "content": "方法 B[E003]。",
                        "evidence": [{"chunk_id": "chunk:b.pdf:0"}],
                    }
                ],
            },
        ],
        "is_local": True,
    }

    final = compare_table_node(state)
    docs = documents_for_state({**state, **final})

    assert [doc["meta"]["chunk_id"] for doc in docs] == [
        "chunk:a.pdf:0",
        "chunk:b.pdf:0",
    ]


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
                        "content": "方法 A[E001]。",
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
                        "content": "方法 B[E002]。",
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
                        "content": "方法 A[E001]。",
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
                        "content": "方法 B[E002]。",
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
                        "content": "方法 A[E999]。",
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
                        "content": "方法 B[E002]。",
                        "evidence": [],
                    },
                ],
            },
        ],
    }
    table_result = compare_table_node(state)
    result = citation_node({**state, **table_result})

    assert "- **a.pdf**：方法 A[E999]。" in result["answer"]
    assert "- **b.pdf**：方法 B[E002]。" in result["answer"]
    assert "引用校验警告" in result["answer"]
    assert "不在本次证据账本" in result["critique"]


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
                        "content": "方法 A[E001]。",
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
                        "content": "方法 B[E002]。",
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
    assert "- **a.pdf**：方法 A[E001]。" in result["answer"]
    assert "- **b.pdf**：方法 B[E002]。" in result["answer"]
    assert "引用校验警告" in result["answer"]
    assert result["critique"]
    assert "每个来自证据的事实句" not in result["answer"]

    finalized = workflow.citation_finalize_node(
        {"task_type": "compare", **state, **table_result, **result}
    )
    occurrences = {
        entry["evidence_id"]: len(entry["occurrences"])
        for entry in finalized["citation_ledger"]
    }
    assert occurrences == {"E001": 1, "E002": 1}
    assert "[E001]" not in finalized["answer"]
    assert "[E002]" not in finalized["answer"]


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
                        "content": "方法 A[E999]。",
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
                        "content": "方法 B[E002]。",
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
    assert "没有 Evidence ID" in result["critique"]
    assert "不在本次证据账本" in result["critique"]


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
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("all no-evidence cells must not generate a conclusion")
        ),
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
                        "status": "no_evidence",
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
                        "status": "no_evidence",
                        "evidence": [],
                    },
                ],
            },
        ],
    }
    table_result = compare_table_node(state)
    result = citation_node({**state, **table_result})
    monkeypatch.setattr(
        claim_evidence_verifier,
        "get_settings",
        lambda: SimpleNamespace(claim_verification_enabled=True),
    )
    audit = workflow.claim_audit_node(
        {"task_type": "compare", **state, **table_result, **result}
    )

    assert "- **a.pdf**：文档中未明确说明。" in result["answer"]
    assert "- **b.pdf**：文档中未明确说明。" in result["answer"]
    assert "引用校验警告" not in result["answer"]
    assert result["critique"] == ""
    assert table_result["compare_conclusion"] == ""
    assert audit["claim_audit"]["status"] == "not_run"
    assert audit["claim_audit"]["reason_code"] == "no_evidence_units"


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
    assert [
        result["compare_docs_by_source"][source][0]["retrieval"]["evidence_id"]
        for source in sources
    ] == ["E001", "E002", "E003"]
    assert [entry["evidence_id"] for entry in result["evidence_ledger"]] == [
        "E001",
        "E002",
        "E003",
    ]
    for index, profile in enumerate(result["document_profiles"], start=1):
        assert [cell["dimension_id"] for cell in profile["cells"]] == [
            "method",
            "data",
            "metrics",
        ]
        assert all(cell["source"] == profile["source"] for cell in profile["cells"])
        assert all(
            cell["content"].endswith(f"[E{index:03d}]。") for cell in profile["cells"]
        )


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
        lambda state, table_answer: "两篇文档都提供了可对比事实[E001][E002]。",
    )

    result = workflow.app.invoke(
        {"messages": []},
        config={
            "configurable": {
                "query": "请对比 a.pdf 和 b.pdf 的方法和指标",
                "doc_id": "kb",
                "is_local": False,
                "evidence_unit_structured_client": _support_all_evidence_units,
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
