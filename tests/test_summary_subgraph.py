import json
from types import SimpleNamespace

from cogdoc.agents import claim_evidence_verifier, summary_generator
from cogdoc.agents.claim_evidence_verifier import (
    CLAIM_AUDIT_EXEMPTION_GUIDANCE,
    CLAIM_AUDIT_EXEMPTION_UPSTREAM_ERROR,
)
from cogdoc.graph import workflow
from cogdoc.graph.subgraphs import summary
from cogdoc.graph.subgraphs.summary import (
    document_loader_check,
    document_loader_node,
    global_summary_node,
    section_planner_node,
    section_summary_node,
)


# 定义 FakeMessage 数据结构。
class FakeMessage:
    # 初始化 FakeMessage 实例。
    def __init__(self, content):
        self.content = content


# 定义 FakeSummaryLLM 数据结构。
class FakeSummaryLLM:
    # 返回支持结构化输出的测试替身。
    def with_structured_output(self, schema):
        return FakeStructuredRouter(schema)

    # 调用测试替身并返回预设结果。
    def invoke(self, messages):
        return FakeMessage("该章节内容来自文档。")


# 定义 FakeStructuredRouter 数据结构。
class FakeStructuredRouter:
    # 初始化 FakeStructuredRouter 实例。
    def __init__(self, schema):
        self.schema = schema

    # 调用测试替身并返回预设结果。
    def invoke(self, messages):
        return self.schema(task_type="summary", reason="用户要求总结文档")


def _support_all_evidence_units(schema, messages):
    payload = json.loads(messages[1]["content"])["untrusted_data"]["evidence_units"]
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


# 验证 document loader selects named source 场景。
def test_document_loader_selects_named_source(monkeypatch):
    engine = FakeEngine([_doc("a.pdf", 1, 0), _doc("b.pdf", 1, 0)])
    monkeypatch.setattr(summary.RetrieverFactory, "get_engine", lambda doc_id: engine)

    result = document_loader_node({"query": "请总结 a.pdf", "doc_id": "kb"})

    assert result["summary_source"] == "a.pdf"
    assert [doc["meta"]["source"] for doc in result["summary_docs"]] == ["a.pdf"]
    assert result["steps_trace"][0]["step_name"] == "summary_document_loader"


# 验证 document loader returns actionable message when ambiguous 场景。
def test_document_loader_returns_actionable_message_when_ambiguous(monkeypatch):
    engine = FakeEngine([_doc("a.pdf", 1, 0), _doc("b.pdf", 1, 0)])
    monkeypatch.setattr(summary.RetrieverFactory, "get_engine", lambda doc_id: engine)

    result = document_loader_node({"query": "请总结这篇文档", "doc_id": "kb"})

    assert result["summary_docs"] == []
    assert "明确指定" in result["answer"]
    assert "a.pdf" in result["answer"] and "b.pdf" in result["answer"]
    assert result["claim_audit_exemption"] == {
        "reason_code": CLAIM_AUDIT_EXEMPTION_GUIDANCE,
        "answer": result["answer"],
    }
    assert document_loader_check(result) == "__end__"


# 验证 document loader resolves referential source 场景。
def test_document_loader_resolves_referential_source(monkeypatch):
    engine = FakeEngine([_doc("a.pdf", 1, 0), _doc("b.pdf", 1, 0)])
    monkeypatch.setattr(summary.RetrieverFactory, "get_engine", lambda doc_id: engine)
    # 「这个文件」字面匹配不到，靠近期对话消解出 a.pdf。
    monkeypatch.setattr(
        summary, "resolve_summary_source", lambda *args, **kwargs: "a.pdf"
    )

    result = document_loader_node(
        {
            "query": "总结这个文件",
            "doc_id": "kb",
            "chat_history": [
                {"role": "user", "content": "讲讲 a.pdf", "timestamp": None}
            ],
        }
    )

    assert result["summary_source"] == "a.pdf"
    assert result["steps_trace"][0]["step_name"] == "summary_source_resolution"


# 验证 document loader skips resolution without history 场景。
def test_document_loader_skips_resolution_without_history(monkeypatch):
    engine = FakeEngine([_doc("a.pdf", 1, 0), _doc("b.pdf", 1, 0)])
    monkeypatch.setattr(summary.RetrieverFactory, "get_engine", lambda doc_id: engine)

    # 记录失败结果。
    def fail(*args, **kwargs):
        raise AssertionError("无历史时不应触发指代消解")

    monkeypatch.setattr(summary, "resolve_summary_source", fail)

    result = document_loader_node({"query": "总结这个文件", "doc_id": "kb"})

    assert result["summary_docs"] == []
    assert "可直接说出文件名" in result["answer"]


# 验证 document loader check continues when docs exist 场景。
def test_document_loader_check_continues_when_docs_exist():
    assert (
        document_loader_check({"summary_docs": [_doc("a.pdf", 1, 0)]})
        == "section_planner_node"
    )


# 验证 section planner node delegates to agent 场景。
def test_section_planner_node_delegates_to_agent():
    result = section_planner_node({"query": "总结 a.pdf"})

    assert [plan["title"] for plan in result["summary_section_plans"]] == [
        "背景与目标",
        "方案与流程",
        "规则与要求",
        "价值与产出",
        "限制与注意事项",
    ]


# 验证 section summary node generates each planned section 场景。
def test_section_summary_node_generates_each_planned_section(monkeypatch):
    # 定义 FakeLLM 数据结构。
    class FakeLLM:
        # 调用测试替身并返回预设结果。
        def invoke(self, messages):
            return FakeMessage("该章节内容来自文档。[a.pdf:P1]")

    monkeypatch.setattr(
        summary_generator.Generator, "_get_client", lambda is_local=False: FakeLLM()
    )

    result = section_summary_node(
        {
            "summary_source": "a.pdf",
            "summary_docs": [_doc("a.pdf", 1, 0)],
            "summary_section_plans": [
                {"section_id": "one", "title": "研究问题", "instruction": "概括问题"},
                {"section_id": "two", "title": "方法", "instruction": "概括方法"},
            ],
        }
    )

    assert [section["title"] for section in result["summary_section_results"]] == [
        "研究问题",
        "方法",
    ]
    assert (
        result["summary_section_results"][0]["content"] == "该章节内容来自文档[E001]。"
    )
    assert (
        result["summary_section_results"][0]["evidence"][0]["chunk_id"]
        == "chunk:a.pdf:0"
    )
    assert result["summary_section_results"][0]["evidence"][0]["evidence_id"] == (
        "E001"
    )


def test_section_summary_node_does_not_generate_terminal_gate_unit(monkeypatch):
    monkeypatch.setattr(
        summary_generator.Generator,
        "_get_client",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("terminal gate must not call the generator")
        ),
    )

    result = section_summary_node(
        {
            "summary_source": "a.pdf",
            "summary_docs": [],
            "summary_section_plans": [
                {"section_id": "one", "title": "研究问题", "instruction": "概括问题"}
            ],
            "evidence_unit_batch_can_generate": False,
            "evidence_unit_results": [
                {
                    "status": "supported",
                    "gate_action": "terminal",
                    "binding": {"section_id": "one"},
                    "selected_docs": [_doc("a.pdf", 1, 0)],
                }
            ],
        }
    )

    section = result["summary_section_results"][0]
    assert section["status"] == "supported"
    assert section["failure_stage"] == "verification"
    assert section["content"] == "本单元证据处理未完成，请重试。"


# 验证 section summary node retries local no evidence 场景。
def test_section_summary_node_retries_local_no_evidence(monkeypatch):
    # 定义 FakeLLM 数据结构。
    class FakeLLM:
        # 初始化 FakeLLM 实例。
        def __init__(self):
            self.calls = 0

        # 调用测试替身并返回预设结果。
        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return FakeMessage("文档中未明确说明。")
            return FakeMessage("文档说明了竞赛背景和目标。")

    fake_llm = FakeLLM()
    monkeypatch.setattr(
        summary_generator.Generator, "_get_client", lambda is_local=False: fake_llm
    )

    result = section_summary_node(
        {
            "summary_source": "a.pdf",
            "summary_docs": [_doc("a.pdf", 1, 0)],
            "summary_section_plans": [
                {
                    "section_id": "one",
                    "title": "背景与目标",
                    "instruction": "概括背景目标",
                },
            ],
            "is_local": True,
        }
    )

    assert fake_llm.calls == 2
    assert (
        result["summary_section_results"][0]["content"]
        == "文档说明了竞赛背景和目标[E001]。"
    )


# 验证 section summary node limits local context 场景。
def test_section_summary_node_limits_local_context(monkeypatch):
    # 定义 FakeLLM 数据结构。
    class FakeLLM:
        # 调用测试替身并返回预设结果。
        def invoke(self, messages):
            return FakeMessage("这是本地摘要。")

    monkeypatch.setattr(
        summary_generator.Generator, "_get_client", lambda is_local=False: FakeLLM()
    )

    result = section_summary_node(
        {
            "summary_source": "a.pdf",
            "summary_docs": [_doc("a.pdf", idx + 1, idx) for idx in range(10)],
            "summary_section_plans": [
                {"section_id": "one", "title": "研究问题", "instruction": "概括问题"},
            ],
            "is_local": True,
        }
    )

    section = result["summary_section_results"][0]

    assert [item["page"] for item in section["evidence"]] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert section["content"] == (
        "这是本地摘要[E001][E002][E003][E004][E005][E006][E007][E008]。"
    )


# 验证 section summary node uses six chunks for large local documents 场景。
def test_section_summary_node_uses_six_chunks_for_large_local_documents(monkeypatch):
    # 定义 FakeLLM 数据结构。
    class FakeLLM:
        # 调用测试替身并返回预设结果。
        def invoke(self, messages):
            return FakeMessage("这是本地摘要。")

    monkeypatch.setattr(
        summary_generator.Generator, "_get_client", lambda is_local=False: FakeLLM()
    )

    result = section_summary_node(
        {
            "summary_source": "a.pdf",
            "summary_docs": [_doc("a.pdf", idx + 1, idx) for idx in range(18)],
            "summary_section_plans": [
                {"section_id": "one", "title": "研究问题", "instruction": "概括问题"},
            ],
            "is_local": True,
        }
    )

    section = result["summary_section_results"][0]

    assert [item["page"] for item in section["evidence"]] == [1, 2, 3, 4, 5, 6]


# 验证 section summary node keeps small local documents intact 场景。
def test_section_summary_node_keeps_small_local_documents_intact(monkeypatch):
    # 定义 FakeLLM 数据结构。
    class FakeLLM:
        # 调用测试替身并返回预设结果。
        def invoke(self, messages):
            return FakeMessage("这是本地摘要。")

    monkeypatch.setattr(
        summary_generator.Generator, "_get_client", lambda is_local=False: FakeLLM()
    )

    result = section_summary_node(
        {
            "summary_source": "a.pdf",
            "summary_docs": [_doc("a.pdf", idx + 1, idx) for idx in range(8)],
            "summary_section_plans": [
                {"section_id": "one", "title": "研究问题", "instruction": "概括问题"},
            ],
            "is_local": True,
        }
    )

    section = result["summary_section_results"][0]

    assert [item["page"] for item in section["evidence"]] == [1, 2, 3, 4, 5, 6, 7, 8]


# 验证 global summary node builds validated answer 场景。
def test_global_summary_node_builds_validated_answer():
    result = global_summary_node(
        {
            "summary_source": "a.pdf",
            "summary_docs": [_doc("a.pdf", 1, 0)],
            "summary_section_results": [
                {
                    "section_id": "one",
                    "title": "背景与目标",
                    "content": "文档提出了目标问题[E001]。",
                }
            ],
        }
    )

    assert "# a.pdf 结构化摘要" in result["answer"]
    assert "## 背景与目标" in result["answer"]
    assert result["critique"] == ""
    assert result["sources"][0]["source"] == "a.pdf"
    assert result["evidence"][0]["chunk_id"] == "chunk:a.pdf:0"


# 验证 global summary node dedupes section evidence 场景。
def test_global_summary_node_dedupes_section_evidence():
    first_doc = _doc("a.pdf", 1, 0)
    second_doc = _doc("a.pdf", 2, 1)
    section_evidence = [
        {
            "chunk_id": "chunk:a.pdf:1",
            "chunk_index": 1,
            "source": "a.pdf",
            "page": 2,
            "page_start": 2,
            "page_end": 2,
            "text_preview": "a.pdf p2 c1",
        }
    ]

    result = global_summary_node(
        {
            "summary_source": "a.pdf",
            "summary_docs": [first_doc, second_doc],
            "summary_section_results": [
                {
                    "section_id": "one",
                    "title": "研究问题",
                    "content": "文档提出了目标问题[E002]。",
                    "evidence": section_evidence,
                },
                {
                    "section_id": "two",
                    "title": "方法",
                    "content": "文档描述了方法[E002]。",
                    "evidence": section_evidence,
                },
            ],
        }
    )

    assert [item["chunk_id"] for item in result["evidence"]] == ["chunk:a.pdf:1"]


def test_global_summary_same_page_siblings_keep_exact_distinct_ids():
    seen = _doc("a.pdf", 1, 0)
    unseen = _doc("a.pdf", 1, 1)
    section_results = [
        {
            "section_id": "one",
            "title": "研究问题",
            "content": "文档提出了目标问题[E001]。",
            "evidence": [
                {
                    "chunk_id": "chunk:a.pdf:0",
                    "source": "a.pdf",
                    "page": 1,
                }
            ],
        }
    ]
    state = {
        "task_type": "summary",
        "summary_source": "a.pdf",
        "summary_docs": [seen, unseen],
        "summary_section_results": section_results,
    }

    final = global_summary_node(state)

    assert [doc["retrieval"]["evidence_id"] for doc in final["summary_docs"]] == [
        "E001",
        "E002",
    ]
    assert "文档提出了目标问题[E001]。" in final["answer"]
    assert [entry["chunk_id"] for entry in final["evidence_ledger"]] == [
        "chunk:a.pdf:0",
        "chunk:a.pdf:1",
    ]


# 验证 global summary node blocks invalid citation 场景。
def test_global_summary_node_blocks_invalid_citation():
    result = global_summary_node(
        {
            "summary_source": "a.pdf",
            "summary_docs": [_doc("a.pdf", 1, 0)],
            "summary_section_results": [
                {
                    "section_id": "one",
                    "title": "研究问题",
                    "content": "文档提出了目标问题。[a.pdf:P99]",
                }
            ],
        }
    )

    assert "# a.pdf 结构化摘要" in result["answer"]
    assert "引用校验警告" in result["answer"]
    assert "Evidence ID" in result["critique"]


# 验证 global summary node flags substantive uncited section 场景。
def test_global_summary_node_flags_substantive_uncited_section():
    result = global_summary_node(
        {
            "summary_source": "a.pdf",
            "summary_docs": [_doc("a.pdf", 1, 0)],
            "summary_section_results": [
                {
                    "section_id": "one",
                    "title": "研究问题",
                    "content": "文档提出了明确的研究目标。",
                    "evidence": [],
                }
            ],
        }
    )

    assert "引用校验警告" in result["answer"]
    assert result["critique"]


# 验证 global summary node accepts all no evidence sections 场景。
def test_global_summary_node_accepts_all_no_evidence_sections():
    result = global_summary_node(
        {
            "summary_source": "a.pdf",
            "summary_docs": [_doc("a.pdf", 1, 0)],
            "summary_section_results": [
                {
                    "section_id": "one",
                    "title": "研究问题",
                    "content": "文档中未明确说明。",
                    "evidence": [],
                }
            ],
        }
    )

    assert "# a.pdf 结构化摘要" in result["answer"]
    assert "引用校验警告" not in result["answer"]
    assert result["critique"] == ""


def test_global_summary_node_returns_all_no_evidence_answer_without_docs():
    result = global_summary_node(
        {
            "summary_source": "a.pdf",
            "summary_docs": [],
            "summary_section_results": [
                {
                    "section_id": "one",
                    "title": "研究问题",
                    "content": "文档中未明确说明。",
                    "status": "no_evidence",
                    "evidence": [],
                }
            ],
        }
    )

    assert "# a.pdf 结构化摘要" in result["answer"]
    assert "文档中未明确说明。" in result["answer"]
    assert result["evidence"] == []
    assert result["critique"] == ""


def test_global_summary_node_returns_operational_failure_answer_without_docs(
    monkeypatch,
):
    result = global_summary_node(
        {
            "summary_source": "a.pdf",
            "summary_docs": [],
            "summary_section_results": [
                {
                    "section_id": "one",
                    "title": "研究问题",
                    "content": "本单元证据处理未完成，请重试。",
                    "status": "verification_error",
                    "failure_stage": "verification",
                    "evidence": [],
                }
            ],
        }
    )

    assert "本单元证据处理未完成，请重试。" in result["answer"]
    assert result["critique"] == ""
    assert result["claim_audit_exemption"]["reason_code"] == (
        CLAIM_AUDIT_EXEMPTION_UPSTREAM_ERROR
    )
    assert result["error"] == "summary_evidence_units_incomplete"

    monkeypatch.setattr(
        claim_evidence_verifier,
        "get_settings",
        lambda: SimpleNamespace(
            claim_verification_enabled=True
        ),
    )
    audit = workflow.claim_audit_node({"task_type": "summary", **result})

    assert audit["claim_audit"]["status"] == "not_run"
    assert audit["claim_audit"]["reason_code"] == (
        CLAIM_AUDIT_EXEMPTION_UPSTREAM_ERROR
    )


# 验证 workflow routes to summary subgraph smoke 场景。
def test_workflow_routes_to_summary_subgraph_smoke(monkeypatch):
    engine = FakeEngine([_doc("a.pdf", 1, 0), _doc("b.pdf", 1, 0)])
    monkeypatch.setattr(summary.RetrieverFactory, "get_engine", lambda doc_id: engine)
    monkeypatch.setattr(
        summary_generator.Generator,
        "_get_client",
        lambda is_local=False: FakeSummaryLLM(),
    )

    result = workflow.app.invoke(
        {"messages": []},
        config={
            "configurable": {
                "query": "请总结 a.pdf",
                "doc_id": "kb",
                "is_local": False,
                "evidence_unit_structured_client": _support_all_evidence_units,
            }
        },
    )

    assert result["task_type"] == "summary"
    assert result["summary_source"] == "a.pdf"
    assert "# a.pdf 结构化摘要" in result["answer"]
    assert "## 背景与目标" in result["answer"]
    assert result["summary_section_results"][0]["evidence"][0]["source"] == "a.pdf"
    assert result["critique"] == ""
