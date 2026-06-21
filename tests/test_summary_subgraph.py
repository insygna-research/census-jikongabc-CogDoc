from agents import summary_generator
from graph import workflow
from graph.subgraphs import summary
from graph.subgraphs.summary import (
    document_loader_check,
    document_loader_node,
    global_summary_node,
    section_planner_node,
    section_summary_node,
)


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeSummaryLLM:
    def with_structured_output(self, schema):
        return FakeStructuredRouter(schema)

    def invoke(self, messages):
        return FakeMessage("该章节内容来自文档。")


class FakeStructuredRouter:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, messages):
        return self.schema(task_type = "summary", reason = "用户要求总结文档")


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


class FakeEngine:
    def __init__(self, docs):
        self.docs = docs

    def list_sources(self):
        return sorted({doc["meta"]["source"] for doc in self.docs})

    def load_source_chunks(self, source):
        return [doc for doc in self.docs if doc["meta"]["source"] == source]


def test_document_loader_selects_named_source(monkeypatch):
    engine = FakeEngine([_doc("a.pdf", 1, 0), _doc("b.pdf", 1, 0)])
    monkeypatch.setattr(summary.RetrieverFactory, "get_engine", lambda doc_id: engine)

    result = document_loader_node({"query": "请总结 a.pdf", "doc_id": "kb"})

    assert result["summary_source"] == "a.pdf"
    assert [doc["meta"]["source"] for doc in result["summary_docs"]] == ["a.pdf"]
    assert result["steps_trace"][0]["step_name"] == "summary_document_loader"


def test_document_loader_returns_actionable_message_when_ambiguous(monkeypatch):
    engine = FakeEngine([_doc("a.pdf", 1, 0), _doc("b.pdf", 1, 0)])
    monkeypatch.setattr(summary.RetrieverFactory, "get_engine", lambda doc_id: engine)

    result = document_loader_node({"query": "请总结这篇文档", "doc_id": "kb"})

    assert result["summary_docs"] == []
    assert "明确指定" in result["answer"]
    assert "a.pdf" in result["answer"] and "b.pdf" in result["answer"]
    assert document_loader_check(result) == "__end__"


def test_document_loader_check_continues_when_docs_exist():
    assert document_loader_check({"summary_docs": [_doc("a.pdf", 1, 0)]}) == "section_planner_node"


def test_section_planner_node_delegates_to_agent():
    result = section_planner_node({"query": "总结 a.pdf"})

    assert [plan["title"] for plan in result["summary_section_plans"]] == [
        "背景与目标",
        "方案与流程",
        "规则与要求",
        "价值与产出",
        "限制与注意事项",
    ]


def test_section_summary_node_generates_each_planned_section(monkeypatch):
    class FakeLLM:
        def invoke(self, messages):
            return FakeMessage("该章节内容来自文档。[a.pdf:P1]")

    monkeypatch.setattr(summary_generator.Generator, "_get_client", lambda is_local = False: FakeLLM())

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

    assert [section["title"] for section in result["summary_section_results"]] == ["研究问题", "方法"]
    assert result["summary_section_results"][0]["content"] == "该章节内容来自文档。[a.pdf:P1]"
    assert result["summary_section_results"][0]["evidence"][0]["chunk_id"] == "chunk:a.pdf:0"


def test_section_summary_node_retries_local_no_evidence(monkeypatch):
    class FakeLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return FakeMessage("文档中未明确说明。")
            return FakeMessage("文档说明了竞赛背景和目标。")

    fake_llm = FakeLLM()
    monkeypatch.setattr(summary_generator.Generator, "_get_client", lambda is_local = False: fake_llm)

    result = section_summary_node(
        {
            "summary_source": "a.pdf",
            "summary_docs": [_doc("a.pdf", 1, 0)],
            "summary_section_plans": [
                {"section_id": "one", "title": "背景与目标", "instruction": "概括背景目标"},
            ],
            "is_local": True,
        }
    )

    assert fake_llm.calls == 2
    assert result["summary_section_results"][0]["content"] == "文档说明了竞赛背景和目标。[a.pdf:P1]"


def test_section_summary_node_limits_local_context(monkeypatch):
    class FakeLLM:
        def invoke(self, messages):
            return FakeMessage("这是本地摘要。")

    monkeypatch.setattr(summary_generator.Generator, "_get_client", lambda is_local = False: FakeLLM())

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
        "这是本地摘要。[a.pdf:P1][a.pdf:P2][a.pdf:P3][a.pdf:P4]"
        "[a.pdf:P5][a.pdf:P6][a.pdf:P7][a.pdf:P8]"
    )


def test_section_summary_node_uses_six_chunks_for_large_local_documents(monkeypatch):
    class FakeLLM:
        def invoke(self, messages):
            return FakeMessage("这是本地摘要。")

    monkeypatch.setattr(summary_generator.Generator, "_get_client", lambda is_local = False: FakeLLM())

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


def test_section_summary_node_keeps_small_local_documents_intact(monkeypatch):
    class FakeLLM:
        def invoke(self, messages):
            return FakeMessage("这是本地摘要。")

    monkeypatch.setattr(summary_generator.Generator, "_get_client", lambda is_local = False: FakeLLM())

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


def test_global_summary_node_builds_validated_answer():
    result = global_summary_node(
        {
            "summary_source": "a.pdf",
            "summary_docs": [_doc("a.pdf", 1, 0)],
            "summary_section_results": [
                {
                    "section_id": "one",
                    "title": "背景与目标",
                    "content": "文档提出了目标问题。[a.pdf:P1]",
                }
            ],
        }
    )

    assert "# a.pdf 结构化摘要" in result["answer"]
    assert "## 背景与目标" in result["answer"]
    assert result["critique"] == ""
    assert result["sources"][0]["source"] == "a.pdf"
    assert result["evidence"][0]["chunk_id"] == "chunk:a.pdf:0"


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
                    "content": "文档提出了目标问题。[a.pdf:P2]",
                    "evidence": section_evidence,
                },
                {
                    "section_id": "two",
                    "title": "方法",
                    "content": "文档描述了方法。[a.pdf:P2]",
                    "evidence": section_evidence,
                },
            ],
        }
    )

    assert [item["chunk_id"] for item in result["evidence"]] == ["chunk:a.pdf:1"]


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
    assert "页码错误" in result["critique"]


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


def test_workflow_routes_to_summary_subgraph_smoke(monkeypatch):
    engine = FakeEngine([_doc("a.pdf", 1, 0), _doc("b.pdf", 1, 0)])
    monkeypatch.setattr(summary.RetrieverFactory, "get_engine", lambda doc_id: engine)
    monkeypatch.setattr(summary_generator.Generator, "_get_client", lambda is_local = False: FakeSummaryLLM())

    result = workflow.app.invoke(
        {"messages": []},
        config = {
            "configurable": {
                "query": "请总结 a.pdf",
                "doc_id": "kb",
                "is_local": False,
            }
        },
    )

    assert result["task_type"] == "summary"
    assert result["summary_source"] == "a.pdf"
    assert "# a.pdf 结构化摘要" in result["answer"]
    assert "## 背景与目标" in result["answer"]
    assert result["summary_section_results"][0]["evidence"][0]["source"] == "a.pdf"
    assert result["critique"] == ""
