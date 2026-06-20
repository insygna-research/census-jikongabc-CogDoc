from agents import summary_generator
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
    # DocumentLoader 按用户问题中的文件名加载单文档 chunks。
    engine = FakeEngine([_doc("a.pdf", 1, 0), _doc("b.pdf", 1, 0)])
    monkeypatch.setattr(summary.RetrieverFactory, "get_engine", lambda doc_id: engine)

    result = document_loader_node({"query": "请总结 a.pdf", "doc_id": "kb"})

    assert result["summary_source"] == "a.pdf"
    assert [doc["meta"]["source"] for doc in result["summary_docs"]] == ["a.pdf"]
    assert result["steps_trace"][0]["step_name"] == "summary_document_loader"


def test_document_loader_returns_actionable_message_when_ambiguous(monkeypatch):
    # 多文档且用户未点名时返回可操作提示。
    engine = FakeEngine([_doc("a.pdf", 1, 0), _doc("b.pdf", 1, 0)])
    monkeypatch.setattr(summary.RetrieverFactory, "get_engine", lambda doc_id: engine)

    result = document_loader_node({"query": "请总结这篇文档", "doc_id": "kb"})

    assert result["summary_docs"] == []
    assert "明确指定" in result["answer"]
    assert "a.pdf" in result["answer"] and "b.pdf" in result["answer"]
    assert document_loader_check(result) == "__end__"


def test_document_loader_check_continues_when_docs_exist():
    # 文档加载成功后才进入章节规划。
    assert document_loader_check({"summary_docs": [_doc("a.pdf", 1, 0)]}) == "section_planner_node"


def test_section_planner_node_delegates_to_agent():
    # Summary 子图章节规划节点输出固定摘要章节。
    result = section_planner_node({"query": "总结 a.pdf"})

    assert [plan["title"] for plan in result["summary_section_plans"]] == [
        "研究问题",
        "方法与方案",
        "实验与要求",
        "结论与价值",
        "局限与注意事项",
    ]


def test_section_summary_node_generates_each_planned_section(monkeypatch):
    # SectionSummary 为每个章节生成一段带引用摘要。
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


def test_global_summary_node_builds_validated_answer():
    # GlobalSummary 整合章节并复用 citation checker。
    result = global_summary_node(
        {
            "summary_source": "a.pdf",
            "summary_docs": [_doc("a.pdf", 1, 0)],
            "summary_section_results": [
                {
                    "section_id": "one",
                    "title": "研究问题",
                    "content": "文档提出了目标问题。[a.pdf:P1]",
                }
            ],
        }
    )

    assert "# a.pdf 结构化摘要" in result["answer"]
    assert "## 研究问题" in result["answer"]
    assert result["critique"] == ""
    assert result["sources"][0]["source"] == "a.pdf"
    assert result["evidence"][0]["chunk_id"] == "chunk:a.pdf:0"


def test_global_summary_node_blocks_invalid_citation():
    # 引用页码不在文档 chunk 内时保留摘要并附加校验警告。
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
