import json
from contextlib import nullcontext
from types import SimpleNamespace

from cogdoc.agents import compare_profile, summary_generator
from cogdoc.graph.subgraphs import compare, summary
from cogdoc.tools.reranker import BGEReranker


def _doc(source: str, chunk_id: str, text: str) -> dict:
    return {
        "text": text,
        "meta": {
            "chunk_id": chunk_id,
            "source_sha256": f"sha:{source}",
            "local_chunk_index": 0,
            "chunk_index": 0,
            "source": source,
            "page": 1,
            "page_start": 1,
            "page_end": 1,
            "origin": "file",
        },
    }


class _Engine:
    def __init__(self, docs):
        self.docs = list(docs)

    def search(self, query, top_k=3, *, scope=None):
        docs = self.docs
        if scope is not None and scope.allowed_sources:
            docs = [
                doc for doc in docs if doc["meta"]["source"] in scope.allowed_sources
            ]
        return docs[:top_k]

    def load_source_chunks(self, source):
        return [doc for doc in self.docs if doc["meta"]["source"] == source]


class _NoDerived:
    def search(self, kb_id, query, top_k=3, *, scope=None):
        return []


class _NoFeedback:
    def boosts_for_query(self, kb_id, query):
        return {}


class _Message:
    def __init__(self, content):
        self.content = content


class _LLM:
    def invoke(self, messages):
        return _Message("证据闭集支持该单元。")


def _support_all_units(schema, messages):
    payload = json.loads(messages[1]["content"])["untrusted_data"]["evidence_units"]
    return schema(
        assessments=[
            {
                "unit_id": row["unit_id"],
                "status": "supported",
                "evidence_ids": [row["candidate_evidence_ids"][0]],
                "reason": "测试闭集明确支持该单元",
            }
            for row in payload
        ]
    )


def _reject_all_units(schema, messages):
    payload = json.loads(messages[1]["content"])["untrusted_data"]["evidence_units"]
    return schema(
        assessments=[
            {
                "unit_id": row["unit_id"],
                "status": "no_evidence",
                "evidence_ids": [],
                "reason": "候选闭集没有直接证据",
            }
            for row in payload
        ]
    )


def _config(structured_client=_support_all_units):
    return {
        "configurable": {
            "evidence_unit_structured_client": structured_client,
            "state_runtime": SimpleNamespace(
                derived_knowledge_retriever=_NoDerived(),
                retrieval_feedback_store=_NoFeedback(),
            )
        }
    }


def _capture_plan_budget(captured):
    def retrieve(units, **kwargs):
        budget = kwargs["budget"]
        budget.validate_plan_capacity(units)
        captured["unit_count"] = len(units)
        captured["budget"] = budget
        metrics = {
            "ready_count": 0,
            "no_evidence_count": 0,
            "retrieval_error_count": 0,
            "verification_error_count": 0,
            "budget_exhausted_count": 0,
        }
        return SimpleNamespace(
            metrics=metrics,
            grounded_docs=(),
            grounded_docs_by_source={},
            to_state=lambda: {
                "evidence_units": [],
                "evidence_unit_results": [],
                "evidence_unit_metrics": metrics,
            },
        )

    return retrieve


def test_summary_reserves_capacity_for_twenty_five_required_sections(monkeypatch):
    docs = [_doc("a.pdf", "intro", "文档内容。")]
    captured = {}
    monkeypatch.setattr(
        summary, "retrieve_verified_evidence_units", _capture_plan_budget(captured)
    )
    monkeypatch.setattr(summary, "kb_read_lease", lambda _kb: nullcontext())
    monkeypatch.setattr(
        summary.RetrieverFactory, "get_engine", lambda _kb: _Engine(docs)
    )
    plans = [
        {
            "section_id": f"section-{index}",
            "title": f"章节 {index}",
            "instruction": f"概括章节 {index}",
        }
        for index in range(25)
    ]

    summary.section_evidence_node(
        {
            "query": "总结 a.pdf",
            "doc_id": "kb",
            "summary_source": "a.pdf",
            "summary_docs": docs,
            "summary_section_plans": plans,
        },
        _config(),
    )

    assert captured["unit_count"] == 25
    assert captured["budget"].max_total_docs == 25
    assert captured["budget"].max_docs_per_unit == 8


def test_compare_reserves_capacity_for_six_source_default_matrix(monkeypatch):
    sources = [f"doc-{index}.pdf" for index in range(6)]
    docs = [_doc(source, f"chunk-{index}", "文档内容。") for index, source in enumerate(sources)]
    captured = {}
    monkeypatch.setattr(
        compare, "retrieve_verified_evidence_units", _capture_plan_budget(captured)
    )
    monkeypatch.setattr(compare, "kb_read_lease", lambda _kb: nullcontext())
    monkeypatch.setattr(
        compare.RetrieverFactory, "get_engine", lambda _kb: _Engine(docs)
    )

    compare.cell_evidence_node(
        {
            "query": "对比六篇文档",
            "doc_id": "kb",
            "compare_sources": sources,
            "compare_docs_by_source": {
                source: [doc] for source, doc in zip(sources, docs)
            },
            "compare_dimensions": compare_profile.default_compare_dimensions(),
        },
        _config(),
    )

    assert captured["unit_count"] == 36
    assert captured["budget"].max_total_docs == 36
    assert captured["budget"].max_docs_per_unit == 4


def test_summary_retrieves_per_section_and_generator_uses_frozen_closures(
    monkeypatch,
):
    docs = [
        _doc("a.pdf", "method", "文档的方法是分层检索。"),
        _doc("a.pdf", "limits", "文档限制是必须离线运行。"),
    ]
    monkeypatch.setattr(
        summary.RetrieverFactory, "get_engine", lambda _kb: _Engine(docs)
    )
    monkeypatch.setattr(BGEReranker, "default_device", lambda: "cpu")
    monkeypatch.setattr(
        summary_generator.Generator,
        "_get_client_for_node",
        lambda *args, **kwargs: _LLM(),
    )
    state = {
        "query": "总结 a.pdf",
        "doc_id": "kb",
        "summary_source": "a.pdf",
        "summary_docs": docs,
        "summary_section_plans": [
            {"section_id": "method", "title": "方法", "instruction": "概括方法"},
            {"section_id": "limits", "title": "限制", "instruction": "概括限制"},
        ],
    }

    retrieval = summary.section_evidence_node(state, _config())
    generated = summary.section_summary_node({**state, **retrieval})

    assert len(retrieval["evidence_units"]) == 2
    assert [row["status"] for row in retrieval["evidence_unit_results"]] == [
        "supported",
        "supported",
    ]
    assert retrieval["evidence_unit_metrics"]["coverage_rate"] == 1.0
    assert retrieval["evidence_ledger"]
    assert all(
        result["content"].endswith("。") and "[E" in result["content"]
        for result in generated["summary_section_results"]
    )


def test_summary_reverifies_recovery_before_emitting_no_evidence(monkeypatch):
    docs = [_doc("a.pdf", "background", "这里只包含无关背景。")]
    monkeypatch.setattr(
        summary.RetrieverFactory, "get_engine", lambda _kb: _Engine(docs)
    )
    monkeypatch.setattr(BGEReranker, "default_device", lambda: "cpu")
    monkeypatch.setattr(
        summary_generator.Generator,
        "_get_client_for_node",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("no-evidence unit must not call the generator")
        ),
    )
    state = {
        "query": "总结 a.pdf 的限制",
        "doc_id": "kb",
        "summary_source": "a.pdf",
        "summary_docs": docs,
        "summary_section_plans": [
            {"section_id": "limits", "title": "限制", "instruction": "概括限制"}
        ],
    }

    retrieval = summary.section_evidence_node(
        state, _config(_reject_all_units)
    )
    generated = summary.section_summary_node({**state, **retrieval})
    final = summary.global_summary_node({**state, **retrieval, **generated})

    assert retrieval["evidence_unit_metrics"]["targeted_retry_count"] == 1
    assert retrieval["evidence_unit_metrics"]["verification_rounds"] == 2
    assert retrieval["evidence_unit_results"][0]["status"] == "no_evidence"
    assert retrieval["evidence_unit_results"][0]["gate_action"] == "terminal"
    assert retrieval["summary_docs"] == []
    assert generated["summary_section_results"][0]["content"] == "文档中未明确说明。"
    assert "# a.pdf 结构化摘要" in final["answer"]
    assert "文档中未明确说明。" in final["answer"]


def test_compare_retrieves_each_cell_inside_its_source_and_preserves_matrix(
    monkeypatch,
):
    docs = [
        _doc("a.pdf", "a-method", "A 使用分层检索。"),
        _doc("b.pdf", "b-method", "B 使用关键词检索。"),
    ]
    engine = _Engine(docs)
    monkeypatch.setattr(compare.RetrieverFactory, "get_engine", lambda _kb: engine)
    monkeypatch.setattr(BGEReranker, "default_device", lambda: "cpu")
    monkeypatch.setattr(
        compare_profile.Generator,
        "_get_client_for_node",
        lambda *args, **kwargs: _LLM(),
    )
    state = {
        "query": "对比 a.pdf 和 b.pdf",
        "doc_id": "kb",
        "compare_sources": ["a.pdf", "b.pdf"],
        "compare_docs_by_source": {
            "a.pdf": [docs[0]],
            "b.pdf": [docs[1]],
        },
        "compare_dimensions": [
            {"dimension_id": "method", "title": "方法", "instruction": "概括方法"}
        ],
    }

    retrieval = compare.cell_evidence_node(state, _config())
    generated = compare.document_profile_node({**state, **retrieval})

    assert [
        row["selected_docs"][0]["meta"]["source"]
        for row in retrieval["evidence_unit_results"]
    ] == ["a.pdf", "b.pdf"]
    profiles = generated["document_profiles"]
    assert [profile["source"] for profile in profiles] == ["a.pdf", "b.pdf"]
    assert [profile["cells"][0]["evidence"][0]["source"] for profile in profiles] == [
        "a.pdf",
        "b.pdf",
    ]
    assert [
        profile["cells"][0]["evidence"][0]["evidence_id"] for profile in profiles
    ] == ["E001", "E002"]
