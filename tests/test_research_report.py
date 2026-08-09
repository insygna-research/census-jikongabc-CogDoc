import json
from types import SimpleNamespace

from cogdoc.service.research_report import (
    ResolvedResearchEvidence,
    ResolvedResearchSection,
    build_research_report,
    resolve_research_evidence,
)
from cogdoc.service.retriever_factory import RetrieverFactory
from cogdoc.tools.citation_ledger import assign_evidence_ids
from cogdoc.tools.public_citation_ledger import validate_public_citation_ledger
from cogdoc.tools.reranker import BGEReranker


def _job():
    return {
        "job_id": "rj_test",
        "kb_id": "kb",
        "title": "赛事研究",
        "objective": "比较赛事报名条件",
        "sections": [
            {
                "section_id": "s1",
                "title": "报名门槛",
                "research_question": "报名门槛是什么？",
            },
            {
                "section_id": "s2",
                "title": "时间成本",
                "research_question": "时间成本是什么？",
            },
        ],
    }


def _resolved():
    docs, ledger = assign_evidence_ids(
        [
            {
                "text": "参赛者必须在五月一日前提交材料。",
                "meta": {
                    "chunk_id": "rules-1",
                    "source": "rules.pdf",
                    "page": 3,
                    "page_start": 3,
                    "page_end": 3,
                },
                "retrieval": {},
            }
        ]
    )
    return ResolvedResearchEvidence(
        sections=(
            ResolvedResearchSection(
                section_id="s1",
                verification_status="supported",
                docs=tuple(docs),
                evidence=(
                    {
                        "chunk_id": "rules-1",
                        "source": "rules.pdf",
                        "page": 3,
                        "text_preview": "参赛者必须提交材料。",
                    },
                ),
                reason_code="supported",
            ),
            ResolvedResearchSection(
                section_id="s2",
                verification_status="no_evidence",
                reason_code="no_direct_support",
            ),
        ),
        evidence_ledger=tuple(ledger),
        metrics={"supported_count": 1, "no_evidence_count": 1},
    )


def test_research_report_only_generates_supported_sections_and_renders_citations():
    calls = []

    def writer(objective, title, question, context):
        calls.append((objective, title, question, context))
        return "规程明确规定了材料提交截止时间。"

    result = build_research_report(
        _job(),
        evidence_resolver=lambda _job: _resolved(),
        section_writer=writer,
    )

    assert len(calls) == 1
    assert "<Document" in calls[0][3]
    assert result.status == "ready_with_gaps"
    assert "规程明确规定了材料提交截止时间[rules.pdf:P3]。" in result.markdown
    assert "本章节未找到足以支持结论的证据。" in result.markdown
    assert "[E001]" not in result.markdown
    assert len(result.citation_ledger) == 1
    assert result.sections[0]["status"] == "generated"
    assert result.sections[1]["status"] == "no_evidence"


def test_research_report_fails_closed_when_section_generation_errors():
    def failing_writer(*_args):
        raise TimeoutError("model unavailable")

    result = build_research_report(
        _job(),
        evidence_resolver=lambda _job: _resolved(),
        section_writer=failing_writer,
    )

    assert result.status == "ready_with_gaps"
    assert result.citation_ledger == ()
    assert result.sections[0]["status"] == "generation_error"
    assert result.sections[0]["error"] == "TimeoutError"
    assert "未经校验的内容" in result.markdown


def test_research_report_rejects_model_supplied_public_citations():
    result = build_research_report(
        _job(),
        evidence_resolver=lambda _job: _resolved(),
        section_writer=lambda *_args: "伪造定位 [other.pdf:P99]。",
    )

    assert result.sections[0]["status"] == "generation_error"
    assert "other.pdf:P99" not in result.markdown
    assert result.citation_ledger == ()


def test_research_report_selectively_regenerates_rejected_section():
    first = build_research_report(
        _job(),
        evidence_resolver=lambda _job: _resolved(),
        section_writer=lambda *_args: "第一章原始正文。",
    )
    second_docs, second_ledger = assign_evidence_ids(
        [
            {
                "text": "赛事需要连续投入三天。",
                "meta": {
                    "chunk_id": "time-1",
                    "source": "schedule.pdf",
                    "page": 5,
                },
                "retrieval": {},
            }
        ]
    )
    regeneration_job = _job()
    regeneration_job["regeneration_section_ids"] = ["s2"]
    for raw_section, generated in zip(
        regeneration_job["sections"], first.sections, strict=True
    ):
        raw_section.update(generated)
    # Simulate a report created before section-local ledgers were persisted;
    # selective regeneration must recover the preserved ledger from the global one.
    regeneration_job["report"] = {
        "content": first.markdown,
        "citation_ledger": list(first.citation_ledger),
    }
    regeneration_job["sections"][0]["citation_ledger"] = []
    regeneration_job["sections"][0].update(
        {"review_status": "approved", "review_note": "已核对"}
    )
    regeneration_job["sections"][1].update(
        {
            "review_status": "changes_requested",
            "review_note": "补充时间证据",
            "revision_instruction": "补充时间证据",
        }
    )
    writer_calls = []

    result = build_research_report(
        regeneration_job,
        evidence_resolver=lambda _job: ResolvedResearchEvidence(
            sections=(
                ResolvedResearchSection(
                    section_id="s2",
                    verification_status="supported",
                    docs=tuple(second_docs),
                    evidence=({"chunk_id": "time-1", "source": "schedule.pdf"},),
                    reason_code="supported",
                ),
            ),
            evidence_ledger=tuple(second_ledger),
            metrics={"supported_count": 1},
        ),
        section_writer=lambda *_args: writer_calls.append(_args)
        or "第二章修订正文。",
    )

    assert len(writer_calls) == 1
    assert result.sections[0]["content"] == first.sections[0]["content"]
    assert result.sections[0]["review_status"] == "approved"
    assert result.sections[0]["preserved"] is True
    assert result.sections[1]["preserved"] is False
    assert "第一章原始正文[rules.pdf:P3]。" in result.markdown
    assert "第二章修订正文[schedule.pdf:P5]。" in result.markdown
    assert [entry["evidence_id"] for entry in result.citation_ledger] == [
        "E001",
        "E002",
    ]
    assert validate_public_citation_ledger(
        result.markdown, list(result.citation_ledger)
    ).is_valid
    assert result.verification_metrics["selective_regeneration"] is True
    assert result.verification_metrics["regenerated_section_count"] == 1
    assert result.verification_metrics["preserved_section_count"] == 1


def test_research_evidence_resolver_runs_closed_set_verifier(monkeypatch):
    raw_doc = {
        "text": "参赛者必须在五月一日前提交材料。",
        "meta": {
            "chunk_id": "rules-1",
            "source_sha256": "sha:rules.pdf",
            "local_chunk_index": 0,
            "chunk_index": 0,
            "source": "rules.pdf",
            "page": 3,
            "page_start": 3,
            "page_end": 3,
            "origin": "file",
        },
    }

    class Engine:
        def search(self, query, top_k=3, *, scope=None):
            return [raw_doc]

        def load_source_chunks(self, source):
            return [raw_doc] if source == "rules.pdf" else []

    class NoDerived:
        def search(self, kb_id, query, top_k=3, *, scope=None):
            return []

    class NoFeedback:
        def boosts_for_query(self, kb_id, query):
            return {}

    verifier_payloads = []

    def verifier_client(schema, messages):
        payload = json.loads(
            messages[1]["content"].split("\n", 1)[1].rsplit("\n\n", 1)[0]
        )
        verifier_payloads.extend(payload)
        return schema(
            assessments=[
                {
                    "unit_id": row["unit_id"],
                    "status": "supported",
                    "evidence_ids": [row["candidate_evidence_ids"][0]],
                    "reason": "证据直接支持",
                }
                for row in payload
            ]
        )

    monkeypatch.setattr(
        RetrieverFactory,
        "get_engine",
        classmethod(lambda _cls, _kb_id: Engine()),
    )
    monkeypatch.setattr(BGEReranker, "default_device", lambda: "cpu")
    resolved = resolve_research_evidence(
        {
            "kb_id": "kb",
            "objective": "查明报名门槛",
            "regeneration_section_ids": ["s1"],
            "sections": [
                {
                    "section_id": "s1",
                    "title": "报名门槛",
                    "research_question": "报名门槛是什么？",
                    "revision_instruction": "补充核对材料提交日期",
                },
                {
                    "section_id": "s2",
                    "title": "不应重跑",
                    "research_question": "这个章节是否会被检索？",
                },
            ],
        },
        state_runtime=SimpleNamespace(
            derived_knowledge_retriever=NoDerived(),
            retrieval_feedback_store=NoFeedback(),
        ),
        structured_client=verifier_client,
    )

    assert resolved.sections[0].verification_status == "supported"
    assert resolved.sections[0].docs[0]["retrieval"]["evidence_id"] == "E001"
    assert resolved.sections[0].evidence[0]["source"] == "rules.pdf"
    assert resolved.metrics["supported_count"] == 1
    assert "补充核对材料提交日期" in verifier_payloads[0]["instruction"]
    assert [section.section_id for section in resolved.sections] == ["s1"]
