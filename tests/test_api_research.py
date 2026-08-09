import pytest
import asyncio
from httpx import ASGITransport, AsyncClient

from cogdoc.api.app import create_app
from cogdoc.api.ingest import KnowledgeBaseRegistry
from cogdoc.config.settings import Settings
from cogdoc.state_runtime import StateRuntime
from cogdoc.service.research_execution import ResearchExecutionManager


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _make_app(tmp_path, monkeypatch, retrieve=None, report_builder=None):
    import cogdoc.api.app as app_module

    monkeypatch.setattr(app_module, "configure_logging", lambda: None)
    settings = Settings(
        _env_file=None,
        cogdoc_data_dir=str(tmp_path / "data"),
        cogdoc_state_backend="jsonl",
        cogdoc_feedback_store="jsonl",
    )
    runtime = StateRuntime.from_settings(settings)
    registry = KnowledgeBaseRegistry(
        registry_path=str(tmp_path / "registry.json"),
        source_dir_for=lambda kb_id: str(tmp_path / "kb" / kb_id / "sources"),
    )
    registry.create("kb")
    research_manager = ResearchExecutionManager(
        runtime.research_job_store,
        retrieve=retrieve or (lambda _kb_id, _query: []),
        kb_exists=registry.exists,
        report_builder=report_builder,
    )
    app = create_app(
        state_runtime=runtime,
        close_state_runtime_on_shutdown=True,
        kb_registry=registry,
        research_execution_manager=research_manager,
    )
    return app


@pytest.mark.anyio
async def test_research_api_create_list_get_and_update_plan(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            created_response = await client.post(
                "/v1/research-jobs",
                json={
                    "kb_id": "kb",
                    "objective": "比较赛事并形成有证据的选择建议",
                    "section_titles": ["参赛门槛", "时间成本"],
                },
            )
            assert created_response.status_code == 201
            created = created_response.json()["job"]
            assert created["revision"] == 1
            assert [section["title"] for section in created["sections"]] == [
                "参赛门槛",
                "时间成本",
            ]

            listed = await client.get("/v1/research-jobs?kb_id=kb")
            assert listed.status_code == 200
            assert [job["job_id"] for job in listed.json()["jobs"]] == [
                created["job_id"]
            ]

            fetched = await client.get(f"/v1/research-jobs/{created['job_id']}")
            assert fetched.status_code == 200
            assert fetched.json()["job"] == created

            updated_response = await client.put(
                f"/v1/research-jobs/{created['job_id']}/plan",
                json={
                    "expected_revision": 1,
                    "sections": [
                        {
                            "title": "选择建议",
                            "research_question": "各项证据共同支持哪种选择？",
                        }
                    ],
                },
            )
            assert updated_response.status_code == 200
            updated = updated_response.json()["job"]
            assert updated["revision"] == 2
            assert updated["sections"][0]["section_id"] == "s1"

            conflict = await client.put(
                f"/v1/research-jobs/{created['job_id']}/plan",
                json={
                    "expected_revision": 1,
                    "sections": [
                        {"title": "旧计划", "research_question": "会覆盖吗？"}
                    ],
                },
            )
            assert conflict.status_code == 409
            assert conflict.json()["error_code"] == "RESEARCH_JOB_REVISION_CONFLICT"


@pytest.mark.anyio
async def test_research_api_rejects_unknown_kb_and_returns_stable_not_found(
    tmp_path, monkeypatch
):
    app = _make_app(tmp_path, monkeypatch)
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            missing_kb = await client.post(
                "/v1/research-jobs",
                json={"kb_id": "missing", "objective": "研究目标"},
            )
            missing_job = await client.get("/v1/research-jobs/rj_missing")

    assert missing_kb.status_code == 404
    assert missing_kb.json()["error_code"] == "KB_NOT_FOUND"
    assert missing_job.status_code == 404
    assert missing_job.json()["error_code"] == "RESEARCH_JOB_NOT_FOUND"


@pytest.mark.anyio
async def test_research_api_executes_evidence_and_exposes_progress(
    tmp_path, monkeypatch
):
    def retrieve(_kb_id, query):
        return [
            {
                "text": f"{query} 的直接证据",
                "meta": {"chunk_id": query, "source": "rules.pdf", "page": 1},
                "retrieval": {"rerank_score": 0.8},
            }
        ]

    app = _make_app(tmp_path, monkeypatch, retrieve=retrieve)
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            created = await client.post(
                "/v1/research-jobs",
                json={
                    "kb_id": "kb",
                    "objective": "形成证据矩阵",
                    "section_titles": ["门槛", "时间"],
                },
            )
            job_id = created.json()["job"]["job_id"]
            started = await client.post(f"/v1/research-jobs/{job_id}/start")
            assert started.status_code == 202
            assert started.json()["job"]["status"] == "running"

            body = None
            for _ in range(200):
                response = await client.get(f"/v1/research-jobs/{job_id}")
                body = response.json()["job"]
                if body["status"] == "evidence_ready":
                    break
                await asyncio.sleep(0.01)

            assert body["status"] == "evidence_ready"
            assert all(
                section["evidence_status"] == "partial"
                for section in body["sections"]
            )
            assert body["sections"][0]["evidence"][0]["source"] == "rules.pdf"
            conflict = await client.post(f"/v1/research-jobs/{job_id}/start")
            assert conflict.status_code == 409
            assert conflict.json()["error_code"] == "RESEARCH_JOB_STATE_CONFLICT"


@pytest.mark.anyio
async def test_research_api_cancel_and_unknown_action_are_stable(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            created = await client.post(
                "/v1/research-jobs",
                json={"kb_id": "kb", "objective": "可取消任务"},
            )
            job_id = created.json()["job"]["job_id"]
            invalid_resume = await client.post(
                f"/v1/research-jobs/{job_id}/resume"
            )
            cancelled = await client.post(f"/v1/research-jobs/{job_id}/cancel")
            missing = await client.post("/v1/research-jobs/rj_missing/pause")

    assert invalid_resume.status_code == 409
    assert invalid_resume.json()["error_code"] == "RESEARCH_JOB_STATE_CONFLICT"
    assert cancelled.status_code == 200
    assert cancelled.json()["job"]["status"] == "cancelled"
    assert missing.status_code == 404
    assert missing.json()["error_code"] == "RESEARCH_JOB_NOT_FOUND"


@pytest.mark.anyio
async def test_research_api_generates_and_downloads_markdown_report(
    tmp_path, monkeypatch
):
    def report_builder(_job):
        return {
            "status": "ready_with_gaps",
            "markdown": "# 研究报告\n\n## 证据\n\n证据不足。\n",
            "citation_ledger": [],
            "verification_metrics": {"no_evidence_count": 1},
            "sections": [
                {
                    "section_id": "s1",
                    "status": "no_evidence",
                    "verification_status": "no_evidence",
                    "verification_reason_code": "no_direct_support",
                    "content": "证据不足。",
                    "evidence": [],
                    "error": "",
                }
            ],
        }

    app = _make_app(
        tmp_path,
        monkeypatch,
        report_builder=report_builder,
    )
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            created = await client.post(
                "/v1/research-jobs",
                json={
                    "kb_id": "kb",
                    "objective": "形成报告",
                    "section_titles": ["证据"],
                },
            )
            job_id = created.json()["job"]["job_id"]
            before = await client.get(f"/v1/research-jobs/{job_id}/report")
            assert before.status_code == 409

            await client.post(f"/v1/research-jobs/{job_id}/start")
            for _ in range(200):
                current = await client.get(f"/v1/research-jobs/{job_id}")
                if current.json()["job"]["status"] == "evidence_ready":
                    break
                await asyncio.sleep(0.01)
            generated = await client.post(
                f"/v1/research-jobs/{job_id}/generate"
            )
            assert generated.status_code == 202

            body = None
            for _ in range(200):
                current = await client.get(f"/v1/research-jobs/{job_id}")
                body = current.json()["job"]
                if body["status"] == "completed":
                    break
                await asyncio.sleep(0.01)
            downloaded = await client.get(f"/v1/research-jobs/{job_id}/report")

            premature_publish = await client.post(
                f"/v1/research-jobs/{job_id}/publish",
                json={"expected_revision": body["revision"]},
            )
            assert premature_publish.status_code == 409
            reviewed = await client.put(
                f"/v1/research-jobs/{job_id}/review",
                json={
                    "expected_revision": body["revision"],
                    "decisions": [
                        {
                            "section_id": "s1",
                            "decision": "accepted_gap",
                            "note": "接受当前证据限制",
                        }
                    ],
                },
            )
            assert reviewed.status_code == 200
            reviewed_job = reviewed.json()["job"]
            assert reviewed_job["review_status"] == "approved"
            published = await client.post(
                f"/v1/research-jobs/{job_id}/publish",
                json={"expected_revision": reviewed_job["revision"]},
            )
            published_download = await client.get(
                f"/v1/research-jobs/{job_id}/published-report"
            )

    assert body["report_status"] == "ready_with_gaps"
    assert body["sections"][0]["verification_status"] == "no_evidence"
    assert downloaded.status_code == 200
    assert downloaded.text.startswith("# 研究报告")
    assert downloaded.headers["content-disposition"].endswith(f'{job_id}.md"')
    assert published.status_code == 200
    assert published.json()["job"]["review_status"] == "published"
    assert published_download.status_code == 200
    assert published_download.text == downloaded.text


@pytest.mark.anyio
async def test_research_api_regenerates_rejected_report_as_new_version(
    tmp_path, monkeypatch
):
    seen_revision_instructions = []
    seen_regeneration_scopes = []

    def report_builder(job):
        seen_revision_instructions.append(
            job["sections"][0].get("revision_instruction", "")
        )
        seen_regeneration_scopes.append(job.get("regeneration_section_ids", []))
        return {
            "status": "ready",
            "markdown": f"# v{len(seen_revision_instructions)}\n",
            "citation_ledger": [],
            "verification_metrics": {"supported_count": 1},
            "sections": [
                {
                    "section_id": "s1",
                    "status": "generated",
                    "verification_status": "supported",
                    "verification_reason_code": "supported",
                    "content": "正文",
                    "evidence": [],
                    "error": "",
                }
            ],
        }

    app = _make_app(
        tmp_path,
        monkeypatch,
        report_builder=report_builder,
    )
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            created = await client.post(
                "/v1/research-jobs",
                json={
                    "kb_id": "kb",
                    "objective": "形成可修订报告",
                    "section_titles": ["证据"],
                },
            )
            job_id = created.json()["job"]["job_id"]
            await client.post(f"/v1/research-jobs/{job_id}/start")
            first = None
            for _ in range(200):
                current = (await client.get(f"/v1/research-jobs/{job_id}")).json()[
                    "job"
                ]
                if current["status"] == "evidence_ready":
                    await client.post(f"/v1/research-jobs/{job_id}/generate")
                if current["status"] == "completed":
                    first = current
                    break
                await asyncio.sleep(0.01)
            assert first["report_version"] == 1
            rejected = await client.put(
                f"/v1/research-jobs/{job_id}/review",
                json={
                    "expected_revision": first["revision"],
                    "decisions": [
                        {
                            "section_id": "s1",
                            "decision": "changes_requested",
                            "note": "补充明确时间范围",
                        }
                    ],
                },
            )
            rejected_job = rejected.json()["job"]
            regenerated = await client.post(
                f"/v1/research-jobs/{job_id}/generate"
            )
            assert regenerated.status_code == 202
            second = None
            for _ in range(200):
                current = (await client.get(f"/v1/research-jobs/{job_id}")).json()[
                    "job"
                ]
                if current["status"] == "completed":
                    second = current
                    break
                await asyncio.sleep(0.01)

    assert rejected_job["review_status"] == "changes_requested"
    assert second["report_version"] == 2
    assert len(second["report_history"]) == 1
    assert seen_revision_instructions == ["", "补充明确时间范围"]
    assert seen_regeneration_scopes == [[], ["s1"]]
