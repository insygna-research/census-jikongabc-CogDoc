import json
import pytest
from httpx import ASGITransport, AsyncClient
from cogdoc.api.app import create_app
from cogdoc.api.derived_knowledge_store import DerivedKnowledgeStore
from cogdoc.api.feedback_analysis_store import FeedbackAnalysisStore
from cogdoc.api.feedback_store import FeedbackStore
from cogdoc.api.retrieval_feedback_store import RetrievalFeedbackStore


# 声明异步测试使用的后端。
@pytest.fixture
def anyio_backend():
    return "asyncio"


# 构造应用。
def _make_app(tmp_path, monkeypatch, webhook_dispatcher=None):
    import cogdoc.api.app as app_module

    monkeypatch.setattr(app_module, "configure_logging", lambda: None)
    store = FeedbackStore(
        feedback_path=str(tmp_path / "feedback.jsonl"),
        bad_cases_path=str(tmp_path / "bad_cases.jsonl"),
    )
    knowledge_store = DerivedKnowledgeStore(path=str(tmp_path / "knowledge.jsonl"))
    feedback_analysis_store = FeedbackAnalysisStore(
        path=str(tmp_path / "feedback_analysis.jsonl")
    )
    retrieval_feedback_store = RetrievalFeedbackStore(
        path=str(tmp_path / "retrieval_feedback.jsonl")
    )
    return (
        create_app(
            feedback_store=store,
            feedback_analysis_store=feedback_analysis_store,
            knowledge_store=knowledge_store,
            retrieval_feedback_store=retrieval_feedback_store,
            webhook_dispatcher=webhook_dispatcher,
        ),
        tmp_path,
    )


# 发送结果。
async def _post(app, payload):
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.post("/v1/feedback", json=payload)


# 读取逐行对象文件。
def _read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# 验证点赞只记录反馈且不进入坏样本场景。
@pytest.mark.anyio
async def test_thumbs_up_recorded_not_bad_case(tmp_path, monkeypatch):
    app, root = _make_app(tmp_path, monkeypatch)

    resp = await _post(
        app, {"trace_id": "t1", "feedback": "thumbs_up", "kb_id": "kb", "query": "问题"}
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["is_bad_case"] is False and body["feedback_id"]
    feedback = _read_jsonl(root / "feedback.jsonl")
    assert len(feedback) == 1 and feedback[0]["trace_id"] == "t1"
    # 正反馈不进坏样本集。
    assert _read_jsonl(root / "bad_cases.jsonl") == []


# 验证点踩进入坏样本场景。
@pytest.mark.anyio
async def test_thumbs_down_lands_in_bad_cases(tmp_path, monkeypatch):
    app, root = _make_app(tmp_path, monkeypatch)

    resp = await _post(
        app,
        {
            "trace_id": "t2",
            "feedback": "thumbs_down",
            "kb_id": "kb",
            "query": "问题",
            "answer": "错答案",
            "citations": [{"chunk_id": "c1", "source": "a.pdf", "page": 1}],
            "evidence": [
                {
                    "chunk_id": "c1",
                    "source": "a.pdf",
                    "page": 1,
                    "text_preview": "证据",
                }
            ],
        },
    )

    assert resp.status_code == 201 and resp.json()["is_bad_case"] is True
    bad = _read_jsonl(root / "bad_cases.jsonl")
    assert len(bad) == 1 and bad[0]["feedback"] == "thumbs_down"
    assert bad[0]["eval_draft"] == {
        "case_type": "faithfulness",
        "layer": "feedback",
        "query": "问题",
        "answer": "错答案",
        "is_faithful": False,
        "reviewer": "user_feedback",
        "trace_id": "t2",
        "kb_id": "kb",
        "feedback": "thumbs_down",
        "citations": [
            {
                "chunk_id": "c1",
                "source_type": "document",
                "knowledge_id": "",
                "source": "a.pdf",
                "page": 1,
            }
        ],
        "evidence": [
            {
                "chunk_id": "c1",
                "source_type": "document",
                "knowledge_id": "",
                "source": "a.pdf",
                "page": 1,
                "text_preview": "证据",
            }
        ],
    }
    # 同时也进总反馈日志。
    assert len(_read_jsonl(root / "feedback.jsonl")) == 1
    # 未归因的点踩可以是回答问题，不应自动惩罚引用分块。
    assert _read_jsonl(root / "retrieval_feedback.jsonl") == []
    feedback_analysis = _read_jsonl(root / "feedback_analysis.jsonl")
    assert feedback_analysis[0]["recommended_action"] == "adjust_retrieval"
    assert feedback_analysis[0]["target"]["chunk_ids"] == ["c1"]


# 验证一次反馈命中多个分块时调权仍按一次反馈统计。
@pytest.mark.anyio
async def test_retrieval_feedback_counts_one_feedback_with_multiple_chunks(
    tmp_path, monkeypatch
):
    app, root = _make_app(tmp_path, monkeypatch)

    resp = await _post(
        app,
        {
            "trace_id": "t-multi-chunk",
            "feedback": "thumbs_down",
            "feedback_type": "bad_retrieval",
            "kb_id": "kb",
            "query": "问题",
            "citations": [{"chunk_id": "c1", "source": "a.pdf"}],
            "evidence": [{"chunk_id": "c2", "source": "a.pdf"}],
        },
    )

    assert resp.status_code == 201
    raw_rows = _read_jsonl(root / "retrieval_feedback.jsonl")
    assert len(raw_rows) == 1
    assert raw_rows[0]["chunk_count"] == 2
    assert [row["chunk_id"] for row in raw_rows[0]["target_chunks"]] == ["c1", "c2"]

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            listed = await client.get("/v1/retrieval-feedback", params={"kb_id": "kb"})

    assert listed.status_code == 200
    listed_rows = listed.json()["retrieval_feedback"]
    assert len(listed_rows) == 1
    assert listed_rows[0]["chunk_count"] == 2


# 验证同一回答只接受第一条反馈。
@pytest.mark.anyio
async def test_quick_feedback_duplicate_trace_ignored(tmp_path, monkeypatch):
    app, root = _make_app(tmp_path, monkeypatch)
    first = await _post(
        app,
        {
            "trace_id": "t-dup",
            "feedback": "thumbs_down",
            "feedback_type": "bad_retrieval",
            "kb_id": "kb",
            "query": "问题",
            "citations": [{"chunk_id": "c1", "source": "a.pdf"}],
        },
    )
    second = await _post(
        app,
        {
            "trace_id": "t-dup",
            "feedback": "thumbs_up",
            "kb_id": "kb",
            "query": "问题",
            "citations": [{"chunk_id": "c1", "source": "a.pdf"}],
        },
    )

    assert first.status_code == 201
    assert second.status_code == 201
    assert second.json()["status"] == "duplicate_ignored"
    assert second.json()["feedback_id"] == first.json()["feedback_id"]
    assert len(_read_jsonl(root / "feedback.jsonl")) == 1
    assert len(_read_jsonl(root / "retrieval_feedback.jsonl")) == 1
    assert len(_read_jsonl(root / "feedback_analysis.jsonl")) == 1


# 验证点踩后提交纠错仍会进入待审核知识。
@pytest.mark.anyio
async def test_correction_after_thumbs_down_still_creates_pending_knowledge(
    tmp_path, monkeypatch
):
    app, root = _make_app(tmp_path, monkeypatch)
    down = await _post(
        app,
        {
            "trace_id": "t-correct-after-down",
            "feedback": "thumbs_down",
            "kb_id": "kb",
            "query": "问题",
            "citations": [{"chunk_id": "c1", "source": "a.pdf"}],
        },
    )
    correction = await _post(
        app,
        {
            "trace_id": "t-correct-after-down",
            "feedback": "correction",
            "kb_id": "kb",
            "query": "问题",
            "correction_text": "正确说法",
            "citations": [{"chunk_id": "c1", "source": "a.pdf"}],
        },
    )

    assert down.status_code == 201
    assert correction.status_code == 201
    assert correction.json()["status"] == "recorded"
    assert correction.json()["knowledge_status"] == "pending"
    assert len(_read_jsonl(root / "feedback.jsonl")) == 2
    knowledge = _read_jsonl(root / "knowledge.jsonl")
    assert knowledge[0]["text"] == "正确说法"
    assert knowledge[0]["origin"] == "correction"


# 验证纠错样本草稿优先使用纠正答案场景。
@pytest.mark.anyio
async def test_correction_uses_correction_text_in_eval_draft(tmp_path, monkeypatch):
    app, root = _make_app(tmp_path, monkeypatch)

    resp = await _post(
        app,
        {
            "trace_id": "t4",
            "feedback": "correction",
            "kb_id": "kb",
            "query": "问题",
            "answer": "原答案",
            "correction": "纠正后的答案",
            "comment": "引用不支撑结论",
        },
    )

    assert resp.status_code == 201 and resp.json()["is_bad_case"] is True
    draft = _read_jsonl(root / "bad_cases.jsonl")[0]["eval_draft"]
    assert draft["answer"] == "纠正后的答案"
    assert draft["correction"] == "纠正后的答案"
    assert draft["comment"] == "引用不支撑结论"
    knowledge = _read_jsonl(root / "knowledge.jsonl")[0]
    assert knowledge["origin"] == "correction"
    assert knowledge["text"] == "纠正后的答案"


# 验证纠错可以创建待审核知识场景。
@pytest.mark.anyio
async def test_correction_can_create_pending_knowledge(
    tmp_path, monkeypatch, webhook_dispatcher
):
    app, root = _make_app(tmp_path, monkeypatch, webhook_dispatcher)

    resp = await _post(
        app,
        {
            "trace_id": "t5",
            "feedback": "correction",
            "kb_id": "kb",
            "query": "内部报销规则是什么",
            "answer": "旧规则",
            "correction_text": "差旅报销需要在 7 天内提交。",
            "feedback_text": "回答引用了旧规则",
            "save_as_knowledge": True,
            "citations": [{"chunk_id": "c1", "source": "policy.pdf", "page": 2}],
            "related_page_start": 2,
            "related_page_end": 2,
            "related_chunk_text_hash": "hash-c1",
            "related_anchor_text": "差旅报销",
            "created_by": "u1",
        },
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["knowledge_id"].startswith("K")
    assert body["knowledge_status"] == "pending"
    feedback = _read_jsonl(root / "feedback.jsonl")[0]
    assert feedback["comment"] == "回答引用了旧规则"
    assert feedback["correction"] == "差旅报销需要在 7 天内提交。"
    knowledge = _read_jsonl(root / "knowledge.jsonl")[0]
    assert knowledge["origin"] == "correction"
    assert knowledge["created_from_trace_id"] == "t5"
    assert knowledge["related_source"] == "policy.pdf"
    assert knowledge["related_chunk_ids"] == ["c1"]
    assert knowledge["related_page_start"] == 2
    assert knowledge["related_page_end"] == 2
    assert knowledge["related_chunk_text_hash"] == "hash-c1"
    assert knowledge["related_anchor_text"] == "差旅报销"
    # 普通纠错只供审核和派生知识，不代表检索证据有错。
    assert _read_jsonl(root / "retrieval_feedback.jsonl") == []
    assert [event for event, _ in webhook_dispatcher.events] == [
        "knowledge.pending_created"
    ]
    assert webhook_dispatcher.events[0][1]["source"] == "feedback"


# 验证无答案补知识不生成检索调权场景。
@pytest.mark.anyio
async def test_no_evidence_save_as_knowledge_skips_retrieval_feedback(
    tmp_path, monkeypatch, webhook_dispatcher
):
    app, root = _make_app(tmp_path, monkeypatch, webhook_dispatcher)

    resp = await _post(
        app,
        {
            "trace_id": "t6",
            "feedback": "correction",
            "kb_id": "kb",
            "query": "没有答案的问题",
            "answer": "文档中未明确说明。",
            "correction_text": "补充后的正确说法",
            "feedback_text": "人工补充无答案问题",
            "feedback_type": "no_evidence",
            "save_as_knowledge": True,
            "skip_retrieval_feedback": True,
            "citations": [{"chunk_id": "c1", "source": "policy.pdf", "page": 2}],
            "related_page_start": 2,
            "related_page_end": 2,
        },
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["knowledge_id"].startswith("K")
    assert body["knowledge_status"] == "pending"
    knowledge = _read_jsonl(root / "knowledge.jsonl")[0]
    assert knowledge["origin"] == "no_evidence"
    assert knowledge["text"] == "补充后的正确说法"
    assert _read_jsonl(root / "retrieval_feedback.jsonl") == []


# 验证反馈理解结果可以查询场景。
@pytest.mark.anyio
async def test_feedback_analysis_can_be_listed(tmp_path, monkeypatch):
    app, _ = _make_app(tmp_path, monkeypatch)

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            created = await client.post(
                "/v1/feedback",
                json={
                    "trace_id": "t7",
                    "feedback": "correction",
                    "kb_id": "kb",
                    "query": "问题",
                    "correction_text": "正确答案",
                    "feedback_text": "回答错误",
                },
            )
            listed = await client.get(
                "/v1/feedback-analysis",
                params={
                    "kb_id": "kb",
                    "recommended_action": "create_pending_knowledge",
                },
            )

    assert created.status_code == 201
    assert created.json()["feedback_analysis_action"] == "create_pending_knowledge"
    assert created.json()["feedback_analysis_confidence"] >= 0.8
    assert listed.status_code == 200
    rows = listed.json()["feedback_analysis"]
    assert rows[0]["feedback_analysis_id"] == created.json()["feedback_analysis_id"]
    assert rows[0]["extracted_claim"] == "正确答案"


# 验证反馈记录可以按条件查询场景。
@pytest.mark.anyio
async def test_feedback_records_can_be_listed(tmp_path, monkeypatch):
    app, _ = _make_app(tmp_path, monkeypatch)

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            first = await client.post(
                "/v1/feedback",
                json={
                    "trace_id": "t10",
                    "session_id": "s1",
                    "feedback": "thumbs_down",
                    "feedback_type": "bad_retrieval",
                    "kb_id": "kb",
                    "query": "问题",
                },
            )
            await client.post(
                "/v1/feedback",
                json={
                    "trace_id": "t11",
                    "feedback": "thumbs_up",
                    "kb_id": "kb",
                    "query": "问题",
                },
            )
            listed = await client.get(
                "/v1/feedback",
                params={
                    "kb_id": "kb",
                    "feedback": "thumbs_down",
                    "feedback_type": "bad_retrieval",
                    "is_bad_case": True,
                },
            )
            traced = await client.get(
                "/v1/feedback", params={"kb_id": "kb", "trace_id": "t10"}
            )

    assert first.status_code == 201
    assert listed.status_code == 200
    rows = listed.json()["feedback"]
    assert len(rows) == 1
    assert rows[0]["feedback_id"] == first.json()["feedback_id"]
    assert rows[0]["session_id"] == "s1"
    assert traced.json()["feedback"][0]["trace_id"] == "t10"


# 验证反馈理解失败不阻断反馈提交场景。
@pytest.mark.anyio
async def test_feedback_analysis_failure_does_not_block_feedback(tmp_path, monkeypatch):
    import cogdoc.api.routes.feedback as feedback_route

    app, root = _make_app(tmp_path, monkeypatch)

    def broken_analysis(payload):
        raise RuntimeError("broken")

    monkeypatch.setattr(feedback_route, "analyze_feedback", broken_analysis)

    resp = await _post(
        app,
        {
            "trace_id": "t8",
            "feedback": "thumbs_down",
            "kb_id": "kb",
            "query": "问题",
        },
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["feedback_id"]
    assert body["feedback_analysis_id"] is None
    assert _read_jsonl(root / "feedback.jsonl")[0]["trace_id"] == "t8"
    assert _read_jsonl(root / "feedback_analysis.jsonl") == []


# 验证知识草稿创建失败不阻断反馈提交场景。
@pytest.mark.anyio
async def test_knowledge_create_failure_does_not_block_feedback(tmp_path, monkeypatch):
    app, root = _make_app(tmp_path, monkeypatch)

    class BrokenKnowledgeStore:
        def create(self, payload):
            raise RuntimeError("broken")

    app.state.knowledge_store = BrokenKnowledgeStore()

    resp = await _post(
        app,
        {
            "trace_id": "t9",
            "feedback": "correction",
            "kb_id": "kb",
            "query": "问题",
            "correction_text": "正确答案",
            "save_as_knowledge": True,
        },
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["feedback_id"]
    assert body["knowledge_id"] is None
    assert body["knowledge_status"] is None
    assert _read_jsonl(root / "feedback.jsonl")[0]["trace_id"] == "t9"


# 验证检索反馈可以禁用和启用场景。
@pytest.mark.anyio
async def test_retrieval_feedback_can_disable_and_enable(tmp_path, monkeypatch):
    app, root = _make_app(tmp_path, monkeypatch)

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            await client.post(
                "/v1/feedback",
                json={
                    "trace_id": "t6",
                    "feedback": "thumbs_down",
                    "feedback_type": "bad_retrieval",
                    "kb_id": "kb",
                    "query": "问题",
                    "citations": [{"chunk_id": "c1", "source": "a.pdf"}],
                },
            )
            listed = await client.get("/v1/retrieval-feedback", params={"kb_id": "kb"})
            feedback_id = _read_jsonl(root / "retrieval_feedback.jsonl")[0][
                "retrieval_feedback_id"
            ]
            disabled = await client.post(
                f"/v1/retrieval-feedback/{feedback_id}/disable",
                json={"actor": "admin", "reason": "误点"},
            )
            disabled_listed = await client.get(
                "/v1/retrieval-feedback",
                params={"kb_id": "kb", "enabled": False},
            )
            enabled = await client.post(f"/v1/retrieval-feedback/{feedback_id}/enable")

    assert listed.status_code == 200
    assert listed.json()["retrieval_feedback"][0]["chunk_id"] == "c1"
    assert disabled.status_code == 200
    assert disabled.json()["status"] == "disabled"
    assert disabled_listed.status_code == 200
    assert disabled_listed.json()["retrieval_feedback"][0]["enabled"] is False
    assert enabled.status_code == 200
    assert enabled.json()["status"] == "enabled"
    rows = _read_jsonl(root / "retrieval_feedback.jsonl")
    assert rows[-2]["enabled"] is False and rows[-2]["disable_reason"] == "误点"
    assert rows[-1]["enabled"] is True and rows[-1]["disabled_at"] is None


# 验证反馈拒绝非法请求场景。
@pytest.mark.anyio
async def test_feedback_rejects_invalid_payload(tmp_path, monkeypatch):
    app, _ = _make_app(tmp_path, monkeypatch)

    missing_trace = await _post(app, {"feedback": "thumbs_up"})
    bad_type = await _post(app, {"trace_id": "t3", "feedback": "love_it"})

    assert missing_trace.status_code == 422
    assert bad_type.status_code == 422
