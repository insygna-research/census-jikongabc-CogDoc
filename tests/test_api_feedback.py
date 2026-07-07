import json
import pytest
from httpx import ASGITransport, AsyncClient
from cogdoc.api.app import create_app
from cogdoc.api.derived_knowledge_store import DerivedKnowledgeStore
from cogdoc.api.feedback_store import FeedbackStore


# 声明异步测试使用的后端。
@pytest.fixture
def anyio_backend():
    return "asyncio"


# 构造应用。
def _make_app(tmp_path, monkeypatch):
    import cogdoc.api.app as app_module

    monkeypatch.setattr(app_module, "configure_logging", lambda: None)
    store = FeedbackStore(
        feedback_path=str(tmp_path / "feedback.jsonl"),
        bad_cases_path=str(tmp_path / "bad_cases.jsonl"),
    )
    knowledge_store = DerivedKnowledgeStore(path=str(tmp_path / "knowledge.jsonl"))
    return create_app(feedback_store=store, knowledge_store=knowledge_store), tmp_path


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
        "citations": [{"chunk_id": "c1", "source": "a.pdf", "page": 1}],
        "evidence": [
            {
                "chunk_id": "c1",
                "source": "a.pdf",
                "page": 1,
                "text_preview": "证据",
            }
        ],
    }
    # 同时也进总反馈日志。
    assert len(_read_jsonl(root / "feedback.jsonl")) == 1


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


# 验证纠错可以创建待审核知识场景。
@pytest.mark.anyio
async def test_correction_can_create_pending_knowledge(tmp_path, monkeypatch):
    app, root = _make_app(tmp_path, monkeypatch)

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


# 验证反馈拒绝非法请求场景。
@pytest.mark.anyio
async def test_feedback_rejects_invalid_payload(tmp_path, monkeypatch):
    app, _ = _make_app(tmp_path, monkeypatch)

    missing_trace = await _post(app, {"feedback": "thumbs_up"})
    bad_type = await _post(app, {"trace_id": "t3", "feedback": "love_it"})

    assert missing_trace.status_code == 422
    assert bad_type.status_code == 422
