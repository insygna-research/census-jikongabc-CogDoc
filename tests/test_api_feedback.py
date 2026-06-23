import json

import pytest
from httpx import ASGITransport, AsyncClient

from api.app import create_app
from api.feedback_store import FeedbackStore


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _make_app(tmp_path, monkeypatch):
    import api.app as app_module

    monkeypatch.setattr(app_module, "configure_logging", lambda: None)
    store = FeedbackStore(
        feedback_path=str(tmp_path / "feedback.jsonl"),
        bad_cases_path=str(tmp_path / "bad_cases.jsonl"),
    )
    return create_app(feedback_store=store), tmp_path


async def _post(app, payload):
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/v1/feedback", json=payload)


def _read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


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
        },
    )

    assert resp.status_code == 201 and resp.json()["is_bad_case"] is True
    bad = _read_jsonl(root / "bad_cases.jsonl")
    assert len(bad) == 1 and bad[0]["feedback"] == "thumbs_down"
    # 同时也进总反馈日志。
    assert len(_read_jsonl(root / "feedback.jsonl")) == 1


@pytest.mark.anyio
async def test_feedback_rejects_invalid_payload(tmp_path, monkeypatch):
    app, _ = _make_app(tmp_path, monkeypatch)

    missing_trace = await _post(app, {"feedback": "thumbs_up"})
    bad_type = await _post(app, {"trace_id": "t3", "feedback": "love_it"})

    assert missing_trace.status_code == 422
    assert bad_type.status_code == 422
