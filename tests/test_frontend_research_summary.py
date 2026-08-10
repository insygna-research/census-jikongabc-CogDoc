from contextlib import nullcontext

import httpx
import pytest

from cogdoc.frontend import app as frontend_app
from cogdoc.frontend.api_client import CogDocClient
from cogdoc.frontend.app import (
    _research_area,
    _research_summary_cache_key,
    _research_summary_progress_label,
    _research_summary_response_payload,
)


def test_research_summary_client_sends_cursor_and_conditional_header(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return httpx.Response(304)

    monkeypatch.setattr("cogdoc.frontend.api_client.httpx.get", fake_get)
    client = CogDocClient("http://api/", api_key="secret")

    response = client.list_research_job_summaries(
        "kb",
        status="running",
        limit=20,
        cursor="opaque-cursor",
        if_none_match='"rs-etag"',
    )

    assert response.status_code == 304
    assert calls == [
        (
            "http://api/v1/research-jobs/summaries",
            {
                "params": {
                    "kb_id": "kb",
                    "limit": 20,
                    "status": "running",
                    "cursor": "opaque-cursor",
                },
                "timeout": client.timeout,
                "headers": {
                    "Authorization": "Bearer secret",
                    "If-None-Match": '"rs-etag"',
                },
            },
        )
    ]


def test_research_summary_client_omits_empty_optional_query_and_header(monkeypatch):
    captured = {}

    def fake_get(url, **kwargs):
        captured.update({"url": url, **kwargs})
        return httpx.Response(200, json={"jobs": []})

    monkeypatch.setattr("cogdoc.frontend.api_client.httpx.get", fake_get)
    CogDocClient("http://api").list_research_job_summaries("kb")

    assert captured["params"] == {"kb_id": "kb", "limit": 20}
    assert captured["headers"] == {}


def test_research_summary_frontend_helpers_handle_304_and_progress_strictly():
    cached = {"jobs": [{"job_id": "rj_1"}], "next_cursor": None, "has_more": False}

    assert _research_summary_response_payload(httpx.Response(304), cached) is cached
    with pytest.raises(ValueError, match="没有可复用"):
        _research_summary_response_payload(httpx.Response(304), None)
    assert _research_summary_progress_label(
        {
            "section_counts": {
                "total": 4,
                "completed": 2,
                "running": 1,
                "failed": 1,
            }
        }
    ) == "章节 2/4 · 进行中 1 · 失败 1"
    assert _research_summary_progress_label(
        {"section_counts": {"total": True, "completed": 0, "running": 0, "failed": 0}}
    ) == "章节进度未知"
    assert _research_summary_cache_key("http://api", "kb", None) == (
        "research-summaries",
        "http://api",
        "anonymous",
        "kb",
        "",
        "",
    )
    assert _research_summary_cache_key(
        "http://api", "kb", "next", auth_identity="fingerprint"
    ) == (
        "research-summaries",
        "http://api",
        "fingerprint",
        "kb",
        "",
        "next",
    )


class _SessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


class _Column:
    def markdown(self, *_args, **_kwargs):
        return None

    def caption(self, *_args, **_kwargs):
        return None

    def button(self, *_args, **_kwargs):
        return False


class _StreamlitStub:
    def __init__(self):
        self.session_state = _SessionState(
            research_notice=None,
            research_summary_cache={},
            research_summary_pages={},
            research_open_job_by_kb={},
            is_local=False,
        )

    def __getattr__(self, name):
        if name in {"subheader", "caption", "markdown", "info", "error", "success"}:
            return lambda *_args, **_kwargs: None
        raise AttributeError(name)

    def button(self, *_args, **_kwargs):
        return False

    def form(self, *_args, **_kwargs):
        return nullcontext()

    def text_input(self, *_args, **_kwargs):
        return ""

    def text_area(self, *_args, **_kwargs):
        return ""

    def form_submit_button(self, *_args, **_kwargs):
        return False

    def columns(self, spec, **_kwargs):
        return [_Column() for _ in spec]

    def rerun(self):
        raise AssertionError("unselected summary index must not rerun")


class _SummaryOnlyClient:
    base_url = "http://api"

    def __init__(self):
        self.summary_calls = 0
        self.detail_calls = 0

    def list_research_job_summaries(self, *_args, **_kwargs):
        self.summary_calls += 1
        return httpx.Response(
            200,
            json={
                "schema_version": "v1",
                "jobs": [
                    {
                        "job_id": "rj_1",
                        "title": "资格案卷",
                        "objective_preview": "核对报名资格",
                        "status": "running",
                        "revision": 3,
                        "updated_at": "2026-08-10T02:00:00+00:00",
                        "section_counts": {
                            "total": 2,
                            "pending": 1,
                            "running": 1,
                            "completed": 0,
                            "failed": 0,
                        },
                        "provenance_status": "current",
                        "review_status": "not_started",
                    }
                ],
                "next_cursor": None,
                "has_more": False,
            },
            headers={"ETag": '"rs-1"'},
        )

    def get_research_job(self, *_args, **_kwargs):
        self.detail_calls += 1
        raise AssertionError("an unopened dossier must not fetch detail")


def test_research_area_does_not_fetch_or_build_unselected_job_detail(monkeypatch):
    streamlit_stub = _StreamlitStub()
    client = _SummaryOnlyClient()
    monkeypatch.setattr(frontend_app, "st", streamlit_stub)
    monkeypatch.setattr(frontend_app, "_client", lambda: client)

    _research_area("kb")

    assert client.summary_calls == 1
    assert client.detail_calls == 0
    assert streamlit_stub.session_state.research_open_job_by_kb == {}
