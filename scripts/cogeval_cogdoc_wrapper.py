from __future__ import annotations

import os
import uuid
from typing import Any

import httpx
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="CogDoc CogEval Wrapper")

COGDOC_URL = os.getenv("COGDOC_URL", "http://127.0.0.1:8002/v1/chat").strip()
COGDOC_DOC_ID = os.getenv("COGDOC_DOC_ID", "arch_blueprint_2026").strip()
COGDOC_IS_LOCAL = os.getenv("COGDOC_IS_LOCAL", "true").strip().lower() in {
    "1",
    "true",
    "yes",
}
TIMEOUT_SECONDS = float(os.getenv("COGDOC_WRAPPER_TIMEOUT_SECONDS", "60"))


class InvokeRequest(BaseModel):
    message: str | None = None
    query: str | None = None
    input: Any | None = None
    trace_id: str | None = None
    metadata: dict[str, Any] | None = None


def _text(payload: InvokeRequest) -> str:
    if payload.message:
        return payload.message
    if payload.query:
        return payload.query
    if isinstance(payload.input, str):
        return payload.input
    if isinstance(payload.input, dict):
        for key in ("query", "message", "question", "prompt", "input"):
            value = payload.input.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return "请用一句话说明当前服务是否可用。"


def _trace_id(payload: InvokeRequest) -> str:
    if payload.trace_id:
        return payload.trace_id
    if payload.metadata and isinstance(payload.metadata.get("trace_id"), str):
        return str(payload.metadata["trace_id"])
    return uuid.uuid4().hex


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    ready_url = COGDOC_URL.rsplit("/v1/chat", 1)[0] + "/readyz"
    try:
        response = httpx.get(ready_url, timeout=10)
    except httpx.HTTPError as exc:
        return {
            "status": "degraded",
            "cogdoc_status": None,
            "error": type(exc).__name__,
        }
    return {
        "status": "ok" if response.is_success else "degraded",
        "cogdoc_status": response.status_code,
    }


@app.post("/invoke")
def invoke(payload: InvokeRequest) -> dict[str, Any]:
    requested_trace_id = _trace_id(payload)
    query = _text(payload)
    response = httpx.post(
        COGDOC_URL,
        json={"query": query, "doc_id": COGDOC_DOC_ID, "is_local": COGDOC_IS_LOCAL},
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    cogdoc_trace_id = str(data.get("trace_id") or requested_trace_id)
    output = str(data.get("answer") or data.get("output") or "")
    trace = {
        "id": requested_trace_id,
        "trace_id": requested_trace_id,
        "input": {"query": query, "doc_id": COGDOC_DOC_ID},
        "output": output,
        "metadata": {
            "agent_id": "cogdoc-local",
            "agent_version": "v3",
            "cogdoc_trace_id": cogdoc_trace_id,
            "task_type": data.get("task_type"),
        },
        "observations": [
            {
                "id": f"obs_{requested_trace_id}",
                "type": "GENERATION",
                "name": "cogdoc-chat",
                "input": {"query": query},
                "output": output,
                "metadata": {"doc_id": COGDOC_DOC_ID, "cogdoc_trace_id": cogdoc_trace_id},
            }
        ],
    }
    return {
        "output": output,
        "answer": output,
        "status": "SUCCESS",
        "trace_id": requested_trace_id,
        "trace": trace,
        "events": trace["observations"],
        "trace_completeness": "EVENTS",
    }
