import json
import os
from typing import Any, Callable, Iterable, Iterator, Mapping
import httpx

DEFAULT_TIMEOUT = 180.0


class CogDocAPIError(RuntimeError):
    # 后端返回结构化错误体或非预期响应时抛出，供 UI 直接展示。
    def __init__(
        self, message: str, status_code: int | None = None, payload: Any = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


def format_api_error(
    payload: Any, status_code: int | None = None, fallback: str = "请求失败"
) -> str:
    status = f"HTTP {status_code}: " if status_code is not None else ""
    if isinstance(payload, Mapping):
        code = payload.get("error_code")
        message = payload.get("message")
        if code and message:
            return f"{status}[{code}] {message}"
        if message:
            return f"{status}{message}"
    if payload not in (None, ""):
        return f"{status}{fallback}: {payload}"
    return f"{status}{fallback}"


def response_payload(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text[:200]


def _response_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise CogDocAPIError(
            f"HTTP {response.status_code}: 后端返回非 JSON 响应: {response.text[:200]}",
            status_code=response.status_code,
        ) from exc


def _checked_json(response: httpx.Response) -> Any:
    payload = _response_json(response)
    if response.status_code >= 400:
        raise CogDocAPIError(
            format_api_error(payload, response.status_code),
            status_code=response.status_code,
            payload=payload,
        )
    return payload


def _expect_list(payload: Any, label: str) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping) and payload.get("error_code"):
        raise CogDocAPIError(format_api_error(payload), payload=payload)
    raise CogDocAPIError(f"{label}响应格式不符合预期: {payload}")


# 完成 iterSSE事件列表 处理。
def iter_sse_events(lines: Iterable[str]) -> Iterator[tuple[str, dict]]:
    # 把 SSE 行流解析成 (event_name, data)；空行结束一帧，非 JSON data 跳过。
    event_name = "message"
    for line in lines:
        if not line:
            event_name = "message"
            continue
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            raw = line[len("data:") :].strip()
            try:
                yield event_name, json.loads(raw)
            except json.JSONDecodeError:
                continue


# 交付层瘦客户端：只打 /v1，不碰后端智能逻辑。
class CogDocClient:
    # 交付层瘦客户端：只打 /v1，不碰后端智能逻辑。
    def __init__(
        self,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT,
        api_key: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # 后端开启鉴权时带上 key；缺省读环境变量，未配置则不带头（鉴权关闭场景）。
        key = api_key if api_key is not None else os.getenv("COGDOC_API_KEY", "")
        self._headers = {"Authorization": f"Bearer {key}"} if key else {}

    # 拼接结果。
    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    # 列出 knowledge bases。
    def list_knowledge_bases(self) -> list[dict]:
        response = httpx.get(
            self._url("/v1/knowledge-bases"),
            timeout=self.timeout,
            headers=self._headers,
        )
        return _expect_list(_checked_json(response), "知识库列表")

    # 创建 knowledge base。
    def create_knowledge_base(self, kb_id: str) -> httpx.Response:
        return httpx.post(
            self._url("/v1/knowledge-bases"),
            json={"kb_id": kb_id},
            timeout=self.timeout,
            headers=self._headers,
        )

    # 删除 knowledge base。
    def delete_knowledge_base(self, kb_id: str) -> httpx.Response:
        return httpx.delete(
            self._url(f"/v1/knowledge-bases/{kb_id}"),
            timeout=self.timeout,
            headers=self._headers,
        )

    # 列出 documents。
    def list_documents(self, kb_id: str) -> httpx.Response:
        return httpx.get(
            self._url(f"/v1/knowledge-bases/{kb_id}/documents"),
            timeout=self.timeout,
            headers=self._headers,
        )

    # 完成 上传document 处理。
    def upload_document(
        self, kb_id: str, filename: str, content: bytes
    ) -> httpx.Response:
        files = {"file": (filename, content, "application/pdf")}
        return httpx.post(
            self._url(f"/v1/knowledge-bases/{kb_id}/documents"),
            files=files,
            timeout=self.timeout,
            headers=self._headers,
        )

    # 删除 document。
    def delete_document(self, kb_id: str, name: str) -> httpx.Response:
        return httpx.delete(
            self._url(f"/v1/knowledge-bases/{kb_id}/documents/{name}"),
            timeout=self.timeout,
            headers=self._headers,
        )

    # 获取 job。
    def get_job(self, job_id: str) -> httpx.Response:
        return httpx.get(
            self._url(f"/v1/index-jobs/{job_id}"),
            timeout=self.timeout,
            headers=self._headers,
        )

    # 获取 trace。
    def get_trace(self, trace_id: str) -> httpx.Response:
        return httpx.get(
            self._url(f"/v1/traces/{trace_id}"),
            timeout=self.timeout,
            headers=self._headers,
        )

    # 列出 traces。
    def list_traces(self, limit: int = 20) -> httpx.Response:
        return httpx.get(
            self._url("/v1/traces"),
            params={"limit": limit},
            timeout=self.timeout,
            headers=self._headers,
        )

    # 获取 session history。
    def get_session_history(self, session_id: str, kb_id: str) -> httpx.Response:
        return httpx.get(
            self._url(f"/v1/sessions/{session_id}/history"),
            params={"doc_id": kb_id},
            timeout=self.timeout,
            headers=self._headers,
        )

    # 列出 sessions。
    def list_sessions(self, kb_id: str) -> httpx.Response:
        return httpx.get(
            self._url("/v1/sessions"),
            params={"doc_id": kb_id},
            timeout=self.timeout,
            headers=self._headers,
        )

    # 删除 session。
    def delete_session(self, session_id: str, kb_id: str) -> httpx.Response:
        return httpx.delete(
            self._url(f"/v1/sessions/{session_id}"),
            params={"doc_id": kb_id},
            timeout=self.timeout,
            headers=self._headers,
        )

    # 提交 feedback。
    def submit_feedback(
        self,
        trace_id: str,
        feedback: str,
        kb_id: str | None = None,
        query: str | None = None,
        answer: str | None = None,
    ) -> httpx.Response:
        payload = {
            "trace_id": trace_id,
            "feedback": feedback,
            "kb_id": kb_id,
            "query": query,
            "answer": answer,
        }
        return httpx.post(
            self._url("/v1/feedback"),
            json={k: v for k, v in payload.items() if v is not None},
            timeout=self.timeout,
            headers=self._headers,
        )

    # 完成 chat请求体 处理。
    def _chat_payload(
        self, kb_id: str, query: str, mode: str, session_id: str | None, is_local: bool
    ) -> dict:
        payload = {"query": query, "doc_id": kb_id, "mode": mode, "is_local": is_local}
        if session_id:
            payload["session_id"] = session_id
        return payload

    # 流式返回chat。
    def stream_chat(
        self,
        kb_id: str,
        query: str,
        mode: str = "auto",
        session_id: str | None = None,
        is_local: bool = False,
        on_response: Callable[[httpx.Response], None] | None = None,
    ) -> Iterator[tuple[str, dict]]:
        payload = self._chat_payload(kb_id, query, mode, session_id, is_local)
        with httpx.stream(
            "POST",
            self._url("/v1/chat/stream"),
            json=payload,
            timeout=self.timeout,
            headers=self._headers,
        ) as response:
            if on_response is not None:
                on_response(response)
            if response.status_code != 200:
                # 流式响应非 200 不会抛异常，需读出 body 转成 error 事件，避免静默成空答案。
                response.read()
                try:
                    yield "error", response.json()
                except ValueError:
                    yield "error", {"message": response.text}
                return
            yield from iter_sse_events(response.iter_lines())
