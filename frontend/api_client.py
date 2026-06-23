import json
from typing import Iterable, Iterator
import httpx

DEFAULT_TIMEOUT = 180.0


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


class CogDocClient:
    # 交付层瘦客户端：只打 /v1，不碰后端智能逻辑。
    def __init__(self, base_url: str, timeout: float = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def list_knowledge_bases(self) -> list[dict]:
        return httpx.get(self._url("/v1/knowledge-bases"), timeout=self.timeout).json()

    def create_knowledge_base(self, kb_id: str) -> httpx.Response:
        return httpx.post(
            self._url("/v1/knowledge-bases"),
            json={"kb_id": kb_id},
            timeout=self.timeout,
        )

    def list_documents(self, kb_id: str) -> httpx.Response:
        return httpx.get(
            self._url(f"/v1/knowledge-bases/{kb_id}/documents"), timeout=self.timeout
        )

    def upload_document(
        self, kb_id: str, filename: str, content: bytes
    ) -> httpx.Response:
        files = {"file": (filename, content, "application/pdf")}
        return httpx.post(
            self._url(f"/v1/knowledge-bases/{kb_id}/documents"),
            files=files,
            timeout=self.timeout,
        )

    def delete_document(self, kb_id: str, name: str) -> httpx.Response:
        return httpx.delete(
            self._url(f"/v1/knowledge-bases/{kb_id}/documents/{name}"),
            timeout=self.timeout,
        )

    def get_job(self, job_id: str) -> httpx.Response:
        return httpx.get(self._url(f"/v1/index-jobs/{job_id}"), timeout=self.timeout)

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
        )

    def _chat_payload(
        self, kb_id: str, query: str, mode: str, session_id: str | None, is_local: bool
    ) -> dict:
        payload = {"query": query, "doc_id": kb_id, "mode": mode, "is_local": is_local}
        if session_id:
            payload["session_id"] = session_id
        return payload

    def stream_chat(
        self,
        kb_id: str,
        query: str,
        mode: str = "auto",
        session_id: str | None = None,
        is_local: bool = False,
    ) -> Iterator[tuple[str, dict]]:
        payload = self._chat_payload(kb_id, query, mode, session_id, is_local)
        with httpx.stream(
            "POST", self._url("/v1/chat/stream"), json=payload, timeout=self.timeout
        ) as response:
            if response.status_code != 200:
                # 流式响应非 200 不会抛异常，需读出 body 转成 error 事件，避免静默成空答案。
                response.read()
                try:
                    yield "error", response.json()
                except ValueError:
                    yield "error", {"message": response.text}
                return
            yield from iter_sse_events(response.iter_lines())
