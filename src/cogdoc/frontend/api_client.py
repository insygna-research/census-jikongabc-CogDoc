import json
import os
from collections.abc import Mapping
from typing import Any, Callable, Iterable, Iterator
import httpx

DEFAULT_TIMEOUT = 180.0


# 定义接口错误。
class CogDocAPIError(RuntimeError):
    # 后端返回结构化错误体或非预期响应时抛出，供界面直接展示。
    def __init__(
        self, message: str, status_code: int | None = None, payload: Any = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


# 格式化接口错误。
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


# 处理响应载荷。
def response_payload(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text[:200]


# 处理响应对象。
def _response_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise CogDocAPIError(
            f"HTTP {response.status_code}: 后端返回非 JSON 响应: {response.text[:200]}",
            status_code=response.status_code,
        ) from exc


# 处理已校验响应对象。
def _checked_json(response: httpx.Response) -> Any:
    payload = _response_json(response)
    if response.status_code >= 400:
        raise CogDocAPIError(
            format_api_error(payload, response.status_code),
            status_code=response.status_code,
            payload=payload,
        )
    return payload


# 处理预期列表。
def _expect_list(payload: Any, label: str) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping) and payload.get("error_code"):
        raise CogDocAPIError(format_api_error(payload), payload=payload)
    raise CogDocAPIError(f"{label}响应格式不符合预期: {payload}")


# 解析流式事件列表。
def iter_sse_events(lines: Iterable[str]) -> Iterator[tuple[str, dict]]:
    # 把行流解析成事件名和数据；空行结束一帧，非法数据跳过。
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


# 交付层瘦客户端：只打版本接口，不碰后端智能逻辑。
class CogDocClient:
    # 交付层瘦客户端：只打版本接口，不碰后端智能逻辑。
    def __init__(
        self,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT,
        api_key: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # 后端开启鉴权时带上密钥；缺省读环境变量，未配置则不带头。
        key = api_key if api_key is not None else os.getenv("COGDOC_API_KEY", "")
        self._headers = {"Authorization": f"Bearer {key}"} if key else {}

    # 拼接结果。
    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    # 列出知识库。
    def list_knowledge_bases(self) -> list[dict]:
        response = httpx.get(
            self._url("/v1/knowledge-bases"),
            timeout=self.timeout,
            headers=self._headers,
        )
        return _expect_list(_checked_json(response), "知识库列表")

    # 创建知识库。
    def create_knowledge_base(self, kb_id: str) -> httpx.Response:
        return httpx.post(
            self._url("/v1/knowledge-bases"),
            json={"kb_id": kb_id},
            timeout=self.timeout,
            headers=self._headers,
        )

    # 删除知识库。
    def delete_knowledge_base(self, kb_id: str) -> httpx.Response:
        return httpx.delete(
            self._url(f"/v1/knowledge-bases/{kb_id}"),
            timeout=self.timeout,
            headers=self._headers,
        )

    # 列出文档。
    def list_documents(self, kb_id: str) -> httpx.Response:
        return httpx.get(
            self._url(f"/v1/knowledge-bases/{kb_id}/documents"),
            timeout=self.timeout,
            headers=self._headers,
        )

    # 列出知识库来源文件。
    def list_sources(self, kb_id: str) -> httpx.Response:
        return httpx.get(
            self._url(f"/v1/knowledge-bases/{kb_id}/sources"),
            timeout=self.timeout,
            headers=self._headers,
        )

    # 列出来源文件分块。
    def list_source_chunks(
        self,
        kb_id: str,
        source: str,
        offset: int = 0,
        limit: int = 50,
        anchor_text: str | None = None,
    ) -> httpx.Response:
        params = {"offset": offset, "limit": limit, "anchor_text": anchor_text}
        return httpx.get(
            self._url(f"/v1/knowledge-bases/{kb_id}/sources/{source}/chunks"),
            params={k: v for k, v in params.items() if v is not None},
            timeout=self.timeout,
            headers=self._headers,
        )

    # 上传文档。
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

    # 删除文档。
    def delete_document(self, kb_id: str, name: str) -> httpx.Response:
        return httpx.delete(
            self._url(f"/v1/knowledge-bases/{kb_id}/documents/{name}"),
            timeout=self.timeout,
            headers=self._headers,
        )

    # 获取任务。
    def get_job(self, job_id: str) -> httpx.Response:
        return httpx.get(
            self._url(f"/v1/index-jobs/{job_id}"),
            timeout=self.timeout,
            headers=self._headers,
        )

    # 获取跟踪。
    def get_trace(self, trace_id: str) -> httpx.Response:
        return httpx.get(
            self._url(f"/v1/traces/{trace_id}"),
            timeout=self.timeout,
            headers=self._headers,
        )

    # 列出跟踪。
    def list_traces(
        self, limit: int = 20, kb_id: str = "", session_id: str = ""
    ) -> httpx.Response:
        return httpx.get(
            self._url("/v1/traces"),
            params={"limit": limit, "doc_id": kb_id, "session_id": session_id},
            timeout=self.timeout,
            headers=self._headers,
        )

    # 获取会话历史。
    def get_session_history(self, session_id: str, kb_id: str) -> httpx.Response:
        return httpx.get(
            self._url(f"/v1/sessions/{session_id}/history"),
            params={"doc_id": kb_id},
            timeout=self.timeout,
            headers=self._headers,
        )

    # 列出会话。
    def list_sessions(self, kb_id: str) -> httpx.Response:
        return httpx.get(
            self._url("/v1/sessions"),
            params={"doc_id": kb_id},
            timeout=self.timeout,
            headers=self._headers,
        )

    # 删除会话。
    def delete_session(self, session_id: str, kb_id: str) -> httpx.Response:
        return httpx.delete(
            self._url(f"/v1/sessions/{session_id}"),
            params={"doc_id": kb_id},
            timeout=self.timeout,
            headers=self._headers,
        )

    # 提交反馈。
    def submit_feedback(
        self,
        trace_id: str,
        feedback: str,
        kb_id: str | None = None,
        query: str | None = None,
        answer: str | None = None,
        citations: list[dict] | None = None,
        evidence: list[dict] | None = None,
        comment: str | None = None,
        correction: str | None = None,
        feedback_type: str | None = None,
        feedback_text: str | None = None,
        correction_text: str | None = None,
        save_as_knowledge: bool = False,
        related_source: str | None = None,
        related_source_sha256: str | None = None,
        related_chunk_ids: list[str] | None = None,
        related_page_start: int | None = None,
        related_page_end: int | None = None,
        related_chunk_text_hash: str | None = None,
        related_anchor_text: str | None = None,
        certainty: str | None = None,
        created_by: str | None = None,
    ) -> httpx.Response:
        payload = {
            "trace_id": trace_id,
            "feedback": feedback,
            "kb_id": kb_id,
            "query": query,
            "answer": answer,
            "citations": citations or [],
            "evidence": evidence or [],
            "comment": comment,
            "correction": correction,
            "feedback_type": feedback_type,
            "feedback_text": feedback_text,
            "correction_text": correction_text,
            "save_as_knowledge": save_as_knowledge,
            "related_source": related_source,
            "related_source_sha256": related_source_sha256,
            "related_chunk_ids": related_chunk_ids or [],
            "related_page_start": related_page_start,
            "related_page_end": related_page_end,
            "related_chunk_text_hash": related_chunk_text_hash,
            "related_anchor_text": related_anchor_text,
            "certainty": certainty,
            "created_by": created_by,
        }
        return httpx.post(
            self._url("/v1/feedback"),
            json={k: v for k, v in payload.items() if v is not None},
            timeout=self.timeout,
            headers=self._headers,
        )

    # 查询反馈记录。
    def list_feedback(
        self,
        kb_id: str,
        trace_id: str | None = None,
        session_id: str | None = None,
        feedback: str | None = None,
        feedback_type: str | None = None,
        is_bad_case: bool | None = None,
        limit: int = 100,
    ) -> httpx.Response:
        params = {
            "kb_id": kb_id,
            "trace_id": trace_id,
            "session_id": session_id,
            "feedback": feedback,
            "feedback_type": feedback_type,
            "is_bad_case": is_bad_case,
            "limit": limit,
        }
        return httpx.get(
            self._url("/v1/feedback"),
            params={k: v for k, v in params.items() if v is not None},
            timeout=self.timeout,
            headers=self._headers,
        )

    # 查询反馈理解结果。
    def list_feedback_analysis(
        self,
        kb_id: str,
        recommended_action: str | None = None,
        needs_review: bool | None = None,
        limit: int = 100,
    ) -> httpx.Response:
        params = {
            "kb_id": kb_id,
            "recommended_action": recommended_action,
            "needs_review": needs_review,
            "limit": limit,
        }
        return httpx.get(
            self._url("/v1/feedback-analysis"),
            params={k: v for k, v in params.items() if v is not None},
            timeout=self.timeout,
            headers=self._headers,
        )

    # 新增派生知识。
    def create_knowledge(
        self,
        *,
        kb_id: str,
        text: str,
        related_source: str | None = None,
        related_source_sha256: str | None = None,
        related_chunk_ids: list[str] | None = None,
        related_page_start: int | None = None,
        related_page_end: int | None = None,
        related_chunk_text_hash: str | None = None,
        related_anchor_text: str | None = None,
        source_note: str | None = None,
        certainty: str = "medium",
        origin: str = "manual_entry",
        created_from_trace_id: str | None = None,
        created_by: str | None = None,
        enable_immediately: bool = False,
    ) -> httpx.Response:
        payload = {
            "kb_id": kb_id,
            "text": text,
            "related_source": related_source,
            "related_source_sha256": related_source_sha256,
            "related_chunk_ids": related_chunk_ids or [],
            "related_page_start": related_page_start,
            "related_page_end": related_page_end,
            "related_chunk_text_hash": related_chunk_text_hash,
            "related_anchor_text": related_anchor_text,
            "source_note": source_note,
            "certainty": certainty,
            "origin": origin,
            "created_from_trace_id": created_from_trace_id,
            "created_by": created_by,
            "enable_immediately": enable_immediately,
        }
        return httpx.post(
            self._url("/v1/knowledge"),
            json={k: v for k, v in payload.items() if v is not None},
            timeout=self.timeout,
            headers=self._headers,
        )

    # 查询派生知识。
    def list_knowledge(
        self,
        kb_id: str,
        status: str | None = None,
        document_id: str | None = None,
        origin: str | None = None,
        created_by: str | None = None,
        conflict_group_id: str | None = None,
        has_conflict: bool | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
    ) -> httpx.Response:
        params = {
            "kb_id": kb_id,
            "status": status,
            "document_id": document_id,
            "origin": origin,
            "created_by": created_by,
            "conflict_group_id": conflict_group_id,
            "has_conflict": has_conflict,
            "created_after": created_after,
            "created_before": created_before,
        }
        return httpx.get(
            self._url("/v1/knowledge"),
            params={k: v for k, v in params.items() if v is not None},
            timeout=self.timeout,
            headers=self._headers,
        )

    # 查询审核队列摘要。
    def review_queue_summary(
        self,
        kb_id: str,
        document_id: str | None = None,
        origin: str | None = None,
        created_by: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
    ) -> httpx.Response:
        params = {
            "kb_id": kb_id,
            "document_id": document_id,
            "origin": origin,
            "created_by": created_by,
            "created_after": created_after,
            "created_before": created_before,
        }
        return httpx.get(
            self._url("/v1/review-queue"),
            params={k: v for k, v in params.items() if v is not None},
            timeout=self.timeout,
            headers=self._headers,
        )

    # 导出审核队列。
    def review_queue_export(
        self,
        kb_id: str,
        limit: int = 200,
        knowledge_document_id: str | None = None,
        knowledge_origin: str | None = None,
        knowledge_created_by: str | None = None,
        knowledge_created_after: str | None = None,
        knowledge_created_before: str | None = None,
    ) -> httpx.Response:
        params = {
            "kb_id": kb_id,
            "limit": limit,
            "knowledge_document_id": knowledge_document_id,
            "knowledge_origin": knowledge_origin,
            "knowledge_created_by": knowledge_created_by,
            "knowledge_created_after": knowledge_created_after,
            "knowledge_created_before": knowledge_created_before,
        }
        return httpx.get(
            self._url("/v1/review-queue/export"),
            params={k: v for k, v in params.items() if v is not None},
            timeout=self.timeout,
            headers=self._headers,
        )

    # 查询待审核计数。
    def pending_knowledge_count(self, kb_id: str) -> httpx.Response:
        return httpx.get(
            self._url("/v1/knowledge/pending-count"),
            params={"kb_id": kb_id},
            timeout=self.timeout,
            headers=self._headers,
        )

    # 查询反馈闭环指标。
    def feedback_loop_metrics(
        self,
        kb_id: str,
        answer_count: int | None = None,
    ) -> httpx.Response:
        params = {"kb_id": kb_id, "answer_count": answer_count}
        return httpx.get(
            self._url("/v1/feedback-loop-metrics"),
            params={k: v for k, v in params.items() if v is not None},
            timeout=self.timeout,
            headers=self._headers,
        )

    # 审核派生知识。
    def review_knowledge(
        self,
        knowledge_id: str,
        action: str,
        actor: str | None = None,
        note: str | None = None,
        related_document_id: str | None = None,
        related_source: str | None = None,
        related_source_sha256: str | None = None,
        related_chunk_ids: list[str] | None = None,
        related_page_start: int | None = None,
        related_page_end: int | None = None,
        related_chunk_text_hash: str | None = None,
        related_anchor_text: str | None = None,
    ) -> httpx.Response:
        payload = {
            "actor": actor,
            "note": note,
            "related_document_id": related_document_id,
            "related_source": related_source,
            "related_source_sha256": related_source_sha256,
            "related_chunk_ids": related_chunk_ids,
            "related_page_start": related_page_start,
            "related_page_end": related_page_end,
            "related_chunk_text_hash": related_chunk_text_hash,
            "related_anchor_text": related_anchor_text,
        }
        return httpx.post(
            self._url(f"/v1/knowledge/{knowledge_id}/{action}"),
            json={k: v for k, v in payload.items() if v is not None},
            timeout=self.timeout,
            headers=self._headers,
        )

    # 创建知识修订版本。
    def revise_knowledge(
        self,
        knowledge_id: str,
        *,
        text: str,
        related_document_id: str | None = None,
        related_source: str | None = None,
        related_source_sha256: str | None = None,
        related_chunk_ids: list[str] | None = None,
        related_page_start: int | None = None,
        related_page_end: int | None = None,
        related_chunk_text_hash: str | None = None,
        related_anchor_text: str | None = None,
        source_note: str | None = None,
        certainty: str = "medium",
        created_from_trace_id: str | None = None,
        created_by: str | None = None,
        enable_immediately: bool = False,
    ) -> httpx.Response:
        payload = {
            "text": text,
            "related_document_id": related_document_id,
            "related_source": related_source,
            "related_source_sha256": related_source_sha256,
            "related_chunk_ids": related_chunk_ids,
            "related_page_start": related_page_start,
            "related_page_end": related_page_end,
            "related_chunk_text_hash": related_chunk_text_hash,
            "related_anchor_text": related_anchor_text,
            "source_note": source_note,
            "certainty": certainty,
            "created_from_trace_id": created_from_trace_id,
            "created_by": created_by,
            "enable_immediately": enable_immediately,
        }
        return httpx.post(
            self._url(f"/v1/knowledge/{knowledge_id}/revise"),
            json={k: v for k, v in payload.items() if v is not None},
            timeout=self.timeout,
            headers=self._headers,
        )

    # 批量审核派生知识。
    def batch_review_knowledge(
        self,
        knowledge_ids: list[str],
        action: str,
        actor: str | None = None,
        note: str | None = None,
    ) -> httpx.Response:
        payload = {"knowledge_ids": knowledge_ids, "actor": actor, "note": note}
        return httpx.post(
            self._url(f"/v1/knowledge/{action}"),
            json={k: v for k, v in payload.items() if v is not None},
            timeout=self.timeout,
            headers=self._headers,
        )

    # 查询检索调权反馈。
    def list_retrieval_feedback(
        self,
        kb_id: str,
        enabled: bool | None = None,
        limit: int = 100,
    ) -> httpx.Response:
        params = {"kb_id": kb_id, "enabled": enabled, "limit": limit}
        return httpx.get(
            self._url("/v1/retrieval-feedback"),
            params={k: v for k, v in params.items() if v is not None},
            timeout=self.timeout,
            headers=self._headers,
        )

    # 设置检索调权反馈状态。
    def set_retrieval_feedback_enabled(
        self,
        feedback_id: str,
        enabled: bool,
        actor: str | None = None,
        reason: str | None = None,
    ) -> httpx.Response:
        action = "enable" if enabled else "disable"
        payload = {"actor": actor, "reason": reason}
        kwargs = {"timeout": self.timeout, "headers": self._headers}
        if not enabled:
            kwargs["json"] = {k: v for k, v in payload.items() if v is not None}
        return httpx.post(
            self._url(f"/v1/retrieval-feedback/{feedback_id}/{action}"),
            **kwargs,
        )

    # 构造对话请求体。
    def _chat_payload(
        self, kb_id: str, query: str, mode: str, session_id: str | None, is_local: bool
    ) -> dict:
        payload = {"query": query, "doc_id": kb_id, "mode": mode, "is_local": is_local}
        if session_id:
            payload["session_id"] = session_id
        return payload

    # 调用独立摘要接口。
    def summary(
        self,
        kb_id: str,
        query: str,
        session_id: str | None = None,
        is_local: bool = False,
    ) -> httpx.Response:
        payload = {
            "query": query,
            "doc_id": kb_id,
            "session_id": session_id,
            "is_local": is_local,
        }
        return httpx.post(
            self._url("/v1/summary"),
            json={k: v for k, v in payload.items() if v is not None},
            timeout=self.timeout,
            headers=self._headers,
        )

    # 调用独立对比接口。
    def compare(
        self,
        kb_id: str,
        query: str,
        session_id: str | None = None,
        is_local: bool = False,
    ) -> httpx.Response:
        payload = {
            "query": query,
            "doc_id": kb_id,
            "session_id": session_id,
            "is_local": is_local,
        }
        return httpx.post(
            self._url("/v1/compare"),
            json={k: v for k, v in payload.items() if v is not None},
            timeout=self.timeout,
            headers=self._headers,
        )

    # 调用独立检索接口。
    def retrieve(
        self,
        kb_id: str,
        query: str,
        top_k: int = 8,
        rerank: bool = False,
        rerank_top_n: int | None = None,
    ) -> httpx.Response:
        payload = {
            "query": query,
            "doc_id": kb_id,
            "top_k": top_k,
            "rerank": rerank,
            "rerank_top_n": rerank_top_n,
        }
        return httpx.post(
            self._url("/v1/retrieve"),
            json={k: v for k, v in payload.items() if v is not None},
            timeout=self.timeout,
            headers=self._headers,
        )

    # 流式返回对话。
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
                # 流式响应失败不会抛异常，需读出正文转成错误事件，避免静默成空答案。
                response.read()
                try:
                    yield "error", response.json()
                except ValueError:
                    yield "error", {"message": response.text}
                return
            yield from iter_sse_events(response.iter_lines())
