import logging

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from cogdoc.agents.feedback_understanding import FeedbackAnalysis, analyze_feedback
from cogdoc.api.schemas import (
    FeedbackIssueType,
    FeedbackListResponse,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackType,
)
from cogdoc.observability.logger import log_event

router = APIRouter(prefix="/v1", tags=["feedback"])


# 构建纠错派生知识草稿。
def _knowledge_payload(body: FeedbackRequest) -> dict | None:
    correction = body.correction_text or body.correction
    if not body.save_as_knowledge or not correction or not body.kb_id:
        return None
    related_chunk_ids = body.related_chunk_ids or [
        item.chunk_id for item in body.citations if item.chunk_id
    ]
    related_source = body.related_source or next(
        (item.source for item in body.citations if item.source), None
    )
    return {
        "kb_id": body.kb_id,
        "text": correction,
        "related_document_id": body.related_document_id,
        "related_source": related_source,
        "related_source_sha256": body.related_source_sha256,
        "related_chunk_ids": related_chunk_ids,
        "source_note": body.source_note or body.feedback_text or body.comment,
        "certainty": body.certainty,
        "status": "pending",
        "origin": "correction",
        "created_from_trace_id": body.trace_id,
        "created_by": body.created_by,
    }


# 构建反馈理解建议的知识草稿。
def _analysis_knowledge_payload(
    body: FeedbackRequest, analysis: FeedbackAnalysis, confidence: float
) -> dict | None:
    extracted_claim = analysis.get("extracted_claim")
    correction = extracted_claim.strip() if isinstance(extracted_claim, str) else ""
    if (
        not body.kb_id
        or not correction
        or analysis.get("recommended_action") != "create_pending_knowledge"
        or confidence < 0.8
    ):
        return None
    related_chunk_ids = body.related_chunk_ids or [
        item.chunk_id for item in body.citations if item.chunk_id
    ]
    related_source = body.related_source or next(
        (item.source for item in body.citations if item.source), None
    )
    return {
        "kb_id": body.kb_id,
        "text": correction,
        "related_document_id": body.related_document_id,
        "related_source": related_source,
        "related_source_sha256": body.related_source_sha256,
        "related_chunk_ids": related_chunk_ids,
        "source_note": body.source_note or body.feedback_text or body.comment,
        "certainty": "high" if confidence >= 0.9 else body.certainty,
        "status": "pending",
        "origin": "agent_suggested",
        "created_from_trace_id": body.trace_id,
        "created_by": body.created_by,
    }


# 分析反馈，失败时降级为仅记录原始反馈。
def _analyze_feedback_quiet(feedback_id: str, payload: dict) -> FeedbackAnalysis | None:
    try:
        return analyze_feedback(payload)
    except Exception as exc:
        log_event(
            "feedback",
            "feedback_analysis_failed",
            {},
            level=logging.WARNING,
            feedback_id=feedback_id,
            error_class=type(exc).__name__,
        )
        return None


# 记录反馈理解结果，失败时不阻断反馈提交。
def _record_feedback_analysis_quiet(
    request: Request,
    feedback_id: str,
    payload: dict,
    analysis: FeedbackAnalysis,
) -> dict | None:
    try:
        return request.app.state.feedback_analysis_store.record(
            feedback_id, payload, analysis
        )
    except Exception as exc:
        log_event(
            "feedback",
            "feedback_analysis_record_failed",
            {},
            level=logging.WARNING,
            feedback_id=feedback_id,
            error_class=type(exc).__name__,
        )
        return None


# 创建知识草稿，失败时不阻断反馈提交。
def _create_knowledge_quiet(
    request: Request,
    feedback_id: str,
    knowledge_payload: dict,
) -> tuple[str | None, str | None, bool]:
    try:
        knowledge, deduplicated = request.app.state.knowledge_store.create(
            knowledge_payload
        )
        return knowledge["knowledge_id"], knowledge["status"], deduplicated
    except Exception as exc:
        log_event(
            "feedback",
            "knowledge_create_failed",
            {},
            level=logging.WARNING,
            feedback_id=feedback_id,
            error_class=type(exc).__name__,
        )
        return None, None, False


# 提交反馈。
@router.post("/feedback", status_code=201)
async def submit_feedback(body: FeedbackRequest, request: Request):
    # 控制层只落盘，不做评判；坏样本归集逻辑在存储层里。
    payload = body.model_dump(exclude_none=True)
    if body.feedback_text and not payload.get("comment"):
        payload["comment"] = body.feedback_text
    if body.correction_text and not payload.get("correction"):
        payload["correction"] = body.correction_text
    result = request.app.state.feedback_store.record(payload)
    request.app.state.retrieval_feedback_store.record_from_feedback(
        result["feedback_id"], payload
    )
    analysis = _analyze_feedback_quiet(result["feedback_id"], payload)
    analysis_row = None
    if analysis is not None:
        analysis_row = _record_feedback_analysis_quiet(
            request, result["feedback_id"], payload, analysis
        )
    knowledge_id = None
    knowledge_status = None
    knowledge_deduplicated = False
    knowledge_payload = _knowledge_payload(body)
    if knowledge_payload is None and analysis is not None:
        knowledge_payload = _analysis_knowledge_payload(
            body, analysis, float(analysis.get("confidence") or 0.0)
        )
    if knowledge_payload is not None:
        knowledge_id, knowledge_status, knowledge_deduplicated = (
            _create_knowledge_quiet(request, result["feedback_id"], knowledge_payload)
        )
    return FeedbackResponse(
        feedback_id=result["feedback_id"],
        is_bad_case=result["is_bad_case"],
        feedback_analysis_id=(
            analysis_row["feedback_analysis_id"] if analysis_row else None
        ),
        feedback_analysis_action=(
            analysis_row["recommended_action"] if analysis_row else None
        ),
        feedback_analysis_confidence=(
            analysis_row["confidence"] if analysis_row else None
        ),
        knowledge_id=knowledge_id,
        knowledge_status=knowledge_status,
        knowledge_deduplicated=knowledge_deduplicated,
    )


# 查询反馈记录。
@router.get("/feedback", response_model=FeedbackListResponse)
async def list_feedback(
    request: Request,
    kb_id: str = Query(min_length=1),
    trace_id: str | None = None,
    session_id: str | None = None,
    feedback: FeedbackType | None = None,
    feedback_type: FeedbackIssueType | None = None,
    is_bad_case: bool | None = None,
    limit: int = Query(default=100, ge=1, le=500),
):
    rows = request.app.state.feedback_store.list(
        kb_id=kb_id,
        trace_id=trace_id,
        session_id=session_id,
        feedback=feedback.value if feedback is not None else None,
        feedback_type=feedback_type.value if feedback_type is not None else None,
        is_bad_case=is_bad_case,
        limit=limit,
    )
    return FeedbackListResponse(feedback=rows)


# 查询反馈理解结果。
@router.get("/feedback-analysis")
async def list_feedback_analysis(
    request: Request,
    kb_id: str = Query(min_length=1),
    feedback_id: str | None = None,
    trace_id: str | None = None,
    recommended_action: str | None = None,
    needs_review: bool | None = None,
    min_confidence: float | None = Query(default=None, ge=0, le=1),
    limit: int = Query(default=100, ge=1, le=500),
):
    rows = request.app.state.feedback_analysis_store.list(
        kb_id=kb_id,
        feedback_id=feedback_id,
        trace_id=trace_id,
        recommended_action=recommended_action,
        needs_review=needs_review,
        min_confidence=min_confidence,
        limit=limit,
    )
    return {"feedback_analysis": rows}


# 查询检索反馈。
@router.get("/retrieval-feedback")
async def list_retrieval_feedback(
    request: Request,
    kb_id: str = Query(min_length=1),
    enabled: bool | None = None,
    limit: int = Query(default=100, ge=1, le=500),
):
    rows = request.app.state.retrieval_feedback_store.list(
        kb_id=kb_id, enabled=enabled, limit=limit
    )
    return {"retrieval_feedback": rows}


# 禁用检索反馈。
@router.post("/retrieval-feedback/{feedback_id}/disable")
async def disable_retrieval_feedback(feedback_id: str, body: dict, request: Request):
    row = request.app.state.retrieval_feedback_store.set_enabled(
        feedback_id,
        False,
        actor=body.get("actor"),
        reason=body.get("reason"),
    )
    if row is None:
        return JSONResponse(status_code=404, content={"message": "检索反馈不存在"})
    return {"status": "disabled", "retrieval_feedback_id": feedback_id}


# 启用检索反馈。
@router.post("/retrieval-feedback/{feedback_id}/enable")
async def enable_retrieval_feedback(feedback_id: str, request: Request):
    row = request.app.state.retrieval_feedback_store.set_enabled(feedback_id, True)
    if row is None:
        return JSONResponse(status_code=404, content={"message": "检索反馈不存在"})
    return {"status": "enabled", "retrieval_feedback_id": feedback_id}
