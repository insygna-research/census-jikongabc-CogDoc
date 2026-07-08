from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from cogdoc.api.schemas import (
    DerivedKnowledge,
    ErrorCode,
    ErrorResponse,
    FeedbackLoopMetricsResponse,
    KnowledgeBatchReviewRequest,
    KnowledgeBatchReviewResponse,
    KnowledgeConflictCandidate,
    KnowledgeCreateRequest,
    KnowledgeCreateResponse,
    KnowledgeListResponse,
    KnowledgeOrigin,
    KnowledgePendingCountResponse,
    KnowledgeReviewRequest,
    KnowledgeReviseRequest,
    KnowledgeStatus,
    ReviewQueueExportResponse,
    ReviewQueueSummaryResponse,
    build_error_response,
)
from cogdoc.api.time_utils import now_iso
from cogdoc.api.webhooks import notify_pending_created

router = APIRouter(prefix="/v1", tags=["knowledge"])

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
}


# 计算比率。
def _rate(numerator: int, denominator: int | None) -> float | None:
    if denominator is None or denominator <= 0:
        return None
    return round(numerator / denominator, 4)


# 完成 错误响应 处理。
def _error(code: ErrorCode, message: str, status: int) -> JSONResponse:
    return JSONResponse(
        status_code=status, content=build_error_response(code, message).model_dump()
    )


# 构建公开知识视图。
def _public(row: dict) -> DerivedKnowledge:
    return DerivedKnowledge(**row)


# 构建冲突候选视图。
def _conflict_public(row: dict) -> KnowledgeConflictCandidate:
    return KnowledgeConflictCandidate(
        knowledge_id=row["knowledge_id"],
        text=row["text"],
        status=row["status"],
        origin=row.get("origin") or "manual_entry",
        related_source=row.get("related_source"),
        created_at=row["created_at"],
    )


# 构建审核队列摘要。
def _build_review_queue_summary(
    request: Request,
    *,
    kb_id: str,
    document_id: str | None = None,
    origin: KnowledgeOrigin | None = None,
    created_by: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
) -> ReviewQueueSummaryResponse:
    origin_value = origin.value if origin is not None else None
    knowledge = request.app.state.knowledge_store.counts(
        kb_id=kb_id,
        document_id=document_id,
        origin=origin_value,
        created_by=created_by,
        created_after=created_after,
        created_before=created_before,
    )
    knowledge_conflicts = request.app.state.knowledge_store.conflict_counts(
        kb_id=kb_id,
        document_id=document_id,
        origin=origin_value,
        created_by=created_by,
        created_after=created_after,
        created_before=created_before,
    )
    auto_review = request.app.state.knowledge_store.auto_review_counts(
        kb_id=kb_id,
        document_id=document_id,
        origin=origin_value,
        created_by=created_by,
        created_after=created_after,
        created_before=created_before,
    )
    feedback_rows = request.app.state.feedback_store.counts(kb_id=kb_id)
    feedback = request.app.state.feedback_analysis_store.counts(kb_id=kb_id)
    retrieval = request.app.state.retrieval_feedback_store.counts(kb_id=kb_id)
    return ReviewQueueSummaryResponse(
        kb_id=kb_id,
        knowledge=knowledge["by_status"],
        knowledge_origin=knowledge["by_origin"],
        knowledge_conflicts=knowledge_conflicts,
        knowledge_auto_review={
            **auto_review,
            "stale_pending": int(knowledge["by_status"].get("stale", 0)),
        },
        feedback_counts=feedback_rows,
        feedback_analysis={
            **feedback["by_action"],
            "needs_review": feedback["needs_review"],
            "total": feedback["total"],
        },
        feedback_analysis_type=feedback["by_type"],
        retrieval_feedback=retrieval,
    )


# 新增派生知识。
@router.post("/knowledge", status_code=201, responses=_ERROR_RESPONSES)
async def create_knowledge(body: KnowledgeCreateRequest, request: Request):
    payload = body.model_dump(exclude_none=True)
    payload["status"] = (
        KnowledgeStatus.APPROVED.value
        if body.enable_immediately
        else KnowledgeStatus.PENDING.value
    )
    try:
        row, deduplicated = request.app.state.knowledge_store.create(payload)
    except ValueError as exc:
        return _error(ErrorCode.BAD_REQUEST, str(exc), 400)
    conflicts = (
        [] if deduplicated else request.app.state.knowledge_store.conflicts_for(row)
    )
    if not deduplicated:
        notify_pending_created(request.app, row, "knowledge_create")
    return KnowledgeCreateResponse(
        knowledge=_public(row),
        deduplicated=deduplicated,
        requires_review=bool(conflicts),
        conflicts=[_conflict_public(item) for item in conflicts],
    )


# 查询派生知识。
@router.get("/knowledge", response_model=KnowledgeListResponse)
async def list_knowledge(
    request: Request,
    kb_id: str = Query(min_length=1),
    status: KnowledgeStatus | None = None,
    document_id: str | None = None,
    origin: KnowledgeOrigin | None = None,
    created_by: str | None = None,
    conflict_group_id: str | None = None,
    has_conflict: bool | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
):
    rows = request.app.state.knowledge_store.list(
        kb_id=kb_id,
        status=status.value if status is not None else None,
        document_id=document_id,
        origin=origin.value if origin is not None else None,
        created_by=created_by,
        conflict_group_id=conflict_group_id,
        has_conflict=has_conflict,
        created_after=created_after,
        created_before=created_before,
    )
    return KnowledgeListResponse(knowledge=[_public(row) for row in rows])


# 查询待审核计数。
@router.get("/knowledge/pending-count", response_model=KnowledgePendingCountResponse)
async def pending_knowledge_count(
    request: Request,
    kb_id: str = Query(min_length=1),
):
    knowledge = request.app.state.knowledge_store.counts(kb_id=kb_id)
    analysis = request.app.state.feedback_analysis_store.counts(kb_id=kb_id)
    by_status = knowledge["by_status"]
    pending = int(by_status.get(KnowledgeStatus.PENDING.value, 0))
    stale = int(by_status.get(KnowledgeStatus.STALE.value, 0))
    needs_review = int(analysis["needs_review"])
    return KnowledgePendingCountResponse(
        kb_id=kb_id,
        pending=pending,
        stale=stale,
        feedback_analysis_needs_review=needs_review,
        total=pending + stale + needs_review,
    )


# 查询反馈闭环指标。
@router.get("/feedback-loop-metrics", response_model=FeedbackLoopMetricsResponse)
async def feedback_loop_metrics(
    request: Request,
    kb_id: str = Query(min_length=1),
    answer_count: int | None = Query(default=None, ge=0),
):
    feedback = request.app.state.feedback_store.counts(kb_id=kb_id)
    knowledge = request.app.state.knowledge_store.counts(kb_id=kb_id)
    stale_review = request.app.state.knowledge_store.stale_review_counts(kb_id=kb_id)
    analysis = request.app.state.feedback_analysis_store.counts(kb_id=kb_id)
    retrieval = request.app.state.retrieval_feedback_store.counts(kb_id=kb_id)
    by_feedback = feedback["by_feedback"]
    by_type = feedback["by_type"]
    by_status = knowledge["by_status"]
    by_action = analysis["by_action"]
    feedback_total = int(feedback["total"])
    negative_total = int(feedback["bad_cases"])
    correction_total = int(by_feedback.get("correction", 0))
    no_evidence_total = int(by_type.get("no_evidence", 0))
    knowledge_total = int(knowledge["total"])
    approved_total = int(by_status.get(KnowledgeStatus.APPROVED.value, 0))
    rejected_total = int(by_status.get(KnowledgeStatus.REJECTED.value, 0))
    pending_created = int(by_action.get("create_pending_knowledge", 0))
    retrieval_total = int(retrieval["total"])
    retrieval_disabled = int(retrieval["disabled"])
    stale_total = int(stale_review["total"])
    stale_reviewed = int(stale_review["reviewed"])
    return FeedbackLoopMetricsResponse(
        kb_id=kb_id,
        counts={
            "answer_total": answer_count or 0,
            "feedback_total": feedback_total,
            "negative_feedback_total": negative_total,
            "no_evidence_feedback_total": no_evidence_total,
            "correction_feedback_total": correction_total,
            "knowledge_total": knowledge_total,
            "approved_knowledge_total": approved_total,
            "rejected_knowledge_total": rejected_total,
            "pending_created_total": pending_created,
            "retrieval_feedback_total": retrieval_total,
            "retrieval_feedback_disabled": retrieval_disabled,
            "stale_knowledge_total": stale_total,
            "stale_knowledge_reviewed": stale_reviewed,
        },
        rates={
            "feedback_rate": _rate(feedback_total, answer_count),
            "negative_feedback_rate": _rate(negative_total, answer_count),
            "no_evidence_rate": _rate(no_evidence_total, answer_count),
            "pending_approval_rate": _rate(approved_total, knowledge_total),
            "pending_rejection_rate": _rate(rejected_total, knowledge_total),
            "feedback_to_pending_rate": _rate(pending_created, correction_total),
            "retrieval_feedback_rollback_rate": _rate(
                retrieval_disabled, retrieval_total
            ),
            "stale_review_completion_rate": _rate(stale_reviewed, stale_total),
        },
    )


# 查询审核队列摘要。
@router.get("/review-queue", response_model=ReviewQueueSummaryResponse)
async def review_queue_summary(
    request: Request,
    kb_id: str = Query(min_length=1),
    document_id: str | None = None,
    origin: KnowledgeOrigin | None = None,
    created_by: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
):
    return _build_review_queue_summary(
        request,
        kb_id=kb_id,
        document_id=document_id,
        origin=origin,
        created_by=created_by,
        created_after=created_after,
        created_before=created_before,
    )


# 导出当前审核队列。
@router.get("/review-queue/export", response_model=ReviewQueueExportResponse)
async def review_queue_export(
    request: Request,
    kb_id: str = Query(min_length=1),
    knowledge_document_id: str | None = None,
    knowledge_origin: KnowledgeOrigin | None = None,
    knowledge_created_by: str | None = None,
    knowledge_created_after: str | None = None,
    knowledge_created_before: str | None = None,
    limit: int = Query(default=200, ge=1, le=1000),
):
    origin_value = knowledge_origin.value if knowledge_origin is not None else None
    summary = _build_review_queue_summary(
        request,
        kb_id=kb_id,
        document_id=knowledge_document_id,
        origin=knowledge_origin,
        created_by=knowledge_created_by,
        created_after=knowledge_created_after,
        created_before=knowledge_created_before,
    )
    pending = request.app.state.knowledge_store.list(
        kb_id=kb_id,
        status=KnowledgeStatus.PENDING.value,
        document_id=knowledge_document_id,
        origin=origin_value,
        created_by=knowledge_created_by,
        created_after=knowledge_created_after,
        created_before=knowledge_created_before,
        limit=limit,
    )
    stale = request.app.state.knowledge_store.list(
        kb_id=kb_id,
        status=KnowledgeStatus.STALE.value,
        document_id=knowledge_document_id,
        origin=origin_value,
        created_by=knowledge_created_by,
        created_after=knowledge_created_after,
        created_before=knowledge_created_before,
        limit=limit,
    )
    analysis = request.app.state.feedback_analysis_store.list(
        kb_id=kb_id,
        needs_review=True,
        limit=limit,
    )
    retrieval = request.app.state.retrieval_feedback_store.list(
        kb_id=kb_id,
        enabled=True,
        limit=limit,
    )
    feedback = request.app.state.feedback_store.list(
        kb_id=kb_id,
        is_bad_case=True,
        limit=limit,
    )
    auto_review_events = request.app.state.knowledge_store.auto_review_events(
        kb_id=kb_id,
        document_id=knowledge_document_id,
        origin=origin_value,
        created_by=knowledge_created_by,
        created_after=knowledge_created_after,
        created_before=knowledge_created_before,
        limit=limit,
    )
    return ReviewQueueExportResponse(
        kb_id=kb_id,
        generated_at=now_iso(),
        summary=summary,
        pending_knowledge=[_public(row) for row in pending],
        stale_knowledge=[_public(row) for row in stale],
        auto_review_events=auto_review_events,
        feedback_analysis_needs_review=analysis,
        retrieval_feedback_enabled=retrieval,
        feedback_bad_cases=feedback,
    )


# 审核状态流转。
def _set_status(request: Request, knowledge_id: str, status: str, body):
    binding_updates = {
        key: value
        for key, value in {
            "related_document_id": body.related_document_id,
            "related_source": body.related_source,
            "related_source_sha256": body.related_source_sha256,
            "related_chunk_ids": body.related_chunk_ids,
            "related_page_start": body.related_page_start,
            "related_page_end": body.related_page_end,
            "related_chunk_text_hash": body.related_chunk_text_hash,
            "related_anchor_text": body.related_anchor_text,
        }.items()
        if value is not None
    }
    try:
        row = request.app.state.knowledge_store.set_status(
            knowledge_id,
            status,
            actor=body.actor,
            note=body.note,
            binding_updates=binding_updates,
        )
    except ValueError as exc:
        return _error(ErrorCode.BAD_REQUEST, str(exc), 400)
    if row is None:
        return _error(ErrorCode.KNOWLEDGE_NOT_FOUND, f"知识不存在: {knowledge_id}", 404)
    return _public(row)


# 审核通过知识。
@router.post(
    "/knowledge/{knowledge_id}/approve",
    response_model=DerivedKnowledge,
    responses=_ERROR_RESPONSES,
)
async def approve_knowledge(
    knowledge_id: str, body: KnowledgeReviewRequest, request: Request
):
    return _set_status(request, knowledge_id, KnowledgeStatus.APPROVED.value, body)


# 驳回知识。
@router.post(
    "/knowledge/{knowledge_id}/reject",
    response_model=DerivedKnowledge,
    responses=_ERROR_RESPONSES,
)
async def reject_knowledge(
    knowledge_id: str, body: KnowledgeReviewRequest, request: Request
):
    return _set_status(request, knowledge_id, KnowledgeStatus.REJECTED.value, body)


# 归档知识。
@router.post(
    "/knowledge/{knowledge_id}/archive",
    response_model=DerivedKnowledge,
    responses=_ERROR_RESPONSES,
)
async def archive_knowledge(
    knowledge_id: str, body: KnowledgeReviewRequest, request: Request
):
    return _set_status(request, knowledge_id, KnowledgeStatus.ARCHIVED.value, body)


# 创建知识修订版本。
@router.post(
    "/knowledge/{knowledge_id}/revise",
    status_code=201,
    response_model=KnowledgeCreateResponse,
    responses=_ERROR_RESPONSES,
)
async def revise_knowledge(
    knowledge_id: str, body: KnowledgeReviseRequest, request: Request
):
    payload = body.model_dump(exclude_none=True)
    payload["status"] = (
        KnowledgeStatus.APPROVED.value
        if body.enable_immediately
        else KnowledgeStatus.PENDING.value
    )
    try:
        row = request.app.state.knowledge_store.revise(knowledge_id, payload)
    except ValueError as exc:
        return _error(ErrorCode.BAD_REQUEST, str(exc), 400)
    if row is None:
        return _error(ErrorCode.KNOWLEDGE_NOT_FOUND, f"知识不存在: {knowledge_id}", 404)
    notify_pending_created(request.app, row, "knowledge_revision")
    return KnowledgeCreateResponse(knowledge=_public(row), deduplicated=False)


# 批量审核。
def _batch_set_status(request: Request, body: KnowledgeBatchReviewRequest, status: str):
    try:
        updated, missing = request.app.state.knowledge_store.batch_set_status(
            body.knowledge_ids, status, actor=body.actor, note=body.note
        )
    except ValueError as exc:
        return _error(ErrorCode.BAD_REQUEST, str(exc), 400)
    return KnowledgeBatchReviewResponse(
        updated=[_public(row) for row in updated], missing_ids=missing
    )


# 批量审核通过。
@router.post(
    "/knowledge/batch-approve",
    response_model=KnowledgeBatchReviewResponse,
    responses=_ERROR_RESPONSES,
)
async def batch_approve_knowledge(body: KnowledgeBatchReviewRequest, request: Request):
    return _batch_set_status(request, body, KnowledgeStatus.APPROVED.value)


# 批量驳回。
@router.post(
    "/knowledge/batch-reject",
    response_model=KnowledgeBatchReviewResponse,
    responses=_ERROR_RESPONSES,
)
async def batch_reject_knowledge(body: KnowledgeBatchReviewRequest, request: Request):
    return _batch_set_status(request, body, KnowledgeStatus.REJECTED.value)
