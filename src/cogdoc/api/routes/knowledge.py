from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from cogdoc.api.schemas import (
    DerivedKnowledge,
    ErrorCode,
    ErrorResponse,
    KnowledgeBatchReviewRequest,
    KnowledgeBatchReviewResponse,
    KnowledgeConflictCandidate,
    KnowledgeCreateRequest,
    KnowledgeCreateResponse,
    KnowledgeListResponse,
    KnowledgeOrigin,
    KnowledgeReviewRequest,
    KnowledgeReviseRequest,
    KnowledgeStatus,
    ReviewQueueSummaryResponse,
    build_error_response,
)

router = APIRouter(prefix="/v1", tags=["knowledge"])

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
}


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
    created_after: str | None = None,
    created_before: str | None = None,
):
    rows = request.app.state.knowledge_store.list(
        kb_id=kb_id,
        status=status.value if status is not None else None,
        document_id=document_id,
        origin=origin.value if origin is not None else None,
        created_by=created_by,
        created_after=created_after,
        created_before=created_before,
    )
    return KnowledgeListResponse(knowledge=[_public(row) for row in rows])


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
    knowledge = request.app.state.knowledge_store.counts(
        kb_id=kb_id,
        document_id=document_id,
        origin=origin.value if origin is not None else None,
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
        feedback_counts=feedback_rows,
        feedback_analysis={
            **feedback["by_action"],
            "needs_review": feedback["needs_review"],
            "total": feedback["total"],
        },
        feedback_analysis_type=feedback["by_type"],
        retrieval_feedback=retrieval,
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
