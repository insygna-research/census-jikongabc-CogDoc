from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, Response

from cogdoc.api.offload import run_sync
from cogdoc.api.research_job_store import (
    ResearchJobRevisionConflictError,
    ResearchJobStateConflictError,
    ResearchJobStore,
)
from cogdoc.api.schemas import (
    ErrorCode,
    ResearchJob,
    ResearchJobCreate,
    ResearchJobListResponse,
    ResearchJobResponse,
    ResearchPlanUpdate,
    ResearchReportPublishRequest,
    ResearchReportReviewUpdate,
    build_error_response,
)


router = APIRouter(prefix="/v1/research-jobs", tags=["research"])


def _error(code: ErrorCode, message: str, status: int) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=build_error_response(code, message).model_dump(),
    )


def _store(request: Request) -> ResearchJobStore | None:
    return getattr(request.app.state, "research_job_store", None)


def _manager(request: Request):
    return getattr(request.app.state, "research_execution_manager", None)


@router.post("", status_code=201, response_model=ResearchJobResponse)
async def create_research_job(body: ResearchJobCreate, request: Request):
    if not request.app.state.kb_registry.exists(body.kb_id):
        return _error(
            ErrorCode.KB_NOT_FOUND,
            f"知识库不存在: {body.kb_id}",
            404,
        )
    store = _store(request)
    if store is None:
        return _error(ErrorCode.INTERNAL_ERROR, "研究任务存储不可用", 503)
    row = await run_sync(
        request.app.state.offload_executor,
        store.create,
        kb_id=body.kb_id,
        objective=body.objective,
        title=body.title,
        section_titles=body.section_titles,
    )
    return ResearchJobResponse(job=ResearchJob.model_validate(row))


@router.get("", response_model=ResearchJobListResponse)
async def list_research_jobs(
    request: Request,
    kb_id: str | None = None,
    status: Literal[
        "planned",
        "running",
        "paused",
        "evidence_ready",
        "generating",
        "completed",
        "failed",
        "cancelled",
    ]
    | None = None,
    limit: int = Query(default=100, ge=1, le=500),
):
    store = _store(request)
    if store is None:
        return _error(ErrorCode.INTERNAL_ERROR, "研究任务存储不可用", 503)
    rows = await run_sync(
        request.app.state.offload_executor,
        store.list,
        kb_id=kb_id,
        status=status,
        limit=limit,
    )
    return ResearchJobListResponse(
        jobs=[ResearchJob.model_validate(row) for row in rows]
    )


@router.get("/{job_id}", response_model=ResearchJobResponse)
async def get_research_job(job_id: str, request: Request):
    store = _store(request)
    if store is None:
        return _error(ErrorCode.INTERNAL_ERROR, "研究任务存储不可用", 503)
    row = await run_sync(request.app.state.offload_executor, store.get, job_id)
    if row is None:
        return _error(
            ErrorCode.RESEARCH_JOB_NOT_FOUND,
            f"研究任务不存在: {job_id}",
            404,
        )
    return ResearchJobResponse(job=ResearchJob.model_validate(row))


async def _execution_action(job_id: str, request: Request, action: str):
    manager = _manager(request)
    if manager is None:
        return _error(ErrorCode.INTERNAL_ERROR, "研究执行器不可用", 503)
    operation = getattr(manager, action)
    try:
        row = await run_sync(
            request.app.state.offload_executor,
            operation,
            job_id,
        )
    except KeyError:
        return _error(
            ErrorCode.RESEARCH_JOB_NOT_FOUND,
            f"研究任务不存在: {job_id}",
            404,
        )
    except ResearchJobStateConflictError as exc:
        return _error(ErrorCode.RESEARCH_JOB_STATE_CONFLICT, str(exc), 409)
    return ResearchJobResponse(job=ResearchJob.model_validate(row))


@router.post("/{job_id}/start", status_code=202, response_model=ResearchJobResponse)
async def start_research_job(job_id: str, request: Request):
    return await _execution_action(job_id, request, "start")


@router.post("/{job_id}/resume", status_code=202, response_model=ResearchJobResponse)
async def resume_research_job(job_id: str, request: Request):
    return await _execution_action(job_id, request, "resume")


@router.post("/{job_id}/pause", response_model=ResearchJobResponse)
async def pause_research_job(job_id: str, request: Request):
    return await _execution_action(job_id, request, "pause")


@router.post("/{job_id}/cancel", response_model=ResearchJobResponse)
async def cancel_research_job(job_id: str, request: Request):
    return await _execution_action(job_id, request, "cancel")


@router.post(
    "/{job_id}/generate",
    status_code=202,
    response_model=ResearchJobResponse,
)
async def generate_research_report(job_id: str, request: Request):
    return await _execution_action(job_id, request, "compile")


@router.get("/{job_id}/report")
async def download_research_report(job_id: str, request: Request):
    store = _store(request)
    if store is None:
        return _error(ErrorCode.INTERNAL_ERROR, "研究任务存储不可用", 503)
    row = await run_sync(request.app.state.offload_executor, store.get, job_id)
    if row is None:
        return _error(
            ErrorCode.RESEARCH_JOB_NOT_FOUND,
            f"研究任务不存在: {job_id}",
            404,
        )
    report = row.get("report")
    if not isinstance(report, dict) or not str(report.get("content") or ""):
        return _error(
            ErrorCode.RESEARCH_JOB_STATE_CONFLICT,
            "研究报告尚未生成",
            409,
        )
    return Response(
        content=str(report["content"]),
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{job_id}.md"'
        },
    )


@router.put("/{job_id}/review", response_model=ResearchJobResponse)
async def review_research_report(
    job_id: str,
    body: ResearchReportReviewUpdate,
    request: Request,
):
    store = _store(request)
    if store is None:
        return _error(ErrorCode.INTERNAL_ERROR, "研究任务存储不可用", 503)
    try:
        row = await run_sync(
            request.app.state.offload_executor,
            store.review_report,
            job_id,
            decisions=[decision.model_dump() for decision in body.decisions],
            expected_revision=body.expected_revision,
        )
    except KeyError:
        return _error(
            ErrorCode.RESEARCH_JOB_NOT_FOUND,
            f"研究任务不存在: {job_id}",
            404,
        )
    except ResearchJobRevisionConflictError as exc:
        return _error(ErrorCode.RESEARCH_JOB_REVISION_CONFLICT, str(exc), 409)
    except ResearchJobStateConflictError as exc:
        return _error(ErrorCode.RESEARCH_JOB_STATE_CONFLICT, str(exc), 409)
    except ValueError as exc:
        return _error(ErrorCode.BAD_REQUEST, str(exc), 422)
    return ResearchJobResponse(job=ResearchJob.model_validate(row))


@router.post("/{job_id}/publish", response_model=ResearchJobResponse)
async def publish_research_report(
    job_id: str,
    body: ResearchReportPublishRequest,
    request: Request,
):
    store = _store(request)
    if store is None:
        return _error(ErrorCode.INTERNAL_ERROR, "研究任务存储不可用", 503)
    try:
        row = await run_sync(
            request.app.state.offload_executor,
            store.publish_report,
            job_id,
            expected_revision=body.expected_revision,
        )
    except KeyError:
        return _error(
            ErrorCode.RESEARCH_JOB_NOT_FOUND,
            f"研究任务不存在: {job_id}",
            404,
        )
    except ResearchJobRevisionConflictError as exc:
        return _error(ErrorCode.RESEARCH_JOB_REVISION_CONFLICT, str(exc), 409)
    except ResearchJobStateConflictError as exc:
        return _error(ErrorCode.RESEARCH_JOB_STATE_CONFLICT, str(exc), 409)
    return ResearchJobResponse(job=ResearchJob.model_validate(row))


@router.get("/{job_id}/published-report")
async def download_published_research_report(job_id: str, request: Request):
    store = _store(request)
    if store is None:
        return _error(ErrorCode.INTERNAL_ERROR, "研究任务存储不可用", 503)
    row = await run_sync(request.app.state.offload_executor, store.get, job_id)
    if row is None:
        return _error(
            ErrorCode.RESEARCH_JOB_NOT_FOUND,
            f"研究任务不存在: {job_id}",
            404,
        )
    report = row.get("published_report")
    if not isinstance(report, dict) or not str(report.get("content") or ""):
        return _error(
            ErrorCode.RESEARCH_JOB_STATE_CONFLICT,
            "研究报告尚未发布",
            409,
        )
    return Response(
        content=str(report["content"]),
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{job_id}-published.md"'
        },
    )


@router.put("/{job_id}/plan", response_model=ResearchJobResponse)
async def update_research_plan(
    job_id: str,
    body: ResearchPlanUpdate,
    request: Request,
):
    store = _store(request)
    if store is None:
        return _error(ErrorCode.INTERNAL_ERROR, "研究任务存储不可用", 503)
    try:
        row = await run_sync(
            request.app.state.offload_executor,
            store.update_plan,
            job_id,
            sections=[section.model_dump() for section in body.sections],
            expected_revision=body.expected_revision,
        )
    except KeyError:
        return _error(
            ErrorCode.RESEARCH_JOB_NOT_FOUND,
            f"研究任务不存在: {job_id}",
            404,
        )
    except ResearchJobRevisionConflictError as exc:
        return _error(ErrorCode.RESEARCH_JOB_REVISION_CONFLICT, str(exc), 409)
    except ResearchJobStateConflictError as exc:
        return _error(ErrorCode.RESEARCH_JOB_STATE_CONFLICT, str(exc), 409)
    except ValueError as exc:
        return _error(ErrorCode.BAD_REQUEST, str(exc), 422)
    return ResearchJobResponse(job=ResearchJob.model_validate(row))
