import os

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse

from api.ingest import KBExistsError
from api.schemas import (
    Document,
    ErrorCode,
    ErrorResponse,
    IndexJob,
    KnowledgeBase,
    KnowledgeBaseCreate,
    build_error_response,
)
from config.settings import get_settings
from tools.manifest import load_index_manifest

router = APIRouter(prefix="/v1", tags=["documents"])

_PDF_MAGIC = b"%PDF"
_ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    409: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
}


def _error(code: ErrorCode, message: str, status: int) -> JSONResponse:
    return JSONResponse(
        status_code=status, content=build_error_response(code, message).model_dump()
    )


def _kb_documents(kb_id: str) -> list[Document]:
    manifest = load_index_manifest(kb_id)
    return [
        Document(name=doc.get("name", ""), sha256=doc.get("sha256", ""))
        for doc in manifest.get("documents", [])
    ]


@router.post("/knowledge-bases", status_code=201, responses=_ERROR_RESPONSES)
async def create_knowledge_base(body: KnowledgeBaseCreate, request: Request):
    try:
        record = request.app.state.kb_registry.create(body.kb_id)
    except KBExistsError:
        return _error(ErrorCode.KB_EXISTS, f"知识库已存在: {body.kb_id}", 409)
    return KnowledgeBase(**record, document_count=0)


@router.get("/knowledge-bases")
async def list_knowledge_bases(request: Request):
    registry = request.app.state.kb_registry
    return [
        KnowledgeBase(**record, document_count=len(_kb_documents(record["kb_id"])))
        for record in registry.list()
    ]


@router.get("/knowledge-bases/{kb_id}", responses=_ERROR_RESPONSES)
async def get_knowledge_base(kb_id: str, request: Request):
    record = request.app.state.kb_registry.get(kb_id)
    if record is None:
        return _error(ErrorCode.KB_NOT_FOUND, f"知识库不存在: {kb_id}", 404)
    return KnowledgeBase(**record, document_count=len(_kb_documents(kb_id)))


@router.get("/knowledge-bases/{kb_id}/documents", responses=_ERROR_RESPONSES)
async def list_documents(kb_id: str, request: Request):
    if not request.app.state.kb_registry.exists(kb_id):
        return _error(ErrorCode.KB_NOT_FOUND, f"知识库不存在: {kb_id}", 404)
    return _kb_documents(kb_id)


@router.post(
    "/knowledge-bases/{kb_id}/documents", status_code=202, responses=_ERROR_RESPONSES
)
async def upload_document(
    kb_id: str, request: Request, file: UploadFile = File(...)
):
    registry = request.app.state.kb_registry
    if not registry.exists(kb_id):
        return _error(ErrorCode.KB_NOT_FOUND, f"知识库不存在: {kb_id}", 404)

    filename = os.path.basename(file.filename or "")
    if not filename.lower().endswith(".pdf"):
        return _error(ErrorCode.INVALID_PDF, "只接受 .pdf 文件", 400)

    settings = get_settings()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    # 分块读并即时熔断，内存占用以上限为界，不被客户端声明的大小拖垮。
    content = bytearray()
    while True:
        block = await file.read(1024 * 1024)
        if not block:
            break
        content.extend(block)
        if len(content) > max_bytes:
            return _error(
                ErrorCode.FILE_TOO_LARGE, f"文件超过上限 {settings.max_upload_mb}MB", 413
            )
    if not content.startswith(_PDF_MAGIC):
        return _error(ErrorCode.INVALID_PDF, "文件不是合法 PDF", 400)

    source_dir = registry.source_dir(kb_id)
    os.makedirs(source_dir, exist_ok=True)
    with open(os.path.join(source_dir, filename), "wb") as fp:
        fp.write(content)

    job = request.app.state.index_jobs.submit(kb_id)
    return IndexJob(**job)


@router.delete(
    "/knowledge-bases/{kb_id}/documents/{name}",
    status_code=202,
    responses=_ERROR_RESPONSES,
)
async def delete_document(kb_id: str, name: str, request: Request):
    registry = request.app.state.kb_registry
    if not registry.exists(kb_id):
        return _error(ErrorCode.KB_NOT_FOUND, f"知识库不存在: {kb_id}", 404)

    safe_name = os.path.basename(name)
    path = os.path.join(registry.source_dir(kb_id), safe_name)
    if not os.path.exists(path):
        return _error(ErrorCode.DOCUMENT_NOT_FOUND, f"文档不存在: {safe_name}", 404)

    os.remove(path)
    job = request.app.state.index_jobs.submit(kb_id)
    return IndexJob(**job)


@router.get("/index-jobs/{job_id}", responses=_ERROR_RESPONSES)
async def get_index_job(job_id: str, request: Request):
    job = request.app.state.index_jobs.get(job_id)
    if job is None:
        return _error(ErrorCode.JOB_NOT_FOUND, f"任务不存在: {job_id}", 404)
    return IndexJob(**job)
