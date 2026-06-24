import asyncio
import os

from fastapi import APIRouter, File, Request, Response, UploadFile
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
from service.ingest_service import (
    KBCleanupError,
    delete_kb_index_transactional,
    mark_kb_deleted,
)
from service.kb_locks import kb_write_lock
from service.kb_state import KBState
from tools.manifest import load_index_manifest

router = APIRouter(prefix="/v1", tags=["documents"])


# 创建 create kb 相关逻辑。
def _create_kb(kb_id, registry):
    # 与删库尾部互斥：create 与 delete 都持 kb_write_lock，杜绝"删库已删 registry、未落 tombstone" 之间并发 create 把 lifecycle 切 active、随后旧删库又写 deleted 把新 KB 标删的竞态。
    with kb_write_lock(kb_id):
        return registry.create(kb_id)


# 删除 delete kb 相关逻辑。
def _delete_kb(kb_id, registry, index_jobs):
    # registry 删除与落 tombstone 必须与 create 在同一把锁内原子完成。
    with kb_write_lock(kb_id):
        delete_kb_index_transactional(kb_id)  # 内部同一把锁，可重入
        # 先持久化 deleted，再删 registry。后者失败时 KB 记录仍在、读写被 tombstone 拦住， DELETE 可重试；反过来会出现 registry 已消失但 tombstone 未落、无法重试的半删除态。
        mark_kb_deleted(kb_id)
        registry.delete(kb_id)
    # 释放 executor 槽位，允许 KB 重建时创建新 executor，防止 256 上限耗尽。
    index_jobs.release_executor(kb_id)


_PDF_MAGIC = b"%PDF"
_ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    409: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
}


# 构造错误 error 相关逻辑。
def _error(code: ErrorCode, message: str, status: int) -> JSONResponse:
    return JSONResponse(
        status_code=status, content=build_error_response(code, message).model_dump()
    )


# 处理 public job 相关逻辑。
def _public_job(job: dict) -> IndexJob:
    # committed_generation_id 是崩溃对账证据，只存内部 job record，不进入严格 API schema。
    return IndexJob(**{k: v for k, v in job.items() if k != "committed_generation_id"})


# 处理 kb documents 相关逻辑。
def _kb_documents(kb_id: str) -> list[Document]:
    # generation state 是事务提交指针且内含 documents；manifest 是提交后的派生缓存，写失败时可能滞后。
    active = KBState(kb_id).active()
    documents = (
        active.get("documents", [])
        if active is not None
        else load_index_manifest(kb_id).get("documents", [])
    )
    return [
        Document(name=doc.get("name", ""), sha256=doc.get("sha256", ""))
        for doc in documents
    ]


# 创建 create knowledge base 相关逻辑。
@router.post("/knowledge-bases", status_code=201, responses=_ERROR_RESPONSES)
async def create_knowledge_base(body: KnowledgeBaseCreate, request: Request):
    loop = asyncio.get_running_loop()
    index_jobs = request.app.state.index_jobs
    try:
        record = await loop.run_in_executor(
            request.app.state.offload_executor,
            index_jobs.run_blocking,
            body.kb_id,
            _create_kb,
            body.kb_id,
            request.app.state.kb_registry,
        )
    except KBExistsError:
        return _error(ErrorCode.KB_EXISTS, f"知识库已存在: {body.kb_id}", 409)
    return KnowledgeBase(**record, document_count=0)


# 列出 list knowledge bases 相关逻辑。
@router.get("/knowledge-bases")
async def list_knowledge_bases(request: Request):
    registry = request.app.state.kb_registry
    return [
        KnowledgeBase(**record, document_count=len(_kb_documents(record["kb_id"])))
        for record in registry.list()
    ]


# 获取 get knowledge base 相关逻辑。
@router.get("/knowledge-bases/{kb_id}", responses=_ERROR_RESPONSES)
async def get_knowledge_base(kb_id: str, request: Request):
    record = request.app.state.kb_registry.get(kb_id)
    if record is None:
        return _error(ErrorCode.KB_NOT_FOUND, f"知识库不存在: {kb_id}", 404)
    return KnowledgeBase(**record, document_count=len(_kb_documents(kb_id)))


# 删除 delete knowledge base 相关逻辑。
@router.delete("/knowledge-bases/{kb_id}", status_code=204, responses=_ERROR_RESPONSES)
async def delete_knowledge_base(kb_id: str, request: Request):
    registry = request.app.state.kb_registry
    if not registry.exists(kb_id):
        return _error(ErrorCode.KB_NOT_FOUND, f"知识库不存在: {kb_id}", 404)
    loop = asyncio.get_running_loop()
    index_jobs = request.app.state.index_jobs
    # 排进该 KB 的序列化 executor，等待前序入库任务完成再执行。
    try:
        await loop.run_in_executor(
            request.app.state.offload_executor,
            index_jobs.run_blocking,
            kb_id,
            _delete_kb,
            kb_id,
            registry,
            index_jobs,
        )
    except KBCleanupError:
        # 清理不完整：registry 与 manifest 均保留，返回可重试错误而非误报删除成功。
        return _error(
            ErrorCode.KB_CLEANUP_FAILED, f"知识库清理未完成，请重试: {kb_id}", 500
        )
    return Response(status_code=204)


# 列出 list documents 相关逻辑。
@router.get("/knowledge-bases/{kb_id}/documents", responses=_ERROR_RESPONSES)
async def list_documents(kb_id: str, request: Request):
    if not request.app.state.kb_registry.exists(kb_id):
        return _error(ErrorCode.KB_NOT_FOUND, f"知识库不存在: {kb_id}", 404)
    return _kb_documents(kb_id)


# 上传 upload document 相关逻辑。
@router.post(
    "/knowledge-bases/{kb_id}/documents", status_code=202, responses=_ERROR_RESPONSES
)
async def upload_document(kb_id: str, request: Request, file: UploadFile = File(...)):
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
                ErrorCode.FILE_TOO_LARGE,
                f"文件超过上限 {settings.max_upload_mb}MB",
                413,
            )
    if not content.startswith(_PDF_MAGIC):
        return _error(ErrorCode.INVALID_PDF, "文件不是合法 PDF", 400)

    source_dir = registry.source_dir(kb_id)
    # submit_upload 含同步 SQLite 写：放线程池执行，绝不阻塞事件循环（否则 SQLite 锁竞争会冻结整个 API）。
    loop = asyncio.get_running_loop()
    job = await loop.run_in_executor(
        request.app.state.offload_executor,
        request.app.state.index_jobs.submit_upload,
        kb_id,
        source_dir,
        filename,
        bytes(content),
    )
    return _public_job(job)


# 删除 delete document 相关逻辑。
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
    # 同步 SQLite 写下放线程池，不阻塞事件循环；存在性检查仍在 executor command 内完成，路由始终 202。
    loop = asyncio.get_running_loop()
    job = await loop.run_in_executor(
        request.app.state.offload_executor,
        request.app.state.index_jobs.submit_delete_doc,
        kb_id,
        path,
    )
    return _public_job(job)


# 获取 get index job 相关逻辑。
@router.get("/index-jobs/{job_id}", responses=_ERROR_RESPONSES)
async def get_index_job(job_id: str, request: Request):
    job = request.app.state.index_jobs.get(job_id)
    if job is None:
        return _error(ErrorCode.JOB_NOT_FOUND, f"任务不存在: {job_id}", 404)
    return _public_job(job)
