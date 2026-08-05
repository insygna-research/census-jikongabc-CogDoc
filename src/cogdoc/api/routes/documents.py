import os
from fastapi import APIRouter, File, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from cogdoc.api.ingest import KBExistsError
from cogdoc.api.offload import run_sync
from cogdoc.api.schemas import (
    Document,
    ErrorCode,
    ErrorResponse,
    IndexJob,
    KnowledgeBase,
    KnowledgeBaseCreate,
    SourceChunksResponse,
    SourceListResponse,
    build_error_response,
)
from cogdoc.config.settings import get_settings
from cogdoc.observability.trace import delete_trace_files
from cogdoc.service.ingest_service import (
    KBCleanupError,
    delete_kb_index_transactional,
    mark_kb_deleted,
)
from cogdoc.service.source_chunks import (
    chunk_preview,
    source_chunks as read_source_chunks,
)
from cogdoc.service.kb_locks import kb_write_lock
from cogdoc.service.kb_state import KBState
from cogdoc.tools.manifest import load_index_manifest

router = APIRouter(prefix="/v1", tags=["documents"])


# 创建 kb。
def _create_kb(kb_id, registry):
    # 与删库尾部互斥：create 与 delete 都持 kb_write_lock，杜绝"删库已删 registry、未落 tombstone" 之间并发 create 把 lifecycle 切 active、随后旧删库又写 deleted 把新 KB 标删的竞态。
    with kb_write_lock(kb_id):
        return registry.create(kb_id)


# 删除 kb。
def _clear_kb_review_state(kb_id, stores) -> None:
    for store in stores:
        clear_kb = getattr(store, "clear_kb", None)
        if clear_kb is not None:
            clear_kb(kb_id)


# 删除 kb。
def _delete_kb(
    kb_id,
    registry,
    index_jobs,
    session_store=None,
    knowledge_store=None,
    feedback_store=None,
    feedback_analysis_store=None,
    retrieval_feedback_store=None,
    retrieval_eval_draft_store=None,
):
    # registry 删除与落 tombstone 必须与 create 在同一把锁内原子完成。
    try:
        with kb_write_lock(kb_id):
            delete_kb_index_transactional(kb_id)  # 内部同一把锁，可重入
            # 先持久化 deleted，再删 registry。后者失败时 KB 记录仍在、读写被 tombstone 拦住， DELETE 可重试；反过来会出现 registry 已消失但 tombstone 未落、无法重试的半删除态。
            mark_kb_deleted(kb_id)
            try:
                _clear_kb_review_state(
                    kb_id,
                    (
                        knowledge_store,
                        feedback_store,
                        feedback_analysis_store,
                        retrieval_feedback_store,
                        retrieval_eval_draft_store,
                    ),
                )
            except Exception as exc:
                raise KBCleanupError(f"KB 派生/反馈状态删除失败: {kb_id}") from exc
            # 连带清掉该库的会话历史，否则同名新库复用 kb_id 会捡到旧对话。
            try:
                if session_store is not None:
                    session_store.clear_kb(kb_id)
            except Exception as exc:
                # registry 保留到所有幂等状态清理完成后再删，失败时 DELETE 可重试。
                raise KBCleanupError(f"KB 会话状态删除失败: {kb_id}") from exc
            registry.delete(kb_id)
    finally:
        try:
            delete_trace_files(doc_id=kb_id)
        finally:
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


# 完成 错误 处理。
def _error(code: ErrorCode, message: str, status: int) -> JSONResponse:
    return JSONResponse(
        status_code=status, content=build_error_response(code, message).model_dump()
    )


# 完成 公开视图任务 处理。
def _public_job(job: dict) -> IndexJob:
    # committed_generation_id 是崩溃对账证据，只存内部 job record，不进入严格 API schema。
    return IndexJob(**{k: v for k, v in job.items() if k != "committed_generation_id"})


# 完成 知识库documents 处理。
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


# 读取知识库来源文件列表。
def _kb_sources(kb_id: str) -> list[str]:
    from cogdoc.service.retriever_factory import RetrieverFactory
    from cogdoc.service.kb_readers import kb_read_lease

    with kb_read_lease(kb_id):
        return RetrieverFactory.get_engine(kb_id).list_sources()


# 创建 knowledge base。
@router.post("/knowledge-bases", status_code=201, responses=_ERROR_RESPONSES)
async def create_knowledge_base(body: KnowledgeBaseCreate, request: Request):
    index_jobs = request.app.state.index_jobs
    try:
        record = await run_sync(
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


# 列出 knowledge bases。
@router.get("/knowledge-bases")
async def list_knowledge_bases(request: Request):
    registry = request.app.state.kb_registry
    return [
        KnowledgeBase(**record, document_count=len(_kb_documents(record["kb_id"])))
        for record in registry.list()
    ]


# 返回knowledgebase。
@router.get("/knowledge-bases/{kb_id}", responses=_ERROR_RESPONSES)
async def get_knowledge_base(kb_id: str, request: Request):
    record = request.app.state.kb_registry.get(kb_id)
    if record is None:
        return _error(ErrorCode.KB_NOT_FOUND, f"知识库不存在: {kb_id}", 404)
    return KnowledgeBase(**record, document_count=len(_kb_documents(kb_id)))


# 删除 knowledge base。
@router.delete("/knowledge-bases/{kb_id}", status_code=204, responses=_ERROR_RESPONSES)
async def delete_knowledge_base(kb_id: str, request: Request):
    registry = request.app.state.kb_registry
    if not registry.exists(kb_id):
        return _error(ErrorCode.KB_NOT_FOUND, f"知识库不存在: {kb_id}", 404)
    index_jobs = request.app.state.index_jobs
    # 排进该 KB 的序列化 executor，等待前序入库任务完成再执行。
    try:
        await run_sync(
            request.app.state.offload_executor,
            index_jobs.run_blocking,
            kb_id,
            _delete_kb,
            kb_id,
            registry,
            index_jobs,
            request.app.state.session_store,
            request.app.state.knowledge_store,
            request.app.state.feedback_store,
            request.app.state.feedback_analysis_store,
            request.app.state.retrieval_feedback_store,
            request.app.state.retrieval_eval_draft_store,
        )
    except KBCleanupError:
        # 清理不完整：registry 与 manifest 均保留，返回可重试错误而非误报删除成功。
        return _error(
            ErrorCode.KB_CLEANUP_FAILED, f"知识库清理未完成，请重试: {kb_id}", 500
        )
    return Response(status_code=204)


# 列出 documents。
@router.get("/knowledge-bases/{kb_id}/documents", responses=_ERROR_RESPONSES)
async def list_documents(kb_id: str, request: Request):
    if not request.app.state.kb_registry.exists(kb_id):
        return _error(ErrorCode.KB_NOT_FOUND, f"知识库不存在: {kb_id}", 404)
    return _kb_documents(kb_id)


# 列出知识库来源文件。
@router.get(
    "/knowledge-bases/{kb_id}/sources",
    response_model=SourceListResponse,
    responses=_ERROR_RESPONSES,
)
async def list_sources(kb_id: str, request: Request):
    if not request.app.state.kb_registry.exists(kb_id):
        return _error(ErrorCode.KB_NOT_FOUND, f"知识库不存在: {kb_id}", 404)
    source_reader = getattr(request.app.state, "source_list_reader", _kb_sources)
    sources = await run_sync(request.app.state.offload_executor, source_reader, kb_id)
    return SourceListResponse(kb_id=kb_id, sources=sources)


# 查询来源文件 chunks。
@router.get(
    "/knowledge-bases/{kb_id}/sources/{source}/chunks",
    response_model=SourceChunksResponse,
    responses=_ERROR_RESPONSES,
)
async def source_chunks(
    kb_id: str,
    source: str,
    request: Request,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    anchor_text: str | None = None,
):
    if not request.app.state.kb_registry.exists(kb_id):
        return _error(ErrorCode.KB_NOT_FOUND, f"知识库不存在: {kb_id}", 404)
    chunks_reader = getattr(
        request.app.state, "source_chunks_reader", read_source_chunks
    )
    chunks = await run_sync(
        request.app.state.offload_executor, chunks_reader, kb_id, source
    )
    window = chunks[offset : offset + limit]
    return SourceChunksResponse(
        kb_id=kb_id,
        source=source,
        total_count=len(chunks),
        offset=offset,
        limit=limit,
        chunks=[chunk_preview(chunk, anchor_text) for chunk in window],
    )


# 完成 上传document 处理。
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
    job = await run_sync(
        request.app.state.offload_executor,
        request.app.state.index_jobs.submit_upload,
        kb_id,
        source_dir,
        filename,
        bytes(content),
    )
    return _public_job(job)


# 删除 document。
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
    job = await run_sync(
        request.app.state.offload_executor,
        request.app.state.index_jobs.submit_delete_doc,
        kb_id,
        path,
    )
    return _public_job(job)


# 返回索引任务。
@router.get("/index-jobs/{job_id}", responses=_ERROR_RESPONSES)
async def get_index_job(job_id: str, request: Request):
    job = request.app.state.index_jobs.get(job_id)
    if job is None:
        return _error(ErrorCode.JOB_NOT_FOUND, f"任务不存在: {job_id}", 404)
    return _public_job(job)
