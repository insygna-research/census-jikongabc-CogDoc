from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Callable
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from api.access_control import (
    AccessControlMiddleware,
    TokenBucketRateLimiter,
    build_rate_limiter,
)
from api.feedback_store import FeedbackStore
from api.ingest import IndexJobManager, KnowledgeBaseRegistry
from api.metrics import Metrics, MetricsMiddleware
from api.persistence import SqliteJobStore, SqliteSessionStore
from api.routes import (
    chat_router,
    documents_router,
    feedback_router,
    health_router,
)
from api.schemas import ErrorCode, build_error_response
from api.session_store import SessionStore
import logging
from config.settings import get_settings
from observability.logger import configure_logging, log_event
from service.chat_service import ChatResult, run_chat, run_chat_sync
from service.ingest_service import cancel_all_timers, drain_purge_queue
from service.mutation_journal import shared_mutation_journal
from service.process_lock import (
    SingleInstanceError,
    acquire_single_instance_lock,
    locking_supported,
    release_single_instance_lock,
    strict_single_process,
)
from service.sweeper import BackgroundSweeper


ChatRunner = Callable[..., ChatResult]


# 处理 unhandled error response 相关逻辑。
def _unhandled_error_response(exc: Exception) -> JSONResponse:
    # 线程池关闭竞争窗口的调度异常归为暂时不可用，其余未预期异常归为内部错误；都不漏栈。
    if isinstance(exc, RuntimeError) and "shutdown" in str(exc):
        code, status, message = ErrorCode.MODEL_UNAVAILABLE, 503, "服务正在关闭，请重试"
    else:
        code, status, message = ErrorCode.INTERNAL_ERROR, 500, "服务内部错误"
    error = build_error_response(
        code, message, details={"error_class": type(exc).__name__}
    )
    return JSONResponse(status_code=status, content=error.model_dump())


# 创建 create app 相关逻辑。
def create_app(
    *,
    chat_runner: ChatRunner | None = None,
    chat_stream_runner: Callable | None = None,
    session_store: SessionStore | None = None,
    kb_registry: KnowledgeBaseRegistry | None = None,
    index_jobs: IndexJobManager | None = None,
    feedback_store: FeedbackStore | None = None,
    api_keys: set[str] | None = None,
    rate_limiter: TokenBucketRateLimiter | None = None,
    offload_workers: int = 8,
) -> FastAPI:
    # 处理 lifespan 相关逻辑。
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 非 CLI 入口也要在启动时配一次日志，否则节点 log_event 全部静默丢失。
        configure_logging()
        # 单进程独占锁：仅在平台支持 flock 时强制；占用失败=已有实例，严格模式拒绝启动。
        lock_fh = acquire_single_instance_lock()
        if lock_fh is None and strict_single_process():
            # 无法取得锁（已有实例）或平台无 flock（无法保证单实例）：严格模式一律 fail-closed 拒绝启动， 单进程架构下绝不放行可能的并发写。明知后果可设 COGDOC_ALLOW_MULTI=1 显式放行。
            reason = (
                "平台不支持进程锁，无法保证单实例"
                if not locking_supported()
                else "已有 CogDoc 实例运行"
            )
            raise SingleInstanceError(f"{reason}；如确需放行请设 COGDOC_ALLOW_MULTI=1")
        if lock_fh is None:
            log_event(
                "startup", "single_instance_unconfirmed", {}, level=logging.WARNING
            )
        try:
            # 回放上次进程崩溃遗留的源文件 mutation，使源目录与 active 代一致。
            recovered = shared_mutation_journal().recover_all()
            if recovered:
                log_event(
                    "startup",
                    "mutation_journal_recovered",
                    {},
                    level=logging.WARNING,
                    count=len(recovered),
                )
            # 必须在拿到单实例锁且 journal 恢复之后对账；否则第二个 worker 在被拒绝前 就可能把第一实例的 running job 错标失败。
            app.state.index_jobs.reconcile_orphans()
            # 重试上次遗留的删库外部资源清理（Timer 随进程退出丢失，持久队列在此兜底）。
            drain_purge_queue()
            # 后台清扫：僵尸 generation GC、空闲 executor 淘汰、锁表压缩。
            sweeper = BackgroundSweeper(
                kb_ids_provider=lambda: [
                    r["kb_id"] for r in app.state.kb_registry.list()
                ],
                index_jobs=app.state.index_jobs,
            )
            sweeper.start()
            app.state.sweeper = sweeper
            # 鉴权未配置=所有 /v1 对外开放，生产忘配 key 时启动即告警。
            if not app.state.auth_enabled:
                log_event(
                    "startup",
                    "auth_disabled",
                    {},
                    level=logging.WARNING,
                )
            yield
        finally:
            # 每步独立容错，进程锁放最外层 finally，避免某个 shutdown 异常跳过后续清理。
            try:
                sweeper = getattr(app.state, "sweeper", None)
                if sweeper is not None:
                    sweeper.stop()
            except Exception as exc:
                log_event(
                    "shutdown",
                    "sweeper_stop_failed",
                    {},
                    level=logging.ERROR,
                    error_class=type(exc).__name__,
                )
            try:
                # 先排空 offload（其中可能同步等待 per-KB executor）。
                app.state.offload_executor.shutdown(wait=True)
            except Exception as exc:
                log_event(
                    "shutdown",
                    "offload_shutdown_failed",
                    {},
                    level=logging.ERROR,
                    error_class=type(exc).__name__,
                )
            try:
                # 再排空 mutation；它们提交时仍可能新建清理 Timer。
                app.state.index_jobs.shutdown(wait=True)
            except Exception as exc:
                log_event(
                    "shutdown",
                    "index_jobs_shutdown_failed",
                    {},
                    level=logging.ERROR,
                    error_class=type(exc).__name__,
                )
            drained = False
            try:
                # 所有 Timer 生产者都已停止后再统一取消/等待。
                drained = cancel_all_timers()
            except Exception as exc:
                log_event(
                    "shutdown",
                    "timer_shutdown_failed",
                    {},
                    level=logging.ERROR,
                    error_class=type(exc).__name__,
                )
            # 仅在后台线程确已排空时才显式释放进程锁；否则留给进程退出由 OS 释放， 杜绝"锁已放但卡死的清理线程仍在写索引、新进程并发拉起"的窗口。
            if drained:
                release_single_instance_lock(lock_fh)
            else:
                log_event(
                    "shutdown",
                    "lock_release_deferred_threads_alive",
                    {},
                    level=logging.WARNING,
                )

    app = FastAPI(
        title="CogDoc API",
        version="0.2.0",
        lifespan=lifespan,
    )
    # runner/store 可注入，便于脱离真实图与持久态测试交付层。
    app.state.chat_runner = chat_runner or run_chat_sync
    app.state.chat_stream_runner = chat_stream_runner or run_chat
    app.state.session_store = session_store or SessionStore()
    # 有界线程池限制本地算力并发，缓解高并发下精排/嵌入的坏邻居效应。
    app.state.offload_executor = ThreadPoolExecutor(
        max_workers=offload_workers, thread_name_prefix="cogdoc-offload"
    )
    # 入库注册表/任务管理器可注入，便于测试用假入库函数。
    app.state.kb_registry = kb_registry or KnowledgeBaseRegistry()
    # kb_exists 注入用于 _run_with_write 内的防复活检查；注入版跳过检查（测试按需自行传）。
    app.state.index_jobs = index_jobs or IndexJobManager(
        kb_exists=app.state.kb_registry.exists
    )
    app.state.feedback_store = feedback_store or FeedbackStore()

    # 访问控制：key 留空则鉴权关闭；限流默认按 settings 的令牌桶。两者均可注入测试。
    settings = get_settings()
    resolved_keys = settings.api_key_set if api_keys is None else api_keys
    resolved_limiter = rate_limiter or build_rate_limiter(
        settings.rate_limit_per_minute, settings.rate_limit_burst
    )
    app.state.auth_enabled = bool(resolved_keys)
    app.add_middleware(
        AccessControlMiddleware,
        api_keys=resolved_keys,
        rate_limiter=resolved_limiter,
    )
    # 指标中间件在访问控制外层（后加=最外层），故 401/429 也被计入请求统计。
    app.state.metrics = Metrics()
    app.add_middleware(MetricsMiddleware, metrics=app.state.metrics)

    # 处理 handle unexpected 相关逻辑。
    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        return _unhandled_error_response(exc)

    app.include_router(chat_router)
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(feedback_router)
    return app


# 生产入口：会话与入库任务落 SQLite，进程重启不丢；create_app 默认仍是内存版便于测试隔离。
_db_path = get_settings().state_db_path
_kb_registry = KnowledgeBaseRegistry()
app = create_app(
    session_store=SqliteSessionStore(_db_path),
    kb_registry=_kb_registry,
    index_jobs=IndexJobManager(
        job_store=SqliteJobStore(_db_path, reconcile_on_init=False),
        kb_exists=_kb_registry.exists,
    ),
)
