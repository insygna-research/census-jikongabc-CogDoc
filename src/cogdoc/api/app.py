from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Callable
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from cogdoc.api.access_control import (
    AccessControlMiddleware,
    TokenBucketRateLimiter,
    build_rate_limiter,
)
from cogdoc.api.derived_knowledge_store import DerivedKnowledgeStore
from cogdoc.api.feedback_analysis_store import FeedbackAnalysisStore
from cogdoc.api.feedback_store import FeedbackStore, SqliteFeedbackStore
from cogdoc.api.ingest import IndexJobManager, KnowledgeBaseRegistry
from cogdoc.api.metrics import Metrics, MetricsMiddleware
from cogdoc.api.persistence import SqliteJobStore, SqliteSessionStore
from cogdoc.api.retrieval_feedback_store import RetrievalFeedbackStore
from cogdoc.api.routes import (
    agent_router,
    chat_router,
    documents_router,
    feedback_router,
    health_router,
    knowledge_router,
    traces_router,
)
from cogdoc.api.schemas import ErrorCode, build_error_response
from cogdoc.api.session_store import SessionStore
from cogdoc.api.webhooks import WebhookDispatcher
import logging
from cogdoc.config.settings import get_settings
from cogdoc.observability.logger import configure_logging, log_event
from cogdoc.service.chat_service import ChatResult, run_chat, run_chat_sync
from cogdoc.service.ingest_service import cancel_all_timers, drain_purge_queue
from cogdoc.service.mutation_journal import shared_mutation_journal
from cogdoc.service.process_lock import (
    SingleInstanceError,
    acquire_single_instance_lock,
    locking_supported,
    release_single_instance_lock,
    strict_single_process,
)
from cogdoc.service.sweeper import BackgroundSweeper


ChatRunner = Callable[..., ChatResult]


# 创建反馈存储。
def _default_feedback_store():
    settings = get_settings()
    if settings.cogdoc_feedback_store.strip().lower() == "sqlite":
        return SqliteFeedbackStore()
    return FeedbackStore()


# 构建未捕获异常响应。
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


# 创建服务应用。
def create_app(
    *,
    chat_runner: ChatRunner | None = None,
    chat_stream_runner: Callable | None = None,
    session_store: SessionStore | None = None,
    kb_registry: KnowledgeBaseRegistry | None = None,
    index_jobs: IndexJobManager | None = None,
    feedback_store: FeedbackStore | None = None,
    feedback_analysis_store: FeedbackAnalysisStore | None = None,
    knowledge_store: DerivedKnowledgeStore | None = None,
    retrieval_feedback_store: RetrievalFeedbackStore | None = None,
    webhook_dispatcher: WebhookDispatcher | None = None,
    derived_knowledge_index_refresher: Callable | None = None,
    derived_knowledge_index_statuser: Callable | None = None,
    api_keys: set[str] | None = None,
    rate_limiter: TokenBucketRateLimiter | None = None,
    offload_workers: int | None = None,
) -> FastAPI:
    # 管理结果。
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 非命令行入口也要在启动时配置日志，否则节点日志会静默丢失。
        configure_logging()
        # 单进程独占锁，严格模式下拿不到锁就拒绝启动。
        lock_fh = acquire_single_instance_lock()
        if lock_fh is None and strict_single_process():
            # 无法取得锁时严格拒绝启动，避免单进程架构出现并发写。
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
            # 回放上次进程崩溃遗留的源文件变更，使源目录与当前索引代一致。
            recovered = shared_mutation_journal().recover_all()
            if recovered:
                log_event(
                    "startup",
                    "mutation_journal_recovered",
                    {},
                    level=logging.WARNING,
                    count=len(recovered),
                )
            # 必须在拿到单实例锁且变更日志恢复之后对账，避免误改其他实例的任务状态。
            app.state.index_jobs.reconcile_orphans()
            # 重试上次遗留的删库外部资源清理，持久队列在此兜底。
            drain_purge_queue()
            # 后台清扫僵尸索引代、空闲执行器和锁表。
            sweeper = BackgroundSweeper(
                kb_ids_provider=lambda: [
                    r["kb_id"] for r in app.state.kb_registry.list()
                ],
                index_jobs=app.state.index_jobs,
            )
            sweeper.start()
            app.state.sweeper = sweeper
            # 鉴权未配置时接口对外开放，启动时告警。
            if not app.state.auth_enabled:
                log_event(
                    "startup",
                    "auth_disabled",
                    {},
                    level=logging.WARNING,
                )
            yield
        finally:
            # 每步独立容错，进程锁放最外层，避免某个关闭异常跳过后续清理。
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
                # 先排空请求卸载线程池。
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
                # 再排空索引任务，它们提交时仍可能新建清理定时器。
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
                # 所有定时器生产者都停止后再统一取消和等待。
                drained = cancel_all_timers()
            except Exception as exc:
                log_event(
                    "shutdown",
                    "timer_shutdown_failed",
                    {},
                    level=logging.ERROR,
                    error_class=type(exc).__name__,
                )
            # 仅在后台线程确已排空时才显式释放进程锁，否则留给进程退出自动释放。
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
    # 运行器和存储可注入，便于脱离真实图与持久态测试交付层。
    app.state.chat_runner = chat_runner or run_chat_sync
    app.state.chat_stream_runner = chat_stream_runner or run_chat
    app.state.session_store = session_store or SessionStore()
    # 有界线程池限制本地算力并发，缓解高并发下精排/嵌入的坏邻居效应。
    app.state.offload_executor = ThreadPoolExecutor(
        max_workers=offload_workers or get_settings().cogdoc_offload_workers,
        thread_name_prefix="cogdoc-offload",
    )
    # 入库注册表/任务管理器可注入，便于测试用假入库函数。
    app.state.kb_registry = kb_registry or KnowledgeBaseRegistry()
    # 知识库存在性检查用于写入防复活，注入版由测试自行控制。
    app.state.index_jobs = index_jobs or IndexJobManager(
        kb_exists=app.state.kb_registry.exists
    )
    app.state.feedback_store = feedback_store or _default_feedback_store()
    app.state.feedback_analysis_store = (
        feedback_analysis_store or FeedbackAnalysisStore()
    )
    app.state.knowledge_store = knowledge_store or DerivedKnowledgeStore()
    app.state.retrieval_feedback_store = (
        retrieval_feedback_store or RetrievalFeedbackStore()
    )
    app.state.webhook_dispatcher = webhook_dispatcher or WebhookDispatcher()

    # 访问控制留空则鉴权关闭，限流默认按配置令牌桶。
    settings = get_settings()
    app.state.derived_knowledge_index_auto_refresh = (
        settings.cogdoc_derived_knowledge_index_auto_refresh
    )
    app.state.derived_knowledge_index_refresher = derived_knowledge_index_refresher
    app.state.derived_knowledge_index_statuser = derived_knowledge_index_statuser
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

    # 处理未预期异常。
    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        return _unhandled_error_response(exc)

    app.include_router(chat_router)
    app.include_router(agent_router)
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(feedback_router)
    app.include_router(knowledge_router)
    app.include_router(traces_router)
    return app


# 生产入口会话与入库任务落盘，进程重启不丢，默认创建仍便于测试隔离。
_settings = get_settings()
_db_path = _settings.state_db_path
_kb_registry = KnowledgeBaseRegistry()
app = create_app(
    session_store=SqliteSessionStore(_db_path, memory_policy=_settings.memory_policy),
    kb_registry=_kb_registry,
    index_jobs=IndexJobManager(
        job_store=SqliteJobStore(_db_path, reconcile_on_init=False),
        kb_exists=_kb_registry.exists,
    ),
)
