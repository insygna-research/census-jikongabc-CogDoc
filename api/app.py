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


ChatRunner = Callable[..., ChatResult]


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
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 非 CLI 入口也要在启动时配一次日志，否则节点 log_event 全部静默丢失。
        configure_logging()
        # 鉴权未配置=所有 /v1 对外开放，生产忘配 key 时启动即告警。
        if not app.state.auth_enabled:
            log_event(
                "startup",
                "auth_disabled",
                {},
                level=logging.WARNING,
            )
        yield
        app.state.offload_executor.shutdown(wait=False)
        app.state.index_jobs.shutdown()

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
    app.state.index_jobs = index_jobs or IndexJobManager()
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
app = create_app(
    session_store=SqliteSessionStore(_db_path),
    index_jobs=IndexJobManager(job_store=SqliteJobStore(_db_path)),
)
