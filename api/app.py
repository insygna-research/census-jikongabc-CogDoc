from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Callable
from fastapi import FastAPI
from api.routes import chat_router
from api.session_store import SessionStore
from observability.logger import configure_logging
from service.chat_service import ChatResult, run_chat_sync


ChatRunner = Callable[..., ChatResult]


def create_app(
    *,
    chat_runner: ChatRunner | None = None,
    session_store: SessionStore | None = None,
    offload_workers: int = 8,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 非 CLI 入口也要在启动时配一次日志，否则节点 log_event 全部静默丢失。
        configure_logging()
        yield
        app.state.offload_executor.shutdown(wait=False)

    app = FastAPI(
        title="CogDoc API",
        version="0.2.0",
        lifespan=lifespan,
    )
    # runner/store 可注入，便于脱离真实图与持久态测试交付层。
    app.state.chat_runner = chat_runner or run_chat_sync
    app.state.session_store = session_store or SessionStore()
    # 有界线程池限制本地算力并发，缓解高并发下精排/嵌入的坏邻居效应。
    app.state.offload_executor = ThreadPoolExecutor(
        max_workers=offload_workers, thread_name_prefix="cogdoc-offload"
    )
    app.include_router(chat_router)
    return app


app = create_app()
