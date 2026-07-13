import time
from fastapi import Request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.types import ASGIApp, Message, Receive, Scope, Send


# 每个 app 独立 CollectorRegistry：避免多次 create_app 在全局注册表重复注册、测试间互相串数。
class Metrics:
    # 每个 app 独立 CollectorRegistry：避免多次 create_app 在全局注册表重复注册、测试间互相串数。
    def __init__(self):
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "cogdoc_http_requests_total",
            "HTTP 请求总数",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.duration = Histogram(
            "cogdoc_http_request_duration_seconds",
            "HTTP 请求耗时",
            ["method", "route"],
            registry=self.registry,
        )
        self.in_progress = Gauge(
            "cogdoc_http_requests_in_progress",
            "在途 HTTP 请求数",
            registry=self.registry,
        )
        self.chat_results = Counter(
            "cogdoc_chat_results_total",
            "对话产出按任务类型与是否可信计数",
            ["task_type", "valid"],
            registry=self.registry,
        )

    # 渲染。
    def render(self) -> bytes:
        return generate_latest(self.registry)


# 路由 label。
def _route_label(request: Request) -> str:
    # 用路由模板（/v1/index-jobs/{job_id}）而非原始路径，避免路径参数撑爆标签基数。
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path or "unmatched"


# 统计每个请求的计数、耗时与在途数；置于访问控制外层，故 401/429 也计入。
class MetricsMiddleware:
    # 统计每个请求的计数、耗时与在途数；置于访问控制外层，故 401/429 也计入。
    def __init__(self, app: ASGIApp, metrics: Metrics):
        self.app = app
        self._metrics = metrics

    # 分发结果。
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        start = time.perf_counter()
        self._metrics.in_progress.inc()
        # 默认 500：call_next 抛出未被兜底的异常时也记一条并让异常透传，不静默丢指标。
        status = "500"

        async def send_with_status(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = str(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, send_with_status)
        finally:
            self._metrics.in_progress.dec()
            route = _route_label(request)
            elapsed = time.perf_counter() - start
            self._metrics.requests.labels(request.method, route, status).inc()
            self._metrics.duration.labels(request.method, route).observe(elapsed)


__all__ = ["CONTENT_TYPE_LATEST", "Metrics", "MetricsMiddleware"]
