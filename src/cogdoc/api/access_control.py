import time
from collections import OrderedDict
from threading import Lock
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from cogdoc.api.error_mapping import status_for_code
from cogdoc.api.schemas import ErrorCode, build_error_response


# 探针与文档路径永远放行：鉴权/限流不能挡住存活就绪检查与 OpenAPI。
_EXEMPT_PATHS = frozenset(
    {"/healthz", "/readyz", "/metrics", "/docs", "/redoc", "/openapi.json"}
)
# 仅豁免限流（仍走鉴权）：前端刷新/轮询会高频读取这些轻量状态接口。
_RATE_LIMIT_EXEMPT_GET_PATHS = frozenset(
    ("/v1/knowledge-bases", "/v1/sessions", "/v1/traces")
)
_RATE_LIMIT_EXEMPT_GET_PREFIXES = (
    "/v1/index-jobs/",
    "/v1/knowledge-bases/",
    "/v1/sessions/",
    "/v1/traces/",
)


# 判断ratelimitexempt。
def _is_rate_limit_exempt(request: Request) -> bool:
    if request.method != "GET":
        return False
    path = request.url.path
    return path in _RATE_LIMIT_EXEMPT_GET_PATHS or path.startswith(
        _RATE_LIMIT_EXEMPT_GET_PREFIXES
    )


# 按身份分桶的令牌桶：突发容量 capacity，恒定速率 refill_per_second 补充。
class TokenBucketRateLimiter:
    # 按身份分桶的令牌桶：突发容量 capacity，恒定速率 refill_per_second 补充。
    def __init__(
        self,
        capacity: int,
        refill_per_second: float,
        max_identities: int = 10000,
    ):
        self.capacity = capacity
        self.refill_per_second = refill_per_second
        self.max_identities = max_identities
        # OrderedDict 维护按最近访问排序：末尾最新、头部最旧，淘汰 popitem(last=False) 为 O(1)。
        self._buckets: "OrderedDict[str, tuple[float, float]]" = OrderedDict()
        self._lock = Lock()

    # 放行结果。
    def allow(self, identity: str) -> bool:
        # capacity<=0 表示关闭限流，直接放行。
        if self.capacity <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(identity, (float(self.capacity), now))
            tokens = min(self.capacity, tokens + (now - last) * self.refill_per_second)
            allowed = tokens >= 1.0
            if allowed:
                tokens -= 1.0
            self._buckets[identity] = (tokens, now)
            self._buckets.move_to_end(identity)  # 标记为最近活跃
            # 内存无条件有界：超额时从头部（最久未活跃）逐个淘汰，O(overflow) 无需排序。
            while len(self._buckets) > self.max_identities:
                self._buckets.popitem(last=False)
            return allowed


# 完成 提取流程API密钥 处理。
def _extract_api_key(request: Request) -> str | None:
    # 先认 Authorization: Bearer，再退回 X-API-Key。
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    key = request.headers.get("x-api-key", "")
    return key.strip() or None


# 拒绝结果。
def _reject(code: ErrorCode, message: str) -> JSONResponse:
    error = build_error_response(code, message)
    return JSONResponse(status_code=status_for_code(code), content=error.model_dump())


# 统一入口的鉴权 + 限流：先校验 API key，再按身份限流，最后放行到路由。
class AccessControlMiddleware(BaseHTTPMiddleware):
    # 统一入口的鉴权 + 限流：先校验 API key，再按身份限流，最后放行到路由。
    def __init__(
        self, app, *, api_keys: set[str], rate_limiter: TokenBucketRateLimiter
    ):
        super().__init__(app)
        self._api_keys = api_keys
        self._limiter = rate_limiter

    # 分发结果。
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in _EXEMPT_PATHS:
            return await call_next(request)

        # 鉴权：仅当配置了 key 才开启；身份用于限流分桶。
        if self._api_keys:
            key = _extract_api_key(request)
            if key is None:
                return _reject(ErrorCode.UNAUTHORIZED, "缺少 API key")
            if key not in self._api_keys:
                return _reject(ErrorCode.UNAUTHORIZED, "无效的 API key")
            # 限流身份即 key：共享同一 key 的客户端共享额度（按调用方=租户限流，符合预期）。
            identity = key
        else:
            # 鉴权关闭时按客户端 IP 限流，仍能挡住单源洪泛。
            identity = request.client.host if request.client else "anonymous"

        # 高频只读端点过鉴权但不过限流，避免 Streamlit rerun/轮询误杀正常使用。
        if not _is_rate_limit_exempt(request):
            if not self._limiter.allow(identity):
                return _reject(ErrorCode.REQUEST_THROTTLED, "请求过于频繁，请稍后重试")

        return await call_next(request)


# 构建 rate limiter。
def build_rate_limiter(per_minute: int, burst: int) -> TokenBucketRateLimiter:
    # 每分钟速率换算成每秒补充；burst 即令牌桶容量。
    return TokenBucketRateLimiter(capacity=burst, refill_per_second=per_minute / 60.0)
