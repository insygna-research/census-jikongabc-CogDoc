from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from api.metrics import CONTENT_TYPE_LATEST
from tools.rust_core_loader import REQUIRED_NATIVE_SYMBOLS, ensure_rust_core

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz():
    # 存活探针：进程在跑即可，不做依赖检查。
    return {"status": "ok"}


@router.get("/readyz")
async def readyz():
    # 就绪探针：native 扩展是缺了就无法服务的硬依赖。
    try:
        ensure_rust_core(*REQUIRED_NATIVE_SYMBOLS)
    except RuntimeError as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "rust_core": False, "reason": str(exc)},
        )
    return {"status": "ready", "rust_core": True}


@router.get("/metrics")
async def metrics(request: Request):
    # Prometheus 抓取端点：返回每 app 注册表的文本快照，鉴权/限流已豁免。
    return Response(
        content=request.app.state.metrics.render(),
        media_type=CONTENT_TYPE_LATEST,
    )
