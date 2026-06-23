from fastapi import APIRouter
from fastapi.responses import JSONResponse

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
