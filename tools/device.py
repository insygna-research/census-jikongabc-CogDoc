import torch
from config.settings import get_settings


def cuda_free_bytes() -> int:
    # 当前 GPU 的实际空闲显存（已计入其它进程占用）；查询失败按 0 处理。
    try:
        free, _total = torch.cuda.mem_get_info()
        return int(free)
    except Exception:
        return 0


def mps_available() -> bool:
    # 单独封装 MPS 检测，方便测试替换并兼容无 backends 的 torch 构建。
    backend = getattr(getattr(torch, "backends", None), "mps", None)
    return backend is not None and backend.is_available()


def required_cuda_free_bytes(env_var: str) -> int:
    # 阈值支持环境变量按 MB 覆盖，内部统一换算为字节比较。
    return get_settings().cuda_min_free_bytes(env_var)


def resolve_device(min_cuda_free_bytes: int, current_device: str = None, model_loaded: bool = False) -> str:
    # 按当前空闲显存动态判定：够则 GPU 加速、不够回落 CPU，避免小显存/共享卡 OOM。
    if current_device == "cuda" and model_loaded:
        return "cuda"
    if torch.cuda.is_available() and cuda_free_bytes() >= min_cuda_free_bytes:
        return "cuda"
    if mps_available():
        return "mps"
    return "cpu"
