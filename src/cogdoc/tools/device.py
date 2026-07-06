import threading
import torch
from cogdoc.config.settings import get_settings

_MODEL_SEMAPHORES = {}
_MODEL_SEMAPHORE_SIZES = {}
_MODEL_SEMAPHORE_LOCK = threading.Lock()
_TORCH_THREADS_CONFIGURED = None
_TORCH_THREADS_LOCK = threading.Lock()


# 配置 Torch CPU 线程数。
def configure_torch_threads() -> None:
    global _TORCH_THREADS_CONFIGURED
    threads = get_settings().torch_num_threads
    if not threads or threads <= 0 or _TORCH_THREADS_CONFIGURED == threads:
        return
    with _TORCH_THREADS_LOCK:
        if _TORCH_THREADS_CONFIGURED == threads:
            return
        if torch.get_num_threads() != threads:
            torch.set_num_threads(threads)
        _TORCH_THREADS_CONFIGURED = threads


# 返回模型推理并发闸门。
def model_inference_semaphore(kind: str) -> threading.BoundedSemaphore:
    settings = get_settings()
    sizes = {
        "embedder": settings.cogdoc_embedder_max_concurrency,
        "reranker": settings.cogdoc_reranker_max_concurrency,
    }
    if kind not in sizes:
        raise ValueError(f"未知模型推理闸门: {kind}")
    size = max(1, int(sizes[kind]))
    with _MODEL_SEMAPHORE_LOCK:
        if (
            _MODEL_SEMAPHORES.get(kind) is None
            or _MODEL_SEMAPHORE_SIZES.get(kind) != size
        ):
            _MODEL_SEMAPHORES[kind] = threading.BoundedSemaphore(size)
            _MODEL_SEMAPHORE_SIZES[kind] = size
        return _MODEL_SEMAPHORES[kind]


# 完成 CUDAfreebytes 处理。
def cuda_free_bytes() -> int:
    # 当前 GPU 的实际空闲显存（已计入其它进程占用）；查询失败按 0 处理。
    try:
        free, _total = torch.cuda.mem_get_info()
        return int(free)
    except Exception:
        return 0


# 完成 MPSavailable 处理。
def mps_available() -> bool:
    # 单独封装 MPS 检测，方便测试替换并兼容无 backends 的 torch 构建。
    backend = getattr(getattr(torch, "backends", None), "mps", None)
    return backend is not None and backend.is_available()


# 完成 requiredCUDAfreebytes 处理。
def required_cuda_free_bytes(env_var: str) -> int:
    # 阈值支持环境变量按 MB 覆盖，内部统一换算为字节比较。
    return get_settings().cuda_min_free_bytes(env_var)


# 解析 device。
def resolve_device(
    min_cuda_free_bytes: int, current_device: str = None, model_loaded: bool = False
) -> str:
    # 按当前空闲显存动态判定：够则 GPU 加速、不够回落 CPU，避免小显存/共享卡 OOM。
    configure_torch_threads()
    if current_device == "cuda" and model_loaded:
        return "cuda"
    if torch.cuda.is_available() and cuda_free_bytes() >= min_cuda_free_bytes:
        return "cuda"
    if mps_available():
        return "mps"
    return "cpu"
