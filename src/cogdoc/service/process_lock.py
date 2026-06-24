import os
from cogdoc.config.settings import get_settings

try:
    import fcntl
except ImportError:
    fcntl = None  # 非 POSIX 平台：单实例锁降级为无操作

# 进程级独占锁：executor / KB 锁 / Retriever 缓存 / EpochStore 都是进程内对象， 多 worker 或 CLI+API 并发会绕过串行化并发写同一份 JSON/索引，此锁强制单进程。


# 封装 SingleInstanceError 的状态与行为。
class SingleInstanceError(RuntimeError):
    pass


# 处理 locking supported 相关逻辑。
def locking_supported() -> bool:
    # 平台是否支持 flock；不支持时严格模式不能阻止启动（否则非 POSIX 永远起不来）。
    return fcntl is not None


# 处理 acquire single instance lock 相关逻辑。
def acquire_single_instance_lock(lock_path: str | None = None):
    # 成功返回持有的文件句柄（须存活至进程退出）；已被占用返回 None；平台不支持也返回 None。
    if fcntl is None:
        return None
    path = lock_path or os.path.join(get_settings().kb_root, ".cogdoc.lock")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fh = open(path, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


# 释放 release single instance lock 相关逻辑。
def release_single_instance_lock(fh) -> None:
    if fh is None or fcntl is None or getattr(fh, "closed", False):
        return
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


# 处理 strict single process 相关逻辑。
def strict_single_process() -> bool:
    # 默认严格：检测到多实例即拒绝启动（executor/锁/缓存/epoch 均为进程内对象，并发写不安全）。 显式设 COGDOC_ALLOW_MULTI=1 才降级为仅告警（仅供明知后果的运维场景）。
    return os.environ.get("COGDOC_ALLOW_MULTI", "") != "1"
