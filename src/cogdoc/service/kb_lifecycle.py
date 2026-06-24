import json
import logging
import os
import time
from threading import Lock
from cogdoc.config.settings import get_settings
from cogdoc.observability.logger import log_event


# active：正常读写。deleting：禁检索与新 mutation，仅允许重试删库。deleted：tombstone 防旧任务复活。
LIFECYCLE_ACTIVE = "active"
LIFECYCLE_DELETING = "deleting"
LIFECYCLE_DELETED = "deleted"
_VALID = {LIFECYCLE_ACTIVE, LIFECYCLE_DELETING, LIFECYCLE_DELETED}


# 封装 LifecycleStore 的状态与行为。
class LifecycleStore:
    # KB 生命周期标记，存在 KB 目录之外（lifecycle.json），删库不随目录消失，重建同名 KB 才显式切回 active。
    def __init__(self, path: str | None = None):
        self._path = path or os.path.join(get_settings().kb_root, "lifecycle.json")
        self._degraded_path = f"{self._path}.degraded"  # 持久全局 fail-closed 标记
        self._lock = Lock()

    # 加载 load 相关逻辑。
    def _load(self) -> dict:
        # 文件不存在=全新系统，正常返回空表（各 KB 默认 active）。仅结构合法的 dict 才采纳。
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return {}
        if not isinstance(data, dict) or any(
            not isinstance(k, str) or v not in _VALID for k, v in data.items()
        ):
            raise json.JSONDecodeError("lifecycle 结构或状态值损坏", "", 0)
        return data

    # 加载 load or corrupt 相关逻辑。
    def _load_or_corrupt(self) -> tuple:
        # 返回 (data, corrupt)。文件损坏（语法或结构）时 corrupt=True。
        try:
            return self._load(), False
        except json.JSONDecodeError:
            return {}, True

    # 保存 save 相关逻辑。
    def _save(self, data: dict) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        tmp_path = f"{self._path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self._path)

    # 处理 degraded 相关逻辑。
    def _degraded(self) -> bool:
        return os.path.exists(self._degraded_path)

    # 处理 status 相关逻辑。
    def status(self, kb_id: str) -> str:
        # 无记录视为 active：兼容历史 KB。一旦损坏过（degraded 标记常驻），全局 fail-closed 返回 deleting， 直到运维从 .corrupt 备份恢复 lifecycle.json 并删除 .degraded 标记，杜绝其他 tombstone 丢失后误放读。
        with self._lock:
            if self._degraded():
                return LIFECYCLE_DELETING
            data, corrupt = self._load_or_corrupt()
            if corrupt:
                return LIFECYCLE_DELETING
            value = data.get(kb_id)
            if value is None:
                return LIFECYCLE_ACTIVE
            return value if value in _VALID else LIFECYCLE_DELETING

    # 设置 set 相关逻辑。
    def set(self, kb_id: str, status: str) -> None:
        if status not in _VALID:
            raise ValueError(f"invalid lifecycle status: {status}")
        with self._lock:
            if self._degraded():
                raise RuntimeError(
                    f"lifecycle 处于 degraded 状态，需人工恢复: {self._path}"
                )
            data, corrupt = self._load_or_corrupt()
            if corrupt:
                # 损坏文件隔离保留（含其他 KB 的 tombstone），落持久 degraded 标记后立即失败。 不能重建一个只含当前 KB 的文件并返回成功，否则其他 KB tombstone 已丢却被伪装成成功变更。
                self._quarantine_corrupt()
                self._mark_degraded()
                raise RuntimeError(f"lifecycle 损坏已隔离，需人工恢复: {self._path}")
            data[kb_id] = status
            self._save(data)

    # 标记 mark degraded 相关逻辑。
    def _mark_degraded(self) -> None:
        try:
            with open(self._degraded_path, "w", encoding="utf-8") as f:
                f.write(str(int(time.time())))
        except OSError:
            pass

    # 隔离 quarantine corrupt 相关逻辑。
    def _quarantine_corrupt(self) -> None:
        # 把损坏文件改名留存供人工恢复，并记 error 让 tombstone 丢失可观测。
        try:
            os.replace(self._path, f"{self._path}.corrupt-{time.time_ns()}")
        except OSError:
            pass
        log_event(
            "lifecycle",
            "lifecycle_file_corrupt_quarantined",
            {},
            level=logging.ERROR,
        )


_shared: LifecycleStore | None = None
_shared_lock = Lock()


# 处理 shared lifecycle store 相关逻辑。
def shared_lifecycle_store() -> LifecycleStore:
    # 进程内共享单例；双重检查锁防并发重复构造。
    global _shared
    if _shared is None:
        with _shared_lock:
            if _shared is None:
                _shared = LifecycleStore()
    return _shared
