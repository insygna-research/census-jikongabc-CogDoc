import json
import os
import time
from threading import Lock
from config.settings import get_settings


# 封装 EpochCorruptError 的状态与行为。
class EpochCorruptError(Exception):
    # epoch/tombstone 损坏：归零会令 incarnation 防护失效（旧任务被重新合法化），故 fail-closed 抛错。
    pass


# 封装 EpochStore 的状态与行为。
class EpochStore:
    # 全局 epoch/tombstone：存在 KB 目录之外（data/kb/epochs.json），删库不会随目录一起抹掉。 重建同名 KB 时 epoch 续增而非归零，杜绝「删库前在飞的旧任务被重新合法化」。 原子性依赖共享实例 + 外部 kb_write_lock；多进程部署不安全，需改文件/DB 锁（当前单进程约束）。
    def __init__(self, path: str | None = None):
        self._path = path or os.path.join(get_settings().kb_root, "epochs.json")
        self._degraded_path = f"{self._path}.degraded"
        self._lock = Lock()

    # 加载 load 相关逻辑。
    def _load(self) -> dict:
        # 文件不存在=空表。损坏则隔离并抛错：绝不退回空表把损坏解释成 epoch 0。
        if os.path.exists(self._degraded_path):
            raise EpochCorruptError(
                f"epochs 处于 degraded 状态，需人工恢复: {self._path}"
            )
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            self._quarantine_corrupt()
            raise EpochCorruptError(f"epochs 损坏已隔离: {self._path}")
        if not isinstance(data, dict):
            self._quarantine_corrupt()
            raise EpochCorruptError(f"epochs 顶层非 dict 已隔离: {self._path}")
        if any(
            not isinstance(k, str)
            or not isinstance(v, int)
            or isinstance(v, bool)
            or v < 0
            for k, v in data.items()
        ):
            self._quarantine_corrupt()
            raise EpochCorruptError(f"epochs 条目损坏已隔离: {self._path}")
        return data

    # 隔离 quarantine corrupt 相关逻辑。
    def _quarantine_corrupt(self) -> None:
        try:
            os.replace(self._path, f"{self._path}.corrupt-{time.time_ns()}")
        except OSError:
            pass
        try:
            with open(self._degraded_path, "w", encoding="utf-8") as f:
                f.write(str(int(time.time())))
        except OSError:
            pass

    # 保存 save 相关逻辑。
    def _save(self, data: dict) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        tmp_path = f"{self._path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self._path)

    # 获取当前 current 相关逻辑。
    def current(self, kb_id: str) -> int:
        with self._lock:
            return self._load().get(kb_id, 0)

    # 递增 bump 相关逻辑。
    def bump(self, kb_id: str) -> int:
        with self._lock:
            data = self._load()
            nxt = data.get(kb_id, 0) + 1
            data[kb_id] = nxt
            self._save(data)
            return nxt


_shared: EpochStore | None = None
_shared_lock = Lock()


# 处理 shared epoch store 相关逻辑。
def shared_epoch_store() -> EpochStore:
    # 进程内共享单例，保证所有 KBState 实例看到同一份 epoch；双重检查锁防并发重复构造。
    global _shared
    if _shared is None:
        with _shared_lock:
            if _shared is None:
                _shared = EpochStore()
    return _shared
