from threading import Lock, RLock


# 进程内按 kb_id 的写锁注册表：串行化同一知识库的入库、删库与文件增删， 避免「扫描 hash 与解析之间文件被改」「删库与后台任务交错」等竞态。 注意：仅在单进程内有效。多 worker / 多进程部署需改用文件锁、数据库锁或 coordinator 单实例约束。
_registry_lock = Lock()
_locks: dict[str, RLock] = {}
_refs: dict[str, int] = {}  # 锁句柄发放计数：>0 表示有人持有引用，禁止压缩回收


# 封装 _KbLock 的状态与行为。
class _KbLock:
    # kb_write_lock 的句柄：构造即登记引用（即便尚未 acquire，也不会被 compact 回收），退出 with 时注销。 闭合"取得旧锁引用但未 acquire 时被 sweeper 移除、重建同名 KB 拿到第二把锁"的竞态。
    def __init__(self, kb_id: str):
        self._kb_id = kb_id
        with _registry_lock:
            lock = _locks.get(kb_id)
            if lock is None:
                lock = RLock()
                _locks[kb_id] = lock
            self._lock = lock
            _refs[kb_id] = _refs.get(kb_id, 0) + 1

    # 进入上下文并获取资源。
    def __enter__(self):
        self._lock.acquire()
        return self

    # 退出上下文并释放资源。
    def __exit__(self, exc_type, exc, tb):
        self._lock.release()
        with _registry_lock:
            n = _refs.get(self._kb_id, 1) - 1
            if n <= 0:
                _refs.pop(self._kb_id, None)
            else:
                _refs[self._kb_id] = n
        return False


# 处理 kb write lock 相关逻辑。
def kb_write_lock(kb_id: str) -> _KbLock:
    # 返回可重入的 with-句柄：同线程嵌套 with 共享同一底层 RLock（如删库路由再调 delete_kb_index）。
    return _KbLock(kb_id)


# 压缩 compact locks 相关逻辑。
def compact_locks(keep: set[str]) -> int:
    # sweeper 调用：丢弃不在 keep 内、且引用计数为 0 的锁，防一次性 KB ID 撑大锁表。 引用计数从句柄构造起就 >0，故"已发放但未 acquire"的锁不会被回收，杜绝同名 KB 出现两把锁。
    with _registry_lock:
        drop = [k for k in _locks if k not in keep and _refs.get(k, 0) == 0]
        for kb_id in drop:
            del _locks[kb_id]
        return len(drop)
