from contextlib import contextmanager
from threading import Condition


_condition = Condition()
_readers: dict[str, int] = {}


# 完成 知识库读取读租约 处理。
@contextmanager
def kb_read_lease(kb_id: str):
    with _condition:
        _readers[kb_id] = _readers.get(kb_id, 0) + 1
    try:
        yield
    finally:
        with _condition:
            remaining = _readers.get(kb_id, 1) - 1
            if remaining <= 0:
                _readers.pop(kb_id, None)
            else:
                _readers[kb_id] = remaining
            _condition.notify_all()


# 判断是否存在 readers。
def has_readers(kb_id: str) -> bool:
    with _condition:
        return _readers.get(kb_id, 0) > 0
