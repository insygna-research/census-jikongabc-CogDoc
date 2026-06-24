import pytest
from cogdoc.service.process_lock import (
    acquire_single_instance_lock,
    release_single_instance_lock,
)

fcntl = pytest.importorskip("fcntl")


# 验证 single instance lock blocks second。
def test_single_instance_lock_blocks_second(tmp_path):
    # 同一锁文件第二次获取被拒；释放后可再获取。
    path = str(tmp_path / "x.lock")
    fh1 = acquire_single_instance_lock(path)
    assert fh1 is not None
    assert acquire_single_instance_lock(path) is None

    release_single_instance_lock(fh1)
    fh2 = acquire_single_instance_lock(path)
    assert fh2 is not None
    release_single_instance_lock(fh2)
