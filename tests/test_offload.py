from concurrent.futures import ThreadPoolExecutor

import pytest
from cogdoc.api.offload import run_sync


@pytest.fixture
def anyio_backend():
    return "asyncio"


# 验证线程池结果和异常都能稳定回到事件循环。
@pytest.mark.anyio
async def test_run_sync_returns_results_and_propagates_errors():
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        assert await run_sync(executor, lambda value: value + 1, 1) == 2

        def fail():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await run_sync(executor, fail)
    finally:
        executor.shutdown(wait=True)
