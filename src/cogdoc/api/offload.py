import asyncio
from concurrent.futures import Executor
from typing import Callable, ParamSpec, TypeVar


P = ParamSpec("P")
T = TypeVar("T")
_COMPLETION_WATCHDOG_SECONDS = 0.05


# 在线程完成通知未唤醒事件循环时，用短周期 watchdog 收割已完成结果，避免请求永久挂起。
async def run_sync(
    executor: Executor,
    function: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    concurrent_future = executor.submit(function, *args, **kwargs)
    wrapped_future = asyncio.wrap_future(concurrent_future)
    try:
        while True:
            try:
                return await asyncio.wait_for(
                    asyncio.shield(wrapped_future),
                    timeout=_COMPLETION_WATCHDOG_SECONDS,
                )
            except TimeoutError:
                if concurrent_future.done():
                    return concurrent_future.result()
    except asyncio.CancelledError:
        concurrent_future.cancel()
        raise
