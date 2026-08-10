import asyncio
from concurrent.futures import Executor
import math
import time
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
        wrapped_future.cancel()
        concurrent_future.cancel()
        raise


async def run_sync_until(
    executor: Executor,
    function: Callable[P, T],
    *args: P.args,
    deadline_monotonic: float,
    on_cancel: Callable[[], None] | None = None,
    on_timeout: Callable[[], None] | None = None,
    **kwargs: P.kwargs,
) -> T:
    """Run work on an executor while enforcing one absolute queue+run deadline."""

    if not math.isfinite(deadline_monotonic):
        raise ValueError("deadline_monotonic must be finite")
    concurrent_future = executor.submit(function, *args, **kwargs)
    wrapped_future = asyncio.wrap_future(concurrent_future)
    try:
        while True:
            remaining = deadline_monotonic - time.monotonic()
            if remaining <= 0:
                if on_timeout is not None:
                    try:
                        on_timeout()
                    except Exception:
                        pass
                wrapped_future.cancel()
                concurrent_future.cancel()
                raise TimeoutError("executor operation deadline exceeded")
            try:
                return await asyncio.wait_for(
                    asyncio.shield(wrapped_future),
                    timeout=min(_COMPLETION_WATCHDOG_SECONDS, remaining),
                )
            except TimeoutError:
                if concurrent_future.done():
                    return concurrent_future.result()
                if time.monotonic() >= deadline_monotonic:
                    if on_timeout is not None:
                        try:
                            on_timeout()
                        except Exception:
                            pass
                    wrapped_future.cancel()
                    concurrent_future.cancel()
                    raise TimeoutError("executor operation deadline exceeded")
    except asyncio.CancelledError:
        if on_cancel is not None:
            try:
                on_cancel()
            except Exception:
                pass
        wrapped_future.cancel()
        concurrent_future.cancel()
        raise
