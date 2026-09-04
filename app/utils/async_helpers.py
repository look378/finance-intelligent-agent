"""
Async helper utilities.

Common async utility functions used throughout the application.
"""
import asyncio
from typing import TypeVar, Callable, Awaitable, Any, List

T = TypeVar("T")


async def run_in_threadpool(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """
    Run a sync function in a thread pool.

    Useful for blocking I/O operations that don't have async versions.

    Args:
        func: Sync function to run
        *args: Positional arguments to pass to func
        **kwargs: Keyword arguments to pass to func

    Returns:
        The result of func(*args, **kwargs)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args, **kwargs)


async def gather_with_concurrency(
    *coroutines: Awaitable[T],
    concurrency: int = 10,
) -> List[T]:
    """
    Run coroutines with a limit on concurrent executions.

    Args:
        *coroutines: Coroutines to execute
        concurrency: Maximum number of concurrent operations

    Returns:
        List[T]: Results from all coroutines in order
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def run_with_lock(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro

    return await asyncio.gather(*(run_with_lock(c) for c in coroutines))
