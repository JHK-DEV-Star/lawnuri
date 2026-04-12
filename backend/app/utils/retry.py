"""
Retry decorators with exponential backoff and jitter.
Provides both synchronous and asynchronous variants.
"""

import asyncio
import functools
import random
import time
from typing import Any, Callable, Tuple, Type

from app.utils.logger import logger


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
) -> Callable:
    """
    Synchronous retry decorator with exponential backoff and jitter.

    Args:
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay in seconds before the first retry.
        max_delay: Maximum delay in seconds between retries.
        exceptions: Tuple of exception types that trigger a retry.

    Returns:
        Decorated function with retry logic.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: BaseException | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc

                    if attempt == max_retries:
                        logger.error(
                            "[retry] %s failed after %d attempts: %s",
                            func.__name__,
                            max_retries + 1,
                            exc,
                        )
                        raise

                    # Exponential backoff with jitter
                    delay = min(initial_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0, delay * 0.5)
                    sleep_time = delay + jitter

                    logger.warning(
                        "[retry] %s attempt %d/%d failed: %s. "
                        "Retrying in %.2fs...",
                        func.__name__,
                        attempt + 1,
                        max_retries + 1,
                        exc,
                        sleep_time,
                    )
                    time.sleep(sleep_time)

            # Should not reach here, but just in case
            raise last_exception  # type: ignore[misc]

        return wrapper

    return decorator


def retry_with_backoff_async(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
) -> Callable:
    """
    Asynchronous retry decorator with exponential backoff and jitter.

    Args:
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay in seconds before the first retry.
        max_delay: Maximum delay in seconds between retries.
        exceptions: Tuple of exception types that trigger a retry.

    Returns:
        Decorated async function with retry logic.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: BaseException | None = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc

                    if attempt == max_retries:
                        logger.error(
                            "[retry_async] %s failed after %d attempts: %s",
                            func.__name__,
                            max_retries + 1,
                            exc,
                        )
                        raise

                    # Exponential backoff with jitter
                    delay = min(initial_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0, delay * 0.5)
                    sleep_time = delay + jitter

                    logger.warning(
                        "[retry_async] %s attempt %d/%d failed: %s. "
                        "Retrying in %.2fs...",
                        func.__name__,
                        attempt + 1,
                        max_retries + 1,
                        exc,
                        sleep_time,
                    )
                    await asyncio.sleep(sleep_time)

            # Should not reach here, but just in case
            raise last_exception  # type: ignore[misc]

        return wrapper

    return decorator
