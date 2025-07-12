"""General utility helpers for the Computer-Use Agent."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Awaitable, Callable, Coroutine, TypeVar

T = TypeVar("T")
logger = logging.getLogger("cua.utils")

# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def utcnow() -> datetime:  # noqa: D401 – simple helper
    """Return *timezone-aware* current UTC datetime."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------


def async_retry(
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    jitter: float = 0.25,
    retry_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Coroutine[Any, Any, T]]]:
    """Decorator that retries an *async* function using exponential back-off.

    Example::

        @async_retry(attempts=5)
        async def fragile_call():
            ...
    """

    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(fn)
        async def wrapped(*args: Any, **kwargs: Any) -> T:
            for attempt in range(1, attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except retry_exceptions as err:  # type: ignore[misc]
                    if attempt == attempts:
                        logger.exception("Exceeded retry limit for %s", fn.__name__)
                        raise
                    delay = min(base_delay * 2 ** (attempt - 1), max_delay)
                    delay += random.random() * jitter * delay  # add jitter
                    logger.warning(
                        "Retrying %s (attempt %s/%s) in %.2fs – %s",
                        fn.__name__,
                        attempt,
                        attempts,
                        delay,
                        err,
                    )
                    await asyncio.sleep(delay)
            # Unreachable, but satisfies type checker
            raise RuntimeError("async_retry decorator logic error")

        return wrapped

    return decorator


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def to_b64(data: bytes) -> str:
    """Base64-encode *bytes* → *str* helper."""
    import base64

    return base64.b64encode(data).decode()
