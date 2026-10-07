"""Scope fixture: a class large enough to expose its methods, a decorated function."""
import functools


class RetryPolicy:
    """Decides how long to wait between attempts of a failing operation."""

    def __init__(self, base_delay, max_delay, attempts):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.attempts = attempts
        self.history = []

    def delay(self, attempt):
        """Exponential backoff, capped at the maximum delay."""
        value = self.base_delay * (2 ** attempt)
        if value > self.max_delay:
            value = self.max_delay
        self.history.append(value)
        return value

    async def run(self, operation):
        """Runs the operation until it succeeds or the attempts are spent."""
        last_error = None
        for attempt in range(self.attempts):
            try:
                return await operation()
            except Exception as error:  # noqa: BLE001
                last_error = error
                await self.sleep(self.delay(attempt))
        raise last_error

    async def sleep(self, seconds):
        """Waits; a separate method so tests can replace it."""
        import asyncio

        await asyncio.sleep(seconds)


@functools.lru_cache(maxsize=None)
def fibonacci(n):
    """The n-th Fibonacci number, memoized."""
    return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)
