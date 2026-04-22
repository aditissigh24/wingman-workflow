import time
from typing import Optional


class PipelineError(Exception):
    def __init__(self, step: str, message: str, partial_artifacts: Optional[dict] = None):
        self.step = step
        self.partial_artifacts = partial_artifacts or {}
        super().__init__(f"[{step}] {message}")


def poll_until_done(check_fn, interval=10, timeout=300):
    """
    Poll a function until it signals completion.

    check_fn: callable returning (is_done: bool, result, error_message)
    Returns the result when done.
    Raises PipelineError on error or TimeoutError on timeout.
    """
    elapsed = 0
    while elapsed < timeout:
        done, result, error = check_fn()
        if error:
            raise RuntimeError(f"Polling failed: {error}")
        if done:
            return result
        time.sleep(interval)
        elapsed += interval
    raise TimeoutError(f"Operation timed out after {timeout}s")
