"""Long-Running Operation helper.

A small, generic polling primitive used by Phase 4 endpoints whose completion
is signalled by a resource-state transition (``deployments.create``,
``deployments.scale``, ``deployments.rollback``, ``codegen.generate``,
``datasets.upload_from_url``, …).

The shape follows GCP/k8s LROs:

    op = LongRunningOperation(poll=lambda: client.get_deployment(dep_id),
                              success_states={"running"})
    op.wait(timeout=300)      # blocks with exponential backoff
    dep = op.result()         # raises LROFailedError on terminal failure

Fire-and-forget callers can ignore the object; there are no background
threads.
"""

from __future__ import annotations

from collections.abc import Iterable
import time
from typing import Callable, FrozenSet, Optional

from dagnam._core.exceptions import APIError, LROFailedError, LROTimeoutError
from dagnam._types import JsonMapping

DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_POLL_MIN = 2.0
DEFAULT_POLL_MAX = 10.0

_DEFAULT_FAILURE_STATES: FrozenSet[str] = frozenset({"failed"})

# API errors worth retrying mid-wait: the client maps a network failure to
# status 0, a rate limit to 429, and a server hiccup to 5xx. A 4xx (404/403/…)
# is a real, non-transient error and must surface immediately.
_TRANSIENT_STATUS: FrozenSet[int] = frozenset({0, 429, 500, 502, 503, 504})


def _freeze(values: Optional[Iterable[str]]) -> FrozenSet[str]:
    return frozenset(values or ())


def freeze(values: Optional[Iterable[str]]) -> FrozenSet[str]:
    return _freeze(values)


class LongRunningOperation:
    """Polls a provider function until a terminal state is reached.

    ``poll`` must return the latest resource payload (a dict).  Its
    ``state_key`` field — ``"status"`` by default — is compared against
    ``success_states`` and ``failure_states``.
    """

    def __init__(
        self,
        *,
        poll: Callable[[], JsonMapping],
        success_states: Iterable[str],
        failure_states: Iterable[str] = _DEFAULT_FAILURE_STATES,
        state_key: str = "status",
        error_key: str = "error_message",
        name: str = "operation",
        initial: Optional[JsonMapping] = None,
        poll_min: float = DEFAULT_POLL_MIN,
        poll_max: float = DEFAULT_POLL_MAX,
    ) -> None:
        self._poll = poll
        self._success: FrozenSet[str] = _freeze(success_states)
        self._failure: FrozenSet[str] = _freeze(failure_states)
        if not self._success:
            raise ValueError("LongRunningOperation requires at least one success_state")
        self._state_key = state_key
        self.error_key = error_key
        self._name = name
        self._latest: Optional[JsonMapping] = initial
        self._poll_min = max(0.1, float(poll_min))
        self._poll_max = max(self._poll_min, float(poll_max))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    def configure_polling(self, min_interval: float, max_interval: float) -> None:
        self._poll_min = min_interval
        self._poll_max = max(min_interval, max_interval)

    def initial(self) -> Optional[JsonMapping]:
        """Payload captured at construction (if object) — never re-polled."""
        return self._latest

    def status(self) -> JsonMapping:
        """Force one poll and return the latest payload."""
        self._latest = self._poll()
        return self._latest

    def _current_state(self, payload: JsonMapping) -> str:
        value = payload.get(self._state_key)
        return str(value) if value is not None else ""

    def _poll_resilient(
        self,
        deadline: float,
        *,
        sleep: Callable[[float], None],
        now: Callable[[], float],
    ) -> JsonMapping:
        """Poll once, retrying transient API errors until the deadline.

        Transient means network / 429 / 5xx. A non-transient error (e.g. 404)
        propagates immediately; a transient error that persists past the
        deadline is re-raised so the caller sees the real cause rather than a
        bare timeout.
        """
        attempt_delay = self._poll_min
        while True:
            try:
                return self._poll()
            except APIError as exc:
                if exc.status_code not in _TRANSIENT_STATUS:
                    raise
                remaining = deadline - now()
                if remaining <= 0:
                    raise
                sleep(min(attempt_delay, remaining))
                attempt_delay = min(self._poll_max, attempt_delay * 2)

    def done(self) -> bool:
        """True if the most recent payload is in a terminal state.

        Does not poll — inspect ``initial()`` / ``status()`` first if you
        want a fresh read.
        """
        if self._latest is None:
            return False
        state = self._current_state(self._latest)
        return state in self._success or state in self._failure

    # ------------------------------------------------------------------
    # Blocking wait
    # ------------------------------------------------------------------

    def wait(
        self,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        *,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
    ) -> LongRunningOperation:
        """Block until the operation reaches a terminal state.

        Polls with exponential backoff from ``poll_min`` to ``poll_max``
        seconds.  Raises :class:`LROTimeoutError` if the timeout elapses
        before completion.  Returns ``self`` so calls can chain::

            dep = op.wait(timeout=300).result()
        """
        deadline = now() + float(timeout)
        delay = self._poll_min

        # First read — if already terminal, return immediately.
        payload = self._poll_resilient(deadline, sleep=sleep, now=now)
        self._latest = payload
        state = self._current_state(payload)
        if state in self._success or state in self._failure:
            return self

        while now() < deadline:
            remaining = deadline - now()
            sleep(min(delay, max(0.0, remaining)))
            if now() >= deadline:
                break

            payload = self._poll_resilient(deadline, sleep=sleep, now=now)
            self._latest = payload
            state = self._current_state(payload)
            if state in self._success or state in self._failure:
                return self

            delay = min(self._poll_max, delay * 2)

        raise LROTimeoutError(
            f"{self._name} did not reach a terminal state within {timeout:.1f}s "
            f"(last state: {state or 'unknown'!r})"
        )

    # ------------------------------------------------------------------
    # Result extraction
    # ------------------------------------------------------------------

    def result(self) -> JsonMapping:
        """Return the final payload, or raise if the operation failed.

        Callers are expected to have ``wait()``-ed first (or otherwise
        observed a terminal state via ``status()``).  If the operation has
        not yet reached a terminal state, ``LROTimeoutError`` is raised.
        """
        if self._latest is None:
            raise LROTimeoutError(
                f"{self._name} has not been polled yet — call wait() or status() first"
            )
        state = self._current_state(self._latest)
        if state in self._failure:
            detail = self._latest.get(self.error_key)
            raise LROFailedError(state, detail if isinstance(detail, str) else None)
        if state in self._success:
            return self._latest
        raise LROTimeoutError(f"{self._name} is still in non-terminal state {state!r}")


__all__ = ["LongRunningOperation"]
