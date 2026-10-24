"""Record ``openai`` chat completions and Responses calls (sync, async and streaming).

``install`` wraps ``create`` and ``parse`` on the SDK's resource classes; it imports ``openai``
lazily and only there (a missing SDK, or a version without one of the resources, is simply not
patched), so ``import dagnam_record`` never needs the SDK. The wrapper records the call's JSON
body parameters (taken before the call, so later changes to the caller's data do not alter the
record) and the response body, and nothing else: no header, key or organisation id is read.

The ``stream()`` helpers call the wrapped ``create`` and so are recorded through it. Not
recorded in v1 (the call itself is unaffected): ``with_raw_response`` and
``with_streaming_response`` (their result is not a model; counted as ``dropped_error``).

An interrupted or cancelled call is not a failed call: it is counted as ``dropped_partial`` and
nothing is written.
"""

from __future__ import annotations

from typing import Any, Callable, List, Tuple, Union

from dagnam_record._openai_assemble import ChatAssembler, ResponsesAssembler
from dagnam_record._openai_body import request_body, response_body
from dagnam_record._patch import Patcher, Plan
from dagnam_record._recorder import Recorder

PROVIDER = "openai"
_METHODS = ("create", "parse")


def _assembler(endpoint: str) -> Union[ChatAssembler, ResponsesAssembler]:
    return ChatAssembler() if endpoint == "chat.completions" else ResponsesAssembler()


_PATCHER = Patcher(PROVIDER, request_body, response_body, _assembler)


def _targets() -> List[Tuple[Any, str, bool]]:
    """The SDK classes this version has: ``(class, endpoint, is_async)``."""
    found: List[Tuple[Any, str, bool]] = []
    # Lazy, runtime-optional imports: ``import dagnam_record`` must not need the SDK.
    try:
        import openai.resources.chat.completions as chat
    except ImportError:
        pass
    else:
        found += [
            (getattr(chat, "Completions", None), "chat.completions", False),
            (getattr(chat, "AsyncCompletions", None), "chat.completions", True),
        ]
    try:
        import openai.resources.beta as beta
    except ImportError:
        pass
    else:  # older SDKs kept ``parse`` in a separate beta class; newer ones have no such module
        legacy = getattr(getattr(beta, "chat", None), "completions", None)
        found += [
            (getattr(legacy, "Completions", None), "chat.completions", False),
            (getattr(legacy, "AsyncCompletions", None), "chat.completions", True),
        ]
    try:
        import openai.resources.responses as responses
    except ImportError:
        pass
    else:
        found += [
            (getattr(responses, "Responses", None), "responses", False),
            (getattr(responses, "AsyncResponses", None), "responses", True),
        ]
    return found


def _plan() -> List[Plan]:
    def builder(
        endpoint: str, is_async: bool
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return lambda original: _PATCHER.wrap(original, endpoint, is_async=is_async)

    return [
        (cls, method, builder(endpoint, is_async))
        for cls, endpoint, is_async in _targets()
        for method in _METHODS
    ]


def install(recorder: Recorder) -> bool:
    """Record through ``recorder``; True when at least one method is (now) patched.

    Safe to call again: a method is wrapped once, and the newest recorder is used. Returns False
    under the kill switch and when the SDK (or all its known resources) is not there.
    """
    return _PATCHER.install(recorder, _plan())


def uninstall() -> None:
    """Put the SDK's own methods back and stop recording (used by tests and ``dagnam_record.uninstall``)."""
    _PATCHER.uninstall()
