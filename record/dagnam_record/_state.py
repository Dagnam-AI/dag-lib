"""The public API: ``install``, ``record_call`` and ``uninstall``, over one global recorder.

There is one ``Writer`` and one ``Recorder`` per process, kept in ``STATE`` under a lock. The
options and the second-``install`` behaviour are documented on ``install``.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Optional, Union

from dagnam_record import _anthropic, _openai
from dagnam_record._files import DEFAULT_DIR, prune, resolve
from dagnam_record._recorder import KILL_SWITCH, Recorder, recording_disabled, to_plain
from dagnam_record._redact import Option, check, resolve as resolve_redact
from dagnam_record._writer import Writer

LOGGER = logging.getLogger("dagnam_record")
MAX_WINDOW_DAYS = 365
_LOOKS_LIKE_OFF = frozenset({"false", "off", "no"})


@dataclass(frozen=True)
class Config:
    """The checked options of one ``install`` call."""

    window_days: int
    directory: str
    sample: float
    no_raw: bool
    redact: Option


@dataclass(frozen=True)
class Installed:
    """The recorder that ``install`` set up, with the options it was built from."""

    config: Config
    writer: Writer
    recorder: Recorder


class State:
    """What is installed (if anything), behind one lock."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.installed: Optional[Installed] = None
        self.warned = False  # ``record_call`` said once that nothing is installed


STATE = State()


def after_fork_in_child() -> None:
    """A lock another thread held at the fork stays locked in the child: make a free one."""
    STATE.lock = threading.Lock()


getattr(os, "register_at_fork", lambda **hooks: None)(after_in_child=after_fork_in_child)


def _config(
    window_days: object, directory: object, sample: object, no_raw: object, redact: object
) -> Config:
    if (
        not isinstance(window_days, int)
        or isinstance(window_days, bool)
        or not 1 <= window_days <= MAX_WINDOW_DAYS
    ):
        raise ValueError(f"window_days must be a whole number from 1 to {MAX_WINDOW_DAYS}")
    if isinstance(directory, os.PathLike):
        directory = os.fspath(directory)
    if not isinstance(directory, str) or not directory or "\0" in directory:
        raise ValueError("dir must be a non-empty path")
    if isinstance(sample, bool) or not isinstance(sample, (int, float)) or not 0 < sample <= 1:
        raise ValueError("sample must be a number above 0 and at most 1")
    if not isinstance(no_raw, bool):
        raise ValueError("no_raw must be True or False")
    return Config(window_days, resolve(directory), float(sample), no_raw, check(redact))


def _warn_about_lookalikes() -> None:
    value = os.environ.get(KILL_SWITCH)
    if value is not None and value.strip().lower() in _LOOKS_LIKE_OFF:
        LOGGER.warning(
            "dagnam_record: %s=%r does not switch the recorder off; set it to 0 to do so",
            KILL_SWITCH,
            value,
        )


def _configure(config: Config, previous: Optional[Installed]) -> Installed:
    """Build the writer and recorder for ``config`` after stopping the previous ones."""
    redactor = resolve_redact(config.redact)  # may raise ImportError: nothing has changed yet
    if previous is not None:
        previous.writer.close()  # writes its queue and saves its counts, which the new one reads
    writer = Writer(config.directory, window_days=config.window_days)
    prune(config.directory, config.window_days, time.time())
    recorder = Recorder(writer, no_raw=config.no_raw, sample=config.sample, redact=redactor)
    return Installed(config, writer, recorder)


def install(
    *,
    window_days: int = 14,
    dir: Union[str, os.PathLike[str]] = DEFAULT_DIR,
    sample: float = 1.0,
    no_raw: bool = False,
    redact: Union[Callable[[str], str], str, None] = None,
) -> bool:
    """Start recording LLM calls; True when at least one SDK was patched.

    Options (keyword-only; a value outside its range raises ``ValueError`` and changes nothing):

    * ``window_days``: whole days from 1 to 365, default 14. The first ``install`` in a directory
      starts the window; when it ends the recorder stops and logs once. The recorder's own daily
      files dated more than ``window_days`` days ago are deleted whenever the recorder is set up
      (the first ``install``, or one with new options).
    * ``dir``: where the files go, default ``~/.dagnam/traces`` (created ``0700``; files ``0600``).
    * ``sample``: the share of calls kept, above 0 and at most 1, chosen by a hash of each call's
      trace id (see ``_recorder.sampled``).
    * ``no_raw``: write the provider-neutral ``norm`` only, never the request and response bodies.
    * ``redact``: ``None``, a function from text to text (see ``_redact``), or ``"auto"``, which
      needs the optional extra ``dagnam-record[redact]`` (a missing one raises ``ImportError``).

    The kill switch ``DAGNAM_RECORD=0`` is read when ``install`` runs: ``install`` then does
    nothing, validates nothing, writes nothing, and returns ``False``. Only the exact value ``0``
    counts.

    A missing SDK is never an error: ``False`` means no SDK (``openai``, ``anthropic``) was
    patched, or the kill switch is on. The recorder is set up anyway, so ``record_call`` works for
    frameworks that bypass the SDKs. Calling ``install`` again is safe, from any thread: with the
    same options it only patches an SDK that has appeared since (one is never wrapped twice); with
    different options the recorder is reconfigured: the old writer writes what it has queued and
    stops, then a new one starts (the window carries over in the same directory). Lines submitted
    in the instant between the two are counted as dropped. Blocking work is limited to this call
    (a directory and a thread are made); recording itself never blocks.
    """
    if recording_disabled():
        return False
    _warn_about_lookalikes()
    config = _config(window_days, dir, sample, no_raw, redact)
    with STATE.lock:
        installed = STATE.installed
        if installed is None or installed.config != config:
            installed = STATE.installed = _configure(config, installed)
        patched = [_openai.install(installed.recorder), _anthropic.install(installed.recorder)]
    return any(patched)


def _record(
    recorder: Recorder,
    provider: str,
    endpoint: str,
    request: Dict[str, Any],
    response: Dict[str, Any],
    status: int,
    latency_ms: float,
) -> None:
    recorder.record(
        provider,
        endpoint,
        to_plain(request),
        to_plain(response),
        status=status,
        latency_ms=latency_ms,
    )


def record_call(
    *,
    provider: str,
    request: Dict[str, Any],
    response: Dict[str, Any],
    latency_ms: float = 0.0,
    endpoint: str = "custom",
    status: int = 200,
) -> None:
    """Record one call that the SDK patches cannot see. Never raises; a bad call is counted.

    ``provider`` is ``"openai"``, ``"anthropic"`` or ``"other"``, ``endpoint`` is
    ``"chat.completions"``, ``"responses"``, ``"messages"`` or ``"custom"``; ``request`` and
    ``response`` are the JSON bodies (SDK objects are converted). It goes through the same
    sampling, redaction, size limit and writer as a patched call. Before ``install`` (or after
    ``uninstall``) it records nothing and logs one warning, unless the kill switch is on.
    """
    installed = STATE.installed
    if installed is None:
        if not (STATE.warned or recording_disabled()):
            STATE.warned = True
            LOGGER.warning(
                "dagnam_record: record_call() ran without install(); nothing is recorded"
            )
        return
    recorder = installed.recorder
    recorder.fail_open(_record, recorder, provider, endpoint, request, response, status, latency_ms)


def uninstall() -> None:
    """Put the SDKs back, stop recording and close the writer (it writes what is queued first)."""
    with STATE.lock:
        _openai.uninstall()
        _anthropic.uninstall()
        installed, STATE.installed = STATE.installed, None
    if installed is not None:
        installed.writer.close()
