"""Unified CLI error presentation.

Every failure the CLI can hit funnels through one renderer so users always see
the same shape: a title line, aligned detail fields, actionable ``Try:``
suggestions, and the docs link. Rendering is registry-driven — a typed
exception maps to a builder, never a per-command branch — and transport
failures (backend down, DNS, TLS, timeouts) are classified by walking the
exception cause chain instead of leaking raw ``urllib3`` reprs.

All glyphs are plain ASCII so output is safe on legacy Windows code pages.
Minimal ANSI color is applied only when stderr is an interactive
terminal that advertises support; piped/captured output is always plain text.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import cache
import os
import sys
from typing import TYPE_CHECKING, cast

from dagnam.cli.common import DOCS_URL
from dagnam.cli.presentation import sanitize_terminal_text

if TYPE_CHECKING:
    from dagnam._core.exceptions import (
        APIError,
        ArchitectureValidationError,
        InsufficientCreditsError,
        TrainingStateError,
    )

_RESET = "\x1b[0m"
_ANSI = {
    "error": "\x1b[1;31m",  # bold red — the Error: label
    "bold": "\x1b[1m",  # section headers (Try:, Docs:)
    "dim": "\x1b[2m",  # field labels
    "command": "\x1b[36m",  # cyan — suggested commands
}

_INDENT = "    "
_GAP = 3  # spaces between an aligned label/command column and its text


def color_enabled(
    stream: object | None = None,
    *,
    platform: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Whether ANSI styling is safe for ``stream`` (default: stderr).

    ``FORCE_COLOR`` wins, then ``NO_COLOR``/``TERM=dumb`` opt out, then the
    stream must be a real TTY. On Windows, classic conhost may print escape
    codes literally, so color additionally requires a capable host to announce
    itself (Windows Terminal's ``WT_SESSION``, ``ANSICON``, or a ``TERM`` set
    by Git Bash / ConEmu). ``platform``/``environ`` are injectable for tests.
    """
    env = os.environ if environ is None else environ
    if env.get("FORCE_COLOR"):
        return True
    if env.get("NO_COLOR") is not None:
        return False
    if env.get("TERM") == "dumb":
        return False
    out = sys.stderr if stream is None else stream
    is_tty = getattr(out, "isatty", None)
    if is_tty is None or not is_tty():
        return False
    plat = sys.platform if platform is None else platform
    if plat == "win32":
        return bool(env.get("WT_SESSION") or env.get("ANSICON") or env.get("TERM"))
    return True


def _style(text: str, role: str, *, color: bool) -> str:
    if not color:
        return text
    return f"{_ANSI[role]}{text}{_RESET}"


def _cell(text: str, width: int, role: str, *, color: bool) -> str:
    """Style ``text`` and pad to ``width`` measured on the plain text."""
    return _style(text, role, color=color) + " " * (width - len(text))


@dataclass
class ErrorReport:
    """One renderable failure: title + optional details, bullets, suggestions."""

    title: str
    fields: list[tuple[str, str]] = field(default_factory=list)
    bullets: list[str] = field(default_factory=list)
    # (command, note) rows; an empty command renders the note as prose.
    tries: list[tuple[str, str]] = field(default_factory=list)


def render_report(report: ErrorReport, *, color: bool) -> str:
    """Render a report into the unified error block.

    Title, field values, and bullets can carry server-supplied text (e.g. the
    ``Detail`` field is the backend's error message), so they are sanitised of
    terminal control/escape sequences before reaching the TTY.
    """
    lines = [f"{_style('Error:', 'error', color=color)} {sanitize_terminal_text(report.title)}"]
    if report.fields:
        lines.append("")
        width = max(len(label) for label, _ in report.fields) + _GAP
        lines.extend(
            f"{_INDENT}{_cell(label, width, 'dim', color=color)}{sanitize_terminal_text(value)}"
            for label, value in report.fields
        )
    if report.bullets:
        lines.append("")
        lines.extend(f"{_INDENT}- {sanitize_terminal_text(bullet)}" for bullet in report.bullets)
    if report.tries:
        lines.append("")
        lines.append(_style("Try:", "bold", color=color))
        width = max((len(cmd) for cmd, _ in report.tries if cmd), default=0) + _GAP
        for cmd, note in report.tries:
            if cmd:
                lines.append(f"{_INDENT}{_cell(cmd, width, 'command', color=color)}{note}")
            else:
                lines.append(f"{_INDENT}{note}")
    lines.append("")
    lines.append(f"{_style('Docs:', 'bold', color=color)} {DOCS_URL}")
    return "\n".join(lines)


def render_error(exc: BaseException) -> str:
    """Render any exception into the unified error block."""
    return render_report(_report_for(exc), color=color_enabled())


def render_message(message: str, *, hint: str | None = None) -> str:
    """Render a contextual one-line error (the ``error()`` helper) in the frame."""
    tries = [("", hint)] if hint else []
    return render_report(ErrorReport(title=message, tries=tries), color=color_enabled())


def styled_error_label() -> str:
    """The ``Error:`` label alone, styled when stderr supports it (argparse use)."""
    return _style("Error:", "error", color=color_enabled())


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


def _iter_causes(exc: BaseException, *, limit: int = 12) -> Iterator[BaseException]:
    """Walk the full failure chain under ``exc``, bounded and cycle-safe.

    Besides ``__cause__``/``__context__``, this follows urllib3's
    ``MaxRetryError.reason`` attribute and exception args, because ``requests``
    wraps the underlying ``ConnectionRefusedError``/``socket.gaierror`` as an
    argument rather than a dunder link.
    """
    seen: set[int] = set()
    stack = [exc]
    while stack and limit > 0:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        limit -= 1
        yield current
        links = (current.__cause__, current.__context__, getattr(current, "reason", None))
        stack.extend(nxt for nxt in links if isinstance(nxt, BaseException))
        stack.extend(arg for arg in current.args if isinstance(arg, BaseException))


def _api_url() -> str:
    from dagnam._core.auth import get_api_url

    return get_api_url()


def _connect_tries() -> list[tuple[str, str]]:
    return [
        ("dagnam whoami", "check which server the CLI targets"),
        ("dagnam config list", "inspect api_url / DAGNAM_API_URL"),
        ("", "Check your connection and the configured URL, then retry."),
    ]


def _transport_report(exc: BaseException) -> ErrorReport:
    """Classify a status-0 APIError (no HTTP response) by its cause chain."""
    import socket

    import requests

    causes = list(_iter_causes(exc))

    def has(kind: type[BaseException] | tuple[type[BaseException], ...]) -> bool:
        return any(isinstance(cause, kind) for cause in causes)

    tries = _connect_tries()
    if has(requests.exceptions.SSLError):
        cause = "the TLS/SSL handshake failed"
        tries = [
            ("dagnam whoami", "check which server the CLI targets"),
            ("", "Make sure api_url points at the correct https endpoint, then retry."),
        ]
    elif has(socket.gaierror):
        cause = "the host name could not be resolved"
    elif has(ConnectionRefusedError):
        cause = "connection refused - nothing is listening at that address"
    elif has((requests.Timeout, TimeoutError)):
        return ErrorReport(
            title="the Dagnam API did not respond in time",
            fields=[("API URL", _api_url())],
            tries=[
                ("", "The service may be busy right now."),
                ("", "Wait a moment and retry."),
            ],
        )
    else:
        cause = "the connection could not be established"
    return ErrorReport(
        title="cannot connect to the Dagnam API",
        fields=[("API URL", _api_url()), ("Cause", cause)],
        tries=tries,
    )


def _http_report(exc: BaseException) -> ErrorReport:
    """Render an APIError that carries a real HTTP status code."""
    # The registry dispatches by isinstance, so exc IS an APIError here.
    api_exc = cast("APIError", exc)
    code = api_exc.status_code
    detail = api_exc.message
    fields = [("Detail", detail)] if detail else []
    if code == 403:
        return ErrorReport(
            title="permission denied (HTTP 403)",
            fields=fields,
            tries=[
                ("dagnam whoami", "confirm which account and key you are using"),
                ("", "Your plan or role may not allow this action."),
            ],
        )
    if code == 429:
        return ErrorReport(
            title="rate limited by the API (HTTP 429)",
            fields=fields,
            tries=[("", "Wait a moment, then retry.")],
        )
    if code >= 500:
        return ErrorReport(
            title=f"the Dagnam API had an internal error (HTTP {code})",
            fields=fields,
            tries=[
                ("", "This is a server-side problem, not caused by your request."),
                ("", "Retry shortly; if it persists, contact support."),
            ],
        )
    if code in (400, 422):
        return ErrorReport(
            title=f"the request was rejected by the API (HTTP {code})",
            fields=fields,
        )
    return ErrorReport(title=f"the API returned an error (HTTP {code})", fields=fields)


def _api_report(exc: BaseException) -> ErrorReport:
    if cast("APIError", exc).status_code == 0:
        return _transport_report(exc)
    return _http_report(exc)


def _auth_report(exc: BaseException) -> ErrorReport:
    return ErrorReport(
        title="authentication failed",
        fields=[("Detail", str(exc))],
        tries=[
            ("dagnam login", "sign in with a fresh API key"),
            ("dagnam whoami", "see which key and server are configured"),
            ("", "Not logged in yet? Create an API key in your account settings."),
        ],
    )


def _quota_report(exc: BaseException) -> ErrorReport:
    return ErrorReport(
        title="plan limit reached",
        fields=[("Detail", str(exc))],
        tries=[
            ("dagnam usage", "see your current usage and limits"),
            ("", "Free up resources or upgrade your plan to raise the limit."),
        ],
    )


SUPPORT_URL = "https://dagnam.ai/support"
_REQUEST_HINT = f"Ask for more credits at {SUPPORT_URL}"
_SMALLER_HINT = "Or make the run smaller and try again"


def _credit_hints(exc: InsufficientCreditsError) -> list[str]:
    """Hints for the platform's ``next_steps`` that its message does not already give.

    Only steps whose destination exists are translated: the support page for ``request_credits``
    (not for a caller who is not the account's owner, whose refusal carries no balance and says
    the owner must act), and the smaller-run sentence for ``reduce_job_size``. Any other token,
    and any advice the message already contains, is skipped.
    """
    said = exc.message.lower()
    hints: list[str] = []
    if (
        "request_credits" in exc.next_steps
        and exc.available_credits is not None
        and SUPPORT_URL not in said
    ):
        hints.append(_REQUEST_HINT)
    if "reduce_job_size" in exc.next_steps and "smaller" not in said:
        hints.append(_SMALLER_HINT)
    return hints


def _credits_report(exc: BaseException) -> ErrorReport:
    credits = cast("InsufficientCreditsError", exc)
    return ErrorReport(
        title="not enough credits",
        fields=[("Detail", str(exc))],
        tries=[("", hint) for hint in _credit_hints(credits)],
    )


def _not_found_report(list_command: str) -> Callable[[BaseException], ErrorReport]:
    def build(exc: BaseException) -> ErrorReport:
        return ErrorReport(
            title=str(exc),
            tries=[(list_command, "list the available items and check the id")],
        )

    return build


def _validation_report(exc: BaseException) -> ErrorReport:
    return ErrorReport(
        title="the request failed validation",
        fields=[("Detail", str(exc))],
    )


def _architecture_report(exc: BaseException) -> ErrorReport:
    # The registry dispatches by isinstance, so exc IS an ArchitectureValidationError.
    errors = cast("ArchitectureValidationError", exc).errors
    return ErrorReport(
        title="architecture validation failed",
        bullets=[error.message for error in errors],
        tries=[("", "Fix the listed parameters and retry.")],
    )


def _state_report(exc: BaseException) -> ErrorReport:
    return ErrorReport(
        title="action not allowed in the deployment's current state",
        fields=[("Detail", str(exc))],
        tries=[("dagnam deployments get <id>", "check the current lifecycle state")],
    )


_JOB_GET_TRY = ("dagnam training get <id>", "check the job's current status")
_TRAINING_STATE_TITLES: dict[str | None, tuple[str, tuple[str, str]]] = {
    "not_paused": ("the training job is not paused", _JOB_GET_TRY),
    "checkpoint_unavailable": (
        "the paused job's saved checkpoint cannot be read",
        ("dagnam checkpoint list <id>", "see which checkpoints are saved, then restore one"),
    ),
    "not_accepting_checkpoints": ("the training job is not accepting checkpoints", _JOB_GET_TRY),
}
_TRAINING_NOT_READY = (
    "the training job is not ready for that yet",
    ("", "Wait a few minutes, then retry."),
)


def _training_state_report(exc: BaseException) -> ErrorReport:
    state = cast("TrainingStateError", exc)
    title, hint = _TRAINING_STATE_TITLES.get(state.reason, _TRAINING_NOT_READY)
    return ErrorReport(title=title, fields=[("Detail", state.message)], tries=[hint])


def _stream_report(exc: BaseException) -> ErrorReport:
    return ErrorReport(
        title="the live event stream failed",
        fields=[("Detail", str(exc))],
        tries=[("", "Retry; streaming resumes from the server's current state.")],
    )


def _lro_timeout_report(exc: BaseException) -> ErrorReport:
    return ErrorReport(
        title="the operation did not finish within the timeout",
        fields=[("Detail", str(exc))],
        tries=[("", "It may still be running - check its status before retrying.")],
    )


def _lro_failed_report(exc: BaseException) -> ErrorReport:
    return ErrorReport(
        title="the operation failed on the server",
        fields=[("Detail", str(exc))],
    )


def _checksum_report(exc: BaseException) -> ErrorReport:
    return ErrorReport(
        title="a downloaded file failed checksum verification",
        fields=[("Detail", str(exc))],
        tries=[
            ("dagnam cache clear", "drop the corrupted cached copy"),
            ("", "Then retry the download."),
        ],
    )


def _upload_report(exc: BaseException) -> ErrorReport:
    return ErrorReport(
        title="the upload failed",
        fields=[("Detail", str(exc))],
        tries=[("", "Check your connection and retry.")],
    )


def _plain_report(exc: BaseException) -> ErrorReport:
    return ErrorReport(title=str(exc))


def _unexpected_report(exc: BaseException) -> ErrorReport:
    # Deliberately redacts the exception message: internals never leak without
    # an explicit opt-in.
    return ErrorReport(
        title=f"unexpected error ({type(exc).__name__})",
        tries=[("", "Rerun with --debug for the full traceback.")],
    )


@cache
def _registry() -> tuple[tuple[type[BaseException], Callable[[BaseException], ErrorReport]], ...]:
    """Ordered (type, builder) table — most specific first, fallbacks last."""
    from dagnam._core import exceptions as m

    return (
        (m.AuthError, _auth_report),
        (m.InsufficientCreditsError, _credits_report),
        (m.QuotaExceededError, _quota_report),
        (m.ArchitectureValidationError, _architecture_report),
        (m.DeploymentValidationError, _validation_report),
        (m.CodegenValidationError, _validation_report),
        (m.DeploymentStateError, _state_report),
        (m.TrainingStateError, _training_state_report),
        (m.ProjectNotFoundError, _not_found_report("dagnam projects list")),
        (
            m.ArchitectureVersionNotFoundError,
            _not_found_report("dagnam projects versions list <project-id>"),
        ),
        (m.DatasetNotFoundError, _not_found_report("dagnam dataset list")),
        (m.TrainingJobNotFoundError, _not_found_report("dagnam training list")),
        (m.DeploymentNotFoundError, _not_found_report("dagnam deployments list")),
        (m.HubModelNotFoundError, _not_found_report("dagnam hub list")),
        (m.CheckpointNotFoundError, _not_found_report("dagnam checkpoint list <job-id>")),
        (m.ModelNotFoundError, _not_found_report("dagnam models list")),
        (m.StreamError, _stream_report),
        (m.LROTimeoutError, _lro_timeout_report),
        (m.LROFailedError, _lro_failed_report),
        (m.ChecksumError, _checksum_report),
        (m.UploadError, _upload_report),
        (m.APIError, _api_report),
        (m.DagnamError, _plain_report),
    )


def _report_for(exc: BaseException) -> ErrorReport:
    for exc_type, build in _registry():
        if isinstance(exc, exc_type):
            return build(exc)
    return _unexpected_report(exc)
