"""Unified CLI error renderer."""

from __future__ import annotations

import socket
from typing import TYPE_CHECKING

from dagnam_contracts import ParamError
import pytest
import requests

from dagnam._core.exceptions import (
    APIError,
    ArchitectureValidationError,
    ArchitectureVersionNotFoundError,
    AuthError,
    CheckpointNotFoundError,
    ChecksumError,
    CodegenValidationError,
    DagnamError,
    DatasetNotFoundError,
    DeploymentNotFoundError,
    DeploymentStateError,
    DeploymentValidationError,
    HubModelNotFoundError,
    LROFailedError,
    LROTimeoutError,
    ModelNotFoundError,
    ProjectNotFoundError,
    QuotaExceededError,
    StreamError,
    TrainingJobNotFoundError,
    UploadError,
)
from dagnam.cli import errors as errors_mod
from dagnam.cli.common import DOCS_URL

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch

_API_URL = "http://localhost:8000"


@pytest.fixture(autouse=True)
def stable_error_env(monkeypatch: PytestMonkeyPatch) -> None:
    """Pin color off and the API URL so rendering is deterministic."""
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setattr("dagnam._core.auth._api_url", None)
    monkeypatch.setenv("DAGNAM_API_URL", _API_URL)


def _chained(cause: BaseException) -> APIError:
    """Build an APIError the way the transport wrap sites do (``from cause``)."""
    try:
        raise APIError(0, f"Connection failed: {cause}") from cause
    except APIError as exc:
        return exc


# ---------------------------------------------------------------- color gating


class _FakeTty:
    def isatty(self) -> bool:
        return True


class _FakePipe:
    def isatty(self) -> bool:
        return False


def test_color_enabled_force_color_wins() -> None:
    assert errors_mod.color_enabled(_FakePipe(), environ={"FORCE_COLOR": "1"}) is True


def test_color_enabled_no_color_opts_out() -> None:
    assert errors_mod.color_enabled(_FakeTty(), environ={"NO_COLOR": ""}) is False


def test_color_enabled_dumb_term_opts_out() -> None:
    assert errors_mod.color_enabled(_FakeTty(), environ={"TERM": "dumb"}) is False


def test_color_enabled_requires_a_tty() -> None:
    assert errors_mod.color_enabled(_FakePipe(), environ={}) is False
    assert errors_mod.color_enabled(object(), environ={}) is False  # no isatty at all


def test_color_enabled_posix_tty_is_on() -> None:
    assert errors_mod.color_enabled(_FakeTty(), platform="linux", environ={}) is True


def test_color_enabled_windows_needs_a_capable_host() -> None:
    assert errors_mod.color_enabled(_FakeTty(), platform="win32", environ={}) is False
    assert (
        errors_mod.color_enabled(_FakeTty(), platform="win32", environ={"WT_SESSION": "x"}) is True
    )


def test_render_error_colors_when_forced(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setenv("FORCE_COLOR", "1")
    out = errors_mod.render_error(AuthError("bad key"))
    assert "\x1b[1;31mError:\x1b[0m" in out
    assert "\x1b[36mdagnam login\x1b[0m" in out


def test_styled_error_label_plain_without_tty() -> None:
    assert errors_mod.styled_error_label() == "Error:"


# ---------------------------------------------------------------- report layout


def test_render_report_aligns_fields_and_tries() -> None:
    report = errors_mod.ErrorReport(
        title="something broke",
        fields=[("URL", "http://x"), ("Cause", "refused")],
        bullets=["first problem", "second problem"],
        tries=[("dagnam whoami", "check identity"), ("", "Then retry.")],
    )
    out = errors_mod.render_report(report, color=False)
    assert out.splitlines() == [
        "Error: something broke",
        "",
        "    URL     http://x",
        "    Cause   refused",
        "",
        "    - first problem",
        "    - second problem",
        "",
        "Try:",
        "    dagnam whoami   check identity",
        "    Then retry.",
        "",
        f"Docs: {DOCS_URL}",
    ]


def test_render_report_sanitizes_server_supplied_escapes() -> None:
    # A hostile backend error body (the Detail field) must not reach the TTY as
    # a live escape; title and bullets are sanitised too.
    report = errors_mod.ErrorReport(
        title="\x1b]0;pwned\x07broke",
        fields=[("Detail", "\x1b[31mred\x9b2J")],
        bullets=["\x1bhostile bullet"],
    )
    out = errors_mod.render_report(report, color=False)
    assert "\x1b" not in out
    assert "\x9b" not in out
    assert "\x07" not in out
    assert "broke" in out
    assert "red" in out


def test_render_message_with_and_without_hint() -> None:
    with_hint = errors_mod.render_message("bad input", hint="Fix the JSON and retry.")
    assert "Error: bad input" in with_hint
    assert "Try:" in with_hint
    assert "Fix the JSON and retry." in with_hint
    without = errors_mod.render_message("bad input")
    assert "Try:" not in without
    assert DOCS_URL in without


# ---------------------------------------------------------------- transport


def test_connection_refused_is_classified() -> None:
    cause = requests.ConnectionError(ConnectionRefusedError(10061, "refused"))
    out = errors_mod.render_error(_chained(cause))
    assert "Error: cannot connect to the Dagnam API" in out
    assert _API_URL in out
    assert "connection refused" in out
    assert "dagnam whoami" in out
    assert "HTTPConnectionPool" not in out


def test_dns_failure_is_classified() -> None:
    cause = requests.ConnectionError(socket.gaierror(11001, "getaddrinfo failed"))
    out = errors_mod.render_error(_chained(cause))
    assert "host name could not be resolved" in out


def test_ssl_failure_is_classified() -> None:
    out = errors_mod.render_error(_chained(requests.exceptions.SSLError("bad handshake")))
    assert "TLS/SSL handshake failed" in out
    assert "correct https endpoint" in out


def test_timeout_is_classified() -> None:
    out = errors_mod.render_error(_chained(requests.Timeout("read timed out")))
    assert "Error: the Dagnam API did not respond in time" in out
    assert _API_URL in out


def test_builtin_timeout_in_chain_is_classified() -> None:
    out = errors_mod.render_error(_chained(requests.ConnectionError(TimeoutError("slow"))))
    assert "did not respond in time" in out


def test_unclassified_transport_falls_back() -> None:
    out = errors_mod.render_error(APIError(0, "Connection failed: weird"))
    assert "the connection could not be established" in out


def test_urllib3_reason_attribute_is_followed() -> None:
    class FakeRetryError(Exception):
        def __init__(self, reason: BaseException) -> None:
            super().__init__("retries exceeded")
            self.reason = reason

    cause = requests.ConnectionError(FakeRetryError(ConnectionRefusedError("refused")))
    out = errors_mod.render_error(_chained(cause))
    assert "connection refused" in out


def test_cause_walk_is_cycle_safe() -> None:
    first = APIError(0, "Connection failed: loop")
    second = Exception("back-link")
    first.__cause__ = second
    second.__cause__ = first
    out = errors_mod.render_error(first)
    assert "could not be established" in out


def test_cause_walk_is_depth_limited() -> None:
    root = APIError(0, "Connection failed: deep")
    current: BaseException = root
    for _ in range(20):
        nxt = Exception("layer")
        current.__cause__ = nxt
        current = nxt
    current.__cause__ = ConnectionRefusedError("too deep to reach")
    out = errors_mod.render_error(root)
    assert "could not be established" in out  # refused cause was beyond the cap


# ---------------------------------------------------------------- HTTP statuses


def test_http_403_permission_denied() -> None:
    out = errors_mod.render_error(APIError(403, "admin only"))
    assert "Error: permission denied (HTTP 403)" in out
    assert "admin only" in out


def test_http_429_rate_limited() -> None:
    out = errors_mod.render_error(APIError(429, "slow down"))
    assert "rate limited by the API (HTTP 429)" in out


def test_http_5xx_server_fault() -> None:
    out = errors_mod.render_error(APIError(503, "unavailable"))
    assert "internal error (HTTP 503)" in out
    assert "not caused by your request" in out


def test_http_422_rejected() -> None:
    out = errors_mod.render_error(APIError(422, "bad payload"))
    assert "rejected by the API (HTTP 422)" in out


def test_http_other_status_generic() -> None:
    out = errors_mod.render_error(APIError(418, ""))
    assert "the API returned an error (HTTP 418)" in out
    assert "Detail" not in out  # empty detail renders no field


# ---------------------------------------------------------------- typed errors


def test_auth_error_suggests_login() -> None:
    out = errors_mod.render_error(AuthError("No API key found."))
    assert "Error: authentication failed" in out
    assert "dagnam login" in out
    assert "Not logged in yet?" in out


def test_quota_error_suggests_usage() -> None:
    out = errors_mod.render_error(QuotaExceededError("Storage quota exceeded"))
    assert "Error: plan limit reached" in out
    assert "dagnam usage" in out


def test_architecture_validation_lists_problems() -> None:
    errors = [
        ParamError(type="param", message="filters must be >= 1", node_id="n1", severity="error"),
        ParamError(type="param", message="axis out of range", node_id="n2", severity="error"),
    ]
    out = errors_mod.render_error(ArchitectureValidationError(errors))
    assert "architecture validation failed" in out
    assert "- filters must be >= 1" in out
    assert "- axis out of range" in out


@pytest.mark.parametrize("exc_type", [DeploymentValidationError, CodegenValidationError])
def test_validation_errors_show_detail(exc_type: type[DagnamError]) -> None:
    out = errors_mod.render_error(exc_type("field X is invalid"))
    assert "failed validation" in out
    assert "field X is invalid" in out


def test_deployment_state_error() -> None:
    out = errors_mod.render_error(DeploymentStateError("cannot stop a stopped deployment"))
    assert "current state" in out
    assert "dagnam deployments get <id>" in out


@pytest.mark.parametrize(
    ("exc", "list_command"),
    [
        (ProjectNotFoundError("p1"), "dagnam projects list"),
        (ArchitectureVersionNotFoundError("v1"), "dagnam projects versions list <project-id>"),
        (DatasetNotFoundError("d1"), "dagnam dataset list"),
        (TrainingJobNotFoundError("t1"), "dagnam training list"),
        (DeploymentNotFoundError("dep1"), "dagnam deployments list"),
        (HubModelNotFoundError("h1"), "dagnam hub list"),
        (CheckpointNotFoundError("c1"), "dagnam checkpoint list <job-id>"),
        (ModelNotFoundError("md1"), "dagnam models list"),
    ],
)
def test_not_found_errors_suggest_list_command(exc: DagnamError, list_command: str) -> None:
    out = errors_mod.render_error(exc)
    assert "not found" in out
    assert list_command in out


def test_stream_error() -> None:
    out = errors_mod.render_error(StreamError("SSE dropped"))
    assert "live event stream failed" in out


def test_lro_timeout_error() -> None:
    out = errors_mod.render_error(LROTimeoutError("gave up after 300s"))
    assert "did not finish within the timeout" in out


def test_lro_failed_error() -> None:
    out = errors_mod.render_error(LROFailedError("failed", "exploded"))
    assert "failed on the server" in out
    assert "exploded" in out


def test_checksum_error_suggests_cache_clear() -> None:
    out = errors_mod.render_error(ChecksumError("sha mismatch"))
    assert "checksum verification" in out
    assert "dagnam cache clear" in out


def test_upload_error() -> None:
    out = errors_mod.render_error(UploadError("connection dropped mid-upload"))
    assert "Error: the upload failed" in out


def test_plain_dagnam_error_falls_back_to_message() -> None:
    out = errors_mod.render_error(DagnamError("something domain-specific"))
    assert "Error: something domain-specific" in out
    assert DOCS_URL in out


def test_unexpected_error_shows_type_and_redacts_message() -> None:
    out = errors_mod.render_error(ValueError("internal detail"))
    assert "unexpected error (ValueError)" in out
    assert "--debug" in out
    assert "internal detail" not in out
