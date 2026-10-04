"""The platform's contract, read once before anything is uploaded, and judged against this install's."""

from __future__ import annotations

import pytest
from tests.audit._platform import FakePlatform

from dagnam._core.exceptions import APIError, AuthError, ResponseError
from dagnam._types import JsonValue
from dagnam.audit.preflight import (
    PLATFORM_TOO_OLD,
    UPGRADE,
    check_platform,
    release,
    version_gap,
)

OURS = "0.4.2"


@pytest.fixture(autouse=True)
def installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dagnam.audit.preflight.version", lambda _distribution: OURS)


@pytest.mark.parametrize(
    ("text", "parsed"),
    [
        ("0.4.2", (0, 4, 2)),
        ("0.4", (0, 4, 0)),
        ("1.12.30rc1", (1, 12, 30)),
        ("v0.4.2", None),
        ("unknown", None),
        ("", None),
        (None, None),
        (7, None),
    ],
)
def test_a_release_is_read_from_the_front_of_the_string_or_not_at_all(
    text: object, parsed: tuple[int, int, int] | None
) -> None:
    assert release(text) == parsed


@pytest.mark.parametrize("contracts", ["0.4.2", "0.4.0", "0.4.1", "0.5.0rc1", "1.0.0", "0.4"])
def test_a_platform_at_or_ahead_by_minor_or_behind_by_patch_passes_without_a_refusal(
    platform: FakePlatform, contracts: str
) -> None:
    platform.contracts = contracts
    check = check_platform(platform)
    assert check.refusal is None
    assert check.contracts == contracts


@pytest.mark.parametrize("contracts", ["0.4.3", "0.4.99"])
def test_a_newer_patch_on_the_same_minor_is_a_warning_that_names_the_upgrade(
    platform: FakePlatform, contracts: str
) -> None:
    platform.contracts = contracts
    check = check_platform(platform)
    assert check.refusal is None
    assert check.warning is not None
    assert f"dagnam-contracts {contracts} and this install {OURS}" in check.warning
    assert UPGRADE == "pip install -U dagnam-contracts"
    assert f"`{UPGRADE}`" in check.warning


@pytest.mark.parametrize("contracts", ["0.4.2", "0.4.1", "0.5.0", "0.5.9", "1.0.0"])
def test_nothing_else_is_said_before_the_run(platform: FakePlatform, contracts: str) -> None:
    platform.contracts = contracts
    assert check_platform(platform).warning is None


@pytest.mark.parametrize("contracts", ["0.3.9", "0.3.0", "0.0.1"])
def test_a_platform_a_minor_behind_is_refused_with_both_versions(
    platform: FakePlatform, contracts: str
) -> None:
    platform.contracts = contracts
    check = check_platform(platform)
    assert check.contracts == contracts
    assert check.refusal == (
        f"{PLATFORM_TOO_OLD}: the platform at https://x runs dagnam-contracts {contracts} and"
        f" this SDK runs {OURS}. The platform has not been upgraded for this SDK yet; nothing"
        " was uploaded or spent by this run. Run again once it is."
    )


@pytest.mark.parametrize("contracts", [None, "", "unknown", "v0.4.2", 4, ["0.4.2"]])
def test_a_platform_that_does_not_say_is_refused(
    platform: FakePlatform, contracts: JsonValue
) -> None:
    platform.contracts = contracts
    check = check_platform(platform)
    assert check.contracts is None
    assert check.refusal is not None
    assert "does not report its dagnam-contracts version" in check.refusal


@pytest.mark.parametrize(
    "error", [APIError(404, "Not Found"), ResponseError(0, "Expected JSON object, got list")]
)
def test_a_route_that_is_missing_or_answers_something_else_is_refused(
    platform: FakePlatform, error: APIError
) -> None:
    platform.build_error = error
    check = check_platform(platform)
    assert check.refusal is not None
    assert "does not report its dagnam-contracts version" in check.refusal


@pytest.mark.parametrize(
    "error",
    [
        APIError(0, "Request failed: connection refused"),
        APIError(500, "boom"),
        APIError(503, "Service Unavailable"),
        APIError(429, "slow down"),
    ],
)
def test_a_request_that_fails_proves_nothing_and_the_run_goes_on(
    platform: FakePlatform, error: APIError
) -> None:
    platform.build_error = error
    check = check_platform(platform)
    assert (check.refusal, check.warning, check.contracts) == (None, None, None)


def test_an_account_error_is_the_runs_to_report_not_swallowed(platform: FakePlatform) -> None:
    platform.build_error = AuthError("expired key")
    with pytest.raises(AuthError):
        check_platform(platform)


@pytest.mark.parametrize(
    ("theirs", "says"),
    [
        (None, ""),
        ("unknown", ""),
        (OURS, ""),
        (
            "0.4.3",
            f" The platform runs dagnam-contracts 0.4.3, this install {OURS}: `{UPGRADE}`, then scan again.",
        ),
        (
            "0.5.0",
            f" The platform runs dagnam-contracts 0.5.0, this install {OURS}: `{UPGRADE}`, then scan again.",
        ),
        (
            "0.4.1",
            f" The platform runs dagnam-contracts 0.4.1, older than this install's {OURS}: wait for it.",
        ),
        (
            "0.3.0",
            f" The platform runs dagnam-contracts 0.3.0, older than this install's {OURS}: wait for it.",
        ),
    ],
)
def test_a_version_gap_says_which_side_is_newer_and_what_to_do(
    theirs: str | None, says: str
) -> None:
    assert version_gap(theirs, OURS) == says


def test_an_install_whose_version_cannot_be_read_is_never_told_about_a_gap() -> None:
    assert version_gap("0.4.3", "weird") == ""
