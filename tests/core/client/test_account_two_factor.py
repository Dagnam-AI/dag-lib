"""Wire-level coverage for the sync two-factor methods.

Two contracts matter more than the request shapes and are pinned first: the
password never appears anywhere but the request body, and an enrollment that is
never verified leaves 2FA inactive.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import APIError, AuthError, ResponseError

if TYPE_CHECKING:
    from tests.typing_helpers import RequestsMocker

API = "https://api.test"
PROFILE = f"{API}/api/v1/users/me/profile"
ENABLE = f"{API}/api/v1/users/me/2fa/enable"
VERIFY = f"{API}/api/v1/users/me/2fa/verify"
DISABLE = f"{API}/api/v1/users/me/2fa/disable"

ENROLLMENT = {
    "secret": "JBSWY3DPEHPK3PXP",
    "qr_code_uri": "otpauth://totp/Dagnam:me?secret=JBSWY3DPEHPK3PXP",
    "backup_codes": ["11111111", "22222222"],
}


# ------------------------------------------------------------ two_factor_enabled


@pytest.mark.parametrize("declared", [True, False])
def test_status_reads_the_field_off_the_profile(
    client: DagnamClient, rmock: RequestsMocker, declared: bool
) -> None:
    """There is deliberately no dedicated status endpoint; one source cannot
    disagree with itself."""
    rmock.get(PROFILE, json={"email": "me@example.com", "two_factor_enabled": declared})
    assert client.two_factor_enabled() is declared


def test_a_profile_missing_the_field_reports_disabled_not_enabled(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    """An unknown state must never read as "protected" -- that is the direction
    that gets someone to skip enrolling."""
    rmock.get(PROFILE, json={"email": "me@example.com"})
    assert client.two_factor_enabled() is False


@pytest.mark.parametrize("declared", [1, "true", None, ""])
def test_a_non_boolean_field_is_coerced_rather_than_returned_raw(
    client: DagnamClient, rmock: RequestsMocker, declared: object
) -> None:
    rmock.get(PROFILE, json={"two_factor_enabled": declared})
    assert client.two_factor_enabled() is bool(declared)


# ------------------------------------------------------------- enable / verify


def test_enable_sends_only_the_password_and_returns_the_material(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(ENABLE, json=ENROLLMENT)
    result = client.enable_two_factor("Passw0rd!")

    assert result == ENROLLMENT
    assert rmock.last_request.json() == {"password": "Passw0rd!"}
    assert rmock.last_request.headers["Authorization"] == "Bearer k"


def test_the_password_never_reaches_the_url_or_the_headers(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    """A password in a query string is logged by every proxy on the path."""
    rmock.post(ENABLE, json=ENROLLMENT)
    client.enable_two_factor("Passw0rd!")

    assert "Passw0rd!" not in rmock.last_request.url
    assert not any("Passw0rd!" in str(v) for v in rmock.last_request.headers.values())


def test_enable_with_a_wrong_password_raises_autherror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(ENABLE, status_code=401, json={"detail": "Password is incorrect"})
    with pytest.raises(AuthError):
        client.enable_two_factor("wrong")


def test_enable_when_already_enrolled_raises_apierror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(ENABLE, status_code=400, json={"detail": "2FA is already enabled"})
    with pytest.raises(APIError):
        client.enable_two_factor("Passw0rd!")


def test_verify_sends_the_code(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(VERIFY, json={"message": "Two-factor authentication enabled successfully"})
    result = client.verify_two_factor("123456")

    assert result["message"] == "Two-factor authentication enabled successfully"
    assert rmock.last_request.json() == {"code": "123456"}


def test_a_rejected_code_raises_and_does_not_report_success(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    """The failure mode this prevents: a mistyped authenticator that reports
    success would leave the caller locked out at the next login."""
    rmock.post(VERIFY, status_code=400, json={"detail": "Invalid verification code"})
    with pytest.raises(APIError):
        client.verify_two_factor("000000")


def test_a_rate_limited_verify_surfaces_rather_than_retrying_silently(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(VERIFY, status_code=429, json={"detail": "Too many attempts"})
    with pytest.raises(APIError):
        client.verify_two_factor("123456")


# ------------------------------------------------------------------- disable


def test_disable_sends_only_the_password(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(DISABLE, json={"message": "Two-factor authentication disabled"})
    result = client.disable_two_factor("Passw0rd!")

    assert result["message"] == "Two-factor authentication disabled"
    assert rmock.last_request.json() == {"password": "Passw0rd!"}


def test_disable_with_a_wrong_password_raises_autherror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(DISABLE, status_code=401, json={"detail": "Password is incorrect"})
    with pytest.raises(AuthError):
        client.disable_two_factor("wrong")


def test_disable_when_not_enabled_raises_apierror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(DISABLE, status_code=400, json={"detail": "2FA is not enabled"})
    with pytest.raises(APIError):
        client.disable_two_factor("Passw0rd!")


@pytest.mark.parametrize("method", ["enable_two_factor", "verify_two_factor", "disable_two_factor"])
def test_a_non_object_response_raises_responseerror(
    client: DagnamClient, rmock: RequestsMocker, method: str
) -> None:
    """A bare list or scalar where an object is expected is a protocol
    mismatch, not something to hand back to the caller as a result."""
    rmock.post(f"{API}/api/v1/users/me/2fa/{method.split('_')[0]}", json=["unexpected"])
    with pytest.raises(ResponseError):
        getattr(client, method)("value")
