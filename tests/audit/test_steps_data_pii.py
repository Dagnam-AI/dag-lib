"""The PII stop: a privacy control that never relaxes, and says which of its causes it is."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from tests.audit._platform import FakePlatform

from dagnam.audit.state import AuditState
from dagnam.audit.steps import StepContext
from dagnam.audit.steps_data import (
    pii_scan,
    pii_stop,
    resolve_version,
    split,
    upload,
    wait_pii,
    wait_split,
)

REMOVES = " `dagnam audit delete <audit-dir>` removes the uploaded rows and the whole audit."
ERROR_CAP = 500
"""What the platform keeps of a candidate's ``error``: past it, the end of the message is lost."""


@pytest.fixture(autouse=True)
def installed_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the contract this install redacts with, so a release bump cannot move the messages."""
    monkeypatch.setattr("dagnam.audit.steps_data.installed_contract", lambda: "0.4.0")


def _stop(ctx: StepContext) -> str:
    state = AuditState()
    for step in (upload, resolve_version, split, wait_split, pii_scan, wait_pii):
        state = step(state, ctx)
    step_state = ctx.step(state)
    assert step_state.pii_agrees is False
    assert step_state.error is not None
    return step_state.error


def test_a_platform_on_an_older_privacy_contract_is_named_as_the_cause(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    """The server scanned fewer classes than the client redacted, and found nothing.

    The stop stays a stop: a scan that does not know a class cannot vouch for it.
    """
    platform.pii_pass_list = ["PII_EMAIL"]
    assert _stop(make_ctx()) == (
        "pii_disagreement: the platform runs an older privacy contract than this SDK: its scan"
        " does not cover PII_NATIONAL_ID, PII_PAYMENT_CARD, PII_PHONE (it found nothing in the"
        f" classes it does cover). Run again once it is upgraded.{REMOVES}"
    )


def test_a_platform_on_a_newer_privacy_contract_is_named_as_the_cause(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    """The server found matches only in classes these rows were never scanned for."""
    platform.pii_pass_list = [*platform.pii_pass_list, "PII_SECRET", "PII_IP_ADDRESS"]
    platform.pii_counts = {"ds-1": {"PII_SECRET": 185, "PII_IP_ADDRESS": 636, "PII_EMAIL": 0}}
    assert _stop(make_ctx()) == (
        "pii_disagreement: the platform runs a newer privacy contract than these rows were"
        " redacted with: it found PII_IP_ADDRESS (636), PII_SECRET (185) in classes the scan did"
        " not look for. `pip install -U dagnam-contracts` (or `pip install -U dagnam`), then scan"
        f" again.{REMOVES}"
    )


def test_a_newer_platform_patch_is_named_with_both_versions_and_the_cure(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    """Contract 0.4.1 finds more inside the same classes: same class list, different counts."""
    platform.pii_counts = {"ds-1": {"PII_EMAIL": 2}}
    message = _stop(make_ctx(platform_contracts="0.4.1"))
    assert message == (
        "pii_disagreement: the platform found PII_EMAIL (2) in classes this SDK redacts: the"
        " detectors disagree, or the rows changed since the scan. Scan again; report it if it"
        " repeats. The platform runs dagnam-contracts 0.4.1, this install"
        f" 0.4.0: `pip install -U dagnam-contracts`, then scan again.{REMOVES}"
    )


def test_an_older_platform_patch_says_to_wait_for_it(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    platform.pii_counts = {"ds-1": {"PII_EMAIL": 2}}
    message = _stop(make_ctx(platform_contracts="0.3.9"))
    assert (
        " The platform runs dagnam-contracts 0.3.9, older than this install's 0.4.0: wait for it."
        in message
    )
    assert "pip install" not in message


def test_a_newer_platform_minor_names_the_upgrade_for_the_newer_contract_case(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    platform.pii_pass_list = [*platform.pii_pass_list, "PII_SECRET"]
    platform.pii_counts = {"ds-1": {"PII_SECRET": 3}}
    message = _stop(make_ctx(platform_contracts="0.5.0"))
    assert "The platform runs dagnam-contracts 0.5.0, this install 0.4.0" in message
    assert "`pip install -U dagnam-contracts`, then scan again." in message


def test_a_finding_in_a_class_this_sdk_redacts_is_a_real_disagreement(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    """Never explained away as a version gap, whatever else the two contracts differ in."""
    platform.pii_pass_list = ["PII_EMAIL", "PII_SECRET"]
    platform.pii_counts = {"ds-1": {"PII_EMAIL": 2, "PII_SECRET": 1, "PII_PHONE": 0}}
    assert _stop(make_ctx()) == (
        "pii_disagreement: the platform found PII_EMAIL (2), PII_SECRET (1) in classes this SDK"
        " redacts: the detectors disagree, or the rows changed since the scan. Scan again;"
        " report it if it repeats. Not scanned:"
        f" PII_NATIONAL_ID, PII_PAYMENT_CARD, PII_PHONE.{REMOVES}"
    )


def test_a_real_disagreement_on_a_platform_that_scanned_every_class_says_only_that(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    platform.pii_counts = {"ds-1": {"PII_EMAIL": 2, "PII_PHONE": 0}}
    assert _stop(make_ctx()) == (
        "pii_disagreement: the platform found PII_EMAIL (2) in classes this SDK redacts: the"
        " detectors disagree, or the rows changed since the scan. Scan again; report it if it"
        f" repeats.{REMOVES}"
    )


def test_a_finding_only_in_a_new_class_beside_classes_the_platform_lacks_names_both(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    """A platform that is newer in one class and older in another: both facts are in the stop."""
    platform.pii_pass_list = ["PII_EMAIL", "PII_SECRET"]
    platform.pii_counts = {"ds-1": {"PII_SECRET": 4}}
    message = _stop(make_ctx())
    assert message.startswith("pii_disagreement: the platform runs a newer privacy contract")
    assert "it found PII_SECRET (4) in classes the scan did not look for" in message
    assert " Not scanned: PII_NATIONAL_ID, PII_PAYMENT_CARD, PII_PHONE." in message


def test_pii_result_without_counts_or_pass_list_cannot_agree(
    make_ctx: Callable[..., StepContext], platform: FakePlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = make_ctx()
    state = pii_scan(
        wait_split(split(resolve_version(upload(AuditState(), ctx), ctx), ctx), ctx), ctx
    )
    monkeypatch.setattr(
        platform,
        "get_dataset_task_status",
        lambda task_id: {
            "status": "SUCCESS",
            "result": {"counts_by_code": None, "pass_list": None},
        },
    )
    step = ctx.step(wait_pii(state, ctx))
    assert step.pii_agrees is False
    assert step.error == (
        "pii_disagreement: the platform's scan did not say which classes it looked for, so it"
        f" cannot confirm the redaction. Scan and run again.{REMOVES}"
    )


def test_no_message_publishes_a_path_of_this_machine_and_each_says_what_delete_removes(
    make_ctx: Callable[..., StepContext], platform: FakePlatform, audit_dir: Path
) -> None:
    platform.pii_pass_list = ["PII_EMAIL"]
    message = _stop(make_ctx())
    assert str(audit_dir) not in message
    assert "<audit-dir>" in message
    assert "removes the uploaded rows and the whole audit" in message


@pytest.mark.parametrize("contracts", [None, "0.4.1", "0.3.9", "0.5.0"])
def test_the_cause_survives_the_platforms_500_character_cap_in_every_case(
    make_ctx: Callable[..., StepContext], contracts: str | None
) -> None:
    """The platform keeps the first 500 characters of a candidate's error; the cause is first.

    The worst case: dozens of classes on each side, and a version gap besides.
    """
    ctx = make_ctx(platform_contracts=contracts)
    wide = {f"PII_CLASS_{n:02d}": 100 + n for n in range(40)}
    everything = {f"PII_LOCAL_{n:02d}" for n in range(40)}
    cases = {
        "known": pii_stop({**wide, "PII_EMAIL": 9}, set(), {"PII_EMAIL", *everything}, ctx),
        "newer": pii_stop(wide, set(), everything, ctx),
        "older": pii_stop({}, {"PII_EMAIL"}, everything, ctx),
        "silent": pii_stop({}, set(), set(), ctx),
    }
    for name, stop in cases.items():
        error = f"pii_disagreement: {stop}"
        assert len(error) <= ERROR_CAP, (name, len(error))
        assert error.endswith(REMOVES), name
    assert "and 37 more" in cases["newer"]
