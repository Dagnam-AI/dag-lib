"""Deployment keys go to the keyring when it works, else to a 0600 file -- never anywhere else."""

from __future__ import annotations

import json
from pathlib import Path
import stat
import sys

import keyring
from keyring.errors import NoKeyringError
import pytest

from dagnam.audit.secrets import SECRETS_FILE, SERVICE, SecretStore


@pytest.fixture
def real_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the audit-wide autouse block so the real module (with patched calls) is importable."""
    monkeypatch.setitem(sys.modules, "keyring", keyring)


@pytest.fixture
def fake_keyring(real_keyring: None, monkeypatch: pytest.MonkeyPatch) -> dict[tuple[str, str], str]:
    vault: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(keyring, "set_password", lambda s, u, p: vault.__setitem__((s, u), p))
    monkeypatch.setattr(keyring, "get_password", lambda s, u: vault.get((s, u)))
    return vault


def test_keyring_mode_round_trips_and_writes_no_file(
    tmp_path: Path, fake_keyring: dict[tuple[str, str], str]
) -> None:
    store = SecretStore(tmp_path)
    assert store.store("w1/head_tune", "sk-secret") == "keyring"
    assert store.mode == "keyring"
    assert fake_keyring == {(SERVICE, f"{tmp_path.resolve()}::w1/head_tune"): "sk-secret"}
    assert store.load("w1/head_tune") == "sk-secret"
    assert store.load("missing") is None
    assert not (tmp_path / SECRETS_FILE).exists()


def test_file_mode_when_keyring_is_not_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)
    store = SecretStore(tmp_path / "audit")
    assert store.store("w1/head_tune", "sk-1") == "file"
    assert store.store("w1/sft_small", "sk-2") == "file"

    path = tmp_path / "audit" / SECRETS_FILE
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text()) == {"w1/head_tune": "sk-1", "w1/sft_small": "sk-2"}
    assert store.load("w1/head_tune") == "sk-1"
    assert store.load("nope") is None
    assert "pip install 'dagnam[audit]'" in caplog.text
    assert "sk-1" not in caplog.text


def test_keyring_without_a_backend_falls_back_to_the_file(
    real_keyring: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def broken(*_args: str) -> None:
        raise NoKeyringError("no backend")

    monkeypatch.setattr(keyring, "set_password", broken)
    monkeypatch.setattr(keyring, "get_password", broken)
    store = SecretStore(tmp_path)
    assert store.store("k", "v") == "file"
    assert "keyring unavailable" in caplog.text
    assert store.load("k") == "v"
    assert stat.S_IMODE((tmp_path / SECRETS_FILE).stat().st_mode) == 0o600


def test_file_mode_tightens_an_existing_wide_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)
    path = tmp_path / SECRETS_FILE
    path.write_text("{}")
    path.chmod(0o644)
    SecretStore(tmp_path).store("k", "v")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_forget_removes_the_key_from_both_backends(
    tmp_path: Path, fake_keyring: dict[tuple[str, str], str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from keyring.errors import PasswordDeleteError

    def delete_password(service: str, user: str) -> None:
        if (service, user) not in fake_keyring:
            raise PasswordDeleteError(user)
        del fake_keyring[(service, user)]

    monkeypatch.setattr(keyring, "delete_password", delete_password)
    store = SecretStore(tmp_path)
    store.store("w1/head_tune", "sk-secret")
    store.forget("w1/head_tune")
    store.forget("w1/head_tune")  # already gone in the keyring: not an error
    assert fake_keyring == {}
    assert store.load("w1/head_tune") is None

    monkeypatch.setitem(sys.modules, "keyring", None)
    store.store("w1/sft_small", "sk-file")
    store.store("w2/sft_small", "sk-file-2")
    store.forget("w1/sft_small")
    store.forget("never-stored")
    assert json.loads((tmp_path / SECRETS_FILE).read_text()) == {"w2/sft_small": "sk-file-2"}
