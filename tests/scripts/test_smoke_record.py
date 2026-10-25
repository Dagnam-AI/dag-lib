"""``scripts/smoke_record.py``: the release smoke test of an INSTALLED ``dagnam-record``.

The package comes from ``record/`` and its installed metadata is a ``.dist-info`` written into a
temporary directory, so the script runs against a real recorder without building anything.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "smoke_record.py"


def _dist_info(where: Path, version: str, *files: str) -> None:
    info = where / f"dagnam_record-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.4\nName: dagnam-record\nVersion: {version}\n", encoding="utf-8"
    )
    (info / "RECORD").write_text("".join(f"{name},,\n" for name in files), encoding="utf-8")
    for name in files:  # the metadata API lists only the files that exist
        (where / name).parent.mkdir(parents=True, exist_ok=True)
        (where / name).touch()


def _smoke(tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return subprocess.run(  # noqa: S603 - the running interpreter and our own script
        [sys.executable, str(SCRIPT), *arguments],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(home),
            "PYTHONPATH": f"{ROOT / 'record'}{os.pathsep}{tmp_path / 'site'}",
        },
    )


def test_an_installed_package_that_is_the_expected_version_and_works_passes(tmp_path: Path) -> None:
    _dist_info(tmp_path / "site", "0.1.0", "dagnam_record/__init__.py", "dagnam_record/py.typed")
    result = _smoke(tmp_path, "0.1.0")
    assert (result.returncode, result.stderr) == (0, "")
    assert "dagnam-record 0.1.0 installed and working" in result.stdout
    assert not (tmp_path / "home" / ".dagnam").exists()  # the default directory is never made


@pytest.mark.parametrize(
    ("expected", "files", "reason"),
    [
        (
            "0.1.1",
            ("dagnam_record/py.typed",),
            "version: module 0.1.0, metadata 0.1.0, wanted 0.1.1",
        ),
        ("0.1.0", ("tests/test_x.py", "dagnam_record/py.typed"), "test files are installed"),
    ],
)
def test_a_wrong_version_or_shipped_tests_fail_with_the_reason(
    tmp_path: Path, expected: str, files: tuple[str, ...], reason: str
) -> None:
    _dist_info(tmp_path / "site", "0.1.0", *files)
    result = _smoke(tmp_path, expected)
    assert result.returncode == 1
    assert reason in result.stderr


def test_the_module_and_the_installed_metadata_must_agree(tmp_path: Path) -> None:
    _dist_info(tmp_path / "site", "9.9.9", "dagnam_record/py.typed")
    result = _smoke(tmp_path, "9.9.9")
    assert result.returncode == 1
    assert "module 0.1.0, metadata 9.9.9" in result.stderr


def test_it_needs_exactly_the_expected_version(tmp_path: Path) -> None:
    for arguments in ((), ("0.1.0", "extra")):
        result = _smoke(tmp_path, *arguments)
        assert result.returncode == 2
        assert "usage: smoke_record.py" in result.stderr
