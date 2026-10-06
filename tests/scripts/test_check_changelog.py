"""``scripts/check_changelog.py``: a release is refused while its entries sit in the wrong place."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_changelog.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("check_changelog", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_changelog = _load()

READY = """# Changelog

## [Unreleased]

## [0.16.0] - 2026-10-03

### Added

- Something.

## [0.15.0] - 2026-09-27
"""


def test_a_changelog_folded_under_the_version_being_released_is_ready() -> None:
    assert check_changelog.problems(READY, "0.16.0") == []


def test_entries_left_under_unreleased_are_refused() -> None:
    # The state of the repository between releases: the work is under [Unreleased].
    pending = READY.replace("## [Unreleased]\n", "## [Unreleased]\n\n- A change.\n", 1)
    (reason,) = check_changelog.problems(pending, "0.16.0")
    assert "still holds entries" in reason
    only_unreleased = "# Changelog\n\n## [Unreleased]\n\n### Fixed\n\n- A change.\n"
    assert "still holds entries" in check_changelog.problems(only_unreleased, "0.16.0")[0]


def test_a_heading_for_another_version_or_without_a_date_is_refused() -> None:
    (other,) = check_changelog.problems(READY, "0.17.0")
    assert "newest dated heading is not `## [0.17.0]" in other
    undated = READY.replace("[0.16.0] - 2026-10-03", "[0.16.0] - YYYY-MM-DD")
    (reason,) = check_changelog.problems(undated, "0.16.0")
    assert "no real release date" in reason
    bare = READY.replace(" - 2026-10-03", "")
    assert "no real release date" in check_changelog.problems(bare, "0.16.0")[0]


def test_a_changelog_with_no_unreleased_heading_or_no_release_is_refused() -> None:
    no_unreleased = READY.replace("## [Unreleased]\n\n", "")
    assert "no `## [Unreleased]` heading" in check_changelog.problems(no_unreleased, "0.16.0")[0]
    assert len(check_changelog.problems("# Changelog\n", "0.16.0")) == 2


def test_main_reports_every_reason_and_the_exit_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    init = tmp_path / "__init__.py"
    init.write_text('__version__ = "0.16.0"\n', encoding="utf-8")
    changelog = tmp_path / "CHANGELOG.md"

    changelog.write_text(READY, encoding="utf-8")
    assert check_changelog.main(["--changelog", str(changelog), "--init", str(init)]) == 0
    assert "is ready for 0.16.0" in capsys.readouterr().out

    changelog.write_text(READY.replace("## [Unreleased]\n", "## [Unreleased]\n\n- x\n"), "utf-8")
    assert check_changelog.main(["--changelog", str(changelog), "--init", str(init)]) == 1
    assert "still holds entries" in capsys.readouterr().err

    init.write_text("no version here\n", encoding="utf-8")
    assert check_changelog.main(["--changelog", str(changelog), "--init", str(init)]) == 1
    assert "declares no __version__" in capsys.readouterr().err
