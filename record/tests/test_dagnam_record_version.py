"""The package imports on its own and declares one version."""

from __future__ import annotations

import dagnam_record


def test_the_package_declares_its_version() -> None:
    assert dagnam_record.__version__ == "0.1.0"
