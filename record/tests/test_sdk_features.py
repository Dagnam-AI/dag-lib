"""The SDK versions the repo locks have every feature the version-dependent tests skip without."""

from __future__ import annotations

import os

from _sdk_features import FEATURES
import pytest


@pytest.mark.skipif(
    os.environ.get("DAGNAM_RECORD_SDK_MATRIX") == "1",
    reason="the SDK matrix runs other versions on purpose; the tests that need more are skipped",
)
def test_the_locked_sdks_have_every_feature_a_test_may_skip_without() -> None:
    assert [name for name, present in FEATURES.items() if not present] == []
