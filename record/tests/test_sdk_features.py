"""The newest SDKs a row or the main gate runs have every feature the version-dependent tests skip without.

Only a "lowest" matrix row (``DAGNAM_RECORD_SDK_ROW=lowest``) may lack features. Everywhere else,
the main gate, a "highest" row and a source-distribution run, a feature that is gone makes the
guard fail instead of letting the tests that need it skip quietly.
"""

from __future__ import annotations

import os
from typing import Dict

from _sdk_features import FEATURES, lost_features
import pytest


def test_the_features_a_test_may_skip_without_are_found_by_name() -> None:
    assert lost_features({"a": True, "b": False, "c": False}) == ["b", "c"]
    assert lost_features({"a": True}) == []


@pytest.mark.parametrize("features", [{"parse": False}, {"ok": True, "beta_tool": False}])
def test_a_feature_that_vanished_fails_the_guard(features: Dict[str, bool]) -> None:
    with pytest.raises(AssertionError, match="no longer in the SDKs"):
        _assert_none_lost(features)


def _assert_none_lost(features: Dict[str, bool]) -> None:
    lost = lost_features(features)
    assert lost == [], f"no longer in the SDKs this run installed: {lost}"


@pytest.mark.skipif(
    os.environ.get("DAGNAM_RECORD_SDK_ROW") == "lowest",
    reason="a lowest row runs the oldest supported SDKs, which lack features on purpose",
)
def test_the_sdks_have_every_feature_a_test_may_skip_without() -> None:
    _assert_none_lost(FEATURES)
