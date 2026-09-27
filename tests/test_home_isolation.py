"""Every test runs against a throwaway home: no real ~/.dagnam, no credential variables.

Without this, a test that reaches a real command path would send the developer's
stored API key to the live API, and a cache test could write into the real
``~/.dagnam``. The isolation lives in the autouse ``isolated_home`` fixture in
``tests/conftest.py``; these tests prove it holds.
"""

from __future__ import annotations

import os
from pathlib import Path

from dagnam._core import config
from dagnam.data import cache, load
from dagnam.data.loaders.system import dispatch

# Read while this module is collected, before any fixture runs: the process's real home.
_HOME_AT_COLLECTION = Path.home()


def test_each_test_gets_its_own_empty_home() -> None:
    assert Path.home() != _HOME_AT_COLLECTION
    assert list(Path.home().iterdir()) == []


def test_the_real_config_file_cannot_be_read() -> None:
    real_config = _HOME_AT_COLLECTION / ".dagnam" / "config.json"
    assert real_config != config.CONFIG_FILE
    assert Path.home() / ".dagnam" / "config.json" == config.CONFIG_FILE
    assert config.load_config() == {}
    assert "DAGNAM_API_KEY" not in os.environ
    assert "DAGNAM_API_URL" not in os.environ


def test_cache_roots_and_their_imported_copies_follow_the_temp_home() -> None:
    # load and dispatch hold copies made by `from ... import`, so moving HOME alone
    # would leave them pointing at the real ~/.dagnam.
    roots = [cache.DEFAULT_CACHE_DIR, load.DEFAULT_CACHE_DIR, dispatch.SYSTEM_CACHE_ROOT]
    assert not any(root.is_relative_to(_HOME_AT_COLLECTION / ".dagnam") for root in roots)
    assert all(root.is_relative_to(Path.home() / ".dagnam") for root in roots)
