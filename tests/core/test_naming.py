from __future__ import annotations

import re

from dagnam._core.naming import generate_run_name


def test_run_name_format():
    name = generate_run_name(now_str="20260531-0954")
    assert re.fullmatch(r"20260531-0954-[a-z]+-[a-z]+", name), name


def test_run_name_unique_enough():
    names = {generate_run_name() for _ in range(200)}
    assert len(names) > 190
