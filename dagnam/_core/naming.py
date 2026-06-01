"""Generate memorable, non-identifying local run names."""

from __future__ import annotations

import datetime
import secrets

_ADJECTIVES = (
    "amber",
    "bold",
    "brave",
    "calm",
    "clever",
    "cosmic",
    "crimson",
    "dapper",
    "eager",
    "fancy",
    "gentle",
    "golden",
    "happy",
    "jolly",
    "keen",
    "lucky",
    "mellow",
    "nimble",
    "noble",
    "proud",
    "quiet",
    "rapid",
    "sage",
    "shiny",
    "silent",
    "snowy",
    "solar",
    "spry",
    "stellar",
    "swift",
    "tidy",
    "vivid",
    "witty",
    "zesty",
)
_ANIMALS = (
    "ant",
    "bear",
    "bison",
    "cobra",
    "crane",
    "dingo",
    "eagle",
    "falcon",
    "fox",
    "gecko",
    "hawk",
    "heron",
    "ibex",
    "jaguar",
    "koala",
    "lemur",
    "lynx",
    "marten",
    "monkey",
    "newt",
    "otter",
    "panda",
    "puma",
    "quail",
    "raven",
    "seal",
    "shark",
    "tapir",
    "tiger",
    "viper",
    "walrus",
    "yak",
    "zebra",
)


def generate_run_name(now_str: str | None = None) -> str:
    """Return an ordered, memorable run name without exposing machine identity."""
    if now_str is None:
        now_str = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S-%f")
    return f"{now_str}-{secrets.choice(_ADJECTIVES)}-{secrets.choice(_ANIMALS)}"
