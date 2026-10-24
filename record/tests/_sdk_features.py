"""Parts of the SDKs that only some of the versions the recorder supports have.

The CI matrix runs the lowest and the highest ``openai`` and ``anthropic`` the recorder declares
(see ``dagnam_record._compat``). A test of something the lower version does not have is skipped
there, with the reason; ``test_sdk_features.py`` fails if the versions the repo locks lack any of
them, so a skip cannot hide a test in the main gate.
"""

from __future__ import annotations

import inspect
from typing import Dict

import anthropic
import anthropic.lib.streaming as anthropic_streaming
import anthropic.resources.beta.messages as anthropic_beta
import anthropic.resources.messages as anthropic_messages
import openai.resources.chat.completions as openai_chat
import pytest

FEATURES: Dict[str, bool] = {
    "openai chat.completions.parse": hasattr(openai_chat.Completions, "parse"),
    "openai chat.completions.stream": hasattr(openai_chat.Completions, "stream"),
    "anthropic messages.parse": hasattr(anthropic_messages.Messages, "parse"),
    "anthropic messages.count_tokens": hasattr(anthropic_messages.Messages, "count_tokens"),
    "anthropic beta.messages.parse": hasattr(anthropic_beta.Messages, "parse"),
    "anthropic beta.messages.stream": hasattr(anthropic_beta.Messages, "stream"),
    "anthropic BetaMessageStreamManager": hasattr(anthropic_streaming, "BetaMessageStreamManager"),
    "anthropic beta_tool": hasattr(anthropic, "beta_tool"),
    "anthropic mcp_servers": "mcp_servers"
    in inspect.signature(anthropic_beta.Messages.create).parameters,
}


def needs(feature: str) -> pytest.MarkDecorator:
    """Skip a test when the installed SDK lacks ``feature`` (a key of ``FEATURES``)."""
    return pytest.mark.skipif(not FEATURES[feature], reason=f"this SDK version has no {feature}")
