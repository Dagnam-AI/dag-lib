"""The fixtures copied from dag-record are byte-for-byte what they were when copied.

``recorder_*.jsonl`` and ``openai_roundtrip_batch.jsonl`` are written by ``tests/gen_fixtures.py`` in
https://github.com/Dagnam-AI/dag-record and copied here by hand; the SDK's reader tests read them.
Nothing else notices a local edit or a stale copy, so this pins each file's SHA-256 and the
dag-record commit they came from. A deliberate refresh changes the hashes and ``SOURCE_COMMIT`` here
in the same commit as the copy. (It cannot see dag-record regenerating; once ``dagnam-record`` is on
PyPI, a test against the installed package is the better guard.)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

SOURCE_COMMIT = "90d6071"
FIXTURES = Path(__file__).parent / "fixtures"
SHA256 = {
    "openai_roundtrip_batch.jsonl": "8399cdaeca99c0d49942e6f41dfc5ba11c2671a4836c18036357f16796d69e65",
    "recorder_anthropic.jsonl": "09c6eaaddc18fa589bf970679e557fd743717ebbfd70f398bd2841b5ad4a6774",
    "recorder_anthropic_stream.jsonl": "dc6bb6c2657ebd9617eac1a4a92dc05e3abef650cea5049411a3ea2c5a21bb02",
    "recorder_media.jsonl": "c27e662a74005a1a0e6633124fe7506c52adde936e7fd2fd31c14e6e4042e424",
    "recorder_mixed_bad.jsonl": "2e3b403ec694390a2872579e215d548f20f79f26dc5c4c6eaa860324e035cdc3",
    "recorder_norm_only.jsonl": "e42e124149f2277a9d42e9d6dc340a0d571c2e328e149d850a775f20572cf3c7",
    "recorder_openai_chat.jsonl": "79fdece1684fb0d8c4cdc2dac66fe0ed3c50aa8ee4964bca3edce6f7954f4dfa",
    "recorder_openai_parse.jsonl": "783f6b984c49894c5030baaac1ea0dfea7e24e34c647756f5485d3d7d9b20538",
    "recorder_openai_responses.jsonl": "972e1e9537a6328ee1422bf6fe2007bf0c83503640bd3a705c109557d3de9fea",
    "recorder_openai_stream.jsonl": "e70d7be54664b5ade24b44d1f856fae5cf9cd2759f5f1e1722cb82f47dbe2a2e",
    "recorder_other.jsonl": "1d9c6de738fa0081bdafbc3c01cf02a075d891b9fb1969415369a6f0a2043d0d",
}


def test_the_pinned_set_is_every_recorder_fixture_here() -> None:
    on_disk = {path.name for path in FIXTURES.glob("recorder_*.jsonl")} | {
        "openai_roundtrip_batch.jsonl"
    }
    assert on_disk == set(SHA256)


@pytest.mark.parametrize("name", sorted(SHA256))
def test_a_copied_fixture_is_unchanged_since_it_was_copied(name: str) -> None:
    digest = hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest()
    assert digest == SHA256[name], (
        f"{name} differs from dag-record {SOURCE_COMMIT}; refresh the pins with it"
    )
