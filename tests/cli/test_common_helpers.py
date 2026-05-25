"""Tests for the shared CLI helpers added for version/account commands."""

from __future__ import annotations

from dagnam.cli.common import mask_key, resolve_version


class TestMaskKey:
    def test_long_key_shows_prefix_and_suffix(self) -> None:
        assert mask_key("sk_abcdefghijklmnop") == "sk_abc...mnop"

    def test_short_key_fully_masked(self) -> None:
        assert mask_key("sk_123") == "******"

    def test_boundary_ten_chars_fully_masked(self) -> None:
        assert mask_key("1234567890") == "*" * 10


class TestResolveVersion:
    def test_returns_nonempty_string(self) -> None:
        version = resolve_version()
        assert isinstance(version, str)
        assert version

    def test_falls_back_to_dunder_version_when_not_installed(self) -> None:
        import dagnam.cli.common as common_mod

        def _raise(_name: str) -> str:
            from importlib.metadata import PackageNotFoundError

            raise PackageNotFoundError(_name)

        import importlib.metadata as md

        original = md.version
        md.version = _raise  # type: ignore[assignment]
        try:
            from dagnam import __version__

            assert common_mod.resolve_version() == __version__
        finally:
            md.version = original  # type: ignore[assignment]
