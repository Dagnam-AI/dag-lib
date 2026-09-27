"""Integrity checks for the bundled Agent Skill payload."""

from __future__ import annotations

from importlib import resources
import json
from pathlib import Path
import re

from dagnam._agent import install

_ASSETS = Path(str(resources.files("dagnam._agent")))


def _all_subcommands() -> set[str]:
    """Every leaf subcommand string in the parser, e.g. 'training create'."""
    from dagnam.cli.main import build_parser

    names: set[str] = set()

    def walk(parser: object, prefix: str) -> None:
        for action in getattr(parser, "_actions", []):
            choices = getattr(action, "choices", None)
            if isinstance(choices, dict):
                for name, sub in choices.items():
                    full = f"{prefix}{name}".strip()
                    names.add(full)
                    walk(sub, f"{full} ")

    walk(build_parser(), "")
    return names


def test_skill_md_frontmatter_valid() -> None:
    text = (_ASSETS / "skill" / "SKILL.md").read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert m, "SKILL.md must start with YAML frontmatter"
    assert "name: dagnam" in m.group(1)
    assert "description:" in m.group(1)


def test_reference_files_exist_for_every_domain() -> None:
    ref = _ASSETS / "skill" / "reference"
    expected = {
        "datasets",
        "cache",
        "projects",
        "codegen",
        "training",
        "deployments",
        "inference",
        "hub",
        "account",
        "troubleshooting",
    }
    assert {p.stem for p in ref.glob("*.md")} == expected


def test_no_dead_command_references_in_skill_docs() -> None:
    """Every `dagnam <cmd> <sub>` mentioned in SKILL.md/reference must resolve in the parser."""
    valid = _all_subcommands()
    pattern = re.compile(r"`dagnam (\w[\w-]*(?: \w[\w-]*)?)")
    docs = [_ASSETS / "skill" / "SKILL.md", *(_ASSETS / "skill" / "reference").glob("*.md")]
    bad: list[str] = []
    for doc in docs:
        for match in pattern.findall(doc.read_text(encoding="utf-8")):
            # The whole `<cmd> <sub>` must resolve: a live group must not vouch
            # for a removed subcommand under it.
            if match not in valid:
                bad.append(f"{doc.name}: dagnam {match}")
    assert not bad, f"Dead command references: {bad}"


def test_skill_docs_name_only_env_vars_the_sdk_reads() -> None:
    """Every ``DAGNAM_*`` variable the skill mentions must appear in the package source."""
    source = "\n".join(p.read_text(encoding="utf-8") for p in _ASSETS.parent.rglob("*.py"))
    docs = [_ASSETS / "skill" / "SKILL.md", *(_ASSETS / "skill" / "reference").glob("*.md")]
    named = {
        name
        for doc in docs
        for name in re.findall(r"\bDAGNAM_[A-Z_]+\b", doc.read_text(encoding="utf-8"))
    }
    unread = sorted(name for name in named if name not in source)
    assert not unread, f"Skill documents env vars nothing reads: {unread}"


def test_scripts_present() -> None:
    scripts = _ASSETS / "skill" / "scripts"
    assert (scripts / "plan.py").exists()
    assert (scripts / "watch_training.py").exists()


def test_claude_plugin_and_codex_yaml_version_placeholders() -> None:
    """Adapter files ship with the 0.0.0 placeholder the installer stamps at install time."""
    plugin = json.loads((_ASSETS / "claude" / "plugin.json").read_text(encoding="utf-8"))
    assert plugin["version"] == "0.0.0"
    assert "version: 0.0.0" in (_ASSETS / "codex" / "agents" / "openai.yaml").read_text(
        encoding="utf-8"
    )


def test_package_version_resolves() -> None:
    assert re.match(r"^\d+\.\d+", install.package_version())
