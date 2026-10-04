"""The rules both checks of an auto-fix patch apply, and the applied-names check.

Used by check-fix-patch.py in this directory. `path_problem` judges one repository
path; `scan_applied` judges git's own list of what an applied patch changed
(`git diff --raw -z`), which is the check that decides: a parser of patch text and
`git apply` can read the same text as different names, but the list git prints
cannot disagree with what git will commit.
"""

from __future__ import annotations

import re
import unicodedata

# Deleting or renaming away more files than this is not a fix for a red check.
MAX_REMOVED = 3

# The one protected-name set, used by both modes. A coding agent or editor on the
# owner's machine loads these from a checked-out branch without being asked (or acts
# on them when the branch is opened), so a model-authored patch must not write them.
# Three layers, all compared after folding (lower case, NFKC, no trailing dot):
#   1. PROTECTED: exact names, at any depth.
#   2. PROTECTED_PREFIXES: a file name that starts with one (tools add suffixes:
#      `.cursorrules`, `.aider.conf.yml`, `.devcontainer.json`, `.yarnrc.yml`).
#   3. Any dot-directory at any depth, and AGENT_MARKDOWN (a family, not a list:
#      AGENTS.md, AGENTS.override.md, CLAUDE.local.md, GEMINI.md and the like).
# Tools keep their project configuration in a dot-directory (.vscode, .idea, .zed,
# .cursor, .agents, ...), so the third layer ends the chase after each new tool; a
# fix for a red check has no reason to write inside one. Deliberately NOT protected,
# because an ordinary fix edits them: source, tests, dependency manifests and
# lockfiles (pyproject.toml, uv.toml, package.json), `.gitignore`, `conftest.py`,
# lint and format configuration (`.prettierrc*`, `.coveragerc`), version pins
# (`.python-version`, `.nvmrc`), `.env.example`, and `pip.conf` (pip does not read
# a project-directory copy).
PROTECTED = frozenset(
    {
        # git: attributes (filters run commands), submodules, LFS endpoint; CI
        ".git",
        ".github",
        ".gitattributes",
        ".gitmodules",
        ".lfsconfig",
        # tools that run or fetch what a project file names
        ".pre-commit-config.yaml",
        ".envrc",
        ".npmrc",
        ".pnpmfile.cjs",
        "mise.toml",
        "lefthook.yml",
        "lefthook.yaml",
        # agent settings, rules and MCP servers that live in a plain file
        ".mcp.json",
        "mcp.json",
        "opencode.json",
        "opencode.jsonc",
        "copilot-instructions.md",
        ".rules",  # Zed
        ".goosehints",
        ".roorules",
        ".roomodes",
    }
)
PROTECTED_PREFIXES = (
    ".yarnrc",  # yarn runs the file a `yarnPath` setting names
    ".mise",
    ".lefthook",
    ".devcontainer",  # `.devcontainer.json`
    ".claude",
    ".codex",
    ".gemini",
    ".agent",
    ".cursor",
    ".windsurf",
    ".clinerules",
    ".aider",
    ".continue",
)
# CLAUDE.md, CLAUDE.local.md, AGENTS.md, AGENTS.override.md, AGENT.md, GEMINI.md,
# codex.md and any `<family>.<qualifier>.md` of them; not `claude-notes.md`.
AGENT_MARKDOWN = re.compile(r"(agents?|claude|gemini|codex)(\..+)?\.md")

EDITABLE_MODES = ("100644", "100755")  # a file edited or deleted in place
NEW_FILE_MODE = "100644"

# One `git diff --raw -z --no-abbrev` record: modes, two object names, status.
_RAW = re.compile(rb":([0-7]{6}) ([0-7]{6}) [0-9a-f]+(?:\.\.\.)? [0-9a-f]+(?:\.\.\.)? ([A-Z])\d*")


def _fold(component: str) -> str:
    """A name as a case-insensitive, Unicode-normalising file system sees it."""
    text = unicodedata.normalize("NFKC", component)
    text = "".join(c for c in text if unicodedata.category(c) != "Cf")
    return text.rstrip(". ").casefold()


def path_problem(raw: bytes) -> str | None:
    """Why ``raw`` (a repository path, no `a/` or `b/` prefix) is refused, or None."""
    try:
        path = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "path is not valid UTF-8"
    parts = path.split("/")
    odd_chars = any(c in path for c in "\\:") or any(ord(c) < 32 or ord(c) == 127 for c in path)
    for refused, why in (
        (not path, "empty path"),
        (path.startswith("/"), "absolute path"),
        (odd_chars, "backslash, colon or control character in path"),
        (
            any(part in ("", ".", "..") for part in parts),
            "path has an empty, `.` or `..` component",
        ),
        (
            any(re.search(r"~\d", part) for part in parts),
            "path has a `NAME~N` short-name alias component",
        ),
    ):
        if refused:
            return why
    folded = [_fold(part) for part in parts]
    protected = [
        part in PROTECTED
        or part.startswith(PROTECTED_PREFIXES)
        or bool(AGENT_MARKDOWN.fullmatch(part))
        or (depth < len(parts) - 1 and part.startswith("."))  # a dot-directory
        for depth, part in enumerate(folded)
    ]
    if any(protected[:-1]):
        return "path is inside a protected directory"
    return "path is a protected file" if protected[-1] else None


def _entry_problem(status: str, src: str, dst: str) -> str | None:
    """Why a changed path of this status and these modes is refused, or None."""
    if "120000" in (src, dst) or "160000" in (src, dst):
        return "a symlink or submodule"
    if status in ("A", "C"):
        return None if dst == NEW_FILE_MODE else f"a new file with mode {dst}: a new executable"
    if status == "D":
        return None if src in EDITABLE_MODES else f"a deleted file with mode {src}"
    if status in ("M", "R"):
        return None if src == dst and src in EDITABLE_MODES else "a mode change"
    if status == "T":
        return "a type change"
    return f"an unexpected status {status}"


def scan_applied(data: bytes) -> tuple[list[str], int, int]:
    """Judge `git diff --raw -z --no-abbrev` output: (problems, paths, removed).

    Each record is `:srcmode dstmode srcsha dstsha STATUS` and then one NUL-ended
    path, two for a rename or copy. NUL separation keeps a name with a newline in
    it whole, which a line-based listing would split.
    """
    problems: list[str] = []
    if data and not data.endswith(b"\0"):
        problems.append("the list of changed paths does not end with a NUL")
    fields = data.split(b"\0")[:-1]
    count = removed = i = 0
    while i < len(fields):
        meta = _RAW.fullmatch(fields[i])
        if not meta:
            problems.append(f"cannot read the entry {fields[i][:60]!r}")
            break
        status, (src, dst) = meta[3].decode(), (meta[1].decode(), meta[2].decode())
        width = 2 if status in ("R", "C") else 1
        paths = fields[i + 1 : i + 1 + width]
        if len(paths) < width:
            problems.append(f"an entry is cut off after its metadata: {fields[i][:60]!r}")
            break
        i += 1 + width
        count += len(paths)
        removed += status in ("D", "R")
        for path in paths:
            found = path_problem(path) or _entry_problem(status, src, dst)
            if found:
                problems.append(f"{path.decode('utf-8', 'replace')!r}: {found}")
    if not data:
        problems.append("git lists no changed paths")
    if removed > MAX_REMOVED:
        problems.append(f"{removed} files deleted or renamed away; the limit is {MAX_REMOVED}")
    return problems, count, removed
