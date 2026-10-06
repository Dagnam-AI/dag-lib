#!/usr/bin/env python3
"""Tests for check-fix-patch.py. Run: python3 .github/scripts/test_check_fix_patch.py

Every case is a hand-written patch file's text. The checker reads text only, so
no repository is involved. ``CASES`` is (name, patch, expected): ``None`` means
the patch must be accepted, a string is a phrase the refusal must contain.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))  # the checker imports fix_patch_rules from its own directory
_SPEC = importlib.util.spec_from_file_location("check_fix_patch", _HERE / "check-fix-patch.py")
assert _SPEC is not None
assert _SPEC.loader is not None
check = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check)


def edit(path: str, mode: str = "100644") -> str:
    return (
        f"diff --git a/{path} b/{path}\nindex 1111111..2222222 {mode}\n"
        f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-a\n+b\n"
    )


def named(header: str, declared: str) -> str:
    """An edit whose names are written exactly as given (git quotes odd names itself)."""
    return (
        f"diff --git {header}\nindex 1111111..2222222 100644\n"
        f"--- {declared}\n+++ {declared}\n@@ -1 +1 @@\n-a\n+b\n"
    )


def create(path: str, mode: str = "100644") -> str:
    return (
        f"diff --git a/{path} b/{path}\nnew file mode {mode}\nindex 0000000..3333333\n"
        f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1 @@\n+x\n"
    )


def delete(path: str, mode: str = "100644") -> str:
    return (
        f"diff --git a/{path} b/{path}\ndeleted file mode {mode}\nindex 3333333..0000000\n"
        f"--- a/{path}\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n"
    )


def rename(src: str, dst: str) -> str:
    return (
        f"diff --git a/{src} b/{dst}\nsimilarity index 100%\nrename from {src}\nrename to {dst}\n"
    )


def copy(src: str, dst: str) -> str:
    return f"diff --git a/{src} b/{dst}\nsimilarity index 100%\ncopy from {src}\ncopy to {dst}\n"


def mode_change(path: str, old: str = "100644", new: str = "100755") -> str:
    return f"diff --git a/{path} b/{path}\nold mode {old}\nnew mode {new}\n"


BINARY = (
    "diff --git a/logo.png b/logo.png\nnew file mode 100644\nindex 0000000..abc1234\n"
    "GIT binary patch\nliteral 4\nLcmZQzU|?bX00aO6\n\nliteral 0\nHcmV?d00001\n\n"
)


def trick(path: str) -> str:
    """A new file whose `+++` name is a quoted path with text after the closing quote.

    git apply reads the quoted name and ignores what follows it. The `diff --git`
    names differ from the `+++` name, so git takes the `+++` one.
    """
    return (
        "diff --git a/n b/n.md\nnew file mode 100644\n--- /dev/null\n"
        f'+++ "b/{path}"x\n@@ -0,0 +1 @@\n+x\n'
    )


def declared(old: str, new: str, header: str = "a/src/x.py b/src/x.py", extra: str = "") -> str:
    """One text edit with the `---` and `+++` names written exactly as given."""
    return f"diff --git {header}\n{extra}--- {old}\n+++ {new}\n@@ -1 +1 @@\n-a\n+b\n"


PROTECTED_FILES = (
    # git, CI and tool configuration that runs or fetches what it names
    ".gitattributes",
    ".gitmodules",
    ".lfsconfig",
    ".pre-commit-config.yaml",
    ".envrc",
    ".npmrc",
    ".yarnrc",
    ".yarnrc.yml",
    ".pnpmfile.cjs",
    "mise.toml",
    ".mise.toml",
    "lefthook.yml",
    ".lefthook.yml",
    # agent instruction and settings files, by family (a pattern, not today's names)
    "CLAUDE.md",
    "CLAUDE.local.md",
    "CLAUDE.team.md",
    "AGENTS.md",
    "AGENTS.override.md",
    "AGENTS.local.md",
    "AGENT.md",
    "codex.md",
    "GEMINI.md",
    "GEMINI.local.md",
    "copilot-instructions.md",
    ".cursorrules",
    ".windsurfrules",
    ".clinerules",
    ".roorules",
    ".roomodes",
    ".goosehints",
    ".rules",
    ".aider.conf.yml",
    ".aiderignore",
    ".geminiignore",
    ".continuerc.json",
    ".devcontainer.json",
    ".mcp.json",
    "mcp.json",
    "opencode.json",
    "opencode.jsonc",
)
# Directories: the named ones, plus any dot-directory (a tool's project configuration).
PROTECTED_DIRS = (
    ".git",
    ".github",
    ".githooks",
    ".husky",
    ".yarn",
    ".claude",
    ".claude-plugin",
    ".codex",
    ".agents",
    ".gemini",
    ".cursor",
    ".windsurf",
    ".vscode",
    ".idea",
    ".zed",
    ".devcontainer",
    ".continue",
    ".aider",
    ".junie",
    ".futuretool",
)
# Ordinary files a fix for a red check must still be able to change.
ORDINARY = (
    ".gitignore",
    ".gitkeep",
    ".dockerignore",
    ".env.example",
    ".prettierrc.cjs",
    ".python-version",
    ".nvmrc",
    ".coveragerc",
    "uv.toml",
    "uv.lock",
    "pyproject.toml",
    "package.json",
    "yarn.lock",
    "conftest.py",
    "docs/agents-overview.md",
    "docs/claude-code-anthropic.md",
    "docs/agents_notes.md",
    "src/agent.py",
    "tests/test_agents.py",
)

# A removed line whose text starts with "-- a/..." shows up in a hunk as
# "--- a/.github/...", and an added line can start with "diff --git". Inside a
# hunk they are content, not headers.
LOOKALIKES = (
    "diff --git a/src/notes.txt b/src/notes.txt\nindex 1111111..2222222 100644\n"
    "--- a/src/notes.txt\n+++ b/src/notes.txt\n@@ -1,3 +1,3 @@\n"
    "--- a/.github/workflows/ci.yml\n+++ b/.github/workflows/ci.yml\n"
    "+diff --git a/.github/x b/.github/x\n keep\n-old\n"
    "@@ -9,2 +9,2 @@\n context\n-gone\n+here\n\\ No newline at end of file\n"
)

CASES: list[tuple[str, str | bytes, str | None]] = [
    # --- acceptable ---------------------------------------------------------
    ("edit and a new file", edit("src/app.py") + create("tests/test_app.py"), None),
    (
        "edit of a file with a space in its name (git ends the names with a tab)",
        named("a/src/my file.py b/src/my file.py", "a/src/my file.py\t"),
        None,
    ),
    (
        "edit of a file whose quoted name needs escapes",
        named('"a/src/caf\\303\\251.py" "b/src/caf\\303\\251.py"', '"a/src/caf\\303\\251.py"'),
        None,
    ),
    ("edit of an existing executable keeps its mode", edit("scripts/run.sh", "100755"), None),
    ("hunk content that looks like headers", LOOKALIKES, None),
    ("three deletions", "".join(delete(f"old/{n}.py") for n in "abc"), None),
    ("a docs file called notes in a docs folder", edit("docs/notes.md"), None),
    ("a path that only contains the word github", edit("src/github_client.py"), None),
    # --- protected paths ----------------------------------------------------
    ("workflow edit", edit(".github/workflows/ci.yml"), "protected directory"),
    (
        "new workflow",
        create(".github/workflows/evil.yml") + edit("src/app.py"),
        "protected directory",
    ),
    ("other letter case", create(".GitHub/workflows/x.yml"), "protected directory"),
    (
        "rename out of .github",
        rename(".github/workflows/ci.yml", "ci-moved.yml"),
        "protected directory",
    ),
    ("rename into .github", rename("src/app.py", ".github/app.py"), "protected directory"),
    ("copy into .github", copy("src/app.py", ".github/copy.py"), "protected directory"),
    ("delete of a workflow", delete(".github/workflows/ci.yml"), "protected directory"),
    ("git hook", edit(".githooks/pre-push"), "protected directory"),
    ("claude settings", create(".claude/settings.json"), "protected directory"),
    ("nested claude directory", create("pkg/.claude/skills/x.md"), "protected directory"),
    ("a .git directory", edit(".git/config"), "protected directory"),
    ("root CLAUDE.md", edit("CLAUDE.md"), "protected file"),
    ("nested CLAUDE.md", edit("docs/CLAUDE.md"), "protected file"),
    ("AGENTS.md in lower case", edit("agents.md"), "protected file"),
    ("codex.md", edit("codex.md"), "protected file"),
    (".gitattributes", create(".gitattributes"), "protected file"),
    (".gitmodules", edit(".gitmodules"), "protected file"),
    (".mcp.json", create(".mcp.json"), "protected file"),
    (".pre-commit-config.yaml", edit(".pre-commit-config.yaml"), "protected file"),
    (
        "a quoted name that decodes to .github",
        named(
            '"a/.git\\150ub/workflows/x y.yml" "b/.git\\150ub/workflows/x y.yml"',
            '"a/.git\\150ub/workflows/x y.yml"',
        ),
        "protected directory",
    ),
    ("a zero-width character inside .github", edit(".git\u200bhub/x"), "protected directory"),
    ("a full-width dot in front of github", edit("\uff0egithub/x"), "protected directory"),
    ("a trailing dot on .github", edit(".github./x"), "protected directory"),
    ("an 8.3 short-name alias", edit("GITHUB~1/x"), "short-name alias"),
    (
        "a header whose second name is protected and no other line says so",
        "diff --git a/src/x.py b/.github/y.yml\nindex 1111111..2222222 100644\n",
        "protected directory",
    ),
    (
        "a protected file in a second block after an ordinary edit",
        edit("src/app.py") + edit("pkg/CLAUDE.md"),
        "protected file",
    ),
    (
        "a protected header after a complete hunk",
        edit("src/app.py") + "diff --git a/.github/x b/.github/x\nindex 1..2 100644\n",
        "protected directory",
    ),
    # --- paths outside the repository ---------------------------------------
    ("parent directory", create("../outside.txt"), "empty, `.` or `..`"),
    ("parent directory in the middle", edit("src/../../x"), "empty, `.` or `..`"),
    ("absolute path", edit("/etc/passwd"), "absolute path"),
    ("backslash", edit("src\\..\\x"), "backslash"),
    ("drive or stream colon", edit("C:/x"), "colon"),
    # --- file kinds and modes -----------------------------------------------
    ("new symlink", create("link", "120000"), "symlink, submodule or executable"),
    ("new submodule", create("vendor/lib", "160000"), "symlink, submodule or executable"),
    ("edit of a symlink", edit("link", "120000"), "symlink or submodule"),
    ("new executable", create("run.sh", "100755"), "symlink, submodule or executable"),
    ("mode change to executable", mode_change("scripts/run.sh"), "a mode change"),
    (
        "mode change away from executable",
        mode_change("run.sh", "100755", "100644"),
        "a mode change",
    ),
    ("delete of a symlink", delete("link", "120000"), "symlink, submodule or executable"),
    ("binary patch", BINARY, "binary patch"),
    (
        "binary notice",
        "diff --git a/a.png b/a.png\nBinary files a/a.png and b/a.png differ\n",
        "binary patch",
    ),
    # --- deletions ----------------------------------------------------------
    ("four deletions", "".join(delete(f"old/{n}.py") for n in "abcd"), "deleted or renamed away"),
    (
        "four renames count as removals",
        "".join(rename(f"a/{n}.py", f"b/{n}.py") for n in "abcd"),
        "deleted or renamed away",
    ),
    # --- not a patch this check can read ------------------------------------
    ("empty", "", "does not start with"),
    ("text before the first header", "From: someone\n" + edit("src/app.py"), "does not start with"),
    (
        "unknown header line",
        edit("src/app.py").replace("index", "extended", 1),
        "does not understand",
    ),
    ("hunk shorter than declared", edit("src/app.py")[:-6], "ends inside a hunk"),
    (
        "hunk longer than declared",
        edit("src/app.py") + "+c\n",
        "does not understand",
    ),
    (
        "hunk with a line that is not content",
        edit("src/app.py").replace("@@ -1 +1 @@", "@@ -1,2 +1,2 @@") + "!c\n",
        "declared line counts",
    ),
    (
        "junk after a hunk",
        edit("src/app.py") + "do something else\n",
        "does not understand",
    ),
    ("combined diff", "diff --git a/x b/x\n@@@ -1 -1 +1 @@@\n", "hunk header"),
    ("a quote that never closes", 'diff --git "a/x b/x\n', "cannot read the names"),
    (
        "non-UTF-8 path",
        b"diff --git a/\xff.py b/\xff.py\nindex 1111111..2222222 100644\n",
        "not valid UTF-8",
    ),
    # --- every protected name, at any depth, and behind the quoting trick ---
    *[
        case
        for name in PROTECTED_FILES
        for case in (
            (f"root {name}", create(name), "protected file"),
            (f"nested {name}", edit(f"src/deep/{name}"), "protected file"),
            (f"{name} behind a quote and junk", trick(f"src/{name}"), "closing quote"),
        )
    ],
    *[
        case
        for name in PROTECTED_DIRS
        for case in (
            (f"{name} directory below the root", edit(f"pkg/{name}/x.txt"), "protected directory"),
            (
                f"{name} directory behind a quote and junk",
                trick(f"pkg/{name}/x.txt"),
                "closing quote",
            ),
        )
    ],
    ("a protected name used as a directory", edit("src/CLAUDE.md/x.txt"), "protected directory"),
    ("the exact quote trick", trick("src/CLAUDE.md"), "closing quote"),
    ("a quote trick on an unprotected name", trick("src/ok.py"), "closing quote"),
    (
        "a quote trick on a rename target",
        'diff --git a/a b/b\nrename from a\nrename to "c"x\n',
        "closing quote",
    ),
    # --- spellings of a protected name ---------------------------------------
    ("mixed case", edit("src/ClAuDe.MD"), "protected file"),
    ("mixed case directory", edit("src/.GiThUb/x"), "protected directory"),
    (
        "full-width letters (NFKC)",
        edit("src/\uff23\uff2c\uff21\uff35\uff24\uff25.md"),
        "protected file",
    ),
    ("a trailing space", edit("src/CLAUDE.md "), "protected file"),
    ("a tab-terminated name", named("a/src/x b/src/x", "a/src/CLAUDE.md\t"), "agree"),
    (
        "a tab-terminated name that is protected on both sides",
        named("a/src/CLAUDE.md b/src/CLAUDE.md", "a/src/CLAUDE.md\t"),
        "protected file",
    ),
    ("text after the tab", named("a/src/x.py b/src/x.py", "a/src/x.py\tjunk"), "after the tab"),
    ("a date after the name", named("a/src/x.py b/src/x.py", "a/src/x.py  2020-01-01"), "agree"),
    (
        "octal escapes that spell a protected name, quoted",
        named('"a/src/CL\\101UDE.md" "b/src/CL\\101UDE.md"', '"a/src/CL\\101UDE.md"'),
        "protected file",
    ),
    ("octal escapes without quotes", edit("src/CL\\101UDE.md"), "backslash"),
    ("an unquoted name with a double quote", edit('src/x"y.py'), "double quote"),
    ("an escape git does not know", named('"a/x\\q" "b/x\\q"', '"a/x\\q"'), "unknown escape"),
    # --- header lines that do not agree --------------------------------------
    ("`---` and `+++` name different files", declared("a/src/x.py", "b/src/y.py"), "agree"),
    (
        "`+++` differs from `diff --git`",
        declared("/dev/null", "b/src/ok.py", "a/notes b/notes.md", "new file mode 100644\n"),
        "agree",
    ),
    ("`---` differs from `diff --git`", declared("a/src/q.py", "b/src/x.py"), "agree"),
    (
        "rename lines differ from `diff --git`",
        "diff --git a/a b/b\nrename from a\nrename to c\n",
        "agree",
    ),
    (
        "`---` differs from `rename from`",
        declared("a/q", "b/b", "a/a b/b", "rename from a\nrename to b\n"),
        "agree",
    ),
    ("`/dev/null` without a new file mode", declared("/dev/null", "b/src/x.py"), "/dev/null"),
    (
        "a new file mode with a real old name",
        declared("a/src/x.py", "b/src/x.py", extra="new file mode 100644\n"),
        "/dev/null",
    ),
    ("`---` without `+++`", "diff --git a/x b/x\n--- a/x\n@@ -1 +1 @@\n-a\n+b\n", "without `+++`"),
    ("`+++` that does not follow `---`", "diff --git a/x b/x\n+++ b/x\n", "does not follow"),
    ("a name with no a/ or b/ prefix", named("a/x b/x", "x"), "prefix"),
    (
        "a hunk before the file names",
        "diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b\n",
        "before the file names",
    ),
    (
        "file names after a hunk, with no new `diff --git`",
        edit("src/a.py") + "--- a/src/b.py\n+++ b/src/b.py\n@@ -1 +1 @@\n-a\n+b\n",
        "after the file names",
    ),
    (
        "a rename line after a hunk",
        edit("src/a.py") + "rename from q\nrename to src/CLAUDE.md\n",
        "after the file names",
    ),
    (
        "a repeated index line",
        edit("src/x.py").replace("index", "index 1..2 100644\nindex", 1),
        "repeated",
    ),
    ("`rename from` without `rename to`", "diff --git a/a b/b\nrename from a\n", "rename"),
    ("a header line too long to be a path", "diff --git " + "a/" + "x " * 4000 + "\n", "too long"),
    # --- a `diff --git` line with no header lines: git keeps its names for the next block ---
    (
        "a diff --git line with nothing after it",
        "diff --git a/src/ok.py b/src/ok.py\n" + edit("src/other.py"),
        "no header lines",
    ),
    ("a lone diff --git line", "diff --git a/src/ok.py b/src/ok.py\n", "no header lines"),
    # --- ordinary files stay editable, by name and under any directory ---
    *[(f"ordinary {name}", edit(name), None) for name in ORDINARY],
    *[(f"ordinary {name} below the root", edit(f"pkg/sub/{name}"), None) for name in ORDINARY],
]


def run(patch: str | bytes) -> tuple[int, str]:
    data = patch if isinstance(patch, bytes) else patch.encode("utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fix.patch"
        path.write_bytes(data)
        out = io.StringIO()
        with redirect_stdout(out):
            code = check.main(["check-fix-patch.py", str(path)])
    return code, out.getvalue()


class CheckFixPatchTest(unittest.TestCase):
    """The checker accepts what a fix may publish and refuses everything else."""

    def test_every_case(self) -> None:
        """Each hand-written patch is accepted or refused for the stated reason."""
        for name, patch, expected in CASES:
            with self.subTest(name):
                code, output = run(patch)
                if expected is None:
                    assert code == 0, output
                    assert "patch accepted" in output
                else:
                    assert code == 1, output
                    assert "::error::The patch is refused" in output
                    assert expected in output

    def test_every_protected_name_is_already_folded(self) -> None:
        """Names are compared after folding; an entry that folds to something else never matches."""
        rules = sys.modules["fix_patch_rules"]
        for name in (*rules.PROTECTED, *rules.PROTECTED_PREFIXES):
            with self.subTest(name):
                assert rules._fold(name) == name

    def test_the_removal_limit_is_three(self) -> None:
        """Three deleted files pass (see the cases above) and four do not."""
        assert check.MAX_REMOVED == 3

    def test_usage_error(self) -> None:
        """Without a patch file there is nothing to judge."""
        with redirect_stderr(io.StringIO()):
            assert check.main(["check-fix-patch.py"]) == 2

    def test_a_flood_of_problems_stops_reading(self) -> None:
        """A huge hostile patch is refused after a bounded amount of reporting."""
        patch = "".join(edit(f".github/f{n}.yml") for n in range(500))
        code, output = run(patch)
        assert code == 1
        assert output.count("::error::") <= check.MAX_PROBLEMS


if __name__ == "__main__":
    unittest.main()
