#!/usr/bin/env python3
"""Tests for the `--applied-names` mode of check-fix-patch.py.

Run: python3 .github/scripts/test_check_fix_applied.py

The input is what `git diff --cached --raw -z` prints after the patch is applied,
so each case here is git's own reading of the change, not the patch text. The
git-backed tests ask `git apply --numstat -z` (it only parses; it writes nothing
and needs no repository) which names git reads from the patch text.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
# The patch-text tests own the checker loader and the hand-written patches.
_SPEC = importlib.util.spec_from_file_location(
    "test_check_fix_patch", Path(__file__).resolve().parent / "test_check_fix_patch.py"
)
assert _SPEC is not None
assert _SPEC.loader is not None
_PATCH_TESTS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PATCH_TESTS)
check = _PATCH_TESTS.check
trick = _PATCH_TESTS.trick
CASES = _PATCH_TESTS.CASES
PROTECTED_FILES = _PATCH_TESTS.PROTECTED_FILES
PROTECTED_DIRS = _PATCH_TESTS.PROTECTED_DIRS
ORDINARY = _PATCH_TESTS.ORDINARY


def entry(status: str, *paths: str, src: str = "100644", dst: str = "100644") -> bytes:
    """One record of `git diff --raw -z --no-abbrev`: metadata, then each path, NUL-ended."""
    meta = f":{src} {dst} {'1' * 40} {'2' * 40} {status}"
    return "".join(f"{field}\0" for field in (meta, *paths)).encode("utf-8")


NEW = {"src": "000000", "dst": "100644"}
GONE = {"src": "100644", "dst": "000000"}

# What `git diff --cached --raw -z` reports after the patch is applied: git's own
# names, so nothing here depends on how the patch text was written.
APPLIED_CASES: list[tuple[str, bytes, str | None]] = [
    ("an edit", entry("M", "src/app.py"), None),
    ("an edit and a new file", entry("M", "src/app.py") + entry("A", "tests/t.py", **NEW), None),
    ("a rename", entry("R100", "src/a.py", "src/b.py"), None),
    ("a rename of an executable", entry("R90", "x.sh", "y.sh", src="100755", dst="100755"), None),
    ("an executable edited in place", entry("M", "run.sh", src="100755", dst="100755"), None),
    (
        "three deletions",
        b"".join(entry("D", f"old/{n}.py", **GONE) for n in "abc"),
        None,
    ),
    ("a nested CLAUDE.md", entry("M", "src/CLAUDE.md"), "protected file"),
    ("a root CLAUDE.local.md", entry("A", "CLAUDE.local.md", **NEW), "protected file"),
    ("a nested AGENTS.md in other case", entry("A", "a/b/agents.MD", **NEW), "protected file"),
    ("a nested .github", entry("A", "pkg/.github/x.yml", **NEW), "protected directory"),
    ("a .cursor directory", entry("A", ".cursor/rules/x.mdc", **NEW), "protected directory"),
    ("a full-width look-alike", entry("A", "\uff23LAUDE.md", **NEW), "protected file"),
    ("a trailing dot", entry("M", "src/.github./x"), "protected directory"),
    (
        "a rename into a protected name",
        entry("R100", "src/a.py", "docs/AGENTS.md"),
        "protected file",
    ),
    ("a rename into .github", entry("R95", "src/a.py", ".github/a.py"), "protected directory"),
    ("a rename out of .github", entry("R100", ".github/a.yml", "a.yml"), "protected directory"),
    ("a copy into a protected name", entry("C100", "src/a.py", "src/CLAUDE.md"), "protected file"),
    ("a deletion of a protected file", entry("D", "docs/CLAUDE.md", **GONE), "protected file"),
    ("a new symlink", entry("A", "link", src="000000", dst="120000"), "symlink or submodule"),
    ("an edited symlink", entry("M", "link", src="120000", dst="120000"), "symlink or submodule"),
    ("a symlink turned into a file", entry("T", "link", src="120000"), "symlink or submodule"),
    ("a gitlink", entry("A", "vendor/lib", src="000000", dst="160000"), "symlink or submodule"),
    (
        "a deleted gitlink",
        entry("D", "vendor/lib", src="160000", dst="000000"),
        "symlink or submodule",
    ),
    ("a new executable", entry("A", "run.sh", src="000000", dst="100755"), "new executable"),
    (
        "a copy that is executable",
        entry("C100", "a", "b", src="100755", dst="100755"),
        "new executable",
    ),
    ("a file that becomes executable", entry("M", "run.sh", dst="100755"), "mode change"),
    ("a file that stops being executable", entry("M", "run.sh", src="100755"), "mode change"),
    ("a type change", entry("T", "x"), "type change"),
    ("an unmerged entry", entry("U", "x"), "status"),
    ("a path with a newline", entry("A", "src/a\nb.py", **NEW), "control character"),
    (
        "a newline that would split a protected name off, kept whole by -z",
        entry("A", "src/ok.py\nCLAUDE.md", **NEW),
        "control character",
    ),
    ("an absolute path", entry("A", "/etc/passwd", **NEW), "absolute path"),
    ("a parent directory", entry("A", "../x", **NEW), "`..`"),
    (
        "four deletions",
        b"".join(entry("D", f"o/{n}", **GONE) for n in "abcd"),
        "deleted or renamed away",
    ),
    ("no entries", b"", "no changed paths"),
    ("a record that is cut off", entry("M", "src/a.py")[:-1], "not end"),
    ("a record with no path", f":100644 100644 {'1' * 40} {'2' * 40} M\0".encode(), "cut off"),
    ("a rename with one path", entry("R100", "only-one"), "cut off"),
    ("text that is not a record", b"src/app.py\0", "cannot read"),
    (
        "a protected path in a second record",
        entry("M", "ok.py") + entry("M", ".claude/x"),
        "protected",
    ),
    # --- every protected name and directory, and the files that stay editable ---
    *[
        (f"{name} at the root", entry("A", name, **NEW), "protected file")
        for name in PROTECTED_FILES
    ],
    *[
        (f"{name} nested", entry("M", f"src/deep/{name}"), "protected file")
        for name in PROTECTED_FILES
    ],
    *[
        (f"{name} directory", entry("A", f"pkg/{name}/x.txt", **NEW), "protected directory")
        for name in PROTECTED_DIRS
    ],
    ("a rename into AGENTS.override.md", entry("R100", "a.md", "AGENTS.override.md"), "protected"),
    ("a rename into .agents", entry("R100", "a.md", ".agents/skills/x/SKILL.md"), "protected"),
    *[(f"ordinary {name}", entry("M", name), None) for name in ORDINARY],
]


def run_applied(data: bytes) -> tuple[int, str]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "names.raw"
        path.write_bytes(data)
        out = io.StringIO()
        with redirect_stdout(out):
            code = check.main(["check-fix-patch.py", "--applied-names", str(path)])
    return code, out.getvalue()


def git_raw(extra: dict[str, bytes]) -> bytes:
    """What `git diff --raw -z` prints for a rename, an edit and the extra new files.

    Two plain directories compared with `--no-index`: git writes nothing and no
    repository is involved, but the output is git's real record format.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for side in ("a", "b"):
            (root / side / "src").mkdir(parents=True)
        body = b"one\ntwo\nthree\nfour\nfive\n"
        (root / "a/src/old.py").write_bytes(body)
        (root / "b/src/new.py").write_bytes(body)
        (root / "a/src/e.py").write_bytes(b"q")
        (root / "b/src/e.py").write_bytes(b"r")
        for name, content in extra.items():
            (root / "b" / name).parent.mkdir(parents=True, exist_ok=True)
            (root / "b" / name).write_bytes(content)
        done = subprocess.run(
            ["git", "diff", "--no-index", "--raw", "-z", "--no-abbrev", "--find-renames", "a", "b"],  # noqa: S607
            cwd=tmp,
            capture_output=True,
            check=False,
        )
    assert done.returncode == 1, done.stderr  # 1 means "the sides differ"
    # `--no-index` prefixes each path with its side; the job's paths have none.
    return done.stdout.replace(b"\0a/", b"\0").replace(b"\0b/", b"\0")


def git_names(patch: str | bytes) -> tuple[int, list[str]]:
    """The names git itself reads from a patch, by `git apply --numstat -z`.

    That command only parses; it writes nothing and needs no repository, and the
    patch file lives in a temporary directory.
    """
    data = patch if isinstance(patch, bytes) else patch.encode("utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "fix.patch").write_bytes(data)
        # Fixed arguments; git is looked up on PATH (the tests skip when it is absent).
        done = subprocess.run(
            ["git", "apply", "--numstat", "-z", "fix.patch"],  # noqa: S607
            cwd=tmp,
            capture_output=True,
            check=False,
        )
    fields = iter(done.stdout.decode("utf-8", "replace").split("\0"))
    names: list[str] = []
    for field in fields:
        if field.count("\t") < 2:
            continue
        name = field.split("\t", 2)[2]
        names += [name] if name else [next(fields), next(fields)]  # a rename: old, new
    return done.returncode, names


class AppliedNamesTest(unittest.TestCase):
    """Git's list of changed paths is judged by the same rules as the patch text."""

    def test_every_applied_case(self) -> None:
        """Git's own list of changed paths is judged by the same rules."""
        for name, data, expected in APPLIED_CASES:
            with self.subTest(name):
                code, output = run_applied(data)
                if expected is None:
                    assert code == 0, output
                    assert "changes accepted" in output
                else:
                    assert code == 1, output
                    assert "::error::The applied changes are refused" in output
                    assert expected in output

    @unittest.skipUnless(shutil.which("git"), "git is not installed")
    def test_real_git_output_is_read_as_written(self) -> None:
        """A rename and an edit pass; a protected file or a newline name added beside them does not."""
        code, output = run_applied(git_raw({}))
        assert code == 0, output
        assert "applied changes accepted: 3 path(s), 1 removed" in output
        for extra, phrase in (
            ("src/sub/CLAUDE.local.md", "protected file"),
            ("src/ok.py\nCLAUDE.md", "control character"),
            (".husky/pre-commit", "protected directory"),
        ):
            with self.subTest(extra):
                code, output = run_applied(git_raw({extra: b"x"}))
                assert code == 1, output
                assert phrase in output

    def test_the_script_runs_from_another_directory(self) -> None:
        """Run the way the workflow runs it: by path, from a different working directory."""
        script = Path(check.__file__).resolve()
        with tempfile.TemporaryDirectory() as tmp:
            # Our own interpreter and the checked-in script: nothing untrusted.
            done = subprocess.run(  # noqa: S603
                [sys.executable, "-B", str(script), "--applied-names", "-"],
                cwd=tmp,
                input=entry("M", "src/deep/CLAUDE.local.md"),
                capture_output=True,
                check=False,
            )
        assert done.returncode == 1, done.stderr
        assert b"protected file" in done.stdout

    def test_the_applied_names_can_come_from_standard_input(self) -> None:
        """`--applied-names -` reads the raw bytes that git writes to a pipe."""
        stdin = io.TextIOWrapper(io.BytesIO(entry("M", "src/CLAUDE.md")))
        out = io.StringIO()
        old, sys.stdin = sys.stdin, stdin
        try:
            with redirect_stdout(out):
                code = check.main(["check-fix-patch.py", "--applied-names", "-"])
        finally:
            sys.stdin = old
        assert code == 1
        assert "protected file" in out.getvalue()

    @unittest.skipUnless(shutil.which("git"), "git is not installed")
    def test_git_reads_the_names_the_checker_accepted(self) -> None:
        """For every patch accepted here, git's names are ordinary files."""
        for name, patch, expected in CASES:
            if expected is not None:
                continue
            with self.subTest(name):
                code, names = git_names(patch)
                assert code == 0, names
                assert names, "git read no file names"
                for found in names:
                    assert check.path_problem(found.encode("utf-8")) is None, found

    @unittest.skipUnless(shutil.which("git"), "git is not installed")
    def test_the_quote_trick_reaches_a_protected_file_in_git(self) -> None:
        """Git reads the trick patch as `src/CLAUDE.md`; the applied-names check stops it."""
        code, names = git_names(trick("src/CLAUDE.md"))
        assert code == 0
        assert names == ["src/CLAUDE.md"]
        raw = entry("A", *names, **NEW)
        assert run_applied(raw)[0] == 1


if __name__ == "__main__":
    unittest.main()
