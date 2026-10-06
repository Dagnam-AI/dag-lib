#!/usr/bin/env python3
"""Refuse an auto-fix patch that could do more than repair code.

The auto-fix workflow's `publish` job runs this twice. The patch is the output of
a job that ran a pull request's own code and a model, so it is data to inspect,
not a change to trust. A fix for a red check touches source and tests. It has no
reason to touch what runs on the owner's machine when the branch is opened (git
hooks, agent settings and instruction files, tool configuration), what runs in CI
(`.github/`), or anything that is not a plain text file edit.

1. `check-fix-patch.py PATCH_FILE` reads the patch text before anything is applied.
2. `check-fix-patch.py --applied-names FILE` (`-` is standard input) reads git's
   own list of what the applied patch changed, `git diff --cached --raw -z`. Git
   is the authority on names: a parser of patch text and `git apply` can read the
   same text differently, but the list git prints cannot disagree with what git
   will commit. This is the check that decides.

Both apply one rule set, `path_problem` in fix_patch_rules.py (which sits beside
this file and is imported from the directory this file is run from), and refuse:

* any path component, at any depth, that is in PROTECTED (agent instruction and
  tool configuration files and directories) or starts with a PROTECTED_PREFIXES
  entry. Names are compared the way a case-insensitive, Unicode-normalising file
  system would (macOS, Windows), and `NAME~1` short-name aliases are refused;
* an absolute path, a `..` or empty component, a backslash, a colon or a control
  character in a path: a path that is not plainly inside the repository;
* a symlink or submodule, a new executable, and any mode change;
* more than MAX_REMOVED files deleted or renamed away.

The patch text check is an early, cheap refusal and is not the authority: git reads
a few constructions differently (for example a `diff --git` line with nothing after
it), so only the applied-names check may be trusted to see every name. It also refuses a binary patch and anything it does not
understand: the patch must be well-formed git diff text whose hunks match their
declared line counts, whose header lines each have exactly the form git writes
(a quoted name ends at its closing quote, a repeated header line is refused), and
whose `diff --git`, `---`, `+++` and rename or copy names agree.

Exit status: 0 acceptable, 1 refused (reasons on stdout as ::error:: lines), 2
usage. Standard library only.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

from fix_patch_rules import (
    EDITABLE_MODES,
    MAX_REMOVED,
    NEW_FILE_MODE,
    path_problem,
    scan_applied,
)

# Reading stops after this many problems: the patch is refused either way.
MAX_PROBLEMS = 20
# A `diff --git` line longer than this is not two paths; reading it is quadratic.
MAX_HEADER = 4096

_HUNK = re.compile(rb"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_INDEX = re.compile(rb"^index [0-9a-f]+\.\.[0-9a-f]+(?: (\d{6}))?$")
_SIMILARITY = re.compile(rb"^(?:dis)?similarity index \d+%$")
_FILE_MODE = re.compile(rb"^(new file|deleted file) mode (\d{6})$")
_ESCAPES = {b"a": 7, b"b": 8, b"f": 12, b"n": 10, b"r": 13, b"t": 9, b"v": 11, b'"': 34, b"\\": 92}
_MOVES = (b"rename from ", b"rename to ", b"copy from ", b"copy to ")
# Every header line git reads before a hunk; after the file names none may appear.
_HEADER_PREFIXES = (
    b"old mode ",
    b"new mode ",
    b"new file mode ",
    b"deleted file mode ",
    b"index ",
    b"similarity index ",
    b"dissimilarity index ",
    b"rename ",
    b"copy ",
    b"--- ",
    b"+++ ",
    b"GIT binary patch",
    b"Binary files ",
)


class RefusedError(Exception):
    """The patch is outside what an auto-fix may publish."""


def _quoted(text: bytes) -> tuple[bytes, bytes]:
    """Decode the C-quoted string that ``text`` starts with; return it and the rest.

    git ends the name at the closing quote and ignores what follows it, so the
    rest is returned for the caller to refuse rather than dropped.
    """
    out, i = bytearray(), 1
    while i < len(text):
        char = text[i : i + 1]
        if char == b'"':
            return bytes(out), text[i + 1 :]
        if char != b"\\":
            out += char
            i += 1
            continue
        octal = re.match(rb"[0-3][0-7]{2}", text[i + 1 : i + 4])
        escaped = text[i + 1 : i + 2]
        if octal:
            out.append(int(octal[0], 8))
            i += 4
        elif escaped in _ESCAPES:
            out.append(_ESCAPES[escaped])
            i += 2
        else:
            raise RefusedError(f"unknown escape in a quoted path: {text[:80]!r}")
    raise RefusedError(f"a quoted path that never closes: {text[:80]!r}")


def _name(token: bytes, *, tab_ok: bool = False) -> bytes:
    """The one path a header line carries, in exactly the forms git writes.

    Either a fully quoted string, or plain text with no double quote (git quotes
    any name that has one). On a `---` or `+++` line git ends a name that holds a
    space with one tab; nothing may follow it.
    """
    if token.startswith(b'"'):
        name, tail = _quoted(token)
        if tail not in ((b"", b"\t") if tab_ok else (b"",)):
            raise RefusedError(f"text after the closing quote of a quoted path: {token[:80]!r}")
        return name
    if tab_ok and token.endswith(b"\t"):
        token = token[:-1]
    if b'"' in token:
        raise RefusedError(f"a double quote in an unquoted path: {token[:80]!r}")
    if b"\t" in token:
        raise RefusedError(f"text after the tab in a path: {token[:80]!r}")
    return token


def _strip_prefix(name: bytes) -> bytes:
    """Remove the leading `a/` or `b/` (git apply's -p1)."""
    _, slash, rest = name.partition(b"/")
    if not slash:
        raise RefusedError(f"a path with no `a/` or `b/` prefix: {name[:80]!r}")
    return rest


def _header_pairs(rest: bytes) -> list[tuple[bytes, bytes]]:
    """Every (old, new) pair of names the text after `diff --git ` could be read as.

    A name may contain spaces, so the split between the two names is ambiguous
    in general. All splits are returned and all are checked: a name only has to
    look protected under one reading to refuse the patch.
    """
    if len(rest) > MAX_HEADER:
        raise RefusedError("the line is too long to be two paths")
    if rest.startswith(b'"'):
        first, tail = _quoted(rest)
        if not tail.startswith(b" "):
            raise RefusedError("no second name after the first quoted name")
        return [(_strip_prefix(first), _strip_prefix(_name(tail[1:])))]
    pairs = []
    for k, char in enumerate(rest):
        if char == 0x20:
            try:
                pairs.append((_strip_prefix(_name(rest[:k])), _strip_prefix(_name(rest[k + 1 :]))))
            except RefusedError:
                continue  # not a reading that is a pair of git-written names
    if not pairs:
        raise RefusedError("no way to read it as two names")
    return pairs


class _Scan:
    """What one pass over the patch found."""

    def __init__(self) -> None:
        self.problems: list[str] = []
        self.removed = 0
        self.blocks = 0
        self.stopped = False

    def path(self, name: bytes) -> None:
        found = path_problem(name)
        if found:
            self.problems.append(f"{name.decode('utf-8', 'replace')!r}: {found}")

    def fail(self, text: str) -> None:
        self.problems.append(text)


class _Section:
    """The header lines of one `diff --git` block, judged together when it ends."""

    def __init__(self, pairs: list[tuple[bytes, bytes]] | None) -> None:
        self.pairs = pairs  # None: the `diff --git` line itself was unreadable
        self.moves: dict[str, bytes] = {}  # "rename from" and so on, to the name
        self.names: dict[str, bytes | None] = {}  # "---" and "+++"; None is /dev/null
        self.seen: set[str] = set()
        self.lines = 0  # header lines after the `diff --git` line
        self.new_file = False
        self.deleted = False
        self.old_line = 0  # line number of the `---` line
        self.named = False  # both `---` and `+++` read

    def once(self, key: str, n: int, scan: _Scan) -> bool:
        """True the first time; a header line git reads only once must not repeat."""
        if key in self.seen:
            scan.fail(f"line {n}: a repeated header line ({key})")
            return False
        self.seen.add(key)
        return True


def _agree(sec: _Section, scan: _Scan) -> None:
    """The names git would read must be the names every header line gives."""
    if sec.pairs is None:
        return
    if not sec.lines:
        # git skips a `diff --git` line that nothing follows, but keeps its names
        # for the next block, so the two readers would pair names differently.
        scan.fail("a `diff --git` line with no header lines after it")
        return
    if "---" in sec.names and not sec.named:
        scan.fail("a `---` line without `+++`")
        return
    old, new = sec.names.get("---"), sec.names.get("+++")
    if sec.named and ((old is None) != sec.new_file or (new is None) != sec.deleted):
        scan.fail("`/dev/null` and the new file or deleted file mode line do not match")
        return
    kinds = {key.split()[0] for key in sec.moves}
    expected: tuple[bytes, bytes] | None = None
    consistent = True
    if kinds:
        kind = kinds.pop()
        source, target = sec.moves.get(f"{kind} from"), sec.moves.get(f"{kind} to")
        if kinds or source is None or target is None:
            scan.fail("rename or copy lines that are not one `from` and one `to`")
            return
        expected = (source, target)
        consistent = not sec.named or (old, new) == expected
    elif sec.named:
        path = new if new is not None else old
        if path is not None:
            expected = (path, path)
        consistent = old in (None, path) and new in (None, path)
    found = (expected in sec.pairs) if expected else any(o == n for o, n in sec.pairs)
    if not (consistent and found):
        scan.fail(
            "the names in `diff --git`, `---`, `+++` and the rename or copy lines do not agree"
        )


def _hunk(lines: list[bytes], i: int, scan: _Scan) -> int:
    """Consume one hunk (header at ``lines[i]``) by its declared counts."""
    found = _HUNK.match(lines[i])
    if not found:
        scan.fail(f"line {i + 1}: a hunk header git apply would not accept")
        return len(lines)
    old = int(found[2]) if found[2] is not None else 1
    new = int(found[4]) if found[4] is not None else 1
    i += 1
    while old > 0 or new > 0:
        if i >= len(lines):
            scan.fail("the patch ends inside a hunk")
            return len(lines)
        lead = lines[i][:1]
        if lead in (b" ", b"") and old > 0 and new > 0:
            old, new = old - 1, new - 1
        elif lead == b"-" and old > 0:
            old -= 1
        elif lead == b"+" and new > 0:
            new -= 1
        elif lead != b"\\":
            scan.fail(f"line {i + 1}: hunk content does not match the declared line counts")
            return len(lines)
        i += 1
    while i < len(lines) and lines[i][:1] == b"\\":
        i += 1
    return i


def _declared(rest: bytes, scan: _Scan) -> bytes | None:
    """The name on a `---` or `+++` line, without prefix; None for /dev/null."""
    if rest in (b"/dev/null", b"/dev/null\t"):
        return None
    name = _strip_prefix(_name(rest, tab_ok=True))
    scan.path(name)
    return name


def _header_line(line: bytes, n: int, scan: _Scan, sec: _Section) -> None:
    """Judge one line of a file's header section."""
    sec.lines += 1
    if sec.named and line.startswith(_HEADER_PREFIXES):
        # git apply reads a header line after a hunk as the start of another change.
        scan.fail(f"line {n}: a header line after the file names")
        return
    mode = _FILE_MODE.match(line)
    index = _INDEX.match(line)
    if line.startswith((b"old mode ", b"new mode ")):
        scan.fail(f"line {n}: a mode change")
    elif mode:
        allowed = (NEW_FILE_MODE,) if mode[1] == b"new file" else EDITABLE_MODES
        if mode[2].decode() not in allowed:
            scan.fail(
                f"line {n}: {mode[1].decode()} mode {mode[2].decode()} "
                "(a symlink, submodule or executable)"
            )
        if sec.once("file mode", n, scan):
            sec.new_file, sec.deleted = mode[1] == b"new file", mode[1] == b"deleted file"
            scan.removed += sec.deleted
    elif index:
        sec.once("index", n, scan)
        if index[1] and index[1].decode() not in EDITABLE_MODES:
            scan.fail(f"line {n}: the file is mode {index[1].decode()} (a symlink or submodule)")
    elif _SIMILARITY.match(line):
        sec.once("similarity", n, scan)
    elif line.startswith(_MOVES):
        key = next(move for move in _MOVES if line.startswith(move)).decode().strip()
        if sec.once(key, n, scan):
            sec.moves[key] = _name(line[len(key) + 1 :])
            scan.path(sec.moves[key])
            scan.removed += key == "rename from"
    elif line.startswith(b"--- "):
        if sec.once("---", n, scan):
            sec.names["---"], sec.old_line = _declared(line[4:], scan), n
    elif line.startswith(b"+++ "):
        if sec.old_line != n - 1:
            scan.fail(f"line {n}: a `+++` line that does not follow a `---` line")
        elif sec.once("+++", n, scan):
            sec.names["+++"], sec.named = _declared(line[4:], scan), True
    elif line.startswith((b"GIT binary patch", b"Binary files ")):
        scan.fail(f"line {n}: a binary patch")
        scan.stopped = True  # what follows is encoded data, not lines to judge
    else:
        scan.fail(f"line {n}: a line this check does not understand: {line[:60]!r}")


def scan_patch(data: bytes) -> _Scan:
    """Read the whole patch; the result lists every problem found."""
    scan = _Scan()
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    if not lines or not lines[0].startswith(b"diff --git "):
        scan.fail("the patch does not start with a `diff --git` header")
        return scan
    sec = _Section(None)
    i = 0
    while i < len(lines) and not scan.stopped and len(scan.problems) < MAX_PROBLEMS:
        line = lines[i]
        if line.startswith(b"diff --git "):
            _agree(sec, scan)
            scan.blocks += 1
            pairs: list[tuple[bytes, bytes]] | None = None
            try:
                pairs = _header_pairs(line[len(b"diff --git ") :])
            except RefusedError as why:
                scan.fail(f"line {i + 1}: cannot read the names in `diff --git`: {why}")
            for pair in pairs or []:
                for name in pair:
                    scan.path(name)
            sec = _Section(pairs)
            i += 1
        elif line.startswith(b"@@"):
            if not sec.named:
                scan.fail(f"line {i + 1}: a hunk before the file names")
            i = _hunk(lines, i, scan)
        else:
            try:
                _header_line(line, i + 1, scan, sec)
            except RefusedError as why:
                scan.fail(f"line {i + 1}: {why}")
            i += 1
    _agree(sec, scan)
    if scan.removed > MAX_REMOVED:
        scan.fail(f"{scan.removed} files deleted or renamed away; the limit is {MAX_REMOVED}")
    return scan


def _refuse(problems: list[str], subject: str) -> int:
    for problem in dict.fromkeys(problems):
        sys.stdout.write(f"::error::{subject}: {problem}\n")
    sys.stdout.write("An auto-fix may only edit ordinary files outside protected paths.\n")
    return 1


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "--applied-names":
        data = sys.stdin.buffer.read() if argv[2] == "-" else Path(argv[2]).read_bytes()
        problems, count, removed = scan_applied(data)
        if problems:
            return _refuse(problems, "The applied changes are refused")
        sys.stdout.write(f"applied changes accepted: {count} path(s), {removed} removed\n")
        return 0
    if len(argv) != 2:
        sys.stderr.write(__doc__ or "")
        return 2
    scan = scan_patch(Path(argv[1]).read_bytes())
    if scan.problems:
        return _refuse(scan.problems, "The patch is refused")
    sys.stdout.write(f"patch accepted: {scan.blocks} file(s), {scan.removed} removed\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
