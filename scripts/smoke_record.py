#!/usr/bin/env python3
"""Release smoke test: the INSTALLED ``dagnam-record`` is the version being released and works.

Run it with the interpreter of a clean environment that holds only the installed wheel, sdist or
index package, never from a checkout of ``record/``::

    python scripts/smoke_record.py 0.1.0

It checks that the module, the installed metadata and the expected version agree, that the typed
marker ships and the tests do not, then records one call into a temporary directory and runs
``python -m dagnam_record status`` and ``purge`` against it. Standard library only; the recorder
needs no network and none is used.
"""

from __future__ import annotations

import importlib.metadata
import importlib.resources
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def _run(*arguments: str, home: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - the running interpreter and fixed arguments
        [sys.executable, "-m", "dagnam_record", *arguments],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={**os.environ, "HOME": home, "USERPROFILE": home},
    )


def smoke(expected: str) -> list[str]:
    """What is wrong with the installed ``dagnam-record``, as readable lines (empty when fine)."""
    import dagnam_record  # the point of the test is the installed copy

    wrong: list[str] = []
    installed = importlib.metadata.version("dagnam-record")
    if not dagnam_record.__version__ == installed == expected:
        wrong.append(
            f"version: module {dagnam_record.__version__}, metadata {installed}, wanted {expected}"
        )
    if not importlib.resources.files("dagnam_record").joinpath("py.typed").is_file():
        wrong.append("py.typed is not installed")
    shipped = [
        str(f) for f in importlib.metadata.files("dagnam-record") or () if "tests" in f.parts
    ]
    if shipped:
        wrong.append(f"test files are installed: {shipped[:3]}")

    with tempfile.TemporaryDirectory() as work:
        traces = str(Path(work) / "traces")
        if not dagnam_record.install(dir=traces) and not dagnam_record.is_installed():
            wrong.append("install() did not set the recorder up")
        dagnam_record.record_call(
            provider="openai",
            endpoint="chat.completions",
            request={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            response={
                "model": "m",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            },
        )
        dagnam_record.uninstall()  # drains the writer before the commands read the directory
        status = _run("status", "--dir", traces, home=work)
        if status.returncode != 0 or "written: 1" not in status.stdout:
            wrong.append(f"status exited {status.returncode}: {status.stdout!r} {status.stderr!r}")
        dry = _run("purge", "--dry-run", "--dir", traces, home=work)
        if dry.returncode != 0 or "would delete" not in dry.stdout:
            wrong.append(f"purge --dry-run exited {dry.returncode}: {dry.stdout!r} {dry.stderr!r}")
        purged = _run("purge", "--dir", traces, home=work)
        gone = _run("status", "--dir", traces, home=work)
        if purged.returncode != 0 or gone.returncode != 1:
            wrong.append(
                f"purge exited {purged.returncode}, status after it {gone.returncode} (wanted 0 and 1)"
            )
    return wrong


def main(argv: list[str] | None = None) -> int:
    """Exit status: 0 when the installed package passes, else 1 with every reason on stderr."""
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        sys.stderr.write("usage: smoke_record.py <expected version>\n")
        return 2
    wrong = smoke(arguments[0])
    for line in wrong:
        sys.stderr.write(f"{line}\n")
    if not wrong:
        sys.stdout.write(f"dagnam-record {arguments[0]} installed and working\n")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
