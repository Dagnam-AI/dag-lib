"""Regression: ``import dagnam`` must not eagerly load the async/httpx stack."""

from __future__ import annotations

import subprocess
import sys

import pytest


def _run(code: str) -> None:
    result = subprocess.run(  # noqa: S603 - fixed interpreter + literal script, no untrusted input
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, (
        f"subprocess failed (rc={result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_bare_import_does_not_load_httpx_or_aio() -> None:
    _run(
        "import sys, dagnam\n"
        "assert 'httpx' not in sys.modules, sorted(m for m in sys.modules if 'httpx' in m)\n"
        "assert 'dagnam._core.aio' not in sys.modules\n"
        "assert 'dagnam._core.aio.base' not in sys.modules\n"
    )


def test_async_client_resolves_lazily_via_top_level() -> None:
    _run(
        "import sys, dagnam\n"
        "assert 'httpx' not in sys.modules\n"
        "client_cls = dagnam.AsyncDagnamClient\n"
        "assert client_cls.__name__ == 'AsyncDagnamClient'\n"
        "assert 'httpx' in sys.modules\n"
        "assert 'dagnam._core.aio.base' in sys.modules\n"
    )


def test_async_client_resolves_lazily_via_core() -> None:
    _run(
        "import sys\n"
        "import dagnam._core as core\n"
        "assert 'httpx' not in sys.modules\n"
        "assert core.AsyncDagnamClient.__name__ == 'AsyncDagnamClient'\n"
        "assert 'httpx' in sys.modules\n"
    )


def test_core_getattr_rejects_unknown_name() -> None:
    import dagnam._core as core

    with pytest.raises(AttributeError, match="does_not_exist"):
        core.does_not_exist  # noqa: B018


def test_core_getattr_loads_async_client_in_process() -> None:
    """Cover ``__getattr__``'s lazy-load body in-process.

    The subprocess-based tests above run in a child interpreter, so the lines
    they execute in ``dagnam._core.__getattr__`` are not seen by this process's
    coverage. Accessing the attribute here (in-process) exercises the
    import/resolve/cache success path and the falsy branch of the name guard.
    """
    import dagnam._core as core

    assert core.AsyncDagnamClient.__name__ == "AsyncDagnamClient"


def test_aio_submodule_still_eager() -> None:
    _run(
        "import sys, dagnam.aio\n"
        "assert dagnam.aio.AsyncDagnamClient.__name__ == 'AsyncDagnamClient'\n"
        "assert 'httpx' in sys.modules\n"
    )
