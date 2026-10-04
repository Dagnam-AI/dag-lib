"""``scripts/check_platform.py``: the release gate that asks the platform which contract it runs.

Driven over real HTTP against a local server, so the request, the timeout and
the JSON read are the script's own and nothing here is a mock of ``urllib``.
"""

from __future__ import annotations

from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import re
import socket
import threading
from typing import Any, override

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_platform.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("check_platform", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_platform = _load()


class Platform:
    """A local ``/health/build``: ``status`` and ``body`` are what it answers next."""

    def __init__(self) -> None:
        self.url = ""
        self.reset()

    def reset(self) -> None:
        self.status = 200
        self.body: bytes = b"{}"
        self.paths: list[str] = []
        self.agents: list[str] = []
        self.redirect: str | None = None
        """Where ``/health/build`` redirects to, when set."""


@pytest.fixture(scope="module")
def server() -> Iterator[Platform]:
    """One listening socket for the module; each test resets what it answers."""
    fake = Platform()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            fake.paths.append(self.path)
            fake.agents.append(self.headers.get("User-Agent", ""))
            if fake.redirect is not None and self.path == "/health/build":
                self.send_response(302)
                self.send_header("Location", fake.redirect)
                self.end_headers()
                return
            self.send_response(fake.status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(fake.body)

        @override
        def log_message(self, format: str, *args: object) -> None:
            """Keep the test output free of one access-log line per request."""

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    fake.url = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield fake
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@pytest.fixture
def platform(server: Platform) -> Platform:
    server.reset()
    return server


@pytest.fixture
def pyproject(tmp_path: Path) -> Path:
    path = tmp_path / "pyproject.toml"
    path.write_text(
        '[project]\nname = "x"\ndependencies = [\n'
        '    "requests>=2.32.0",\n    "dagnam-contracts>=0.4.0,<0.5",\n]\n',
        encoding="utf-8",
    )
    return path


def _run(platform: Platform, pyproject: Path) -> int:
    return int(check_platform.main([platform.url, "--pyproject", str(pyproject)]))


def test_the_floor_is_read_from_the_declared_contract_requirement(pyproject: Path) -> None:
    assert check_platform.contract_floor(pyproject) == (0, 4)
    # The repository's own declaration parses too, whatever its floor is today.
    major, minor = check_platform.contract_floor(ROOT / "pyproject.toml")
    assert (major, minor) >= (0, 4)


def test_a_project_that_does_not_declare_the_contract_cannot_be_checked(tmp_path: Path) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text('[project]\nname = "x"\ndependencies = ["requests>=2"]\n', encoding="utf-8")
    with pytest.raises(SystemExit, match=r"declares no dagnam-contracts>=X\.Y floor"):
        check_platform.contract_floor(path)


@pytest.mark.parametrize("contracts", ["0.4.0", "0.4.7", "0.5.0", "1.0.0", "0.4.0rc1"])
def test_a_platform_at_or_ahead_of_the_floor_passes(
    platform: Platform, pyproject: Path, capsys: pytest.CaptureFixture[str], contracts: str
) -> None:
    platform.body = json.dumps({"revision": "abc", "version": "1", "contracts": contracts}).encode()
    assert _run(platform, pyproject) == 0
    assert platform.paths == ["/health/build"]
    assert f"runs dagnam-contracts {contracts}" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("status", "body", "says"),
    [
        # Behind the floor: the release would ship an SDK its platform cannot serve.
        (
            200,
            {"revision": "abc", "version": "1", "contracts": "0.3.1"},
            "runs dagnam-contracts 0.3.1",
        ),
        # What the platform answered before it reported the key.
        (200, {"revision": "abc", "version": "1"}, "does not report its dagnam-contracts version"),
        (200, {"contracts": "unknown"}, "does not report its dagnam-contracts version"),
        (200, ["not", "an", "object"], "does not report its dagnam-contracts version"),
        # A platform from before the route, and one that is down: neither proves it is ready.
        (404, {"detail": "Not Found"}, "could not be read"),
        (503, {"detail": "down"}, "could not be read"),
    ],
)
def test_a_platform_behind_the_floor_or_unable_to_say_fails_the_release(
    platform: Platform,
    pyproject: Path,
    capsys: pytest.CaptureFixture[str],
    status: int,
    body: object,
    says: str,
) -> None:
    platform.status, platform.body = status, json.dumps(body).encode()
    assert _run(platform, pyproject) == 1
    err = capsys.readouterr().err
    assert says in err
    assert "this release needs 0.4 or newer" in err
    assert "Deploy the platform first" in err


def test_a_body_that_is_not_json_fails_the_release(
    platform: Platform, pyproject: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    platform.body = b"<html>gateway</html>"
    assert _run(platform, pyproject) == 1
    assert "could not be read" in capsys.readouterr().err


def test_an_unreachable_platform_fails_the_release(
    pyproject: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Port 9 on loopback: nothing listens, so the connection is refused at once.
    assert check_platform.main(["http://127.0.0.1:9", "--pyproject", str(pyproject)]) == 1
    assert "could not be read" in capsys.readouterr().err


def test_the_request_names_itself_and_asks_for_json(platform: Platform, pyproject: Path) -> None:
    """Bot filters in front of an API commonly refuse the default ``Python-urllib`` agent."""
    platform.body = b'{"contracts": "0.4.0"}'
    assert _run(platform, pyproject) == 0
    assert platform.agents == ["dagnam-release-gate"]


def test_a_redirect_on_the_platforms_own_host_is_followed(
    platform: Platform, pyproject: Path
) -> None:
    platform.redirect = "/health/build?via=redirect"
    platform.body = b'{"contracts": "0.4.0"}'
    assert _run(platform, pyproject) == 0
    assert platform.paths == ["/health/build", "/health/build?via=redirect"]


def test_a_redirect_to_another_host_fails_the_release_and_is_not_followed(
    platform: Platform, pyproject: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A gate that reads a version off wherever a redirect leads proves nothing about the platform."""
    reached: list[str] = []

    class Other(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            reached.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"contracts": "9.9.9"}')

        @override
        def log_message(self, format: str, *args: object) -> None:
            """Quiet."""

    elsewhere = ThreadingHTTPServer(("127.0.0.1", 0), Other)
    thread = threading.Thread(target=elsewhere.serve_forever, daemon=True)
    thread.start()
    try:
        platform.redirect = f"http://127.0.0.1:{elsewhere.server_address[1]}/health/build"
        assert _run(platform, pyproject) == 1
    finally:
        elsewhere.shutdown()
        elsewhere.server_close()
        thread.join(timeout=5)
    assert reached == []
    assert "redirected to another host" in capsys.readouterr().err


def test_a_platform_that_never_answers_fails_the_release_within_the_timeout(
    pyproject: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The request has a timeout: a hung platform cannot hang the release job until the runner's limit."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)  # accepts the connection in the kernel and never replies
    monkeypatch.setattr(check_platform, "TIMEOUT_SECONDS", 0.3)
    try:
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        assert check_platform.main([url, "--pyproject", str(pyproject)]) == 1
    finally:
        listener.close()
    assert "could not be read" in capsys.readouterr().err


def test_a_requirement_that_only_starts_with_the_name_is_not_the_contract(tmp_path: Path) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text(
        '[project]\nname = "x"\ndependencies = [\n    "dagnam-contracts-extra>=9.9",\n'
        '    "dagnam-contracts[x]>=0.4.0,<0.5",\n]\n',
        encoding="utf-8",
    )
    assert check_platform.contract_floor(path) == (0, 4)
    path.write_text(
        '[project]\nname = "x"\ndependencies = ["dagnam-contracts-extra>=9.9"]\n', encoding="utf-8"
    )
    with pytest.raises(SystemExit, match=r"declares no dagnam-contracts>=X\.Y floor"):
        check_platform.contract_floor(path)


def test_a_requirement_with_no_lower_bound_is_not_a_floor(tmp_path: Path) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text(
        '[project]\nname = "x"\ndependencies = ["dagnam-contracts<0.5"]\n', encoding="utf-8"
    )
    with pytest.raises(SystemExit, match=r"declares no dagnam-contracts>=X\.Y floor"):
        check_platform.contract_floor(path)


def test_the_script_uses_the_standard_library_only() -> None:
    """The release job runs it on a bare runner, before anything is installed."""
    imported = {
        line.split()[1].split(".")[0]
        for line in SCRIPT.read_text(encoding="utf-8").splitlines()
        if line.startswith(("import ", "from "))
    }
    assert imported <= {
        "__future__",
        "argparse",
        "http",
        "json",
        "pathlib",
        "re",
        "sys",
        "tomllib",
        "typing",
        "urllib",
    }


WORKFLOWS = ROOT / ".github" / "workflows"


def _jobs(workflow: str) -> dict[str, str]:
    """Each job's block of a workflow file, by job id: the text under ``jobs:`` split at its keys."""
    blocks: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in workflow.split("\njobs:\n", 1)[1].splitlines():
        job = re.fullmatch(r"  ([\w-]+):", line)
        if job:
            current = blocks.setdefault(job[1], [])
        elif current is not None:
            current.append(line)
    return {name: "\n".join(lines) for name, lines in blocks.items()}


def _needs(job: str) -> set[str]:
    """The job ids a job waits for: ``needs: a``, ``needs: [a, b]`` or a block list."""
    lines = job.splitlines()
    for index, line in enumerate(lines):
        declared = re.fullmatch(r"    needs:\s*(.*)", line)
        if declared is None:
            continue
        if declared[1]:
            return set(re.findall(r"[\w-]+", declared[1]))
        found: set[str] = set()
        for item in lines[index + 1 :]:
            listed = re.fullmatch(r"      - ([\w-]+)", item)
            if listed is None:
                break
            found.add(listed[1])
        return found
    return set()


def _reaches(jobs: dict[str, str], name: str, target: str) -> bool:
    """Whether ``name`` cannot start before ``target`` has succeeded, through any chain of ``needs``."""
    waiting = _needs(jobs[name])
    return target in waiting or any(_reaches(jobs, other, target) for other in waiting)


def test_the_needs_reader_reads_every_spelling() -> None:
    assert _needs("    needs: platform\n") == {"platform"}
    assert _needs("    needs: [platform, ci]\n") == {"platform", "ci"}
    assert _needs("    runs-on: x\n    needs:\n      - a\n      - b-c\n    steps: []") == {
        "a",
        "b-c",
    }
    assert _needs("    runs-on: x\n") == set()


PLATFORM_STEP = "        run: python scripts/check_platform.py https://api.dagnam.ai"


def _release_problems(workflow: str) -> list[str]:
    """What is wrong with a release workflow, as far as the gate being un-skippable goes.

    What must hold, whatever else the workflow grows: the platform check is a first job with no
    ``needs`` that runs the script; the tagged commit's own CI is run; ``build`` waits for both;
    every job that publishes or installs the release waits, through some chain, for all three;
    and no job can be skipped or allowed to fail.
    """
    jobs = _jobs(workflow)
    problems: list[str] = []
    if _needs(jobs["platform"]):
        problems.append("platform is not a first job")
    if PLATFORM_STEP not in jobs["platform"].splitlines():  # the whole line: no suffix, no comment
        problems.append("platform does not run the script")
    if "uses: ./.github/workflows/dag-lib-ci.yml" not in jobs["ci"]:
        problems.append("ci does not call the CI workflow")
    if not _needs(jobs["build"]) >= {"platform", "ci"}:
        problems.append("build does not wait for platform and ci")
    for name in jobs:
        if name in {"platform", "ci", "build"}:
            continue
        problems += [
            f"{name} can start before {gate}"
            for gate in ("build", "platform", "ci")
            if not _reaches(jobs, name, gate)
        ]
    for name, job in jobs.items():
        # At the job or at any step: a skipped or tolerated step is as good as none.
        if re.search(r"^\s+(if|continue-on-error):", job, re.M):
            problems.append(f"{name} can be skipped or fail without failing the release")
    return problems


def test_the_release_workflow_cannot_publish_without_the_gate_and_ci() -> None:
    workflow = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
    assert _release_problems(workflow) == []
    assert len(_jobs(workflow)) > 3  # and there is something after the build to protect


@pytest.mark.parametrize(
    ("old", "new", "problem"),
    [
        ("    needs: [platform, ci]\n", "    needs: [platform]\n", "build does not wait"),
        ("    needs: [platform, ci]\n", "    needs: [ci]\n", "build does not wait"),
        (
            "    name: Publish to TestPyPI\n    needs: build\n",
            "    name: Publish to TestPyPI\n    needs: []\n",
            "publish-testpypi can start before",
        ),
        (
            "    name: Publish to PyPI\n    needs: smoke-testpypi\n",
            "    name: Publish to PyPI\n",
            "publish-pypi can start before",
        ),
        (
            "    name: CI on the tagged commit\n",
            "    name: CI on the tagged commit\n    if: false\n",
            "ci can be skipped",
        ),
        (
            "    timeout-minutes: 15\n",
            "    timeout-minutes: 15\n    continue-on-error: true\n",
            "build can be skipped",
        ),
        (
            "    name: Platform runs this release's contract\n",
            "    name: Platform runs this release's contract\n    needs: ci\n",
            "platform is not a first job",
        ),
        ("python scripts/check_platform.py", "python scripts/other.py", "does not run the script"),
        (
            PLATFORM_STEP.removeprefix("        "),
            PLATFORM_STEP.removeprefix("        ") + " || true",
            "does not run the script",
        ),
        (
            PLATFORM_STEP.removeprefix("        "),
            PLATFORM_STEP.removeprefix("        ") + "/v1",
            "does not run the script",
        ),
        (PLATFORM_STEP, "        # " + PLATFORM_STEP.strip(), "does not run the script"),
        (
            "      - name: Check the platform's contract version\n",
            "      - name: Check the platform's contract version\n        if: false\n",
            "platform can be skipped",
        ),
        (
            "      - name: Check the platform's contract version\n",
            "      - name: Check the platform's contract version\n        continue-on-error: true\n",
            "platform can be skipped",
        ),
        (
            "uses: ./.github/workflows/dag-lib-ci.yml",
            "uses: ./.github/workflows/x.yml",
            "does not call",
        ),
    ],
)
def test_a_workflow_that_lets_a_release_around_the_gate_is_caught(
    old: str, new: str, problem: str
) -> None:
    workflow = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
    assert old in workflow, old
    assert any(problem in found for found in _release_problems(workflow.replace(old, new, 1)))


def test_every_third_party_action_is_pinned_and_every_job_has_a_timeout() -> None:
    """A tag publishes to PyPI: an action moved under it or a job that hangs both cost a release."""
    for file in ("release.yml", "dag-lib-ci.yml"):
        for name, job in _jobs((WORKFLOWS / file).read_text(encoding="utf-8")).items():
            if re.search(r"^    uses: ", job, re.M):
                continue  # a call of a reusable workflow: its own jobs carry the timeouts
            assert re.search(r"^    timeout-minutes: \d+$", job, re.M), f"{file}:{name} has none"
            for action in re.findall(r"uses: (\S+)", job):
                local = action.startswith("./")
                assert local or re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", action), action


def test_the_workflow_the_release_calls_is_one_it_may_call() -> None:
    """``ci`` runs the CI file on the tagged commit, which that file must say it allows."""
    called = (WORKFLOWS / "dag-lib-ci.yml").read_text(encoding="utf-8")
    assert re.search(r"^  workflow_call:", called, re.M)
