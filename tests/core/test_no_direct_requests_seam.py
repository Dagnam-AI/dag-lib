"""Guard against reintroducing a direct ``requests.<verb>`` call in a migrated mixin.

Every non-streaming, non-multipart client call must go through
``BaseDagnamClient._request`` (so it inherits transport mapping + transient
retry). The only functions still allowed to call ``requests.<verb>`` directly are
the ones ``_request`` cannot express: SSE/chunked streaming reads, multipart
uploads, and the unauthenticated bootstrap POSTs (``_request`` always attaches
the Authorization header). The allowlist keys on ``(filename, function)`` via
``ast`` so it is exact regardless of how the streaming/multipart kwarg is spelled.
"""

from __future__ import annotations

import ast
from pathlib import Path

_CLIENT_DIR = Path(__file__).resolve().parents[2] / "dagnam" / "_core" / "client"
_VERBS = {"get", "post", "put", "delete", "patch", "request"}

# (filename, enclosing function) sites that legitimately call ``requests.<verb>``
# directly because ``_request()`` cannot express what they need.
_ALLOWED_SITES = {
    # Streaming reads — a streamed body cannot be replayed on retry.
    ("base.py", "_get_stream"),  # streaming GET, (connect, read) timeout tuple
    ("base.py", "_get_stream_no_auth"),  # streaming GET, no auth header
    ("datasets.py", "download_dataset"),  # streaming download with Range/resume
    ("codegen.py", "download_code"),  # conditional streaming: stream=bool(dest_path)
    ("deployments.py", "open_deployment_stream"),  # stream=True SSE read
    ("inference.py", "open_inference_stream"),  # stream=True SSE read
    ("training.py", "open_training_stream"),  # stream=True SSE read
    # Multipart uploads — _request has no files= parameter.
    ("datasets.py", "upload_dataset"),  # multipart, timeout=None
    ("account.py", "upload_profile_photo"),  # multipart files={"file": ...}
    ("projects.py", "upload_project_thumbnail"),  # multipart files={"file": ...}
    ("hub.py", "upload_model_file"),  # multipart files={"file": ...}
    ("models.py", "upload_model_artifact_direct"),  # multipart files={"file": ...}
    ("training.py", "upload_run_artifact"),  # multipart files={"file": ...}
    # Unauthenticated bootstrap POSTs — _request always attaches the bearer
    # auth header these two must NOT send.
    ("account.py", "register"),
    ("account.py", "login_for_bootstrap"),
}


def _direct_call_functions(path: Path) -> set[str]:
    """Names of functions in ``path`` whose body calls ``requests.<verb>(...)`` directly."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "requests"
                and call.func.attr in _VERBS
            ):
                hits.add(node.name)
    return hits


def test_no_unmigrated_requests_calls() -> None:
    offenders: list[str] = []
    for path in sorted(_CLIENT_DIR.glob("*.py")):
        for func_name in _direct_call_functions(path):
            if (path.name, func_name) not in _ALLOWED_SITES:
                offenders.append(f"{path.name}:{func_name}")
    assert not offenders, f"Unmigrated direct requests calls: {offenders}"
