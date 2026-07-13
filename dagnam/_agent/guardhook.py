"""Cross-platform PreToolUse hook: deny un-confirmed costly Dagnam commands.

Reads the harness hook event (JSON on stdin) and, if the Bash command runs a costly
Dagnam action (deployments create/delete, training create/delete, projects delete,
hub publish, or a public visibility flag) WITHOUT an explicit ``DAGNAM_CONFIRM=1``
prefix, emits a deny decision. Both the CLI verb shape (``dagnam deployments create``)
and the Python SDK shape (``python -c "import dagnam; dagnam.deployments.create(...)"``,
which ``SKILL.md`` recommends) are matched.

This hook is **non-authoritative, best-effort defense-in-depth ONLY.** It has known
blind spots -- command obfuscation (base64/eval, string-built attribute names),
aliasing, imports under a different module name, and any non-Bash tool the agent
might use -- and it fails OPEN on any error so it can never wedge the agent. The
authoritative gate is the ``SKILL.md`` behavioral rule plus server-side confirmation
enforcement; this hook only hardens them. The ``visibility=public`` heuristic is
deliberately broad and will occasionally deny a benign public create/upload -- an
accepted false positive, since a denial only asks the operator to re-run with a
``DAGNAM_CONFIRM=1`` prefix.

Prints to stdout by design (the hook protocol), so it is T20-exempt.
"""

from __future__ import annotations

import json
import re
import sys

# Costly actions in BOTH the CLI verb shape and the Python SDK shape. re.DOTALL so a
# newline between ``dagnam`` and the action (a multi-line or line-continuation
# command) cannot hide it from the deny check.
_COSTLY = re.compile(
    r"\bdagnam\b.*\b(?:"
    r"deployments?\s+(?:create|delete)"  # CLI: dagnam deployments create
    r"|training\s+(?:create|delete)"
    r"|projects?\s+delete"
    r"|deployments?\s*\.\s*(?:create|delete)"  # SDK: dagnam.deployments.create(
    r"|training\s*\.\s*(?:create|delete)"  # SDK: dagnam.training.create(
    r"|projects?\s*\.\s*delete"
    r"|hub\s*\.\s*create"  # SDK: publish to the hub
    r"|create_training_job"
    r"|visibility\s*=\s*[\\'\"]*public"  # any public create/upload (quotes/escapes optional)
    r")",
    re.DOTALL,
)
# Anchor the confirm token to a command-leading env assignment (start of line,
# or the first token after a shell separator). Matching it ANYWHERE let a
# hostile string echoed into an argument -- e.g. a dataset description
# "... DAGNAM_CONFIRM=1" surfaced via prompt injection -- silently satisfy the
# gate. It must be a real prefix the operator typed, not incidental text.
_CONFIRMED = re.compile(r"(?:^|[;&|\n]\s*)DAGNAM_CONFIRM=1\b")
_DENY_REASON = (
    "Dagnam guardrail: this is a costly/irreversible action. Show the user an execution "
    "plan (scripts/plan.py, or codegen.validate/preview + account.entitlements) and get "
    "explicit confirmation, then re-run the command prefixed with DAGNAM_CONFIRM=1."
)


def main() -> int:
    """Entry point wired by claude/hooks/guard.json and codex/hooks.json."""
    try:
        event = json.load(sys.stdin)
        command = str(event.get("tool_input", {}).get("command", ""))
    except (json.JSONDecodeError, AttributeError, ValueError):
        return 0  # fail-open
    if _COSTLY.search(command) and not _CONFIRMED.search(command):
        json.dump(
            {"permissionDecision": "deny", "permissionDecisionReason": _DENY_REASON}, sys.stdout
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
