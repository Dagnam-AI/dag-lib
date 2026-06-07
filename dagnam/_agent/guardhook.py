"""Cross-platform PreToolUse hook: deny un-confirmed costly Dagnam commands.

Reads the harness hook event (JSON on stdin) and, if the Bash command runs a costly
Dagnam verb (deployments create/delete, training create/delete, projects delete)
WITHOUT an explicit ``DAGNAM_CONFIRM=1`` prefix, emits a deny decision. Fail-open on
any error -- the ``SKILL.md`` behavioral gate is the real guarantee; this hook only
hardens it and must never wedge the agent.

Prints to stdout by design (the hook protocol), so it is T20-exempt.
"""

from __future__ import annotations

import json
import re
import sys

# Real, costly CLI verbs only (no `deployments scale` / `hub publish` -- those are
# SDK-only and not reachable as a Bash `dagnam ...` command).
_COSTLY = re.compile(
    r"\bdagnam\b.*\b(?:"
    r"deployments?\s+(?:create|delete)"
    r"|training\s+(?:create|delete)"
    r"|projects?\s+delete"
    r")\b"
)
_CONFIRMED = re.compile(r"\bDAGNAM_CONFIRM=1\b")
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
