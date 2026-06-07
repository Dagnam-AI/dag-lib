"""Dry-run preview for a costly Dagnam action.

Usage: ``python plan.py --action deploy --param key=value ...``

A thin shim over :func:`dagnam._agent.runner.plan_main`; all logic and tests live
in ``runner.py`` so this stays trivial.
"""

from __future__ import annotations

from dagnam._agent.runner import plan_main

if __name__ == "__main__":
    raise SystemExit(plan_main())
