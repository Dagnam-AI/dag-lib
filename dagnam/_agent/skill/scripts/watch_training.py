"""Stream a Dagnam training job to completion.

Usage: ``python watch_training.py <job_id>``

A thin shim over :func:`dagnam._agent.runner.watch_main`; all logic and tests live
in ``runner.py`` so this stays trivial.
"""

from __future__ import annotations

from dagnam._agent.runner import watch_main

if __name__ == "__main__":
    raise SystemExit(watch_main())
