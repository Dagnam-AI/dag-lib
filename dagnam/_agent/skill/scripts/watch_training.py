"""Stream a Dagnam training job to completion.

Usage: ``python watch_training.py <job_id>``

Exit code: 0 complete (or the stream ended with the job still running), 1 failed or
cancelled, 3 paused because the account could not fund the next stretch (add credits, then
``dagnam training resume <job_id>``).

A thin shim over :func:`dagnam._agent.runner.watch_main`; all logic and tests live
in ``runner.py`` so this stays trivial.
"""

from __future__ import annotations

from dagnam._agent.runner import watch_main

if __name__ == "__main__":
    raise SystemExit(watch_main())
