"""dagnam-record: a local, stdlib-only recorder of a training run's workload.

It has no dependencies and opens no network connection. Nothing leaves the machine
until the owner uploads the file it writes.

    import dagnam_record
    dagnam_record.install()  # records openai and anthropic calls under ~/.dagnam/traces

``install`` is documented in ``dagnam_record._state``; ``python -m dagnam_record status|purge``
shows and deletes what was recorded.
"""

from dagnam_record._state import install, record_call, uninstall

__version__ = "0.1.0"

__all__ = ["__version__", "install", "record_call", "uninstall"]
