"""One audit directory driven through ``run``, ``cancel`` and ``delete`` against the teardown world.

``cancel`` and ``delete`` are the SDK's own (:func:`dagnam.audit.cleanup.cancel_audit` and
:func:`~dagnam.audit.cleanup.delete_audit`, over a real ``state.json``); the run is
:class:`tests.audit._teardown_dir.Directory`'s.
"""

from __future__ import annotations

from typing import override

from tests.audit._teardown_client import as_cleanup_client
from tests.audit._teardown_dir import Directory

from dagnam.audit.claims import ClaimError, claim_recorded
from dagnam.audit.cleanup import AuditDeletedError, cancel_audit, delete_audit, receipt_rows
from dagnam.audit.receipt_rows import exit_status
from dagnam.audit.state import AuditState, save_state


class Sdk(Directory):
    """A directory with the SDK's teardown commands."""

    @override
    def _claim(self, state: AuditState) -> bool:
        try:
            claim_recorded(self.client, state)
        except ClaimError:
            return False
        finally:
            save_state(self.audit_dir, state)
        return True

    def delete(self, failure: str | None = None) -> int:
        """``audit delete``: the exit status comes from the receipt alone."""
        self.client.audit_failure = failure
        receipt = delete_audit(self.audit_dir, as_cleanup_client(self.client))
        self.answered = not any(r.get("code") == "not_answered" for r in receipt_rows(receipt))
        return self._exit(
            "delete", exit_status(receipt_rows(receipt), receipt.get("audit_status"), verb="delete")
        )

    def cancel(self, failure: str | None = None) -> int:
        """``audit cancel``."""
        self.client.audit_failure = failure
        try:
            receipt = cancel_audit(self.audit_dir, as_cleanup_client(self.client))
        except AuditDeletedError:  # the command refuses a deleted audit: exit 1, nothing asked
            return self._exit("cancel", 1)
        self.answered = not any(r.get("code") == "not_answered" for r in receipt_rows(receipt))
        return self._exit(
            "cancel", exit_status(receipt_rows(receipt), receipt.get("audit_status"), verb="cancel")
        )
