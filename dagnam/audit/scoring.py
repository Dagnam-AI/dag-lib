"""Client-side agreement between a candidate and the teacher on the holdout (spec D3).

The scorers themselves live in ``dagnam_contracts.audit.scoring``: the SDK
scores a holdout on the customer's machine and the platform scores one on a
worker, and both must reach the same number for the same rows. This module is
the SDK's name for them -- labels and short spans score as normalized exact
match (with macro-F1 alongside), JSON objects score per field, micro-averaged,
and every score carries ``n`` and a 95% Wilson interval on the headline value
that the frontier decides on the lower bound of.
"""

from __future__ import annotations

from dagnam_contracts.audit.scoring import (
    Z95,
    Agreement,
    modal_keys,
    score_json,
    score_labels,
    wilson_interval,
)

__all__ = ["Z95", "Agreement", "modal_keys", "score_json", "score_labels", "wilson_interval"]
