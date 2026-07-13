"""Logging contract for the dagnam library.

A PyPI library must never configure the root logger or attach output handlers by
default — that is the consuming application's decision. We attach only a
``NullHandler`` to the ``dagnam`` logger and expose ``enable_debug_logging`` as a
convenience for users who don't know the stdlib ``logging`` API. A
``RedactingFilter`` scrubs credentials so a ``DEBUG``-level URL never leaks a
token.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import IO, override

__all__ = [
    "CACHE_LOGGER",
    "HTTP_LOGGER",
    "LIBRARY_LOGGER_NAME",
    "LRO_LOGGER",
    "SSE_LOGGER",
    "RedactingFilter",
    "enable_debug_logging",
]

LIBRARY_LOGGER_NAME = "dagnam"

HTTP_LOGGER = "dagnam.http"
CACHE_LOGGER = "dagnam.cache"
LRO_LOGGER = "dagnam.lro"
SSE_LOGGER = "dagnam.sse"

_CHILD_LOGGER_NAMES = (HTTP_LOGGER, CACHE_LOGGER, LRO_LOGGER, SSE_LOGGER)

# Credential-bearing query params. Kept deliberately BROAD and in sync with
# dagnam._core.client.base.scrub_secret_params (`token|signature|credential|sig|key`):
# a presigned S3/GCS/CDN URL carries its secret as X-Amz-Signature / X-Amz-Credential /
# Signature / sig / AWSAccessKeyId (matched via the `key` alternative), NOT as `token`.
# A narrower set would leave those unredacted in a logged retry URL. The `[\w.-]*`
# wings match hyphenated prefixes like `x-amz-`.
_QUERY_CRED_RE = re.compile(
    r"([?&][\w.-]*(?:token|signature|credential|sig|key)[\w.-]*=)[^&\s]+", re.IGNORECASE
)
# NOTE: the credential value is the REST of the line, not just the next
# whitespace-delimited token — "Authorization: Bearer sk_live_ABC" has a
# two-token value, and a `\S+` capture would leave "sk_live_ABC" unredacted.
# `.+` consumes everything after the separator instead.
_HEADER_CRED_RE = re.compile(
    r"(authorization|x-api-key|idempotency-key)(['\"]?\s*[:=]\s*).+",
    re.IGNORECASE,
)


class RedactingFilter(logging.Filter):
    """Scrub credential query params and auth-header text from log messages."""

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        scrubbed = _QUERY_CRED_RE.sub(r"\1<redacted>", message)
        scrubbed = _HEADER_CRED_RE.sub(r"\1\2<redacted>", scrubbed)
        record.msg = scrubbed
        record.args = ()
        return True


# Attach one filter instance to EACH child logger at import time — not only to
# the enable_debug_logging() convenience handler. A Filter on a Logger only
# fires for records originating at that logger, so this is what makes redaction
# work regardless of where a consumer attaches their own handler.
for _child_name in _CHILD_LOGGER_NAMES:
    logging.getLogger(_child_name).addFilter(RedactingFilter())


def enable_debug_logging(level: int = logging.DEBUG, stream: IO[str] | None = None) -> None:
    """Attach a redacting ``StreamHandler`` to the ``dagnam`` logger at ``level``.

    Convenience for interactive debugging; production apps configure their own
    handlers (and should add :class:`RedactingFilter`).
    """
    logger = logging.getLogger(LIBRARY_LOGGER_NAME)
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    handler.addFilter(RedactingFilter())
    logger.addHandler(handler)
    logger.setLevel(level)
