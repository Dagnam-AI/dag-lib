"""Unit tests for the dagnam library logging contract (dagnam._core._logging)."""

from __future__ import annotations

import io
import logging

import dagnam
from dagnam._core._logging import RedactingFilter, enable_debug_logging


def test_null_handler_attached_by_default():
    handlers = logging.getLogger("dagnam").handlers
    assert any(isinstance(h, logging.NullHandler) for h in handlers)


def test_redacting_filter_scrubs_query_credentials():
    rec = logging.LogRecord(
        "dagnam.http",
        logging.DEBUG,
        __file__,
        0,
        "GET https://api/x?token=SECRET&a=1",
        (),
        None,
    )
    assert RedactingFilter().filter(rec) is True
    assert "SECRET" not in rec.getMessage()
    assert "<redacted>" in rec.getMessage()


def test_redacting_filter_scrubs_auth_header_text():
    rec = logging.LogRecord(
        "dagnam.http",
        logging.DEBUG,
        __file__,
        0,
        "Authorization: Bearer sk_live_ABC",
        (),
        None,
    )
    RedactingFilter().filter(rec)
    assert "sk_live_ABC" not in rec.getMessage()


def test_child_logger_redacts_even_with_unfiltered_consumer_handler():
    """The advertised idiom: a consumer attaches their OWN plain handler (no
    filter) to the parent 'dagnam' logger. Redaction must still happen because
    RedactingFilter lives on the 'dagnam.http' CHILD logger itself and mutates
    the record before it ever reaches the consumer's handler."""
    stream = io.StringIO()
    parent_logger = logging.getLogger("dagnam")
    http_logger = logging.getLogger("dagnam.http")
    plain_handler = logging.StreamHandler(stream)
    plain_handler.setFormatter(logging.Formatter("%(message)s"))
    parent_logger.addHandler(plain_handler)
    previous_level = http_logger.level
    http_logger.setLevel(logging.DEBUG)
    try:
        http_logger.debug("Authorization: Bearer sk_live_XYZ")
    finally:
        parent_logger.removeHandler(plain_handler)
        http_logger.setLevel(previous_level)
    output = stream.getvalue()
    assert "sk_live_XYZ" not in output
    assert "<redacted>" in output


def test_enable_debug_logging_adds_filtered_handler(tmp_path):
    stream = io.StringIO()
    logger = logging.getLogger("dagnam")
    before = list(logger.handlers)
    try:
        enable_debug_logging(level=logging.DEBUG, stream=stream)
        logging.getLogger("dagnam.http").debug("GET https://api/x?api_key=SECRET")
        assert "SECRET" not in stream.getvalue()
        assert "<redacted>" in stream.getvalue()
    finally:
        for h in list(logger.handlers):
            if h not in before:
                logger.removeHandler(h)
        logger.setLevel(logging.NOTSET)


def test_enable_debug_logging_defaults_to_stderr():
    logger = logging.getLogger("dagnam")
    before = list(logger.handlers)
    try:
        enable_debug_logging()
        added = [h for h in logger.handlers if h not in before]
        assert added
        assert isinstance(added[0], logging.StreamHandler)
    finally:
        for h in list(logger.handlers):
            if h not in before:
                logger.removeHandler(h)
        logger.setLevel(logging.NOTSET)


def test_enable_debug_logging_exported():
    assert dagnam.enable_debug_logging is enable_debug_logging


def test_redacting_filter_scrubs_presigned_url_signature():
    # A presigned S3/GCS/CDN URL carries its credential as X-Amz-Signature /
    # X-Amz-Credential / Signature / sig / AWSAccessKeyId — NOT `token`. The
    # retry logger logs such URLs, so all of these must be redacted.
    msg = (
        "retrying GET https://b.s3.amazonaws.com/o?"
        "X-Amz-Credential=AKIAEXAMPLE&X-Amz-Signature=SECRETSIG&"
        "Signature=DEADBEEF&sig=abc&AWSAccessKeyId=AKIA2&a=1 after status=503"
    )
    rec = logging.LogRecord("dagnam.http", logging.DEBUG, __file__, 0, msg, (), None)
    RedactingFilter().filter(rec)
    out = rec.getMessage()
    for secret in ("SECRETSIG", "AKIAEXAMPLE", "DEADBEEF", "AKIA2"):
        assert secret not in out, secret
    assert "a=1" in out  # non-credential params preserved
