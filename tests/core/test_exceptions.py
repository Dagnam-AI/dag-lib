"""Unit tests for dagnam exception hierarchy."""

import pytest

from dagnam._contracts import ParamError
from dagnam._core.client.common import raise_for_generic, safe_response_text
from dagnam._core.exceptions import (
    APIError,
    ArchitectureValidationError,
    AuthError,
    ChecksumError,
    DagnamError,
    DatasetNotFoundError,
)


def _perr(message: str, node_id: str = "c1") -> ParamError:
    return ParamError(type="parameter_error", message=message, node_id=node_id, severity="error")


class TestArchitectureValidationError:
    def test_carries_param_errors_and_summarizes(self) -> None:
        e = ArchitectureValidationError([_perr("conv: padding bad")])
        assert isinstance(e, DagnamError)
        assert e.errors[0].node_id == "c1"
        assert "padding" in str(e)

    def test_summary_truncates_after_three_errors(self) -> None:
        e = ArchitectureValidationError([_perr(f"err {i}") for i in range(5)])
        assert "(+2 more)" in str(e)


class TestExceptionHierarchy:
    """All custom exceptions inherit from DagnamError."""

    def test_autherror_is_dagnamerror(self) -> None:
        assert issubclass(AuthError, DagnamError)

    def test_dataset_not_founderror_is_dagnamerror(self) -> None:
        assert issubclass(DatasetNotFoundError, DagnamError)

    def test_apierror_is_dagnamerror(self) -> None:
        assert issubclass(APIError, DagnamError)

    def test_checksumerror_is_dagnamerror(self) -> None:
        assert issubclass(ChecksumError, DagnamError)

    def test_dagnamerror_is_exception(self) -> None:
        assert issubclass(DagnamError, Exception)

    def test_catch_all_with_dagnamerror(self) -> None:
        """All library exceptions can be caught with a single except DagnamError."""
        errors: list[DagnamError] = [
            AuthError("some error"),
            DatasetNotFoundError("ds-123"),
            APIError(500, "server error"),
            ChecksumError("some error"),
        ]
        for err in errors:
            with pytest.raises(DagnamError):
                raise err


class TestDatasetNotFoundError:
    def test_stores_dataset_id(self) -> None:
        err = DatasetNotFoundError("abc-123")
        assert err.dataset_id == "abc-123"

    def test_message_contains_dataset_id(self) -> None:
        err = DatasetNotFoundError("my-dataset")
        assert "my-dataset" in str(err)
        assert str(err) == "Dataset 'my-dataset' not found"


class TestAPIError:
    def test_stores_status_code_and_message(self) -> None:
        err = APIError(503, "Service Unavailable")
        assert err.status_code == 503
        assert err.message == "Service Unavailable"

    def test_message_format(self) -> None:
        err = APIError(500, "Internal Server Error")
        assert str(err) == "API error 500: Internal Server Error"

    def test_response_body_is_truncated_in_generic_mapper(self) -> None:
        class Response:
            def __init__(self) -> None:
                self.ok = False
                self.status_code = 500
                self.text = "x" * 5000
                self.headers = {"Content-Type": "text/plain"}
                self.content = self.text.encode()

        with pytest.raises(APIError) as exc_info:
            raise_for_generic(Response())

        assert len(exc_info.value.message) < 2200
        assert "truncated" in exc_info.value.message

    def test_binary_response_body_is_not_dumped(self) -> None:
        class Response:
            def __init__(self) -> None:
                self.ok = False
                self.status_code = 500
                self.text = "\x00" * 5000
                self.headers = {"Content-Type": "application/octet-stream"}
                self.content = b"\x00" * 5000

        with pytest.raises(APIError) as exc_info:
            raise_for_generic(Response())

        assert exc_info.value.message == "<5000 bytes of application/octet-stream>"

    def test_streaming_error_body_is_not_materialized(self) -> None:
        class Response:
            _content = False

            def __init__(self) -> None:
                self.headers = {"Content-Type": "application/octet-stream"}

            @property
            def content(self):
                raise AssertionError("streaming body should not be read")

            @property
            def text(self):
                raise AssertionError("streaming text should not be read")

        # Intentionally partial streaming response fake (no status_code).
        assert (
            safe_response_text(Response())  # pyright: ignore[reportArgumentType]
            == "<streaming application/octet-stream body omitted>"
        )


class TestChecksumError:
    def test_accepts_message(self) -> None:
        err = ChecksumError("checksum mismatch")
        assert "checksum mismatch" in str(err)


class TestAuthError:
    def test_accepts_message(self) -> None:
        err = AuthError("No API key found")
        assert "No API key found" in str(err)
