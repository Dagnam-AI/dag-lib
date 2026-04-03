"""Custom exception classes for the dagnam library."""


class DagnamError(Exception):
    """Base exception for all dagnam errors."""


class AuthError(DagnamError):
    """No API key found or authentication failed (401)."""


class DatasetNotFoundError(DagnamError):
    """Dataset ID not found (404)."""

    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        super().__init__(f"Dataset '{dataset_id}' not found")


class APIError(DagnamError):
    """General API communication failure."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API error {status_code}: {message}")


class ChecksumError(DagnamError):
    """Downloaded file checksum does not match server-reported checksum."""
