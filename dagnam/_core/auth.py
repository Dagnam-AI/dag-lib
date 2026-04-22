"""Authentication resolution for the dagnam library.

Resolves API key and API URL using a three-tier priority chain:
  override parameter → module-level inline state → environment variable → config file → default/error
"""

import os
from typing import Optional

from dagnam._core.config import get_config_value
from dagnam._core.exceptions import AuthError

_DEFAULT_API_URL = "https://api.dagnam.ai"

# Module-level inline state, set via configure()
_api_key: Optional[str] = None
_api_url: Optional[str] = None


def configure(api_key: Optional[str] = None, api_url: Optional[str] = None) -> None:
    """Store inline credentials in module-level state."""
    global _api_key, _api_url
    _api_key = api_key
    _api_url = api_url


def get_api_key(override: Optional[str] = None) -> str:
    """Resolve API key: override → inline → DAGNAM_API_KEY env var → config file.

    Raises AuthError if no key found in any source.
    """
    if override is not None:
        return override

    if _api_key is not None:
        return _api_key

    env_key = os.environ.get("DAGNAM_API_KEY")
    if env_key is not None:
        return env_key

    config_key = get_config_value("api_key")
    if config_key is not None:
        return config_key

    raise AuthError(
        "No API key found. Set DAGNAM_API_KEY environment variable, "
        "run 'dagnam login', or call dagnam.configure(api_key='...')"
    )


def get_api_url(override: Optional[str] = None) -> str:
    """Resolve API URL: override → inline → DAGNAM_API_URL env var → config file → default."""
    if override is not None:
        return override

    if _api_url is not None:
        return _api_url

    env_url = os.environ.get("DAGNAM_API_URL")
    if env_url is not None:
        return env_url

    config_url = get_config_value("api_url")
    if config_url is not None:
        return config_url

    return _DEFAULT_API_URL
