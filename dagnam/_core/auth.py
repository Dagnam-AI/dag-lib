"""Authentication resolution for the dagnam library.

Resolves API key and API URL using a three-tier priority chain:
  override parameter → module-level inline state → environment variable → config file → default/error
"""

import logging
import os
from typing import Optional
from urllib.parse import urlparse

from dagnam._core.config import get_config_value
from dagnam._core.exceptions import AuthError

_logger = logging.getLogger(__name__)

_DEFAULT_API_URL = "https://api.dagnam.ai"

# Module-level inline state, set via configure()
_api_key: Optional[str] = None
_api_url: Optional[str] = None

# Warn-once-per-process guard for the api_url trust check. Process-global by
# design: the check is warn-only and must not spam every request.
_api_url_warned = False
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _warn_if_untrusted_api_url(url: str) -> None:
    """Warn once when the credential target is non-default or cleartext http.

    A poisoned config file or ``DAGNAM_API_URL`` can redirect the
    ``Authorization: Bearer <key>`` header to an attacker-controlled host, and
    a non-local ``http://`` target sends the key in cleartext. Only the
    tamperable sources (env/config) are checked; an explicit ``override=`` or
    an inline ``configure(api_url=…)`` is a deliberate caller choice and stays
    silent. Warn-only — the request is never blocked.
    """
    global _api_url_warned
    if _api_url_warned:
        return
    parsed = urlparse(url)
    host = parsed.hostname or ""
    non_default = host != urlparse(_DEFAULT_API_URL).hostname
    cleartext = parsed.scheme == "http" and host not in _LOCAL_HOSTS
    if non_default or cleartext:
        _api_url_warned = True
        _logger.warning(
            "Dagnam credentials will be sent to %s%s. If you did not configure this, "
            "your ~/.dagnam/config.json or DAGNAM_API_URL may have been tampered with.",
            url,
            " over cleartext http" if cleartext else "",
        )


def configure(api_key: Optional[str] = None, api_url: Optional[str] = None) -> None:
    """Store inline credentials in module-level state."""
    global _api_key, _api_url
    _api_key = api_key
    _api_url = api_url


def get_api_key(override: Optional[str] = None) -> str:
    """Resolve API key: override → inline → DAGNAM_API_KEY env var → config file.

    Raises AuthError if no key found in object source.
    """
    if override is not None:
        return override

    if _api_key is not None:
        return _api_key

    env_key = os.environ.get("DAGNAM_API_KEY")
    if env_key is not None:
        return env_key

    config_key = get_config_value("api_key")
    if isinstance(config_key, str):
        return config_key

    raise AuthError(
        "No API key found. Set DAGNAM_API_KEY environment variable, "
        "run 'dagnam login', or call dagnam.configure(api_key='...')"
    )


def get_api_url(override: Optional[str] = None) -> str:
    """Resolve API URL: override → inline → DAGNAM_API_URL env var → config file → default.

    Warns (once per process) when the resolved URL comes from the tamperable
    sources (``DAGNAM_API_URL`` or the config file) and targets a non-default
    or cleartext host; ``override`` and inline ``configure()`` values are
    deliberate caller choices and stay silent.
    """
    if override is not None:
        return override

    if _api_url is not None:
        return _api_url

    env_url = os.environ.get("DAGNAM_API_URL")
    if env_url is not None:
        _warn_if_untrusted_api_url(env_url)
        return env_url

    config_url = get_config_value("api_url")
    if isinstance(config_url, str):
        _warn_if_untrusted_api_url(config_url)
        return config_url

    return _DEFAULT_API_URL
