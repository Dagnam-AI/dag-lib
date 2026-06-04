"""Shared fixtures for the synchronous client wire-level tests."""

from __future__ import annotations

import pytest
import requests_mock as rm_module

from dagnam._core.client import DagnamClient

API = "https://api.test"


@pytest.fixture
def client() -> DagnamClient:
    return DagnamClient(API, "k")


@pytest.fixture
def rmock():
    with rm_module.Mocker() as m:
        yield m
