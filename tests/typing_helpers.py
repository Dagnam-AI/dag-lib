"""Shared typing helpers for strict test modules."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Protocol

import httpx
import pytest

from dagnam.data.dataset import DagnamDataset

type JsonScalar = str | int | float | bool | None
type JsonObject = dict[str, JsonValue]
type JsonArray = list[JsonValue]
type JsonValue = JsonScalar | JsonObject | JsonArray

type CliRunner = Callable[[list[str]], int]
type PytestMonkeyPatch = pytest.MonkeyPatch
type StrCapture = pytest.CaptureFixture[str]
type LogCapture = pytest.LogCaptureFixture
type JsonIterable = Iterable[JsonValue]
type ObjectIterator = Iterator[object]
type ObjectTransform = Callable[[object], object]
type StreamOpener = Callable[[str | None], object]


class RequestsRecord(Protocol):
    headers: Mapping[str, str]
    method: str
    path: str
    qs: dict[str, list[str]]
    text: str | None

    def json(self) -> JsonObject: ...


class RequestsMocker(Protocol):
    last_request: RequestsRecord

    def get(self, url: str, **kwargs: object) -> object: ...
    def post(self, url: str, **kwargs: object) -> object: ...
    def put(self, url: str, **kwargs: object) -> object: ...
    def patch(self, url: str, **kwargs: object) -> object: ...
    def delete(self, url: str, **kwargs: object) -> object: ...


class RespxCall(Protocol):
    request: httpx.Request


class RespxRoute(Protocol):
    calls: Sequence[RespxCall]
    called: bool

    def mock(self, **kwargs: object) -> RespxRoute: ...


class RespxMockRouter(Protocol):
    def get(self, url: str) -> RespxRoute: ...
    def post(self, url: str) -> RespxRoute: ...
    def put(self, url: str) -> RespxRoute: ...
    def patch(self, url: str) -> RespxRoute: ...
    def delete(self, url: str) -> RespxRoute: ...


class DatasetFactory(Protocol):
    def __call__(self, **kwargs: object) -> DagnamDataset: ...
