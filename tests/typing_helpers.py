"""Shared typing helpers for strict test modules."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Protocol, TypeAlias

import httpx
import pytest

from dagnam.data.dataset import DagnamDataset

JsonScalar: TypeAlias = str | int | float | bool | None
JsonObject: TypeAlias = dict[str, "JsonValue"]
JsonArray: TypeAlias = list["JsonValue"]
JsonValue: TypeAlias = JsonScalar | JsonObject | JsonArray

CliRunner: TypeAlias = Callable[[list[str]], None]
PytestMonkeyPatch: TypeAlias = pytest.MonkeyPatch
StrCapture: TypeAlias = pytest.CaptureFixture[str]
LogCapture: TypeAlias = pytest.LogCaptureFixture
JsonIterable: TypeAlias = Iterable[JsonValue]
ObjectIterator: TypeAlias = Iterator[object]
ObjectTransform: TypeAlias = Callable[[object], object]
StreamOpener: TypeAlias = Callable[[str | None], object]


class RequestsRecord(Protocol):
    headers: Mapping[str, str]
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

    def mock(self, **kwargs: object) -> RespxRoute: ...


class RespxMockRouter(Protocol):
    def get(self, url: str) -> RespxRoute: ...
    def post(self, url: str) -> RespxRoute: ...
    def put(self, url: str) -> RespxRoute: ...
    def patch(self, url: str) -> RespxRoute: ...
    def delete(self, url: str) -> RespxRoute: ...


class DatasetFactory(Protocol):
    def __call__(self, **kwargs: object) -> DagnamDataset: ...
