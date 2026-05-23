"""Shared structural types used by the SDK."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import BinaryIO, Protocol, TypeAlias, TypeGuard, cast, runtime_checkable

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
JsonArray: TypeAlias = list[JsonValue]
JsonMapping: TypeAlias = Mapping[str, JsonValue]

QueryScalar: TypeAlias = str | int | float | bool | None
QueryValue: TypeAlias = QueryScalar | Sequence[QueryScalar]
QueryParams: TypeAlias = Mapping[str, QueryValue]
FormData: TypeAlias = Mapping[str, str]
UploadFile: TypeAlias = tuple[str, BinaryIO] | tuple[str, BinaryIO, str]
UploadFiles: TypeAlias = Mapping[str, UploadFile]


class ResponseLike(Protocol):
    """Small response surface shared by requests and httpx responses."""

    @property
    def status_code(self) -> object: ...

    @property
    def headers(self) -> Mapping[str, str]: ...

    @property
    def text(self) -> object: ...

    @property
    def content(self) -> bytes: ...


class JsonResponseLike(ResponseLike, Protocol):
    """Response object that can decode a JSON body."""

    def json(self) -> object: ...


class IndexedDataset(Protocol):
    """Dataset object that supports length and integer indexing."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> object: ...


@runtime_checkable
class SupportsNumpy(Protocol):
    """Object exposing a NumPy conversion method."""

    def numpy(self) -> object: ...


class TensorflowDataset(Protocol):
    """Small tf.data.Dataset surface used by framework adapters."""

    def shuffle(self, buffer_size: int, seed: int | None = None) -> TensorflowDataset: ...

    def map(
        self,
        map_func: Callable[..., object],
        num_parallel_calls: object = None,
    ) -> TensorflowDataset: ...

    def batch(self, batch_size: int) -> TensorflowDataset: ...

    def prefetch(self, buffer_size: object) -> TensorflowDataset: ...

    def take(self, count: int) -> TensorflowDataset: ...

    def skip(self, count: int) -> TensorflowDataset: ...

    def enumerate(self) -> TensorflowDataset: ...

    def filter(self, predicate: Callable[..., object]) -> TensorflowDataset: ...

    def __iter__(self) -> Iterator[object]: ...


class TensorflowDatasetFactory(Protocol):
    """Factory namespace for tf.data.Dataset constructors."""

    def from_tensor_slices(self, tensors: object) -> TensorflowDataset: ...


class TensorflowExperimentalData(Protocol):
    """Small tf.data.experimental surface used by the SDK."""

    UNKNOWN_CARDINALITY: int

    def cardinality(self, dataset: TensorflowDataset) -> SupportsNumpy: ...


class TensorflowDataNamespace(Protocol):
    """Small tf.data namespace used by the SDK."""

    AUTOTUNE: object
    Dataset: TensorflowDatasetFactory
    experimental: TensorflowExperimentalData


class TensorflowModule(Protocol):
    """Small TensorFlow module surface used by framework adapters."""

    data: TensorflowDataNamespace


NativeSplit: TypeAlias = IndexedDataset | tuple[Sequence[object], Sequence[object]]


def _type_name(value: object) -> str:
    return value.__class__.__name__


def is_json_value(value: object) -> TypeGuard[JsonValue]:
    """Return whether *value* is JSON-compatible."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        items = cast(list[object], value)
        return all(is_json_value(item) for item in items)
    if isinstance(value, dict):
        items = cast(dict[object, object], value)
        return all(isinstance(key, str) and is_json_value(item) for key, item in items.items())
    return False


def ensure_json_value(value: object) -> JsonValue:
    """Return a JSON value or raise when a response body is not JSON-compatible."""
    if is_json_value(value):
        return value
    raise TypeError(f"Expected JSON-compatible value, got {_type_name(value)}")


def ensure_json_object(value: object) -> JsonObject:
    """Return a JSON object or raise when a response body is not an object."""
    if isinstance(value, dict):
        items = cast(dict[object, object], value)
        if all(isinstance(key, str) and is_json_value(item) for key, item in items.items()):
            return cast(JsonObject, value)
    raise TypeError(f"Expected JSON object, got {_type_name(cast(object, value))}")


def ensure_json_array(value: object) -> JsonArray:
    """Return a JSON array or raise when a response body is not an array."""
    if isinstance(value, list):
        items = cast(list[object], value)
        if all(is_json_value(item) for item in items):
            return cast(JsonArray, value)
    raise TypeError(f"Expected JSON array, got {_type_name(cast(object, value))}")
