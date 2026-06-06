"""Coverage for dagnam.data.loaders.audio.transforms (framework adapters)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from tests.data.loaders._audio_helpers import (
    audio_meta,
    build_audio_folder,
    build_split_audio,
    install_fake_torchaudio,
    load_waveform_stub,
)

from dagnam.data.dataset import DagnamDataset

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch


# ---------------------------------------------------------------- transforms (framework adapters)


def test_create_pytorch_loader_unsplit(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    install_fake_torchaudio(monkeypatch)
    from dagnam.data.loaders.audio.transforms import create_pytorch_loader

    build_audio_folder(tmp_path, per_class=4)
    ds = DagnamDataset(audio_meta(num_samples=8), tmp_path)
    loader = create_pytorch_loader(
        ds, split="train", batch_size=2, num_workers=0, val_ratio=0.2, test_ratio=0.2, seed=0
    )
    batch = next(iter(loader))
    assert batch[0].shape[0] >= 1


def test_create_pytorch_loader_split(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    install_fake_torchaudio(monkeypatch)
    from dagnam.data.loaders.audio.transforms import create_pytorch_loader

    build_split_audio(tmp_path)
    ds = DagnamDataset(audio_meta(num_samples=12), tmp_path)
    loader = create_pytorch_loader(ds, split="val", batch_size=2, num_workers=0)
    batch = next(iter(loader))
    assert batch[0].shape[0] >= 1


def test_create_pytorch_loader_uses_meta_audio_cfg(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    install_fake_torchaudio(monkeypatch)
    from dagnam.data.loaders.audio.transforms import create_pytorch_loader

    build_audio_folder(tmp_path, per_class=2)
    ds = DagnamDataset(audio_meta(num_samples=4), tmp_path)
    loader = create_pytorch_loader(
        ds, split="train", batch_size=1, num_workers=0, val_ratio=0.25, test_ratio=0.25
    )
    next(iter(loader))


def test_create_tensorflow_dataset(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("tensorflow")
    from dagnam.data.loaders.audio import io as audio_io
    from dagnam.data.loaders.audio.transforms import create_tensorflow_dataset

    # Stub out load_waveform_py to skip needing soundfile/torchaudio.
    monkeypatch.setattr(
        audio_io,
        "load_waveform_py",
        load_waveform_stub,
    )
    # Also patch the symbol the transforms module imported earlier.
    from dagnam.data.loaders.audio import transforms as audio_transforms

    monkeypatch.setattr(
        audio_transforms,
        "load_waveform_py",
        load_waveform_stub,
    )

    build_audio_folder(tmp_path, per_class=2)
    ds = DagnamDataset(audio_meta(num_samples=4), tmp_path)
    tf_ds = create_tensorflow_dataset(
        ds, split="train", batch_size=2, shuffle=False, val_ratio=0.25, test_ratio=0.25, seed=0
    )
    next(iter(tf_ds))


def test_create_tensorflow_dataset_with_map_fns(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    from dagnam.data.loaders.audio import transforms as audio_transforms

    monkeypatch.setattr(
        audio_transforms,
        "load_waveform_py",
        load_waveform_stub,
    )

    def map_sample(waveform: object, label: object) -> tuple[object, object]:
        return waveform, label

    build_audio_folder(tmp_path, per_class=2)
    ds = DagnamDataset(audio_meta(num_samples=4), tmp_path)
    tf_ds = audio_transforms.create_tensorflow_dataset(
        ds,
        split="train",
        batch_size=2,
        shuffle=True,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=0,
        map_fn=map_sample,
        batch_map_fn=map_sample,
    )
    next(iter(tf_ds))


def test_create_flax_dataset(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("jax")
    from dagnam.data.loaders.audio import transforms as audio_transforms

    monkeypatch.setattr(
        audio_transforms,
        "load_waveform_py",
        load_waveform_stub,
    )

    build_audio_folder(tmp_path, per_class=2)
    ds = DagnamDataset(audio_meta(num_samples=4), tmp_path)
    batches = audio_transforms.create_flax_dataset(
        ds, split="train", batch_size=2, shuffle=False, val_ratio=0.25, test_ratio=0.25, seed=0
    )
    assert batches


def test_create_flax_dataset_with_transforms(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("jax")
    from dagnam.data.loaders.audio import transforms as audio_transforms

    monkeypatch.setattr(
        audio_transforms,
        "load_waveform_py",
        load_waveform_stub,
    )

    build_audio_folder(tmp_path, per_class=2)
    ds = DagnamDataset(audio_meta(num_samples=4), tmp_path)
    batches = audio_transforms.create_flax_dataset(
        ds,
        split="train",
        batch_size=2,
        shuffle=True,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=0,
        transform_fn=lambda w: w,
        batch_transform_fn=lambda f, lbl: (f, lbl),
    )
    assert batches


# ---------------------------------------------------------------- pure decode/setting helpers


class _NumpyScalar:
    """Minimal SupportsNumpy stand-in whose .numpy() returns a fixed value."""

    def __init__(self, value: object) -> None:
        self._value = value

    def numpy(self) -> object:
        return self._value


def test_decode_path_tensor_from_bytes() -> None:
    from dagnam.data.loaders.audio.transforms import _decode_path_tensor

    assert _decode_path_tensor(b"clip.wav") == "clip.wav"


def test_decode_path_tensor_from_str() -> None:
    from dagnam.data.loaders.audio.transforms import _decode_path_tensor

    assert _decode_path_tensor("clip.wav") == "clip.wav"


def test_decode_path_tensor_unwraps_numpy_bytes() -> None:
    from dagnam.data.loaders.audio.transforms import _decode_path_tensor

    # A tensor-like value whose .numpy() returns bytes is decoded to str.
    assert _decode_path_tensor(_NumpyScalar(b"path/clip.wav")) == "path/clip.wav"


def test_decode_path_tensor_falls_back_to_str() -> None:
    from dagnam.data.loaders.audio.transforms import _decode_path_tensor

    # A value that is neither bytes nor str is coerced via str().
    assert _decode_path_tensor(1234) == "1234"


def test_decode_label_tensor_from_int() -> None:
    import numpy as np

    from dagnam.data.loaders.audio.transforms import _decode_label_tensor

    result = _decode_label_tensor(7)
    assert result == np.int64(7)
    assert type(result) is np.int64


def test_decode_label_tensor_unwraps_numpy_generic() -> None:
    import numpy as np

    from dagnam.data.loaders.audio.transforms import _decode_label_tensor

    # SupportsNumpy returning a numpy generic → .item() then np.int64().
    result = _decode_label_tensor(_NumpyScalar(np.int32(3)))
    assert result == np.int64(3)


def test_decode_label_tensor_from_float() -> None:
    import numpy as np

    from dagnam.data.loaders.audio.transforms import _decode_label_tensor

    assert _decode_label_tensor(2.0) == np.int64(2)


def test_decode_label_tensor_rejects_non_scalar() -> None:
    from dagnam.data.loaders.audio.transforms import _decode_label_tensor

    with pytest.raises(TypeError, match="Expected TensorFlow scalar label"):
        _decode_label_tensor([1, 2, 3])


def test_int_setting_returns_default_for_none_and_containers() -> None:
    from dagnam.data.loaders.audio.transforms import _int_setting

    assert _int_setting(None, 16000) == 16000
    assert _int_setting(True, 16000) == 16000
    assert _int_setting({"a": 1}, 16000) == 16000
    assert _int_setting([1, 2], 16000) == 16000


def test_int_setting_coerces_numeric_and_string() -> None:
    from dagnam.data.loaders.audio.transforms import _int_setting

    assert _int_setting(22050, 16000) == 22050
    assert _int_setting(8000.0, 16000) == 8000
    assert _int_setting("44100", 16000) == 44100


def test_int_setting_returns_default_for_unhandled_type() -> None:
    from dagnam.data.loaders.audio.transforms import _int_setting

    # A tuple is neither a container guard case nor str/int/float → final default.
    assert _int_setting((1, 2), 16000) == 16000


# ---------------------------------------------------------------- fake-TF synchronous pipeline
#
# The real ``tf.data`` ``map``/``py_function`` closures inside
# ``create_tensorflow_dataset`` execute inside TensorFlow's graph runtime on
# worker threads, which coverage.py cannot trace even though the existing
# ``test_create_tensorflow_dataset`` test exercises them end-to-end. To get the
# tracer to follow the Python bodies of ``_load_one``/``_map`` (and the
# ``shuffle=None`` default), we substitute a minimal synchronous fake whose
# ``map`` runs the user function eagerly on the main thread.


class _FakeScalar:
    """A tensor-like scalar exposing the surface used by the map closure."""

    def __init__(self, value: object) -> None:
        self._value = value
        self.shape_set: object = None

    def numpy(self) -> object:
        return self._value

    def set_shape(self, shape: object) -> None:
        self.shape_set = shape


class _FakeTfDataset:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self._rows = rows

    def shuffle(self, *_args: object, **_kwargs: object) -> _FakeTfDataset:
        return self

    def map(self, fn: object, *_args: object, **_kwargs: object) -> _FakeTfDataset:
        mapper = cast("Callable[..., object]", fn)
        new_rows: list[tuple[object, ...]] = []
        for row in self._rows:
            result = mapper(*row)
            new_rows.append(tuple(result) if isinstance(result, tuple) else (result,))
        return _FakeTfDataset(new_rows)

    def batch(self, _batch_size: object, *_args: object, **_kwargs: object) -> _FakeTfDataset:
        return self

    def prefetch(self, *_args: object, **_kwargs: object) -> _FakeTfDataset:
        return self

    def __iter__(self) -> Iterator[tuple[object, ...]]:
        return iter(self._rows)


class _FakeDatasetFactory:
    @staticmethod
    def from_tensor_slices(tensors: tuple[Iterable[object], Iterable[object]]) -> _FakeTfDataset:
        paths, labels = tensors
        return _FakeTfDataset([(p, l) for p, l in zip(paths, labels, strict=False)])


class _FakeTfData:
    AUTOTUNE = -1
    Dataset = _FakeDatasetFactory()


class _FakeTfModule:
    data = _FakeTfData()
    float32 = "float32"
    int64 = "int64"

    @staticmethod
    def py_function(func: object, inp: list[object], _tout: object) -> tuple[object, object]:
        fn = cast("Callable[..., tuple[object, object]]", func)
        waveform, label = fn(*inp)
        return _FakeScalar(waveform), _FakeScalar(label)


def test_create_tensorflow_dataset_runs_map_closures_synchronously(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    """Cover the ``_map``/``_load_one`` closures by running the pipeline eagerly.

    Uses a synchronous fake TF module so coverage can trace the Python bodies
    that TensorFlow would otherwise execute in untraceable graph workers.
    """
    import numpy as np

    from dagnam.data.loaders.audio import transforms as audio_transforms

    monkeypatch.setattr(audio_transforms, "_load_tensorflow", lambda: _FakeTfModule())
    monkeypatch.setattr(audio_transforms, "load_waveform_py", load_waveform_stub)

    build_audio_folder(tmp_path, per_class=2)
    ds = DagnamDataset(audio_meta(num_samples=4), tmp_path)
    # shuffle omitted → defaults to True for the 'train' split (covers line 249).
    tf_ds = audio_transforms.create_tensorflow_dataset(
        ds, split="train", batch_size=2, val_ratio=0.25, test_ratio=0.25, seed=0
    )
    rows = list(tf_ds)
    assert rows
    # The fake ``tf.py_function`` wraps each element as a pair of ``_FakeScalar``.
    waveform, label = cast("tuple[_FakeScalar, _FakeScalar]", rows[0])
    # The decoded waveform is the stubbed zero array at the target length.
    assert np.asarray(waveform.numpy()).shape[0] > 0
    # set_shape was invoked by the _map closure.
    assert waveform.shape_set == [int(5.0 * 16000)]
    assert label.shape_set == []


# ---------------------------------------------------------------- create_pytorch_loader branches
#
# The metadata-resolution block in ``create_pytorch_loader`` (lines 162-177) has
# several "skip" legs that the happy-path tests above never exercise: explicit
# ``shuffle``, a non-None ``_meta_audio``, an explicit ``sample_rate``, a dataset
# without ``_raw_meta``, and a non-dict ``audio`` config. These fakes drive each
# alternate leg. The loader only reads ``data_dir`` plus those metadata hooks, so
# a lightweight stand-in suffices.


class _FakeAudioDataset:
    """Minimal DatasetMixinBase stand-in exposing only the attributes the loader reads."""

    def __init__(
        self,
        data_dir: Path,
        *,
        meta_audio: object = None,
        raw_meta: object | None = None,
        has_raw_meta: bool = True,
    ) -> None:
        self.data_dir = data_dir
        if meta_audio is not None:
            self._meta_audio = meta_audio
        if has_raw_meta:
            self._raw_meta: object = raw_meta if raw_meta is not None else {}
            self.raw_meta: object = raw_meta if raw_meta is not None else {}


def test_pytorch_loader_explicit_shuffle_and_sample_rate(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    """Explicit ``shuffle`` and ``sample_rate`` skip the default-resolution legs.

    Covers branches 162->166 (shuffle not None) and 171->180 (sample_rate not None).
    Also sets ``_meta_audio`` to a non-None value to cover 167->171.
    """
    install_fake_torchaudio(monkeypatch)
    from dagnam.data.loaders.audio.transforms import create_pytorch_loader

    build_audio_folder(tmp_path, per_class=2)
    fake = _FakeAudioDataset(tmp_path, meta_audio={"sample_rate": 16000})
    loader = create_pytorch_loader(
        cast("object", fake),  # type: ignore[arg-type]
        split="train",
        batch_size=1,
        num_workers=0,
        shuffle=False,
        sample_rate=22050,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=0,
    )
    batch = next(iter(loader))
    assert batch[0].shape[0] >= 1


def test_pytorch_loader_without_raw_meta_uses_default_sample_rate(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    """A dataset lacking ``_raw_meta`` falls through to the 16000 default.

    Covers branch 173->180 (no ``_raw_meta`` attribute) while ``sample_rate`` is
    None so the line-171 ``if`` body is entered.
    """
    install_fake_torchaudio(monkeypatch)
    from dagnam.data.loaders.audio.transforms import create_pytorch_loader

    build_audio_folder(tmp_path, per_class=2)
    fake = _FakeAudioDataset(tmp_path, meta_audio={}, has_raw_meta=False)
    loader = create_pytorch_loader(
        cast("object", fake),  # type: ignore[arg-type]
        split="train",
        batch_size=1,
        num_workers=0,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=0,
    )
    batch = next(iter(loader))
    assert batch[0].shape[0] >= 1


def test_pytorch_loader_non_dict_audio_cfg_uses_default_sample_rate(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    """A non-dict ``audio`` config in raw_meta skips the per-key resolution.

    Covers branch 175->180 (``audio_cfg`` is not a dict).
    """
    install_fake_torchaudio(monkeypatch)
    from dagnam.data.loaders.audio.transforms import create_pytorch_loader

    build_audio_folder(tmp_path, per_class=2)
    fake = _FakeAudioDataset(tmp_path, meta_audio={}, raw_meta={"audio": "not-a-dict"})
    loader = create_pytorch_loader(
        cast("object", fake),  # type: ignore[arg-type]
        split="train",
        batch_size=1,
        num_workers=0,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=0,
    )
    batch = next(iter(loader))
    assert batch[0].shape[0] >= 1


def test_create_flax_dataset_default_shuffle(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    """Omitting ``shuffle`` defaults it to True for the train split (covers line 306)."""
    pytest.importorskip("jax")
    from dagnam.data.loaders.audio import transforms as audio_transforms

    monkeypatch.setattr(audio_transforms, "load_waveform_py", load_waveform_stub)

    build_audio_folder(tmp_path, per_class=2)
    ds = DagnamDataset(audio_meta(num_samples=4), tmp_path)
    batches = audio_transforms.create_flax_dataset(
        ds, split="train", batch_size=2, val_ratio=0.25, test_ratio=0.25, seed=0
    )
    assert batches
