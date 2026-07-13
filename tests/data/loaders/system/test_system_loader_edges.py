from __future__ import annotations

import hashlib
from pathlib import Path
import tarfile
from typing import ClassVar, cast
import wave

import numpy as np
import pytest
from tests.typing_helpers import PytestMonkeyPatch

from dagnam._core.exceptions import APIError
from dagnam._types import JsonObject, NativeSplit, TensorflowDataset
from dagnam.data.dataset import DagnamDataset
from dagnam.data.loaders.system import dispatch
from dagnam.data.loaders.system.bound_dataset import BoundNativeDataset
from dagnam.data.loaders.system.column_store import Column, ColumnStore
from dagnam.data.loaders.system.decoders._helpers import extensions, safe_extract_tar, spec_dict
from dagnam.data.loaders.system.decoders.array import ArrayDecoder
from dagnam.data.loaders.system.decoders.audio_folder import AudioFolderDecoder, read_wav
from dagnam.data.loaders.system.decoders.base import DecodeError
from dagnam.data.loaders.system.decoders.image_folder import ImageFolderDecoder
from dagnam.data.loaders.system.decoders.image_mask_folder import ImageMaskFolderDecoder
from dagnam.data.loaders.system.decoders.tabular import TabularDecoder
from dagnam.data.loaders.system.decoders.text import TextDecoder
from dagnam.data.loaders.system.transform_executor import apply_transform

PIL_Image = pytest.importorskip("PIL.Image")


def test_columnstore_empty_missing_column_and_misconfigured_lazy() -> None:
    assert len(ColumnStore({})) == 0
    with pytest.raises(KeyError, match="column 'missing'"):
        ColumnStore({"x": Column.eager(np.arange(1))}).column("missing")
    broken = Column(None, None, None)
    with pytest.raises(IndexError, match="lazy column"):
        broken[0]


def test_bound_dataset_requires_input_and_can_default_target_to_input() -> None:
    store = ColumnStore({"x": Column.eager(np.zeros((1, 2), dtype=np.float32))})
    with pytest.raises(ValueError, match="input_column"):
        BoundNativeDataset(store, {}, [])[0]
    x, y = BoundNativeDataset(store, {"input_column": "x"}, [])[0]
    assert np.asarray(x).tolist() == np.asarray(y).tolist()


def test_helper_validation_and_safe_extract_rejects_escape(tmp_path: Path) -> None:
    with pytest.raises(DecodeError, match="missing layout"):
        spec_dict({}, "image")
    with pytest.raises(DecodeError, match="layout ext"):
        extensions({"ext": ".jpg"})

    tarball = tmp_path / "bad.tar.gz"
    payload = b"bad"
    info = tarfile.TarInfo("../escape.txt")
    info.size = len(payload)
    with tarfile.open(tarball, "w:gz") as archive:
        archive.addfile(info, fileobj=__import__("io").BytesIO(payload))
    with pytest.raises(DecodeError, match="escapes destination"):
        safe_extract_tar(tarball, tmp_path / "out")


def _make_tar(path: Path, names_sizes: list[tuple[str, int]]) -> None:
    import io

    with tarfile.open(path, "w:gz") as archive:
        for name, size in names_sizes:
            info = tarfile.TarInfo(name)
            info.size = size
            archive.addfile(info, fileobj=io.BytesIO(b"x" * size))


def test_safe_extract_tar_rejects_decompression_bomb(
    tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    from dagnam.data.loaders.system.decoders import _helpers

    # Too many members.
    many = tmp_path / "many.tar.gz"
    _make_tar(many, [(f"f{i}.txt", 1) for i in range(4)])
    monkeypatch.setattr(_helpers, "_MAX_TAR_MEMBERS", 2)
    with pytest.raises(DecodeError, match="too many members"):
        safe_extract_tar(many, tmp_path / "out-many")

    # Total uncompressed size over the cap.
    monkeypatch.setattr(_helpers, "_MAX_TAR_MEMBERS", 200_000)
    monkeypatch.setattr(_helpers, "_MAX_TAR_UNCOMPRESSED_BYTES", 4)
    big = tmp_path / "big.tar.gz"
    _make_tar(big, [("a.bin", 8)])
    with pytest.raises(DecodeError, match="uncompressed size exceeds"):
        safe_extract_tar(big, tmp_path / "out-big")


def test_array_decoder_missing_artifact_and_key(tmp_path: Path) -> None:
    decoder = ArrayDecoder()
    with pytest.raises(DecodeError, match=r"no \.npz"):
        decoder.decode(tmp_path, {"x": {"key": "x"}}, "train")
    np.savez(tmp_path / "d.npz", x=np.arange(2))
    with pytest.raises(DecodeError, match="missing"):
        decoder.decode(tmp_path, {"y": {"key": "missing"}}, "train")


def test_audio_decoder_error_paths_and_stereo_wav(tmp_path: Path) -> None:
    decoder = AudioFolderDecoder()
    layout = cast("dict[str, object]", {"audio": {"dir": "audio", "ext": [".wav"]}})
    # A configured dir that is absent falls back to the (empty) artifact root.
    with pytest.raises(DecodeError, match="no label subdirectories"):
        decoder.decode(tmp_path, layout, "train")
    (tmp_path / "audio").mkdir()
    with pytest.raises(DecodeError, match="no label subdirectories"):
        decoder.decode(tmp_path, layout, "train")
    (tmp_path / "audio" / "yes").mkdir()
    with pytest.raises(DecodeError, match="no audio files"):
        decoder.decode(tmp_path, layout, "train")

    wav_path = tmp_path / "audio" / "yes" / "a.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(np.asarray([1000, 3000, 5000, 7000], dtype=np.int16).tobytes())
    assert read_wav(wav_path).shape == (2,)

    bad = tmp_path / "audio" / "yes" / "bad.wav"
    with wave.open(str(bad), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(1)
        wav.setframerate(16_000)
        wav.writeframes(np.asarray([1, 2], dtype=np.uint8).tobytes())
    with pytest.raises(DecodeError, match="sample width"):
        read_wav(bad)


def test_image_folder_decoder_error_paths(tmp_path: Path) -> None:
    decoder = ImageFolderDecoder()
    layout = cast("dict[str, object]", {"image": {"dir": "images", "ext": [".jpg"]}})
    with pytest.raises(DecodeError, match="does not exist"):
        decoder.decode(tmp_path, layout, "train")
    (tmp_path / "images").mkdir()
    with pytest.raises(DecodeError, match="no class"):
        decoder.decode(tmp_path, layout, "train")
    (tmp_path / "images" / "cat").mkdir()
    with pytest.raises(DecodeError, match="no images"):
        decoder.decode(tmp_path, layout, "train")


def test_image_mask_decoder_error_paths_and_cached_unpack(tmp_path: Path) -> None:
    decoder = ImageMaskFolderDecoder()
    no_mask = cast("dict[str, object]", {"image": {"dir": "images", "ext": [".jpg"]}})
    with pytest.raises(DecodeError, match="mask column"):
        decoder.decode(tmp_path, no_mask, "train")

    layout = cast(
        "dict[str, object]",
        {
            "image": {"dir": "images", "ext": [".jpg"]},
            "mask": {"dir": "masks", "ext": [".png"]},
        },
    )
    with pytest.raises(DecodeError, match="missing image/mask"):
        decoder.decode(tmp_path, layout, "train")

    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    PIL_Image.new("RGB", (2, 2)).save(tmp_path / "images" / "a.jpg")
    with pytest.raises(DecodeError, match="no paired"):
        decoder.decode(tmp_path, layout, "train")

    unpacked = tmp_path / "_unpacked_image_mask_folder" / "root"
    (unpacked / "images").mkdir(parents=True)
    (unpacked / "masks").mkdir()
    (tmp_path / "oxford-pets.tar.gz").touch()
    PIL_Image.new("RGB", (2, 2)).save(unpacked / "images" / "a.jpg")
    PIL_Image.new("L", (2, 2)).save(unpacked / "masks" / "a.png")
    assert len(decoder.decode(tmp_path, layout, "train")) == 1


def test_tabular_and_text_decoder_error_paths(tmp_path: Path) -> None:
    with pytest.raises(DecodeError, match="no csv/parquet"):
        TabularDecoder().decode(tmp_path, {"x": {"column": "x"}}, "train")
    (tmp_path / "d.csv").write_text("x\n1\n", encoding="utf-8")
    with pytest.raises(DecodeError, match="missing"):
        TabularDecoder().decode(tmp_path, {"y": {"column": "y"}}, "train")

    text_decoder = TextDecoder()
    with pytest.raises(DecodeError, match="requires text layout"):
        text_decoder.decode(tmp_path, {}, "train")
    with pytest.raises(DecodeError, match=r"layout\.text\.file"):
        text_decoder.decode(tmp_path, {"text": {}}, "train")
    with pytest.raises(DecodeError, match="does not exist"):
        text_decoder.decode(tmp_path, {"text": {"file": "missing.txt"}}, "train")


def test_transform_executor_edges() -> None:
    with pytest.raises(ValueError, match="size"):
        apply_transform(np.zeros((2, 2)), {"kind": "image_resize", "params": {"size": [2]}}, None)
    mask = apply_transform(
        np.asarray([[3, 1]], dtype=np.uint8),
        {"kind": "mask", "params": {"remap": "contiguous_long"}},
        None,
    )
    assert mask.tolist() == [[1, 0]]
    plain_mask = apply_transform(
        np.asarray([[3]], dtype=np.uint8), {"kind": "mask", "params": {}}, None
    )
    assert plain_mask.dtype == np.int64
    tokens = apply_transform(
        np.asarray([1, 2]),
        {"kind": "tokenize", "params": {"sequence_length": 4}},
        None,
    )
    assert tokens.tolist() == [1, 2, 0, 0]
    no_pad = apply_transform(
        np.asarray([1, 2]),
        {"kind": "tokenize", "params": {"sequence_length": 2}},
        None,
    )
    assert no_pad.tolist() == [1, 2]


def test_converter_compatibility_prefers_framework_native_when_also_generic() -> None:
    tf = pytest.importorskip("tensorflow")
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from dagnam.data.loaders.flax import FlaxBatch

    meta: JsonObject = {
        "id": "mixed-native",
        "name": "Mixed Native",
        "format": "native",
        "dataset_type": "image",
        "num_samples": 2,
        "num_classes": 2,
    }
    ds = DagnamDataset(meta, data_dir=None)
    ds.native_train = cast(
        "NativeSplit",
        (
            np.zeros((2, 1), dtype=np.float32).tolist(),
            np.asarray([0, 1], dtype=np.int64).tolist(),
        ),
    )
    ds.native_train_tf = cast(
        "TensorflowDataset",
        tf.data.Dataset.from_tensor_slices(
            (np.ones((2, 1), dtype=np.float32), np.asarray([1, 0], dtype=np.int64))
        ),
    )
    ds.native_train_flax = [
        FlaxBatch(
            features=jnp.asarray(np.ones((2, 1), dtype=np.float32)), labels=jnp.asarray([1, 0])
        )
    ]

    batch = cast(
        "tuple[object, object]",
        next(iter(ds.to_tensorflow_dataset(split="train", batch_size=1, shuffle=False))),
    )
    assert batch[0]
    assert ds.to_flax_dataset(split="train", batch_size=1, shuffle=False)


class _Response:
    headers: ClassVar[dict[str, str]] = {"Content-Length": "3"}

    def __init__(self, chunks: list[bytes], status: int = 200) -> None:
        self._chunks = chunks
        self.status_code = status

    def close(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        del chunk_size
        yield from self._chunks

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


def test_dispatch_artifact_helpers(tmp_path: Path, monkeypatch: PytestMonkeyPatch) -> None:
    assert dispatch.detect_installed_framework() == "generic"
    assert dispatch._artifact_filename({"filename": "a.npz"}) == "a.npz"  # type: ignore[attr-defined]
    assert dispatch._artifact_filename({"artifact": {"filename": "b.npz"}}) == "b.npz"  # type: ignore[attr-defined]
    assert dispatch._artifact_filename({"file_path": str(tmp_path / "c.npz")}) == "c.npz"  # type: ignore[attr-defined]
    assert dispatch._artifact_filename({}) is None  # type: ignore[attr-defined]
    assert dispatch._artifact_source({"download_url": "u", "checksum": "c"}) == ("u", "c")  # type: ignore[attr-defined]
    assert dispatch._artifact_source({"artifact": {"download_url": "au", "checksum": "ac"}}) == (  # type: ignore[attr-defined]
        "au",
        "ac",
    )

    source = tmp_path / "source.bin"
    source.write_bytes(b"abc")
    dispatch._copy_local_artifact(source, source)  # type: ignore[attr-defined]
    copied = tmp_path / "copied.bin"
    dispatch._copy_local_artifact(source, copied)  # type: ignore[attr-defined]
    assert copied.read_bytes() == b"abc"

    # The empty middle chunk exercises the ``if chunk:`` skip guard in
    # ``_download_artifact`` (a real transport never yields an empty chunk).
    monkeypatch.setattr(
        "dagnam.data.loaders.system.dispatch.requests.get",
        lambda *a, **k: _Response([b"a", b"", b"b"]),
    )
    downloaded = tmp_path / "downloaded.bin"
    dispatch._download_artifact("https://example.test/d.bin", downloaded)  # type: ignore[attr-defined]
    assert downloaded.read_bytes() == b"ab"

    # A non-transient error status is mapped to APIError and not retried; the
    # staged .tmp file is cleaned up on failure.
    monkeypatch.setattr(
        "dagnam.data.loaders.system.dispatch.requests.get",
        lambda *a, **k: _Response([], status=404),
    )
    failed = tmp_path / "failed.bin"
    with pytest.raises(APIError):
        dispatch._download_artifact("https://example.test/f.bin", failed)  # type: ignore[attr-defined]
    assert not failed.with_suffix(".bin.tmp").exists()


def test_dispatch_artifact_dir_variants(tmp_path: Path, monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setattr(dispatch, "SYSTEM_CACHE_ROOT", tmp_path / "cache")
    source = tmp_path / "local.npz"
    source.write_bytes(b"local")
    meta: JsonObject = {"id": "local", "filename": "local.npz", "file_path": str(source)}
    assert (dispatch._artifact_dir(meta) / "local.npz").read_bytes() == b"local"  # type: ignore[attr-defined]
    no_url = {"id": "no-url", "filename": "missing.npz"}
    assert dispatch._artifact_dir(no_url) == tmp_path / "cache" / "no-url"  # type: ignore[attr-defined]

    # A download_url that is a local filesystem path (not http/https) is
    # hostile server input and must be refused, never silently copied (the old
    # Path(url).exists() local-copy branch was an arbitrary-file-read vector).
    remote_source = tmp_path / "remote.npz"
    remote_source.write_bytes(b"remote")
    meta = {"id": "path-url", "filename": "remote.npz", "download_url": str(remote_source)}
    with pytest.raises(ValueError, match="non-http"):
        dispatch._artifact_dir(meta)  # type: ignore[attr-defined]

    checksum_file = tmp_path / "checked.bin"
    checksum = hashlib.sha256(b"abc").hexdigest()
    monkeypatch.setattr(dispatch, "_download_artifact", lambda _url, dest: dest.write_bytes(b"abc"))
    dispatch._ensure_verified_file("https://example.test/checked", checksum_file, checksum)  # type: ignore[attr-defined]
    assert checksum_file.read_bytes() == b"abc"

    monkeypatch.setattr(
        dispatch,
        "_ensure_verified_file",
        lambda _url, dest, _checksum: dest.write_bytes(b"checked"),
    )
    meta = {
        "id": "checksum",
        "filename": "checksum.npz",
        "download_url": "https://example.test/c",
        "checksum": "sha256:abc",
    }
    assert (dispatch._artifact_dir(meta) / "checksum.npz").read_bytes() == b"checked"  # type: ignore[attr-defined]

    nonexistent = {
        "id": "nonexistent-file-path",
        "filename": "missing-source.npz",
        "file_path": str(tmp_path / "does-not-exist.npz"),
    }
    assert dispatch._artifact_dir(nonexistent) == tmp_path / "cache" / "nonexistent-file-path"  # type: ignore[attr-defined]

    monkeypatch.setattr(dispatch, "_download_artifact", lambda _url, dest: dest.write_bytes(b"net"))
    meta = {"id": "download", "filename": "download.npz", "download_url": "https://example.test/d"}
    assert (dispatch._artifact_dir(meta) / "download.npz").read_bytes() == b"net"  # type: ignore[attr-defined]
    assert dispatch._artifact_dir(meta) == tmp_path / "cache" / "download"  # type: ignore[attr-defined]
    assert dispatch._artifact_dir({"id": "empty"}) == tmp_path / "cache" / "empty"  # type: ignore[attr-defined]


def test_dispatch_artifact_dir_rejects_server_path_traversal(
    tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    """A hostile server descriptor cannot escape SYSTEM_CACHE_ROOT.

    Both the dataset id and the artifact filename come from the server. A
    traversal payload in either must land strictly inside the cache root, never
    at an arbitrary path such as ~/.bashrc.
    """
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(dispatch, "SYSTEM_CACHE_ROOT", cache_root)
    outside = tmp_path / "outside.txt"

    captured: dict[str, Path] = {}
    monkeypatch.setattr(
        dispatch,
        "_download_artifact",
        lambda _url, dest: captured.__setitem__("dest", dest) or dest.write_bytes(b"x"),
    )

    # Traversal in the filename resolves to a basename inside the cache dir.
    meta: JsonObject = {
        "id": "ds",
        "filename": f"{'../' * 8}{outside.name}",
        "download_url": "https://example.test/a",
    }
    dispatch._artifact_dir(meta)  # type: ignore[attr-defined]
    assert not outside.exists()
    assert cache_root.resolve() in captured["dest"].resolve().parents

    # Traversal in the dataset id is percent-encoded into a single component.
    captured.clear()
    meta = {
        "id": f"{'../' * 8}evil",
        "filename": "a.npz",
        "download_url": "https://example.test/a",
    }
    dispatch._artifact_dir(meta)  # type: ignore[attr-defined]
    assert cache_root.resolve() in captured["dest"].resolve().parents

    # A filename that reduces to nothing usable writes no file and returns the
    # cache dir rather than a bogus destination.
    captured.clear()
    result = dispatch._artifact_dir(  # type: ignore[attr-defined]
        {"id": "ds", "filename": "..", "download_url": "https://example.test/a"}
    )
    assert result == cache_root / "ds"
    assert "dest" not in captured


def test_safe_artifact_filename_reduces_to_basename() -> None:
    assert dispatch._safe_artifact_filename("../../.bashrc") == ".bashrc"  # type: ignore[attr-defined]
    assert dispatch._safe_artifact_filename("a/b/c.npz") == "c.npz"  # type: ignore[attr-defined]
    assert dispatch._safe_artifact_filename("C:\\x\\d.bin") == "d.bin"  # type: ignore[attr-defined]
    assert dispatch._safe_artifact_filename("..") == ""  # type: ignore[attr-defined]
    assert dispatch._safe_artifact_filename("plain.npz") == "plain.npz"  # type: ignore[attr-defined]


def test_download_artifact_rejects_non_http_scheme(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-http"):
        dispatch._download_artifact("file:///etc/passwd", tmp_path / "x")  # type: ignore[attr-defined]


def test_download_artifact_rejects_oversized_content_length(
    tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    from dagnam._core.exceptions import DownloadTooLargeError

    class _Resp:
        headers: ClassVar[dict[str, str]] = {"Content-Length": "100"}
        status_code: ClassVar[int] = 200

        def raise_for_status(self) -> None:
            pass

        def iter_content(self, chunk_size: int) -> object:
            yield b"x" * 100

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *_a: object) -> None:
            pass

    monkeypatch.setattr(dispatch, "resolve_max_download_bytes", lambda: 4)
    monkeypatch.setattr(dispatch.requests, "get", lambda *_a, **_k: _Resp())
    with pytest.raises(DownloadTooLargeError):
        dispatch._download_artifact("https://example.test/a", tmp_path / "a.bin")
    assert not (tmp_path / "a.bin").exists()
    assert not (tmp_path / "a.bin.tmp").exists()


def test_download_artifact_aborts_body_over_cap(
    tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    from dagnam._core.exceptions import DownloadTooLargeError

    class _Resp:
        headers: ClassVar[dict[str, str]] = {}
        status_code: ClassVar[int] = 200

        def raise_for_status(self) -> None:
            pass

        def iter_content(self, chunk_size: int) -> object:
            yield b"xx"
            yield b"xx"
            yield b"xx"

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *_a: object) -> None:
            pass

    monkeypatch.setattr(dispatch, "resolve_max_download_bytes", lambda: 4)
    monkeypatch.setattr(dispatch.requests, "get", lambda *_a, **_k: _Resp())
    with pytest.raises(DownloadTooLargeError):
        dispatch._download_artifact("https://example.test/a", tmp_path / "a.bin")
    assert not (tmp_path / "a.bin").exists()
    assert not (tmp_path / "a.bin.tmp").exists()
