from __future__ import annotations

from pathlib import Path
import tarfile
import wave

import numpy as np

from dagnam.data.loaders.system.decoders import get_decoder

_LAYOUT_NO_DIR: dict[str, object] = {
    "audio": {"ext": [".wav"], "label_subdirs": True},
    "label": {"from": "subdir"},
}


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = (np.linspace(-0.5, 0.5, 16, dtype=np.float32) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(samples.tobytes())


def _make_tarball(tarball: Path, members: list[Path], staging: Path) -> None:
    with tarfile.open(tarball, "w:gz") as archive:
        for item in members:
            archive.add(item, arcname=str(item.relative_to(staging)))


def test_system_audio_folder_reads_wav_and_labels_from_sorted_subdirs(tmp_path: Path) -> None:
    _write_wav(tmp_path / "audio" / "yes" / "0.wav")
    _write_wav(tmp_path / "audio" / "no" / "0.wav")

    store = get_decoder("audio_folder").decode(
        tmp_path,
        {
            "audio": {"dir": "audio/", "ext": [".wav"], "label_subdirs": True},
            "label": {"from": "subdir"},
        },
        "train",
    )

    assert len(store) == 2
    assert sorted(int(store.column("label")[i]) for i in range(2)) == [0, 1]
    assert store.column("audio")[0].dtype == np.float32
    assert store.column("audio")[0].shape == (16,)


def test_system_audio_folder_extracts_tarball_with_root_class_dirs(tmp_path: Path) -> None:
    """Speech Commands ships a .tar.gz with class dirs at the archive root and no
    ``audio/`` wrapper; the decoder must extract it and resolve the root itself."""
    staging = tmp_path / "staging"
    _write_wav(staging / "yes" / "0.wav")
    _write_wav(staging / "no" / "0.wav")
    _make_tarball(tmp_path / "speech_commands_v0.02.tar.gz", sorted(staging.iterdir()), staging)

    store = get_decoder("audio_folder").decode(tmp_path, _LAYOUT_NO_DIR, "train")

    assert len(store) == 2
    assert sorted(int(store.column("label")[i]) for i in range(2)) == [0, 1]


def test_system_audio_folder_tarball_single_root_dir_is_reused_on_second_decode(
    tmp_path: Path,
) -> None:
    """A single-top-dir archive collapses to that dir; a second decode reuses the
    already-unpacked tree rather than re-extracting."""
    staging = tmp_path / "staging"
    _write_wav(staging / "speech" / "yes" / "0.wav")
    _write_wav(staging / "speech" / "no" / "0.wav")
    _make_tarball(tmp_path / "data.tar.gz", [staging / "speech"], staging)

    decoder = get_decoder("audio_folder")
    first = decoder.decode(tmp_path, _LAYOUT_NO_DIR, "train")
    second = decoder.decode(tmp_path, _LAYOUT_NO_DIR, "train")

    assert len(first) == len(second) == 2


def test_system_audio_folder_falls_back_when_configured_dir_absent(tmp_path: Path) -> None:
    """An explicit ``dir`` that does not exist falls back to the extracted root."""
    staging = tmp_path / "staging"
    _write_wav(staging / "yes" / "0.wav")
    _write_wav(staging / "no" / "0.wav")
    _make_tarball(tmp_path / "clips.tar.gz", sorted(staging.iterdir()), staging)

    store = get_decoder("audio_folder").decode(
        tmp_path,
        {"audio": {"dir": "missing/", "ext": [".wav"], "label_subdirs": True}, "label": {}},
        "train",
    )

    assert len(store) == 2


def test_system_audio_folder_excludes_background_noise_dir(tmp_path: Path) -> None:
    _write_wav(tmp_path / "audio" / "yes" / "0.wav")
    _write_wav(tmp_path / "audio" / "no" / "0.wav")
    _write_wav(tmp_path / "audio" / "_background_noise_" / "0.wav")

    store = get_decoder("audio_folder").decode(
        tmp_path,
        {
            "audio": {"dir": "audio/", "ext": [".wav"], "label_subdirs": True},
            "label": {"from": "subdir"},
        },
        "train",
    )

    assert len(store) == 2
    assert sorted(int(store.column("label")[i]) for i in range(2)) == [0, 1]


def test_system_audio_folder_rejects_traversal_dir(tmp_path: Path) -> None:
    import pytest

    from dagnam.data.loaders.system.decoders.base import DecodeError

    with pytest.raises(DecodeError):
        get_decoder("audio_folder").decode(
            tmp_path,
            {"audio": {"dir": "../../../etc", "ext": [".wav"]}, "label": {"from": "subdir"}},
            "train",
        )
