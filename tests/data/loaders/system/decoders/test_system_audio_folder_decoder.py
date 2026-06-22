from __future__ import annotations

from pathlib import Path
import wave

import numpy as np

from dagnam.data.loaders.system.decoders import get_decoder


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = (np.linspace(-0.5, 0.5, 16, dtype=np.float32) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(samples.tobytes())


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
