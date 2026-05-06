"""Tests for audio folder dataset loader."""

from pathlib import Path
from unittest.mock import patch

import pytest

from dagnam.data.dataset import DagnamDataset


# ------------------------------------------------------------------
# DagnamDataset dispatch tests
# ------------------------------------------------------------------


class TestAudioFolderDispatch:
    """Tests that DagnamDataset dispatches audio_folder to the audio loader."""

    def test_audio_folder_dispatches_to_loader(self, tmp_path: Path):
        """audio_folder format routes to audio_loader.create_pytorch_loader."""
        ds = DagnamDataset(
            {
                "id": "aud-1",
                "name": "Audio",
                "format": "audio_folder",
                "dataset_type": "audio",
                "num_samples": 2,
                "num_classes": 2,
                "class_names": ["yes", "no"],
                "feature_schema": None,
                "audio": {"sample_rate": 16000, "n_mels": 64},
            },
            tmp_path,
        )
        with patch(
            "dagnam.data.loaders.audio_loader.create_pytorch_loader",
            return_value="loader",
        ) as mock_create:
            result = ds.to_pytorch_loader(split="train", batch_size=2, num_workers=0)

        assert result == "loader"
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["split"] == "train"
        assert call_kwargs["batch_size"] == 2
        assert call_kwargs["num_workers"] == 0

    def test_audio_dataset_type_dispatches_to_audio_loader(self, tmp_path: Path):
        """dataset_type='audio' with non-tabular format routes to audio loader."""
        ds = DagnamDataset(
            {
                "id": "aud-2",
                "name": "Audio Custom",
                "format": "custom_audio",
                "dataset_type": "audio",
                "num_samples": 10,
                "num_classes": 3,
            },
            tmp_path,
        )
        with patch(
            "dagnam.data.loaders.audio_loader.create_pytorch_loader",
            return_value="audio_loader",
        ) as mock_create:
            result = ds.to_pytorch_loader(split="train", batch_size=4, num_workers=0)

        assert result == "audio_loader"
        mock_create.assert_called_once()

    def test_audio_import_error_message(self, tmp_path: Path):
        """Raises ImportError with helpful message when torchaudio is missing."""
        ds = DagnamDataset(
            {
                "id": "aud-3",
                "name": "Audio",
                "format": "audio_folder",
                "dataset_type": "audio",
                "num_samples": 2,
                "num_classes": 2,
            },
            tmp_path,
        )
        # Mock the audio_loader module to raise ImportError
        with patch(
            "dagnam.data.loaders.audio_loader.create_pytorch_loader",
            side_effect=ImportError("dagnam[audio]"),
        ):
            with pytest.raises(ImportError, match="dagnam\\[audio\\]"):
                ds.to_pytorch_loader(split="train", batch_size=2, num_workers=0)

    def test_csv_audio_dataset_uses_csv_loader(self, tmp_path: Path):
        """CSV format with audio dataset_type still uses csv_loader."""
        (tmp_path / "data.csv").write_text(
            "feature,label\n1.0,yes\n2.0,no\n", encoding="utf-8"
        )
        ds = DagnamDataset(
            {
                "id": "aud-csv",
                "name": "Audio CSV",
                "format": "csv",
                "dataset_type": "audio",
                "num_samples": 2,
                "num_classes": 2,
                "class_names": ["yes", "no"],
                "feature_schema": {
                    "columns": [
                        {"name": "feature", "type": "numeric"},
                        {"name": "label", "type": "categorical"},
                    ]
                },
            },
            tmp_path,
        )
        # CSV format should use csv_loader even if dataset_type is audio
        with patch(
            "dagnam.data.loaders.csv_loader.create_pytorch_loader",
            return_value="csv_loader",
        ) as mock_csv:
            result = ds.to_pytorch_loader(split="train", batch_size=2, num_workers=0)

        assert result == "csv_loader"
        mock_csv.assert_called_once()
