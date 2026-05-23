"""Tests for column_roles parameter in DagnamDataset.to_pytorch_loader()."""

from pathlib import Path
from unittest.mock import patch

from dagnam.data.dataset import DagnamDataset


class TestColumnRolesForwarding:
    """Tests that column_roles is forwarded to tabular loaders."""

    def test_to_pytorch_loader_forwards_column_roles_to_csv_loader(self, tmp_path: Path) -> None:
        """column_roles kwarg is passed through to csv_loader."""
        (tmp_path / "data.csv").write_text("x,label,ignore_me\n1,a,0\n2,b,1\n", encoding="utf-8")
        ds = DagnamDataset(
            {
                "id": "tab-1",
                "name": "Tabular",
                "format": "csv",
                "dataset_type": "tabular",
                "num_samples": 2,
                "num_classes": 2,
                "class_names": ["a", "b"],
                "feature_schema": None,
            },
            tmp_path,
        )
        with patch(
            "dagnam.data.loaders.csv.create_pytorch_loader",
            return_value="loader",
        ) as mock_create:
            result = ds.to_pytorch_loader(
                split="train",
                batch_size=1,
                num_workers=0,
                column_roles={"x": "feature", "label": "target", "ignore_me": "ignore"},
            )

        assert result == "loader"
        assert mock_create.call_args.kwargs["column_roles"] == {
            "x": "feature",
            "label": "target",
            "ignore_me": "ignore",
        }

    def test_to_pytorch_loader_forwards_column_roles_to_json_loader(self, tmp_path: Path) -> None:
        """column_roles kwarg is passed through to json_loader."""
        (tmp_path / "data.jsonl").write_text(
            '{"x": 1, "label": "a"}\n{"x": 2, "label": "b"}\n',
            encoding="utf-8",
        )
        ds = DagnamDataset(
            {
                "id": "tab-2",
                "name": "JSON Tabular",
                "format": "jsonl",
                "dataset_type": "tabular",
                "num_samples": 2,
                "num_classes": 2,
                "class_names": ["a", "b"],
                "feature_schema": None,
            },
            tmp_path,
        )
        with patch(
            "dagnam.data.loaders.json_array.create_pytorch_loader",
            return_value="loader",
        ) as mock_create:
            result = ds.to_pytorch_loader(
                split="train",
                batch_size=1,
                num_workers=0,
                column_roles={"x": "feature", "label": "target"},
            )

        assert result == "loader"
        assert mock_create.call_args.kwargs["column_roles"] == {
            "x": "feature",
            "label": "target",
        }

    def test_column_roles_none_by_default(self, tmp_path: Path) -> None:
        """column_roles defaults to None when not provided."""
        (tmp_path / "data.csv").write_text("x,label\n1,a\n2,b\n", encoding="utf-8")
        ds = DagnamDataset(
            {
                "id": "tab-3",
                "name": "Tabular",
                "format": "csv",
                "dataset_type": "tabular",
                "num_samples": 2,
                "num_classes": 2,
                "class_names": ["a", "b"],
                "feature_schema": None,
            },
            tmp_path,
        )
        with patch(
            "dagnam.data.loaders.csv.create_pytorch_loader",
            return_value="loader",
        ) as mock_create:
            ds.to_pytorch_loader(split="train", batch_size=1, num_workers=0)

        # column_roles should not be in kwargs or should be None
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs.get("column_roles") is None

    def test_column_roles_not_passed_to_image_loader(self, tmp_path: Path) -> None:
        """column_roles is NOT passed to image_folder_loader."""
        ds = DagnamDataset(
            {
                "id": "img-roles",
                "name": "Images",
                "format": "image_folder",
                "dataset_type": "image",
                "num_samples": 4,
                "num_classes": 2,
            },
            tmp_path,
        )
        with patch(
            "dagnam.data.loaders.image_folder.create_pytorch_loader",
            return_value="loader",
        ) as mock_create:
            ds.to_pytorch_loader(
                split="train",
                batch_size=2,
                num_workers=0,
                column_roles={"x": "feature"},
            )

        # column_roles should NOT be in the call kwargs for image loader
        call_kwargs = mock_create.call_args.kwargs
        assert "column_roles" not in call_kwargs

    def test_column_roles_not_passed_to_audio_loader(self, tmp_path: Path) -> None:
        """column_roles is NOT passed to audio_loader."""
        ds = DagnamDataset(
            {
                "id": "aud-roles",
                "name": "Audio",
                "format": "audio_folder",
                "dataset_type": "audio",
                "num_samples": 4,
                "num_classes": 2,
            },
            tmp_path,
        )
        with patch(
            "dagnam.data.loaders.audio.create_pytorch_loader",
            return_value="loader",
        ) as mock_create:
            ds.to_pytorch_loader(
                split="train",
                batch_size=2,
                num_workers=0,
                column_roles={"x": "feature"},
            )

        # column_roles should NOT be in the call kwargs for audio loader
        call_kwargs = mock_create.call_args.kwargs
        assert "column_roles" not in call_kwargs
