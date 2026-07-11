"""Dataset command handlers."""

from __future__ import annotations

import argparse
import base64
from copy import deepcopy
from pathlib import Path
import re
import sys
from typing import TYPE_CHECKING

from dagnam._types import JsonObject
from dagnam.cli.common import add_collection_output_args, confirm_or_abort, error, print_json
from dagnam.cli.presentation import Column, emit_result, render_table

if TYPE_CHECKING:
    from dagnam.cli.common import SubParsersAction

# Magic-number prefixes for the image formats a dataset preview can carry, mapped
# to the file extension used when decoding a base64 sample to disk.
_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpg"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"BM", "bmp"),
)


def _render_datasets(result: object) -> str:
    datasets = result if isinstance(result, list) else []
    if not datasets:
        return "No datasets found."
    rows = [dataset for dataset in datasets if isinstance(dataset, dict)]
    return render_table(
        (
            Column("ID", "id", 40),
            Column("Name", "name", 25),
            Column("Format", "format", 14),
            Column("Samples", "num_samples", 10, "right"),
            Column("Type", "dataset_type", 12),
        ),
        rows,
    )


def _redact_dataset_meta(meta: JsonObject, *, show_download_url: bool) -> JsonObject:
    result = deepcopy(meta)
    if "download_url" in result and not show_download_url:
        result["download_url"] = "<redacted>"
    return result


def cmd_dataset_list(args: argparse.Namespace) -> None:
    from dagnam._core.auth import get_api_key, get_api_url
    from dagnam._core.client import DagnamClient

    api_key = args.api_key or get_api_key()
    api_url = args.api_url or get_api_url()

    client = DagnamClient(api_url, api_key)
    datasets = client.list_datasets(type=args.type, search=args.search)

    emit_result(
        datasets,
        output=args.output,
        json_stdout=args.json or args.verbose,
        render_human=_render_datasets,
    )


def cmd_dataset_download(args: argparse.Namespace) -> None:
    import dagnam

    dataset_id: str = args.dataset_id
    output_dir = Path(args.output_dir)
    show_progress = not (args.no_progress or not sys.stderr.isatty())
    dagnam.load_dataset(dataset_id, cache_dir=str(output_dir), show_progress=show_progress)

    print(f"Dataset '{dataset_id}' downloaded to {output_dir / dataset_id}")


def cmd_dataset_info(args: argparse.Namespace) -> None:
    from dagnam._core.auth import get_api_key, get_api_url
    from dagnam._core.client import DagnamClient

    dataset_id: str = args.dataset_id

    api_key = get_api_key()
    api_url = get_api_url()

    client = DagnamClient(api_url, api_key)
    meta = client.get_dataset_meta(dataset_id)

    safe_meta = _redact_dataset_meta(meta, show_download_url=args.show_download_url)
    if args.output:
        from dagnam.cli.common import write_json_file

        write_json_file(args.output, safe_meta)
    if args.json:
        from dagnam.cli.common import print_json

        print_json(safe_meta)
        return
    for key, value in safe_meta.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        elif isinstance(value, list):
            print(f"{key}: {', '.join(str(v) for v in value)}")
        else:
            print(f"{key}: {value}")


def _decode_image_bytes(value: object) -> tuple[bytes, str] | None:
    """Return ``(image_bytes, extension)`` when ``value`` is base64 image data.

    Accepts both a bare base64 string and a ``data:image/...;base64,`` URI. The
    decoded bytes must begin with a known image magic number (PNG/JPEG/GIF/BMP/
    WebP) to qualify; anything else returns ``None`` so tabular string fields are
    never mistaken for images.
    """
    if not isinstance(value, str) or len(value) < 8:
        return None
    payload = (
        value.split(";base64,", 1)[1]
        if value.startswith("data:") and ";base64," in value
        else value
    )
    try:
        raw = base64.b64decode(payload, validate=True)
    except ValueError:
        # binascii.Error (bad alphabet/padding) is a ValueError subclass.
        return None
    for magic, ext in _IMAGE_MAGIC:
        if raw.startswith(magic):
            return raw, ext
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return raw, "webp"
    return None


def _sanitize_component(text: object) -> str:
    """Reduce an arbitrary value to a filesystem-safe filename component."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._")
    return cleaned or "x"


def _preview_samples(
    samples: list[JsonObject], dataset_id: str, out_dir: Path
) -> tuple[list[Path], list[dict[str, object]]]:
    """Split preview samples into saved image files and tabular row dicts.

    For each sample field carrying base64 image data, the image is decoded and
    written under ``out_dir`` with a sanitized filename, and the row records that
    filename in place of the raw bytes; every other field passes through as-is.
    """
    saved: list[Path] = []
    rows: list[dict[str, object]] = []
    for index, sample in enumerate(samples):
        row: dict[str, object] = {}
        for key, value in sample.items():
            decoded = _decode_image_bytes(value)
            if decoded is None:
                row[key] = value
                continue
            image_bytes, ext = decoded
            filename = (
                f"{_sanitize_component(dataset_id)}-sample{index}-{_sanitize_component(key)}.{ext}"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            dest = out_dir / filename
            dest.write_bytes(image_bytes)
            saved.append(dest)
            row[key] = filename
        rows.append(row)
    return saved, rows


def _render_preview_table(rows: list[dict[str, object]]) -> str:
    """Render preview rows as a width-aware table over the union of their keys."""
    ordered_keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in ordered_keys:
                ordered_keys.append(key)
    columns = tuple(Column(key, key, 32) for key in ordered_keys)
    return render_table(columns, rows)


def cmd_dataset_preview(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.preview_dataset(args.dataset_id, rows=args.rows)
    if args.json:
        print_json(result)
        return
    samples_value = result.get("samples")
    statistics_value = result.get("statistics")
    samples = (
        [s for s in samples_value if isinstance(s, dict)] if isinstance(samples_value, list) else []
    )
    # ``_preview_samples`` emits exactly one row per sample, so ``rows`` is
    # non-empty iff the backend returned any samples.
    saved, rows = _preview_samples(samples, args.dataset_id, Path(args.out))
    for path in saved:
        print(f"Saved image to {path}")
    if rows:
        print(_render_preview_table(rows))
    else:
        print("No samples returned.")
    if isinstance(statistics_value, dict) and statistics_value:
        print("\nStatistics:")
        for key, value in statistics_value.items():
            print(f"  {key}: {value}")


def cmd_dataset_update(args: argparse.Namespace) -> None:
    import dagnam

    result = dagnam.update_dataset(
        args.dataset_id,
        name=args.name,
        description=args.description,
        visibility=args.visibility,
    )
    print_json(result)


def cmd_dataset_delete(args: argparse.Namespace) -> None:
    import dagnam

    confirm_or_abort(
        f"This will permanently delete dataset {args.dataset_id}.",
        assume_yes=args.yes,
    )
    dagnam.delete_dataset(args.dataset_id)
    print(f"Dataset {args.dataset_id} deleted.")


def cmd_dataset_roles(args: argparse.Namespace) -> None:
    import dagnam

    column_roles: dict[str, str] = {}
    for item in args.set:
        column, sep, role = item.partition("=")
        if not sep or not column:
            error(f"Invalid --set value {item!r}; expected col=role.")
        column_roles[column] = role
    result = dagnam.update_dataset_roles(
        args.dataset_id, column_roles, task_type_hint=args.task_type_hint
    )
    print_json(result)


def cmd_dataset_upload(args: argparse.Namespace) -> None:
    import dagnam

    path = Path(args.path)
    if not path.is_file():
        error(f"No such file: {path}")
    name = args.name or path.stem
    result = dagnam.datasets.upload(
        str(path),
        name,
        args.type,
        args.format,
        description=args.description,
        visibility=args.visibility,
        license=args.license,
    )
    print_json(result)


def cmd_dataset_import_url(args: argparse.Namespace) -> None:
    import dagnam

    op = dagnam.datasets.upload_from_url(
        args.url,
        args.name,
        args.type,
        args.format,
        description=args.description,
        visibility=args.visibility,
    )
    if args.no_wait:
        print_json(op.initial())
        return
    print_json(op.wait().result())


def register_dataset(subparsers: SubParsersAction) -> None:
    """Register the ``dataset`` command group on the top-level subparsers."""
    dataset = subparsers.add_parser(
        "dataset",
        help="Browse and download datasets.",
        description="List, inspect, and download datasets from Dagnam.AI.",
    )
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)
    dataset_list = dataset_sub.add_parser(
        "list", help="List datasets.", description="List available datasets."
    )
    dataset_list.add_argument("--api-url", help="Override the API base URL.")
    dataset_list.add_argument("--api-key", help="Override the API key.")
    dataset_list.add_argument(
        "--type",
        default="all",
        help="Filter by dataset type: image, text, audio, video, tabular, custom, or all.",
    )
    dataset_list.add_argument("--search", help="Filter by name/description substring.")
    add_collection_output_args(dataset_list)
    dataset_list.set_defaults(func=cmd_dataset_list)
    dataset_info = dataset_sub.add_parser(
        "info", help="Show dataset metadata.", description="Show metadata for one dataset."
    )
    dataset_info.add_argument("dataset_id", help="ID of the dataset to inspect.")
    dataset_info.add_argument("--api-url", help="Override the API base URL.")
    dataset_info.add_argument("--api-key", help="Override the API key.")
    dataset_info.add_argument(
        "--show-download-url",
        action="store_true",
        help="Include signed download_url values in output. By default these are redacted.",
    )
    dataset_info.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    dataset_info.add_argument("--output", help="Write the redacted JSON response to this path.")
    dataset_info.set_defaults(func=cmd_dataset_info)
    dataset_download = dataset_sub.add_parser(
        "download",
        help="Download a dataset.",
        description="Download a dataset to a local directory.",
    )
    dataset_download.add_argument("dataset_id", help="ID of the dataset to download.")
    dataset_download.add_argument(
        "--output-dir", default=".", help="Destination directory (default: current dir)."
    )
    dataset_download.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable download progress output. Progress is also hidden when stderr is redirected.",
    )
    dataset_download.set_defaults(func=cmd_dataset_download)
    dataset_preview = dataset_sub.add_parser(
        "preview",
        help="Preview dataset samples.",
        description="Preview a dataset's samples and statistics; decodes image samples to disk.",
    )
    dataset_preview.add_argument("dataset_id", help="ID of the dataset to preview.")
    dataset_preview.add_argument(
        "--rows", type=int, default=10, help="Number of samples to fetch (1-100, default: 10)."
    )
    dataset_preview.add_argument(
        "--out", default=".", help="Directory for decoded image samples (default: current dir)."
    )
    dataset_preview.add_argument(
        "--json", action="store_true", help="Print the raw preview JSON instead of a table."
    )
    dataset_preview.set_defaults(func=cmd_dataset_preview)
    dataset_update = dataset_sub.add_parser(
        "update",
        help="Update dataset metadata.",
        description="Update a dataset's name, description, and/or visibility.",
    )
    dataset_update.add_argument("dataset_id", help="ID of the dataset to update.")
    dataset_update.add_argument("--name", help="New dataset name.")
    dataset_update.add_argument("--description", help="New dataset description.")
    dataset_update.add_argument("--visibility", help="New visibility (private or public).")
    dataset_update.set_defaults(func=cmd_dataset_update)
    dataset_delete = dataset_sub.add_parser(
        "delete",
        help="Delete a dataset.",
        description="Delete a dataset permanently.",
    )
    dataset_delete.add_argument("dataset_id", help="ID of the dataset to delete.")
    dataset_delete.add_argument(
        "--yes", action="store_true", help="Skip the typed confirmation prompt."
    )
    dataset_delete.set_defaults(func=cmd_dataset_delete)
    dataset_roles = dataset_sub.add_parser(
        "roles",
        help="Set dataset column roles.",
        description="Assign per-column roles (and an optional task-type hint) for a dataset.",
    )
    dataset_roles.add_argument("dataset_id", help="ID of the dataset.")
    dataset_roles.add_argument(
        "--set",
        action="append",
        required=True,
        metavar="COL=ROLE",
        help="Assign a column role, e.g. --set species=target (repeatable).",
    )
    dataset_roles.add_argument(
        "--task-type-hint", help="Optional task-type hint (e.g. classification)."
    )
    dataset_roles.set_defaults(func=cmd_dataset_roles)
    dataset_upload = dataset_sub.add_parser(
        "upload", help="Upload a local dataset file.", description="Upload a dataset file."
    )
    dataset_upload.add_argument("path", help="Path to the dataset file.")
    dataset_upload.add_argument("--name", help="Dataset name (default: file stem).")
    dataset_upload.add_argument(
        "--type", required=True, help="Dataset type (e.g. tabular, image, audio)."
    )
    dataset_upload.add_argument("--format", required=True, help="File format (e.g. csv, parquet).")
    dataset_upload.add_argument("--description", help="Dataset description.")
    dataset_upload.add_argument(
        "--visibility", default="private", help="public or private (default: private)."
    )
    dataset_upload.add_argument("--license", help="Dataset license identifier.")
    dataset_upload.set_defaults(func=cmd_dataset_upload)
    dataset_import = dataset_sub.add_parser(
        "import-url",
        help="Import a dataset from a URL.",
        description="Start a server-side dataset import and wait for it to finish.",
    )
    dataset_import.add_argument("url", help="Source URL of the dataset file.")
    dataset_import.add_argument("--name", required=True, help="Dataset name.")
    dataset_import.add_argument("--type", required=True, help="Dataset type.")
    dataset_import.add_argument("--format", required=True, help="File format.")
    dataset_import.add_argument("--description", help="Dataset description.")
    dataset_import.add_argument(
        "--visibility", default="private", help="public or private (default: private)."
    )
    dataset_import.add_argument(
        "--no-wait", action="store_true", help="Print the ingestion task and return immediately."
    )
    dataset_import.set_defaults(func=cmd_dataset_import_url)
