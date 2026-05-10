"""Dataset command handlers."""

from __future__ import annotations

import argparse

from dagnam.cli.common import _error


def _cmd_dataset_list(args: argparse.Namespace) -> None:
    from dagnam._core.auth import get_api_key, get_api_url
    from dagnam._core.client import DagnamClient
    from dagnam._core.exceptions import DagnamError

    try:
        api_key = get_api_key()
        api_url = get_api_url()
    except DagnamError as exc:
        _error(str(exc))

    client = DagnamClient(api_url, api_key)
    try:
        datasets = client.list_datasets()
    except DagnamError as exc:
        _error(str(exc))

    if not datasets:
        print("No datasets found.")
        return

    # Print formatted table
    header = f"{'ID':<40} {'Name':<25} {'Format':<8} {'Samples':>8} {'Type':<12}"
    print(header)
    print("-" * len(header))
    for ds in datasets:
        print(
            f"{ds.get('id', 'N/A'):<40} "
            f"{ds.get('name', 'N/A'):<25} "
            f"{ds.get('format', 'N/A'):<8} "
            f"{ds.get('num_samples', 'N/A'):>8} "
            f"{ds.get('dataset_type', 'N/A'):<12}"
        )


def _cmd_dataset_download(args: argparse.Namespace) -> None:
    import dagnam
    from dagnam._core.exceptions import DagnamError
    from dagnam.data.cache import DEFAULT_CACHE_DIR

    dataset_id: str = args.dataset_id
    try:
        dagnam.load_dataset(dataset_id)
    except DagnamError as exc:
        _error(str(exc))

    print(f"Dataset '{dataset_id}' downloaded to {DEFAULT_CACHE_DIR / dataset_id}")


def _cmd_dataset_info(args: argparse.Namespace) -> None:
    from dagnam._core.auth import get_api_key, get_api_url
    from dagnam._core.client import DagnamClient
    from dagnam._core.exceptions import DagnamError

    dataset_id: str = args.dataset_id

    try:
        api_key = get_api_key()
        api_url = get_api_url()
    except DagnamError as exc:
        _error(str(exc))

    client = DagnamClient(api_url, api_key)
    try:
        meta = client.get_dataset_meta(dataset_id)
    except DagnamError as exc:
        _error(str(exc))

    for key, value in meta.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        elif isinstance(value, list):
            print(f"{key}: {', '.join(str(v) for v in value)}")
        else:
            print(f"{key}: {value}")
