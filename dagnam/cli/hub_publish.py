"""Hub publishing command handlers: ``hub publish`` and ``hub version create``."""

from __future__ import annotations

import argparse
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
import sys
from typing import TYPE_CHECKING, NoReturn

from dagnam._types import ensure_json_object
from dagnam.cli.common import load_json_arg, print_json, print_next_step
from dagnam.cli.errors import render_message
from dagnam.resources.hub_publish import LICENSES

if TYPE_CHECKING:
    from dagnam._types import JsonObject
    from dagnam.cli.common import SubParsersAction

FILE_KINDS_HELP = (
    ".safetensors, .onnx, .pt, .pth, .h5, or one of config.json, tokenizer.json, "
    "tokenizer_config.json, generation_config.json, special_tokens_map.json. PyTorch (.pt, .pth) "
    "and HDF5 (.h5) weights are converted to .safetensors, and the platform inspects every file "
    "before it is published. Files above about 500 MB are not supported yet."
)
_FILE_HELP = f"A file (repeat for several): {FILE_KINDS_HELP}"


def _usage_error(message: str) -> NoReturn:
    """Print ``message`` in the CLI error frame and exit 2, like a bad flag combination."""
    print(render_message(message), file=sys.stderr)
    sys.exit(2)


@contextmanager
def sdk_usage_errors() -> Generator[None]:
    """Show an SDK pre-check message (``ValueError``, missing file) as a usage error.

    Without this the CLI's catch-all hides the message behind "unexpected error".
    """
    try:
        yield
    except (ValueError, FileNotFoundError) as exc:
        _usage_error(str(exc))


def _progress(path: str, index: int, total: int, state: str) -> None:
    print(f"[{index}/{total}] {Path(path).name}: {state}", file=sys.stderr)


def _check_design_args(args: argparse.Namespace) -> None:
    if args.architecture_version_id and not args.project_id:
        _usage_error("--architecture-version needs --project")


def design_from_args(args: argparse.Namespace) -> JsonObject | None:
    return ensure_json_object(load_json_arg(args.architecture)) if args.architecture else None


def cmd_hub_publish(args: argparse.Namespace) -> None:
    import dagnam

    _check_design_args(args)
    stage = args.stage
    if stage is None and not args.file and not args.registry_version_id:
        stage = "design"
    if args.file and stage is None and not args.registry_version_id:
        _usage_error("--stage is required with --file")
    if stage == "design" and not (args.architecture or args.project_id):
        _usage_error("a design needs --architecture or --project")

    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None
    with sdk_usage_errors():
        result = dagnam.hub.publish(
            name=args.name,
            description=args.description,
            task_type=args.task_type,
            framework=args.framework,
            files=args.file,
            stage=stage,
            base=args.base,
            architecture=design_from_args(args),
            project_id=args.project_id,
            architecture_version_id=args.architecture_version_id,
            registry_version_id=args.registry_version_id,
            model_id=args.model_id,
            version=args.version,
            changelog=args.changelog,
            license=args.license,
            license_name=args.license_name,
            license_url=args.license_url,
            visibility=args.visibility,
            tags=tags,
            metadata=ensure_json_object(load_json_arg(args.metadata)) if args.metadata else None,
            on_file_progress=_progress,
        )
    print_json(result)
    model = result.get("model")
    model_id = model.get("id") if isinstance(model, dict) else None
    print_next_step(f"dagnam hub get {model_id or '<model-id>'}")


def cmd_hub_version_create(args: argparse.Namespace) -> None:
    import dagnam

    _check_design_args(args)
    with sdk_usage_errors():
        result = dagnam.hub.publish_version(
            args.model_id,
            args.version,
            files=args.file,
            changelog=args.changelog,
            architecture=design_from_args(args),
            project_id=args.project_id,
            architecture_version_id=args.architecture_version_id,
            on_file_progress=_progress,
        )
    print_json(result)


def add_design_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--architecture", help="A design file (@path): a Studio design or a dag.json."
    )
    parser.add_argument("--project", dest="project_id", help="Use this project's design.")
    parser.add_argument(
        "--architecture-version",
        dest="architecture_version_id",
        help="The project's architecture version (default: its current one).",
    )


def register_hub_publish(hub_sub: SubParsersAction) -> None:
    """Add ``hub publish`` and ``hub version create`` to the hub command group."""
    publish = hub_sub.add_parser(
        "publish",
        help="Publish a model or a design to the hub.",
        description=(
            "Create a hub page with its first version, upload the files, and publish it. "
            "A design needs no files."
        ),
    )
    publish.add_argument("--name", required=True, help="Model name.")
    publish.add_argument("--description", required=True, help="Model description.")
    publish.add_argument(
        "--task-type",
        required=True,
        dest="task_type",
        help=(
            "Task type: classification, detection, segmentation, generation, nlp, audio, "
            "multimodal or other."
        ),
    )
    publish.add_argument("--framework", required=True, help="Framework (e.g. pytorch).")
    publish.add_argument(
        "--stage",
        help=(
            "design, pretrained, finetuned, adapter, quantized, merged or distilled (the last "
            "five are post-trained stages; pre-trained and fine-tuned are accepted). Default: "
            "design when no --file and no --registry-version is given; required with --file."
        ),
    )
    add_design_args(publish)
    publish.add_argument(
        "--base", help="The model this one derives from: a Hub model id or a name."
    )
    publish.add_argument(
        "--registry-version",
        dest="registry_version_id",
        help="Publish a ready registry version you own as a card.",
    )
    publish.add_argument(
        "--license",
        default="mit",
        help="One of: " + ", ".join(LICENSES) + " (default: mit).",
    )
    publish.add_argument("--license-name", help="Display name of a custom licence.")
    publish.add_argument(
        "--license-url",
        help="https link to a custom licence's text (needed unless the page is private).",
    )
    publish.add_argument("--metadata", help="Model card fields as JSON (@path or a literal).")
    publish.add_argument("--version", default="1.0.0", help="The first version (default: 1.0.0).")
    publish.add_argument("--changelog", help="Changelog of the first version.")
    publish.add_argument("--file", action="append", default=[], help=_FILE_HELP)
    publish.add_argument(
        "--visibility", default="public", help="public, private or unlisted (default: public)."
    )
    publish.add_argument("--tags", help="Comma-separated tags.")
    publish.add_argument(
        "--model-id",
        dest="model_id",
        help="Resume a draft: upload into this model instead of creating a new one.",
    )
    publish.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON (publish always prints JSON; accepted for compatibility).",
    )
    publish.set_defaults(func=cmd_hub_publish)

    version = hub_sub.add_parser(
        "version", help="Manage versions of a model.", description="Manage versions of a model."
    )
    version_sub = version.add_subparsers(dest="version_command", required=True)
    create = version_sub.add_parser(
        "create",
        help="Publish a new version of a model.",
        description="Create a draft version, upload its files, and publish it.",
    )
    create.add_argument("model_id", help="ID of the hub model.")
    create.add_argument("--version", required=True, help="Version string (e.g. 1.1.0).")
    create.add_argument("--changelog", help="Changelog of this version.")
    create.add_argument("--file", action="append", default=[], help=_FILE_HELP)
    add_design_args(create)
    create.set_defaults(func=cmd_hub_version_create)
