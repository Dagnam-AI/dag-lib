"""Model Hub publishing: create -> upload files -> (version) -> finalize, as one call.

Kept apart from :mod:`dagnam.resources.hub` (the per-route functions) so each file stays
readable; ``dagnam.hub.publish`` is re-exported from there.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Optional
from uuid import UUID

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import (
    APIError,
    HubError,
    HubModelNotFoundError,
    HubRuleError,
    PayloadTooLargeError,
    UploadError,
)
from dagnam._core.resolver import resolve_client
from dagnam._types import JsonObject, JsonValue
from dagnam.resources.hub_files import await_clean_files, replace_refused_copy, stored_name

_RETRYABLE_STATUS = frozenset({0, 408, 429})

STAGES = ("design", "pretrained", "finetuned", "adapter", "quantized", "merged", "distilled")
POST_TRAINED = STAGES[2:]
STAGE_ALIASES = {"pre-trained": "pretrained", "fine-tuned": "finetuned"}
LICENSES = (
    "mit",
    "apache2",
    "gpl",
    "proprietary",
    "custom",
    "bsd-3-clause",
    "gpl-3.0",
    "agpl-3.0",
    "lgpl-3.0",
    "mpl-2.0",
    "cc0-1.0",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "cc-by-nc-4.0",
    "cc-by-nc-sa-4.0",
    "openrail-m",
    "llama-community",
    "gemma",
)


def normalize_stage(value: str) -> str:
    """A stage name the platform accepts; ``pre-trained`` and ``fine-tuned`` are accepted too."""
    stage = STAGE_ALIASES.get(value.strip().lower(), value.strip().lower())
    if stage == "post-trained":
        raise ValueError(f"Choose the post-trained stage: {', '.join(POST_TRAINED)}")
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {value!r}; use one of {', '.join(STAGES)}")
    return stage


def split_architecture(obj: JsonObject) -> tuple[str, JsonObject]:
    """The request field a design object goes in: Studio state, or a codegen ``dag.json``."""
    return ("architecture", obj) if "diagram_state" in obj else ("architecture_ir", obj)


def base_fields(base: str) -> JsonObject:
    """``base_model_id`` for a page id (a UUID), else ``base_model_name`` for a plain name."""
    try:
        UUID(base)
    except ValueError:
        return {"base_model_name": base}
    return {"base_model_id": base}


def _source_fields(
    architecture: Optional[JsonObject],
    project_id: Optional[str],
    architecture_version_id: Optional[str],
) -> JsonObject:
    """The architecture source of a create or version request (at most one)."""
    if architecture is not None and project_id is not None:
        raise ValueError("Give either architecture or project_id, not both")
    if architecture_version_id is not None and project_id is None:
        raise ValueError("architecture_version_id needs project_id")
    fields: JsonObject = {}
    if architecture is not None:
        key, value = split_architecture(architecture)
        fields[key] = value
    if project_id is not None:
        fields["project_id"] = project_id
    if architecture_version_id is not None:
        fields["architecture_version_id"] = architecture_version_id
    return fields


def _retryable(exc: Exception) -> bool:
    """Only a transport failure (0), a timeout (408), a rate limit (429) or a 5xx can succeed again.

    A rejected file, a conflict, a request above the size ceiling (413) and a local file that
    changed mid-send (``OSError``) answer the same way on every attempt.
    """
    return isinstance(exc, APIError) and (
        exc.status_code in _RETRYABLE_STATUS or exc.status_code >= 500
    )


def create_payload(
    *,
    name: str,
    description: str,
    task_type: str,
    framework: str,
    license: str,
    visibility: str,
    tags: Optional[list[str]],
    metadata: Optional[JsonObject],
) -> JsonObject:
    """The body of ``POST /hub/models``, shared by ``hub.create`` and :func:`publish`."""
    payload: JsonObject = {
        "name": name,
        "description": description,
        "task_type": task_type,
        "framework": framework,
        "license": license,
        "visibility": visibility,
    }
    if tags is not None:
        payload["tags"] = [str(tag) for tag in tags]
    if metadata is not None:
        payload["metadata"] = metadata
    return payload


def finalize(
    model_id: str | UUID,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Publish a draft model: ``POST /hub/models/{id}/finalize``.

    Raises ``HubModelNotFoundError`` when the platform answers 404 and ``APIError`` for any
    other refusal; the model stays a draft in both cases.
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.finalize_hub_model(str(model_id))


def _check_files(files: Sequence[str]) -> None:
    """``FileNotFoundError`` for a missing file, ``ValueError`` for two with one stored name."""
    for file_path in files:
        if not Path(file_path).is_file():
            raise FileNotFoundError(f"No such file: {file_path}")
    # Names are compared ignoring case: refusing a pair that only differs by case costs a
    # rename, accepting one the hub treats as a duplicate would finalize a resume without it.
    stored = [stored_name(file_path).casefold() for file_path in files]
    duplicates = sorted({entry for entry in stored if stored.count(entry) > 1})
    if duplicates:
        raise ValueError(
            "Two files would be stored under one name (the hub keeps one file per name, and a "
            f"PyTorch .pt/.pth or HDF5 .h5 file is stored as <stem>.safetensors): {', '.join(duplicates)}"
        )


def _draft(model_id: str, version_id: Optional[str]) -> tuple[str, str]:
    """What a halted publish names (the draft) and how to carry on from it."""
    if version_id is None:
        return f"hub model {model_id}", f"resume with hub.publish(..., model_id={model_id!r})"
    return (
        f"version {version_id} of hub model {model_id}",
        "the version is still a draft: add the rest with "
        f"hub.upload_file(..., version_id={version_id!r}) and publish it with "
        f"hub.finalize_version({model_id!r}, {version_id!r}), or remove it with "
        f"hub.delete_version({model_id!r}, {version_id!r})",
    )


def _upload_all(
    resolved: DagnamClient,
    model_id: str,
    files: Sequence[str],
    version_id: Optional[str],
    on_file_progress: Optional[Callable[[str, int, int, str], object]],
    max_retries_per_file: int,
    resume: bool,
) -> list[JsonValue]:
    """Upload each file (into ``version_id`` when given), retrying only what can succeed again."""
    total = len(files)
    uploaded: list[JsonValue] = []
    for index, file_path in enumerate(files, start=1):
        if on_file_progress is not None:
            on_file_progress(file_path, index, total, "uploading")
        attempts = 0
        replaced = False
        while True:
            try:
                record = resolved.upload_model_file(model_id, file_path, version_id)
                uploaded.append(record)
                if on_file_progress is not None:
                    on_file_progress(file_path, index, total, "uploaded")
                    if record.get("scan_status") == "pending":
                        on_file_progress(file_path, index, total, "inspecting")
                break
            except (APIError, HubError, UploadError, PayloadTooLargeError, OSError) as exc:
                if (
                    isinstance(exc, APIError)
                    and not isinstance(exc, HubRuleError)
                    and exc.status_code == 409
                    and (resume or attempts)
                ):
                    # A refused copy from an earlier run is not "already there": remove it, retry once.
                    if resume and not replaced:
                        try:
                            replaced = replace_refused_copy(
                                resolved, model_id, version_id, file_path
                            )
                        except (APIError, HubError, HubModelNotFoundError) as cleanup:
                            subject, advice = _draft(model_id, version_id)
                            raise UploadError(
                                f"Publish of {subject} halted: could not remove the refused copy "
                                f"of {file_path} ({cleanup}). Fix the issue, then {advice}."
                            ) from cleanup
                        if replaced:
                            continue
                    if on_file_progress is not None:
                        on_file_progress(file_path, index, total, "skipped")
                    break
                attempts += 1
                if attempts > max_retries_per_file or not _retryable(exc):
                    if on_file_progress is not None:
                        on_file_progress(file_path, index, total, "failed")
                    done = ", ".join(str(Path(p).name) for p in files[: index - 1]) or "none"
                    subject, advice = _draft(model_id, version_id)
                    raise UploadError(
                        f"Publish of {subject} halted: {file_path} failed "
                        f"after {attempts} attempt(s) ({exc}). Uploaded so far: {done}. "
                        f"Fix the issue, then {advice}."
                    ) from exc
                if on_file_progress is not None:
                    on_file_progress(file_path, index, total, "retrying")
    return uploaded


def publish(
    *,
    name: str,
    description: str,
    task_type: str,
    framework: str,
    files: Sequence[str] = (),
    stage: Optional[str] = None,
    base: Optional[str] = None,
    architecture: Optional[JsonObject] = None,
    project_id: Optional[str] = None,
    architecture_version_id: Optional[str] = None,
    registry_version_id: Optional[str] = None,
    version: str = "1.0.0",
    changelog: Optional[str] = None,
    license: str = "mit",
    license_url: Optional[str] = None,
    license_name: Optional[str] = None,
    visibility: str = "public",
    tags: Optional[list[str]] = None,
    metadata: Optional[JsonObject] = None,
    model_id: Optional[str] = None,
    max_retries_per_file: int = 2,
    on_file_progress: Optional[Callable[[str, int, int, str], object]] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Publish a model to the hub: create -> upload files -> finalize.

    ``architecture`` publishes a design with no weights: a Studio ``{diagram_state,
    architecture_config}`` object or a codegen ``dag.json`` object. ``project_id`` (with
    ``architecture_version_id``) takes the design from one of your Studio projects instead.
    ``stage`` is ``design``, ``pretrained`` or one of the post-trained stages ``finetuned``,
    ``adapter``, ``quantized``, ``merged``, ``distilled`` (``pre-trained`` and ``fine-tuned``
    are accepted); leave it out for the platform to decide. ``base`` names the model this one
    was made from: a Hub page id, or a plain name such as ``Qwen/Qwen2.5-0.5B-Instruct``.
    ``registry_version_id`` publishes a registry version as a card and takes no stage, design
    or files. ``version`` and ``changelog`` name the first version.

    With ``model_id`` the create step is skipped and the files go into that draft: a file the
    draft already holds (the platform answers 409) is reported ``skipped`` and the rest are
    uploaded, so a halted publish is resumed by calling again with the id from its error.
    ``name``, ``description``, ``task_type``, ``framework`` and the design are not sent on a
    resume.

    ``on_file_progress(path, index, total, state)`` receives per-file states:
    ``uploading`` -> (``retrying`` ...) -> ``uploaded`` (then ``inspecting`` when the platform
    inspects uploads) | ``failed`` | ``skipped``, and ``refused`` for a file the platform refused.

    Before finalizing, ``publish`` waits for the platform to inspect every file of the draft
    (a platform that reports no scan state is not waited for). A refused file is removed and
    ``UploadError`` names it with the platform's reason; ``InspectionTimeoutError`` says the
    wait ran out and the draft is kept.

    An upload is retried up to ``max_retries_per_file`` times, and only on a transport
    failure, a timeout (408), a rate limit (429) or a server error; any other 4xx, a file
    above the size ceiling and a local file that changed mid-send halt at once. A 409 after
    such a retry means the first attempt stored the file; it is ``skipped``.

    Files above about 500 MB are not supported yet: the platform refuses the request
    whatever the plan allows. PyTorch ``.pt``/``.pth`` and HDF5 ``.h5`` files are converted by the
    platform and stored as ``<stem>.safetensors``; ``.safetensors``, ``.onnx`` and the JSON
    sidecars (``config.json``, ``tokenizer.json``, ...) are stored as they are.

    Raises ``ValueError`` before any network call for an unknown stage or licence, a custom
    licence without ``license_url`` on a page that is not private, a post-trained stage without
    ``base`` (unless resuming), the design stage with files, a registry card combined with a
    stage, design or files, two design sources, or two files that would be stored under one
    name (``stored_name``); ``FileNotFoundError`` if a local file is missing; ``UploadError`` if
    a file still fails (the draft and its uploaded files are kept: resume with ``model_id``);
    ``HubError`` if the platform refuses the finalize call (the model is still a draft and the
    error names it: ``hub.finalize(model_id)`` retries that step alone). The result is ``{"model", "files",
    "version", "finalized"}``; ``version`` is the published model's ``latest_version`` and
    ``finalized`` is always ``True``, kept for callers that read it.
    """
    _check_files(files)
    if license not in LICENSES:
        raise ValueError(f"Unknown licence {license!r}; use one of {', '.join(LICENSES)}")
    if license == "custom" and visibility != "private" and not license_url:
        raise ValueError("A public or unlisted model with the custom licence needs license_url")
    source = _source_fields(architecture, project_id, architecture_version_id)
    if registry_version_id is not None and (stage or source or files):
        raise ValueError("A registry card takes no stage, design or files")
    checked = normalize_stage(stage) if stage is not None else None
    if checked == "design" and files:
        raise ValueError("A design takes no files")
    if checked in POST_TRAINED and base is None and model_id is None:
        raise ValueError(f"A {checked} model needs base (the model it was made from)")
    extra: JsonObject = {**source, "version": version}
    if checked is not None:
        extra["stage"] = checked
    if base is not None:
        extra.update(base_fields(base))
    if registry_version_id is not None:
        extra["registry_version_id"] = registry_version_id
    if changelog is not None:
        extra["changelog"] = changelog
    if license_url is not None:
        extra["license_url"] = license_url
    if license_name is not None:
        extra["license_name"] = license_name

    resolved = resolve_client(client, api_key, api_url)
    resume = model_id is not None
    if model_id is None:
        created = resolved.create_hub_model(
            {
                **create_payload(
                    name=name,
                    description=description,
                    task_type=task_type,
                    framework=framework,
                    license=license,
                    visibility=visibility,
                    tags=tags,
                    metadata=metadata,
                ),
                **extra,
            }
        )
        model_id = str(created["id"])

    uploaded = _upload_all(
        resolved, model_id, files, None, on_file_progress, max_retries_per_file, resume
    )
    await_clean_files(
        resolved,
        model_id,
        None,
        files,
        on_file_progress,
        f"resume with hub.publish(..., model_id={model_id!r})",
    )
    try:
        model = finalize(model_id, client=resolved)
    except HubRuleError as exc:
        raise HubError(
            f"Hub model {model_id} is still a draft: {exc.message} ({exc.code}). "
            f"Retry with hub.finalize({model_id!r}) once that is resolved."
        ) from exc
    except (HubModelNotFoundError, APIError) as exc:
        code, detail = (exc.status_code, exc.message) if isinstance(exc, APIError) else (404, exc)
        raise HubError(
            f"Hub model {model_id} has its files but is still a draft: the platform answered "
            f"HTTP {code} to the finalize call ({detail}). Fix what it names (hub.update can "
            f"change the base or licence), then retry that step with hub.finalize({model_id!r})."
        ) from exc

    return {
        "model": model,
        "files": uploaded,
        "version": model.get("latest_version"),
        "finalized": True,
    }


def publish_version(
    model_id: str | UUID,
    version: str,
    *,
    changelog: Optional[str] = None,
    files: Sequence[str] = (),
    architecture: Optional[JsonObject] = None,
    project_id: Optional[str] = None,
    architecture_version_id: Optional[str] = None,
    max_retries_per_file: int = 2,
    on_file_progress: Optional[Callable[[str, int, int, str], object]] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Publish a new version of a model you manage: create a draft, upload files, finalize.

    ``architecture`` or ``project_id`` (with ``architecture_version_id``) attach a design
    snapshot as in :func:`publish`. Returns the published version. Like :func:`publish` it
    waits for the platform to inspect the files first and removes a refused one. A failed or
    refused upload raises ``UploadError`` and a refused finalize raises ``HubError``; both leave the draft version
    (named in the error): it cannot be resumed by this function, so finish it with
    ``hub.finalize_version`` or remove it with ``hub.delete_version``.
    """
    _check_files(files)
    payload: JsonObject = {
        "version": version,
        **_source_fields(architecture, project_id, architecture_version_id),
    }
    if changelog is not None:
        payload["changelog"] = changelog
    resolved = resolve_client(client, api_key, api_url)
    mid = str(model_id)
    draft = resolved.create_hub_model_version(mid, payload)
    version_id = str(draft["id"])
    _upload_all(
        resolved, mid, files, version_id, on_file_progress, max_retries_per_file, resume=False
    )
    await_clean_files(
        resolved,
        mid,
        version_id,
        files,
        on_file_progress,
        f"add replacements with hub.upload_file(..., version_id={version_id!r}) and publish it "
        f"with hub.finalize_version({mid!r}, {version_id!r}), or remove it with "
        f"hub.delete_version({mid!r}, {version_id!r})",
    )
    try:
        return resolved.finalize_hub_model_version(mid, version_id)
    except HubRuleError as exc:
        raise HubError(
            f"Version {version_id} of hub model {mid} is still a draft: {exc.message} "
            f"({exc.code}). Fix that, then publish it with "
            f"hub.finalize_version({mid!r}, {version_id!r}), or remove it with "
            f"hub.delete_version({mid!r}, {version_id!r})."
        ) from exc
    except (HubModelNotFoundError, HubError, APIError) as exc:
        raise HubError(
            f"Version {version_id} of hub model {mid} is still a draft: the platform refused to "
            f"publish it ({exc}). Fix that, then publish it with "
            f"hub.finalize_version({mid!r}, {version_id!r}), or remove it with "
            f"hub.delete_version({mid!r}, {version_id!r})."
        ) from exc


__all__ = [
    "LICENSES",
    "POST_TRAINED",
    "STAGES",
    "STAGE_ALIASES",
    "base_fields",
    "create_payload",
    "finalize",
    "normalize_stage",
    "publish",
    "publish_version",
    "split_architecture",
    "stored_name",
]
