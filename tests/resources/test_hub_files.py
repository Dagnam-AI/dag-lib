"""``dagnam.hub.wait_for_inspection``, ``delete_file`` and ``link_design``
(``dagnam/resources/hub_files.py``)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from dagnam import hub
from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import HubError, InspectionTimeoutError
from dagnam._types import JsonObject
from dagnam.resources import hub_files


class _Clock:
    """A fake ``time`` module: ``sleep`` moves ``monotonic`` forward, nothing really waits."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(
        hub_files, "time", SimpleNamespace(sleep=fake.sleep, monotonic=fake.monotonic)
    )
    return fake


def _listing(*states: str | None) -> JsonObject:
    files: list[JsonObject] = [
        {"id": f"f{n}", **({} if state is None else {"scan_status": state})}
        for n, state in enumerate(states)
    ]
    return {"model_id": "m1", "version_id": "v1", "files": list(files)}


class TestWaitForInspection:
    def test_it_returns_every_file_as_soon_as_none_is_pending(self, clock: _Clock) -> None:
        client = MagicMock(spec=DagnamClient)
        client.list_hub_model_files.side_effect = [
            _listing("pending", "clean"),
            _listing("pending", "clean"),
            _listing("rejected", "clean"),
        ]
        files = hub.wait_for_inspection("m1", client=client)
        assert [entry["scan_status"] for entry in files] == ["rejected", "clean"]
        assert client.list_hub_model_files.call_count == 3
        assert clock.sleeps == [3.0, 3.0]

    def test_a_file_an_earlier_run_uploaded_is_waited_for_too(self, clock: _Clock) -> None:
        client = MagicMock(spec=DagnamClient)
        client.list_hub_model_files.side_effect = [_listing("pending"), _listing("clean")]
        assert len(hub.wait_for_inspection("m1", poll=1.5, client=client)) == 1
        assert clock.sleeps == [1.5]

    def test_it_passes_the_version_through(self, clock: _Clock) -> None:
        client = MagicMock(spec=DagnamClient)
        client.list_hub_model_files.return_value = _listing("clean")
        hub.wait_for_inspection("m1", version_id="v9", client=client)
        hub.wait_for_inspection("m1", client=client)
        assert [call.args for call in client.list_hub_model_files.call_args_list] == [
            ("m1",),
            ("m1",),
        ]
        assert [call.kwargs for call in client.list_hub_model_files.call_args_list] == [
            {"version_id": "v9"},
            {"version_id": None},
        ]

    def test_a_platform_without_inspection_reports_no_state(self, clock: _Clock) -> None:
        client = MagicMock(spec=DagnamClient)
        client.list_hub_model_files.return_value = _listing(None, None)
        assert len(hub.wait_for_inspection("m1", client=client)) == 2
        assert clock.sleeps == []

    def test_a_listing_without_a_files_list_reads_as_empty(self, clock: _Clock) -> None:
        client = MagicMock(spec=DagnamClient)
        client.list_hub_model_files.side_effect = [{"files": "nope"}, {}]
        assert hub.wait_for_inspection("m1", client=client) == []
        assert hub.wait_for_inspection("m1", client=client) == []

    def test_a_non_object_entry_is_ignored(self, clock: _Clock) -> None:
        client = MagicMock(spec=DagnamClient)
        client.list_hub_model_files.return_value = {"files": ["x", {"id": "f1"}]}
        assert hub.wait_for_inspection("m1", client=client) == [{"id": "f1"}]

    def test_it_raises_after_the_timeout_and_names_the_next_step(self, clock: _Clock) -> None:
        client = MagicMock(spec=DagnamClient)
        client.list_hub_model_files.return_value = _listing("pending")
        with pytest.raises(InspectionTimeoutError, match=r"m1.*hub\.finalize") as raised:
            hub.wait_for_inspection("m1", timeout=10.0, poll=4.0, client=client)
        assert isinstance(raised.value, HubError)
        assert clock.sleeps == [4.0, 4.0, 4.0]

    def test_it_is_exported(self) -> None:
        assert {"delete_file", "link_design", "wait_for_inspection"} <= set(hub.__all__)


class TestDeleteFile:
    def test_it_delegates_to_the_client(self) -> None:
        client = MagicMock(spec=DagnamClient)
        assert hub.delete_file("m1", "f1", client=client) is None
        client.delete_hub_model_file.assert_called_once_with("m1", "f1")


class TestLinkDesign:
    def test_a_project_sends_project_id_and_version(self) -> None:
        client = MagicMock(spec=DagnamClient)
        client.link_hub_model_design.return_value = {"id": "v1"}
        out = hub.link_design(
            "m1", "v1", project_id="p1", architecture_version_id="a1", client=client
        )
        assert out == {"id": "v1"}
        client.link_hub_model_design.assert_called_once_with(
            "m1", "v1", {"project_id": "p1", "architecture_version_id": "a1"}
        )

    def test_a_project_alone_sends_only_its_id(self) -> None:
        client = MagicMock(spec=DagnamClient)
        hub.link_design("m1", "v1", project_id="p1", client=client)
        client.link_hub_model_design.assert_called_once_with("m1", "v1", {"project_id": "p1"})

    def test_a_design_object_is_sent_as_architecture(self) -> None:
        client = MagicMock(spec=DagnamClient)
        design: JsonObject = {"diagram_state": {}, "architecture_config": {}}
        hub.link_design("m1", "v1", architecture=design, client=client)
        client.link_hub_model_design.assert_called_once_with("m1", "v1", {"architecture": design})

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"project_id": "p1", "architecture": {"diagram_state": {}}},
            {"architecture": {"diagram_state": {}}, "architecture_version_id": "a1"},
        ],
    )
    def test_a_bad_combination_is_refused_before_any_request(self, kwargs: dict[str, Any]) -> None:
        client = MagicMock(spec=DagnamClient)
        with pytest.raises(ValueError, match=r"project_id|architecture"):
            hub.link_design("m1", "v1", client=client, **kwargs)
        client.link_hub_model_design.assert_not_called()
