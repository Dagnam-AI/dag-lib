"""Installing and removing the Anthropic patch, and the clients that share the patched classes."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, List, Tuple

from _anthropic_bodies import PATH, message_body, text_events
from _anthropic_calls import ARGS, run
from _mock_server import Reply
import anthropic
from anthropic.lib import streaming
from anthropic.lib.bedrock import AnthropicBedrock, AsyncAnthropicBedrock
import anthropic.lib.bedrock._beta_messages as bedrock_beta
from anthropic.lib.vertex import AnthropicVertex, AsyncAnthropicVertex
import anthropic.lib.vertex._beta_messages as vertex_beta
import anthropic.resources.beta.messages as beta_messages
import anthropic.resources.messages as messages_module
import httpx

from dagnam_record import _anthropic

if TYPE_CHECKING:
    import pytest

VERTEX_CREDENTIAL = "vertex-must-not-be-recorded"
ABSENT = (
    "anthropic",
    "anthropic.resources",
    "anthropic.resources.messages",
    "anthropic.resources.beta",
    "anthropic.resources.beta.messages",
    "anthropic.lib",
    "anthropic.lib.streaming",
    "anthropic.lib.bedrock",
    "anthropic.lib.bedrock._beta_messages",
    "anthropic.lib.vertex",
    "anthropic.lib.vertex._beta_messages",
)


def patched_methods() -> List[Tuple[Any, str]]:
    """Every (class, method) the patch wraps."""
    classes = [
        messages_module.Messages,
        messages_module.AsyncMessages,
        beta_messages.Messages,
        beta_messages.AsyncMessages,
        bedrock_beta.Messages,
        bedrock_beta.AsyncMessages,
        vertex_beta.Messages,
        vertex_beta.AsyncMessages,
    ]
    found = [(cls, name) for cls in classes for name in ("create", "parse", "stream")]
    managers = ("MessageStreamManager", "BetaMessageStreamManager")
    found += [(getattr(streaming, name), "__enter__") for name in managers]
    found += [(getattr(streaming, f"Async{name}"), "__aenter__") for name in managers[:1]]
    found += [(streaming.BetaAsyncMessageStreamManager, "__aenter__")]
    return found


def is_patched(cls: Any, name: str) -> bool:
    return getattr(getattr(cls, name), "__dagnam_record__", False) is True


def test_every_known_method_is_patched_and_uninstall_puts_the_sdks_own_back(arig: Any) -> None:
    methods = patched_methods()
    assert all(is_patched(cls, name) for cls, name in methods)
    originals = [getattr(cls, name).__wrapped__ for cls, name in methods]
    _anthropic.uninstall()
    assert not any(is_patched(cls, name) for cls, name in methods)
    assert [getattr(cls, name) for cls, name in methods] == originals
    _anthropic.uninstall()  # a second uninstall is a no-op


def test_install_twice_wraps_once_and_a_second_recorder_replaces_the_first(
    arig: Any, make: Any, tmp_path: Any
) -> None:
    wrapper = messages_module.Messages.create
    other = make(tmp_path / "other")
    assert _anthropic.install(type(arig.recorder)(other)) is True
    assert messages_module.Messages.create is wrapper  # not wrapped a second time
    arig.server.on(PATH, Reply(body=message_body()))
    with arig.anthropic() as client:
        client.messages.create(**ARGS)
    assert arig.lines() == []
    other.drain()
    assert len(list((tmp_path / "other").glob("*.jsonl"))) == 1


def test_after_the_uninstall_calls_are_not_recorded(arig: Any) -> None:
    _anthropic.uninstall()
    arig.server.on(PATH, Reply(body=message_body()))
    with arig.anthropic() as client:
        client.messages.create(**ARGS)
    assert arig.lines() == []
    assert arig.writer.counts["dropped_error"] == 0


def test_wrappers_someone_kept_pass_straight_through_after_the_uninstall(arig: Any) -> None:
    kept_sync = messages_module.Messages.create
    kept_async = messages_module.AsyncMessages.create
    kept_stream = messages_module.Messages.stream
    kept_enter = streaming.MessageStreamManager.__enter__
    _anthropic.uninstall()
    arig.server.on(PATH, Reply(body=message_body()), Reply(body=message_body()))
    with arig.anthropic() as client:
        assert kept_sync(client.messages, **ARGS).stop_reason == "end_turn"
        manager = kept_stream(client.messages, **ARGS)
        arig.server.on(PATH, Reply(events=text_events()))
        with kept_enter(manager) as stream:
            assert stream.get_final_text() == "Hello"

    async def call(client: anthropic.AsyncAnthropic) -> Any:
        arig.server.on(PATH, Reply(body=message_body()))
        return await kept_async(client.messages, **ARGS)

    assert run(call, arig).stop_reason == "end_turn"
    assert arig.lines() == []
    assert arig.writer.counts["dropped_error"] == 0


def test_a_manager_made_before_the_uninstall_is_not_recorded_when_entered_after(
    arig: Any,
) -> None:
    arig.server.on(PATH, Reply(events=text_events()))
    with arig.anthropic() as client:
        manager = client.messages.stream(**ARGS)
        _anthropic.uninstall()
        with manager as stream:
            stream.get_final_message()
    assert arig.lines() == []
    assert arig.writer.counts["dropped_partial"] == 0


def test_the_kill_switch_patches_nothing(arig: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _anthropic.uninstall()
    monkeypatch.setenv("DAGNAM_RECORD", "0")
    assert _anthropic.install(arig.recorder) is False
    assert not any(is_patched(cls, name) for cls, name in patched_methods())


def test_an_absent_sdk_is_not_an_error(arig: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _anthropic.uninstall()
    for name in ABSENT:
        monkeypatch.setitem(sys.modules, name, None)
    assert _anthropic.install(arig.recorder) is False


def test_an_sdk_without_one_resource_still_patches_the_others(
    arig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _anthropic.uninstall()
    for name in ("anthropic.resources.beta.messages", "anthropic.lib.bedrock._beta_messages"):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delattr(vertex_beta, "AsyncMessages")
    monkeypatch.delattr(messages_module.Messages, "parse")
    monkeypatch.delattr(streaming, "BetaMessageStreamManager")
    assert _anthropic.install(arig.recorder) is True
    assert is_patched(messages_module.Messages, "create")
    assert is_patched(vertex_beta.Messages, "create")
    assert not hasattr(messages_module.Messages, "parse")
    assert not is_patched(beta_messages.Messages, "create")
    assert not is_patched(bedrock_beta.Messages, "create")
    assert is_patched(streaming.MessageStreamManager, "__enter__")
    _anthropic.uninstall()
    assert not is_patched(messages_module.Messages, "create")


def test_an_sdk_without_the_streaming_helpers_still_records_create(
    arig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _anthropic.uninstall()
    monkeypatch.setitem(sys.modules, "anthropic.lib.streaming", None)
    assert _anthropic.install(arig.recorder) is True
    assert not is_patched(streaming.MessageStreamManager, "__enter__")
    arig.server.on(PATH, Reply(body=message_body()))
    with arig.anthropic() as client:
        client.messages.create(**ARGS)
    assert len(arig.lines()) == 1


def test_an_sdk_with_no_known_resource_is_not_patched(
    arig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _anthropic.uninstall()
    for cls, name in patched_methods():
        if name != "__aenter__" and name != "__enter__":
            monkeypatch.delattr(cls, name)
    assert _anthropic.install(arig.recorder) is False
    assert not is_patched(streaming.MessageStreamManager, "__enter__")  # nothing to record with


def test_the_installed_sdk_classes_are_the_ones_every_client_family_uses(arig: Any) -> None:
    fake = "not-a-credential"
    clients = [
        anthropic.Anthropic(api_key=fake),
        AnthropicBedrock(aws_region="us-east-1", aws_access_key=fake, aws_secret_key=fake),
        AnthropicVertex(region="us-east5", project_id="p", access_token=fake),
    ]
    for client in clients:
        assert is_patched(type(client.messages), "create")
        assert is_patched(type(client.messages), "stream")
        assert is_patched(type(client.beta.messages), "create")
        assert is_patched(type(client.beta.messages), "stream")
    async_clients = [
        anthropic.AsyncAnthropic(api_key=fake),
        AsyncAnthropicBedrock(aws_region="us-east-1", aws_access_key=fake, aws_secret_key=fake),
        AsyncAnthropicVertex(region="us-east5", project_id="p", access_token=fake),
    ]
    for async_client in async_clients:
        assert is_patched(type(async_client.messages), "create")
        assert is_patched(type(async_client.beta.messages), "create")


def test_a_vertex_client_is_recorded_through_its_real_request_path(arig: Any) -> None:
    arig.server.on(
        "/projects/p/locations/us-east5/publishers/anthropic/models/m:rawPredict",
        Reply(body=message_body()),
    )
    client = AnthropicVertex(
        region="us-east5",
        project_id="p",
        access_token=VERTEX_CREDENTIAL,
        base_url=arig.server.root,
        http_client=httpx.Client(trust_env=False),
        max_retries=0,
    )
    with client:
        client.messages.create(
            model="m", max_tokens=8, messages=[{"role": "user", "content": "Hi"}]
        )
    (line,) = arig.lines()
    assert line["norm"]["response"] == "Hello there"
    assert VERTEX_CREDENTIAL not in arig.raw()[0]
    assert arig.server.seen[0].headers["authorization"] == "Bearer " + VERTEX_CREDENTIAL


def test_a_message_stream_without_the_expected_attribute_is_counted_and_unharmed(
    arig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A future SDK may build its message stream differently: the call must still work."""
    _anthropic.uninstall()

    class Manager:
        def __enter__(self) -> Any:
            return SimpleNamespace(opened=True)

        def __exit__(self, *exc: Any) -> None:
            return None

    monkeypatch.setattr(streaming, "MessageStreamManager", Manager)
    assert _anthropic.install(arig.recorder)
    manager = Manager()
    setattr(manager, _anthropic.CALL, ("messages", {"model": "m"}))
    with manager as stream:
        assert stream.opened is True
    assert not hasattr(stream, "_raw_stream")  # nothing was put in its place
    assert arig.writer.counts["dropped_error"] == 1
    assert arig.raw() == []


def test_a_manager_that_cannot_carry_the_request_is_counted_and_returned_as_it_is(
    arig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Slotted:
        __slots__ = ()

    _anthropic.uninstall()
    monkeypatch.setattr(messages_module.Messages, "stream", lambda self, **kwargs: Slotted())
    assert _anthropic.install(arig.recorder)
    with arig.anthropic() as client:
        assert isinstance(client.messages.stream(**ARGS), Slotted)
    assert arig.writer.counts["dropped_error"] == 1
    _anthropic.uninstall()  # before monkeypatch puts the SDK's function back, or it would be restored over it


def test_the_request_noted_on_a_manager_is_taken_before_the_sdk_sees_the_arguments(
    arig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def stream(self: Any, **kwargs: Any) -> Any:
        kwargs["messages"][0]["content"] = "changed by the SDK"
        return SimpleNamespace()

    _anthropic.uninstall()
    monkeypatch.setattr(messages_module.Messages, "stream", stream)
    assert _anthropic.install(arig.recorder)
    messages: Any = [{"role": "user", "content": "original"}]
    with arig.anthropic() as client:
        manager = client.messages.stream(model="m", max_tokens=8, messages=messages)
    _anthropic.uninstall()  # before monkeypatch puts the SDK's function back
    endpoint, request = getattr(manager, _anthropic.CALL)
    assert endpoint == "messages"
    assert request["messages"] == [{"role": "user", "content": "original"}]
