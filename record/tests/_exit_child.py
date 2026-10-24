"""A child interpreter for test_stream_exit: record one scenario against the mock server, then exit.

Usage: ``python _exit_child.py <scenario> <trace directory>``. Garbage collection is switched off
so that a stream in a reference cycle is still there when the interpreter exits, and nothing
closes the writer by hand: the exit hooks have to do all of it.
"""

from __future__ import annotations

import gc
import os
import queue
import sys
from typing import Any, Optional

from _anthropic_bodies import PATH, text_events
from _mock_server import MockServer, Reply
from _openai_bodies import CHAT_PATH, MODEL, chat_text_events
import anthropic
import httpx
import openai
from typing_extensions import override

from dagnam_record import _anthropic, _openai
from dagnam_record._recorder import Recorder
from dagnam_record._writer import Writer

ARGS: Any = {
    "model": "claude-sonnet-4-5",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "Hi"}],
}
HELD: Any = []  # module globals live until the interpreter is torn down


class CollectingQueue(queue.Queue[Optional[str]]):
    """A queue that runs a garbage collection whenever a line is put on it."""

    @override
    def put_nowait(self, item: Optional[str]) -> None:
        gc.collect()  # submit() holds the writer's gate here
        super().put_nowait(item)


class Hooked(Writer):
    """A writer, not started, whose ``submit`` runs a collection inside its gate."""

    def hit(self) -> None:
        self._queue = CollectingQueue(maxsize=10)
        self.submit("{}")
        self.close()


def anthropic_client(server: MockServer) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key="k",
        base_url=server.root,
        max_retries=0,
        http_client=httpx.Client(trust_env=False),
    )


def openai_client(server: MockServer) -> openai.OpenAI:
    return openai.OpenAI(
        api_key="k",
        base_url=server.url,
        max_retries=0,
        http_client=httpx.Client(trust_env=False),
    )


def read_until(stream: Any, kind: str) -> None:
    for event in stream:
        if event.type == kind:
            break


def anthropic_local(client: anthropic.Anthropic, kind: str) -> None:
    stream = client.messages.create(stream=True, **ARGS)
    read_until(stream, kind)  # dropped here, in a reference cycle


def main(scenario: str, directory: str) -> None:
    gc.disable()
    server = MockServer()
    server.on(PATH, *[Reply(events=text_events("Hel", "lo")) for _ in range(4)])
    server.on(CHAT_PATH, *[Reply(events=chat_text_events()) for _ in range(4)])
    writer = Writer(directory)
    assert _anthropic.install(Recorder(writer))
    assert _openai.install(Recorder(writer))
    client = anthropic_client(server)
    if scenario == "anthropic_local":
        anthropic_local(client, "message_stop")
    elif scenario == "anthropic_global":
        HELD.append(client.messages.create(stream=True, **ARGS))
        read_until(HELD[0], "message_stop")
    elif scenario == "anthropic_unfinished":
        anthropic_local(client, "content_block_stop")
        HELD.append(client.messages.create(stream=True, **ARGS))
        read_until(HELD[0], "content_block_start")
    elif scenario == "anthropic_helper":
        manager = client.messages.stream(**ARGS)
        HELD.append(manager)
        stream = manager.__enter__()  # never exited, as on an SDK whose exit skips the raw stream
        HELD.append(stream)
        read_until(stream, "message_stop")
    elif scenario == "anthropic_helper_unfinished":
        manager = client.messages.stream(**ARGS)
        HELD.append(manager)
        HELD.append(manager.__enter__())
        next(iter(HELD[1]))
    elif scenario == "openai_local":
        openai_local(openai_client(server))
    elif scenario == "openai_global":
        HELD.append(openai_open(openai_client(server)))
        through_stop(HELD[0])
    elif scenario == "openai_unfinished":
        stream = openai_open(openai_client(server))
        next(iter(stream))
    elif scenario == "fork_child_drops_inherited":
        HELD.append(client.messages.create(stream=True, **ARGS))
        read_until(HELD[0], "message_stop")  # finished, never closed, held by the parent
        pid = os.fork()
        if pid == 0:
            HELD.clear()  # the child lets go of the parent's stream ...
            gc.collect()
            anthropic_local(anthropic_client(server), "message_stop")  # ... and makes its own call
            sys.exit(0)
        os.waitpid(pid, 0)
    elif scenario == "gc_in_submit":
        anthropic_local(client, "message_stop")
        Hooked(
            directory + "-hooked", start=False
        ).hit()  # a collection inside submit(), under its lock
    elif scenario == "late_writer":
        anthropic_local(client, "message_stop")  # the exit hook is registered by now
        later = Writer(directory + "-later")
        assert _anthropic.install(
            Recorder(later)
        )  # a writer made after the hook: install re-orders
        anthropic_local(client, "message_stop")
    else:
        raise SystemExit("unknown scenario " + scenario)
    sys.stdout.write("done\n")


def openai_open(client: openai.OpenAI) -> Any:
    return client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
        stream_options={"include_usage": True},
    )


def through_stop(stream: Any) -> None:
    iterator = iter(stream)
    for _ in range(4):
        next(iterator)


def openai_local(client: openai.OpenAI) -> None:
    through_stop(openai_open(client))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
