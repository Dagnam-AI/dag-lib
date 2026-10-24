"""A local HTTP server that plays canned OpenAI-shaped replies (JSON or server-sent events).

The recorder tests drive the real ``openai`` client against it, so the patch is checked against
the SDK's own request building, response parsing and streaming, never against hand-built objects.
It lives in the tests only: the recorder package opens no socket.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Any, Dict, List, Optional, Tuple, cast

from typing_extensions import override

Event = Tuple[Optional[str], Any]
"""One server-sent event: its ``event:`` name (or ``None``) and its data (``"[DONE]"`` or JSON)."""


class Reply:
    """What the server answers: a JSON ``body`` or a list of ``events``, with a status."""

    def __init__(
        self,
        status: int = 200,
        body: Any = None,
        events: Optional[List[Event]] = None,
        cut: bool = False,
    ) -> None:
        self.status = status
        self.body = body
        self.events = events
        self.cut = cut  # promise more bytes than are sent, then hang up (a dropped connection)

    def payload(self) -> Tuple[str, bytes]:
        """The content type and the bytes of the body."""
        if self.events is None:
            return "application/json", json.dumps(self.body).encode("utf-8")
        text = ""
        for name, data in self.events:
            text += f"event: {name}\n" if name else ""
            text += "data: " + (data if isinstance(data, str) else json.dumps(data)) + "\n\n"
        return "text/event-stream", text.encode("utf-8")


class Seen:
    """One request the server received: the path, the JSON body and the headers."""

    def __init__(self, path: str, body: Any, headers: Dict[str, str]) -> None:
        self.path = path
        self.body = body
        self.headers = headers


class _Handler(BaseHTTPRequestHandler):
    @override
    def log_message(self, format: str, *args: Any) -> None:
        """Stay quiet."""

    def do_POST(self) -> None:
        """Record the request and play the next reply scripted for its path."""
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        body = json.loads(raw) if raw else None
        state = cast("_Server", self.server)
        path = self.path.split("?")[0]
        state.seen.append(Seen(path, body, {k.lower(): v for k, v in self.headers.items()}))
        reply = state.next_reply(path)
        kind, data = reply.payload()
        self.send_response(reply.status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data) + (1000 if reply.cut else 0)))
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()
        self.close_connection = True


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.seen: List[Seen] = []
        self.script: Dict[str, List[Reply]] = {}
        self.lock = threading.Lock()

    def next_reply(self, path: str) -> Reply:
        with self.lock:
            queued = self.script.get(path) or [Reply(404, {"error": {"message": "no reply"}})]
            return queued.pop(0) if len(queued) > 1 else queued[0]


class MockServer:
    """Start with ``MockServer()``; ``on(path, *replies)`` scripts a path (the last one repeats)."""

    def __init__(self) -> None:
        self._server = _Server()
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        self._thread.start()

    @property
    def url(self) -> str:
        """The OpenAI-style base URL (ends in ``/v1``)."""
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    @property
    def seen(self) -> List[Seen]:
        """Every request received so far, in order."""
        return self._server.seen

    def on(self, path: str, *replies: Reply) -> None:
        """Script the replies for a path such as ``/v1/chat/completions``."""
        with self._server.lock:
            self._server.script[path] = list(replies)

    def close(self) -> None:
        """Stop serving and wait (bounded) for the thread."""
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(5)
