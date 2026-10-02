"""A local stand-in for the Messages API, for tool steps.

A tool step resumes Claude Code at a deferred call so that it runs exactly that tool.
After the tool, the engine asks the model to continue (twice, tested on Claude Code
2.1.273 and 2.1.287, even with ``max_turns=1``). Those requests go here instead of to
the real model: they cost nothing, and the conversation never leaves the Worker for
them. Every request gets the same short answer.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

ANSWER = "ok"
"""What the stand-in model answers."""


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def _send(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        self._send({})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("content-length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("not an object")
        except ValueError:
            error = {"type": "invalid_request_error", "message": "not a JSON object"}
            return self._send({"type": "error", "error": error}, status=400)
        if "count_tokens" in self.path:
            return self._send({"input_tokens": 1})
        if not self.path.startswith("/v1/messages"):
            return self._send({})
        message = {
            "id": "msg_tool_step",
            "type": "message",
            "role": "assistant",
            "model": body.get("model", "stand-in"),
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        if not body.get("stream"):
            text = [{"type": "text", "text": ANSWER}]
            return self._send({**message, "content": text, "stop_reason": "end_turn"})
        events: list[dict[str, Any]] = [
            {"type": "message_start", "message": message},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": ANSWER},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
            {"type": "message_stop"},
        ]
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("connection", "close")
        self.end_headers()
        for event in events:
            self.wfile.write(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
            )
        self.wfile.flush()
        self.close_connection = True


class StandInModel:
    """A Messages API on 127.0.0.1 that answers every request with a short text.

    One per runner: it starts with the first tool step and serves until the Worker
    process ends.
    """

    def __init__(self) -> None:
        """Create it; it starts on first use."""
        self._server: ThreadingHTTPServer | None = None
        self._lock = threading.Lock()
        self._count_lock = threading.Lock()
        self.requests = 0
        """POST requests answered (the engine's model calls in tool steps)."""

    @property
    def base_url(self) -> str:
        """The URL for ``ANTHROPIC_BASE_URL``; starts the server if needed."""
        with self._lock:
            if self._server is None:
                stand_in = self

                class Counting(_Handler):
                    def do_POST(self) -> None:
                        with stand_in._count_lock:  # handlers run in threads
                            stand_in.requests += 1
                        super().do_POST()

                self._server = ThreadingHTTPServer(("127.0.0.1", 0), Counting)
                self._server.daemon_threads = True
                threading.Thread(
                    target=self._server.serve_forever,
                    name="claude-tool-step-model",
                    daemon=True,
                ).start()
            return f"http://127.0.0.1:{self._server.server_address[1]}"
