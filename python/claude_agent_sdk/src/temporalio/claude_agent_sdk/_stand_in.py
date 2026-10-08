"""A local stand-in for the Messages API, for tool steps.

A tool step runs one Claude Code tool call that paused a segment. Claude Code resumes a
copy of the conversation that ends before the call and asks the model for its next
message; the stand-in answers with the call itself: the exact ``tool_use`` block Claude
sent, with its id and input, as the conversation recorded it. Claude Code then runs the
call in an ordinary turn, and ``max_turns=1`` ends the turn after it (the bounded native
call replay of Brian Strauch's hybrid prototype, temporalio/ai-integrations). No real
model is asked, and the conversation never leaves the Worker for it.

Each step registers its call under a key of its own (``serve``): only the step's engine
gets that key, as its ``ANTHROPIC_API_KEY``. The call is sent once; any other model call
of that engine gets a short text, and is counted.
"""

from __future__ import annotations

import copy
import hmac
import json
import secrets
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar

ANSWER = "ok"
"""What the stand-in answers to every model call but the one that gets the step's call."""

MAX_BODY_BYTES = 1024**3
"""Largest request body read (the engine sends the whole conversation)."""


@dataclass
class RecordedCall:
    """A tool step's call, as the stand-in serves it.

    Attributes:
        key: The API key of the step's engine.
        block: The ``tool_use`` block to answer with.
        served: How many times the block was sent (one, when all went well).
        other: Model calls answered with ``ANSWER`` instead, after the block was sent
            or in a request that did not offer the call's tool.
    """

    key: str
    block: dict[str, Any]
    served: int = 0
    other: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


def _offers(body: dict[str, Any], name: str) -> bool:
    """Whether a model call offers the tool ``name`` (the engine's main loop does)."""
    tools = body.get("tools")
    return isinstance(tools, list) and any(
        isinstance(t, dict) and t.get("name") == name for t in tools
    )


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    stand_in: ClassVar[StandInModel]  # set by the server's own subclass

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

    def _refuse(self, status: int, kind: str, message: str) -> None:
        """Answer without reading the body, and close the connection."""
        self.close_connection = True
        self._send(
            {"type": "error", "error": {"type": kind, "message": message}}, status
        )

    def do_POST(self) -> None:
        call = self.stand_in.call_for(self.headers.get("x-api-key", ""))
        if call is None:
            return self._refuse(401, "authentication_error", "not a tool step's key")
        self.stand_in.counted()
        try:
            length = int(self.headers.get("content-length", 0))
        except ValueError:
            length = -1
        if length < 0:  # read() would wait for the end of the connection
            return self._refuse(400, "invalid_request_error", "bad content-length")
        if length > MAX_BODY_BYTES:
            return self._refuse(413, "request_too_large", "body too large")
        try:
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
        with call.lock:  # handlers run in threads
            first = call.served == 0 and _offers(body, str(call.block.get("name")))
            if first:
                call.served += 1
            else:
                call.other += 1
        if first:
            block = call.block
            content = [
                {
                    "type": "tool_use",
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "input": copy.deepcopy(block.get("input") or {}),
                }
            ]
        else:
            content = [{"type": "text", "text": ANSWER}]
        self._answer(body, content)

    def _answer(self, body: dict[str, Any], content: list[dict[str, Any]]) -> None:
        """The model's message: the step's call, or the short text."""
        stop = "tool_use" if content[0]["type"] == "tool_use" else "end_turn"
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
            return self._send({**message, "content": content, "stop_reason": stop})
        events: list[dict[str, Any]] = [{"type": "message_start", "message": message}]
        for index, block in enumerate(content):
            start: dict[str, Any]
            delta: dict[str, Any]
            if block["type"] == "tool_use":
                start = {**block, "input": {}}
                delta = {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(block["input"]),
                }
            else:
                start = {"type": "text", "text": ""}
                delta = {"type": "text_delta", "text": block["text"]}
            events += [
                {"type": "content_block_start", "index": index, "content_block": start},
                {"type": "content_block_delta", "index": index, "delta": delta},
                {"type": "content_block_stop", "index": index},
            ]
        events += [
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop, "stop_sequence": None},
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
    """A Messages API on 127.0.0.1 that answers a tool step's engine with its call.

    One per runner: it starts with the first tool step and serves until the Worker
    process ends. It answers model calls (POST) only when they carry the key of a step
    that is running: another program on the machine gets 401 before the body of its
    request is read.
    """

    def __init__(self) -> None:
        """Create it; it starts on first use."""
        self._server: ThreadingHTTPServer | None = None
        self._lock = threading.Lock()
        self._calls: dict[str, RecordedCall] = {}
        self.requests = 0
        """POST requests with the key of a running step (its engine's calls)."""

    def counted(self) -> None:
        """Count one request with a running step's key."""
        with self._lock:
            self.requests += 1

    def serve(self, block: dict[str, Any]) -> RecordedCall:
        """Answer the engine that sends the returned key with ``block``, once."""
        call = RecordedCall(key=secrets.token_hex(16), block=copy.deepcopy(block))
        with self._lock:
            self._calls[call.key] = call
        return call

    def done(self, call: RecordedCall) -> None:
        """Stop answering the step's engine (its key gets 401 from now on)."""
        with self._lock:
            self._calls.pop(call.key, None)

    def call_for(self, key: str) -> RecordedCall | None:
        """The running step that ``key`` belongs to, compared in constant time."""
        given = key.encode("latin-1", "replace")
        with self._lock:
            calls = list(self._calls.values())
        found = None
        for call in calls:
            if hmac.compare_digest(given, call.key.encode()):
                found = call
        return found

    @property
    def base_url(self) -> str:
        """The URL for ``ANTHROPIC_BASE_URL``; starts the server if needed."""
        with self._lock:
            if self._server is None:
                handler = type("StandInHandler", (_Handler,), {"stand_in": self})
                self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
                self._server.daemon_threads = True
                threading.Thread(
                    target=self._server.serve_forever,
                    name="claude-tool-step-model",
                    daemon=True,
                ).start()
            return f"http://127.0.0.1:{self._server.server_address[1]}"
