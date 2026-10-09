"""A deterministic fake TypeSafe HTTP transport for the typesafe SDK.

An ``httpx2.MockTransport`` adapter answers ``POST /v1/systemone`` in
process, exercising the SDK's real request/response path with no credentials.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx2
from typesafe_sdk import (
    AsyncTypeSafeClient,
    NoulAnswer,
    RetryPolicy,
    SystemOneResponse,
)

MODEL = "jev-1.13.0"


class BillingResponse(SystemOneResponse):
    billing: NoulAnswer


def fake_response(
    status: int = 200,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> httpx2.Response:
    """Build a raw response for the given status, body, and headers."""
    return httpx2.Response(
        status_code=status,
        json=body if body is not None else {},
        headers=headers or {},
        request=httpx2.Request("POST", "https://fake.typesafe.ai/v1/systemone"),
    )


def answer_for(question: dict[str, Any]) -> dict[str, Any]:
    """Synthesize the well-typed answer for one question shape."""
    kind = question.get("type")
    if kind == "choice":
        criteria = question["criteria"]
        return {
            "type": "choice",
            "choice": next(iter(criteria)),
            "confidence": 0.9,
            "probabilities": {name: 1.0 / len(criteria) for name in criteria},
        }
    if kind == "score":
        levels = question["criteria"]
        return {
            "type": "score",
            "score": (len(levels) - 1) / 2,
            "confidence": 0.85,
            "probabilities": {str(i): 1.0 / len(levels) for i in range(len(levels))},
            "legend": {str(i): description for i, description in enumerate(levels)},
        }
    if kind == "noul":
        return {"type": "noul", "noul": 0.8}
    raise ValueError(f"unrecognized question type {kind!r}")


def fake_typesafe_responder(
    handler: Callable[[dict[str, Any]], httpx2.Response],
    *,
    model: str | None = None,
    retry: RetryPolicy | None = None,
) -> AsyncTypeSafeClient:
    """A client whose requests are answered by ``handler``.

    Args:
        handler: Maps the decoded request body to a response.
        model: A default model the call-time ``model=None`` falls back to,
            set only when a test pins one.
        retry: An SDK retry policy, set only when a test pins one.
    """

    def transport(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content) if request.content else {}
        return handler(body)

    return AsyncTypeSafeClient(
        api_key="fake",
        transport=httpx2.MockTransport(transport),
        model=model,
        retry=retry,
    )


def mock_client(
    status: int = 200,
    headers: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
    *,
    model: str | None = None,
    retry: RetryPolicy | None = None,
) -> AsyncTypeSafeClient:
    """A client that answers every question with a canned well-typed body.

    Args:
        status: HTTP status every request answers with; ``200`` yields
            answers for each question.
        headers: Extra headers on every response.
        extra: Fields merged into every answer the fake returns.
        model: A default model the call-time ``model=None`` falls back to,
            set only when a test pins one.
        retry: An SDK retry policy, set only when a test pins one.
    """

    def handler(body: dict[str, Any]) -> httpx2.Response:
        if status != 200:
            return fake_response(status, {"message": "boom"}, headers)
        answers = {
            name: answer_for(question) | (extra or {})
            for name, question in body["questions"].items()
        }
        return fake_response(
            200,
            {
                "model": MODEL,
                "answers": answers,
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
            headers,
        )

    return fake_typesafe_responder(handler, model=model, retry=retry)
