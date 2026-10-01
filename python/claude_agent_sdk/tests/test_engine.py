"""The real Claude Code engine (bundled in claude-agent-sdk) against a local fake Messages API.

These tests drive ClaudeAgentSdkRunner directly, segment by segment, the way the
Workflow does (holding the conversation, by default, or with a session store), so they
pin down the engine behavior the plugin relies on.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import DeferredToolUse, ResultMessage, SystemMessage

from temporalio.claude_agent_sdk import (
    ClaudeAgentSdkRunner,
    FileSessionStore,
    SegmentInput,
    SegmentOutput,
    ToolOutcome,
    ToolSpec,
    _runner,
)
from tests.helpers.fake_messages_api import (
    FakeMessagesAPI,
    engine_env,
    history_of,
    start_with_policy,
)
from tests.refund import shop
from tests.refund.policy import refund_policy

ORDER_SCHEMA = {
    "type": "object",
    "properties": {"order_id": {"type": "string"}},
    "required": ["order_id"],
}
TOOLS = [
    ToolSpec("look_up_order", "Look up an order.", ORDER_SCHEMA),
    ToolSpec("issue_refund", "Refund money.", {"type": "object"}),
    ToolSpec("email_customer", "Email the customer.", {"type": "object"}),
]


def _make_runner(
    api: FakeMessagesAPI, tmp_path: Path, mode: str = "held"
) -> ClaudeAgentSdkRunner:
    return ClaudeAgentSdkRunner(
        session_store=FileSessionStore(tmp_path / "store") if mode == "store" else None,
        cwd=str(tmp_path),
        env=engine_env(api, str(tmp_path / "cfg")),
    )


def _fake_tool(name: str, args: dict[str, Any]) -> ToolOutcome:
    if name == "look_up_order":
        return ToolOutcome(shop.ORDERS[args["order_id"]])
    if name == "issue_refund":
        return ToolOutcome({"refund_id": "R-000001", "amount": args["amount"]})
    return ToolOutcome({"sent": True})


async def _drive(
    runner: ClaudeAgentSdkRunner,
    prompt: str,
    tools: list[ToolSpec],
    limit: int = 8,
    added: list[int] | None = None,
) -> tuple[list[tuple[str, dict[str, Any]]], SegmentOutput]:
    """Run segments like the Workflow does: pause, run the tool, resume with its result.

    Without a session store, it holds the conversation like the Workflow, and records
    how many entries each segment added in ``added``.
    """
    held = runner._store is None  # type: ignore[reportPrivateUsage]
    transcript: list[dict[str, Any]] = []

    def hold(out: SegmentOutput) -> SegmentOutput:
        nonlocal transcript
        if held and not out.is_error:
            assert out.transcript_keep is not None
            transcript = transcript[: out.transcript_keep] + out.transcript_add
            if added is not None:
                added.append(len(out.transcript_add))
        return out

    seg = hold(
        await runner.run(
            SegmentInput(
                session_id=str(uuid.uuid4()),
                prompt=prompt,
                tools=tools,
                transcript=[] if held else None,
            ),
            1,
        )
    )
    paused: list[tuple[str, dict[str, Any]]] = []
    while seg.deferred is not None and len(paused) < limit:
        call = seg.deferred
        paused.append((call.name, call.input))
        seg = hold(
            await runner.run(
                SegmentInput(
                    session_id=seg.session_id,
                    prompt=None,
                    tools=tools,
                    checkpoint=seg.checkpoint,
                    injected={call.id: _fake_tool(call.name, call.input)},
                    segment_index=len(paused),
                    transcript=list(transcript) if held else None,
                ),
                1,
            )
        )
    return paused, seg


@pytest.mark.parametrize("mode", ["held", "store"])
async def test_engine_pauses_at_every_durable_call_and_resumes(
    tmp_path: Path, mode: str
) -> None:
    api = start_with_policy(refund_policy)
    try:
        runner = _make_runner(api, tmp_path, mode)
        paused, final = await _drive(
            runner, "Order A-1001 arrived broken, I want my money back.", TOOLS
        )
    finally:
        api.stop()
    assert [name for name, _ in paused] == [
        "look_up_order",
        "issue_refund",
        "email_customer",
    ]
    assert not final.is_error, final.error
    assert final.result is not None and final.result.startswith(
        "Done. Refunded 49.99 EUR"
    )
    assert api.errors == []  # every request followed the real API's tool rules
    assert runner.stub_calls == 0  # the engine never ran a durable tool itself


@pytest.mark.parametrize("mode", ["held", "store"])
async def test_parallel_calls_are_serialized_not_lost(
    tmp_path: Path, mode: str
) -> None:
    """Claude asks for two tools in one message: one pauses, the other is told to retry."""
    wanted = ("A-1001", "A-1002")
    ref: list[FakeMessagesAPI] = []

    def both_at_once(body: dict[str, Any]) -> list[dict[str, Any]]:
        uses, _, history = history_of(body)
        if not uses:
            return [ref[0].tool_use("look_up_order", {"order_id": o}) for o in wanted]
        done = {
            h.content["order_id"]
            for h in history
            if not h.is_error
            and isinstance(h.content, dict)
            and "order_id" in h.content
        }
        missing = [o for o in wanted if o not in done]
        if missing:
            return [ref[0].tool_use("look_up_order", {"order_id": missing[0]})]
        return [{"type": "text", "text": "FINAL: looked up " + ", ".join(wanted)}]

    api = FakeMessagesAPI(both_at_once)
    ref.append(api)
    api.start()
    try:
        runner = _make_runner(api, tmp_path, mode)
        paused, final = await _drive(
            runner, "Check orders A-1001 and A-1002.", TOOLS[:1]
        )
    finally:
        api.stop()
    assert sorted(args["order_id"] for _, args in paused) == ["A-1001", "A-1002"]
    assert final.result == "FINAL: looked up A-1001, A-1002"
    assert api.errors == []
    assert runner.stub_calls == 0


@pytest.mark.parametrize(
    ("reported", "parsed"),
    [
        ("2.1.259 (Claude Code)", (2, 1, 259)),
        ("2.1.273", (2, 1, 273)),
        ("(unknown version)", None),
        ("", None),
    ],
)
def test_engine_versions_are_parsed(
    reported: str, parsed: tuple[int, ...] | None
) -> None:
    assert _runner._version(reported) == parsed


async def test_engine_older_than_the_minimum_is_refused_before_it_starts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Claude Code 2.1.259 drops a tool result when Claude calls two tools at once (tested)."""

    async def old_version(self: ClaudeAgentSdkRunner) -> str:
        del self
        return "2.1.259 (Claude Code)"

    def must_not_start(**kwargs: Any) -> Any:
        del kwargs
        raise AssertionError("the engine must not start")

    monkeypatch.setattr(ClaudeAgentSdkRunner, "_engine_version", old_version)
    monkeypatch.setattr(_runner, "query", must_not_start)
    runner = ClaudeAgentSdkRunner(
        session_store=FileSessionStore(tmp_path),
        env={"ANTHROPIC_API_KEY": "sk-ant-fake-not-real"},
    )
    out = await runner.run(
        SegmentInput(
            session_id=str(uuid.uuid4()), prompt="Refund A-1001.", tools=TOOLS
        ),
        1,
    )
    assert out.is_error and out.deferred is None and out.result is None
    assert out.error is not None and "older than 2.1.273" in out.error


async def test_old_engine_found_at_start_never_hands_back_a_tool_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If the version could not be read up front, the engine's own report still stops it."""

    async def unknown_version(self: ClaudeAgentSdkRunner) -> None:
        del self

    async def old_engine(**kwargs: Any) -> AsyncIterator[Any]:
        del kwargs
        yield SystemMessage(subtype="init", data={"claude_code_version": "2.1.259"})
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="s",
            stop_reason="tool_deferred",
            deferred_tool_use=DeferredToolUse(
                id="toolu_1", name="mcp__durable__issue_refund", input={}
            ),
        )

    monkeypatch.setattr(ClaudeAgentSdkRunner, "_engine_version", unknown_version)
    monkeypatch.setattr(_runner, "query", old_engine)
    runner = ClaudeAgentSdkRunner(
        session_store=FileSessionStore(tmp_path),
        env={"ANTHROPIC_API_KEY": "sk-ant-fake-not-real"},
    )
    out = await runner.run(
        SegmentInput(
            session_id=str(uuid.uuid4()), prompt="Refund A-1001.", tools=TOOLS
        ),
        1,
    )
    assert out.is_error and out.deferred is None  # the refund is never handed back
    assert out.error is not None and "older than 2.1.273" in out.error


@pytest.mark.parametrize(
    ("max_turns", "budget", "stop"),
    [(None, None, None), (1, None, "turns"), (None, 0.0001, "budget")],
    ids=["runs-inside", "max-turns", "max-budget"],
)
async def test_builtin_tools_run_inside_the_segment(
    tmp_path: Path, max_turns: int | None, budget: float | None, stop: str | None
) -> None:
    """A built-in tool runs in the engine; a turn or budget cap ends the segment for good."""
    ref: list[FakeMessagesAPI] = []

    def glob_then_answer(body: dict[str, Any]) -> list[dict[str, Any]]:
        answered = any(
            block.get("type") == "tool_result"
            for message in body.get("messages", [])
            if isinstance(message.get("content"), list)
            for block in message["content"]
        )
        if answered:
            return [{"type": "text", "text": "FINAL: looked around"}]
        return [
            {
                "type": "tool_use",
                "id": ref[0].next_id("toolu_glob"),
                "name": "Glob",
                "input": {"pattern": "*.txt"},
            }
        ]

    api = FakeMessagesAPI(glob_then_answer)
    ref.append(api)
    api.start()
    try:
        runner = _make_runner(api, tmp_path)
        runner._max_budget = budget  # type: ignore[reportPrivateUsage]
        out = await runner.run(
            SegmentInput(
                session_id=str(uuid.uuid4()),
                prompt="Look around.",
                tools=TOOLS,
                builtin_tools=["Glob"],
                max_turns=max_turns,
            ),
            1,
        )
    finally:
        api.stop()
    assert out.deferred is None
    if stop is None:
        assert out.result == "FINAL: looked around" and out.checkpoint is not None
    else:
        assert out.is_error and out.error is not None and stop in out.error
    assert api.errors == [] and runner.stub_calls == 0
