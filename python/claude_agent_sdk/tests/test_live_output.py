"""Live output: the agent's events through Workflow Streams, in order, across Continue-As-New."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import pytest

from temporalio.claude_agent_sdk import (
    ClaudeAgentPlugin,
    ClaudeAgentSdkRunner,
    FileSessionStore,
    SegmentRunner,
    follow_agent,
)
from temporalio.claude_agent_sdk.testing import ScriptedClaude
from temporalio.client import Client
from temporalio.worker import Worker
from tests.endless.activities import ALL
from tests.endless.policy import count_policy
from tests.endless.workflows import LongTaskWorkflow, TaskOptions
from tests.helpers.fake_messages_api import engine_env, start_with_policy

pytestmark = pytest.mark.timeout(180)


async def read_until_done(
    client: Client, workflow_id: str, from_offset: int = 0
) -> list[dict[str, Any]]:
    """Every event from ``from_offset`` through ``done`` (or the end of the stream)."""
    events: list[dict[str, Any]] = []
    async for event in follow_agent(client, workflow_id, from_offset=from_offset):
        events.append(event)
        if event["type"] in ("done", "error"):
            break
    return events


def worker(client: Client, queue: str, runner: SegmentRunner) -> Worker:
    """A Worker for the live agents."""
    return Worker(
        client,
        task_queue=queue,
        workflows=[LongTaskWorkflow],
        activities=ALL,
        plugins=[ClaudeAgentPlugin(runner, heartbeat_every=1.0)],
    )


def check_sequence(events: list[dict[str, Any]], steps: int) -> None:
    """The events tell the whole story, once each, in order."""
    offsets = [e["offset"] for e in events]
    assert offsets == list(
        range(offsets[0], offsets[0] + len(events))
    )  # no gap, no duplicate
    kinds = [e["type"] for e in events]
    assert kinds[0] == "prompt" and kinds[-1] == "done"
    calls = [e for e in events if e["type"] == "tool_call"]
    results = [e for e in events if e["type"] == "tool_result"]
    assert [c["input"]["n"] for c in calls] == list(range(1, steps + 1))
    assert [r["status"] for r in results] == ["done"] * steps
    for call in calls:  # each call's result comes after it
        assert kinds.index("tool_result", events.index(call)) > events.index(call)
    assert {"type": "text", "text": f"counted to {steps}"}.items() <= next(
        e for e in events if e["type"] == "text"
    ).items()
    assert events[-1]["result"] == f"counted to {steps}"


@pytest.mark.usefixtures("shop_dir")
async def test_live_events_arrive_in_order_across_continue_as_new(
    client: Client,
) -> None:
    queue = f"live-{uuid.uuid4().hex[:8]}"
    async with worker(client, queue, ScriptedClaude(count_policy)):
        handle = await client.start_workflow(
            LongTaskWorkflow.run,
            args=["count to 12", TaskOptions(after_events=60, live=True), None],
            id=queue,
            task_queue=queue,
        )
        events = await asyncio.wait_for(read_until_done(client, queue), 120)
        assert await handle.result() == "counted to 12"
    check_sequence(events, 12)
    assert any(
        e["type"] == "continued_as_new" for e in events
    )  # it followed the new runs


@pytest.mark.usefixtures("shop_dir")
async def test_a_subscriber_that_reconnects_misses_nothing(client: Client) -> None:
    queue = f"reconnect-{uuid.uuid4().hex[:8]}"
    async with worker(
        client,
        queue,
        ScriptedClaude(count_policy, think_seconds=0.05),
    ):
        await client.start_workflow(
            LongTaskWorkflow.run,
            args=["count to 10", TaskOptions(after_events=60, live=True), None],
            id=queue,
            task_queue=queue,
        )
        first: list[dict[str, Any]] = []
        async for event in follow_agent(client, queue):  # disconnect after 7 events
            first.append(event)
            if len(first) == 7:
                break
        rest = await asyncio.wait_for(
            read_until_done(client, queue, first[-1]["offset"] + 1), 120
        )
    check_sequence(first + rest, 10)


@pytest.mark.timeout(240)
@pytest.mark.usefixtures("shop_dir")
async def test_real_engine_text_streams_live(client: Client, tmp_path: Path) -> None:
    api = start_with_policy(count_policy)
    (tmp_path / "work").mkdir()
    runner = ClaudeAgentSdkRunner(
        session_store=FileSessionStore(tmp_path / "sessions"),
        cwd=str(tmp_path / "work"),
        env=engine_env(api, str(tmp_path / "cfg")),
    )
    queue = f"live-real-{uuid.uuid4().hex[:8]}"
    try:
        async with worker(client, queue, runner):
            await client.start_workflow(
                LongTaskWorkflow.run,
                args=["count to 3", TaskOptions(after_events=40, live=True), None],
                id=queue,
                task_queue=queue,
            )
            events = await asyncio.wait_for(read_until_done(client, queue), 200)
    finally:
        api.stop()
    check_sequence(
        events, 3
    )  # the text event came from the engine, through the Activity
    assert api.errors == [] and runner.stub_calls == 0
