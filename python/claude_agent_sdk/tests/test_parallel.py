"""Several tool calls in one message: they all run at once, and Claude gets every result.

The engine keeps one paused call per run, so the first durable call of a message
pauses it and the calls after it are denied. The segment reports those calls; the
Workflow runs them all, each its own Activity (with its own approval), and the next
segment puts every result where Claude reads it.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from temporalio.claude_agent_sdk import (
    ClaudeAgentPlugin,
    ClaudeAgentSdkRunner,
    FileSessionStore,
    follow_agent,
)
from temporalio.claude_agent_sdk.testing import ScriptedClaude
from temporalio.client import Client, WorkflowHandle
from temporalio.worker import Worker
from tests.endless.activities import ALL as COUNTING
from tests.endless.workflows import LongTaskWorkflow, TaskOptions
from tests.helpers.fake_messages_api import engine_env, start_with_policy
from tests.parallel.policy import parallel_policy
from tests.parallel.workflows import ParallelWorkflow
from tests.refund import shop
from tests.test_crash import wait_until

pytestmark = pytest.mark.timeout(240)


def worker(client: Client, queue: str, runner: Any) -> Worker:
    """A Worker for the agents these tests use."""
    return Worker(
        client,
        task_queue=queue,
        workflows=[ParallelWorkflow, LongTaskWorkflow],
        activities=COUNTING,
        plugins=[ClaudeAgentPlugin(runner, heartbeat_every=1.0)],
    )


def counted() -> list[str]:
    """The step numbers ``count`` really executed, in order."""
    return [e["detail"] for e in shop.executions("count")]


async def history_of_steps(
    handle: WorkflowHandle[Any, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Every segment's input, and the order of the tool Activities' events."""
    inputs: list[dict[str, Any]] = []
    order: list[str] = []
    kinds: dict[int, str] = {}
    async for event in handle.fetch_history_events():
        if event.HasField("activity_task_scheduled_event_attributes"):
            sched = event.activity_task_scheduled_event_attributes
            kinds[event.event_id] = sched.activity_type.name
            if sched.activity_type.name == "run_claude_segment":
                inputs.append(json.loads(sched.input.payloads[0].data))
            else:
                order.append(f"start {sched.activity_type.name}")
        if event.HasField("activity_task_completed_event_attributes"):
            done = event.activity_task_completed_event_attributes
            kind = kinds.get(done.scheduled_event_id)
            if kind is not None and kind != "run_claude_segment":
                order.append(f"end {kind}")
    return inputs, order


@pytest.mark.usefixtures("shop_dir")
@pytest.mark.parametrize("mode", ["held", "store"])
async def test_calls_in_one_message_run_at_once(
    client: Client, tmp_path: Path, mode: str
) -> None:
    queue = f"par-{uuid.uuid4().hex[:8]}"
    folder = tmp_path / "fake" if mode == "store" else None
    async with worker(client, queue, ScriptedClaude(parallel_policy, folder)):
        handle = await client.start_workflow(
            ParallelWorkflow.run,
            args=[["parallel 3"], None],
            id=queue,
            task_queue=queue,
        )
        events: list[dict[str, Any]] = []
        async for event in follow_agent(client, queue):
            events.append(event)
            if event["type"] in ("done", "error"):
                break
        answers = await asyncio.wait_for(handle.result(), 60)
        inputs, order = await history_of_steps(handle)
    assert answers == ["counted 3 at once"]
    assert sorted(counted()) == ["1", "2", "3"]  # each call ran once
    assert order[:3] == ["start count"] * 3  # all three started before any ended
    assert len(inputs) == 2  # the calls, then the answer
    assert len(inputs[1]["injected"]) == 3  # every result went back at once
    kinds = [e["type"] for e in events if e["type"] in ("tool_call", "tool_result")]
    assert kinds == ["tool_call"] * 3 + ["tool_result"] * 3


@pytest.mark.usefixtures("shop_dir")
async def test_each_call_waits_only_for_its_own_approval(client: Client) -> None:
    """The calls that need no approval finish while the one that needs it waits."""
    queue = f"parappr-{uuid.uuid4().hex[:8]}"
    async with worker(client, queue, ScriptedClaude(parallel_policy)):
        handle = await client.start_workflow(
            ParallelWorkflow.run,
            args=[["parallel 2 and publish"], None],
            id=queue,
            task_queue=queue,
        )
        pending: list[dict[str, Any]] = []

        async def approval_requested() -> None:
            while not pending:
                pending.extend(await handle.query(ParallelWorkflow.pending_approvals))
                await asyncio.sleep(0.1)

        await asyncio.wait_for(approval_requested(), 30)
        await wait_until(lambda: len(counted()) == 2)  # the others ran meanwhile
        assert pending[0]["name"] == "publish"
        assert shop.executions("publish") == []
        await handle.execute_update(
            ParallelWorkflow.review, args=[pending[0]["id"], True]
        )
        answers = await asyncio.wait_for(handle.result(), 60)
        calls = await handle.query(ParallelWorkflow.tool_calls)
    assert answers == ["counted 2 at once"]
    assert len(shop.executions("publish")) == 1
    assert [(c["name"], c["status"]) for c in calls] == [
        ("count", "done"),
        ("publish", "done"),
        ("count", "done"),
    ]


@pytest.mark.usefixtures("shop_dir")
async def test_a_stopped_task_answers_every_call_it_left_waiting(
    client: Client,
) -> None:
    """``max_segments`` stops a task before its calls run; the next task tells Claude
    that none of them ran (the scripted runner fails a step that leaves one out)."""
    queue = f"parstop-{uuid.uuid4().hex[:8]}"
    async with worker(client, queue, ScriptedClaude(parallel_policy)):
        handle = await client.start_workflow(
            ParallelWorkflow.run,
            args=[["parallel 2", "parallel 0"], 1],
            id=queue,
            task_queue=queue,
        )
        answers = await asyncio.wait_for(handle.result(), 60)
        inputs, _ = await history_of_steps(handle)
    assert answers[0].startswith(
        "error: ApplicationError: Stopped after 1 segments, before running tool calls "
    )
    assert answers[1] == "counted 0 at once"
    assert counted() == []  # nothing ran
    delivered = inputs[1]["injected"]
    assert len(delivered) == 2 and all(r["is_error"] for r in delivered.values())
    assert all("did not run" in r["content"] for r in delivered.values())


@pytest.mark.usefixtures("shop_dir")
async def test_continue_as_new_right_after_calls_in_one_message(client: Client) -> None:
    """Every result of the message waits in the state and reaches Claude in the new run."""
    queue = f"parcan-{uuid.uuid4().hex[:8]}"
    async with worker(client, queue, ScriptedClaude(parallel_policy)):
        handle = await client.start_workflow(
            LongTaskWorkflow.run,
            args=["parallel 3", TaskOptions(after_events=12), None],
            id=queue,
            task_queue=queue,
        )
        result = await asyncio.wait_for(handle.result(), 60)
        progress = await handle.query(LongTaskWorkflow.progress)
    assert result == "counted 3 at once"
    assert progress == {"runs": 2, "tool_calls": 3}
    assert sorted(counted()) == ["1", "2", "3"]


@pytest.mark.timeout(300)
@pytest.mark.usefixtures("shop_dir")
@pytest.mark.parametrize("mode", ["held", "store"])
async def test_real_engine_runs_calls_in_one_message_at_once(
    client: Client, tmp_path: Path, mode: str
) -> None:
    """Three calls in one message on the real engine, and Continue-As-New while their
    results wait: each runs once, and Claude reads all three."""
    api = start_with_policy(parallel_policy)
    (tmp_path / "work").mkdir()
    runner = ClaudeAgentSdkRunner(
        session_store=(
            FileSessionStore(tmp_path / "sessions") if mode == "store" else None
        ),
        cwd=str(tmp_path / "work"),
        env=engine_env(api, str(tmp_path / "cfg")),
    )
    queue = f"parreal-{uuid.uuid4().hex[:8]}"
    try:
        async with worker(client, queue, runner):
            handle = await client.start_workflow(
                LongTaskWorkflow.run,
                args=["parallel 3", TaskOptions(after_events=12), None],
                id=queue,
                task_queue=queue,
            )
            result = await asyncio.wait_for(handle.result(), 240)
            progress = await handle.query(LongTaskWorkflow.progress)
    finally:
        api.stop()
    assert result == "counted 3 at once"
    assert progress == {"runs": 2, "tool_calls": 3}
    assert sorted(counted()) == ["1", "2", "3"]
    assert api.errors == [] and runner.stub_calls == 0
