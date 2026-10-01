"""Record the golden histories in ``tests/histories`` (run: python -m tests.record_histories).

Each scenario runs with the scripted runner on a local dev server; every run of its
Workflow is saved as JSON. ``test_replay.py`` replays them, so a change to the
Workflow code that would break Workflows already running fails a test.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from temporalio.claude_agent_sdk import ClaudeAgentPlugin
from temporalio.claude_agent_sdk.testing import ScriptedClaude
from temporalio.client import Client, WorkflowHandle
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from tests import DEV_SERVER_DOWNLOAD_VERSION
from tests.endless.activities import ALL as COUNTING
from tests.endless.policy import count_policy
from tests.endless.workflows import ChatWorkflow, LongTaskWorkflow, TaskOptions
from tests.engine_tools.policy import shell_policy
from tests.engine_tools.workflows import ShellOptions, ShellWorkflow
from tests.lifecycle.workflows import TasksWorkflow
from tests.parallel.policy import parallel_policy
from tests.parallel.workflows import ParallelWorkflow
from tests.refund.activities import ALL as SHOP
from tests.refund.policy import refund_policy
from tests.refund.workflows import MANAGER, RefundAgentWorkflow

HISTORIES = Path(__file__).with_name("histories")
WORKFLOWS = [
    RefundAgentWorkflow,
    ParallelWorkflow,
    LongTaskWorkflow,
    ChatWorkflow,
    ShellWorkflow,
    TasksWorkflow,
]
"""Every Workflow the golden histories use (the replay test needs them all)."""


async def _first_pending(handle: WorkflowHandle[Any, Any], query: Any) -> str:
    for _ in range(600):
        pending = await handle.query(query)
        if pending:
            return pending[0]["id"]
        await asyncio.sleep(0.05)
    raise TimeoutError("no approval request")


async def refund(client: Client, queue: str) -> str:
    """A refund with a manager's approval: one durable call at a time."""
    handle = await client.start_workflow(
        RefundAgentWorkflow.run,
        "Order A-1001 arrived broken, I want my money back.",
        id=queue,
        task_queue=queue,
    )
    tool_use_id = await _first_pending(handle, RefundAgentWorkflow.pending_approvals)
    await handle.execute_update(
        RefundAgentWorkflow.review, args=[tool_use_id, True, MANAGER]
    )
    await handle.result()
    return queue


async def parallel(client: Client, queue: str) -> str:
    """Three calls in one message, one of them waiting for approval; live output."""
    handle = await client.start_workflow(
        ParallelWorkflow.run,
        args=[["parallel 2 and publish"], None],
        id=queue,
        task_queue=queue,
    )
    tool_use_id = await _first_pending(handle, ParallelWorkflow.pending_approvals)
    await handle.execute_update(ParallelWorkflow.review, args=[tool_use_id, True])
    await handle.result()
    return queue


async def continue_as_new(client: Client, queue: str) -> str:
    """A long task that continues as new twice, carrying its conversation."""
    handle = await client.start_workflow(
        LongTaskWorkflow.run,
        args=["count to 6", TaskOptions(after_events=24), None],
        id=queue,
        task_queue=queue,
    )
    await handle.result()
    return queue


async def chat(client: Client, queue: str) -> str:
    """A chat: tasks on one session, with Continue-As-New between messages."""
    handle = await client.start_workflow(
        ChatWorkflow.run, 3, id=queue, task_queue=queue
    )
    for text in ("count to 2", "count to 4", "count to 6"):
        await handle.signal(ChatWorkflow.send, text)
    await handle.result()
    return queue


async def bash(client: Client, queue: str) -> str:
    """A Claude Code Bash call that waits for approval, then runs as its own Activity."""
    handle = await client.start_workflow(
        ShellWorkflow.run,
        args=["run: make test", ShellOptions(tool_approvals=["Bash"])],
        id=queue,
        task_queue=queue,
    )
    tool_use_id = await _first_pending(handle, ShellWorkflow.pending_approvals)
    await handle.execute_update(ShellWorkflow.review, args=[tool_use_id, True])
    await handle.result()
    return queue


async def failed_task(client: Client, queue: str) -> str:
    """A task stopped by max_segments, then a task that answers the waiting call."""
    handle = await client.start_workflow(
        TasksWorkflow.run, ["count to 5", "count to 1"], id=queue, task_queue=queue
    )
    await handle.result()
    return queue


Scenario = Callable[[Client, str], Awaitable[str]]
SCENARIOS: dict[str, tuple[Scenario, Callable[[], Any], str]] = {
    "refund-approval": (refund, lambda: refund_policy, "held"),
    "refund-approval-store": (refund, lambda: refund_policy, "store"),
    "parallel-approval": (parallel, lambda: parallel_policy, "held"),
    "continue-as-new": (continue_as_new, lambda: count_policy, "held"),
    "chat-continue-as-new": (chat, lambda: count_policy, "held"),
    "bash-approval": (bash, lambda: shell_policy, "held"),
    "failed-task": (failed_task, lambda: count_policy, "held"),
}
"""Name: (how to run it, the scripted policy, where the conversation lives)."""


async def record(client: Client, name: str, folder: Path) -> list[Path]:
    """Run one scenario and save every run's history; return the files."""
    scenario, policy, mode = SCENARIOS[name]
    runner = ScriptedClaude(
        policy(),
        folder / name if mode == "store" else None,
        engine_tools={"Bash": lambda args: f"pretend output of {args['command']}"},
    )
    queue = f"golden-{name}-{uuid.uuid4().hex[:8]}"
    async with Worker(
        client,
        task_queue=queue,
        workflows=WORKFLOWS,
        activities=[*COUNTING, *SHOP],
        plugins=[ClaudeAgentPlugin(runner, heartbeat_every=1.0)],
    ):
        workflow_id = await scenario(client, queue)
    runs = [
        execution.run_id
        async for execution in client.list_workflows(f'WorkflowId = "{workflow_id}"')
    ]
    runs.reverse()  # oldest first
    files: list[Path] = []
    for number, run_id in enumerate(runs, start=1):
        history = await client.get_workflow_handle(
            workflow_id, run_id=run_id
        ).fetch_history()
        suffix = f"-run-{number}" if len(runs) > 1 else ""
        path = HISTORIES / f"{name}{suffix}.json"
        path.write_text(history.to_json(), encoding="utf-8")
        files.append(path)
    return files


async def main() -> None:
    """Record every scenario."""
    HISTORIES.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="golden-"))
    os.environ["SHOP_DIR"] = str(work / "shop")
    env = await WorkflowEnvironment.start_local(
        dev_server_download_version=DEV_SERVER_DOWNLOAD_VERSION
    )
    try:
        for name in SCENARIOS:
            for path in await record(env.client, name, work):
                print(f"recorded {path.name}")
    finally:
        await env.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
