"""A Worker process for the chaos tests: the real runner, talking to the test's fake model.

Settings come from the environment, so every Worker the test starts after a kill is
the same: ``CHAOS_ENGINE_ENV`` (the runner's ``env``, as JSON), ``ENGINE_CWD``,
``RUNNER_MODE`` (``held`` or ``store``), ``SESSION_DIR`` and ``SHOP_DIR``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from temporalio.claude_agent_sdk import (
    ClaudeAgentPlugin,
    ClaudeAgentSdkRunner,
    FileSessionStore,
)
from temporalio.client import Client
from temporalio.worker import Worker
from tests.endless.activities import ALL
from tests.engine_tools.workflows import ShellWorkflow
from tests.helpers.workers import FAIL_FAST


async def main() -> None:
    """Run until killed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", required=True)
    parser.add_argument("--task-queue", required=True)
    args = parser.parse_args()
    store = os.environ.get("RUNNER_MODE", "held") == "store"
    runner = ClaudeAgentSdkRunner(
        session_store=FileSessionStore(os.environ["SESSION_DIR"]) if store else None,
        cwd=os.environ["ENGINE_CWD"],
        env=json.loads(os.environ["CHAOS_ENGINE_ENV"]),
    )
    client = await Client.connect(args.address)
    worker = Worker(
        client,
        task_queue=args.task_queue,
        workflows=[ShellWorkflow],
        activities=ALL,
        plugins=[ClaudeAgentPlugin(runner, heartbeat_every=1.0)],
        **FAIL_FAST,
    )
    print("worker ready", flush=True)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
