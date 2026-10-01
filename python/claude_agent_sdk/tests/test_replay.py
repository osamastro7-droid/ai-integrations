"""Golden histories: recorded Workflows must replay the same way.

``python -m tests.record_histories`` recorded them with the scripted runner: a refund
with approval (with the conversation in the Workflow, and in a store), calls in one
message, Continue-As-New in a long task and in a chat, a Claude Code tool with
approval, and a task stopped by ``max_segments``. A change to the Workflow code that
would break Workflows already running fails here.

The ``first-version-*`` histories were recorded with the plugin as first published
(commit 766c647: one call at a time, the conversation in a session store), so
Workflows started with it keep running after an upgrade.
"""

from __future__ import annotations

import warnings

import pytest

from temporalio.claude_agent_sdk import ClaudeAgentPlugin
from temporalio.claude_agent_sdk.testing import ScriptedClaude
from temporalio.client import WorkflowHistory
from temporalio.worker import Replayer
from tests.endless.policy import count_policy
from tests.record_histories import HISTORIES, SCENARIOS, WORKFLOWS

FILES = sorted(p.name for p in HISTORIES.glob("*.json"))
FIRST_VERSION = "first-version-"


def test_every_scenario_has_a_golden_history() -> None:
    names = {name.split("-run-")[0].removesuffix(".json") for name in FILES}
    first = {n for n in names if n.startswith(FIRST_VERSION)}
    assert names - first == set(SCENARIOS)
    # The first version had no calls in parallel, no Claude Code tools as
    # Activities, and no conversation in the Workflow.
    assert {n.removeprefix(FIRST_VERSION) for n in first} == {
        "refund-approval",
        "continue-as-new",
        "chat-continue-as-new",
        "failed-task",
    }


@pytest.mark.parametrize("file_name", FILES)
async def test_replay(file_name: str) -> None:
    history = WorkflowHistory.from_json(
        file_name, (HISTORIES / file_name).read_text(encoding="utf-8")
    )
    with warnings.catch_warnings(record=True) as recorder:
        warnings.filterwarnings(
            "always", message=r"Module .* was imported after initial workflow load"
        )
        await Replayer(
            workflows=WORKFLOWS,
            plugins=[ClaudeAgentPlugin(ScriptedClaude(count_policy))],
        ).replay_workflow(history)
    # Sandbox imports during an activation count toward the deadlock timeout.
    assert not [
        str(w.message)
        for w in recorder
        if "was imported after initial workflow load" in str(w.message)
    ]
