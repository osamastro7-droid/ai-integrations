"""External pre-execution policy failures preserve the tool-step outcome contract."""

import uuid
from pathlib import Path
from typing import Any

import pytest

from temporalio import activity
from temporalio.api.failure.v1 import Failure
from temporalio.claude_agent_sdk import TOOL_CALL_REFUSED, ClaudeAgentPlugin
from temporalio.claude_agent_sdk._models import ToolStepInput
from temporalio.claude_agent_sdk._workflow import (
    _failed_step,
    _may_try_again,
    _refusals_final,
)
from temporalio.claude_agent_sdk.testing import HistoryItem
from temporalio.client import Client
from temporalio.common import RetryPolicy
from temporalio.converter import DefaultFailureConverter, DefaultPayloadConverter
from temporalio.exceptions import ActivityError, ApplicationError, RetryState
from temporalio.worker import (
    ActivityInboundInterceptor,
    ExecuteActivityInput,
    Interceptor,
    Worker,
)
from tests.endless.activities import ALL as COUNTING
from tests.engine_tools.policy import shell_policy
from tests.engine_tools.workflows import ShellOptions, ShellWorkflow
from tests.helpers.fake_messages_api import history_of, start_with_policy
from tests.helpers.workers import FAIL_FAST
from tests.test_engine_tools import make_runner

REFUSED_TEXT = (
    "This call was refused by policy before it ran. The same call will be refused "
    "again unless its authorization changes."
)


def activity_failure(cause: ApplicationError) -> ActivityError:
    failure = Failure()
    converter, payload = DefaultFailureConverter(), DefaultPayloadConverter()
    converter.to_failure(cause, payload, failure)
    restored = converter.from_failure(failure, payload)
    error = ActivityError(
        "Activity task failed",
        scheduled_event_id=1,
        started_event_id=2,
        identity="worker",
        activity_type="run_claude_tool_step",
        activity_id="tool-x",
        retry_state=RetryState.NON_RETRYABLE_FAILURE,
    )
    error.__cause__ = restored
    return error


def test_external_policy_refusal_survives_wire_without_retry_or_private_reason() -> (
    None
):
    # An interceptor raises this before the Activity body: no body enrichment.
    error = activity_failure(
        ApplicationError(
            "private policy configuration",
            {"try_again": True},
            type=TOOL_CALL_REFUSED,
            non_retryable=True,
        )
    )
    record: dict[str, str] = {}
    result = _failed_step(error, record)
    assert record["status"] == "refused" and result.is_error
    assert result.content == REFUSED_TEXT
    assert "private" not in str(result.content)
    assert not _may_try_again(error)
    assert isinstance(error.cause, ApplicationError) and error.cause.non_retryable


@pytest.mark.parametrize(
    "kind,status,text,retry",
    [
        ("ToolCallNotRun", "not run", "This call did not run", True),
        ("ToolCallInterrupted", "interrupted", "so it may have run", False),
        ("OtherFailure", "interrupted", "so it may have run", False),
    ],
)
def test_existing_failure_semantics_are_preserved(
    kind: str, status: str, text: str, retry: bool
) -> None:
    error = activity_failure(
        ApplicationError("failure", {"try_again": True}, type=kind)
    )
    record: dict[str, str] = {}
    result = _failed_step(error, record)
    assert record["status"] == status and result.is_error
    assert text in str(result.content)
    assert _may_try_again(error) is retry


def test_refusals_are_never_retried_whatever_the_policy_says() -> None:
    assert _refusals_final(None).non_retryable_error_types == [TOOL_CALL_REFUSED]
    mine = RetryPolicy(maximum_attempts=5, non_retryable_error_types=["Mine"])
    final = _refusals_final(mine)
    assert final.maximum_attempts == 5
    assert final.non_retryable_error_types == ["Mine", TOOL_CALL_REFUSED]
    assert mine.non_retryable_error_types == ["Mine"]  # the caller's is unchanged
    assert _refusals_final(final) is final


class RefuseTool(Interceptor):
    """A Worker interceptor standing in for a policy engine (no Tenuo): it refuses
    every attempt at one tool before the Activity body, and leaves out
    ``non_retryable`` on purpose: the plugin must make the refusal final itself."""

    def __init__(self, tool: str) -> None:
        self.tool = tool
        self.refused: list[str] = []

    def intercept_activity(
        self, next: ActivityInboundInterceptor
    ) -> ActivityInboundInterceptor:
        return _Refuse(next, self)


class _Refuse(ActivityInboundInterceptor):
    def __init__(self, next: ActivityInboundInterceptor, policy: RefuseTool) -> None:
        super().__init__(next)
        self.policy = policy

    async def execute_activity(self, input: ExecuteActivityInput) -> Any:
        kind = activity.info().activity_type
        step = input.args[0] if input.args else None
        name = step.call.name if isinstance(step, ToolStepInput) else kind
        if name == self.policy.tool:
            self.policy.refused.append(activity.info().activity_id)
            raise ApplicationError("private policy reason", type=TOOL_CALL_REFUSED)
        return await super().execute_activity(input)


async def run_refusing(
    client: Client, tmp_path: Path, tool: str, prompt: str, options: ShellOptions
) -> tuple[str, RefuseTool, list[dict[str, Any]], list[HistoryItem]]:
    api = start_with_policy(shell_policy)
    policy = RefuseTool(tool)
    queue = f"refuse-{uuid.uuid4().hex[:8]}"
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[ShellWorkflow],
            activities=COUNTING,
            interceptors=[policy],
            plugins=[
                ClaudeAgentPlugin(make_runner(tmp_path, api), heartbeat_every=1.0)
            ],
            **FAIL_FAST,
        ):
            result = await client.execute_workflow(
                ShellWorkflow.run, args=[prompt, options], id=queue, task_queue=queue
            )
            calls = await client.get_workflow_handle(queue).query(
                ShellWorkflow.tool_calls
            )
    finally:
        api.stop()
    assert api.errors == []
    seen = [h for body in api.requests for h in history_of(body)[2] if h.name == tool]
    return result, policy, calls, seen


@pytest.mark.timeout(240)
@pytest.mark.usefixtures("shop_dir")
async def test_real_engine_refused_bash_step_never_runs_and_claude_is_told(
    client: Client, tmp_path: Path
) -> None:
    """Bash is repeatable here, so only the plugin's own non-retryable type stops
    Temporal from trying the refused step again."""
    effects = tmp_path / "effects.log"
    result, policy, calls, seen = await run_refusing(
        client,
        tmp_path,
        "Bash",
        f'run: echo ran >> "{effects.as_posix()}"',
        ShellOptions(repeatable_tools=["Bash"]),
    )
    assert not effects.exists()  # the command never ran
    assert len(policy.refused) == 1  # and was not tried again
    assert result == f"error: {REFUSED_TEXT}"
    assert [c["status"] for c in calls if c["name"] == "Bash"] == ["refused"]
    assert seen and all(h.is_error and "private" not in str(h.content) for h in seen)


@pytest.mark.timeout(240)
@pytest.mark.usefixtures("shop_dir")
async def test_real_engine_refused_durable_tool_gets_the_refusal_not_its_message(
    client: Client, tmp_path: Path
) -> None:
    """``count`` has its own retry policy (five attempts): one refusal ends it."""
    effects = tmp_path / "effects.log"
    result, policy, calls, seen = await run_refusing(
        client,
        tmp_path,
        "count",
        f'count then run: echo ran >> "{effects.as_posix()}" && echo hi',
        ShellOptions(),
    )
    assert len(policy.refused) == 1
    assert [c["status"] for c in calls if c["name"] == "count"] == ["refused"]
    assert seen and all(h.is_error and h.content == REFUSED_TEXT for h in seen)
    assert result == "hi" and effects.read_text().split() == ["ran"]
