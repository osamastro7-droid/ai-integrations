"""Test helpers for this integration."""

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from datetime import timedelta
from typing import TypeVar

from temporalio.api.enums.v1 import EventType
from temporalio.client import Client, WorkflowHandle
from temporalio.service import RPCError, RPCStatusCode
from temporalio.worker import Worker, WorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner


def new_worker(
    client: Client,
    *workflows: type,
    activities: Sequence[Callable] = [],
    task_queue: str | None = None,
    workflow_runner: WorkflowRunner = SandboxedWorkflowRunner(),
    max_cached_workflows: int = 1000,
    workflow_failure_exception_types: Sequence[type[BaseException]] = [],
    **kwargs,  # type:ignore[reportMissingParameterType]
) -> Worker:
    return Worker(
        client,
        task_queue=task_queue or str(uuid.uuid4()),
        workflows=workflows,
        activities=activities,
        workflow_runner=workflow_runner,
        max_cached_workflows=max_cached_workflows,
        workflow_failure_exception_types=workflow_failure_exception_types,
        **kwargs,
    )


T = TypeVar("T")


async def assert_eventually(
    fn: Callable[[], Awaitable[T]],
    *,
    timeout: timedelta = timedelta(seconds=10),
    interval: timedelta = timedelta(milliseconds=200),
    retry_on_rpc_cancelled: bool = True,
) -> T:
    start_sec = time.monotonic()
    while True:
        try:
            res = await fn()
            return res
        except AssertionError:
            if timedelta(seconds=time.monotonic() - start_sec) >= timeout:
                raise
        except RPCError as e:
            if not (retry_on_rpc_cancelled and e.status == RPCStatusCode.CANCELLED):
                raise
            if timedelta(seconds=time.monotonic() - start_sec) >= timeout:
                raise
        await asyncio.sleep(interval.total_seconds())


async def assert_eq_eventually(
    expected: T,
    fn: Callable[[], Awaitable[T]],
    *,
    timeout: timedelta = timedelta(seconds=10),
    interval: timedelta = timedelta(milliseconds=200),
) -> None:
    async def check() -> None:
        assert expected == await fn()

    await assert_eventually(check, timeout=timeout, interval=interval)


async def assert_event_subsequence(
    wf_handle: WorkflowHandle,
    expected_events: list[EventType.ValueType],
    timeout: timedelta = timedelta(seconds=5),
) -> None:
    """
    Given a workflow handle and a sequence of event types, assert that the workflow's history
    contains that subsequence of events in the order specified.
    """

    async def check():
        history = await wf_handle.fetch_history()

        _all_events = iter(history.events)
        _expected_events = iter(expected_events)

        previous_expected_event_type_name = None
        for expected_event_type in _expected_events:
            expected_event_type_name = EventType.Name(expected_event_type).removeprefix(
                "EVENT_TYPE_"
            )
            has_expected = next(
                (e for e in _all_events if e.event_type == expected_event_type),
                None,
            )
            if not has_expected:
                if previous_expected_event_type_name is not None:
                    prefix = f"After {previous_expected_event_type_name}, "
                else:
                    prefix = ""
                raise AssertionError(
                    f"{prefix}expected {expected_event_type_name} in workflow {wf_handle.id}"
                )
            previous_expected_event_type_name = expected_event_type_name

    await assert_eventually(check, timeout=timeout)
