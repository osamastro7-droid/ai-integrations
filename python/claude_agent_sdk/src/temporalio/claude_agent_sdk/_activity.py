"""Activity side: runs one model segment through a pluggable runner."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Protocol

from temporalio import activity

from ._events import emit, publishing
from ._models import SegmentInput, SegmentOutput
from ._workflow import SEGMENT_ACTIVITY_NAME


class SegmentRunner(Protocol):
    """Runs one model segment. Implemented by ``ClaudeAgentSdkRunner`` and ``ScriptedClaude``.

    The contract that makes retries safe: every segment that is not an error returns a
    ``checkpoint``, and the next segment of the session receives it. A segment that
    runs again (``attempt`` > 1, or ``inp.fork``) must continue from
    ``inp.checkpoint`` and ignore anything an unfinished attempt wrote after it.

    A runner that lets the Workflow hold the conversation reads it from
    ``inp.conversation`` (or ``inp.transcript``) and returns what changed in
    ``transcript_keep`` and ``transcript_add``; one that keeps it elsewhere returns
    ``transcript_keep=None``. Both report ``external_storage``, so the Workflow knows
    whether a large state can move to a new run.
    """

    async def run(self, inp: SegmentInput, attempt: int) -> SegmentOutput:
        """Run one segment.

        Args:
            inp: What to run.
            attempt: The Activity attempt number, starting at 1.

        Returns:
            Where the segment stopped: a pause at a tool call, or a final answer.
        """
        ...


def make_segment_activity(
    runner: SegmentRunner, *, heartbeat_every: float = 5.0
) -> Callable[..., Any]:
    """Build the segment Activity around a runner.

    Args:
        runner: Runs the segment.
        heartbeat_every: Seconds between heartbeats while the segment runs (more often
            when the segment's heartbeat timeout needs it: at least three per timeout).

    Returns:
        The Activity function, named ``run_claude_segment``.
    """

    @activity.defn(name=SEGMENT_ACTIVITY_NAME)
    async def run_claude_segment(inp: SegmentInput) -> SegmentOutput:
        info = activity.info()
        every = heartbeat_every
        if info.heartbeat_timeout:
            every = min(every, info.heartbeat_timeout.total_seconds() / 3)

        async def beat() -> None:
            while True:
                activity.heartbeat(inp.segment_index)
                await asyncio.sleep(every)

        attempt = info.attempt
        beater = asyncio.create_task(beat())
        try:
            async with publishing(
                inp.live_output, segment=inp.segment_index, attempt=attempt
            ):
                if attempt > 1:
                    emit({"type": "retry"})  # this segment's earlier text is superseded
                return await runner.run(inp, attempt)
        finally:
            beater.cancel()

    return run_claude_segment
