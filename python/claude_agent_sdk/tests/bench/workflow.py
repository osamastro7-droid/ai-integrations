"""The segment benchmark Workflow, as in the hybrid prototype."""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.claude_agent_sdk import DurableClaudeAgent, activity_as_tool

with workflow.unsafe.imports_passed_through():
    from tests.bench.workload import SCHEMA, segment_echo


@workflow.defn
class SegmentBenchmarkWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        agent = DurableClaudeAgent(
            tools=[
                activity_as_tool(
                    segment_echo,
                    name="echo",
                    input_schema=SCHEMA,
                    start_to_close_timeout=timedelta(seconds=30),
                )
            ],
            max_segments=None,
        )
        return await agent.run(prompt)
