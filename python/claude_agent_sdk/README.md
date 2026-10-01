# Durable Claude Agent SDK agents on Temporal

> ⚠️ **Experimental.** The API may change.

Temporal integration for Anthropic's [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview), published as [`temporalio-claude-agent-sdk`](https://pypi.org/project/temporalio-claude-agent-sdk/) and imported as `temporalio.claude_agent_sdk`.

- **Every durable tool call Claude makes is its own Temporal Activity.** Finished calls never run again after a crash, retries follow your retry policy, and the Activity ID (`tool-<tool_use_id>`) doubles as an idempotency key for the systems a tool touches.
- **Tools can wait for a human.** Mark a tool `needs_approval=True` and the agent waits, for minutes or weeks, until someone approves or rejects the call.
- **Crashes resume cleanly, on any Worker.** Each model step ends at a checkpoint that Temporal records. A step that runs again (after a crash, a timeout, or on another machine) starts from that checkpoint, so nothing a failed attempt added to the conversation reaches Claude.

## Install

```bash
uv add temporalio-claude-agent-sdk
```

It depends on `claude-agent-sdk>=0.2.153`, which bundles Claude Code 2.1.273 or newer (see [Requirements](#requirements-and-limits)).

**Windows.** claude-agent-sdk 0.2.160 to 0.2.163 publish no Windows wheel (the bundled Windows engine is over PyPI's 100 MiB file limit), and their source package has no engine, so the plugin skips those versions on Windows. If a newer release misses its Windows wheel too, pin `claude-agent-sdk` to a version that has one, or install Claude Code natively and pass `cli_path` to the runner.

## Quick start

A Workflow with an agent inside. Tools are ordinary Activities that take one dict argument:

```python
from datetime import timedelta
from typing import Any

from temporalio import activity, workflow
from temporalio.claude_agent_sdk import DurableClaudeAgent, activity_as_tool


@activity.defn
async def look_up_order(args: dict[str, Any]) -> dict[str, Any]:
    """Look up an order by id."""
    ...


@activity.defn
async def issue_refund(args: dict[str, Any]) -> dict[str, Any]:
    """Refund an order. Moves real money."""
    key = activity.info().activity_id  # stable across retries: use it as the idempotency key
    ...


@workflow.defn
class RefundAgent:
    def __init__(self) -> None:
        self.agent = DurableClaudeAgent(
            system_prompt="You handle refund requests for an online store.",
            tools=[
                activity_as_tool(look_up_order),
                activity_as_tool(issue_refund, needs_approval=True),
            ],
            approvers=["manager@shop.example"],
        )

    @workflow.run
    async def run(self, request: str) -> str:
        return await self.agent.run(request)

    @workflow.update
    def review(self, tool_use_id: str, approved: bool, approver: str) -> None:
        self.agent.decide(tool_use_id, approved, approver)

    @review.validator
    def check_review(self, tool_use_id: str, approved: bool, approver: str) -> None:
        self.agent.validate_decision(tool_use_id, approver)  # refused Updates never reach history

    @workflow.query
    def pending_approvals(self) -> list[dict[str, Any]]:
        return self.agent.pending_approvals()
```

Tool names use 1 to 50 letters, digits, `_` or `-` (Claude Code renames anything else, and the call could never run). Claude's arguments are not checked against `input_schema`: validate them in the Activity, like any untrusted input. `activity_as_tool` also takes `schedule_to_close_timeout`, `heartbeat_timeout` and `task_queue`.

The Worker runs Claude through the plugin:

```python
from temporalio.claude_agent_sdk import ClaudeAgentPlugin, ClaudeAgentSdkRunner, FileSessionStore

runner = ClaudeAgentSdkRunner(
    session_store=FileSessionStore("/shared/claude-sessions"),
    cwd="/srv/agent",
)
worker = Worker(
    client,
    task_queue="agents",
    workflows=[RefundAgent],
    activities=[look_up_order, issue_refund],
    plugins=[ClaudeAgentPlugin(runner)],
)
```

The conversation lives in a session store that every Worker must reach. `FileSessionStore` is for tests and one machine; for production, implement `SessionStore` on your database or object storage (the Claude Agent SDK repository has example stores for S3, Redis and Postgres). The store keys sessions by the engine's working directory, so give every Worker the same `cwd`. The runner refuses a path for which Claude Code and the SDK would compute different keys (for example with decomposed Unicode or emoji).

Pass the plugin to the Worker, or to the Client the Worker is built from, not both. Each segment starts a Claude Code process: 0.7 to 0.9 seconds and up to about 270 MB of memory each (measured on Linux with an instant local model), so cap parallel segments with the Worker's `max_concurrent_activities`.

Other [`ClaudeAgentOptions`](https://code.claude.com/docs/en/agent-sdk/python) go in the runner's `extra_options` (for example `permission_mode`, `agents`, `hooks`, `setting_sources`, `thinking`). `env`, `mcp_servers` and `allowed_tools` are merged with the plugin's own, and `system_prompt` (a string, or a preset such as Claude Code's own prompt) is the default for agents that set none. Options the agent or the plugin sets (`model`, `tools`, `max_turns`, `cwd`, `settings`, session and resume options, and the same engine flags in `extra_args`) are refused. The runner sets `permission_mode="default"` unless you pass one: since Claude Code 2.1.285, a run without one uses auto mode when telemetry is off or the provider is Bedrock, Vertex or Foundry, and auto mode asks the model whether each tool call may run. External MCP servers run inside the segment, like built-in tools, and hooks from settings you load must not decide on the plugin's `mcp__durable__` tools.

`approvers` checks the name the caller passes. It is not authentication: control who may send Updates with Temporal's own access control.

## Long-running agents

A Workflow's history holds at most 51,200 events or 50 MB (Temporal's default limits), and every durable tool call adds about 12 events. So one run fits about 4,000 tool calls, and fewer when results are large. Continue-As-New starts a fresh history. The conversation stays in the session store, so the agent's state is small.

Turn on `auto_continue_as_new` and give the Workflow a `state` argument:

```python
from temporalio.claude_agent_sdk import AgentState, DurableClaudeAgent


@workflow.defn
class ResearchAgent:
    @workflow.init
    def __init__(self, prompt: str, state: AgentState | None = None) -> None:
        self.agent = DurableClaudeAgent(
            tools=[...], state=state, max_segments=None, auto_continue_as_new=True
        )

    @workflow.run
    async def run(self, prompt: str, state: AgentState | None = None) -> str:
        return await self.agent.run(prompt)  # continues as new inside when needed
```

- When the server suggests it (`is_continue_as_new_suggested()`, which counts history length, history size and Updates), the agent continues as new at the next safe point between two tool calls. It lets running Update and Signal handlers finish and continues as new with `[prompt, state]`. The new run calls `agent.run(prompt)` again; the agent sees the unfinished task in its state and continues it instead of sending the prompt again.
- It is off by default, because the new run's arguments must match your run method. For other arguments, pass `continue_as_new_args`, a function that builds them from the `AgentState`. `continue_as_new_after_events` uses a fixed history length instead of the server's suggestion.
- Call `agent.run()` from the Workflow's run method, not from a separate asyncio task, and let no handler wait for the task to finish: continuing as new waits for every handler to finish. With `auto_continue_as_new`, a call from a handler fails at once: an Update handler fails its Update, a Signal handler fails the Workflow.
- **Chats.** Calling `agent.run(message)` again continues the same Claude session, so Claude remembers earlier turns. Turns without tool calls never reach a point between tool calls, so between messages, when no handler is waiting for an answer, call `await agent.continue_as_new()` from the run method if `agent.should_continue_as_new()` is true; `continue_as_new_args` carries your own state, such as an inbox. At the start of a new run, if `agent.busy` is true, `await agent.run()` finishes the task that was interrupted.
- `max_segments` caps a task across all its runs (default 50). The task stops before it runs a tool whose result could no longer reach Claude. Use `None` for tasks that may take thousands of steps.
- **Start with every argument.** Temporal applies a Workflow's argument types only when the caller passes as many arguments as `run` declares. If `run` has other typed arguments besides `state`, start the Workflow with `state=None` included, or those arguments arrive as plain dicts.

## Failures and cancellation

- A task fails with an `ApplicationError` at `max_segments`, when a step reports an error that running it again cannot fix (Claude Code's `max_turns` or `max_budget_usd`, a broken pause, or a Messages API request that would be refused again: invalid (400), an unknown model (404), too large (413), or a prompt too long for the model), or if the engine hands back a tool call that already ran. Other engine and API errors, such as a low credit balance, a rejected API key, rate limits or overload, fail the step's Activity, and Temporal retries it with `segment_retry_policy` (by default without limit, so the step continues once the cause is fixed; each retry copies the session). When the Activity fails for good, the task fails with an `ActivityError`. Catch `temporalio.exceptions.FailureError` for both.
- After a failed task, the agent can take the next one: it continues from the last checkpoint, and a tool call Claude was still waiting for gets an error result, delivered with the next prompt.
- Cancelling the Workflow cancels the running step or tool call, and the Workflow ends as cancelled. By default Temporal does not wait for a cancelled Activity. `activity_as_tool(..., cancellation_type=workflow.ActivityCancellationType.WAIT_CANCELLATION_COMPLETED)` waits until the call finishes or acknowledges the cancellation through a heartbeat, so the history records what really happened; `segment_cancellation_type` does the same for a running step.
- A running step hears of a cancel (or of its own timeout) with its next heartbeat: at most 0.8 × `segment_heartbeat_timeout` later (30 seconds by default, so up to 24 seconds). From then on the engine cannot start a built-in tool while the SDK stops it, which takes up to about 10 seconds (a durable call only pauses the run). If the Worker process dies, its engine runs until its current turn ends, built-in tools included, unless the Worker's child processes stop with it (as in a container or a systemd service).

## Large tool results

Every tool result is stored in the Workflow's history twice: as the tool Activity's result, and in the next step's input. A single payload over 2 MB cannot be recorded at all (the SDK stops it with `[TMPRL1103] Attempted to upload payloads with size that exceeded the error limit`), and results of a few hundred KB fill the 50 MB history quickly.

The plugin works with Temporal's [External Storage](https://docs.temporal.io/external-storage) unchanged. Payloads over a threshold (256 KiB by default) go to your store, and the history keeps small references. Configure it on the Client, as for any Temporal application; Workers built from that Client use it too:

```python
from temporalio.converter import DataConverter, ExternalStorage

client = await Client.connect(
    "localhost:7233",
    data_converter=DataConverter(external_storage=ExternalStorage(drivers=[my_s3_driver])),
)
```

Use the same External Storage on every Client that starts or queries these Workflows. Measured on a local dev server at Temporal's default limits, with 40 tool results of 1 MB: without it, the run was terminated after 25 results ("Workflow history size exceeds limit"); with it, the task finished in one run with 84 KB of history.

## Live output

Turn on `live_output=True` and the agent publishes its events through Temporal's [Workflow Streams](https://docs.temporal.io/develop/python/workflows/workflow-streams). A UI reads them with `follow_agent`, which follows Continue-As-New by itself:

```python
from temporalio.claude_agent_sdk import follow_agent

async for event in follow_agent(client, workflow_id):
    print(event["type"], event.get("text") or event.get("name") or event.get("result"))
    if event["type"] in ("done", "error", "cancelled"):
        break
```

- Event types: `prompt`, `text`, `tool_call`, `approval_needed`, `tool_result`, `retry`, `continued_as_new`, `done`, `error` and `cancelled`. Every event has the time it happened (`at`) and its `offset`.
- Claude's text arrives one assistant message at a time, not token by token. `text` and `retry` events carry the `segment` and `attempt` that produced them: after a `retry` event, the earlier text of that segment is superseded.
- After a disconnect, continue from the last offset plus one. Continue-As-New carries only the newest events (`live_output_keep`, 1,000, and `live_output_keep_bytes`, 256 KiB); asked for an older offset, the stream starts at the oldest event it still has, so compare offsets to see a gap.
- A `text`, `result` or `error` longer than 32,768 characters, or a tool `input` longer than that as JSON, is cut, and the event gets `truncated: true`. The full answer is the Workflow's result.
- After its final event, a task waits `live_output_linger` (500 ms) so subscribers receive it before the Workflow closes.
- Measured on a local dev server, three runs of 203 events: 107 to 110 ms median and 161 to 162 ms at the 95th percentile, from event to subscriber. Workflow Streams is built for UIs and progress, not real-time voice.
- Continue-As-New carries the stream in the new run's input, next to the agent's state. The stream shrinks to fit, so a large tool result waiting to be delivered still fits under Temporal's 2 MB payload limit. The rest of the input is measured before any codec or External Storage, which can only overstate it, so the stream may keep fewer events than would fit.
- **Every subscriber poll is an Update**, and a Workflow accepts at most 10 Updates in flight and 2,000 per run (Temporal's defaults). Idle subscribers hold Updates in flight, so with many direct subscribers an Update approval can be refused (tested: with 12, it failed with `RESOURCE_EXHAUSTED`). Approve by Signal instead (`decide()` ignores invalid decisions there), or fan events out to viewers through one subscriber in your backend. For long runs with live output, turn on `auto_continue_as_new`: the server counts the polls toward its suggestion.
- Create the agent while the Workflow is being initialized (in its `__init__`), where Workflow Streams registers its handlers; later, the constructor raises.

## How it works

The agent loop runs in *segments*. A segment is one Activity that runs the Claude Code engine from a prompt (or a tool result) until Claude either calls a durable tool or finishes.

1. Durable tools are declared to Claude as SDK MCP tools.
2. A `PreToolUse` command hook answers `defer` whenever Claude calls one ([documented in the hooks guide](https://code.claude.com/docs/en/hooks)). The engine stops with `stop_reason: "tool_deferred"`, and the segment returns the call.
3. The Workflow runs the call as its own Activity, after an approval if the tool needs one.
4. The next segment resumes the session with the result as a normal `tool_result` message.

**Checkpoints.** After a segment, the runner reads the session back from the store and returns its last transcript entry, where the engine would resume; the Workflow records that checkpoint with the segment's result. Reading it back also proves the turn reached the store. A segment that runs again continues in a copy of the session that ends at the checkpoint (`fork_session_via_store`), and Claude decides again from there, with new tool call ids. So does a segment whose session went on past its checkpoint, for example after a Workflow reset: the check rides on the load the SDK does anyway. After Claude sent several durable calls at once, the checkpoint is the paused call's deferral marker (the denied calls' results come after it, and the engine would not resume the paused call past them), so the next segment also continues in a copy. Tested on the real engine: a Worker killed while Claude is answering, an attempt that hangs past its timeout, and results lost after a step finished.

**Fail closed.** If a Claude Code version ever runs a durable tool itself, or ignores a pause, the segment fails with a non-retryable error that names the engine version. Nothing runs outside Temporal. If the engine ever hands back a tool call id that already ran (the agent remembers every call of the current run and the last 256 before it), the Workflow stops instead of running it twice.

## Requirements and limits

- **Claude Code 2.1.273 or newer.** Tested: when Claude calls two tools in one message, Claude Code 2.1.259 replaces the paused call's result with `[Tool result missing due to internal error]`, so Claude asks for the same tool again. The runner checks `claude -v` before starting the engine and refuses older engines.
- **One durable tool call at a time.** The engine keeps one paused call per run. The runner asks Claude for one call per message; if Claude sends several, the first durable call pauses and every call after it in the same message, durable or built-in, is told to call again after its result.
- **Claude Code's built-in tools** (Bash, Edit, and so on) are off unless you pass `builtin_tools`. When enabled, they run inside the segment Activity, not as their own Activities, on the Worker's disk in the runner's `cwd` (shared by every agent on that Worker). They can run again when a segment runs again, possibly on another Worker with a different disk, and files a failed attempt wrote are not rolled back. Give them work that is safe to repeat, or make the work a durable tool.
- **A session store every Worker can reach**, and the same `cwd` on every Worker. Every segment reads the whole session twice (the SDK to resume it, the runner to record the checkpoint), so reading grows with the session's length. A segment that runs again reads it once more and writes a copy; the earlier copy stays in the store, so remove old sessions with your store's own retention.
- **Subagents run in the foreground, without durable tools.** The runner turns off Claude Code's background tasks (a background subagent kept the engine working after it paused). A subagent (Claude Code's `Agent` tool) can use built-in tools; a durable tool call from a subagent cannot pause the run, so the step fails closed.
- **Long conversations.** Continue-As-New keeps the Workflow's history small, but the conversation itself grows until Claude Code compacts it. A request the model refuses as too long fails the task.
- **Logins that survive resumes.** Use `ANTHROPIC_API_KEY`, Amazon Bedrock, Google Vertex AI, Microsoft Foundry, or `CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token`. A Claude app login cannot refresh itself when a session is resumed from a session store.

## Security

- Credentials stay on Workers: the engine inherits the Worker's environment (API keys, cloud credentials, proxies). They reach Workflow history only if a built-in tool shows them to Claude (for example Bash running `env`), so give Workers that enable built-in tools only the secrets they need.
- The conversation does: prompts, Claude's text, tool arguments and results are in the Workflow's history and in the session store. Encrypt payloads with a codec and protect the store like any other data store.
- Claude chooses tool arguments. Treat them as untrusted input in your Activities, and require approval (`needs_approval=True`) for tools that move money or delete data.

## Testing your agents

`temporalio.claude_agent_sdk.testing.ScriptedClaude` is a segment runner that plays Claude with a Python policy. It needs no engine and no API key. It keeps sessions in a folder, so tests can kill a Worker and continue on a new one, and its checkpoints behave like the real runner's: a segment that runs again decides again, with a new tool call id.

## Documentation

- Repository conventions: [`AGENTS.md`](https://github.com/temporalio/ai-integrations/blob/main/AGENTS.md)

## Develop

```bash
make sync   # install (non-editable) into .venv
make lint
make test   # the real Claude Code engine against a local fake Messages API; no credentials
```
