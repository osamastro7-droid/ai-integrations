"""What keeps Claude Code's own tools inside the plugin's guarantees.

A command or MCP tool that runs as its own Activity must run once, with its approval,
in the Worker's environment, and talk to no model; nothing a tool prints, a hook in
``extra_options`` answers, or a settings file sets may change that. Each test here
failed before the change it guards.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import HookMatcher

from temporalio.claude_agent_sdk import (
    ClaudeAgentSdkRunner,
    FileSessionStore,
    SegmentInput,
    ToolSpec,
    ToolStepInput,
    _defer_hook,
    _runner,
    _stand_in,
)
from temporalio.exceptions import ApplicationError
from tests.engine_tools.policy import shell_policy
from tests.helpers.fake_messages_api import (
    FakeMessagesAPI,
    engine_env,
    history_of,
    start_with_policy,
)

pytestmark = pytest.mark.timeout(240)
TOOLS = [ToolSpec("count", "Count one step.", {"type": "object"})]


def make_runner(
    tmp_path: Path, api: FakeMessagesAPI, cfg: str = "cfg", **kw: Any
) -> ClaudeAgentSdkRunner:
    (tmp_path / "work").mkdir(exist_ok=True)
    return ClaudeAgentSdkRunner(
        cwd=str(tmp_path / "work"), env=engine_env(api, str(tmp_path / cfg)), **kw
    )


async def pause_at_bash(runner: ClaudeAgentSdkRunner, command: str) -> Any:
    out = await runner.run(
        SegmentInput(
            session_id=str(uuid.uuid4()),
            prompt=f"run: {command}",
            tools=TOOLS,
            builtin_tools=["Bash"],
            transcript=[],
            tool_activities=["Bash"],
        ),
        1,
    )
    assert out.deferred is not None and out.deferred.kind == "engine", out
    return out


def step_for(out: Any) -> ToolStepInput:
    return ToolStepInput(
        session_id=out.session_id,
        checkpoint=out.checkpoint,
        call=out.deferred,
        tools=TOOLS,
        builtin_tools=["Bash"],
        transcript=out.transcript_add,
    )


def lines(path: Path) -> list[str]:
    return path.read_text().split() if path.exists() else []


# ---- a tool step runs once, talks to no model, and keeps the Worker's environment ----


async def test_a_tool_step_keeps_its_result_when_the_engine_fails_afterwards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model call after the tool fails: the call ran, so the step returns its
    result instead of failing, which would make Temporal run the command again."""
    effects = tmp_path / "work" / "effects.log"
    api = start_with_policy(shell_policy)
    runner = make_runner(tmp_path, api)

    def refuse(handler: Any) -> None:
        handler.rfile.read(int(handler.headers.get("content-length", 0)))
        error = {"type": "invalid_request_error", "message": "stand-in refuses"}
        handler._send({"type": "error", "error": error}, status=400)

    try:
        out = await pause_at_bash(runner, "echo ran >> effects.log && echo local")
        monkeypatch.setattr(_stand_in._Handler, "do_POST", refuse)
        outcome = await runner.run_tool_step(step_for(out), 1)
    finally:
        api.stop()
    assert outcome.content == "local" and not outcome.is_error
    assert lines(effects) == ["ran"]


@pytest.mark.parametrize("where", ["user settings", "global config"])
async def test_settings_files_cannot_send_a_tool_steps_model_calls_elsewhere(
    tmp_path: Path, where: str
) -> None:
    """Provider settings in Claude Code's settings files would win over the Worker's
    environment; a tool step's model calls must still reach only the stand-in."""
    api = start_with_policy(shell_policy)
    provider = start_with_policy(shell_policy)  # where the settings point
    cfg = tmp_path / "cfg-x"
    cfg.mkdir()
    env = {"ANTHROPIC_BASE_URL": provider.base_url}
    extra: dict[str, Any] = {}
    if where == "global config":
        (cfg / ".claude.json").write_text(json.dumps({"env": env}))
    else:
        (cfg / "settings.json").write_text(json.dumps({"env": env}))
        extra = {"setting_sources": ["user"]}
    plain = make_runner(tmp_path, api)
    configured = make_runner(tmp_path, api, cfg="cfg-x", extra_options=extra)
    try:
        out = await pause_at_bash(plain, "echo local")
        outcome = await configured.run_tool_step(step_for(out), 1)
    finally:
        api.stop()
        provider.stop()
    assert outcome.content == "local"
    assert provider.requests == []
    assert configured._stand_in.requests >= 1  # type: ignore[reportPrivateUsage]


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows ignores case in variable names: NO_PROXY and no_proxy are one",
)
async def test_a_command_in_a_tool_step_sees_the_workers_environment(
    tmp_path: Path,
) -> None:
    """The step's engine talks to the stand-in, but the command gets the Worker's
    provider settings back (a script that calls the Anthropic API works as in a
    segment), and proxy exceptions are merged, not replaced."""
    api = start_with_policy(shell_policy)
    url = api.base_url
    runner = make_runner(tmp_path, api)
    runner._env["no_proxy"] = "internal.example"  # type: ignore[reportPrivateUsage]
    command = 'echo "url=$ANTHROPIC_BASE_URL key=$ANTHROPIC_API_KEY np=$no_proxy"'
    try:
        out = await pause_at_bash(runner, command)
        outcome = await runner.run_tool_step(step_for(out), 1)
    finally:
        api.stop()
    assert outcome.content == (
        f"url={url} key=sk-ant-fake-not-real np=internal.example"
    )


@pytest.mark.parametrize("words", [_defer_hook.STOPPED, _defer_hook.STEP_ONLY])
async def test_a_command_that_prints_the_hooks_words_still_ran(
    tmp_path: Path, words: str
) -> None:
    """The step decides "did not run" from the hook's own records, not from the
    output, which a command (or an MCP tool) controls."""
    effects = tmp_path / "work" / "effects.log"
    api = start_with_policy(shell_policy)
    runner = make_runner(tmp_path, api)
    try:
        out = await pause_at_bash(runner, f"echo ran >> effects.log && echo '{words}'")
        outcome = await runner.run_tool_step(step_for(out), 1)
        # Inside a segment too: such output does not stop the step.
        inside = await runner.run(
            SegmentInput(
                session_id=str(uuid.uuid4()),
                prompt=f"run: echo '{words}'",
                tools=TOOLS,
                builtin_tools=["Bash"],
                transcript=[],
                tool_activities=[],
            ),
            1,
        )
    finally:
        api.stop()
    assert outcome.content == words and lines(effects) == ["ran"]
    assert not inside.is_error and inside.result == words


@pytest.mark.parametrize("activities", [[], ["Bash"]], ids=["segment", "tool-step"])
async def test_every_command_starts_in_the_working_directory(
    tmp_path: Path, activities: list[str]
) -> None:
    """Each segment and tool step is a new engine run, so a `cd` cannot carry over;
    the plugin makes that the rule everywhere, and tells Claude."""
    (tmp_path / "work" / "sub").mkdir(parents=True)
    ref: list[FakeMessagesAPI] = []

    def decide(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, _, history = history_of(body)
        bash = [h for h in history if h.name == "Bash"]
        if not bash:
            return [ref[0].call("Bash", {"command": "cd sub && echo 1 > here.txt"})]
        if len(bash) == 1:
            return [ref[0].call("Bash", {"command": "echo 2 > here.txt"})]
        return [{"type": "text", "text": "FINAL"}]

    api = FakeMessagesAPI(decide).start()
    ref.append(api)
    runner = make_runner(tmp_path, api)
    transcript: list[dict[str, Any]] = []
    inp = SegmentInput(
        session_id=str(uuid.uuid4()),
        prompt="go",
        tools=TOOLS,
        builtin_tools=["Bash"],
        transcript=[],
        tool_activities=activities,
    )
    try:
        out = await runner.run(inp, 1)
        for index in range(1, 4):
            assert not out.is_error, out.error
            transcript = transcript[: out.transcript_keep] + out.transcript_add
            if out.deferred is None:
                break
            assert out.checkpoint is not None
            outcome = await runner.run_tool_step(
                ToolStepInput(
                    session_id=out.session_id,
                    checkpoint=out.checkpoint,
                    call=out.deferred,
                    tools=TOOLS,
                    builtin_tools=["Bash"],
                    transcript=list(transcript),
                ),
                1,
            )
            inp = SegmentInput(
                session_id=out.session_id,
                prompt=None,
                tools=TOOLS,
                builtin_tools=["Bash"],
                checkpoint=out.checkpoint,
                injected={out.deferred.id: outcome},
                transcript=list(transcript),
                tool_activities=activities,
                segment_index=index,
            )
            out = await runner.run(inp, 1)
    finally:
        api.stop()
    assert out.result == "FINAL"
    work = tmp_path / "work"
    # The second command did not start in sub/, where the first one went.
    assert (work / "sub" / "here.txt").read_text().split() == ["1"]
    assert (work / "here.txt").read_text().split() == ["2"]
    systems = [json.dumps(r.get("system")) for r in api.requests]
    assert systems and all(_runner.SHELL_HINT[:40] in s for s in systems)


# ---- calls that must wait for the Workflow never run inside the segment ----


async def test_a_subagent_cannot_run_a_command_that_runs_as_an_activity(
    tmp_path: Path,
) -> None:
    """With tool_approvals=["Bash"], a subagent's command would run with no approval;
    it is denied with a hint, so commands only ever run as Activities."""
    effects = tmp_path / "work" / "effects.log"
    task = "Run the cleanup command."
    ref: list[FakeMessagesAPI] = []
    seen: list[str] = []

    def decide(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, texts, history = history_of(body)
        if any(task in t for t in texts) and not any(
            t.startswith("Clean up.") for t in texts
        ):  # the subagent's conversation
            if history:
                seen.extend(str(h.content) for h in history)
                return [{"type": "text", "text": "done"}]
            return [ref[0].call("Bash", {"command": "echo ran >> effects.log"})]
        if not any(h.name == "Agent" for h in history):
            return [
                ref[0].call(
                    "Agent",
                    {
                        "description": "cleanup",
                        "prompt": task,
                        "subagent_type": "general-purpose",
                    },
                )
            ]
        return [{"type": "text", "text": "FINAL"}]

    api = FakeMessagesAPI(decide).start()
    ref.append(api)
    runner = make_runner(tmp_path, api)
    try:
        out = await runner.run(
            SegmentInput(
                session_id=str(uuid.uuid4()),
                prompt="Clean up.",
                tools=TOOLS,
                builtin_tools=["Bash", "Agent"],
                transcript=[],
                tool_activities=["Bash"],
            ),
            1,
        )
    finally:
        api.stop()
    assert out.result == "FINAL" and lines(effects) == []
    assert seen and "only the main agent can call it" in seen[0]
    # The subagent's own steps are not part of the conversation the Workflow keeps;
    # its report is, as the Agent call's result.
    calls = [
        block.get("name")
        for entry in out.transcript_add
        for block in (entry.get("message") or {}).get("content") or []
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]
    assert calls == ["Agent"]


async def test_a_broken_hook_cannot_let_a_command_run_inside_the_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the hook gives no answer (it crashed), the command is not pre-approved, so
    Claude Code refuses a command that changes something instead of running it."""
    monkeypatch.setattr(
        _runner,
        "_hook_entry",
        lambda: {
            "type": "command",
            "command": sys.executable,
            "args": ["-c", "import sys; sys.exit(1)"],
        },
    )
    effects = tmp_path / "work" / "effects.log"
    api = start_with_policy(shell_policy)
    runner = make_runner(tmp_path, api)
    try:
        out = await runner.run(
            SegmentInput(
                session_id=str(uuid.uuid4()),
                prompt="run: echo ran >> effects.log",
                tools=TOOLS,
                builtin_tools=["Bash"],
                transcript=[],
                tool_activities=["Bash"],
            ),
            1,
        )
    finally:
        api.stop()
    assert out.deferred is None and lines(effects) == []


@pytest.mark.parametrize("order", [["refund"], ["look_up", "refund"]])
async def test_a_hook_in_extra_options_cannot_decide_on_durable_calls(
    tmp_path: Path, order: list[str]
) -> None:
    """A deny from your own hook was overridden when the call came second in a
    message (the Workflow ran it with the paused one). Such hooks now stop the step
    with an error that says what to change, before any call runs."""

    async def no_refunds(event: Any, tool_use_id: Any, context: Any) -> Any:
        del tool_use_id, context
        if event.get("tool_name") == "mcp__durable__refund":
            output = {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "refunds are not allowed",
            }
            return {"hookSpecificOutput": output}
        return {}

    ref: list[FakeMessagesAPI] = []

    def decide(body: dict[str, Any]) -> list[dict[str, Any]]:
        uses, _, _ = history_of(body)
        if not uses:
            return [ref[0].tool_use(name, {"order": "A-1"}) for name in order]
        return [{"type": "text", "text": "FINAL"}]

    api = FakeMessagesAPI(decide).start()
    ref.append(api)
    runner = make_runner(
        tmp_path,
        api,
        extra_options={
            "hooks": {"PreToolUse": [HookMatcher(matcher=None, hooks=[no_refunds])]}
        },
    )
    tools = [ToolSpec(n, n, {"type": "object"}) for n in ("look_up", "refund")]
    try:
        out = await runner.run(
            SegmentInput(
                session_id=str(uuid.uuid4()),
                prompt="Refund A-1.",
                tools=tools,
                transcript=[],
            ),
            1,
        )
    finally:
        api.stop()
    assert out.is_error and out.error is not None
    assert "deny on mcp__durable__refund" in out.error
    assert out.deferred is None and out.siblings == []


# ---- the Worker's disk, options and the stand-in ----


async def test_a_session_already_on_disk_is_left_alone(tmp_path: Path) -> None:
    """Claude Code refuses a session id that already has a transcript; the runner
    used to remove that transcript afterwards as if it were its own."""
    api = start_with_policy(shell_policy)
    runner = make_runner(tmp_path, api)
    sid = str(uuid.uuid4())
    transcript, folder = runner._local_copy(sid)  # type: ignore[reportPrivateUsage]
    folder.mkdir(parents=True)
    entry = {
        "type": "user",
        "uuid": str(uuid.uuid4()),
        "parentUuid": None,
        "sessionId": sid,
        "message": {"role": "user", "content": "a session of someone else"},
    }
    transcript.write_text(json.dumps(entry) + "\n")
    (folder / "kept.txt").write_text("kept")
    try:
        try:
            await runner.run(
                SegmentInput(session_id=sid, prompt="run: echo hi", tools=TOOLS),
                1,
            )
        except Exception:  # the engine refuses the id; whatever it does, the files stay
            pass
    finally:
        api.stop()
    assert transcript.exists() and (folder / "kept.txt").exists()


def test_file_checkpointing_is_refused(tmp_path: Path) -> None:
    (tmp_path / "work").mkdir()
    with pytest.raises(ValueError, match="enable_file_checkpointing"):
        ClaudeAgentSdkRunner(
            cwd=str(tmp_path / "work"),
            env={"ANTHROPIC_API_KEY": "x"},
            extra_options={"enable_file_checkpointing": True},
        )


def test_the_stand_in_refuses_a_body_that_is_not_json() -> None:
    stand_in = _stand_in.StandInModel()
    request = urllib.request.Request(
        f"{stand_in.base_url}/v1/messages", data=b"not json", method="POST"
    )
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(request, timeout=10)  # noqa: S310 (local stand-in)
    assert err.value.code == 400 and stand_in.requests == 1


async def test_a_tool_step_on_a_worker_set_up_the_other_way_is_retried(
    tmp_path: Path,
) -> None:
    """Like a segment, a tool step that lands on a Worker that keeps conversations
    elsewhere fails in a way Temporal retries, until a matching Worker takes it."""
    api = start_with_policy(shell_policy)
    held = make_runner(tmp_path, api)
    stored = make_runner(
        tmp_path, api, session_store=FileSessionStore(tmp_path / "sessions")
    )
    try:
        out = await pause_at_bash(held, "echo hi")
        with pytest.raises(RuntimeError, match="kept in its Workflow.*Retrying"):
            await stored.run_tool_step(step_for(out), 1)
        no_conversation = step_for(out)
        no_conversation.transcript = None
        with pytest.raises(RuntimeError, match="session store this runner.*Retrying"):
            await held.run_tool_step(no_conversation, 1)
        with pytest.raises(ApplicationError):  # a real mismatch is final
            wrong = step_for(out)
            wrong.checkpoint = str(uuid.uuid4())
            await held.run_tool_step(wrong, 1)
    finally:
        api.stop()
