"""The real runner: drives the Claude Agent SDK and its bundled Claude Code engine.

How one segment works:

1. Durable tools are declared to Claude as in-process SDK MCP tools.
2. A settings-file PreToolUse command hook answers "defer" when Claude calls one, or a
   Claude Code tool in ``tool_activities``. The run stops and the SDK returns
   ``ResultMessage.deferred_tool_use``. The Workflow runs the call as a Temporal
   Activity: a durable tool's own Activity, or a tool step (``run_tool_step``).
3. The next segment resumes the session and sends the stored result as a normal
   ``tool_result`` message for that ``tool_use_id``. Claude continues and can pause
   again. In a segment, the hook keeps answering "defer" (never "allow") for these
   calls.

A tool step resumes a copy of the session that ends where it paused at a Claude Code
call, and the hook allows exactly that call, so Claude gets Claude Code's own result.
The engine's model calls after the tool go to a stand-in on 127.0.0.1
(``_stand_in``), and the command still sees the Worker's own environment.

Checkpoints make retries clean. After a segment, the runner reads the session back
from the session store and returns where the next segment must continue (the last
transcript entry, or the paused call's deferral marker after parallel calls) as the
segment's checkpoint; the Workflow stores it with the segment's result.

Where the conversation lives:

- By default (no ``session_store``), in the Workflow. The segment reads the committed
  transcript with a Query on its own Workflow, seeds a fresh in-memory session store
  with it (up to the checkpoint), runs, and returns what changed
  (``transcript_keep`` and ``transcript_add``); the Workflow splices it in. Every
  attempt starts from what the Workflow committed, so a retry can never build on
  an unfinished attempt, and any Worker can run any step with no shared storage.
- With a ``session_store``, in the store. Reading the checkpoint back also proves
  the turn reached the store, so a segment only commits what every Worker can
  resume. A segment that runs again (a retry after a crash or a timeout, or the
  first segment after a failed task) cannot trust what an unfinished attempt wrote,
  so it continues in a copy of the session that ends at the checkpoint
  (``fork_session_via_store``), and Claude decides again from there.

Why not "resume and let the hook allow the deferred call"? On that auto-resume path
the engine ignores a later "defer" when the resume also sends a user message, and the
SDK's resume always does (reported: anthropics/claude-code#97196).

Why a copy, and not ``resume_session_at``? Tested with Claude Code 2.1.281: resuming
at the checkpoint of a session that holds a later attempt's entries (in place or with
``fork_session``) makes the engine treat the paused call as interrupted: it answers
the call with an error placeholder and drops the delivered result. In a copy that
ends at the checkpoint, the paused call resumes normally. The same drop happens when a
user row follows the paused call (anthropics/claude-code#97358), which is why the
checkpoint after parallel calls is the deferral marker (see ``_resume_point``).

Parallel calls: the engine keeps one paused call per run, so the hook defers the first
call of a message that runs as an Activity and denies the calls after it. The segment
reports the denied durable calls (``siblings``); the Workflow runs them with the
paused one, and the next segment puts their results where the engine resumes (see
``_deliver``), so Claude sees every call of the message with its result. A denied
Claude Code call keeps its denial, and Claude calls it again.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import copy
import dataclasses
import fnmatch
import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import unicodedata
import uuid
import warnings
from collections.abc import AsyncGenerator, AsyncIterator, Collection
from pathlib import Path
from typing import Any, Literal, cast

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    InMemorySessionStore,
    MirrorErrorMessage,
    ResultError,
    ResultMessage,
    SessionStore,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    UserMessage,
    create_sdk_mcp_server,
    fork_session_via_store,
    project_key_for_directory,
    query,
    tool,
)
from temporalio import activity
from temporalio.exceptions import ApplicationError

from ._conversation import external_storage_on, read_conversation, too_large
from ._defer_hook import ANSWERED, REASON_KEYS, STOPPED, WORKER_LOCK, denial_name
from ._defer_hook import NOT_RUN as NOT_RUN_REASON
from ._events import emit
from ._launcher import CHECK as LAUNCHER_CHECK
from ._launcher import WORKER_PID
from ._models import (
    DeferredCall,
    SegmentInput,
    SegmentOutput,
    ToolOutcome,
    ToolSpec,
    ToolStepInput,
)
from ._stand_in import StandInModel

ENV_AUTH = (
    "ANTHROPIC_API_KEY",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
)
"""Logins that survive resumes.

A Claude app login (Keychain OAuth) does not: when the SDK resumes a session (every
segment after the first) it copies the login without its refresh token, so once the
short-lived access token expires, resumed segments fail with "OAuth session expired".
"""

SERVER = "durable"
PREFIX = f"mcp__{SERVER}__"
ONE_TOOL_HINT = "Call at most one tool per message, then wait for its result before calling another."
"""Asks Claude for one call per message (``one_tool_at_a_time``; off by default)."""

PAUSE_CONTRACT = (
    "Durable tools never run inside the engine (the in-engine tool only returns an "
    "error), so nothing ran outside Temporal. This segment stops instead of "
    "continuing. Durable tools cannot be called from subagents; otherwise, use a "
    "Claude Code version that passes this plugin's test suite."
)

MIN_BUFFER_BYTES = 64 * 1024 * 1024
"""Smallest limit for one message from the engine (the SDK's default is 1 MiB).

The engine echoes each delivered tool result as one JSON line, so a result over the
SDK's default could never reach Claude.
"""

STEP_PROVIDER_ENV = {
    # Provider settings in settings files (user, project, .claude.json, managed) no
    # longer apply, so they cannot send a tool step's model calls elsewhere.
    "CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST": "1",
    "ANTHROPIC_AUTH_TOKEN": "",
    "CLAUDE_CODE_OAUTH_TOKEN": "",
    "ANTHROPIC_UNIX_SOCKET": "",
    "CLAUDE_CODE_API_BASE_URL": "",
    "CLAUDE_CODE_USE_BEDROCK": "",
    "CLAUDE_CODE_USE_VERTEX": "",
    "CLAUDE_CODE_USE_FOUNDRY": "",
    "CLAUDE_CODE_USE_ANTHROPIC_AWS": "",
    "CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD": "",
    "CLAUDE_CODE_USE_MANTLE": "",
    "CLAUDE_CODE_USE_GATEWAY": "",
}
"""A tool step talks to the local stand-in model only, whatever provider the Worker uses.

``ANTHROPIC_API_KEY`` is the stand-in's own key (``StandInModel.key``).
"""

SAVED_OUTPUT_TAIL = 4096
"""Bytes of the end of an output Claude Code saved to a file, added to the preview."""

ENGINE_ENV = {
    "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
    "CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR": "1",
}
"""Set for every engine run.

With background tasks (for example a subagent running in the background), the engine
keeps working after it paused at a durable call. And every shell command starts in the
working directory: a ``cd`` would otherwise carry over inside one engine run but not
to the next one (each segment and tool step is a new run), so a later command could
run somewhere else than Claude expects.
"""

SHELL_HINT = (
    "Each shell command starts in the working directory: a `cd` does not carry over "
    "to the next command, so use absolute paths or `cd <dir> && <command>`."
)
"""Added to the system prompt when Bash or PowerShell is on: Claude Code's own tool
description says the working directory persists, which is not so here."""

_SHELL_TOOLS = ("Bash", "PowerShell")

DEFAULT_PERMISSION_MODE = "default"
"""Permission mode unless ``extra_options`` sets one.

Since Claude Code 2.1.285, a run with no permission mode uses auto mode when telemetry
is off or the provider is Bedrock, Vertex or Foundry. Auto mode asks the model whether
each tool call is safe, and blocked a subagent in this plugin's tests (claude-agent-sdk
0.2.163). Setting the mode keeps every engine version on the same rules: tools in
``allowed_tools`` run, others are refused.
"""

_RESERVED_OPTIONS = {
    "tools": "DurableClaudeAgent(builtin_tools=...)",
    "model": "DurableClaudeAgent(model=...) or ClaudeAgentSdkRunner(model=...)",
    "max_turns": "DurableClaudeAgent(max_turns=...)",
    "max_budget_usd": "ClaudeAgentSdkRunner(max_budget_usd=...)",
    "cwd": "ClaudeAgentSdkRunner(cwd=...)",
    "cli_path": "ClaudeAgentSdkRunner(cli_path=...)",
    "session_store": "ClaudeAgentSdkRunner(session_store=...)",
}
"""``extra_options`` keys the agent or the runner already sets, with where to set them."""

_MANAGED_OPTIONS = frozenset(
    {
        "settings",
        "strict_mcp_config",
        "resume",
        "session_id",
        "continue_conversation",
        "fork_session",
        "resume_session_at",
        "resume_drops_turn",
    }
)
"""``extra_options`` keys the plugin needs for pausing, resuming and checkpoints."""

_MANAGED_FLAGS = frozenset(
    {
        "settings",
        "resume",
        "session-id",
        "continue",
        "fork-session",
        "resume-session-at",
        "mcp-config",
        "strict-mcp-config",
        "system-prompt",
        "tools",
        "allowedTools",
        "allowed-tools",
        "model",
        "max-turns",
    }
)
"""Engine flags the plugin sets, refused in ``extra_options["extra_args"]`` too."""


def _check_extra_options(extra: dict[str, Any]) -> None:
    """Refuse ``extra_options`` that would replace what the plugin relies on.

    ``env``, ``mcp_servers``, ``allowed_tools`` and ``max_buffer_size`` are merged
    with the plugin's own values instead.

    Raises:
        ValueError: If ``extra`` sets a reserved option.
    """
    problems = [
        f"{key} (use {_RESERVED_OPTIONS[key]})"
        for key in sorted(extra)
        if key in _RESERVED_OPTIONS
    ] + [
        f"{key} (the plugin sets it to pause and resume sessions)"
        for key in sorted(extra)
        if key in _MANAGED_OPTIONS
    ]
    flags = extra.get("extra_args") or {}
    problems += [
        f"extra_args[{flag!r}] (the plugin sets that engine flag)"
        for flag in sorted(flags)
        if flag.lstrip("-") in _MANAGED_FLAGS
    ]
    if extra.get("enable_file_checkpointing"):
        problems.append(
            "enable_file_checkpointing (Claude Code cannot combine it with the session "
            "store every engine run uses)"
        )
    servers = extra.get("mcp_servers") or {}
    if not isinstance(servers, dict):
        problems.append("mcp_servers (pass a dict of servers; they are merged)")
    elif SERVER in servers:
        problems.append(f"mcp_servers[{SERVER!r}] (the durable tools use that name)")
    if problems:
        raise ValueError("extra_options cannot set: " + "; ".join(problems))


def _engine_cwd(directory: str) -> str:
    """The working directory as a process started in ``directory`` reports it.

    That is the path the engine derives its session key from: on macOS, the form
    stored on disk, which can be decomposed even when ``directory`` is not.
    """
    code = "import os, sys; sys.stdout.buffer.write(os.fsencode(os.getcwd()))"
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=directory,
        capture_output=True,
        check=True,
        timeout=60,
    ).stdout
    return os.fsdecode(out)


def _key_mismatch(directory: str) -> str | None:
    """How Claude Code and the SDK would store this directory's sessions under two keys.

    The SDK derives the key from the NFC form of the real path and counts code
    points; the engine uses its working directory as it is and counts UTF-16 units.
    They agree when the engine's path is exactly the SDK's and has no character
    outside the Basic Multilingual Plane. Otherwise no segment could find its own
    turn in the store, and every one would be retried.

    Returns:
        A description of the difference, or None if the keys agree.
    """
    ours = unicodedata.normalize("NFC", os.path.realpath(directory))
    engine = _engine_cwd(directory)
    if engine != ours:
        return f"the engine sees {engine!r}, the SDK uses {ours!r}"
    if any(ord(char) > 0xFFFF for char in engine):
        return "it has characters outside the Basic Multilingual Plane (such as emoji)"
    return None


MIN_ENGINE_VERSION = (2, 1, 273)
"""Oldest Claude Code engine that keeps every tool result (bundled in claude-agent-sdk 0.2.153).

Tested: when Claude calls two tools in one message, Claude Code 2.1.259 replaces the
result of the paused call with "[Tool result missing due to internal error]" in the
next request, so Claude asks for the same tool again and it would run twice.
"""


def _version(text: str) -> tuple[int, ...] | None:
    parts = text.split()[0].split(".") if text.strip() else []
    if len(parts) < 3 or not all(p.isdigit() for p in parts[:3]):
        return None
    return tuple(int(p) for p in parts[:3])


def _too_old(reported: str | None) -> bool:
    parsed = _version(reported or "")
    return parsed is not None and parsed < MIN_ENGINE_VERSION


def _too_old_output(session_id: str, reported: str) -> SegmentOutput:
    minimum = ".".join(map(str, MIN_ENGINE_VERSION))
    return SegmentOutput(
        session_id=session_id,
        is_error=True,
        error=(
            f"Claude Code {reported} is older than {minimum}. Older engines can drop a "
            "tool result when Claude calls several tools in one message; Claude then "
            "asks for the tool again, so it could run twice. Use claude-agent-sdk "
            f"0.2.153 or newer, or set cli_path to Claude Code {minimum} or newer. "
            "No durable tool ran."
        ),
    )


FINAL_RESULT_ERRORS = frozenset({"error_max_turns", "error_max_budget_usd"})
"""Result subtypes that running the segment again cannot fix."""


_FINAL_TERMINAL_REASONS = frozenset({"prompt_too_long", "image_error"})


def _final_api_error(err: ResultError) -> bool:
    """Whether the same request would be refused again however often it is sent.

    Final: the engine's own verdicts (prompt too long, image error), an invalid
    request (400), an unknown model (404) and a request too large (413): the model,
    tools and conversation come from the segment's input, which a retry reuses.
    Retried: a low credit balance, authentication and permissions (fixed outside the
    request), rate limits, overload and server errors.
    """
    if err.terminal_reason in _FINAL_TERMINAL_REASONS:
        return True
    if err.api_error_status in (404, 413):
        return True
    if err.api_error_status == 400:
        text = f"{err.result or ''} {' '.join(err.errors)} {err}".lower()
        return "credit balance" not in text
    return False


_TRANSCRIPT_TYPES = frozenset({"user", "assistant", "attachment", "system"})
_OPTIONAL_STORE_METHODS = (
    "list_sessions",
    "list_session_summaries",
    "delete",
    "list_subkeys",
)


def _is_transcript(entry: Any) -> bool:
    return (
        isinstance(entry, dict)
        and isinstance(entry.get("uuid"), str)
        and entry.get("type") in _TRANSCRIPT_TYPES
        and not entry.get("isSidechain")
    )


_COST_STATE = "cost-state"


def _is_cost_state(entry: Any) -> bool:
    """Whether ``entry`` is a saved cost total (see ``_without_cost_state``)."""
    return isinstance(entry, dict) and entry.get("type") == _COST_STATE


def _without_cost_state(entries: list[Any]) -> list[Any]:
    """The entries without the session's saved cost totals.

    Claude Code 2.1.277 and newer write the session's running cost at the end of a run
    (a ``cost-state`` entry) and start a resumed run from it, so the run's result
    reports the session's total instead of what the run spent, and the Workflow would
    count earlier steps again. Without the entry every engine reports each run's own
    cost (``max_budget_usd`` stays per run either way, tested on 2.1.287).
    """
    return [e for e in entries if not _is_cost_state(e)]


def _last_entry(entries: list[Any]) -> str | None:
    """The uuid of the session's last transcript entry: where the engine resumes."""
    for entry in reversed(entries):
        if _is_transcript(entry):
            return entry["uuid"]
    return None


def _resume_point(entries: list[Any], paused_call: str | None) -> str | None:
    """Where the next segment must continue after this one.

    Normally the last transcript entry. But when Claude sent several durable calls
    at once, the engine writes the denied calls' results after the paused call's
    deferral marker, and a session that ends there no longer resumes the paused
    call (tested: its delivered result is replaced with "[Tool result missing due
    to internal error]"). Then the checkpoint is the marker, so the next segment
    continues in a copy that ends at it.
    """
    leaf: str | None = None
    marker: str | None = None
    user_after_marker = False
    for entry in entries:
        if not _is_transcript(entry):
            continue
        leaf = entry["uuid"]
        attachment = entry.get("attachment") if entry["type"] == "attachment" else None
        if (
            paused_call is not None
            and isinstance(attachment, dict)
            and attachment.get("type") == "hook_deferred_tool"
            and attachment.get("toolUseID") == paused_call
        ):
            marker, user_after_marker = leaf, False
        elif marker is not None and entry["type"] == "user":
            user_after_marker = True
    return marker if marker is not None and user_after_marker else leaf


def _result_ids(entry: Any) -> list[str]:
    """The tool calls a transcript entry answers (a user entry with tool results)."""
    if not isinstance(entry, dict) or entry.get("type") != "user":
        return []
    content = (entry.get("message") or {}).get("content")
    if not isinstance(content, list):
        return []
    return [
        str(block.get("tool_use_id"))
        for block in content
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]


def _marker_of(entry: Any) -> str | None:
    """The paused call's id, if ``entry`` is a deferral marker."""
    if not isinstance(entry, dict) or entry.get("type") != "attachment":
        return None
    attachment = entry.get("attachment")
    if isinstance(attachment, dict) and attachment.get("type") == "hook_deferred_tool":
        return str(attachment.get("toolUseID"))
    return None


def _siblings(
    entries: list[Any], paused_call: str, denials: dict[str, str]
) -> list[DeferredCall]:
    """The other durable calls of the paused message, denied after the pause.

    Their denials come after the paused call's deferral marker, and the hook recorded
    each one as "not run" (``denials``): a call denied for any other reason, or by
    anything else, is never run.
    """
    marker = None
    for index, entry in enumerate(entries):
        if _marker_of(entry) == paused_call:
            marker = index
    if marker is None:
        return []
    not_run = REASON_KEYS[NOT_RUN_REASON]
    denied = {
        tid
        for entry in entries[marker + 1 :]
        for tid in _result_ids(entry)
        if denials.get(denial_name(tid)) == not_run
    }
    calls: list[DeferredCall] = []
    for entry in entries[:marker]:
        if not isinstance(entry, dict) or entry.get("type") != "assistant":
            continue
        for block in (entry.get("message") or {}).get("content") or []:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("id") in denied
                and str(block.get("name", "")).startswith(PREFIX)
                and all(c.id != block["id"] for c in calls)
            ):
                calls.append(
                    DeferredCall(
                        id=block["id"],
                        name=str(block["name"]).removeprefix(PREFIX),
                        input=dict(block.get("input") or {}),
                    )
                )
    return calls


def _deliver(
    entries: list[Any], checkpoint: str, results: dict[str, ToolOutcome]
) -> tuple[list[Any], set[str]] | None:
    """Put the results of the calls denied after a pause where the engine resumes.

    After a pause in a message with several calls, the session ends with the paused
    call's hook entries (the last is the deferral marker, the checkpoint), followed
    by the denials of the calls after it. Resuming at the marker would drop those
    calls: the engine strips tool calls whose results come after where it resumes.
    So the denials move before the paused call's hook entries, and the denials of
    durable calls get their real results. The engine's own links between entries
    stay as they are. Tested on Claude Code 2.1.273 and 2.1.287, with built-in and
    durable calls mixed in one message: Claude sees each call of the message with
    its result (a built-in call denied after the pause keeps the denial, and Claude
    calls it again).

    Returns:
        The entries to resume from, and the calls whose results they now hold; or
        None if the checkpoint is not a deferral marker followed by denials.
    """
    marker = next(
        (
            i
            for i, e in enumerate(entries)
            if isinstance(e, dict) and e.get("uuid") == checkpoint
        ),
        None,
    )
    paused = _marker_of(entries[marker]) if marker is not None else None
    if marker is None or paused is None:
        return None
    first = marker  # the paused call's hook entries end at the marker
    while first > 0:
        previous = entries[first - 1]
        attachment = previous.get("attachment") if isinstance(previous, dict) else None
        if not (isinstance(attachment, dict) and attachment.get("toolUseID") == paused):
            break
        first -= 1
    moved: list[Any] = []
    delivered: set[str] = set()
    for entry in entries[marker + 1 :]:
        ids = _result_ids(entry)
        if not ids or paused in ids:
            continue
        entry = copy.deepcopy(entry)
        for block in entry["message"]["content"]:
            outcome = results.get(str(block.get("tool_use_id")))
            if block.get("type") != "tool_result" or outcome is None:
                continue
            block["content"] = _result_content(outcome)
            if outcome.is_error:
                block["is_error"] = True
            else:
                block.pop("is_error", None)
            entry["toolUseResult"] = block["content"]
            delivered.add(str(block["tool_use_id"]))
        moved.append(entry)
    if not moved:
        return None
    return [*entries[:first], *moved, *entries[first : marker + 1]], delivered


def _seed(committed: list[Any], checkpoint: str) -> list[Any] | None:
    """The committed conversation up to the checkpoint, where the next run resumes.

    That is all of it, unless the checkpoint is the paused call's deferral marker
    (see ``_resume_point``): then the entries after the marker are left out.

    Returns:
        The entries, or None if the checkpoint is not in the conversation.
    """
    if _last_entry(committed) == checkpoint:
        return committed
    for index in range(len(committed) - 1, -1, -1):
        entry = committed[index]
        if _is_transcript(entry) and entry["uuid"] == checkpoint:
            return committed[: index + 1]
    return None


def _kept(committed: list[Any], entries: list[Any]) -> tuple[int, int]:
    """How much of the committed conversation the session still starts with.

    Saved cost totals do not count: the engine never resumes from one, and those an
    older version left in the conversation stay where they are (every seed and every
    store load leaves them out), so the step sends nothing again because of them.

    Returns:
        The number of committed entries to keep, and the number of the session's
        entries they cover.
    """
    i = j = 0
    while True:
        while i < len(committed) and _is_cost_state(committed[i]):
            i += 1
        while j < len(entries) and _is_cost_state(entries[j]):
            j += 1
        if i == len(committed) or j == len(entries):
            return i, j
        if committed[i] is not entries[j] and committed[i] != entries[j]:
            return i, j
        i += 1
        j += 1


def _attempt_session_id(session_id: str, attempt: int) -> str:
    """The id of a new session: a fresh one for each attempt.

    A failed attempt may have left a partial transcript, in the session store and
    in the engine's own folder.
    """
    if attempt == 1:
        return session_id
    return str(uuid.uuid5(uuid.UUID(session_id), f"attempt-{attempt}"))


class _SessionMoved(Exception):
    """The session no longer ends at the checkpoint (for example after a Workflow reset)."""


def _moved(err: BaseException) -> bool:
    seen: BaseException | None = err
    while seen is not None:
        if isinstance(seen, _SessionMoved):
            return True
        seen = seen.__cause__ or seen.__context__
    return False


class _GuardedStore:
    """Passes the session store to the SDK, without the saved cost totals.

    The engine loads every session it resumes through it, so it never starts from the
    session's running cost (see ``_without_cost_state``); the store itself keeps what
    the engine wrote. With a checkpoint, it also checks the session when it is
    resumed in place: that is right only while the session still ends at the
    checkpoint. The check rides on the load the SDK does anyway, so it costs nothing
    extra.
    """

    def __init__(self, inner: Any, session_id: str, checkpoint: str | None) -> None:
        self._inner = inner
        self._session_id = session_id
        self._checkpoint = checkpoint

    async def append(self, key: Any, entries: Any) -> None:
        await self._inner.append(key, entries)

    async def load(self, key: Any) -> Any:
        entries = await self._inner.load(key)
        if (
            entries
            and self._checkpoint is not None
            and key.get("session_id") == self._session_id
            and not key.get("subpath")
            and _last_entry(entries) != self._checkpoint
        ):
            raise _SessionMoved(
                f"Session {self._session_id} continued after checkpoint "
                f"{self._checkpoint}"
            )
        return _without_cost_state(entries) if entries else entries


def _guarded(inner: Any, session_id: str, checkpoint: str | None) -> Any:
    """``inner`` for the SDK, keeping only the optional methods it has."""
    namespace: dict[str, Any] = {}
    for name in _OPTIONAL_STORE_METHODS:
        # The rule the SDK applies: present, and not the protocol's default.
        present = getattr(inner, name, None) is not None
        if present and getattr(type(inner), name, None) is not getattr(
            SessionStore, name, None
        ):
            namespace[name] = _forward(name)
    cls = type("GuardedSessionStore", (_GuardedStore,), namespace)
    return cls(inner, session_id, checkpoint)


def _forward(name: str) -> Any:
    async def method(self: _GuardedStore, *args: Any, **kwargs: Any) -> Any:
        return await getattr(self._inner, name)(*args, **kwargs)  # type: ignore[reportPrivateUsage]

    method.__name__ = name
    return method


def _hook_entry() -> dict[str, Any]:
    """The PreToolUse command hook (a function so tests can simulate other engines).

    Exec form: Claude Code starts the program directly, with no shell, so no path
    needs quoting. (Through a shell, Git Bash on Windows drops the backslashes of a
    Windows path, and PowerShell, its fallback, needs other quoting.) The hook file
    is run as a plain script: it only needs the standard library, and importing
    this package would add about a second to every durable tool call. Isolated mode
    and no ``site``, like the launcher: nothing from the Worker's environment or its
    ``.pth`` files runs before each decision.
    """
    hook = Path(__file__).with_name("_defer_hook.py")
    return {
        "type": "command",
        "command": sys.executable,
        "args": ["-I", "-S", str(hook)],
    }


def _hook_folder() -> str:
    """A new folder with the settings file that registers the hook for every tool."""
    hook_dir = tempfile.mkdtemp(prefix="tca-hook-")
    settings = {
        "hooks": {
            "PreToolUse": [
                # Every tool: built-in calls after a pause are denied too.
                {"matcher": ".*", "hooks": [_hook_entry()]}
            ]
        }
    }
    Path(hook_dir, "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    return hook_dir


async def _stop_hooks_when_cancelled(hook_dir: str) -> None:
    """Once the segment Activity is cancelled (or timed out), the hook denies every call.

    The SDK gives the engine a few seconds to exit, and a cancel reaches the Worker
    only with a heartbeat, so the engine could otherwise still run a tool after the
    Workflow moved on.
    """
    await activity.wait_for_cancelled()
    try:
        Path(hook_dir, "stop").touch()
    except OSError:
        pass  # the folder is already gone, and the hook denies without it


def _hook_denials(hook_dir: str) -> dict[str, str]:
    """The calls the hook denied in this run: record name (``denial_name``) to reason key."""
    try:
        return {
            path.name: path.read_text(encoding="utf-8").strip()
            for path in Path(hook_dir, "denied").iterdir()
        }
    except OSError:
        return {}


class _Feed:
    """An engine's input: the messages for its turns, then, once ``close`` is called,
    its end.

    The SDK writes each message to the engine as it comes, and closes the engine's
    input when the stream ends; the engine then exits after its turn. A warm engine's
    input stays open between segments: the next segment's message goes in with
    ``put``.
    """

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self.put(messages)

    def put(self, messages: list[dict[str, Any]]) -> None:
        for message in messages:
            self._queue.put_nowait(message)

    def close(self) -> None:
        self._queue.put_nowait(None)

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        while (message := await self._queue.get()) is not None:
            yield message


async def _as_messages(prompt: Any) -> list[dict[str, Any]]:
    """An engine run's input as user messages.

    A text prompt becomes the message the one-shot query writes for it; the runner's
    own message stream is read as it is.
    """
    if isinstance(prompt, str):
        return [
            {
                "type": "user",
                "session_id": "",
                "message": {"role": "user", "content": prompt},
                "parent_tool_use_id": None,
            }
        ]
    return [message async for message in prompt]


def _agents_running(client: Any) -> bool:
    """Whether delegated agents the engine started still run (the SDK tracks them)."""
    return bool(getattr(getattr(client, "_query", None), "_inflight_tasks", None))


async def _turn_messages(
    client: Any,
    feed: _Feed,
    resumed: bool,
    *,
    stay: bool = False,
    answered: Collection[str] = (),
) -> AsyncGenerator[Any, None]:
    """An engine's messages until it exits. Its input ends once its own turn is over.

    The one-shot query ends the input at the first result that comes while no
    delegated agent runs. A resumed engine first repeats the result its session
    paused with, before its ``init`` message (tested on Claude Code 2.1.273 and
    2.1.288), so the input ended before the resumed turn even started: hooks and
    permission callbacks from ``extra_options``, which the engine asks over its
    input, were never asked, and Claude Code refused their calls. This ends the
    input at the turn's own result instead, once no agent runs: any result but
    one such repeat (a pause, before ``init``). Like the one-shot query, it reads
    on until the engine exits: the last result counts, and an engine that exits
    after an error result raises ``ResultError``.

    With ``stay``, an engine that paused cleanly stays running instead (it can stay
    warm, see ``warm_engines``): when its turn's result came with no error and no
    task started, at a call other than one of ``answered`` (whose results this run
    brings: a resumed engine repeats such a pause), this returns at that result, and
    the input stays open.
    """
    started = not resumed
    repeated = False
    tasks = False
    ending = False
    async for message in client.receive_messages():
        yield message
        if ending:
            continue
        if isinstance(message, SystemMessage):
            started = started or message.subtype == "init"
            tasks = tasks or message.subtype == "task_started"
        elif isinstance(message, ResultMessage):
            paused = message.deferred_tool_use
            if (
                not started
                and not repeated
                and paused is not None
                and not message.is_error
            ):
                repeated = True  # the pause the session ended with, repeated
                continue
            if (
                stay
                and not tasks
                and not message.is_error
                and paused is not None
                and paused.id not in answered
            ):
                return  # the engine waits, its input open, for the next segment
            if not _agents_running(client):  # each one wakes the engine again
                feed.close()
                ending = True


async def _disconnect(client: Any) -> None:
    """Stop an engine started through the streaming client, and clean up after it.

    The SDK closes the engine's input, waits for it to exit (and ends it if it does
    not), and removes its resume folder. If that wait is cut short (a second cancel,
    an event loop that closes), the engine still gets SIGTERM and the folder goes.
    """
    process = getattr(getattr(client, "_transport", None), "_process", None)
    resumed = getattr(client, "_materialized", None)
    try:
        with contextlib.suppress(Exception):
            await client.disconnect()
    finally:
        _stop_now(process, resumed)


EXIT_GRACE_SECONDS = 10.0
"""How long an engine that got SIGTERM may still write to its resume folder."""


def _stop_now(process: Any, resumed: Any) -> None:
    """End an engine without waiting: SIGTERM if it still runs; its resume folder goes.

    An engine that is still exiting can write a few lines to the folder after it is
    removed (session metadata, no conversation): it is removed again after
    ``EXIT_GRACE_SECONDS`` while the event loop runs, and when the Worker exits.
    """
    running = process is not None and process.returncode is None
    if running:
        with contextlib.suppress(OSError):
            process.terminate()
    if resumed is None:
        return
    folder = str(resumed.config_dir)
    shutil.rmtree(folder, ignore_errors=True)
    if running:
        with contextlib.suppress(RuntimeError):  # no event loop runs
            asyncio.get_running_loop().call_later(
                EXIT_GRACE_SECONDS, shutil.rmtree, folder, True
            )
        atexit.register(shutil.rmtree, folder, True)


async def _engine_messages(
    options: dict[str, Any], messages: list[dict[str, Any]], resumed: bool
) -> AsyncGenerator[Any, None]:
    """Start an engine, send it ``messages``, and yield what it says until it exits.

    See ``_turn_messages`` for when its input ends.
    """
    feed = _Feed(messages)
    client = ClaudeSDKClient(ClaudeAgentOptions(**options))
    try:
        await client.connect(feed.stream())
        async for message in _turn_messages(client, feed, resumed):
            yield message
    finally:
        await _disconnect(client)


async def _connected(
    client: Any, feed: _Feed, resumed: bool, answered: Collection[str]
) -> AsyncGenerator[Any, None]:
    """A new engine's messages, for an engine that may stay warm (``stay``).

    The runner, not this, ends the engine: it may keep it for the next segment.
    """
    await client.connect(feed.stream())
    async for message in _turn_messages(
        client, feed, resumed, stay=True, answered=answered
    ):
        yield message


@dataclasses.dataclass
class _WarmEngine:
    """An engine that paused and stays running for the session's next segment.

    The durable boundary is still the pause: the Workflow committed the checkpoint
    before the engine waits here, so a lost warm engine only means a cold start that
    resumes from the checkpoint.
    """

    client: Any
    """The SDK's streaming client, connected to the engine."""
    feed: _Feed
    """The engine's input: the next segment's message goes here."""
    session_id: str
    checkpoint: str
    """Where the engine paused: the next segment must continue exactly there."""
    paused: str
    """The call it paused at: the next segment must bring exactly its result."""
    shape: str
    """The agent's tools and settings: the next segment must have the same."""
    hook_dir: str
    lock: int | None
    store: Any
    """The session store the engine mirrors into."""
    cost: float
    """The engine's running cost total, to report each segment's own cost."""
    version: str
    ran_inside: list[str]
    violations: list[str]
    buffer: int
    """The engine's ``max_buffer_size``: a bigger result needs a new engine."""
    timer: asyncio.TimerHandle | None = None


def _forget_local_copy(paths: tuple[Path, ...]) -> None:
    """Remove the engine's own copy of a new session from its config folder.

    Claude Code writes a new session's transcript (and tool outputs too large to show
    Claude in full) under ``<config folder>/projects``; resumed sessions run in a
    temporary folder the SDK removes. With the conversation in the Workflow, that
    copy is never read again, so it does not stay on the Worker's disk. Only a copy
    the engine created is removed (see ``_run_held``), once the engine ended (such an
    engine never stays warm: see ``_parks``).
    """
    if not paths:
        return
    transcript, folder = paths
    try:
        transcript.unlink(missing_ok=True)
    except OSError:
        pass  # best effort, for example a file still open on Windows
    shutil.rmtree(folder, ignore_errors=True)


def _next_hook_turn(hook_dir: str, injected: dict[str, Any]) -> None:
    """Prepare a warm engine's hook folder for its next turn (a new segment).

    The hook allows one paused call per segment and records its denials per segment;
    the calls answered since the engine started are in ``answered`` (its environment
    is fixed when it starts).
    """
    with contextlib.suppress(OSError):
        os.remove(os.path.join(hook_dir, "paused_call"))
    shutil.rmtree(os.path.join(hook_dir, "denied"), ignore_errors=True)
    path = os.path.join(hook_dir, ANSWERED)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("".join(f"{call}\n" for call in injected))


_ending: set[asyncio.Task[None]] = set()
"""Warm engines being ended in the background (kept until done)."""


def _engine_alive(client: Any) -> bool:
    """Whether the engine behind a streaming client still runs."""
    process = getattr(getattr(client, "_transport", None), "_process", None)
    return process is not None and process.returncode is None


# ---- the engine ends with its Worker ----

if sys.platform == "win32":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    def _lock_file(fd: int) -> None:
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _unlock_file(fd: int) -> None:
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_ulonglong)
            for name in (
                "ReadOperationCount",
                "WriteOperationCount",
                "OtherOperationCount",
                "ReadTransferCount",
                "WriteTransferCount",
                "OtherTransferCount",
            )
        ]

    class _BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimits),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    _JOB: list[Any] = []
    """The job's handle, never closed: Windows closes it when this process ends."""
    _KERNEL32: list[Any] = []

    def _kernel32() -> Any:
        if _KERNEL32:
            return _KERNEL32[0]
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.IsProcessInJob.restype = wintypes.BOOL
        kernel32.IsProcessInJob.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.BOOL),
        ]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        _KERNEL32.append(kernel32)
        return kernel32

    def _in_worker_job(pid: int) -> bool:
        """Whether process ``pid`` is in the Worker's job."""
        if not _JOB:
            return False
        kernel32 = _kernel32()
        process = kernel32.OpenProcess(0x1000, False, pid)  # query limited information
        if not process:
            return False
        try:
            inside = wintypes.BOOL()
            return bool(
                kernel32.IsProcessInJob(process, _JOB[0], ctypes.byref(inside))
                and inside.value
            )
        finally:
            kernel32.CloseHandle(process)

    def _worker_job() -> str | None:
        """Create this Worker process's job object, once.

        Its handle stays open for the life of this process: when the process ends,
        however it ends, Windows closes it and ends every process in the job
        (``KILL_ON_JOB_CLOSE``), with the processes they started (those start in the
        job).

        Returns:
            What went wrong, or None.
        """
        if _JOB:
            return None
        kernel32 = _kernel32()
        job = kernel32.CreateJobObjectW(None, None)
        info = _ExtendedLimits()
        # Kill on close; breakaway allowed, for programs that ask for it.
        info.BasicLimitInformation.LimitFlags = 0x2000 | 0x800
        if not job or not kernel32.SetInformationJobObject(
            job, 9, ctypes.byref(info), ctypes.sizeof(info)
        ):
            return (
                f"cannot create a job object (Windows error {ctypes.get_last_error()})"
            )
        _JOB.append(job)
        return None

    def _end_with_worker(pid: int) -> str | None:
        """Put process ``pid`` in the job that ends it when this Worker process ends.

        The Worker process and its other child processes are not in it.

        Returns:
            What went wrong, or None (the runner decides, see ``engine_cleanup``).
        """
        problem = _worker_job()
        if problem is not None:
            return problem
        kernel32 = _kernel32()
        if _in_worker_job(pid):
            return None
        # Query, quota, terminate.
        process = kernel32.OpenProcess(0x1000 | 0x0100 | 0x0001, False, pid)
        if not process:
            return None  # it has ended
        try:
            if kernel32.AssignProcessToJobObject(_JOB[0], process):
                return None
            error = ctypes.get_last_error()
            code = wintypes.DWORD()
            if kernel32.GetExitCodeProcess(process, ctypes.byref(code)) and (
                code.value != 259  # STILL_ACTIVE
            ):
                return None  # it ended (an engine that failed at its start)
            return f"cannot put Claude Code in the job object (Windows error {error})"
        finally:
            kernel32.CloseHandle(process)

else:
    import fcntl

    def _lock_file(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock_file(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)

    def _worker_job() -> str | None:
        """Nothing to do here: on Linux and macOS the launcher does it (``_launcher``)."""
        return None

    def _end_with_worker(pid: int) -> str | None:
        """Nothing to do here: on Linux and macOS the launcher does it (``_launcher``)."""
        del pid
        return None


_warned: set[str] = set()


def _warn_once(message: str, key: str) -> None:
    """Warn the first time something goes wrong in this process (one ``key`` each)."""
    if key not in _warned:
        _warned.add(key)
        warnings.warn(message, stacklevel=3)


CLEANUP_UNAVAILABLE = "EngineCleanupUnavailable"
"""Type of the ``ApplicationError`` of a step stopped because its engine could not be
made to end with the Worker process (``engine_cleanup="required"``)."""


def _cleanup_problem(problem: str, *, required: bool, key: str) -> None:
    """Stop the step (``engine_cleanup="required"``), or warn once (``"best_effort"``).

    Args:
        problem: What went wrong, and what it means for a Worker process that dies.
        required: Whether the runner requires the engine to end with its Worker.
        key: One warning per key and Worker process.

    Raises:
        ApplicationError: If ``required`` (type ``CLEANUP_UNAVAILABLE``). Retryable:
            another Worker, or this one once fixed, can run the step.
    """
    message = f"temporalio.claude_agent_sdk: {problem}"
    if required:
        raise ApplicationError(
            f'{message} The step stopped, because engine_cleanup is "required" '
            '(engine_cleanup="best_effort" goes on anyway, with a warning).',
            type=CLEANUP_UNAVAILABLE,
        )
    _warn_once(
        f'{message} Going on anyway, because engine_cleanup is "best_effort".',
        key=key,
    )


def _job_problem(problem: str) -> str:
    """``problem`` with what it means: the engine would outlive a dead Worker."""
    return (
        f"{problem}: Claude Code would not end with this Worker process, and would "
        "finish its current turn if the process died."
    )


def _child_pids() -> set[int]:
    """The processes the SDK has started in this Worker process.

    It reads the SDK's own list (``_ACTIVE_CHILDREN``); a test checks that the SDK
    still keeps it.
    """
    try:
        from claude_agent_sdk._internal.transport import subprocess_cli

        children = list(subprocess_cli._ACTIVE_CHILDREN)
    except Exception:
        return set()
    return {
        pid
        for pid in (getattr(c, "pid", None) for c in children)
        if isinstance(pid, int)
    }


def _engines_end_with_worker() -> dict[int, str]:
    """Windows: put the Claude Code processes the SDK has started in the Worker's job.

    Every one is tried (``_child_pids``).

    Returns:
        The processes that could not join it, with what went wrong (as
        ``_job_problem`` words it); 0 stands for all of them. Empty elsewhere.
    """
    if sys.platform != "win32":
        return {}
    failed: dict[int, str] = {}
    try:
        for pid in _child_pids():
            problem = _end_with_worker(pid)
            if problem is not None:
                failed[pid] = _job_problem(problem)
    except Exception as err:
        failed[0] = _job_problem(f"cannot put Claude Code in a job object ({err!r})")
    return failed


_JOB_POLL = sys.platform == "win32"
"""Whether ``_JobWatch`` looks while the engine starts (tests set it elsewhere too)."""


class _JobWatch:
    """Windows: a step's engine joins the Worker's job object, or ``engine_cleanup``
    decides. Elsewhere the launcher did it before the engine started.

    It looks every 5 ms while the engine starts, on the Worker's event loop, so the
    engine is in the job within milliseconds, normally long before it starts a
    process of its own (an MCP server, a command); a process it starts before it
    joins (on a loop blocked for longer) stays outside the job. If this step's
    engine cannot join it and cleanup is required, the engine ends at once: the hook
    denies every call from then on, and the engine is terminated, not asked to
    finish its turn. (It cannot tell its engine from one that another step starts at
    the same moment, so such an engine's failure stops this step too; a job that
    fails, fails for every engine of the Worker.)
    """

    def __init__(self, hook_dir: str, *, required: bool) -> None:
        self._hook_dir = hook_dir
        self._required = required
        self._before = _child_pids()  # other steps' engines: not this step's
        self.problems: list[str] = []
        self._poll: asyncio.Future[None] | None = None
        if _JOB_POLL:
            self._poll = asyncio.ensure_future(self._polling())

    async def _polling(self) -> None:
        while True:
            self._check()
            await asyncio.sleep(0.005)

    def _check(self) -> None:
        failed = {
            pid: problem
            for pid, problem in _engines_end_with_worker().items()
            if pid not in self._before
        }
        for problem in failed.values():
            if problem not in self.problems:
                self.problems.append(problem)
        if failed and self._required:
            with contextlib.suppress(OSError):
                Path(self._hook_dir, "stop").touch()
            for pid in failed:
                if pid > 0:
                    with contextlib.suppress(OSError):
                        os.kill(pid, signal.SIGTERM)  # TerminateProcess on Windows

    def stop(self) -> None:
        if self._poll is not None:
            self._poll.cancel()

    def joined(self) -> None:
        """At the engine's first message, or when it ended without one.

        Raises:
            ApplicationError: If this step's engine could not join the job and
                cleanup is required.
        """
        self.stop()
        self._check()
        if self.problems:
            _cleanup_problem(self.problems[0], required=self._required, key="job")


def _hold_worker_lock(hook_dir: str, *, required: bool) -> int | None:
    """Lock ``<hook folder>/worker.lock`` for this run: the hook's sign that the Worker lives.

    The hook can take the lock only once this process is gone (the operating system
    releases it), and then denies every call. Python opens files non-inheritable, so
    the engine never holds the lock. Where locks do not work (some network or FUSE
    file systems), ``engine_cleanup`` decides: the step stops (``required``), or the
    file goes, the hook takes the Worker to be there, and a warning says so.

    Returns:
        The file descriptor to pass to ``_release_worker_lock``, or None.

    Raises:
        ApplicationError: If locks do not work and ``required``.
    """
    path = os.path.join(hook_dir, WORKER_LOCK)
    fd: int | None = None
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        _lock_file(fd)
        return fd
    except OSError as err:
        if fd is not None:
            os.close(fd)
        with contextlib.suppress(OSError):
            os.remove(path)
        _cleanup_problem(
            f"cannot lock files in {tempfile.gettempdir()} ({err.strerror or err}): "
            "without the lock, the hook cannot tell that a dead Worker's engine "
            "should stop. Use a temporary folder where file locks work (for example, "
            "set TMPDIR to a folder on a local disk).",
            required=required,
            key="lock",
        )
        return None


def _release_worker_lock(fd: int | None) -> None:
    if fd is None:
        return
    try:
        _unlock_file(fd)
    except OSError:
        pass  # closing releases it too
    os.close(fd)


_LAUNCHER = Path(__file__).with_name("_launcher.py")
_launch_dir: list[str] = []
_launch_scripts: dict[str, str | None] = {}
"""Engine path to its checked launcher script, or None when the script cannot run."""


def _private_folder(path: str) -> bool:
    """Whether ``path`` is still the private folder this process made."""
    try:
        info = os.lstat(path)
    except OSError:
        return False
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
        return False
    getuid = getattr(os, "getuid", None)
    return getuid is None or info.st_uid == getuid()


def _sh_quote(text: str) -> str:
    """``text`` as one word for ``/bin/sh``."""
    return "'" + text.replace("'", "'\"'\"'") + "'"


def _write_launch_script(engine: str) -> str:
    """A ``/bin/sh`` script that starts ``engine`` through the launcher (``_launcher``).

    Isolated mode and no ``site``: the launcher needs only the standard library.
    The SDK's version probe before every start (``-v`` alone) goes straight to the
    engine, which prints its version and exits: no Python in between, about 20 ms
    of every step. Written to a new name and moved in place, so a run never sees
    half a script.
    """
    if not _launch_dir or not _private_folder(_launch_dir[0]):
        # A cleaner of temporary files may remove an old one: never write into a
        # folder of that name that someone else may have made since.
        folder = tempfile.mkdtemp(prefix="tca-launch-")
        atexit.register(shutil.rmtree, folder, True)
        _launch_dir[:] = [folder]
    path = Path(_launch_dir[0], hashlib.sha256(engine.encode()).hexdigest()[:16])
    command = " ".join(
        _sh_quote(part) for part in (sys.executable, "-I", "-S", str(_LAUNCHER), engine)
    )
    probe = f'if [ "$#" -eq 1 ] && [ "$1" = -v ]; then exec {_sh_quote(engine)} -v; fi'
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}")
    temporary.write_text(f'#!/bin/sh\n{probe}\nexec {command} "$@"\n', encoding="utf-8")
    temporary.chmod(0o700)
    os.replace(temporary, path)
    return str(path)


async def _launch_problem(script: str) -> str | None:
    """Why ``script`` cannot start the launcher here, or None if it can."""
    try:
        proc = await asyncio.create_subprocess_exec(
            script,
            LAUNCHER_CHECK,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except OSError as err:
        return err.strerror or str(err)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), 30)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return "no answer in 30 seconds"
    if proc.returncode == 0 and out.strip() == b"ok":
        return None
    return f"exit status {proc.returncode}"


def _command_env(
    env: dict[str, str],
    hook_dir: str,
    cwd: str | None,
    overrides: dict[str, str],
    extra: dict[str, str],
) -> dict[str, str]:
    """The engine's environment, with a file that gives each command the Worker's own.

    Claude Code runs ``CLAUDE_ENV_FILE`` before each Bash command, in the command's
    shell. This file puts back what the Worker had for every variable the engine
    gets in ``overrides`` (the values travel in ``TCA_KEEP_*`` variables, never in
    the file), then runs a ``CLAUDE_ENV_FILE`` of your own, as Claude Code would,
    and removes every ``TCA_*`` variable: commands see none of the plugin's own.

    Args:
        env: The engine's environment so far.
        hook_dir: The step's folder, where the file goes.
        cwd: The engine's working directory (a relative ``CLAUDE_ENV_FILE`` of your
            own is found from there, as Claude Code finds it).
        overrides: Variables the engine gets but commands must not.
        extra: The plugin's own variables to add for the engine (and its hook).

    Returns:
        The engine's environment.
    """
    before = {**os.environ, **env}
    path = str(Path(hook_dir, "command_env.sh"))
    overrides = {**overrides, "CLAUDE_ENV_FILE": path}
    keep: dict[str, str] = {}
    lines = [
        "# temporalio.claude_agent_sdk: give the command the Worker's own environment."
    ]
    for name in overrides:
        if before.get(name) is None:
            lines.append(f"unset {name}")
        else:
            keep[f"TCA_KEEP_{name}"] = before[name]
            lines.append(f'export {name}="$TCA_KEEP_{name}"')
    user_file = before.get("CLAUDE_ENV_FILE")
    if user_file:
        user_file = os.path.join(cwd or os.getcwd(), user_file)
        lines += [
            'if [ -f "$TCA_USER_ENV_FILE" ] && [ -r "$TCA_USER_ENV_FILE" ]; then',
            '  . "$TCA_USER_ENV_FILE"',
            "fi",
        ]
    result = {
        **env,
        **overrides,
        **keep,
        **({"TCA_USER_ENV_FILE": user_file} if user_file else {}),
        **extra,
    }
    # Names a shell takes (a stray name in the Worker's environment would print an
    # error into every command's output).
    hidden = sorted(
        n
        for n in {*os.environ, *result}
        if n.startswith("TCA_") and n.isidentifier() and n.isascii()
    )
    lines.append("unset " + " ".join(hidden))
    # Git Bash runs it on Windows too: LF line endings on every platform.
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return result


def _says_only(content: Any, reason: str) -> bool:
    """Whether a tool result is the hook's denial ``reason`` and nothing else.

    Claude Code records the reason as it is (2.1.273), or after
    ``PreToolUse:<tool> hook error: `` (2.1.287).
    """
    text = _text(content).strip() if content is not None else ""
    if text == reason:
        return True
    head, sep, rest = text.partition(" hook error: ")
    return bool(sep) and head.startswith("PreToolUse:") and rest == reason


def _hook_said_stopped(message: UserMessage) -> bool:
    """Whether a tool result in ``message`` is the hook's "step stopped" denial.

    The hook writes no record when it cannot see its folder, so this matches the
    denial itself: an error that is exactly the reason, never a tool's output that
    merely contains it.
    """
    if isinstance(message.content, str):
        return False
    return any(
        isinstance(block, ToolResultBlock)
        and bool(block.is_error)
        and _says_only(block.content, STOPPED)
        for block in message.content
    )


def _runs_as_activity(name: str, patterns: list[str]) -> bool:
    """Whether a Claude Code tool runs as an Activity: it matches ``tool_activities``.

    ``name`` can also be a permission rule, such as ``Bash(npm *)``.
    """
    tool = name.split("(", 1)[0].strip()
    return any(fnmatch.fnmatchcase(tool, p) for p in patterns if p)


def _decision_of(output: Any) -> str | None:
    """The decision a PreToolUse hook's output makes, or None if it makes none."""
    if not isinstance(output, dict):
        return None
    specific = output.get("hookSpecificOutput")
    if isinstance(specific, dict) and specific.get("permissionDecision"):
        return str(specific["permissionDecision"])
    if output.get("decision") in ("block", "approve"):
        return str(output["decision"])
    if output.get("continue") is False:
        return "stop"
    return None


def _guard_hooks(hooks: Any, decides_alone: Any, violations: list[str]) -> Any:
    """Wrap ``extra_options["hooks"]`` so they cannot decide on calls the plugin decides.

    A ``PreToolUse`` callback that returns a decision for a tool that runs as an
    Activity (a durable tool, or one of ``tool_activities``) would override the
    plugin's pause or approval: its decision is dropped, recorded in ``violations``,
    and the step then fails (see ``_hook_violation``).
    """
    if not isinstance(hooks, dict) or not hooks.get("PreToolUse"):
        return hooks

    def wrap(callback: Any) -> Any:
        async def guarded(input_data: Any, tool_use_id: Any, context: Any) -> Any:
            output = await callback(input_data, tool_use_id, context)
            name = str((input_data or {}).get("tool_name") or "")
            decision = _decision_of(output)
            if decision is not None and decides_alone(name):
                violations.append(f"{decision} on {name}")
                return {}
            return output

        return guarded

    matchers = [
        dataclasses.replace(m, hooks=[wrap(cb) for cb in m.hooks])
        if isinstance(m, HookMatcher)
        else m
        for m in hooks["PreToolUse"]
    ]
    return {**hooks, "PreToolUse": matchers}


def _hook_violation(violations: list[str]) -> str | None:
    if not violations:
        return None
    return (
        "A PreToolUse hook in extra_options decided on a call that runs as its own "
        f"Activity ({', '.join(sorted(set(violations)))}). The Workflow decides on "
        "those calls (needs_approval, tool_approvals), so make the hook return no "
        "decision for durable tools and the tools in tool_activities. This step "
        "stopped before any such call ran."
    )


def _as_outcome(value: Any) -> ToolOutcome:
    if isinstance(value, ToolOutcome):
        return value
    return ToolOutcome(
        content=value.get("content"), is_error=bool(value.get("is_error"))
    )


_SAVED_OUTPUT = re.compile(
    r"<persisted-output>\r?\nOutput too large \([^)\r\n]*\)\. "
    r"Full output saved to: ([^\r\n]+)\r?\n"
)
"""How Claude Code's preview of a large output starts, naming the file with the rest."""


def _saved_output_tail(content: Any, project_key: str, session_id: str) -> str | None:
    """The end of an output Claude Code saved to a file in this step, if it did.

    Read it while the engine runs: the file is in the folder of the resumed session,
    ``<temp>/claude-resume-*/projects/<project_key>/<session_id>/tool-results``, which
    the SDK removes. Any other path is ignored: the text names it, and a tool's output
    (a command's, or an MCP tool's from an issue body) can contain such text, so a
    file anywhere else is never read into the conversation.
    """
    found = _SAVED_OUTPUT.match(content) if isinstance(content, str) else None
    if found is None:
        return None
    path = Path(found.group(1).strip())
    layout = path.parts[-6:-1]
    expected = ("projects", project_key, session_id, "tool-results")
    if len(layout) != 5 or not layout[0].startswith("claude-resume-"):
        return None
    if layout[1:] != expected or path.suffix != ".txt":
        return None
    try:
        if not stat.S_ISREG(os.lstat(path).st_mode):
            return None  # not a link to a file elsewhere
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - SAVED_OUTPUT_TAIL))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None


def _result_content(outcome: ToolOutcome) -> Any:
    """A tool result's content for Claude: content blocks as they are, else text."""
    return outcome.blocks if outcome.blocks is not None else _text(outcome.content)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


class ClaudeAgentSdkRunner:
    """Runs each segment with the Claude Agent SDK and its bundled Claude Code engine."""

    def __init__(
        self,
        *,
        session_store: Any = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        cli_path: str | None = None,
        extra_options: dict[str, Any] | None = None,
        one_tool_at_a_time: bool = False,
        model: str | None = None,
        max_budget_usd: float | None = None,
        warm_engines: int = 0,
        warm_seconds: float = 30.0,
        engine_cleanup: Literal["required", "best_effort"] = "required",
    ) -> None:
        """Create the runner.

        Args:
            session_store: None (the default) keeps each conversation in its
                Workflow: nothing to set up, and any Worker can run any step. Or a
                Claude Agent SDK ``SessionStore`` that every Worker can reach, to
                keep conversations there instead (``FileSessionStore`` is for tests
                and one machine). Every Worker of a task queue needs the same choice.
            cwd: Working directory of the engine. Sessions are keyed by it, so give
                every Worker the same one.
            env: Extra environment variables for the engine. The engine also
                inherits the Worker's environment.
            cli_path: Path of a Claude Code executable to use instead of the bundled one.
            extra_options: More ``ClaudeAgentOptions`` fields (for example
                ``permission_mode``, ``agents``, ``hooks``, ``setting_sources``,
                ``thinking``). ``env``, ``mcp_servers`` and ``allowed_tools`` are
                merged with the plugin's; ``system_prompt`` (a string or a preset) is
                the default for agents that set none. Options the agent or the plugin
                sets itself, and ``extra_args`` for the same engine flags, are
                refused.
            one_tool_at_a_time: Ask Claude for one tool call per message. Not needed
                for durable tools: when Claude calls several at once, they all run.
            model: Default model when the agent does not set one.
            max_budget_usd: Cost cap per segment.
            warm_engines: Experimental. How many paused engines this Worker keeps
                running (0, the default: none). When the session's next segment
                runs on this Worker within ``warm_seconds``, at the same checkpoint
                and with only the result of the paused call, the result goes to the
                running engine as its next message: no new engine starts and
                resumes the session (with a local fake model, a tool round was
                about 7 times faster: 100 ms instead of 740 ms). In any other case
                (another Worker, a retry, a reset, a longer wait) the warm engine
                ends and the segment resumes from the checkpoint as usual. Each
                warm engine keeps its process and memory (a few hundred MB);
                ``ClaudeAgentPlugin`` ends them when its Worker stops. No engine
                stays warm for segments with ``max_turns`` or ``max_budget_usd``,
                calls answered in one message, Claude Code tools run as their own
                Activities, or a runner whose ``extra_options`` has hooks, a
                permission or ``stderr`` callback, or in-process MCP servers; with
                the conversation in the Workflow, neither does the engine that
                starts it. A resumed engine runs ``PreToolUse`` hooks again for the
                call whose result it gets; a warm one does not.
            warm_seconds: How long a paused engine stays warm.
            engine_cleanup: Whether each engine must end with this Worker process.
                ``"required"`` (the default): if the engine cannot start through its
                launcher (Linux, macOS), cannot join the Worker's job object
                (Windows), or the Worker cannot hold its lock file, the step stops
                with a retryable ``ApplicationError`` of type
                ``EngineCleanupUnavailable`` that says what to fix, before Claude
                Code runs anything (an engine can fail to join the job only after it
                started: the step then ends it at once). ``"best_effort"``: the step
                goes on with a warning, and if the Worker process dies, its engine
                finishes its current turn, built-in tools included.

        Raises:
            ValueError: If ``extra_options`` sets an option the plugin manages, or
                Claude Code and the SDK would derive different session keys for the
                working directory (for example decomposed Unicode or emoji in its
                path, or a path given in another form than its real one), or
                ``warm_engines`` is negative, or ``warm_seconds`` not positive, or
                ``engine_cleanup`` is not ``"required"`` or ``"best_effort"``.
        """
        _check_extra_options(extra_options or {})
        if warm_engines < 0 or warm_seconds <= 0:
            raise ValueError("warm_engines must be 0 or more, warm_seconds above 0")
        cleanup = cast("str", engine_cleanup)  # a caller may pass anything
        if cleanup not in ("required", "best_effort"):
            raise ValueError(
                f'engine_cleanup must be "required" or "best_effort", not {cleanup!r}'
            )
        directory = cwd or os.getcwd()
        mismatch = _key_mismatch(directory)
        if mismatch is not None:
            raise ValueError(
                f"cwd {directory!r}: Claude Code and the Claude Agent SDK would derive "
                f"different session keys ({mismatch}), so resumed sessions would not "
                "be found. Pass the path exactly as os.path.realpath gives it, or use "
                "a directory with a plain ASCII path."
            )
        self._store = session_store
        self._cwd = cwd
        self._env = env or {}
        self._cli_path = cli_path
        self._extra = extra_options or {}
        self._one_tool = one_tool_at_a_time
        self._model = model
        self._max_budget = max_budget_usd
        self._warm_cap = warm_engines
        self._warm_seconds = warm_seconds
        self._cleanup_required = cleanup == "required"
        self._warm: dict[tuple[str, str], _WarmEngine] = {}
        """Paused engines by session and checkpoint (see ``warm_engines``)."""
        self.stub_calls = 0
        """Durable tools the engine ran itself. Stays 0 while the engine honors defer."""
        self._stand_in = StandInModel()
        self._versions: dict[str, str | None] = {}
        self._launch: str | None = (
            None  # the checked launcher script (``_launch_path``)
        )

        def is_set(name: str) -> bool:
            value = self._env.get(name) or os.environ.get(name, "")
            return value.lower() not in ("", "0", "false", "no")

        if not any(is_set(name) for name in ENV_AUTH):
            warnings.warn(
                "temporalio.claude_agent_sdk: no API key, cloud provider, or "
                "CLAUDE_CODE_OAUTH_TOKEN is set. A Claude app login cannot refresh "
                "itself when a session is resumed (every step after the first), so "
                "long-running agents can fail with 'OAuth session expired'. Set "
                "ANTHROPIC_API_KEY, "
                "use Bedrock or Vertex, or run `claude setup-token` and set "
                "CLAUDE_CODE_OAUTH_TOKEN.",
                stacklevel=2,
            )

    def _engine_path(self) -> str | None:
        """The Claude Code executable the SDK will start (same order the SDK uses)."""
        if self._cli_path:
            return str(self._cli_path)
        import claude_agent_sdk

        name = "claude.exe" if os.name == "nt" else "claude"
        bundled = Path(claude_agent_sdk.__file__).parent / "_bundled" / name
        return str(bundled) if bundled.is_file() else shutil.which("claude")

    async def _launch_path(self) -> str | None:
        """The launcher script for this runner's engine, checked (Linux, macOS).

        The SDK starts the engine through it, so the engine ends with this Worker
        process (see ``_launcher``). When the script cannot run (for example a
        temporary folder mounted noexec), ``engine_cleanup`` decides: the step stops,
        and the next one checks again (``"required"``), or the engine starts
        directly, with a warning once (``"best_effort"``).

        Raises:
            ApplicationError: If the script cannot run and cleanup is required.
        """
        if sys.platform == "win32":
            return None
        engine = self._engine_path()
        if engine is None:
            return None  # the SDK reports the missing engine
        known = engine in _launch_scripts
        script = _launch_scripts.get(engine)
        if script is not None and not (
            os.path.isfile(script) and _private_folder(os.path.dirname(script))
        ):
            # A cleaner of temporary files removed it (a Worker idle for days).
            script = _launch_scripts[engine] = _write_launch_script(engine)
        if script is None and (not known or self._cleanup_required):
            # Checked the first time, and again for a runner that requires it.
            script = _write_launch_script(engine)
            problem = await _launch_problem(script)
            if problem is None:
                _launch_scripts[engine] = script
                return script
            if not self._cleanup_required:
                _launch_scripts[engine] = None  # not checked again
            _cleanup_problem(
                f"cannot run {script} ({problem}): without it, Claude Code would not "
                "end with this Worker process, and would finish its current turn if "
                "the process died. Make the temporary folder allow running programs "
                "(for example, set TMPDIR to a folder on a local disk).",
                required=self._cleanup_required,
                key=f"launch {engine}",
            )
            return None
        return script

    async def _prepare_engine(self) -> None:
        """Arrange for engines to end with this Worker process.

        Linux and macOS: the launcher (``_launch_path``). Windows: the Worker's job
        object, created once; each engine joins it as it starts (``_JobWatch``).

        Raises:
            ApplicationError: If that cannot be arranged and cleanup is required,
                before any engine starts.
        """
        self._launch = await self._launch_path()
        problem = _worker_job()
        if problem is not None:
            _cleanup_problem(
                _job_problem(problem), required=self._cleanup_required, key="job"
            )

    async def _engine_version(self) -> str | None:
        """``claude -v`` of the engine, checked once per executable (None if unknown)."""
        path = self._engine_path()
        if path is None:
            return None
        if path not in self._versions:
            reported: str | None = None
            try:
                proc = await asyncio.create_subprocess_exec(
                    path,
                    "-v",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                try:
                    out, _ = await asyncio.wait_for(proc.communicate(), 30)
                    reported = out.decode(errors="replace").strip() or None
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
            except OSError:
                reported = None
            self._versions[path] = reported
        return self._versions[path]

    async def _start(
        self, inp: SegmentInput, attempt: int, injected: dict[str, ToolOutcome]
    ) -> tuple[str, bool, set[str]]:
        """Where a segment starts in the session store.

        A segment that runs again continues in a copy of the session that ends at
        the checkpoint, because an unfinished attempt may have written after it.

        Returns:
            The session, whether to resume it, and the calls whose results are
            already in it (see ``_copy``).
        """
        if inp.checkpoint is None:  # nothing committed yet: a new session
            return _attempt_session_id(inp.session_id, attempt), False, set()
        if attempt == 1 and not inp.fork:
            return inp.session_id, True, set()
        session_id, delivered = await self._copy(inp, injected)
        return session_id, True, delivered

    async def _copy(
        self, inp: SegmentInput, injected: dict[str, ToolOutcome]
    ) -> tuple[str, set[str]]:
        """A copy of the session that ends at the checkpoint.

        After a pause in a message with several calls, the copy also holds the
        results of the calls denied after the pause (see ``_deliver``).

        Returns:
            The copy's id, and the calls whose results it holds.
        """
        assert inp.checkpoint is not None
        key = {
            "project_key": project_key_for_directory(self._cwd),
            "session_id": inp.session_id,
        }
        entries = cast("list[dict[str, Any]]", await self._store.load(key) or [])
        moved = _deliver(entries, inp.checkpoint, injected)
        if moved is None:
            forked = await fork_session_via_store(
                self._store,
                inp.session_id,
                directory=self._cwd,
                up_to_message_id=inp.checkpoint,
            )
            return forked.session_id, set()
        seed, delivered = moved
        copy_id = str(uuid.uuid4())
        await self._store.append(
            {**key, "session_id": copy_id},
            [{**e, "sessionId": copy_id} if "sessionId" in e else e for e in seed],
        )
        return copy_id, delivered

    async def _checkpoint(
        self,
        store: Any,
        session_id: str,
        assistant_uuid: str | None,
        paused_call: str | None,
        resumed_at: str | None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Where the next segment continues, read back from the store (see ``_resume_point``).

        Reading it back also proves that the turn reached the session store.

        Returns:
            The checkpoint, and the session's entries.

        Raises:
            RuntimeError: If the store does not have the turn (Temporal retries).
        """
        key = {
            "project_key": project_key_for_directory(self._cwd),
            "session_id": session_id,
        }
        entries = cast("list[dict[str, Any]]", await store.load(key) or [])
        stored = assistant_uuid is None or any(
            e.get("uuid") == assistant_uuid for e in entries
        )
        leaf = _resume_point(entries, paused_call)
        if not stored or leaf is None or leaf == resumed_at:  # nothing new stored
            raise RuntimeError(
                f"This segment's turn did not reach the session store (session "
                f"{session_id}), so it could not be continued. Retrying."
            )
        return leaf, entries

    def _may_park(self, inp: SegmentInput) -> bool:
        """Whether this segment's engine may stay warm when it pauses.

        Not with a limit on turns or cost: the engine counts them per run. Not with
        hooks, a permission callback, a ``stderr`` callback or in-process MCP servers
        in ``extra_options``: the SDK runs them in tasks of the segment that started
        the engine, so in a later segment they would see that segment's Activity
        (its info, heartbeats and cancellation).
        """
        servers = self._extra.get("mcp_servers")
        return (
            self._warm_cap > 0
            and inp.max_turns is None
            and self._max_budget is None
            and not self._extra.get("hooks")
            and self._extra.get("can_use_tool") is None
            and self._extra.get("stderr") is None
            and not (
                isinstance(servers, dict)
                and any(
                    isinstance(c, dict) and c.get("type") == "sdk"
                    for c in servers.values()
                )
            )
        )

    @staticmethod
    def _shape(inp: SegmentInput) -> str:
        """What an engine is started with that the next segment must share."""
        return json.dumps(
            [
                inp.system_prompt,
                inp.model,
                [[t.name, t.description, t.input_schema] for t in inp.tools],
                inp.builtin_tools,
                inp.tool_activities,
            ],
            sort_keys=True,
            default=str,
        )

    def _take_warm(
        self, inp: SegmentInput, injected: dict[str, ToolOutcome], attempt: int
    ) -> _WarmEngine | None:
        """The warm engine this segment continues, taken from the pool, or None.

        Never for a retry: an earlier attempt may still run somewhere and go on from
        the same checkpoint; a retry starts a new engine, as without warm engines.
        """
        if inp.checkpoint is None:
            return None
        warm = self._warm.pop((inp.session_id, inp.checkpoint), None)
        if warm is None:
            return None
        if warm.timer is not None:
            warm.timer.cancel()
        payload = sum(len(_text(_result_content(o))) for o in injected.values())
        if (
            attempt == 1
            and inp.prompt is None
            and not inp.fork
            and self._may_park(inp)
            and set(injected) == {warm.paused}
            and warm.shape == self._shape(inp)
            and 8 * payload <= warm.buffer
            and _engine_alive(warm.client)
        ):
            return warm
        self._end_warm(warm)
        return None

    def _park(self, warm: _WarmEngine) -> None:
        """Keep a paused engine for ``warm_seconds``, ending the oldest beyond the cap."""
        key = (warm.session_id, warm.checkpoint)
        old = self._warm.pop(key, None)
        if old is not None:
            self._end_warm(old)
        while self._warm and len(self._warm) >= self._warm_cap:
            self._end_warm(self._warm.pop(next(iter(self._warm))))
        self._warm[key] = warm
        warm.timer = asyncio.get_running_loop().call_later(
            self._warm_seconds, self._expire, key, warm
        )

    def _expire(self, key: tuple[str, str], warm: _WarmEngine) -> None:
        if self._warm.get(key) is warm:
            del self._warm[key]
            self._end_warm(warm)

    @staticmethod
    def _end_warm(warm: _WarmEngine) -> None:
        """End a warm engine in the background; its hook folder and lock go too.

        The folder and the lock go once the ending task is done, also if it was
        cancelled before it ran (an event loop that closes): the engine then gets
        SIGTERM at once.
        """
        if warm.timer is not None:
            warm.timer.cancel()
        process = getattr(getattr(warm.client, "_transport", None), "_process", None)
        resumed = getattr(warm.client, "_materialized", None)
        task = asyncio.ensure_future(_disconnect(warm.client))
        _ending.add(task)

        def ended(task: asyncio.Task[None]) -> None:
            _ending.discard(task)
            _stop_now(process, resumed)
            _release_worker_lock(warm.lock)
            shutil.rmtree(warm.hook_dir, ignore_errors=True)

        task.add_done_callback(ended)

    async def _ends_at_checkpoint(self, inp: SegmentInput) -> bool:
        """Whether the shared session still ends at the segment's checkpoint.

        A warm engine knows the session as it left it. When a segment from the same
        checkpoint ran on another Worker (a retry, or a reset), the session went on
        there, and only a new engine continues it right (in a copy, ``_SessionMoved``).
        """
        key = {
            "project_key": project_key_for_directory(self._cwd),
            "session_id": inp.session_id,
        }
        entries = cast("list[Any]", await self._store.load(key) or [])
        return _last_entry(entries) == inp.checkpoint

    def _end_all_warm(self) -> None:
        """End every warm engine, each in a task of its own (the Worker stopped).

        Nothing waits for them here: ``async with Worker`` cancels the Worker's run
        once it shut down, and a run that still waited would then cancel the code
        that comes after the ``async with`` block.
        """
        while self._warm:
            self._end_warm(self._warm.pop(next(iter(self._warm))))

    async def run(self, inp: SegmentInput, attempt: int) -> SegmentOutput:
        """Run one segment until Claude pauses at a durable tool call or finishes.

        Args:
            inp: The segment input.
            attempt: The Activity attempt number, starting at 1.

        Returns:
            The pause or the final answer, with the segment's checkpoint (and, when
            the Workflow holds the conversation, what changed in it). ``is_error``
            is set when the engine broke the pause contract, or when retrying cannot
            help.

        Raises:
            RuntimeError: If the transcript did not reach the session store, the
                Workflow did not serve its conversation, or the conversation is kept
                where this runner does not keep it (a session store, or the
                Workflow). Temporal retries the segment.
        """
        reported = await self._engine_version()
        if reported is not None and _too_old(reported):
            return _too_old_output(inp.session_id, reported)  # before the engine starts
        await self._prepare_engine()
        injected = {k: _as_outcome(v) for k, v in inp.injected.items()}
        if self._store is None:
            return await self._run_held(inp, injected, attempt)
        if inp.transcript or (
            inp.conversation is not None and inp.conversation.entries
        ):
            raise RuntimeError(
                f"Session {inp.session_id} is kept in its Workflow, but this Worker's "
                "runner has a session store. Every Worker of a task queue needs the "
                "same choice: create this runner without session_store. Retrying."
            )
        warm = self._take_warm(inp, injected, attempt)
        if warm is not None:
            try:
                fresh = await self._ends_at_checkpoint(inp)
            except BaseException:
                self._end_warm(warm)
                raise
            if not fresh:
                self._end_warm(warm)
                warm = None
        if warm is not None:
            return await self._run_engine(
                inp,
                injected,
                warm.session_id,
                True,
                warm.store,
                guard=inp.checkpoint,
                delivered=set(),
                warm=warm,
            )
        session_id, resume, delivered = await self._start(inp, attempt, injected)
        if resume and len(injected) == len(delivered) and inp.prompt is None:
            return SegmentOutput(
                session_id=session_id,
                is_error=True,
                error="Nothing to send: no tool result and no prompt",
            )
        in_place = resume and session_id == inp.session_id
        try:
            return await self._run_engine(
                inp,
                injected,
                session_id,
                resume,
                self._store,
                guard=inp.checkpoint if in_place else None,
                delivered=delivered,
            )
        except _SessionMoved:
            # The session went on after the checkpoint (after a pause in a message
            # with several calls, or a Workflow reset to an earlier point): continue
            # in a copy that ends there.
            session_id, delivered = await self._copy(inp, injected)
            return await self._run_engine(
                inp, injected, session_id, True, self._store, delivered=delivered
            )

    async def _run_held(
        self, inp: SegmentInput, injected: dict[str, ToolOutcome], attempt: int
    ) -> SegmentOutput:
        """Run a segment of a conversation the Workflow holds.

        The engine resumes from a fresh in-memory session store that holds the
        committed conversation up to the checkpoint, so every attempt starts from
        what the Workflow committed.
        """
        committed = await read_conversation(inp)
        warm = self._take_warm(inp, injected, attempt)
        if warm is not None:
            try:
                # The conversation must hold the checkpoint, and no calls answered in
                # the same message (their results go into the conversation, which
                # only a new engine reads).
                clean = (
                    bool(committed)
                    and _seed(committed, warm.checkpoint) is not None
                    and _deliver(committed, warm.checkpoint, injected) is None
                )
            except BaseException:
                self._end_warm(warm)
                raise
            if clean:
                return await self._run_engine(
                    inp,
                    injected,
                    warm.session_id,
                    True,
                    warm.store,
                    committed=committed,
                    delivered=set(),
                    warm=warm,
                )
            self._end_warm(warm)
        delivered: set[str] = set()  # results the seed already holds
        if inp.checkpoint is None:
            if committed:
                return SegmentOutput(
                    session_id=inp.session_id,
                    is_error=True,
                    error=(
                        f"The Workflow holds {len(committed)} conversation entries but "
                        "no checkpoint, so it is unclear where Claude would continue."
                    ),
                )
            session_id, seed, resume = (
                _attempt_session_id(inp.session_id, attempt),
                [],
                False,
            )
        else:
            if not committed:
                raise RuntimeError(
                    f"The Workflow holds no conversation for session {inp.session_id} "
                    f"(checkpoint {inp.checkpoint}): it is kept in a session store this "
                    "runner does not have, or AgentState.transcript was not carried to "
                    "this run. Give this Worker's runner the session store the "
                    "conversation started with. Retrying."
                )
            moved = _deliver(committed, inp.checkpoint, injected)
            found = moved[0] if moved is not None else _seed(committed, inp.checkpoint)
            if found is None:
                return SegmentOutput(
                    session_id=inp.session_id,
                    is_error=True,
                    error=(
                        f"Checkpoint {inp.checkpoint} is not in the conversation the "
                        "Workflow holds, so it is unclear where Claude would continue."
                    ),
                )
            if moved is not None:
                delivered = moved[1]
            session_id, seed, resume = inp.session_id, _without_cost_state(found), True
        if resume and len(injected) == len(delivered) and inp.prompt is None:
            return SegmentOutput(
                session_id=session_id,
                is_error=True,
                error="Nothing to send: no tool result and no prompt",
            )
        store = InMemorySessionStore()
        if seed:
            key = {
                "project_key": project_key_for_directory(self._cwd),
                "session_id": session_id,
            }
            await store.append(key, seed)  # type: ignore[arg-type]
        # A copy that is already there is not this run's (Claude Code then refuses the
        # session id), so it stays.
        local = self._local_copy(session_id)
        ours = not resume and not any(p.exists() for p in local)
        return await self._run_engine(
            inp,
            injected,
            session_id,
            resume,
            store,
            committed=committed,
            delivered=delivered,
            local_copy=local if ours else (),
        )

    async def run_tool_step(self, step: ToolStepInput, attempt: int) -> ToolOutcome:
        """Run one Claude Code tool call that paused a segment, as its own Activity.

        Claude Code resumes a copy of the session that ends where it paused at the
        call (in memory: a session store is not written), and the hook lets it run
        exactly that call. Afterwards the engine asks the model to go on; a local
        stand-in answers, so no real model call is made. Tested on Claude Code
        2.1.273 and 2.1.287: the tool runs once, with the engine's own result.

        The command still sees the Worker's own environment (see ``_step_env``), and
        a result the step already has is kept even if the engine fails afterwards,
        so the Activity is not retried for a call that ran.

        Args:
            step: The call, and where its session paused.
            attempt: The Activity attempt number.

        Returns:
            What the tool returned, as Claude Code would show it to Claude.

        Raises:
            ApplicationError: If the session did not pause at this call, the engine
                did not run it, or a hook in ``extra_options`` tried to decide on it
                (not retried: Claude sees the error).
            RuntimeError: If the Workflow did not serve its conversation, or the
                conversation is kept where this runner does not keep it (retried).
        """
        del attempt
        await self._prepare_engine()
        call = step.call
        key = {
            "project_key": project_key_for_directory(self._cwd),
            "session_id": step.session_id,
        }
        held = step.transcript is not None or (
            step.conversation is not None and step.conversation.entries > 0
        )
        if self._store is None and not held:
            raise RuntimeError(
                f"Session {step.session_id} is kept in a session store this runner "
                "does not have. Every Worker of a task queue needs the same choice: "
                "give this runner the session store. Retrying."
            )
        if self._store is not None and held:
            raise RuntimeError(
                f"Session {step.session_id} is kept in its Workflow, but this Worker's "
                "runner has a session store. Every Worker of a task queue needs the "
                "same choice: create this runner without session_store. Retrying."
            )
        if self._store is None:
            entries = await read_conversation(step)
        else:
            entries = cast("list[Any]", await self._store.load(key) or [])
        seed = _seed(entries, step.checkpoint)
        if seed is None or not any(_marker_of(e) == call.id for e in seed):
            raise ApplicationError(
                f"Session {step.session_id} did not pause at tool call {call.id} "
                f"({call.name}), so the call cannot run.",
                non_retryable=True,
            )
        store = InMemorySessionStore()
        await store.append(key, seed)  # type: ignore[arg-type]
        hook_dir = _hook_folder()
        inp = SegmentInput(
            session_id=step.session_id,
            prompt=None,
            tools=step.tools,
            builtin_tools=step.builtin_tools,
            checkpoint=step.checkpoint,
        )
        violations: list[str] = []
        options = self._engine_options(
            inp,
            {},
            step.session_id,
            True,
            store,
            None,
            hook_dir,
            self._durable_server(step.tools, []),
            violations,
        )
        options["env"] = self._step_env(options["env"], hook_dir, call.id)
        result: ToolResultBlock | None = None
        saved: str | None = None
        failure: Exception | None = None
        denials: dict[str, str] = {}
        lock: int | None = None
        stopper = (
            asyncio.ensure_future(_stop_hooks_when_cancelled(hook_dir))
            if activity.in_activity()
            else None
        )
        job = _JobWatch(hook_dir, required=self._cleanup_required)
        joined = False  # the engine's first message came (``_JobWatch.joined``)
        try:
            lock = _hold_worker_lock(hook_dir, required=self._cleanup_required)
            engine = cast(  # query() is typed as an iterator; it is a generator
                "AsyncGenerator[Any, None]",
                query(prompt="", options=ClaudeAgentOptions(**options)),
            )
            async with contextlib.aclosing(engine) as messages:
                async for message in messages:
                    if not joined:
                        joined = True
                        job.joined()
                    if not isinstance(message, UserMessage) or isinstance(
                        message.content, str
                    ):
                        continue
                    for block in message.content:
                        if (
                            isinstance(block, ToolResultBlock)
                            and block.tool_use_id == call.id
                        ):
                            result = block
                            # The engine's temporary folder still exists: read the
                            # end of an output it saved to a file before it goes.
                            saved = _saved_output_tail(
                                block.content, key["project_key"], step.session_id
                            )
            if not joined:
                joined = True
                job.joined()  # the engine ended without a message
        except Exception as err:
            if result is None:
                if not joined:
                    joined = True
                    job.joined()  # the engine ended before its first message: why
                raise
            # The call ran; the engine failed afterwards (for example the model call
            # that follows the tool). Running the step again would run it again.
            failure = err
        finally:
            if stopper is not None:
                stopper.cancel()
            job.stop()
            denials = _hook_denials(hook_dir)
            _release_worker_lock(lock)
            shutil.rmtree(hook_dir, ignore_errors=True)
        violation = _hook_violation(violations)
        if violation is not None:
            raise ApplicationError(violation, non_retryable=True)
        text = _text(result.content) if result is not None else ""
        if (
            result is None
            or denial_name(call.id) in denials
            or (bool(result.is_error) and _says_only(result.content, STOPPED))
        ):
            raise ApplicationError(
                f"Claude Code did not run tool call {call.id} ({call.name}) in its "
                f"step: {text or 'no result'}",
                non_retryable=True,
            )
        if failure is not None and activity.in_activity():
            activity.logger.warning(
                "Tool call %s (%s) ran; the engine failed after it (%s). Its result "
                "is kept.",
                call.id,
                call.name,
                failure,
            )
        is_error = bool(result.is_error)
        if isinstance(result.content, list):
            return ToolOutcome(blocks=list(result.content), is_error=is_error)
        if saved is not None:
            text += (
                "\n\nThe file named above was removed when the step that ran this "
                f"call ended. The output ends with:\n{saved}"
            )
        return ToolOutcome(content=text, is_error=is_error)

    def _step_env(
        self, env: dict[str, str], hook_dir: str, call_id: str
    ) -> dict[str, str]:
        """The engine's environment in a tool step, and the command's.

        The engine talks to the local stand-in model (``STEP_PROVIDER_ENV``,
        ``ANTHROPIC_BASE_URL``, and the stand-in's own key). The command gets the
        Worker's own environment back (see ``_command_env``), so a command that calls
        the Anthropic API reaches the Worker's provider as it would in a segment.
        """
        before = {**os.environ, **env}
        hosts: list[str] = []
        for name in ("NO_PROXY", "no_proxy"):
            for host in (before.get(name) or "").split(","):
                if host.strip() and host.strip() not in hosts:
                    hosts.append(host.strip())
        if "127.0.0.1" not in hosts:
            hosts.append("127.0.0.1")
        overrides = {
            **STEP_PROVIDER_ENV,
            "ANTHROPIC_API_KEY": self._stand_in.key,
            "ANTHROPIC_BASE_URL": self._stand_in.base_url,
            "NO_PROXY": ",".join(hosts),
            "no_proxy": ",".join(hosts),
        }
        return _command_env(
            env, hook_dir, self._cwd, overrides, {"TCA_ALLOW_ID": call_id}
        )

    def _durable_server(self, tools: list[ToolSpec], ran_inside: list[str]) -> Any:
        """The durable tools as SDK MCP tools.

        The engine never runs them while the hook defers; if it does, the call is
        recorded in ``ran_inside``.
        """

        def make_stub(spec: ToolSpec) -> Any:
            @tool(spec.name, spec.description, spec.input_schema)
            async def stub(args: dict[str, Any]) -> dict[str, Any]:
                del args
                self.stub_calls += 1  # never happens while the hook defers
                ran_inside.append(spec.name)
                return {
                    "content": [
                        {"type": "text", "text": "This tool must run through Temporal."}
                    ],
                    "is_error": True,
                }

            return stub

        return create_sdk_mcp_server(SERVER, tools=[make_stub(t) for t in tools])

    def _local_copy(self, session_id: str) -> tuple[Path, ...]:
        """Where Claude Code writes a new session in its config folder.

        Nothing for an id this runner would never start.
        """
        try:
            uuid.UUID(session_id)  # only ever a session this runner started
        except ValueError:
            return ()
        env = {**self._env, **(self._extra.get("env") or {})}
        config = env.get("CLAUDE_CONFIG_DIR") or os.environ.get("CLAUDE_CONFIG_DIR")
        base = Path(config) if config else Path.home() / ".claude"
        folder = (
            Path(unicodedata.normalize("NFC", str(base)))
            / "projects"
            / project_key_for_directory(self._cwd)
        )
        return folder / f"{session_id}.jsonl", folder / session_id

    async def _run_engine(
        self,
        inp: SegmentInput,
        injected: dict[str, ToolOutcome],
        session_id: str,
        resume: bool,
        store: Any,
        *,
        guard: str | None = None,
        committed: list[dict[str, Any]] | None = None,
        delivered: set[str] | None = None,
        warm: _WarmEngine | None = None,
        local_copy: tuple[Path, ...] = (),
    ) -> SegmentOutput:
        """Run the engine once, on the session in ``store``.

        Args:
            inp: The segment input.
            injected: Tool results to deliver.
            session_id: The session to start or resume.
            resume: Whether to resume it.
            store: The session store the SDK reads and writes.
            guard: Resuming requires the session to end at this checkpoint.
            committed: The conversation the Workflow holds, when it holds one: the
                output then says what changed in it.
            delivered: Calls whose results the session already holds (see
                ``_deliver``); the others go in the message that resumes it.
            warm: An engine still running since it paused at the call whose result
                is in ``injected`` (see ``warm_engines``): the results go to it as
                its next message, instead of a new engine resuming the session.
            local_copy: The engine's own copy of the new session it starts, to
                remove once the engine ended (see ``_forget_local_copy``).

        Raises:
            _SessionMoved: If the session does not end at ``guard``.
        """
        options: dict[str, Any] = {}
        hook_dir = warm.hook_dir if warm is not None else _hook_folder()
        # Durable tools the engine ran itself (must stay empty), and decisions hooks
        # in extra_options tried to make.
        ran_inside: list[str] = warm.ran_inside if warm is not None else []
        violations: list[str] = warm.violations if warm is not None else []
        result: ResultMessage | None = None
        # A warm engine reported its version when it started.
        engine_version = warm.version if warm is not None else "(unknown version)"
        paused_by_hook: str | None = None
        last_assistant: str | None = None
        store_error: str | None = None
        stopped_by_hook = False
        denials: dict[str, str] = {}
        lock: int | None = warm.lock if warm is not None else None
        # The engine's client and input when it may stay warm (see warm_engines).
        client: Any = warm.client if warm is not None else None
        feed: _Feed | None = warm.feed if warm is not None else None
        parked = False
        stopper: asyncio.Future[None] | None = None
        job: _JobWatch | None = None  # a new engine joins the Worker's job (Windows)
        joined = False
        try:
            if warm is None:
                options = self._engine_options(
                    inp,
                    injected,
                    session_id,
                    resume,
                    store,
                    guard,
                    hook_dir,
                    self._durable_server(inp.tools, ran_inside),
                    violations,
                )
                # A command run inside the segment sees none of the plugin's variables.
                options["env"] = _command_env(
                    options["env"], hook_dir, self._cwd, {}, {}
                )
            else:
                ran_inside.clear()
                violations.clear()
                _next_hook_turn(hook_dir, injected)
            prompt: Any
            in_message = {
                k: v for k, v in injected.items() if k not in (delivered or ())
            }
            if not resume:
                prompt = inp.prompt or ""
            elif in_message:
                prompt = self._user_message(session_id, in_message, inp.prompt)
            else:
                prompt = inp.prompt  # a new task on the session
            # When the Activity is cancelled or times out, deny every later tool call
            # while the SDK shuts the engine down.
            if activity.in_activity():
                stopper = asyncio.ensure_future(_stop_hooks_when_cancelled(hook_dir))
            try:
                messages: AsyncGenerator[Any, None]
                if warm is not None:  # a warm engine: its next message
                    assert feed is not None
                    feed.put(await _as_messages(prompt))
                    messages = _turn_messages(
                        client, feed, False, stay=True, answered=set(injected)
                    )
                else:
                    job = _JobWatch(hook_dir, required=self._cleanup_required)
                    lock = _hold_worker_lock(hook_dir, required=self._cleanup_required)
                    first = await _as_messages(prompt)
                    if self._may_park(inp) and not local_copy:
                        feed = _Feed(first)
                        client = ClaudeSDKClient(ClaudeAgentOptions(**options))
                        messages = _connected(client, feed, resume, set(injected))
                    else:
                        messages = _engine_messages(options, first, resume)
                async with contextlib.aclosing(messages):
                    async for message in messages:
                        if job is not None and not joined:
                            joined = True
                            job.joined()
                        if isinstance(message, MirrorErrorMessage):
                            store_error = message.error or "unknown error"
                        elif (
                            isinstance(message, SystemMessage)
                            and message.subtype == "init"
                        ):
                            version = message.data.get("claude_code_version")
                            engine_version = str(version or engine_version)
                        elif (
                            isinstance(message, AssistantMessage)
                            and message.parent_tool_use_id is None  # not a subagent's
                        ):
                            last_assistant = message.uuid or last_assistant
                            for block in message.content:
                                if isinstance(block, TextBlock) and block.text.strip():
                                    emit({"type": "text", "text": block.text})
                        elif isinstance(message, UserMessage):
                            stopped_by_hook = stopped_by_hook or _hook_said_stopped(
                                message
                            )
                        elif isinstance(message, ResultMessage):
                            result = message
                if job is not None and not joined:
                    joined = True
                    job.joined()  # the engine ended without a message
                marker = Path(hook_dir) / "paused_call"  # written when the hook defers
                if marker.exists():
                    paused_by_hook = marker.read_text(encoding="utf-8").strip() or None
                denials = _hook_denials(hook_dir)
            except ResultError as err:
                final = err.subtype in FINAL_RESULT_ERRORS or _final_api_error(err)
                if not final:
                    raise  # other engine errors: let Temporal retry the segment
                # Retrying will not help.
                return SegmentOutput(
                    session_id=session_id, is_error=True, error=str(err)
                )
            except RuntimeError as err:
                if _moved(err):
                    raise _SessionMoved(str(err)) from err
                if job is not None and not joined:
                    job.joined()  # the engine ended before its first message: why
                raise
            except Exception:
                if job is not None and not joined:
                    job.joined()
                raise
            finally:
                if stopper is not None:
                    stopper.cancel()
                if job is not None:
                    job.stop()
            if warm is not None and result is None:
                # The engine ended after it was taken (it was alive then): a new
                # engine can do the step.
                raise RuntimeError(
                    "The warm engine ended before its turn's result. Retrying."
                )
            out = await self._segment_output(
                inp,
                injected,
                session_id,
                resume,
                store,
                guard,
                committed,
                warm,
                ran_inside,
                violations,
                hook_dir,
                result,
                engine_version,
                paused_by_hook,
                last_assistant,
                store_error,
                stopped_by_hook,
                denials,
            )
            if feed is not None and self._parks(
                out, hook_dir, client, local_copy, denials
            ):
                assert out.deferred is not None and out.checkpoint is not None
                assert result is not None
                self._park(
                    _WarmEngine(
                        client=client,
                        feed=feed,
                        session_id=out.session_id,
                        checkpoint=out.checkpoint,
                        paused=out.deferred.id,
                        shape=self._shape(inp),
                        hook_dir=hook_dir,
                        lock=lock,
                        store=store,
                        cost=float(result.total_cost_usd or 0.0),
                        version=engine_version,
                        ran_inside=ran_inside,
                        violations=violations,
                        buffer=(
                            warm.buffer
                            if warm is not None
                            else int(options["max_buffer_size"])
                        ),
                    )
                )
                parked = True
            return out
        finally:
            if not parked:
                try:
                    if client is not None:
                        await _disconnect(client)
                finally:
                    _release_worker_lock(lock)
                    # The hook denies from now on.
                    shutil.rmtree(hook_dir, ignore_errors=True)
                    _forget_local_copy(local_copy)

    def _parks(
        self,
        out: SegmentOutput,
        hook_dir: str,
        client: Any,
        local_copy: tuple[Path, ...],
        denials: dict[str, str],
    ) -> bool:
        """Whether an engine that just ran stays warm: it paused at one durable call,
        cleanly.

        Not at a Claude Code tool that runs as its own Activity (a tool step): a new
        engine checks such a call again when its result arrives, and a running one
        does not (tested: an Edit run apart is refused on resume, and taken by a
        running engine). So that result always goes to a new engine.

        Not an engine that started a conversation the Workflow holds: it writes the
        conversation to its own config folder (``_forget_local_copy``), and a Worker
        that dies while it waits would leave it there. A resumed engine runs in a
        temporary folder, as each segment's engine does today.

        Not after the hook denied a call in the same turn: the denial comes after the
        pause in the session, so the next segment needs a new engine anyway.
        """
        return (
            not local_copy
            and not denials
            and out.deferred is not None
            and out.deferred.kind == "durable"
            and not out.is_error
            and not out.siblings
            and out.checkpoint is not None
            and not os.path.exists(os.path.join(hook_dir, "stop"))
            and _engine_alive(client)
        )

    async def _segment_output(
        self,
        inp: SegmentInput,
        injected: dict[str, ToolOutcome],
        session_id: str,
        resume: bool,
        store: Any,
        guard: str | None,
        committed: list[dict[str, Any]] | None,
        warm: _WarmEngine | None,
        ran_inside: list[str],
        violations: list[str],
        hook_dir: str,
        result: ResultMessage | None,
        engine_version: str,
        paused_by_hook: str | None,
        last_assistant: str | None,
        store_error: str | None,
        stopped_by_hook: bool,
        denials: dict[str, str],
    ) -> SegmentOutput:
        """What one engine run committed: the pause or the answer, or why it failed.

        Raises:
            RuntimeError: If the turn did not reach the session store (retried).
        """
        violation = _hook_violation(violations)
        if violation is not None:
            return SegmentOutput(session_id=session_id, is_error=True, error=violation)
        if stopped_by_hook or REASON_KEYS[STOPPED] in denials.values():
            # Not cancelled, yet the hook denied a call as stopped: it could not see
            # this step's folder. Fail closed rather than commit what Claude said
            # without the tool.
            return SegmentOutput(
                session_id=session_id,
                is_error=True,
                error=(
                    "The pause hook could not see this step's folder "
                    f"({hook_dir}), so it denied Claude's tool calls. Run Claude Code "
                    "where it sees the Worker's temporary folder (for example, not in "
                    "a sandbox with its own /tmp). No tool ran."
                ),
            )
        if store_error is not None:
            raise RuntimeError(
                f"The session store did not save part of this segment's transcript "
                f"({store_error}). Retrying."
            )
        if _too_old(engine_version):
            # The engine reports its version at start. If the check above could not
            # read it, refuse here: the Workflow then never runs the paused tool.
            return _too_old_output(session_id, engine_version)
        if result is None:
            return SegmentOutput(
                session_id=session_id,
                is_error=True,
                error="Claude returned no result message",
            )
        sid = result.session_id or session_id
        cost = float(result.total_cost_usd or 0.0)
        if warm is not None and cost >= warm.cost:
            cost -= warm.cost  # a warm engine reports its running total
        deferred = result.deferred_tool_use
        broken = self._pause_contract_problem(
            ran_inside,
            paused_by_hook,
            deferred,
            set(injected),
            engine_version,
            result.stop_reason,
        )
        if broken:
            return SegmentOutput(
                session_id=sid, is_error=True, error=broken, cost_usd=cost
            )
        if deferred is None and result.is_error:
            return SegmentOutput(
                session_id=sid,
                is_error=True,
                error=str(result.errors or result.subtype),
                cost_usd=cost,
            )
        # A run with nothing new after where it resumed did not reach the store.
        resumed_at = (
            guard if committed is None else (inp.checkpoint if resume else None)
        )
        checkpoint, entries = await self._checkpoint(
            store,
            sid,
            last_assistant,
            deferred.id if deferred is not None else None,
            resumed_at,
        )
        if deferred is not None:
            out = SegmentOutput(
                session_id=sid,
                deferred=DeferredCall(
                    id=deferred.id,
                    name=deferred.name.removeprefix(PREFIX),
                    input=dict(deferred.input),
                    # A Claude Code tool that runs as its own Activity (a tool step).
                    kind="durable" if deferred.name.startswith(PREFIX) else "engine",
                ),
                checkpoint=checkpoint,
                cost_usd=cost,
                siblings=_siblings(entries, deferred.id, denials),
            )
        else:
            out = SegmentOutput(
                session_id=sid,
                result=result.result,
                checkpoint=checkpoint,
                cost_usd=cost,
            )
        out.external_storage = external_storage_on()
        if committed is not None:
            # The engine only appends to what it resumed from, which is the committed
            # conversation, cut after the checkpoint when that is the deferral marker.
            # Saved cost totals never join the conversation (``_kept``).
            keep, covered = _kept(committed, entries)
            out.transcript_keep = keep
            out.transcript_add = _without_cost_state(entries[covered:])
            problem = too_large(out)
            if problem is not None:
                return SegmentOutput(
                    session_id=sid, is_error=True, error=problem, cost_usd=cost
                )
        return out

    def _engine_options(
        self,
        inp: SegmentInput,
        injected: dict[str, ToolOutcome],
        session_id: str,
        resume: bool,
        store: Any,
        guard: str | None,
        hook_dir: str,
        durable_server: Any,
        violations: list[str],
    ) -> dict[str, Any]:
        """The ``ClaudeAgentOptions`` fields of one engine run, ``extra_options`` merged in."""
        extra = dict(self._extra)
        default_prompt = extra.pop("system_prompt", None)
        hints = [
            *([ONE_TOOL_HINT] if self._one_tool else []),
            *([SHELL_HINT] if set(_SHELL_TOOLS) & set(inp.builtin_tools) else []),
        ]
        hint = "\n\n".join(hints) or None
        activities = list(inp.tool_activities)

        def decides_alone(name: str) -> bool:  # the plugin decides on these calls
            return name.startswith(PREFIX) or _runs_as_activity(name, activities)

        if "hooks" in extra:
            extra["hooks"] = _guard_hooks(extra["hooks"], decides_alone, violations)
        system_prompt: Any
        if inp.system_prompt is None and isinstance(default_prompt, dict):
            preset = cast("dict[str, Any]", default_prompt)  # e.g. Claude Code's own
            append = "\n\n".join(p for p in (preset.get("append"), hint) if p)
            system_prompt = {**preset, "append": append} if append else dict(preset)
        else:
            base = (
                inp.system_prompt if inp.system_prompt is not None else default_prompt
            )
            system_prompt = "\n\n".join(p for p in (base, hint) if p) or None
        durable_names = [PREFIX + t.name for t in inp.tools]
        payload = sum(len(_text(_result_content(o))) for o in injected.values())
        payload += len(inp.prompt or "")
        options: dict[str, Any] = {
            "system_prompt": system_prompt,
            "model": inp.model or self._model,
            "tools": list(inp.builtin_tools),  # no built-in tools unless asked for
            "max_turns": inp.max_turns,
            "mcp_servers": {
                **(extra.pop("mcp_servers", None) or {}),
                SERVER: durable_server,
            },
            "strict_mcp_config": True,
            "setting_sources": [],
            # Tools that run as Activities are not pre-approved: the hook defers or
            # allows them call by call, so if it ever gives no answer (it failed, or
            # an engine ignores defer), Claude Code refuses a command that changes
            # something instead of running it inside the segment.
            "allowed_tools": list(
                dict.fromkeys(
                    [
                        *durable_names,
                        *(
                            t
                            for t in [
                                *inp.builtin_tools,
                                *(extra.pop("allowed_tools", None) or []),
                            ]
                            if not _runs_as_activity(t, activities)
                        ),
                    ]
                )
            ),
            "settings": str(Path(hook_dir) / "settings.json"),
            "session_store": _guarded(store, session_id, guard),
            "cwd": self._cwd,
            "env": {
                **self._env,
                **(extra.pop("env", None) or {}),
                **ENGINE_ENV,
                "TCA_HOOK_DIR": hook_dir,
                "TCA_ANSWERED_IDS": " ".join(injected),
                "TCA_TOOL_ACTIVITIES": "\n".join(inp.tool_activities),
                **({WORKER_PID: str(os.getpid())} if self._launch else {}),
            },
            # Through the launcher, the engine ends with this Worker process.
            "cli_path": self._launch or self._cli_path,
            "permission_mode": DEFAULT_PERMISSION_MODE,  # extra_options may change it
            # The engine echoes each delivered result as one line: room for any result.
            "max_buffer_size": max(
                MIN_BUFFER_BYTES,
                8 * payload,
                int(extra.pop("max_buffer_size", None) or 0),
            ),
        }
        if self._max_budget is not None:
            options["max_budget_usd"] = self._max_budget  # safety cap per segment
        if resume:
            options["resume"] = session_id
        else:
            options["session_id"] = session_id
        options.update(
            extra
        )  # only options the plugin leaves alone (checked in __init__)
        return options

    @staticmethod
    def _pause_contract_problem(
        ran_inside: list[str],
        paused_by_hook: str | None,
        deferred: Any,
        answered: set[str],
        version: str,
        stop_reason: Any,
    ) -> str | None:
        """Fail closed if the engine did not honor the pause.

        Returns:
            An error message, or None if the engine behaved.
        """
        if ran_inside:
            return (
                f"Claude Code {version} ran durable tool(s) "
                f"{', '.join(sorted(set(ran_inside)))} inside the engine instead of "
                f"pausing. {PAUSE_CONTRACT}"
            )
        if deferred is not None and deferred.id in answered:
            return (
                f"Claude Code {version} paused again at tool call {deferred.id}, whose "
                f"result was just delivered. {PAUSE_CONTRACT}"
            )
        if paused_by_hook and (deferred is None or deferred.id != paused_by_hook):
            got = (
                f"paused at {deferred.id}" if deferred is not None else "did not pause"
            )
            return (
                f"The pause hook deferred tool call {paused_by_hook}, but Claude Code "
                f"{version} {got} (stop_reason={stop_reason}). {PAUSE_CONTRACT}"
            )
        return None

    @staticmethod
    async def _user_message(
        session_id: str, injected: dict[str, ToolOutcome], prompt: str | None
    ) -> AsyncIterator[dict[str, Any]]:
        """One user message: the tool results first, then the prompt of a new task."""
        blocks: list[dict[str, Any]] = [
            {
                "type": "tool_result",
                "tool_use_id": tid,
                "content": _result_content(o),
                "is_error": o.is_error,
            }
            for tid, o in injected.items()
        ]
        if prompt:
            blocks.append({"type": "text", "text": prompt})
        yield {
            "type": "user",
            "message": {"role": "user", "content": blocks},
            "parent_tool_use_id": None,
            "session_id": session_id,
        }
