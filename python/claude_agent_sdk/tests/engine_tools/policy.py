"""Scripted behavior that calls Claude Code tools."""

from __future__ import annotations

import re

from temporalio.claude_agent_sdk.testing import Final, HistoryItem, ToolCall


def shell_policy(prompt: str, history: list[HistoryItem]) -> ToolCall | Final:
    """'run: <command>' runs it with Bash, 'note: <text>' calls the notes MCP tool,
    'count then run: <command>' counts first. The answer is what the tool returned."""
    count = re.match(r"count then run: (.*)", prompt, re.S)
    if count and not any(h.name == "count" for h in history):
        return ToolCall("count", {"n": 1})
    match = re.match(r"(?:count then )?run: (.*)", prompt, re.S)
    note = re.match(r"note: (.*)", prompt, re.S)
    name = "Bash" if match else "mcp__notes__add_note"
    done = [h for h in history if h.name == name]
    if not done:
        if match:
            return ToolCall("Bash", {"command": match.group(1), "description": "run"})
        return ToolCall(name, {"text": note.group(1) if note else ""})
    last = done[-1]
    return Final(f"{'error: ' if last.is_error else ''}{last.content}")


def together_policy(prompt: str, history: list[HistoryItem]) -> list[ToolCall] | Final:
    """'together: <command>' calls Bash and ``count`` in one message (in the order
    'bash first' or 'count first' says); a call that did not run is asked again."""
    command = prompt.split(": ", 1)[1]
    bash = [h for h in history if h.name == "Bash" and not h.is_error]
    counted = [h for h in history if h.name == "count" and not h.is_error]
    if bash and counted:
        return Final(f"{bash[-1].content} and counted {counted[-1].content['n']}")
    calls = []
    if not bash:
        calls.append(ToolCall("Bash", {"command": command, "description": "run"}))
    if not counted:
        calls.append(ToolCall("count", {"n": 1}))
    return calls if prompt.startswith("bash first") else calls[::-1]
