"""PreToolUse command hook for the Claude Code engine (settings based, not in process).

The engine runs this file as a plain script (standard library only, so it starts in
tens of milliseconds) for every durable tool call. It always answers "defer", so the
Workflow runs the tool as a Temporal Activity. It never answers "allow": on a resumed paused call, "allow" sends
the engine down its auto-resume path, where a later "defer" is ignored.

Parallel calls: the engine keeps only one paused call per run. So the first new
durable call in a run is deferred, and any other call after it in the same run,
durable or built-in, is denied. The Workflow runs the denied durable calls with the
paused one, and the next segment puts their real results in place of the denials. A
denied built-in call keeps the message, so Claude calls it again. (A built-in call
allowed to run after the pause would have its result cut from the session, and
Claude would run it again.)

Stopped runs: when the segment Activity is cancelled or times out, the runner writes
``$TCA_HOOK_DIR/stop``, so an engine that is still shutting down cannot start another
built-in tool (a durable call still defers: that ends the run, and no one runs it).
If the folder is missing (the run ended, or the engine cannot see the Worker's
temporary folder), every call is denied, and the runner fails a step that finished
that way.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

NOT_RUN = (
    "This call did not run: another tool call in this message paused the run. "
    "Call this tool again after you get that result."
)
"""The reason Claude sees for a built-in call denied after the pause."""

STOPPED = "This step was stopped (cancelled or timed out). Do not call tools."
"""The reason Claude sees when the segment is no longer running (the runner looks for it)."""


DURABLE_PREFIX = "mcp__durable__"
"""Names of durable tools as the engine sees them (the runner's ``PREFIX``)."""


def _deny(reason: str = NOT_RUN) -> dict[str, Any]:
    return {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": reason,
    }


def decide(event: dict[str, Any]) -> dict[str, Any]:
    """Return the ``hookSpecificOutput`` for one PreToolUse event.

    Also records the paused call in ``$TCA_HOOK_DIR/paused_call``, so the runner can
    check that the engine really paused there.

    Args:
        event: The PreToolUse event the engine sent on stdin.

    Returns:
        The hook decision.
    """
    tool_use_id = str(event.get("tool_use_id") or "")
    answered = set(os.environ.get("TCA_ANSWERED_IDS", "").split())
    run_dir = os.environ.get("TCA_HOOK_DIR")
    marker = os.path.join(run_dir, "paused_call") if run_dir else None
    output: dict[str, Any]
    if run_dir is not None and not os.path.isdir(run_dir):
        output = _deny(STOPPED)  # the run ended, or this hook cannot see its folder
    elif not str(event.get("tool_name") or "").startswith(DURABLE_PREFIX):
        # A built-in tool runs normally, unless the step was stopped or a durable
        # call already paused this run.
        if run_dir is not None and os.path.exists(os.path.join(run_dir, "stop")):
            output = _deny(STOPPED)
        elif marker is not None and os.path.exists(marker):
            output = _deny()
        else:
            output = {"hookEventName": "PreToolUse"}
    else:
        output = {"hookEventName": "PreToolUse", "permissionDecision": "defer"}
        # On resume the engine re-announces the call that was just answered. It must
        # not take the "one paused call per run" slot, or Claude's next call is denied.
        if marker is not None and tool_use_id not in answered:
            try:
                # Atomic: exactly one call wins the slot.
                fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(tool_use_id)
            except FileExistsError:
                with open(marker, encoding="utf-8") as handle:
                    if handle.read().strip() != tool_use_id:
                        output = _deny()
    log = os.environ.get("TCA_HOOK_LOG")
    if log:  # debugging aid: one line per decision
        with open(log, "a", encoding="utf-8") as handle:
            decision = output.get("permissionDecision", "none")
            handle.write(f"{tool_use_id} {decision}\n")
    return output


def main() -> None:
    """Read one event from stdin and print the decision.

    The engine sends UTF-8; read bytes, so a Windows code page cannot garble or
    reject the tool input. The answer is ASCII (``json.dumps`` escapes the rest).
    """
    event = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    print(json.dumps({"hookSpecificOutput": decide(event)}))


if __name__ == "__main__":
    main()
