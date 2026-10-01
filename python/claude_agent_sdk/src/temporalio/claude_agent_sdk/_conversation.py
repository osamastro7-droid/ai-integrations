"""The conversation the Workflow holds: the Query that serves it, and the step side.

Without a session store, the Claude session's transcript lives in the Workflow, like
the conversation of the other AI integrations in this repository. A segment reads the
committed transcript with a Query on its own Workflow (so the transcript is not
copied into every step's input, and the history grows with what each step adds, not
with the whole conversation), seeds an in-memory session store with it, and returns
what it changed. Every attempt starts from the committed transcript, so nothing a
failed attempt wrote can reach Claude, and any Worker can run any step.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from temporalio import activity

from ._models import ConversationPage, SegmentInput, SegmentOutput, ToolStepInput

QUERY = "__temporal_claude_agent_transcript"
"""The Workflow Query that returns the conversation, a page at a time."""

PAGE_BYTES = 1024 * 1024
"""Most bytes of entries in one page (Temporal checks a Query result against 2 MiB)."""

PAYLOAD_LIMIT_BYTES = 2 * 1024 * 1024 - 64 * 1024
"""Most bytes one payload may carry without External Storage.

Temporal refuses payloads over 2 MiB by default; 64 KiB stay free for the encoding
and for a payload codec.
"""


def json_bytes(value: Any) -> int:
    r"""Size of ``value`` as Temporal's default JSON converter writes it, in bytes.

    The converter escapes non-ASCII characters (``\uXXXX``), so text in, for
    example, Arabic or Chinese takes two to three times its UTF-8 size.
    """
    return len(json.dumps(value, separators=(",", ":"), default=str))


def entry_bytes(entry: Any) -> int:
    """Size of one transcript entry in a payload, in bytes."""
    return json_bytes(entry)


def page(
    entries: list[dict[str, Any]],
    sizes: list[int],
    start: int,
    limit: int = PAGE_BYTES,
) -> ConversationPage:
    """The entries from ``start`` that fit in ``limit`` bytes (always at least one).

    Args:
        entries: The whole conversation.
        sizes: The size of each entry (``entry_bytes``).
        start: The first entry to return.
        limit: Most bytes in the page.

    Returns:
        The page and the conversation's length.
    """
    end = start = max(0, start)
    used = 0
    while end < len(entries) and (end == start or used + sizes[end] <= limit):
        used += sizes[end]
        end += 1
    return ConversationPage(entries=entries[start:end], total=len(entries))


async def read_conversation(
    inp: SegmentInput | ToolStepInput,
) -> list[dict[str, Any]]:
    """The committed conversation a step continues: inline, or read from the Workflow.

    Args:
        inp: The segment (or tool step) input.

    Returns:
        The transcript entries, oldest first (empty for a new conversation).

    Raises:
        RuntimeError: If the Workflow did not serve the conversation it announced
            (Temporal retries the step), or a conversation the Workflow holds is read
            outside the segment Activity.
    """
    if inp.transcript is not None:
        return list(inp.transcript)
    ref = inp.conversation
    if ref is None or ref.entries == 0:
        return []
    if not activity.in_activity():
        raise RuntimeError(
            "A conversation the Workflow holds can only be read inside the segment "
            "Activity; pass SegmentInput.transcript to run a segment directly."
        )
    info = activity.info()
    if info.workflow_id is None:  # a standalone Activity has no Workflow to ask
        raise RuntimeError(
            "This segment Activity was not started by a Workflow, so no Workflow "
            "holds its conversation; pass SegmentInput.transcript instead."
        )
    handle = activity.client().get_workflow_handle(
        info.workflow_id, run_id=info.workflow_run_id
    )
    entries: list[dict[str, Any]] = []
    while len(entries) < ref.entries:
        got = await handle.query(
            ref.query, args=[ref.agent, len(entries)], result_type=ConversationPage
        )
        if not got.entries or got.total != ref.entries:
            raise RuntimeError(
                f"The Workflow has {got.total} conversation entries (page at "
                f"{len(entries)}: {len(got.entries)}), but this step was scheduled "
                f"with {ref.entries}. Retrying."
            )
        entries.extend(got.entries)
    return entries


def external_storage_on() -> bool:
    """Whether this Worker's data converter moves large payloads to External Storage."""
    if not activity.in_activity():
        return False
    try:
        return activity.client().data_converter.external_storage is not None
    except RuntimeError:  # no client, for example in an ActivityEnvironment
        return False


def too_large(out: SegmentOutput) -> str | None:
    """Explain why ``out`` cannot be recorded as one payload, or None if it can."""
    if out.external_storage:
        return None
    size = json_bytes(dataclasses.asdict(out))
    if size <= PAYLOAD_LIMIT_BYTES:
        return None
    return (
        f"This step's output is {size / 1024 / 1024:.1f} MB with what it added to the "
        "conversation, more than one Temporal payload holds without External Storage "
        "(2 MB by default), so the step was not committed. Configure External "
        "Storage on the Client (see the README), or give the runner a session store."
    )
