"""Workflow-side durable TypeSafe calls.

Workflow code imports :class:`TemporalTypeSafe` and :class:`SystemOneResult` from
here. The module imports only workflow-side code, so a sandboxed workflow
never executes the plugin's worker-only imports.
"""

from __future__ import annotations

from temporalio.typesafe._types import SystemOneResult
from temporalio.typesafe._workflow import TemporalTypeSafe

__all__ = [
    "SystemOneResult",
    "TemporalTypeSafe",
]
