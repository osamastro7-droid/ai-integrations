"""TypeSafe decision API for Temporal workflows.

Questions and answers are the typesafe SDK's own ``Choice``, ``Noul`` and
``Score`` models; the workflow stays with durable Activities. Worker code
registers :class:`~temporalio.typesafe.TypeSafePlugin`; workflow code imports
:class:`~temporalio.typesafe.workflow.TemporalTypeSafe`. The wire is
protocol-level: Jev, Laya, and Ollaya differ only by ``base_url`` and
credentials.

Every public symbol loads lazily, so importing the workflow module never
executes worker-only imports.

This package is Pre-release and may change in future versions.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from temporalio.typesafe._plugin import TypeSafePlugin
    from temporalio.typesafe._types import SystemOneResult
    from temporalio.typesafe._workflow import TemporalTypeSafe

__all__ = [
    "SystemOneResult",
    "TemporalTypeSafe",
    "TypeSafePlugin",
]


def __getattr__(name: str) -> Any:
    """Load each public API symbol without importing worker-only code into workflows."""
    if name == "TypeSafePlugin":
        from temporalio.typesafe import _plugin

        return getattr(_plugin, name)
    if name == "SystemOneResult":
        from temporalio.typesafe import _types

        return getattr(_types, name)
    if name == "TemporalTypeSafe":
        from temporalio.typesafe import _workflow

        return getattr(_workflow, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return public API names for interactive completion."""
    return list(__all__)
