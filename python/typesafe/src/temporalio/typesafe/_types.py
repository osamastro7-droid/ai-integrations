"""Serializable glue between the SDK's native models and the Activity wire.

Questions and answers stay the SDK's own ``Choice``/``Noul``/``Score`` and
``SystemOneResponse`` pydantic models; nothing here duplicates their shape.
The registry maps a durable name to a worker-validated
``SystemOneResponse`` subclass, exactly the design the S1 reviewer thread
suggested.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from typesafe_sdk import SystemOneResponse, TypeSafeError

_RESPONSE_MODELS: dict[str, type[SystemOneResponse]] = {}
"""Durable name to ``SystemOneResponse`` subclass, shared worker and workflow side."""


def register_response_model(
    name: str,
    model: type[SystemOneResponse],
) -> None:
    """Register ``model`` under a durable ``name`` the workflow calls with.

    The same class may re-register (plugins are rebuilt per Worker); two
    different classes sharing a name reject, so a stale Worker process never
    decodes history against the wrong schema.
    """
    existing = _RESPONSE_MODELS.get(name)
    if existing is not None and existing is not model:
        raise TypeSafeError(
            f"response model name {name!r} is already registered by "
            f"{existing.__name__!r}"
        )
    _RESPONSE_MODELS[name] = model


def lookup_response_model(name: str | None) -> type[SystemOneResponse] | None:
    """The named registry entry, or ``None`` for the SDK's default parsing."""
    if name is None:
        return None
    model = _RESPONSE_MODELS.get(name)
    if model is None:
        # TypeSafeError, so the Activity's error translation makes this a
        # non-retryable TypeSafeClientError instead of a retry loop.
        raise TypeSafeError(f"response model {name!r} is not registered")
    return model


@dataclasses.dataclass(kw_only=True)
class SystemOneInput:
    """The ``system_one`` Activity's single input: one unified JSON payload.

    Keyword-only, and declared to open with the model name.
    """

    model_name: str | None = None
    state: Any
    questions: dict[str, Any]
    response_model: str | None = None


@dataclasses.dataclass
class SystemOneResult:
    """One request's native response plus its backend request ID.

    ``response`` is the SDK's own ``SystemOneResponse`` (answers, served
    model, and token usage), rebuilt from the recorded history payload. The
    ``request_id`` is the provider's correlation ID reported alongside it.
    """

    response: SystemOneResponse
    request_id: str | None = None
