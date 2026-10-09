"""HTTP status to Temporal retry behavior.

The mapping keys off HTTP status only, so it does not depend on a vendor's
exception class. The protocol's ``retry-after``/``retry-after-ms`` hint becomes
Temporal's ``next_retry_delay``, leaving timing to the caller's RetryPolicy.

Retryable: connection errors, timeouts, 5xx, 429, and the "overloaded" 529.
Non-retryable: 422 and 400 rejections, 401/403 auth, 404 missing model, and
local SDK misuse.
"""

from __future__ import annotations

from datetime import timedelta
from email.utils import parsedate_to_datetime
from typing import Any, NoReturn

from typesafe_sdk import (
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
    TypeSafeError,
)

from temporalio.exceptions import ApplicationError

_RETRY_AFTER_MS = "retry-after-ms"
_RETRY_AFTER = "retry-after"

_RETRYABLE_STATUS = {408, 429, 529, *range(500, 600)}
_STATUS_TYPE = {
    400: "TypeSafeBadRequest",
    401: "TypeSafeUnauthorized",
    403: "TypeSafeForbidden",
    404: "TypeSafeModelNotFound",
    422: "TypeSafeRequestValidationError",
}


def translate_exception(err: BaseException, *details: Any) -> NoReturn:
    """Convert a failure into a Temporal ApplicationError, raised in place.

    A 200 whose body fails the schema
    (``TypeSafeAPIResponseValidationError``) is a contract problem, so it is
    non-retryable and keeps the SDK's ``field_path``. The backend's
    ``request_id`` joins the details as ``{"request-id": …}`` when reported.
    """
    request_id = getattr(err, "request_id", None)
    if request_id is not None:
        details = (*details, {"request-id": request_id})
    if isinstance(err, TypeSafeAPIResponseValidationError):
        raise ApplicationError(
            f"TypeSafe returned invalid response data at {err.field_path}",
            *details,
            err.field_path,
            type="TypeSafeResponseValidationError",
            non_retryable=True,
        ) from err
    if isinstance(err, TypeSafeAPIError) and err.status not in _RETRYABLE_STATUS:
        non_retryable = True
        error_type = _STATUS_TYPE.get(err.status, "TypeSafeServerError")
    elif isinstance(err, TypeSafeAPIConnectionError | TypeSafeAPIError):
        non_retryable = False
        error_type = (
            "TypeSafeConnectionError"
            if not isinstance(err, TypeSafeAPIError)
            else f"TypeSafeHttpError{err.status}"
        )
    elif isinstance(err, TypeSafeError):
        # Local misuse never fixes itself.
        raise ApplicationError(
            str(err), *details, type="TypeSafeClientError", non_retryable=True
        ) from err
    else:
        non_retryable = False
        error_type = "TypeSafeError"

    delay = None if non_retryable else retry_after_delay(err)
    raise ApplicationError(
        _reason(err),
        *details,
        type=error_type,
        non_retryable=non_retryable,
        next_retry_delay=delay,
    ) from err


def retry_after_delay(err: BaseException) -> timedelta | None:
    """The protocol's Retry-After hint as a Temporal delay.

    ``retry-after-ms`` is milliseconds; ``retry-after`` is seconds or an
    HTTP-date. Both are zero-clamped, since a negative delay retries hot.
    """
    headers = getattr(err, "headers", None)
    if headers is None:
        return None
    raw_ms = headers.get(_RETRY_AFTER_MS)
    if raw_ms is not None:
        try:
            return timedelta(milliseconds=max(0.0, float(raw_ms)))
        except ValueError:
            pass
    raw_seconds = headers.get(_RETRY_AFTER)
    if raw_seconds is not None:
        try:
            return timedelta(seconds=max(0.0, float(raw_seconds)))
        except ValueError:
            pass
        try:
            seconds = parsedate_to_datetime(raw_seconds).timestamp()
            return timedelta(seconds=max(0.0, seconds - _now().timestamp()))
        except (ValueError, TypeError, OverflowError):
            pass
    return None


def _reason(err: BaseException) -> str:
    """Describe the failure for the activity's history entry."""
    body = getattr(err, "body", None)
    status = getattr(err, "status", None)
    if status is not None and body:
        return f"TypeSafe API returned status {status}"
    return str(err) or type(err).__name__


def _now() -> Any:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)
