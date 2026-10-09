"""HTTP-semantics to ApplicationError translation, raised in place."""

from __future__ import annotations

import datetime

import httpx2
import pytest
from typesafe_sdk import (
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
    TypeSafeError,
)

from temporalio.typesafe._errors import retry_after_delay, translate_exception


def _api_error(
    status: int, body: object = None, headers: dict[str, str] | None = None
) -> TypeSafeAPIError:
    return TypeSafeAPIError(status, body, httpx2.Headers(headers or {}))


def test_retryable_statuses_map_to_retryable_application_error() -> None:
    from temporalio.exceptions import ApplicationError

    for status in (408, 429, 500, 502, 529):
        with pytest.raises(ApplicationError) as err:
            translate_exception(_api_error(status))
        assert err.value.non_retryable is False, status


def test_malformed_request_is_non_retryable() -> None:
    from temporalio.exceptions import ApplicationError

    for status in (400, 422, 401, 403, 404):
        with pytest.raises(ApplicationError) as err:
            translate_exception(_api_error(status, {"message": "bad"}))
        assert err.value.non_retryable is True
        assert err.value.type is not None and err.value.type.startswith("TypeSafe")


def test_response_validation_error_preserves_field_path() -> None:
    from temporalio.exceptions import ApplicationError

    err = TypeSafeAPIResponseValidationError(
        200,
        {},
        httpx2.Headers({}),
        "answers.billing.type",
    )
    with pytest.raises(ApplicationError) as raised:
        translate_exception(err)
    assert raised.value.non_retryable is True
    assert "answers.billing.type" in raised.value.message


def test_local_sdk_error_is_client_error() -> None:
    from temporalio.exceptions import ApplicationError

    with pytest.raises(ApplicationError) as err:
        translate_exception(TypeSafeError("questions are empty"))
    assert err.value.non_retryable is True
    assert err.value.type == "TypeSafeClientError"


def test_retry_after_ms_becomes_next_retry_delay() -> None:
    err = _api_error(429, headers={"retry-after-ms": "250"})
    assert retry_after_delay(err) == datetime.timedelta(milliseconds=250)


def test_retry_after_seconds_converted() -> None:
    err = _api_error(503, headers={"retry-after": "3"})
    assert retry_after_delay(err) == datetime.timedelta(seconds=3)


def test_retry_after_http_date_converted_from_now() -> None:
    future = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
        seconds=7
    )
    err = _api_error(
        429, headers={"retry-after": future.strftime("%a, %d %b %Y %H:%M:%S GMT")}
    )
    delay = retry_after_delay(err)
    assert delay is not None
    assert 0 < delay.total_seconds() <= 7.5


def test_missing_or_negative_retry_after_is_none() -> None:
    assert retry_after_delay(_api_error(500)) is None
    negative = _api_error(429, headers={"retry-after": "-5"})
    assert retry_after_delay(negative) == datetime.timedelta(0)
