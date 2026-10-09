"""Wire-payload production and response reconstruction against native models."""

from __future__ import annotations

from typing import Any, cast

import pytest
from typesafe_sdk import (
    Choice,
    ChoiceAnswer,
    Noul,
    Score,
    SystemOneResponse,
    TypeSafeError,
)

from temporalio.typesafe._types import (
    SystemOneResult,
    lookup_response_model,
    register_response_model,
)
from temporalio.typesafe._workflow import _decode_result, _questions_payload
from tests.conftest import question_payload
from tests.helpers.fake_typesafe import MODEL, answer_for

_TAGGED_MODEL_NAME = "test-types-tagged"


class TaggedResponse(SystemOneResponse):
    tag: str


def test_payload_drops_unset_optionals() -> None:
    assert question_payload(Noul(instructions="Is this spam?")) == {
        "type": "noul",
        "instructions": "Is this spam?",
    }


def test_payload_encodes_choice_and_score_types() -> None:
    choice = question_payload(Choice(criteria={"a": "first", "b": "second"}))
    assert choice["type"] == "choice"
    assert choice["criteria"] == {"a": "first", "b": "second"}
    score = question_payload(Score(criteria=["sparse", "full"]))
    assert score["type"] == "score"
    assert score["criteria"] == ["sparse", "full"]


def test_payload_passes_dict_questions_through() -> None:
    question: dict[str, Any] = {"type": "noul", "instructions": "raw", "extra": 1}
    assert question_payload(dict(question)) == question


def test_payload_rejects_unsupported_question() -> None:
    with pytest.raises(TypeError, match="unsupported question"):
        _questions_payload({"x": cast(Any, "not-a-question")})


def test_lookup_returns_none_for_default_parsing() -> None:
    assert lookup_response_model(None) is None


def test_register_and_lookup_round_trip() -> None:
    register_response_model("round-trip", SystemOneResponse)
    assert lookup_response_model("round-trip") is SystemOneResponse


def test_duplicate_name_with_other_class_rejects() -> None:
    register_response_model("solo", SystemOneResponse)

    class Other(SystemOneResponse):
        pass

    with pytest.raises(TypeSafeError, match="already registered"):
        register_response_model("solo", Other)


def test_missing_name_raises_typesafe_error() -> None:
    with pytest.raises(TypeSafeError, match="not registered"):
        lookup_response_model("never-registered")


def _response_body() -> dict[str, object]:
    """One well-typed request payload, exactly as the Activity records it."""
    question = {"type": "choice", "criteria": {"a": "first"}}
    return {
        "model": MODEL,
        "answers": {"verdict": answer_for(question)},
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }


def test_decode_result_rebuilds_default_response() -> None:
    result = _decode_result(
        {**_response_body(), "request_id": "req-1"},
        None,
    )
    assert isinstance(result, SystemOneResult)
    assert result.request_id == "req-1"
    assert result.response.model == MODEL
    answer = result.response.answers["verdict"]
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice == "a"


def test_decode_result_omits_absent_request_id() -> None:
    result = _decode_result(_response_body(), None)
    assert result.request_id is None


def test_decode_result_resolves_registered_subclass() -> None:
    register_response_model(_TAGGED_MODEL_NAME, TaggedResponse)
    result = _decode_result(
        {**_response_body(), "tag": "v"},
        _TAGGED_MODEL_NAME,
    )
    assert type(result.response) is TaggedResponse
    assert result.response.tag == "v"


def test_decode_result_requires_the_subclass_field() -> None:
    register_response_model(_TAGGED_MODEL_NAME, TaggedResponse)
    with pytest.raises(ValueError, match="tag"):
        _decode_result(_response_body(), _TAGGED_MODEL_NAME)
