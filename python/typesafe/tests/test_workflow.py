"""End-to-end durable calls: workflow to Activity to a fake TypeSafe backend."""

from __future__ import annotations

import asyncio
import uuid
from datetime import timedelta
from typing import Any

import pytest
from typesafe_sdk import Choice, Noul, NoulAnswer, RetryPolicy, Score, ScoreAnswer

from temporalio import workflow
from temporalio.client import Client, WorkflowFailureError
from temporalio.contrib.pydantic import PydanticPayloadConverter
from temporalio.converter import (
    DataConverter,
    DefaultFailureConverterWithEncodedAttributes,
)
from temporalio.exceptions import ApplicationError, TemporalError
from temporalio.typesafe import TypeSafePlugin
from temporalio.typesafe.workflow import SystemOneResult, TemporalTypeSafe
from temporalio.worker import Replayer
from tests.helpers import new_worker
from tests.helpers.codec import CompressionCodec
from tests.helpers.fake_typesafe import (
    MODEL,
    BillingResponse,
    fake_response,
    fake_typesafe_responder,
    mock_client,
)

with workflow.unsafe.imports_passed_through():
    # typesafe_sdk reaches the sandbox through the workflow module's import
    # graph; passing it through keeps urllib.request out of module validation.
    import typesafe_sdk  # noqa: F401  # pyright: ignore[reportUnusedImport]


@workflow.defn
class SystemOneWorkflow:
    """Ask three questions of one state and return the raw typed result."""

    @workflow.run
    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        result = await TemporalTypeSafe().system_one(
            state,
            {
                "spam": Noul(instructions="Is this spam?"),
                "priority": Score(criteria=["low", "medium", "high"]),
                "target": Choice(criteria={"a": None, "b": None}),
            },
        )
        response = result.response
        return {
            "answers": {
                name: vars(answer) for name, answer in response.answers.items()
            },
            "model": response.model,
            "usage": vars(response.usage),
            "request_id": result.request_id,
        }


@workflow.defn
class ParallelSystemOneWorkflow:
    """Fan one score question out over several states."""

    @workflow.run
    async def run(self, states: list[dict[str, Any]]) -> list[dict[str, Any]]:
        typesafe = TemporalTypeSafe()
        questions = {"priority": Score(criteria=["low", "high"])}
        answers = await asyncio.gather(
            *(typesafe.system_one(state, questions) for state in states)
        )
        return [
            {name: vars(answer) for name, answer in one.response.answers.items()}
            for one in answers
        ]


@workflow.defn
class PinnedModelWorkflow:
    """Ask under an instance default and a per-call override back to back.

    The captured request bodies pin the recorded model: the instance's
    default when the call names none, and the per-call name when it does.
    """

    @workflow.run
    async def run(
        self,
        per_call: str,
    ) -> dict[str, Any]:
        typesafe = TemporalTypeSafe(model="jev-instance-pin")
        default_named = await typesafe.system_one(
            "state", {"n": Noul(instructions="Is this spam?")}
        )
        override_named = await typesafe.system_one(
            "state",
            {"n": Noul(instructions="Is this spam?")},
            model=per_call,
        )
        return {
            "instance": vars(default_named),
            "per_call": vars(override_named),
        }


@workflow.defn
class FailingSystemOneWorkflow:
    """Trigger a non-retryable backend rejection through the durable path."""

    @workflow.run
    async def run(self, _mode: str) -> dict[str, Any]:
        result = await TemporalTypeSafe(
            activity_config={"start_to_close_timeout": timedelta(seconds=5)}
        ).system_one("state", {"spam": Noul(instructions="Is this spam?")})
        return {name: vars(answer) for name, answer in result.response.answers.items()}


def test_activity_config_merges_per_call_over_instance() -> None:
    """Per-call entries win; the instance's other entries survive the merge."""
    proxy = TemporalTypeSafe(
        activity_config={
            "start_to_close_timeout": timedelta(seconds=30),
            "summary": "instance",
        }
    )
    merged = proxy._merged_config({"summary": "call", "heartbeat_timeout": None})
    assert merged.get("start_to_close_timeout") == timedelta(seconds=30)
    assert merged.get("summary") == "call"
    assert "heartbeat_timeout" in merged


def test_activity_config_adds_the_default_attempt_budget() -> None:
    """The 30s default applies only when the config names neither timeout."""
    default = TemporalTypeSafe()._merged_config(None)
    assert default.get("start_to_close_timeout") == timedelta(seconds=30)
    scheduled = TemporalTypeSafe(
        activity_config={"schedule_to_close_timeout": timedelta(minutes=1)}
    )._merged_config(None)
    assert scheduled.get("start_to_close_timeout") is None


def _plugin_with_status(status: int) -> TypeSafePlugin:
    """A plugin whose client answers everything with a canned HTTP status."""
    return TypeSafePlugin(mock_client(status=status, retry=RetryPolicy(max_retries=0)))


async def test_workflow_system_one_returns_typed_answers(client: Client) -> None:
    plugin = TypeSafePlugin(mock_client(retry=RetryPolicy(max_retries=0)))
    async with new_worker(client, SystemOneWorkflow, plugins=[plugin]) as worker:
        handle = await client.start_workflow(
            SystemOneWorkflow.run,
            {"message": "Hello"},
            id=f"system-one-one-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        result = await handle.result()

    # Cross-boundary payloads are JSON: int dict keys stringify and the `type`
    # tag stays in the activity payload, so these asserts read the plain JSON
    # result. Typed decoding is pinned in tests/test_types.py.
    assert result["answers"]["spam"]["noul"] == pytest.approx(0.8)
    assert result["answers"]["priority"]["score"] == pytest.approx(1.0)
    assert result["answers"]["priority"]["legend"] == {
        "0": "low",
        "1": "medium",
        "2": "high",
    }
    assert result["answers"]["target"]["choice"] == "a"
    assert result["model"] == "jev-1.13.0"
    assert result["usage"] == {"input_tokens": 10, "output_tokens": 2}
    assert result["request_id"] is None

    assert "temporalio.typesafe.system_one" in await _activity_names(handle)
    history = await handle.fetch_history()
    await Replayer(workflows=[SystemOneWorkflow], plugins=[plugin]).replay_workflow(
        history
    )


@workflow.defn
class ResultWorkflow:
    """Return the plugin's own result object across the workflow boundary."""

    @workflow.run
    async def run(self, state: dict[str, Any]) -> SystemOneResult:
        return await TemporalTypeSafe().system_one(
            state,
            {
                "spam": Noul(instructions="Is this spam?"),
                "priority": Score(criteria=["low", "high"]),
            },
        )


@workflow.defn
class BillingWorkflow:
    """Decode through a registered response-model subclass end to end."""

    @workflow.run
    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        result = await TemporalTypeSafe().system_one(
            state,
            {"billing": Noul(instructions="Is this about billing?")},
            response_model="billing-boundary",
        )
        answer = result.response.answers["billing"]
        assert isinstance(answer, NoulAnswer)
        return {"billing": answer.noul}


async def test_registered_response_model_survives_the_activity_boundary(
    client: Client,
) -> None:
    plugin = TypeSafePlugin(
        mock_client(retry=RetryPolicy(max_retries=0)),
        response_models={"billing-boundary": BillingResponse},
    )
    async with new_worker(client, BillingWorkflow, plugins=[plugin]) as worker:
        handle = await client.start_workflow(
            BillingWorkflow.run,
            {"message": "Invoice"},
            id=f"system-one-billing-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        result = await handle.result()
        history = await handle.fetch_history()

    assert result["billing"] == pytest.approx(0.8)
    await Replayer(workflows=[BillingWorkflow], plugins=[plugin]).replay_workflow(
        history
    )


@pytest.mark.parametrize("use_codec", [False, True])
async def test_workflow_returns_typed_system_one_result(
    client: Client, use_codec: bool
) -> None:
    """The public result round-trips with the caller's codec and integer score keys.

    The plugin reconfigures the Client, so both the workflow's encoding and the
    client's decoding of the return annotation go through the Pydantic
    converter, restoring native score-map integer keys.
    """
    codec = CompressionCodec() if use_codec else None
    supplied_converter = DataConverter(
        payload_codec=codec,
        failure_converter_class=DefaultFailureConverterWithEncodedAttributes,
    )
    plugin = TypeSafePlugin(
        mock_client(
            headers={"x-typesafe-request-id": "req-typed-result"},
            retry=RetryPolicy(max_retries=0),
        )
    )
    config = client.config()
    config["data_converter"] = supplied_converter
    config["plugins"] = [plugin]
    typed_client = Client(**config)
    assert (
        typed_client.data_converter.payload_converter_class is PydanticPayloadConverter
    )
    assert typed_client.data_converter.payload_codec is codec
    assert (
        typed_client.data_converter.failure_converter_class
        is DefaultFailureConverterWithEncodedAttributes
    )
    async with new_worker(typed_client, ResultWorkflow) as worker:
        handle = await typed_client.start_workflow(
            ResultWorkflow.run,
            {"message": "Hello"},
            id=f"system-one-raw-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        result = await handle.result()
        history = await handle.fetch_history()
    await Replayer(
        workflows=[ResultWorkflow],
        plugins=[plugin],
        data_converter=supplied_converter,
    ).replay_workflow(history)

    [completed_payload] = [
        event.workflow_execution_completed_event_attributes.result.payloads[0]
        for event in history.events
        if event.HasField("workflow_execution_completed_event_attributes")
    ]
    assert completed_payload.metadata["encoding"] == (
        b"binary/zlib" if use_codec else b"json/plain"
    )

    assert isinstance(result, SystemOneResult)
    assert result.request_id == "req-typed-result"
    assert result.response.model == MODEL
    spam = result.response.answers["spam"]
    assert isinstance(spam, NoulAnswer) and spam.noul == pytest.approx(0.8)
    priority = result.response.answers["priority"]
    assert isinstance(priority, ScoreAnswer)
    assert priority.probabilities == {0: 0.5, 1: 0.5}
    assert priority.legend == {0: "low", 1: "high"}
    assert all(type(key) is int for key in priority.probabilities)
    assert all(type(key) is int for key in priority.legend)


async def test_workflow_with_injected_client_survives_a_stopped_worker(
    client: Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The plugin keeps calling through one injected client and never closes it."""
    injected = mock_client(retry=RetryPolicy(max_retries=0))
    closes: list[None] = []

    async def spy() -> None:
        closes.append(None)

    monkeypatch.setattr(injected, "aclose", spy)
    plugin = TypeSafePlugin(injected)

    async with new_worker(client, SystemOneWorkflow, plugins=[plugin]) as worker:
        handle = await client.start_workflow(
            SystemOneWorkflow.run,
            {"message": "First"},
            id=f"system-one-injected-first-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        await handle.result()

    async with new_worker(client, SystemOneWorkflow, plugins=[plugin]) as worker:
        handle = await client.start_workflow(
            SystemOneWorkflow.run,
            {"message": "Second"},
            id=f"system-one-injected-second-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        result = await handle.result()

    assert closes == []
    assert result["model"] == MODEL


async def test_workflow_records_backend_request_id(client: Client) -> None:
    plugin = TypeSafePlugin(
        mock_client(
            headers={"x-typesafe-request-id": "req_durable"},
            retry=RetryPolicy(max_retries=0),
        )
    )
    async with new_worker(client, SystemOneWorkflow, plugins=[plugin]) as worker:
        handle = await client.start_workflow(
            SystemOneWorkflow.run,
            {"message": "Hello"},
            id=f"system-one-request-id-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        result = await handle.result()

    assert result["request_id"] == "req_durable"
    history = await handle.fetch_history()
    await Replayer(workflows=[SystemOneWorkflow], plugins=[plugin]).replay_workflow(
        history
    )


async def test_system_one_records_named_models_into_requests(client: Client) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(body: dict[str, Any]) -> Any:
        bodies.append(body)
        return fake_response(
            200,
            {
                "model": "jev-pin",
                "answers": {"n": {"type": "noul", "noul": 0.5}},
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    captured_client = fake_typesafe_responder(
        handler, model="jev-pin", retry=RetryPolicy(max_retries=0)
    )
    plugin = TypeSafePlugin(captured_client)
    async with new_worker(client, PinnedModelWorkflow, plugins=[plugin]) as worker:
        handle = await client.start_workflow(
            PinnedModelWorkflow.run,
            "jev-per-call",
            id=f"system-one-pinned-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        await handle.result()

    assert bodies[0]["model"] == "jev-instance-pin"
    assert bodies[1]["model"] == "jev-per-call"


async def test_parallel_system_one_calls_preserve_duplicate_states(
    client: Client,
) -> None:
    requested_states: list[dict[str, Any]] = []

    def handler(body: dict[str, Any]) -> Any:
        state = body["state"]
        requested_states.append(state)
        return fake_response(
            body={
                "model": MODEL,
                "answers": {
                    "priority": {
                        "type": "score",
                        "score": float(state["n"]),
                        "confidence": 0.85,
                        "probabilities": {
                            "0": float(1 - state["n"]),
                            "1": float(state["n"]),
                        },
                        "legend": {"0": "low", "1": "high"},
                    }
                },
                "usage": {"input_tokens": 10, "output_tokens": 2},
            }
        )

    plugin = TypeSafePlugin(
        fake_typesafe_responder(handler, retry=RetryPolicy(max_retries=0))
    )
    states = [{"n": 0}, {"n": 1}, {"n": 1}]
    async with new_worker(
        client, ParallelSystemOneWorkflow, plugins=[plugin]
    ) as worker:
        handle = await client.start_workflow(
            ParallelSystemOneWorkflow.run,
            states,
            id=f"system-one-many-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        result = await handle.result()

    assert [one["priority"]["score"] for one in result] == [
        pytest.approx(0),
        pytest.approx(1),
        pytest.approx(1),
    ]
    assert sorted(state["n"] for state in requested_states) == [0, 1, 1]
    assert (await _activity_names(handle)).count("temporalio.typesafe.system_one") == 3
    await Replayer(
        workflows=[ParallelSystemOneWorkflow], plugins=[plugin]
    ).replay_workflow(await handle.fetch_history())


async def test_non_retryable_backend_rejection_fails_workflow(
    client: Client,
) -> None:
    plugin = _plugin_with_status(422)
    async with new_worker(client, FailingSystemOneWorkflow, plugins=[plugin]) as worker:
        with pytest.raises(WorkflowFailureError) as failure:
            await client.execute_workflow(
                FailingSystemOneWorkflow.run,
                "422",
                id=f"system-one-fail-{uuid.uuid4()}",
                task_queue=worker.task_queue,
            )
    root = failure.value
    found: list[ApplicationError] = []
    while isinstance(root, TemporalError):
        if isinstance(root, ApplicationError):
            found.append(root)
        root = root.cause  # type: ignore[assignment]
    [ours] = [e for e in found if e.type == "TypeSafeRequestValidationError"]
    # The SDK's own TypeSafeUnprocessableEntityError rides the chain as a
    # reconstructed ApplicationError whose message carries the request label;
    # MockTransport never opens a socket, so that label is fixture text.
    assert ours.message == "TypeSafe API returned status 422"
    assert ours.non_retryable


async def _activity_names(handle: Any) -> list[str]:
    names: list[str] = []
    async for event in handle.fetch_history_events():
        if event.HasField("activity_task_scheduled_event_attributes"):
            names.append(
                event.activity_task_scheduled_event_attributes.activity_type.name
            )
    return names
