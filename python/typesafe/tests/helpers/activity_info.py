"""Activity ``Info`` stubs for the deadline-clamp tests."""

from datetime import datetime, timedelta, timezone

from temporalio.activity import Info
from temporalio.common import Priority


def fake_info(
    start_to_close: timedelta | None,
    schedule_to_close: timedelta | None = None,
    elapsed: timedelta = timedelta(),
) -> Info:
    """An ``Info`` carrying only the deadline fields the clamp reads.

    ``elapsed`` moves the attempt's start into the past.
    """
    started = datetime.now(timezone.utc) - elapsed
    return Info(
        activity_id="1",
        activity_type="temporalio.typesafe.system_one",
        attempt=1,
        current_attempt_scheduled_time=started,
        heartbeat_details=(),
        heartbeat_timeout=None,
        is_local=False,
        namespace="default",
        schedule_to_close_timeout=schedule_to_close,
        scheduled_time=started,
        start_to_close_timeout=start_to_close,
        started_time=started,
        task_queue="test",
        task_token=b"",
        workflow_id="wf",
        workflow_namespace="default",
        workflow_run_id="run",
        workflow_type="wf",
        priority=Priority(),
        retry_policy=None,
    )
