"""Interaction events: what a rater did on a subject, and events of the run. See
docs/rater_app.md, "Event logging".

The types and the limits of a batch are in ``app.schemas`` (``EVENT_TYPES``,
``MAX_EVENTS_PER_CALL``, ``MAX_EVENT_PAYLOAD_BYTES``). The event of a skip is written by
``app.db.record_skip``, not here: a page cannot send one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from sqlmodel import Session, col, select

from surfcensus.app.db import utc_now
from surfcensus.app.models import Assignment, Event
from surfcensus.app.schemas import SERVER_EVENT_TYPES, EventIn

CONFIG_CHANGE = "config_change"
"""The event of the run that says a setting changed: ``{"setting": ..., "old": ...,
"new": ...}``."""


def payload_json(payload: Mapping[str, object]) -> str:
    """A payload as the ``events`` table holds it: compact JSON with sorted keys."""
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def store_events(session: Session, assignment: Assignment, events: Sequence[EventIn]) -> int:
    """Store a batch of events of one assignment as they were sent, the payload as compact
    JSON with sorted keys; returns how many. An unknown type is stored as given. Not
    committed.

    The rows are added in the order of the batch, so ``event_id`` keeps the order in which
    a page logged them. ``ValueError`` for an assignment without an id.
    """
    assignment_id = assignment.assignment_id
    if assignment_id is None:
        raise ValueError("the assignment has no id yet: flush it first")
    session.add_all(
        Event(
            assignment_id=assignment_id,
            ts=event.ts,
            type=event.type,
            payload_json=payload_json(event.payload),
        )
        for event in events
    )
    session.flush()
    return len(events)


def record_run_event(
    session: Session, type: str, payload: Mapping[str, object], *, now: str | None = None
) -> Event:
    """Store an event of the run, without an assignment: a changed setting
    (``config_change``, with the setting, its old and its new value). Not committed.

    ``ts`` is ``now``, else the present time. ``ValueError`` for an empty type and for a
    type that belongs to an assignment (``SERVER_EVENT_TYPES``).
    """
    if not type:
        raise ValueError("an event has a type")
    if type in SERVER_EVENT_TYPES:
        raise ValueError(f"an event of type {type!r} belongs to an assignment")
    event = Event(
        assignment_id=None,
        ts=utc_now() if now is None else now,
        type=type,
        payload_json=payload_json(payload),
    )
    session.add(event)
    session.flush()
    return event


def record_setting(
    session: Session, setting: str, value: object, *, now: str | None = None
) -> Event | None:
    """Record the value a setting of the app has now, when it is not the one the events of
    the run say it has: a ``config_change`` with ``old`` (None at the first record) and
    ``new``. Returns the event, or None when nothing changed. Not committed.

    The server calls this when it starts, for the settings that decide what a rating
    means (whether the queue is blind, whether marks are required, the snap radius), so
    the database says under which of them each submission was made.
    """
    rows = session.exec(
        select(Event.payload_json)
        .where(col(Event.assignment_id).is_(None), Event.type == CONFIG_CHANGE)
        .order_by(col(Event.event_id).desc())
    ).all()
    known, old = False, None
    for text in rows:
        try:
            payload = json.loads(text)
        except ValueError:
            continue
        if isinstance(payload, dict) and payload.get("setting") == setting:
            known, old = True, payload.get("new")
            break
    if known and old == value:
        return None
    return record_run_event(
        session, CONFIG_CHANGE, {"setting": setting, "old": old, "new": value}, now=now
    )
