"""Interaction events: what a rater did on a subject, and events of the run. See
docs/rater_app.md, "Event logging".

The types and the limits of a batch are in ``app.schemas`` (``EVENT_TYPES``,
``MAX_EVENTS_PER_CALL``, ``MAX_EVENT_PAYLOAD_BYTES``).

Interface of milestone 4; the functions are written by package 4.G.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sqlmodel import Session

from surfcensus.app.models import Assignment, Event
from surfcensus.app.schemas import EventIn

_OWNER = "package 4.G (server and events)"


def store_events(session: Session, assignment: Assignment, events: Sequence[EventIn]) -> int:
    """Store a batch of events of one assignment as they were sent, the payload as compact
    JSON with sorted keys; returns how many. An unknown type is stored as given. Not
    committed."""
    raise NotImplementedError(f"app.events.store_events: {_OWNER}")


def record_run_event(
    session: Session, type: str, payload: Mapping[str, object], *, now: str | None = None
) -> Event:
    """Store an event of the run, without an assignment: a changed setting
    (``config_change``, with the setting, its old and its new value). Not committed."""
    raise NotImplementedError(f"app.events.record_run_event: {_OWNER}")
