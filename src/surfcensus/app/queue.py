"""Which subject a rater gets next. See docs/rater_app.md, "Assignment logic".

Interface of milestone 4. ``QueuePlan`` and ``Next`` are final; the functions are written
by package 4.D, the local branch in milestone 4 and the team branch in milestone 5.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from sqlmodel import Session

from surfcensus.app.models import Assignment, Rater
from surfcensus.app.schemas import AppMode
from surfcensus.calibration import CalibrationSet
from surfcensus.config import AppConfig
from surfcensus.store import PublicCatalog

_OWNER = "package 4.D (queue)"


@dataclass(frozen=True)
class QueuePlan:
    """What the queue knows of the run at one moment, read from the public catalog."""

    mode: AppMode
    order: tuple[str, ...]
    """The selected subjects in queue order. Local, not blind: the subjects flagged
    ``partial_score`` by ``subject_uid``, then by ``primary_rank`` ascending, then the
    subjects without a score by ``subject_uid``. Blind: a seeded shuffle of the sorted
    uids. Team: sorted uids."""
    render_keys: Mapping[str, str]
    """Subjects whose render is ready -> its ``render_key``."""
    failed: Mapping[str, str]
    """Selected subjects whose render failed -> the reason."""
    pending: int
    """Selected subjects with no render row yet."""
    outdated: int
    """Selected subjects whose render was made with another config hash."""
    calibration: tuple[str, ...] = ()
    raters_per_subject: int = 1
    repeat_rate: float = 0.0
    repeat_gap: int = 200
    seed: int = 0


@dataclass(frozen=True)
class Next:
    status: Literal["assignment", "rendering", "done"]
    assignment: Assignment | None = None
    prefetch: tuple[tuple[str, str], ...] = ()
    """The next two ready ``(subject_uid, render_key)`` after the assignment."""


def build_plan(
    public: PublicCatalog,
    scores: pd.DataFrame | None,
    app: AppConfig,
    mode: AppMode,
    *,
    render_config_hash: str,
    calibration: CalibrationSet | None = None,
) -> QueuePlan:
    """The plan of the present state of the run.

    The selected subjects are the rows of ``selection``. ``scores`` are the primary rows of
    ``scores.parquet`` (``store.primary_scores``), or None when the run has none. A render
    is ready when its row is ``done`` and its config hash is ``render_config_hash``, the
    hash of the running version's render config; a ``done`` row of another hash counts as
    outdated.
    """
    raise NotImplementedError(f"app.queue.build_plan: {_OWNER}")


def next_assignment(session: Session, rater: Rater, plan: QueuePlan) -> Next:
    """The rater's open assignment, or a new one, or why there is none.

    Idempotent: an open assignment is returned again. Local: the first subject of
    ``plan.order`` that is ready and not done for this rater, where done means a completed
    ``main`` assignment whose ``render_key`` is the subject's current one; ``rendering``
    when nothing is ready and renders are pending or outdated; ``done`` when nothing is
    left. A subject whose render failed is never offered. With ``repeat_rate > 0`` an
    assignment is, with that probability, a subject the rater completed at least
    ``repeat_gap`` assignments ago and has not repeated, stored with kind ``repeat``. Every
    draw uses ``random.Random(f"{seed}:{rater_id}:{seq}")``. A new assignment is flushed,
    not committed.
    """
    raise NotImplementedError(f"app.queue.next_assignment: {_OWNER}")


def previous_assignment(session: Session, rater: Rater) -> Assignment | None:
    """The rater's most recent completed assignment, the one ``Backspace`` goes back to."""
    raise NotImplementedError(f"app.queue.previous_assignment: {_OWNER}")


def has_work_left(session: Session, rater: Rater, plan: QueuePlan) -> bool:
    """Whether the rater has an open assignment or :func:`next_assignment` would give one
    or answer ``rendering``. While true, the unblinded routes are closed to the rater. It
    creates no assignment."""
    raise NotImplementedError(f"app.queue.has_work_left: {_OWNER}")
