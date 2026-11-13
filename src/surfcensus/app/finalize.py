"""Final labels and final marks: from one rating, from several, and for subjects that cannot
be rated. See docs/rater_app.md and docs/rating_protocol.md.

Who writes ``final_labels`` and ``final_marks``. With one rater in the database, each
submission rewrites the subject's rows (method ``single``), so review and export read the
same two tables in every mode (:func:`record_single`). As soon as a second rater exists
those rows are removed and nothing is written per submission; the labels then come from
:func:`finalize`. A selected subject whose render failed gets an ``unratable`` label from
the render's reason (:func:`sync_unratable`). Rows of method ``adjudicated`` are never
touched here.

Interface of milestone 4. ``FinalizeReport`` is final; the functions are written by
package 4.F.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from sqlmodel import Session

from surfcensus.app.queue import QueuePlan
from surfcensus.protocol import Protocol

_OWNER = "package 4.F (finalize)"


@dataclass(frozen=True)
class FinalizeReport:
    consensus: int
    """Subjects given a consensus label."""
    single: int
    """Subjects with one rating."""
    unratable: int
    """Selected subjects whose render failed."""
    open: tuple[str, ...]
    """``subject_uid``s that need adjudication, sorted."""


def sync_unratable(session: Session, plan: QueuePlan, protocol: Protocol) -> int:
    """Give every subject of ``plan.failed`` the final label ``unratable`` (method
    ``single``, value ``protocol.unratable_index(render_reason(reason))`` with
    ``surfcensus.protocol.render_reason``), and
    remove such a label from a subject whose render is ready again and that has no rating.
    Returns how many subjects carry the label. Runs when the server starts and in
    :func:`finalize`. Not committed."""
    raise NotImplementedError(f"app.finalize.sync_unratable: {_OWNER}")


def record_single(session: Session, subject_uid: str, protocol: Protocol) -> None:
    """What a submission does to the final tables, called in the transaction that stores it.

    With one rater in the database: replace the subject's ``final_labels`` (method
    ``single``: one row per criterion, or the ``unratable`` row) and ``final_marks`` by the
    current submission of the rater's latest completed ``main`` assignment of the subject.
    With more than one: delete every row of method ``single`` that comes from a rating,
    and the final marks of those subjects, and write nothing. Calibration and repeat
    assignments write nothing. Not committed."""
    raise NotImplementedError(f"app.finalize.record_single: {_OWNER}")


def finalize(session: Session, plan: QueuePlan, protocol: Protocol) -> FinalizeReport:
    """Rewrite ``final_labels`` and ``final_marks`` from the current submissions of the
    ``main`` assignments (calibration and repeats excluded; per rater and subject the
    latest), and call :func:`sync_unratable`.

    One rating: its values, method ``single``. Two or more: where all raters are within one
    scale step on every criterion, the lower value and the union of the marks, method
    ``consensus``; where a criterion differs by two steps or more, or one rater said
    unratable and another did not, no label, and the subject is in ``open``. Not committed.
    """
    raise NotImplementedError(f"app.finalize.finalize: {_OWNER}")


def freeze_calibration(
    run_dir: Path | str, *, subjects: Sequence[str] | None = None, version: str = "1"
) -> Path:
    """Write ``calibration.yaml`` from ``final_labels`` and ``final_marks``: every subject
    with a final label, or those named (``subject_uid``s), with empty justifications and no
    examples; and add the ``selection`` rows with reason ``calibration``.

    Raises ``ConfigError("invalid", ...)`` when the file exists (it is never overwritten),
    when a named subject has no final label, and when no subject has one.
    """
    raise NotImplementedError(f"app.finalize.freeze_calibration: {_OWNER}")
