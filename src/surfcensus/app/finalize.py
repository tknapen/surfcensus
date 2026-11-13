"""Final labels and final marks: from one rating, from several, and for subjects that cannot
be rated. See docs/rater_app.md and docs/rating_protocol.md.

Who writes ``final_labels`` and ``final_marks``. With one rater in the database, each
submission rewrites the subject's rows (method ``single``), so review and export read the
same two tables in every mode (:func:`record_single`). As soon as a second rater exists
those rows are removed and nothing is written per submission; the labels then come from
:func:`finalize`. A selected subject whose render failed gets an ``unratable`` label from
the render's reason (:func:`sync_unratable`). Rows of method ``adjudicated`` are never
touched here.

What the functions agree on:

- A *rating* of a subject is the current submission of a rater's latest completed ``main``
  assignment of it, the one made last (``seq``), whatever render it shows. Calibration and
  repeat assignments are no ratings. The queue offers a subject again when its render
  changed, so once a rater has nothing left the latest rating is of the present render.
- An assignment without a submission is no rating, completed or not. A skip stores no
  submission and leaves the assignment open, so a subject that a rater skipped has, from
  that rater, what the rater said of it before (on an earlier render), or nothing. A
  subject that no rater has said anything about has no label from a rating, and gets
  the label of a failed render like any other.
- A subject with a label of method ``adjudicated`` belongs to the adjudicator: none of its
  labels and none of its final marks is written, changed or removed here.
- The label that says why a render failed is for a subject that nobody rated. A subject
  that was rated and whose render fails afterwards keeps what its ratings give.
- A table is rewritten row by row: a row that is already as it should be is left as it is,
  so a second call changes nothing and a final mark keeps its id.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pydantic import ValidationError
from sqlalchemy import func
from sqlmodel import Session, col, select

from surfcensus.app import APP_DB_FILE
from surfcensus.app.db import app_db_mode, open_app_db
from surfcensus.app.models import Assignment, FinalLabel, FinalMark, Mark, Rater, Rating, Submission
from surfcensus.app.queue import QueuePlan
from surfcensus.app.schemas import UNRATABLE_CRITERION
from surfcensus.calibration import (
    CALIBRATION_FILE,
    CalibrationSet,
    CalibrationSubject,
    validate_calibration,
    write_calibration,
)
from surfcensus.config import ConfigError, validation_detail
from surfcensus.log import get_logger
from surfcensus.protocol import (
    CRITERION_IDS,
    OTHER_MARK_CLASS,
    PROTOCOL_FILE,
    Protocol,
    load_protocol,
    render_reason,
)
from surfcensus.store import PublicCatalog

SINGLE = "single"
CONSENSUS = "consensus"
ADJUDICATED = "adjudicated"

CALIBRATION_REASON = "calibration"
"""The ``selection.reason``, and the stratum, of the subjects of a calibration set."""

MAX_STEP = 1
"""Raters who differ by at most this many scale steps on every criterion agree: the label
is the lower value. Further apart, the subject needs adjudication."""

_Labels = Mapping[str, tuple[int | None, str]]
"""Criterion -> (value, method): the final labels a subject should have."""

_MarkRow = tuple[str, float, float, float, str | None, int | None, int | None, str]
"""A final mark without its id and subject: class, tkr x, y, z, hemisphere, pial vertex,
fsaverage6 vertex, source mark ids as JSON."""

_NOTHING: tuple[dict[str, tuple[int | None, str]], tuple[_MarkRow, ...]] = ({}, ())


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


# --- what is in the database --------------------------------------------------------------------


@dataclass(frozen=True)
class _Rated:
    """One rater's rating of a subject."""

    rater_id: int
    unratable: str | None
    values: Mapping[str, int]
    marks: tuple[tuple[int, _MarkRow], ...]
    """``(mark_id, row)`` of the marks of the submission, by ``mark_id``."""


def _current_main(
    session: Session, subject_uid: str | None = None
) -> list[tuple[str, int, int, str | None]]:
    """``(subject_uid, rater_id, submission_id, unratable_reason)`` of the current
    submission of every completed ``main`` assignment, of one subject or of all; a rater's
    in the order the assignments were made."""
    statement = (
        select(
            Assignment.subject_uid,
            Assignment.rater_id,
            Submission.submission_id,
            Submission.unratable_reason,
        )
        .join(Submission, col(Submission.assignment_id) == col(Assignment.assignment_id))
        .where(
            Assignment.kind == "main",
            col(Assignment.completed_at).is_not(None),
            col(Submission.current).is_(True),
        )
        .order_by(col(Assignment.rater_id), col(Assignment.seq))
    )
    if subject_uid is not None:
        statement = statement.where(Assignment.subject_uid == subject_uid)
    rows: list[tuple[str, int, int, str | None]] = []
    for uid, rater_id, submission_id, reason in session.exec(statement).all():
        assert submission_id is not None
        rows.append((uid, rater_id, submission_id, reason))
    return rows


def _rated_subjects(session: Session) -> set[str]:
    """The subjects that have a rating by anyone."""
    return {uid for uid, _, _, _ in _current_main(session)}


def _ratings(session: Session, subject_uid: str | None = None) -> dict[str, list[_Rated]]:
    """Subject -> its ratings, one per rater, by ``rater_id``; of one subject or of all."""
    latest: dict[tuple[str, int], tuple[int, str | None]] = {}
    for uid, rater_id, submission_id, reason in _current_main(session, subject_uid):
        latest[(uid, rater_id)] = (submission_id, reason)  # a later seq replaces an earlier
    kept = {submission_id for submission_id, _ in latest.values()}

    values_of = select(Rating.submission_id, Rating.criterion, Rating.value)
    marks_of = (
        select(
            Mark.mark_id,
            Mark.submission_id,
            Mark.mark_class,
            Mark.tkr_x,
            Mark.tkr_y,
            Mark.tkr_z,
            Mark.hemi,
            Mark.pial_vertex,
            Mark.fsavg6_vertex,
        )
        .where(col(Mark.deleted_ts).is_(None))
        .order_by(col(Mark.mark_id))
    )
    if subject_uid is not None:
        values_of = values_of.join(
            Assignment, col(Assignment.assignment_id) == col(Rating.assignment_id)
        ).where(Assignment.subject_uid == subject_uid)
        marks_of = marks_of.join(
            Assignment, col(Assignment.assignment_id) == col(Mark.assignment_id)
        ).where(Assignment.subject_uid == subject_uid)
    values: dict[int, dict[str, int]] = {}
    for submission_id, criterion, value in session.exec(values_of).all():
        if submission_id in kept:
            values.setdefault(submission_id, {})[criterion] = value
    marks: dict[int, list[tuple[int, _MarkRow]]] = {}
    for mark_id, submission_id, mark_class, x, y, z, hemi, pial, fsavg in session.exec(
        marks_of
    ).all():
        if submission_id in kept:
            assert mark_id is not None
            row = (mark_class, x, y, z, hemi, pial, fsavg, json.dumps([mark_id]))
            marks.setdefault(submission_id, []).append((mark_id, row))

    ratings: dict[str, list[_Rated]] = {}
    for (uid, rater_id), (submission_id, reason) in sorted(latest.items()):
        ratings.setdefault(uid, []).append(
            _Rated(
                rater_id,
                reason,
                values.get(submission_id, {}),
                tuple(marks.get(submission_id, ())),
            )
        )
    return ratings


def _final_rows(
    session: Session, subject_uid: str | None = None
) -> tuple[dict[str, list[FinalLabel]], dict[str, list[FinalMark]]]:
    """The final labels and the final marks that are there, by subject; the marks of a
    subject by id."""
    labels_of = select(FinalLabel)
    marks_of = select(FinalMark).order_by(col(FinalMark.final_mark_id))
    if subject_uid is not None:
        labels_of = labels_of.where(FinalLabel.subject_uid == subject_uid)
        marks_of = marks_of.where(FinalMark.subject_uid == subject_uid)
    labels: dict[str, list[FinalLabel]] = {}
    for label in session.exec(labels_of).all():
        labels.setdefault(label.subject_uid, []).append(label)
    marks: dict[str, list[FinalMark]] = {}
    for mark in session.exec(marks_of).all():
        marks.setdefault(mark.subject_uid, []).append(mark)
    return labels, marks


def _adjudicated(labels: Sequence[FinalLabel]) -> bool:
    return any(label.method == ADJUDICATED for label in labels)


def _from_render(labels: Sequence[FinalLabel]) -> bool:
    """Whether the labels of a subject that nobody rated say why its render failed."""
    return any(
        label.criterion == UNRATABLE_CRITERION and label.method == SINGLE for label in labels
    )


# --- writing ------------------------------------------------------------------------------------


def _mark_row(mark: FinalMark) -> _MarkRow:
    return (
        mark.mark_class,
        mark.tkr_x,
        mark.tkr_y,
        mark.tkr_z,
        mark.hemi,
        mark.pial_vertex,
        mark.fsavg6_vertex,
        mark.source_mark_ids,
    )


def _set_final(
    session: Session,
    subject_uid: str,
    labels: _Labels,
    marks: Sequence[_MarkRow],
    have_labels: Sequence[FinalLabel],
    have_marks: Sequence[FinalMark],
) -> None:
    """Make the final labels and marks of a subject those given, touching only what
    differs. The marks are replaced together or not at all, so they keep their order."""
    for label in have_labels:
        wanted = labels.get(label.criterion)
        if wanted is None:
            session.delete(label)
        elif (label.value, label.method) != wanted:
            label.value, label.method = wanted
            session.add(label)
    there = {label.criterion for label in have_labels}
    for criterion, (value, method) in labels.items():
        if criterion not in there:
            session.add(
                FinalLabel(subject_uid=subject_uid, criterion=criterion, value=value, method=method)
            )
    if [_mark_row(mark) for mark in have_marks] == list(marks):
        return
    for mark in have_marks:
        session.delete(mark)
    for mark_class, x, y, z, hemi, pial, fsavg, sources in marks:
        session.add(
            FinalMark(
                subject_uid=subject_uid,
                mark_class=mark_class,
                tkr_x=x,
                tkr_y=y,
                tkr_z=z,
                hemi=hemi,
                pial_vertex=pial,
                fsavg6_vertex=fsavg,
                source_mark_ids=sources,
            )
        )


def _reason_index(protocol: Protocol, reason: str, subject_uid: str) -> int:
    try:
        return protocol.unratable_index(reason)
    except KeyError:
        raise ConfigError(
            "invalid",
            f"a rating of {subject_uid} says unratable for the reason {reason!r}, which "
            f"{PROTOCOL_FILE} does not list; put the reason back to give the subject a label",
        ) from None


def _of_one(
    rated: _Rated, method: str, protocol: Protocol, subject_uid: str
) -> tuple[dict[str, tuple[int | None, str]], tuple[_MarkRow, ...]]:
    """The labels and marks that one rating gives: its values and its marks as they are."""
    if rated.unratable is not None:
        index = _reason_index(protocol, rated.unratable, subject_uid)
        return {UNRATABLE_CRITERION: (index, method)}, ()
    labels: dict[str, tuple[int | None, str]] = {
        criterion: (rated.values[criterion], method) for criterion in sorted(rated.values)
    }
    return labels, tuple(row for _, row in rated.marks)


def _consensus(
    ratings: Sequence[_Rated], protocol: Protocol, subject_uid: str
) -> tuple[dict[str, tuple[int | None, str]], tuple[_MarkRow, ...]] | None:
    """The labels and marks on which two or more ratings agree, or None when they do not.

    Unratable: every rater must say so, for the same reason. Values: the raters rated the
    same criteria and are within :data:`MAX_STEP` on each; the label is the lowest value.
    Marks: those of every rater, by ``mark_id``, without the marks of a criterion whose
    label is the lowest value of the scale: the label says that there is no such error,
    and a final mark says where one is. Marks of class X belong to no criterion and stay.
    """
    reasons = {rated.unratable for rated in ratings}
    if reasons != {None}:
        if len(reasons) > 1:
            return None
        (reason,) = reasons
        assert reason is not None
        return {UNRATABLE_CRITERION: (_reason_index(protocol, reason, subject_uid), CONSENSUS)}, ()
    criteria = sorted(ratings[0].values)
    if any(sorted(rated.values) != criteria for rated in ratings):
        return None
    labels: dict[str, tuple[int | None, str]] = {}
    for criterion in criteria:
        given = [rated.values[criterion] for rated in ratings]
        if max(given) - min(given) > MAX_STEP:
            return None
        labels[criterion] = (min(given), CONSENSUS)
    none = min(protocol.values())
    no_error = {criterion for criterion, (value, _) in labels.items() if value == none}
    marks = sorted(
        (mark_id, row)
        for rated in ratings
        for mark_id, row in rated.marks
        if row[0] == OTHER_MARK_CLASS or row[0] not in no_error
    )
    return labels, tuple(row for _, row in marks)


# --- the three writers --------------------------------------------------------------------------


def sync_unratable(session: Session, plan: QueuePlan, protocol: Protocol) -> int:
    """Give every subject of ``plan.failed`` the final label ``unratable`` (method
    ``single``, value ``protocol.unratable_index(render_reason(reason))`` with
    ``surfcensus.protocol.render_reason``), and
    remove such a label from a subject whose render is ready again and that has no rating.
    Returns how many subjects carry the label. Runs when the server starts and in
    :func:`finalize`. Not committed.

    The label is for a subject that nobody rated. A subject of ``plan.failed`` that has a
    rating, or a label of method ``adjudicated``, is left as it is and is not counted: what
    it was rated as stays its label (after ``finalize``, when there are several raters). A
    label of a subject whose render is neither failed nor ready (pending, or made with
    another configuration) stays until the render is one or the other.
    """
    rated = _rated_subjects(session)
    labels, marks = _final_rows(session)
    carrying = 0
    for subject_uid in sorted(plan.failed):
        have = labels.get(subject_uid, [])
        if subject_uid in rated or _adjudicated(have):
            continue
        index = protocol.unratable_index(render_reason(plan.failed[subject_uid]))
        wanted = {UNRATABLE_CRITERION: (index, SINGLE)}
        _set_final(session, subject_uid, wanted, (), have, marks.get(subject_uid, []))
        carrying += 1
    for subject_uid, have in labels.items():
        ready = subject_uid in plan.render_keys and subject_uid not in plan.failed
        if ready and subject_uid not in rated and not _adjudicated(have) and _from_render(have):
            _set_final(session, subject_uid, {}, (), have, marks.get(subject_uid, []))
    session.flush()
    return carrying


def _drop_per_submission(session: Session) -> None:
    """Remove the labels of method ``single`` of every rated subject, and its final marks."""
    single = session.exec(select(FinalLabel).where(FinalLabel.method == SINGLE)).all()
    if not single:
        return
    rated = _rated_subjects(session)
    adjudicated = set(
        session.exec(select(FinalLabel.subject_uid).where(FinalLabel.method == ADJUDICATED)).all()
    )
    dropped: set[str] = set()
    for label in single:
        if label.subject_uid in rated and label.subject_uid not in adjudicated:
            session.delete(label)
            dropped.add(label.subject_uid)
    for subject_uid in sorted(dropped):
        for mark in session.exec(
            select(FinalMark).where(FinalMark.subject_uid == subject_uid)
        ).all():
            session.delete(mark)


def _drop_stale_consensus(session: Session, subject_uid: str, protocol: Protocol) -> None:
    """Remove the ``consensus`` labels of a subject, and its final marks, when its ratings
    no longer give them: a rater revised a rating after :func:`finalize`."""
    labels, marks = _final_rows(session, subject_uid)
    have, have_marks = labels.get(subject_uid, []), marks.get(subject_uid, [])
    if _adjudicated(have) or not any(label.method == CONSENSUS for label in have):
        return
    rated = _ratings(session, subject_uid).get(subject_uid, [])
    try:
        agreed = _consensus(rated, protocol, subject_uid) if len(rated) > 1 else None
    except ConfigError as exc:
        get_logger().warning("the final label of %s is left as it is: %s", subject_uid, exc.detail)
        return
    if agreed is not None:
        now = {label.criterion: (label.value, label.method) for label in have}
        if (now, [_mark_row(mark) for mark in have_marks]) == (agreed[0], list(agreed[1])):
            return
    _set_final(session, subject_uid, {}, (), have, have_marks)


def record_single(session: Session, subject_uid: str, protocol: Protocol) -> None:
    """What a submission does to the final tables, called in the transaction that stores it.

    With one rater in the database: replace the subject's ``final_labels`` (method
    ``single``: one row per criterion, or the ``unratable`` row) and ``final_marks`` by the
    current submission of the rater's latest completed ``main`` assignment of the subject.
    With more than one: delete every row of method ``single`` that comes from a rating,
    and the final marks of those subjects, and write nothing. Calibration and repeat
    assignments write nothing. Not committed.

    A row of method ``single`` comes from a rating when its subject has one; the label of
    a subject that nobody rated says why its render failed and stays. With more than one
    rater this also removes the ``single`` rows that :func:`finalize` wrote for subjects
    with one rating. Its ``consensus`` rows stay, but for those of ``subject_uid`` when
    the subject's ratings no longer give them (a rating of it was revised): a label that
    the ratings contradict is not left for review and export to read. Either way the
    labels are those of ``finalize`` no longer after a submission, and it has to run again.

    Never raises for what an earlier rating holds. The rating that gives the label need not
    be the one that was just stored (a repeat of the subject was submitted, say), and it
    can say unratable for a reason that ``protocol.yaml`` has lost since. The subject's
    rows are then left as they are, with a warning: the submission in whose transaction
    this runs is no worse for it, and :func:`finalize` names the rating.
    """
    n_raters = session.exec(select(func.count()).select_from(Rater)).one()
    if n_raters > 1:
        _drop_per_submission(session)
        _drop_stale_consensus(session, subject_uid, protocol)
        session.flush()
        return
    ratings = _ratings(session, subject_uid).get(subject_uid)
    if not ratings:
        return
    labels, marks = _final_rows(session, subject_uid)
    have = labels.get(subject_uid, [])
    if _adjudicated(have):
        return
    try:
        wanted_labels, wanted_marks = _of_one(ratings[0], SINGLE, protocol, subject_uid)
    except ConfigError as exc:
        get_logger().warning("the final label of %s is left as it is: %s", subject_uid, exc.detail)
        return
    _set_final(session, subject_uid, wanted_labels, wanted_marks, have, marks.get(subject_uid, []))
    session.flush()


def finalize(session: Session, plan: QueuePlan, protocol: Protocol) -> FinalizeReport:
    """Rewrite ``final_labels`` and ``final_marks`` from the current submissions of the
    ``main`` assignments (calibration and repeats excluded; per rater and subject the
    latest), and call :func:`sync_unratable`.

    One rating: its values, method ``single``. Two or more: where all raters are within one
    scale step on every criterion, the lower value and the union of the marks, method
    ``consensus``; where a criterion differs by two steps or more, or one rater said
    unratable and another did not, no label, and the subject is in ``open``. Not committed.

    Every rated subject is labelled, in the plan or not. The union leaves out the marks
    of a criterion whose consensus value is the lowest of the scale (one rater saw a minor
    error and marked it, the other saw none: the label is none, and the subject has no
    final mark of that class); the raters' own marks stay in ``marks``. Raters who all say
    unratable agree when they give the same reason; with different reasons, and when they
    rated different criteria, the subject is open. An open subject loses the labels and
    final marks it had. A subject with a label of method ``adjudicated`` is not looked at
    and is in no count. ``unratable`` counts the subjects that carry the label of a failed
    render, which are the subjects of ``plan.failed`` that nobody rated.

    Raises ``ConfigError("invalid", ...)``, before a row is changed, for a rating that says
    unratable for a reason the protocol no longer lists.
    """
    ratings = _ratings(session)
    labels, marks = _final_rows(session)
    n_consensus = n_single = 0
    needs_adjudication: list[str] = []
    wanted: dict[str, tuple[dict[str, tuple[int | None, str]], tuple[_MarkRow, ...]]] = {}
    for subject_uid in sorted(set(ratings) | set(labels) | set(marks)):
        have_labels = labels.get(subject_uid, [])
        if _adjudicated(have_labels):
            continue
        rated = ratings.get(subject_uid, [])
        if not rated:
            # what a failed render gave is sync_unratable's; nothing else explains a row
            kept = {
                label.criterion: (label.value, label.method)
                for label in have_labels
                if label.criterion == UNRATABLE_CRITERION and label.method == SINGLE
            }
            wanted[subject_uid] = (kept, ())
        elif len(rated) == 1:
            wanted[subject_uid] = _of_one(rated[0], SINGLE, protocol, subject_uid)
            n_single += 1
        else:
            agreed = _consensus(rated, protocol, subject_uid)
            if agreed is None:
                needs_adjudication.append(subject_uid)
            else:
                n_consensus += 1
            wanted[subject_uid] = _NOTHING if agreed is None else agreed
    # every label is known before a row is written: a rating that cannot be labelled
    # raises above, with the tables as they were
    for subject_uid, (wanted_labels, wanted_marks) in wanted.items():
        _set_final(
            session,
            subject_uid,
            wanted_labels,
            wanted_marks,
            labels.get(subject_uid, []),
            marks.get(subject_uid, []),
        )
    session.flush()
    n_unratable = sync_unratable(session, plan, protocol)
    return FinalizeReport(n_consensus, n_single, n_unratable, tuple(needs_adjudication))


# --- the calibration set ------------------------------------------------------------------------


def _calibration_subject(
    subject_uid: str, labels: Sequence[FinalLabel], marks: Sequence[FinalMark], protocol: Protocol
) -> CalibrationSubject:
    values = {
        label.criterion: label.value
        for label in labels
        if label.criterion in CRITERION_IDS and label.value is not None
    }
    unratable = None
    for label in labels:
        if label.criterion != UNRATABLE_CRITERION:
            continue
        reasons = protocol.unratable_reasons
        if label.value is None or not 0 <= label.value < len(reasons):
            raise ConfigError(
                "invalid",
                f"the final label of {subject_uid} names unratable reason {label.value!r}, "
                f"and {PROTOCOL_FILE} has the reasons 0 to {len(reasons) - 1}",
            )
        unratable = reasons[label.value]
    try:
        return CalibrationSubject.model_validate(
            {
                "subject_uid": subject_uid,
                "values": values,
                "unratable": unratable,
                "marks": [
                    {"class": mark.mark_class, "tkr": (mark.tkr_x, mark.tkr_y, mark.tkr_z)}
                    for mark in marks
                ],
            }
        )
    except ValidationError as exc:
        raise ConfigError(
            "invalid",
            f"the final labels of {subject_uid} make no calibration subject: "
            f"{validation_detail(exc)}",
        ) from exc


def _choose(
    named: Sequence[str] | None, labels: Mapping[str, Sequence[FinalLabel]], rated: set[str]
) -> list[str]:
    """The subjects of the calibration set, sorted: those named, else every subject with a
    final label that comes from a rating or from adjudication."""

    def unrated(subject_uid: str) -> bool:
        return subject_uid not in rated and not _adjudicated(labels[subject_uid])

    if not named:
        chosen = sorted(subject_uid for subject_uid in labels if not unrated(subject_uid))
        if not chosen:
            raise ConfigError(
                "invalid",
                "no subject has a final label from a rating; rate subjects first, and with "
                "more than one rater run `surfcensus finalize`",
            )
        return chosen
    chosen = sorted(set(named))
    for subject_uid in chosen:
        if subject_uid not in labels:
            raise ConfigError("invalid", f"subject {subject_uid} has no final label")
        if unrated(subject_uid):
            raise ConfigError(
                "invalid",
                f"subject {subject_uid} was not rated: its final label says why it could not "
                "be rendered, and a calibration subject is one that raters are shown",
            )
    return chosen


def freeze_calibration(
    run_dir: Path | str, *, subjects: Sequence[str] | None = None, version: str = "1"
) -> Path:
    """Write ``calibration.yaml`` from ``final_labels`` and ``final_marks``: every subject
    with a final label, or those named (``subject_uid``s), with empty justifications and no
    examples; and add the ``selection`` rows with reason ``calibration``.

    Raises ``ConfigError("invalid", ...)`` when the file exists (it is never overwritten),
    when a named subject has no final label, and when no subject has one.

    A subject that nobody rated, whose label only says why its render failed, is left out,
    and refused when it is named: raters are shown a calibration subject. No name, or an
    empty list of names, means every subject. The subjects are written in the order of
    their ``subject_uid``, the marks of a subject in the order of the final marks. The set
    is checked against ``protocol.yaml`` before anything is written
    (``calibration.validate_calibration``), so labels of other criteria or values than the
    protocol's are refused as well. The ``selection`` rows of reason ``calibration`` that
    were there are replaced (stratum ``calibration``, ``inclusion_prob`` 1, seed 0); they
    are written first and the file last, so the file says that the set is whole.
    ``ConfigError("missing", ...)`` when the run has no ``protocol.yaml``.
    """
    run_dir = Path(run_dir)
    if (run_dir / CALIBRATION_FILE).exists():
        raise ConfigError(
            "invalid",
            f"{CALIBRATION_FILE} exists in {run_dir} and is not overwritten; remove it to "
            "freeze the calibration set again",
        )
    with PublicCatalog.open(run_dir, readonly=False) as public:
        mode = app_db_mode(run_dir)
        if mode is None:
            raise ConfigError(
                "invalid", f"{run_dir} has no {APP_DB_FILE} yet: no subject has a final label"
            )
        protocol, _ = load_protocol(run_dir)
        engine = open_app_db(run_dir, run_id=public.run_id(), mode=mode)
        try:
            with Session(engine) as session:
                labels, marks = _final_rows(session)
                chosen = _choose(subjects, labels, _rated_subjects(session))
                members = tuple(
                    _calibration_subject(
                        subject_uid, labels[subject_uid], marks.get(subject_uid, []), protocol
                    )
                    for subject_uid in chosen
                )
        finally:
            engine.dispose()
        try:
            calibration = CalibrationSet(version=version, subjects=members)
        except ValidationError as exc:
            raise ConfigError("invalid", f"{CALIBRATION_FILE}: {validation_detail(exc)}") from exc
        validate_calibration(
            calibration,
            protocol,
            subjects=public.subject_uids(present=None),
            render=lambda _subject_uid: None,
        )
        rows = pd.DataFrame(
            {
                "subject_uid": chosen,
                "reason": CALIBRATION_REASON,
                "stratum": CALIBRATION_REASON,
                "inclusion_prob": 1.0,
                "seed": 0,
            }
        )
        public.replace_selection(rows, [CALIBRATION_REASON])
    return write_calibration(run_dir, calibration)
