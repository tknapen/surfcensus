"""Which subject a rater gets next. See docs/rater_app.md, "Assignment logic".

The local branch is written (milestone 4); the team branch (calibration first, fewest
ratings first, several raters per subject) is milestone 5 and raises ``NotImplementedError``.

The queue reads two things: a :class:`QueuePlan`, which is the state of the run as the
public catalog has it at one moment (``build_plan``), and the rater's own assignments in
``app.sqlite``. It writes one thing, a new row of ``assignments``.

An assignment is tied to the render it shows (``render_key``). An open assignment is
*servable* while its subject's render is ready under that key. One whose render changed or
is no longer ready is left as it is, open and never completed, and the subject is offered
again under its present render: a submission on the earlier tiles would be refused by the
server, and returning the same assignment again would leave the page asking forever.

A skip (``skip_assignment``) defers a subject and never drops it. The assignment stays
open, with the time of the skip, and no other assignment is made for the subject. It
*waits*: it is offered again after every subject the rater has not been shown yet, the one
skipped longest ago first, so a rater who skips whatever comes goes round the skipped
subjects in the order of skipping. While a render of the selection is awaited the rater
waits for it, since that subject has not been shown either. The queue is done only when
every subject has a submission. A skip is not a completed rating and gives the subject to
nobody else: in a census a rater cannot avoid a subject by skipping it, which would bias
who rates what. The team branch has to keep the same rule within each of its stages: a
skipped calibration subject comes back after the calibration subjects not yet shown, and
before the first main one.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from sqlalchemy import func, or_
from sqlmodel import Session, col, select

from surfcensus.app.db import record_skip, utc_now
from surfcensus.app.models import Assignment, Rater
from surfcensus.app.schemas import AppMode, SkipIn
from surfcensus.calibration import CalibrationSet
from surfcensus.config import AppConfig
from surfcensus.store import PublicCatalog, queue_order

_TEAM = "the team branch of the queue is milestone 5 (package 5.F)"

PREFETCH = 2
"""How many subjects that come next a page is told of."""


@dataclass(frozen=True)
class QueuePlan:
    """What the queue knows of the run at one moment, read from the public catalog."""

    mode: AppMode
    order: tuple[str, ...]
    """The selected subjects in queue order. Local: ``store.queue_order``, which is also
    the order a local run renders in (not blind: the subjects flagged ``partial_score`` by
    ``subject_uid``, then by ``primary_rank`` ascending, then the subjects without a score
    by ``subject_uid``; blind: a shuffle of the sorted uids with the seed of the run).
    Team: sorted uids."""
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
    """The next two ready ``(subject_uid, render_key)`` after the assignment: the subjects
    not shown yet in queue order, then the skipped ones in the order of skipping."""


class SkipRefused(ValueError):
    """The assignment is not open for the rater, so it cannot be skipped: it has a
    submission, it is another rater's, or its render is no longer the subject's. The
    server answers 409 and the page asks for its next subject again."""


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
    ``scores.parquet`` (``store.primary_scores``), or None when the run has none; the local
    order is ``store.queue_order(selected, scores, blind=app.blind, seed=app.queue_seed)``,
    so the queue and the renderer of a local run go through the subjects alike. A render
    is ready when its row is ``done`` and its config hash is ``render_config_hash``, the
    hash of the running version's render config; a ``done`` row of another hash counts as
    outdated.

    So does a ``failed`` row of another hash: ``surfcensus render`` tries such a subject
    again, so its failure is not yet one of the present configuration. Only a failure
    recorded under ``render_config_hash`` is in ``failed``. Every selected subject is
    therefore in exactly one of ``render_keys``, ``failed``, ``pending`` and ``outdated``.
    """
    if mode not in ("local", "team"):
        raise ValueError(f"mode is local or team, not {mode!r}")
    selected = sorted({str(uid) for uid in public.selection()["subject_uid"]})
    renders = public.renders()
    rows = {
        str(uid): (status, config_hash, render_key, reason)
        for uid, status, config_hash, render_key, reason in zip(
            renders["subject_uid"],
            renders["status"],
            renders["config_hash"],
            renders["render_key"],
            renders["reason"],
            strict=True,
        )
    }
    render_keys: dict[str, str] = {}
    failed: dict[str, str] = {}
    pending = outdated = 0
    for uid in selected:
        row = rows.get(uid)
        if row is None:
            pending += 1
            continue
        status, config_hash, render_key, reason = row
        if config_hash != render_config_hash:
            outdated += 1
        elif status == "done" and isinstance(render_key, str) and render_key:
            render_keys[uid] = render_key
        elif status == "failed":
            failed[uid] = reason if isinstance(reason, str) and reason else "render failure"
        else:
            # a done row without a key names no tiles a rating could be tied to
            outdated += 1
    if mode == "local":
        order = queue_order(selected, scores, blind=app.blind, seed=app.queue_seed)
    else:
        order = selected
    in_calibration = () if calibration is None else calibration.subjects
    return QueuePlan(
        mode=mode,
        order=tuple(order),
        render_keys=render_keys,
        failed=failed,
        pending=pending,
        outdated=outdated,
        calibration=tuple(subject.subject_uid for subject in in_calibration),
        raters_per_subject=app.raters_per_subject,
        repeat_rate=app.repeat_rate,
        repeat_gap=app.repeat_gap,
        seed=app.queue_seed,
    )


# --- a rater's assignments ----------------------------------------------------------------------


@dataclass(frozen=True)
class _Open:
    """An assignment without a submission."""

    assignment_id: int
    subject_uid: str
    render_key: str
    kind: str
    seq: int
    skipped_at: str | None


@dataclass(frozen=True)
class _History:
    """A rater's assignments, as far as the queue needs them."""

    last_seq: int
    main: Mapping[tuple[str, str], int]
    """``(subject_uid, render_key)`` of the completed ``main`` assignments -> the latest
    ``seq`` among them. Keyed by both: subjects with the same images have one key."""
    repeated: frozenset[tuple[str, str]]
    """``(subject_uid, render_key)`` of the completed ``repeat`` assignments."""
    open: tuple[_Open, ...]
    """The assignments that are not completed, the latest first."""


def _rater_id(rater: Rater) -> int:
    if rater.rater_id is None:
        raise ValueError("the rater has no id yet: flush it first")
    return rater.rater_id


def _history(session: Session, rater_id: int) -> _History:
    rows = session.exec(
        select(
            Assignment.assignment_id,
            Assignment.subject_uid,
            Assignment.kind,
            Assignment.seq,
            Assignment.render_key,
            Assignment.completed_at,
            Assignment.skipped_at,
        )
        .where(Assignment.rater_id == rater_id)
        .order_by(col(Assignment.seq))
    ).all()
    main: dict[tuple[str, str], int] = {}
    repeated: set[tuple[str, str]] = set()
    still_open: list[_Open] = []
    last_seq = 0
    for assignment_id, subject_uid, kind, seq, render_key, completed_at, skipped_at in rows:
        last_seq = seq
        shown = (subject_uid, render_key)
        if completed_at is None:
            assert assignment_id is not None
            still_open.append(_Open(assignment_id, subject_uid, render_key, kind, seq, skipped_at))
        elif kind == "main":
            main[shown] = seq
        elif kind == "repeat":
            repeated.add(shown)
    return _History(last_seq, main, frozenset(repeated), tuple(reversed(still_open)))


def _current(plan: QueuePlan, subject_uid: str) -> tuple[str, str] | None:
    """The subject with the key of its ready render, or None for a subject that cannot be
    shown now."""
    render_key = plan.render_keys.get(subject_uid)
    if render_key is None or subject_uid in plan.failed:
        return None
    return subject_uid, render_key


def _to_rate(history: _History, plan: QueuePlan) -> list[str]:
    """The subjects that are ready and not done for the rater, in queue order."""
    return [
        uid
        for uid in plan.order
        if (shown := _current(plan, uid)) is not None and shown not in history.main
    ]


def _is_servable(plan: QueuePlan, subject_uid: str, render_key: str) -> bool:
    return _current(plan, subject_uid) == (subject_uid, render_key)


def _servable(history: _History, plan: QueuePlan) -> _Open | None:
    """The latest open assignment that was not skipped and whose render is the subject's
    ready one: the one the page shows."""
    for assignment in history.open:
        if assignment.skipped_at is None and _is_servable(
            plan, assignment.subject_uid, assignment.render_key
        ):
            return assignment
    return None


def _waiting(history: _History, plan: QueuePlan) -> list[_Open]:
    """The skipped assignments whose render is the subject's ready one, the one skipped
    longest ago first; of two skipped at one time, the one made first."""
    skipped = [
        assignment
        for assignment in history.open
        if assignment.skipped_at is not None
        and _is_servable(plan, assignment.subject_uid, assignment.render_key)
    ]
    return sorted(skipped, key=lambda assignment: (assignment.skipped_at, assignment.seq))


def _repeat(history: _History, plan: QueuePlan, rng: random.Random, seq: int) -> str | None:
    """The subject to show again as assignment ``seq``, or None when the draw says no or no
    subject can be repeated: one whose completed ``main`` assignment of the present render
    lies at least ``repeat_gap`` assignments back and that was not repeated on that render.
    A repeat that was skipped is open and waits; its subject gets no second one meanwhile.
    """
    if plan.repeat_rate <= 0 or rng.random() >= plan.repeat_rate:
        return None
    waiting = {
        (assignment.subject_uid, assignment.render_key)
        for assignment in history.open
        if assignment.kind == "repeat"
    }
    candidates = sorted(
        uid
        for (uid, render_key), rated_seq in history.main.items()
        if _current(plan, uid) == (uid, render_key)
        and seq - rated_seq >= plan.repeat_gap
        and (uid, render_key) not in history.repeated
        and (uid, render_key) not in waiting
    )
    return rng.choice(candidates) if candidates else None


@dataclass(frozen=True)
class _Decision:
    """What the rater gets when no open assignment is on the page: a new assignment
    (``subject_uid``), a skipped one that waited (``back``), or neither."""

    status: Literal["assignment", "rendering", "done"]
    kind: Literal["main", "repeat"] = "main"
    subject_uid: str | None = None
    seq: int = 0
    back: _Open | None = None


def _decide(
    history: _History, rater_id: int, plan: QueuePlan, fresh: list[str], waiting: list[_Open]
) -> _Decision:
    """``fresh`` are the subjects to rate that the rater has not been shown, in queue
    order; ``waiting`` the skipped assignments, in the order they come back."""
    seq = history.last_seq + 1
    # one generator per assignment, so the queue can be traced from the database
    rng = random.Random(f"{plan.seed}:{rater_id}:{seq}")
    repeat = _repeat(history, plan, rng, seq)
    if repeat is not None:
        return _Decision("assignment", "repeat", repeat, seq)
    if fresh:
        return _Decision("assignment", "main", fresh[0], seq)
    if plan.pending + plan.outdated > 0:
        # a subject that is still being rendered has not been shown either
        return _Decision("rendering")
    if waiting:
        return _Decision("assignment", back=waiting[0])
    return _Decision("done")


def _fresh(to_rate: list[str], waiting: list[_Open]) -> list[str]:
    skipped = {assignment.subject_uid for assignment in waiting}
    return [uid for uid in to_rate if uid not in skipped]


def _prefetch(
    plan: QueuePlan, to_rate: list[str], waiting: list[_Open], shown: str
) -> tuple[tuple[str, str], ...]:
    upcoming = _fresh(to_rate, waiting) + [assignment.subject_uid for assignment in waiting]
    ahead = [uid for uid in upcoming if uid != shown][:PREFETCH]
    return tuple((uid, plan.render_keys[uid]) for uid in ahead)


def _local_only(plan: QueuePlan) -> None:
    if plan.mode != "local":
        raise NotImplementedError(f"app.queue: {_TEAM}")


# --- the queue ----------------------------------------------------------------------------------


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

    An open assignment is returned only while it is servable (see the module): the
    subject's render is ready and has the assignment's ``render_key``. Of several, the one
    made last, which is the one the page shows. A repeat is of the present render: the
    rater completed these tiles as ``main`` and has not repeated them. The draw for a
    repeat comes before the walk, so a repeat can be the last assignment before ``done``
    and can be given while renders are awaited.

    A skipped assignment (see the module) is open and is not the one on the page. It is
    returned when the rater has been shown everything else: no open assignment that was
    not skipped, no repeat drawn, no subject left that the rater has not been shown, no
    render awaited. Of several, the one skipped longest ago. No second assignment is made
    for its subject while its render stays the same.
    """
    _local_only(plan)
    rater_id = _rater_id(rater)
    history = _history(session, rater_id)
    to_rate = _to_rate(history, plan)
    waiting = _waiting(history, plan)

    def existing(shown: _Open) -> Next:
        assignment = session.get(Assignment, shown.assignment_id)
        assert assignment is not None
        return Next("assignment", assignment, _prefetch(plan, to_rate, waiting, shown.subject_uid))

    servable = _servable(history, plan)
    if servable is not None:
        return existing(servable)
    decision = _decide(history, rater_id, plan, _fresh(to_rate, waiting), waiting)
    if decision.back is not None:
        return existing(decision.back)
    if decision.subject_uid is None:
        return Next(decision.status)
    assignment = Assignment(
        rater_id=rater_id,
        subject_uid=decision.subject_uid,
        kind=decision.kind,
        seq=decision.seq,
        render_key=plan.render_keys[decision.subject_uid],
        assigned_at=utc_now(),
    )
    session.add(assignment)
    session.flush()
    prefetch = _prefetch(plan, to_rate, waiting, decision.subject_uid)
    return Next("assignment", assignment, prefetch)


def is_open_for(rater: Rater, assignment: Assignment, plan: QueuePlan) -> bool:
    """Whether the assignment is one the rater can still rate for the first time, or skip:
    it is the rater's, it has no submission, and its render is the subject's ready one."""
    return (
        assignment.rater_id == _rater_id(rater)
        and assignment.completed_at is None
        and _is_servable(plan, assignment.subject_uid, assignment.render_key)
    )


def skip_assignment(
    session: Session,
    rater: Rater,
    assignment: Assignment,
    skip: SkipIn,
    plan: QueuePlan,
    *,
    now: str | None = None,
) -> Next:
    """The rater skips the assignment: what ``POST /api/skip`` does. Returns what
    :func:`next_assignment` gives afterwards.

    The skip is recorded on the assignment and as an event (``app.db.record_skip``); no
    rating, mark or note is stored, and the assignment stays open and waits (see the
    module). A request that was sent twice, with the same ``client_ts``, is recorded once
    and answered alike. When the skipped assignment is the only thing left for the rater,
    it is returned again. Flushed, not committed.

    Raises :class:`SkipRefused`, with nothing stored, for an assignment that is not open
    for the rater (:func:`is_open_for`), and ``ValueError`` for a skip that names another
    assignment.
    """
    _local_only(plan)
    if skip.assignment_id != assignment.assignment_id:
        raise ValueError(
            f"the skip is of assignment {skip.assignment_id}, not of {assignment.assignment_id}"
        )
    if not is_open_for(rater, assignment, plan):
        raise SkipRefused(f"assignment {assignment.assignment_id} is not open for the rater")
    record_skip(session, assignment, skip, now=now)
    return next_assignment(session, rater, plan)


def previous_assignment(session: Session, rater: Rater) -> Assignment | None:
    """The assignment the rater left last, rated or skipped: the one ``Backspace`` goes
    back to.

    An assignment is left when its first submission is stored (``completed_at``, which a
    revision leaves) and when it is skipped (``skipped_at``); the later of the two counts.
    Of two left at one time, the one made later. A skipped assignment comes back as it is:
    open, without a submission, and its skips stay recorded. Whether its render is still
    the subject's is for the caller to check: this reads no plan.
    """
    left = func.max(
        func.coalesce(col(Assignment.completed_at), ""),
        func.coalesce(col(Assignment.skipped_at), ""),
    )
    return session.exec(
        select(Assignment)
        .where(
            Assignment.rater_id == _rater_id(rater),
            or_(col(Assignment.completed_at).is_not(None), col(Assignment.skipped_at).is_not(None)),
        )
        .order_by(left.desc(), col(Assignment.seq).desc())
        .limit(1)
    ).first()


def has_work_left(session: Session, rater: Rater, plan: QueuePlan) -> bool:
    """Whether the rater has an open assignment or :func:`next_assignment` would give one
    or answer ``rendering``. While true, the unblinded routes are closed to the rater. It
    creates no assignment.

    An open assignment counts while it is servable, skipped or not; one whose render
    changed or failed is never returned again and closes nothing.
    """
    _local_only(plan)
    rater_id = _rater_id(rater)
    history = _history(session, rater_id)
    if _servable(history, plan) is not None:
        return True
    waiting = _waiting(history, plan)
    fresh = _fresh(_to_rate(history, plan), waiting)
    return _decide(history, rater_id, plan, fresh, waiting).status != "done"


def subjects_left(session: Session, rater: Rater, plan: QueuePlan) -> int:
    """How many subjects of the plan the rater still has to rate, as far as the plan says:
    those that are ready and not done, and those whose render is pending or outdated. A
    subject whose render failed is never offered and is not counted; neither is a repeat.
    A skipped subject is still to rate and is counted. The ``remaining`` of the progress
    line. It creates no assignment."""
    _local_only(plan)
    history = _history(session, _rater_id(rater))
    return len(_to_rate(history, plan)) + plan.pending + plan.outdated


def subjects_skipped(
    session: Session, rater: Rater, plan: QueuePlan, shown: Assignment | None = None
) -> int:
    """How many subjects the rater skipped that wait to be offered again: the skipped
    assignments whose render is the subject's ready one, without ``shown``, the assignment
    that the answer puts on the page (it does not wait any more). The ``skipped`` of the
    progress line. It creates no assignment."""
    _local_only(plan)
    waiting = _waiting(_history(session, _rater_id(rater)), plan)
    on_page = None if shown is None else shown.assignment_id
    return sum(assignment.assignment_id != on_page for assignment in waiting)
