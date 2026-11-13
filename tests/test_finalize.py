"""Tests of surfcensus/app/finalize.py: final labels and final marks from one rating, from
several and for a failed render, and the calibration set frozen from them.

The tests of the three writers give them a plan made by hand and a new ``app.sqlite``;
ratings are stored as the server stores them (``app.db.store_submission``). The tests of
``freeze_calibration`` work on a copy of ``small_run``, since it reads a run directory.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlmodel")

from sqlalchemy.engine import Engine
from sqlmodel import Session, col, select

from surfcensus.app import APP_DB_FILE
from surfcensus.app.db import app_db_path, get_or_create_rater, open_app_db, store_submission
from surfcensus.app.finalize import (
    CALIBRATION_REASON,
    FinalizeReport,
    finalize,
    freeze_calibration,
    record_single,
    sync_unratable,
)
from surfcensus.app.models import Assignment, FinalLabel, FinalMark, Mark, Rater, Submission
from surfcensus.app.queue import QueuePlan, build_plan
from surfcensus.app.schemas import MarkIn, RatingIn, ResolvedMark
from surfcensus.calibration import (
    CALIBRATION_FILE,
    CalibrationSet,
    dump_calibration,
    load_calibration,
    validate_calibration,
)
from surfcensus.config import ConfigError, load_config
from surfcensus.protocol import PROTOCOL_FILE, Protocol
from surfcensus.store import PublicCatalog, primary_scores

RUN_ID = "0f1e2d3c4b5a6978"
S1, S2, S3, S4, S5 = (f"{n:016x}" for n in range(1, 6))
NONE, MINOR, MAJOR = 0, 1, 2
T = "2026-11-13T10:00:00.000Z"


# --- a database, raters who rate, and what the final tables hold --------------------------------


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    engine = open_app_db(tmp_path, run_id=RUN_ID, mode="local")
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    with Session(engine) as session:
        yield session


@pytest.fixture
def ada(session: Session) -> Rater:
    return get_or_create_rater(session, "ada", role="lead")


@pytest.fixture
def ben(session: Session, ada: Rater) -> Rater:
    """A second rater; ``ada`` is made first."""
    return get_or_create_rater(session, "ben", role="lead")


def plan_of(
    ready: Sequence[str] = (),
    failed: Mapping[str, str] | None = None,
    *,
    pending: Sequence[str] = (),
) -> QueuePlan:
    """A local plan: the subjects that are ready, those whose render failed with their
    reasons, and those in the order that are neither (counted as pending)."""
    failed = {} if failed is None else dict(failed)
    order = tuple(sorted({*ready, *failed, *pending}))
    keys = {uid: f"key-{uid}" for uid in ready}
    return QueuePlan("local", order, keys, failed, len(pending), 0)


def values(a: int, b: int = NONE, c: int = NONE) -> dict[str, int]:
    return {"A": a, "B": b, "C": c}


def mark(mark_class: str, x: float, *, vertex: int | None = None) -> dict[str, Any]:
    """A mark of a class at tkr ``(x, 2x, -x)``, on the left pial surface when ``vertex``
    is given and off the surfaces otherwise."""
    return {"class": mark_class, "x": x, "vertex": vertex}


def submit(
    session: Session,
    assignment: Assignment,
    given: Mapping[str, int] | None = None,
    *,
    unratable: str | None = None,
    marks: Sequence[Mapping[str, Any]] = (),
) -> list[Mark]:
    """One more submission of the assignment, stored as the server stores it; its marks."""
    assert assignment.assignment_id is not None
    posted = tuple(
        MarkIn.model_validate(
            {
                "client_id": f"m{n}",
                "class": spec["class"],
                "panel": "sagittal",
                "slice": 3,
                "px": 30.5,
                "py": 28.25,
            }
        )
        for n, spec in enumerate(marks)
    )
    resolved = [
        ResolvedMark(
            hemi=None if spec["vertex"] is None else "lh",
            pial_vertex=spec["vertex"],
            pial_dist_mm=None if spec["vertex"] is None else 0.5,
            fsavg6_vertex=None if spec["vertex"] is None else spec["vertex"] + 1000,
            vertex_surface=None if spec["vertex"] is None else "pial",
            box_dist_mm=0.0,
            vox=(1.0, 2.0, 3.0),
            tkr=(spec["x"], 2 * spec["x"], -spec["x"]),
            ras=(0.0, 0.0, 0.0),
        )
        for spec in marks
    ]
    rating = RatingIn(
        assignment_id=assignment.assignment_id,
        protocol_version="0.1-draft",
        values={} if unratable is not None else dict(values(NONE) if given is None else given),
        unratable=unratable,
        marks=posted,
        ms_on_subject=1,
    )
    _, stored = store_submission(
        session, assignment, rating, resolved, protocol_sha256="9c", app_version="0"
    )
    return stored


def rate(
    session: Session,
    rater: Rater,
    subject_uid: str,
    given: Mapping[str, int] | None = None,
    *,
    unratable: str | None = None,
    marks: Sequence[Mapping[str, Any]] = (),
    kind: str = "main",
    key: str = "key",
) -> Assignment:
    """A new assignment of the subject to the rater, completed with one submission."""
    assert rater.rater_id is not None
    earlier = session.exec(
        select(Assignment.seq)
        .where(Assignment.rater_id == rater.rater_id)
        .order_by(col(Assignment.seq).desc())
    ).first()
    assignment = Assignment(
        rater_id=rater.rater_id,
        subject_uid=subject_uid,
        kind=kind,
        seq=(earlier or 0) + 1,
        render_key=key,
        assigned_at=T,
    )
    session.add(assignment)
    session.flush()
    submit(session, assignment, given, unratable=unratable, marks=marks)
    return assignment


def labels(session: Session) -> dict[str, dict[str, tuple[int | None, str]]]:
    """Subject -> criterion -> (value, method), as the final labels are now."""
    session.flush()
    found: dict[str, dict[str, tuple[int | None, str]]] = {}
    for label in session.exec(select(FinalLabel)).all():
        found.setdefault(label.subject_uid, {})[label.criterion] = (label.value, label.method)
    return found


def final_marks(session: Session) -> dict[str, list[tuple[str, float, list[int]]]]:
    """Subject -> (class, tkr x, source mark ids) of its final marks, by id."""
    session.flush()
    found: dict[str, list[tuple[str, float, list[int]]]] = {}
    for final in session.exec(select(FinalMark).order_by(col(FinalMark.final_mark_id))).all():
        found.setdefault(final.subject_uid, []).append(
            (final.mark_class, final.tkr_x, json.loads(final.source_mark_ids))
        )
    return found


def final_mark_ids(session: Session) -> list[int | None]:
    session.flush()
    return list(
        session.exec(select(FinalMark.final_mark_id).order_by(col(FinalMark.final_mark_id))).all()
    )


def live_mark_ids(session: Session, assignment: Assignment) -> list[int]:
    found = session.exec(
        select(Mark.mark_id)
        .where(Mark.assignment_id == assignment.assignment_id, col(Mark.deleted_ts).is_(None))
        .order_by(col(Mark.mark_id))
    ).all()
    return [mark_id for mark_id in found if mark_id is not None]


def single(a: int, b: int = NONE, c: int = NONE) -> dict[str, tuple[int | None, str]]:
    return {name: (value, "single") for name, value in values(a, b, c).items()}


def consensus(a: int, b: int = NONE, c: int = NONE) -> dict[str, tuple[int | None, str]]:
    return {name: (value, "consensus") for name, value in values(a, b, c).items()}


def adjudicate(session: Session, subject_uid: str, a: int, x: float) -> None:
    """The rows an adjudicator leaves: labels of method ``adjudicated`` and a final mark."""
    for criterion, value in values(a).items():
        session.add(
            FinalLabel(
                subject_uid=subject_uid, criterion=criterion, value=value, method="adjudicated"
            )
        )
    session.add(
        FinalMark(
            subject_uid=subject_uid,
            mark_class="A",
            tkr_x=x,
            tkr_y=0.0,
            tkr_z=0.0,
            source_mark_ids="[]",
        )
    )
    session.flush()


# --- record_single: one rater -------------------------------------------------------------------


def test_one_rating_gives_single_labels_and_its_marks(
    session: Session, ada: Rater, shipped_protocol: Protocol
) -> None:
    given = values(MAJOR, NONE, MINOR)
    assignment = rate(
        session, ada, S1, given, marks=[mark("A", 1.5, vertex=211), mark("C", 2.5), mark("X", 3.5)]
    )
    record_single(session, S1, shipped_protocol)

    assert labels(session) == {S1: single(MAJOR, NONE, MINOR)}
    ids = live_mark_ids(session, assignment)
    assert final_marks(session) == {
        S1: [("A", 1.5, [ids[0]]), ("C", 2.5, [ids[1]]), ("X", 3.5, [ids[2]])]
    }
    on_surface, off_surface, _ = session.exec(
        select(FinalMark).order_by(col(FinalMark.final_mark_id))
    ).all()
    assert (on_surface.tkr_x, on_surface.tkr_y, on_surface.tkr_z) == (1.5, 3.0, -1.5)
    assert (on_surface.hemi, on_surface.pial_vertex, on_surface.fsavg6_vertex) == ("lh", 211, 1211)
    # a mark off the surfaces is a final mark all the same, without a vertex
    assert (off_surface.hemi, off_surface.pial_vertex, off_surface.fsavg6_vertex) == (
        None,
        None,
        None,
    )


def test_a_revision_replaces_the_labels_and_the_marks(
    session: Session, ada: Rater, shipped_protocol: Protocol
) -> None:
    assignment = rate(session, ada, S1, values(MAJOR), marks=[mark("A", 1.5), mark("A", 2.5)])
    record_single(session, S1, shipped_protocol)
    revised = submit(session, assignment, values(MINOR, MINOR), marks=[mark("B", 7.5)])
    record_single(session, S1, shipped_protocol)

    assert labels(session) == {S1: single(MINOR, MINOR)}
    assert final_marks(session) == {S1: [("B", 7.5, [revised[0].mark_id])]}


def test_an_unratable_rating_gives_the_one_row_with_the_position_of_its_reason(
    session: Session, ada: Rater, shipped_protocol: Protocol
) -> None:
    assignment = rate(session, ada, S1, values(MAJOR), marks=[mark("A", 1.5)])
    record_single(session, S1, shipped_protocol)
    submit(session, assignment, unratable="image too poor to judge")
    record_single(session, S1, shipped_protocol)

    assert shipped_protocol.unratable_reasons[3] == "image too poor to judge"
    assert labels(session) == {S1: {"unratable": (3, "single")}}
    assert final_marks(session) == {}

    # and back: the unratable row goes when the rater rates the subject after all
    submit(session, assignment, values(NONE))
    record_single(session, S1, shipped_protocol)
    assert labels(session) == {S1: single(NONE)}


def test_the_position_follows_the_protocol_that_is_given(
    session: Session, ada: Rater, shipped_protocol: Protocol
) -> None:
    reasons = tuple(reversed(shipped_protocol.unratable_reasons))
    protocol = shipped_protocol.model_copy(update={"unratable_reasons": reasons})
    rate(session, ada, S1, unratable="image too poor to judge")
    record_single(session, S1, protocol)
    assert labels(session) == {S1: {"unratable": (0, "single")}}


def test_only_the_subject_of_the_submission_is_rewritten(
    session: Session, ada: Rater, shipped_protocol: Protocol
) -> None:
    rate(session, ada, S1, values(MAJOR), marks=[mark("A", 1.5)])
    record_single(session, S1, shipped_protocol)
    before = final_mark_ids(session)
    rate(session, ada, S2, values(MINOR), marks=[mark("A", 2.5)])
    record_single(session, S2, shipped_protocol)

    assert labels(session) == {S1: single(MAJOR), S2: single(MINOR)}
    assert final_mark_ids(session)[: len(before)] == before
    assert [uid for uid in final_marks(session)] == [S1, S2]


def test_a_second_call_changes_nothing_and_the_final_marks_keep_their_ids(
    session: Session, ada: Rater, shipped_protocol: Protocol
) -> None:
    rate(session, ada, S1, values(MAJOR), marks=[mark("A", 1.5), mark("X", 2.5)])
    record_single(session, S1, shipped_protocol)
    first = (labels(session), final_marks(session), final_mark_ids(session))
    record_single(session, S1, shipped_protocol)
    assert (labels(session), final_marks(session), final_mark_ids(session)) == first
    assert not session.dirty and not session.new and not session.deleted


@pytest.mark.parametrize("kind", ["repeat", "calibration"])
def test_a_repeat_and_a_calibration_assignment_write_nothing(
    session: Session, ada: Rater, shipped_protocol: Protocol, kind: str
) -> None:
    rate(session, ada, S1, values(MAJOR), marks=[mark("A", 1.5)])
    record_single(session, S1, shipped_protocol)
    first = (labels(session), final_marks(session), final_mark_ids(session))

    rate(session, ada, S1, values(NONE), marks=[mark("B", 9.5)], kind=kind)
    record_single(session, S1, shipped_protocol)
    assert (labels(session), final_marks(session), final_mark_ids(session)) == first

    # and for a subject that has nothing but such an assignment there is no label at all
    rate(session, ada, S2, values(MINOR), kind=kind)
    record_single(session, S2, shipped_protocol)
    assert S2 not in labels(session)


def test_the_latest_main_assignment_of_a_subject_is_its_rating(
    session: Session, ada: Rater, shipped_protocol: Protocol
) -> None:
    first = rate(session, ada, S1, values(MAJOR), marks=[mark("A", 1.5)], key="tiles-1")
    rate(session, ada, S2, values(NONE))
    rate(session, ada, S1, values(MINOR), marks=[mark("A", 4.5)], key="tiles-2")
    record_single(session, S1, shipped_protocol)
    assert labels(session)[S1] == single(MINOR)
    assert [x for _, x, _ in final_marks(session)[S1]] == [4.5]

    # a revision of the earlier assignment does not bring it back
    submit(session, first, values(MAJOR, MAJOR))
    record_single(session, S1, shipped_protocol)
    assert labels(session)[S1] == single(MINOR)


def test_record_single_is_not_committed(
    engine: Engine, session: Session, ada: Rater, shipped_protocol: Protocol
) -> None:
    rate(session, ada, S1, values(MAJOR), marks=[mark("A", 1.5)])
    session.commit()
    record_single(session, S1, shipped_protocol)
    assert labels(session) == {S1: single(MAJOR)}
    session.rollback()
    session.close()
    with Session(engine) as later:
        assert labels(later) == {} and final_marks(later) == {}


def test_a_rating_replaces_the_label_of_a_failed_render(
    session: Session, ada: Rater, shipped_protocol: Protocol
) -> None:
    assert sync_unratable(session, plan_of(failed={S1: "missing surface"}), shipped_protocol) == 1
    rate(session, ada, S1, values(MINOR))
    record_single(session, S1, shipped_protocol)
    assert labels(session) == {S1: single(MINOR)}


def test_record_single_leaves_an_adjudicated_subject_alone(
    session: Session, ada: Rater, shipped_protocol: Protocol
) -> None:
    adjudicate(session, S1, MAJOR, 8.5)
    rate(session, ada, S1, values(NONE), marks=[mark("B", 1.5)])
    record_single(session, S1, shipped_protocol)
    assert labels(session) == {
        S1: {name: (value, "adjudicated") for name, value in values(MAJOR).items()}
    }
    assert final_marks(session) == {S1: [("A", 8.5, [])]}


# --- record_single: the second rater ------------------------------------------------------------


def test_the_second_raters_arrival_removes_the_per_submission_labels(
    session: Session, ada: Rater, shipped_protocol: Protocol
) -> None:
    rate(session, ada, S1, values(MAJOR), marks=[mark("A", 1.5)])
    record_single(session, S1, shipped_protocol)
    rate(session, ada, S2, unratable="image too poor to judge")
    record_single(session, S2, shipped_protocol)
    # S3 was never rated: its label says why its render failed
    sync_unratable(session, plan_of([S1, S2], {S3: "wrong orientation"}), shipped_protocol)
    adjudicate(session, S4, MINOR, 8.5)
    rate(session, ada, S4, values(NONE))
    assert set(labels(session)) == {S1, S2, S3, S4}

    ben = get_or_create_rater(session, "ben")
    rate(session, ben, S5, values(MINOR), marks=[mark("A", 3.5)])
    record_single(session, S5, shipped_protocol)

    # the labels that came from ratings are gone, with their marks, and none is written
    found = labels(session)
    assert set(found) == {S3, S4}
    assert found[S3] == {"unratable": (1, "single")}
    assert found[S4]["A"] == (MINOR, "adjudicated")
    assert final_marks(session) == {S4: [("A", 8.5, [])]}


def test_with_two_raters_no_submission_writes_a_label(
    session: Session, ada: Rater, ben: Rater, shipped_protocol: Protocol
) -> None:
    for rater in (ada, ben, ada):
        rate(session, rater, S1, values(MAJOR), marks=[mark("A", 1.5)])
        record_single(session, S1, shipped_protocol)
        assert labels(session) == {} and final_marks(session) == {}


def test_a_second_rater_who_has_not_rated_yet_counts(
    session: Session, ada: Rater, shipped_protocol: Protocol
) -> None:
    rate(session, ada, S1, values(MAJOR))
    record_single(session, S1, shipped_protocol)
    assert labels(session) == {S1: single(MAJOR)}
    get_or_create_rater(session, "ben")
    rate(session, ada, S2, values(MINOR))
    record_single(session, S2, shipped_protocol)
    assert labels(session) == {}


def test_a_submission_after_finalize_removes_its_single_rows_and_keeps_the_consensus(
    session: Session, ada: Rater, ben: Rater, shipped_protocol: Protocol
) -> None:
    rate(session, ada, S1, values(MAJOR), marks=[mark("A", 1.5)])
    rate(session, ben, S1, values(MINOR), marks=[mark("A", 2.5)])
    rate(session, ada, S2, values(MINOR), marks=[mark("A", 3.5)])
    finalize(session, plan_of([S1, S2]), shipped_protocol)
    assert labels(session) == {S1: consensus(MINOR), S2: single(MINOR)}

    rate(session, ben, S3, values(NONE))
    record_single(session, S3, shipped_protocol)
    assert labels(session) == {S1: consensus(MINOR)}
    assert set(final_marks(session)) == {S1}


# --- sync_unratable -----------------------------------------------------------------------------


def test_a_failed_render_gets_its_unratable_label(
    session: Session, shipped_protocol: Protocol
) -> None:
    failed = {S1: "missing surface", S2: "wrong orientation", S3: "render failure"}
    assert sync_unratable(session, plan_of([S4], failed), shipped_protocol) == 3
    assert shipped_protocol.unratable_reasons[:3] == (
        "render failure",
        "wrong orientation",
        "missing surface",
    )
    assert labels(session) == {
        S1: {"unratable": (2, "single")},
        S2: {"unratable": (1, "single")},
        S3: {"unratable": (0, "single")},
    }
    assert final_marks(session) == {}


def test_a_reason_of_the_run_counts_as_render_failure(
    session: Session, shipped_protocol: Protocol
) -> None:
    assert sync_unratable(session, plan_of(failed={S1: "source changed"}), shipped_protocol) == 1
    assert labels(session) == {S1: {"unratable": (0, "single")}}


def test_sync_is_idempotent_and_follows_a_changed_reason(
    session: Session, shipped_protocol: Protocol
) -> None:
    plan = plan_of(failed={S1: "missing surface"})
    assert sync_unratable(session, plan, shipped_protocol) == 1
    assert sync_unratable(session, plan, shipped_protocol) == 1
    assert labels(session) == {S1: {"unratable": (2, "single")}}
    assert not session.dirty and not session.new

    assert sync_unratable(session, plan_of(failed={S1: "wrong orientation"}), shipped_protocol) == 1
    assert labels(session) == {S1: {"unratable": (1, "single")}}


def test_the_label_goes_when_the_render_is_ready_again_and_nobody_rated(
    session: Session, shipped_protocol: Protocol
) -> None:
    sync_unratable(
        session, plan_of(failed={S1: "missing surface", S2: "missing surface"}), shipped_protocol
    )
    assert sync_unratable(session, plan_of([S1], {S2: "missing surface"}), shipped_protocol) == 1
    assert labels(session) == {S2: {"unratable": (2, "single")}}


def test_the_label_stays_while_the_render_is_neither_failed_nor_ready(
    session: Session, shipped_protocol: Protocol
) -> None:
    sync_unratable(session, plan_of(failed={S1: "missing surface"}), shipped_protocol)
    assert sync_unratable(session, plan_of(pending=[S1]), shipped_protocol) == 0
    assert labels(session) == {S1: {"unratable": (2, "single")}}


def test_an_unratable_label_from_a_rating_stays_when_the_render_is_ready(
    session: Session, ada: Rater, shipped_protocol: Protocol
) -> None:
    rate(session, ada, S1, unratable="image too poor to judge")
    record_single(session, S1, shipped_protocol)
    assert sync_unratable(session, plan_of([S1]), shipped_protocol) == 0
    assert labels(session) == {S1: {"unratable": (3, "single")}}


def test_a_rated_subject_whose_render_fails_keeps_what_it_was_rated_as(
    session: Session, ada: Rater, shipped_protocol: Protocol
) -> None:
    rate(session, ada, S1, values(MAJOR), marks=[mark("A", 1.5)])
    record_single(session, S1, shipped_protocol)
    before = (labels(session), final_marks(session))
    plan = plan_of(failed={S1: "missing surface", S2: "missing surface"})
    assert sync_unratable(session, plan, shipped_protocol) == 1
    assert (labels(session)[S1], final_marks(session)) == (before[0][S1], before[1])

    # with two raters a rated subject has no label until finalize, and gets none here
    get_or_create_rater(session, "ben")
    rate(session, ada, S3, values(NONE))
    record_single(session, S3, shipped_protocol)
    assert sync_unratable(session, plan, shipped_protocol) == 1
    assert set(labels(session)) == {S2}


@pytest.mark.parametrize("kind", ["repeat", "calibration"])
def test_only_a_main_assignment_is_a_rating_for_the_label_of_a_failed_render(
    session: Session, ada: Rater, shipped_protocol: Protocol, kind: str
) -> None:
    rate(session, ada, S1, values(MAJOR), kind=kind)
    assert sync_unratable(session, plan_of(failed={S1: "missing surface"}), shipped_protocol) == 1
    assert labels(session) == {S1: {"unratable": (2, "single")}}


def test_sync_leaves_an_adjudicated_subject_alone(
    session: Session, shipped_protocol: Protocol
) -> None:
    adjudicate(session, S1, MAJOR, 8.5)
    session.add(FinalLabel(subject_uid=S2, criterion="unratable", value=3, method="adjudicated"))
    session.flush()
    before = (labels(session), final_marks(session))
    assert sync_unratable(session, plan_of([S2], {S1: "missing surface"}), shipped_protocol) == 0
    assert (labels(session), final_marks(session)) == before


def test_sync_is_not_committed(
    engine: Engine, session: Session, shipped_protocol: Protocol
) -> None:
    sync_unratable(session, plan_of(failed={S1: "missing surface"}), shipped_protocol)
    session.rollback()
    session.close()
    with Session(engine) as later:
        assert labels(later) == {}


# --- finalize -----------------------------------------------------------------------------------


def test_one_rating_is_single_also_when_there_are_two_raters(
    session: Session, ada: Rater, ben: Rater, shipped_protocol: Protocol
) -> None:
    first = rate(session, ada, S1, values(MAJOR, MINOR), marks=[mark("A", 1.5), mark("B", 2.5)])
    rate(session, ben, S2, unratable="wrong orientation")
    report = finalize(session, plan_of([S1, S2]), shipped_protocol)

    assert report == FinalizeReport(consensus=0, single=2, unratable=0, open=())
    assert labels(session) == {S1: single(MAJOR, MINOR), S2: {"unratable": (1, "single")}}
    ids = live_mark_ids(session, first)
    assert final_marks(session) == {S1: [("A", 1.5, [ids[0]]), ("B", 2.5, [ids[1]])]}


def test_two_raters_within_one_step_give_the_lower_value_and_the_union_of_the_marks(
    session: Session, ada: Rater, ben: Rater, shipped_protocol: Protocol
) -> None:
    by_ada = rate(
        session, ada, S1, values(MAJOR, MINOR, MINOR), marks=[mark("A", 1.5), mark("B", 2.5)]
    )
    by_ben = rate(
        session, ben, S1, values(MINOR, MAJOR, MINOR), marks=[mark("A", 3.5), mark("C", 4.5)]
    )
    report = finalize(session, plan_of([S1]), shipped_protocol)

    assert report == FinalizeReport(consensus=1, single=0, unratable=0, open=())
    assert labels(session) == {S1: consensus(MINOR, MINOR, MINOR)}
    a, b = live_mark_ids(session, by_ada), live_mark_ids(session, by_ben)
    assert final_marks(session) == {
        S1: [("A", 1.5, [a[0]]), ("B", 2.5, [a[1]]), ("A", 3.5, [b[0]]), ("C", 4.5, [b[1]])]
    }


def test_the_union_leaves_out_the_marks_of_a_criterion_whose_consensus_is_none(
    session: Session, ada: Rater, ben: Rater, shipped_protocol: Protocol
) -> None:
    rate(
        session,
        ada,
        S1,
        values(MINOR, MINOR),
        marks=[mark("A", 1.5), mark("B", 2.5), mark("X", 3.5), mark("C", 4.5)],
    )
    rate(session, ben, S1, values(NONE, MINOR), marks=[mark("B", 5.5), mark("X", 6.5)])
    finalize(session, plan_of([S1]), shipped_protocol)

    assert labels(session) == {S1: consensus(NONE, MINOR, NONE)}
    # A and C are none: no final mark says where such an error is. X belongs to no criterion.
    assert [(c, x) for c, x, _ in final_marks(session)[S1]] == [
        ("B", 2.5),
        ("X", 3.5),
        ("B", 5.5),
        ("X", 6.5),
    ]
    # the raters' own marks are all still there
    assert len(session.exec(select(Mark).where(col(Mark.deleted_ts).is_(None))).all()) == 6


def test_two_steps_apart_is_open_and_has_no_label(
    session: Session, ada: Rater, ben: Rater, shipped_protocol: Protocol
) -> None:
    rate(session, ada, S1, values(MAJOR, MINOR), marks=[mark("A", 1.5)])
    rate(session, ben, S1, values(NONE, MINOR), marks=[mark("B", 2.5)])
    rate(session, ada, S2, values(MINOR))
    rate(session, ben, S2, values(MINOR))
    report = finalize(session, plan_of([S1, S2]), shipped_protocol)

    assert report == FinalizeReport(consensus=1, single=0, unratable=0, open=(S1,))
    assert labels(session) == {S2: consensus(MINOR)}
    assert final_marks(session) == {}


@pytest.mark.parametrize("criterion", ["A", "B", "C"])
def test_one_criterion_two_steps_apart_is_enough(
    session: Session, ada: Rater, ben: Rater, shipped_protocol: Protocol, criterion: str
) -> None:
    rate(session, ada, S1, values(MINOR, MINOR, MINOR) | {criterion: MAJOR})
    rate(session, ben, S1, values(MINOR, MINOR, MINOR) | {criterion: NONE})
    assert finalize(session, plan_of([S1]), shipped_protocol).open == (S1,)
    assert labels(session) == {}


def test_unratable_against_rated_is_open(
    session: Session, ada: Rater, ben: Rater, shipped_protocol: Protocol
) -> None:
    rate(session, ada, S1, unratable="image too poor to judge")
    rate(session, ben, S1, values(NONE))
    report = finalize(session, plan_of([S1]), shipped_protocol)
    assert report == FinalizeReport(consensus=0, single=0, unratable=0, open=(S1,))
    assert labels(session) == {}


def test_raters_who_all_say_unratable_agree_on_one_reason_only(
    session: Session, ada: Rater, ben: Rater, shipped_protocol: Protocol
) -> None:
    for rater in (ada, ben):
        rate(session, rater, S1, unratable="image too poor to judge")
    rate(session, ada, S2, unratable="image too poor to judge")
    rate(session, ben, S2, unratable="wrong orientation")
    report = finalize(session, plan_of([S1, S2]), shipped_protocol)
    assert report == FinalizeReport(consensus=1, single=0, unratable=0, open=(S2,))
    assert labels(session) == {S1: {"unratable": (3, "consensus")}}


def test_raters_who_rated_other_criteria_do_not_agree(
    session: Session, ada: Rater, ben: Rater, shipped_protocol: Protocol
) -> None:
    rate(session, ada, S1, {"A": MINOR, "B": NONE})
    rate(session, ben, S1, values(MINOR))
    assert finalize(session, plan_of([S1]), shipped_protocol).open == (S1,)


def test_three_raters_agree_when_all_are_within_one_step(
    session: Session, ada: Rater, ben: Rater, shipped_protocol: Protocol
) -> None:
    cyd = get_or_create_rater(session, "cyd")
    for rater, a, b in ((ada, MAJOR, NONE), (ben, MINOR, MINOR), (cyd, MAJOR, MINOR)):
        rate(session, rater, S1, values(a, b), marks=[mark("A", float(a))])
    # S2: each pair but one is within a step
    for rater, a in ((ada, NONE), (ben, MINOR), (cyd, MAJOR)):
        rate(session, rater, S2, values(a))
    report = finalize(session, plan_of([S1, S2]), shipped_protocol)
    assert report == FinalizeReport(consensus=1, single=0, unratable=0, open=(S2,))
    assert labels(session) == {S1: consensus(MINOR, NONE)}
    assert len(final_marks(session)[S1]) == 3


def test_finalize_takes_an_open_subjects_earlier_label_away(
    session: Session, ada: Rater, ben: Rater, shipped_protocol: Protocol
) -> None:
    rate(session, ada, S1, values(MINOR), marks=[mark("A", 1.5)])
    by_ben = rate(session, ben, S1, values(MINOR), marks=[mark("A", 2.5)])
    finalize(session, plan_of([S1]), shipped_protocol)
    assert labels(session) == {S1: consensus(MINOR)} and len(final_marks(session)[S1]) == 2

    # a revision one more step away, and nothing is left of the consensus
    submit(session, by_ben, values(MAJOR, MAJOR), marks=[mark("B", 3.5)])
    assert finalize(session, plan_of([S1]), shipped_protocol) == FinalizeReport(0, 0, 0, (S1,))
    assert labels(session) == {} and final_marks(session) == {}

    # and back within a step: the label returns
    submit(session, by_ben, values(MAJOR), marks=[mark("A", 3.5)])
    assert finalize(session, plan_of([S1]), shipped_protocol) == FinalizeReport(1, 0, 0, ())
    assert labels(session) == {S1: consensus(MINOR)}
    assert [x for _, x, _ in final_marks(session)[S1]] == [1.5, 3.5]


def test_calibration_and_repeats_are_no_ratings_and_the_latest_main_is(
    session: Session, ada: Rater, ben: Rater, shipped_protocol: Protocol
) -> None:
    rate(session, ada, S1, values(MAJOR), key="tiles-1")
    rate(session, ada, S1, values(MINOR), key="tiles-2")
    rate(session, ada, S1, values(NONE), kind="repeat", key="tiles-2")
    rate(session, ben, S1, values(MAJOR), kind="calibration")
    rate(session, ben, S2, values(MAJOR), kind="calibration")
    report = finalize(session, plan_of([S1, S2]), shipped_protocol)
    assert report == FinalizeReport(consensus=0, single=1, unratable=0, open=())
    assert labels(session) == {S1: single(MINOR)}


def test_finalize_reads_current_submissions_only(
    session: Session, ada: Rater, ben: Rater, shipped_protocol: Protocol
) -> None:
    by_ada = rate(session, ada, S1, values(MAJOR), marks=[mark("A", 1.5)])
    rate(session, ben, S1, values(NONE))
    revised = submit(session, by_ada, values(MINOR, MINOR), marks=[mark("B", 2.5)])
    report = finalize(session, plan_of([S1]), shipped_protocol)
    assert report.open == ()
    assert labels(session) == {S1: consensus(NONE, NONE)}
    assert final_marks(session) == {}
    assert revised[0].deleted_ts is None


def test_finalize_never_touches_adjudicated_rows(
    session: Session, ada: Rater, ben: Rater, shipped_protocol: Protocol
) -> None:
    adjudicate(session, S1, MAJOR, 8.5)
    rate(session, ada, S1, values(MAJOR), marks=[mark("A", 1.5)])
    rate(session, ben, S1, values(NONE))
    adjudicate(session, S2, MINOR, 9.5)
    ids = final_mark_ids(session)
    report = finalize(session, plan_of([S1], {S2: "missing surface"}), shipped_protocol)

    # neither open nor counted: the adjudicator has decided
    assert report == FinalizeReport(consensus=0, single=0, unratable=0, open=())
    assert labels(session)[S1]["A"] == (MAJOR, "adjudicated")
    assert labels(session)[S2] == {
        name: (value, "adjudicated") for name, value in values(MINOR).items()
    }
    assert final_marks(session) == {S1: [("A", 8.5, [])], S2: [("A", 9.5, [])]}
    assert final_mark_ids(session) == ids


def test_finalize_gives_a_failed_render_its_label_and_counts_it(
    session: Session, ada: Rater, ben: Rater, shipped_protocol: Protocol
) -> None:
    rate(session, ada, S1, values(MINOR))
    rate(session, ben, S1, values(MINOR))
    rate(session, ada, S2, values(MAJOR))
    plan = plan_of([S1], {S2: "missing surface", S3: "missing surface", S4: "source changed"})
    report = finalize(session, plan, shipped_protocol)

    # S2 was rated before its render failed: it is a subject with one rating
    assert report == FinalizeReport(consensus=1, single=1, unratable=2, open=())
    assert labels(session) == {
        S1: consensus(MINOR),
        S2: single(MAJOR),
        S3: {"unratable": (2, "single")},
        S4: {"unratable": (0, "single")},
    }
    # a second call finds the same, and the render that is ready again loses the label
    assert finalize(session, plan, shipped_protocol) == report
    assert (
        finalize(session, plan_of([S1, S3], {S4: "source changed"}), shipped_protocol).unratable
        == 1
    )
    assert S3 not in labels(session)


def test_finalize_with_one_rater_leaves_what_the_submissions_wrote(
    session: Session, ada: Rater, shipped_protocol: Protocol
) -> None:
    rate(session, ada, S1, values(MAJOR), marks=[mark("A", 1.5), mark("X", 2.5)])
    record_single(session, S1, shipped_protocol)
    rate(session, ada, S2, unratable="image too poor to judge")
    record_single(session, S2, shipped_protocol)
    before = (labels(session), final_marks(session), final_mark_ids(session))

    report = finalize(session, plan_of([S1, S2]), shipped_protocol)
    assert report == FinalizeReport(consensus=0, single=2, unratable=0, open=())
    assert (labels(session), final_marks(session), final_mark_ids(session)) == before
    assert not session.dirty and not session.new and not session.deleted


def test_finalize_labels_a_rated_subject_that_the_plan_does_not_have(
    session: Session, ada: Rater, shipped_protocol: Protocol
) -> None:
    rate(session, ada, S1, values(MINOR))
    assert finalize(session, plan_of(), shipped_protocol).single == 1
    assert labels(session) == {S1: single(MINOR)}


def test_finalize_removes_rows_that_no_rating_explains(
    session: Session, ada: Rater, shipped_protocol: Protocol
) -> None:
    session.add(FinalLabel(subject_uid=S1, criterion="A", value=MAJOR, method="consensus"))
    session.add(
        FinalMark(
            subject_uid=S1, mark_class="A", tkr_x=1.0, tkr_y=0.0, tkr_z=0.0, source_mark_ids="[]"
        )
    )
    session.flush()
    report = finalize(session, plan_of([S1]), shipped_protocol)
    assert report == FinalizeReport(consensus=0, single=0, unratable=0, open=())
    assert labels(session) == {} and final_marks(session) == {}


def test_open_subjects_are_sorted(
    session: Session, ada: Rater, ben: Rater, shipped_protocol: Protocol
) -> None:
    for subject_uid in (S3, S1, S2):
        rate(session, ada, subject_uid, values(MAJOR))
        rate(session, ben, subject_uid, values(NONE))
    assert finalize(session, plan_of(), shipped_protocol).open == (S1, S2, S3)


def test_a_reason_the_protocol_lost_stops_finalize_before_it_writes(
    session: Session, ada: Rater, ben: Rater, shipped_protocol: Protocol
) -> None:
    rate(session, ada, S1, values(MAJOR))
    record_single(session, S1, shipped_protocol)
    get_or_create_rater(session, "ben")
    rate(session, ada, S2, unratable="image too poor to judge")
    session.add(FinalLabel(subject_uid=S3, criterion="A", value=MINOR, method="consensus"))
    before = labels(session)

    reasons = shipped_protocol.unratable_reasons[:3]
    shorter = shipped_protocol.model_copy(update={"unratable_reasons": reasons})
    with pytest.raises(ConfigError) as caught:
        finalize(session, plan_of([S1, S2]), shorter)
    assert caught.value.reason == "invalid"
    assert S2 in caught.value.detail and "image too poor to judge" in caught.value.detail
    assert PROTOCOL_FILE in caught.value.detail
    assert labels(session) == before
    assert not session.dirty and not session.new and not session.deleted


def test_finalize_is_not_committed(
    engine: Engine, session: Session, ada: Rater, ben: Rater, shipped_protocol: Protocol
) -> None:
    rate(session, ada, S1, values(MINOR))
    rate(session, ben, S1, values(MINOR))
    session.commit()
    finalize(session, plan_of([S1], {S2: "missing surface"}), shipped_protocol)
    assert set(labels(session)) == {S1, S2}
    session.rollback()
    session.close()
    with Session(engine) as later:
        assert labels(later) == {}


# --- on a run: a plan from the catalog, and the calibration set ---------------------------------


def rated_run(
    app_engine: Engine, uids: Mapping[str, str], protocol: Protocol, *, second_rater: bool = False
) -> dict[str, list[int]]:
    """Rate the three subjects of ``app_run`` and commit: ``sub-000`` major on A with two
    marks, ``sub-001`` unratable, ``sub-002`` none. With a second rater who agrees within a
    step on ``sub-000``, the labels come from ``finalize``. Returns the live mark ids of
    ``sub-000`` per rater name."""
    plan = plan_of(list(uids.values()))
    marks: dict[str, list[int]] = {}
    with Session(app_engine) as session:
        ada = get_or_create_rater(session, "ada", role="lead")
        first = rate(
            session,
            ada,
            uids["sub-000"],
            values(MAJOR, NONE, MINOR),
            marks=[mark("A", 1.5, vertex=7), mark("C", -2.25)],
        )
        record_single(session, uids["sub-000"], protocol)
        rate(session, ada, uids["sub-001"], unratable="image too poor to judge")
        record_single(session, uids["sub-001"], protocol)
        rate(session, ada, uids["sub-002"], values(NONE))
        record_single(session, uids["sub-002"], protocol)
        marks["ada"] = live_mark_ids(session, first)
        if second_rater:
            ben = get_or_create_rater(session, "ben", role="lead")
            second = rate(
                session, ben, uids["sub-000"], values(MINOR, NONE, MINOR), marks=[mark("A", 3.5)]
            )
            record_single(session, uids["sub-000"], protocol)
            assert labels(session) == {}
            finalize(session, plan, protocol)
            marks["ben"] = live_mark_ids(session, second)
        session.commit()
    return marks


def selection_rows(run_dir: Path) -> list[tuple[str, str, str, float, int]]:
    with PublicCatalog.open(run_dir) as public:
        frame = public.selection()
    return [
        (str(uid), str(reason), str(stratum), float(prob), int(seed))
        for uid, reason, stratum, prob, seed in frame.itertuples(index=False, name=None)
    ]


def test_finalize_on_the_plan_of_a_run(
    app_run: Path, app_engine: Engine, small_run_uids: dict[str, str], shipped_protocol: Protocol
) -> None:
    config = load_config(app_run)
    with PublicCatalog.open(app_run) as public:
        plan = build_plan(
            public,
            primary_scores(app_run),
            config.app,
            "local",
            render_config_hash=config.render.config_hash(),
        )
    assert set(plan.render_keys) == set(small_run_uids.values()) and not plan.failed
    rated_run(app_engine, small_run_uids, shipped_protocol, second_rater=True)
    with Session(app_engine) as session:
        report = finalize(session, plan, shipped_protocol)
        assert report == FinalizeReport(consensus=1, single=2, unratable=0, open=())
        # a render of the run that failed: the plan says so, and the subject is labelled
        failed = QueuePlan(
            "local", plan.order, {}, dict.fromkeys(plan.order, "render failure"), 0, 0
        )
        assert finalize(session, failed, shipped_protocol) == report


def test_freeze_writes_the_final_labels_and_marks_of_every_rated_subject(
    app_run: Path, app_engine: Engine, small_run_uids: dict[str, str], shipped_protocol: Protocol
) -> None:
    rated_run(app_engine, small_run_uids, shipped_protocol)
    before = selection_rows(app_run)
    uids = sorted(small_run_uids.values())
    assert [row[:2] for row in before] == [(uid, "all") for uid in uids]

    path = freeze_calibration(app_run)
    assert path == app_run / CALIBRATION_FILE

    loaded = load_calibration(app_run)
    assert loaded is not None
    calibration, _ = loaded
    assert calibration.version == "1"
    assert [subject.subject_uid for subject in calibration.subjects] == uids
    major = calibration.subject(small_run_uids["sub-000"])
    unratable = calibration.subject(small_run_uids["sub-001"])
    clean = calibration.subject(small_run_uids["sub-002"])
    assert major is not None and unratable is not None and clean is not None
    assert major.values == values(MAJOR, NONE, MINOR) and major.unratable is None
    assert [(m.mark_class, m.tkr) for m in major.marks] == [
        ("A", (1.5, 3.0, -1.5)),
        ("C", (-2.25, -4.5, 2.25)),
    ]
    assert unratable.values == {} and unratable.unratable == "image too poor to judge"
    assert unratable.marks == ()
    assert clean.values == values(NONE) and clean.marks == ()
    # justifications and examples are written by hand afterwards
    assert all(s.justification == "" and s.examples == () for s in calibration.subjects)
    assert path.read_text(encoding="utf-8") == dump_calibration(calibration)

    # the file passes the check the server makes at start
    with PublicCatalog.open(app_run) as public:
        known = public.subject_uids(present=None)
    validate_calibration(calibration, shipped_protocol, subjects=known, render=lambda _uid: None)

    # the selection gained the calibration rows and kept the others
    after = selection_rows(app_run)
    assert [row for row in after if row[1] != CALIBRATION_REASON] == before
    assert [row for row in after if row[1] == CALIBRATION_REASON] == [
        (uid, "calibration", "calibration", 1.0, 0) for uid in uids
    ]


def test_freeze_takes_the_consensus_labels_and_marks_of_finalize(
    app_run: Path, app_engine: Engine, small_run_uids: dict[str, str], shipped_protocol: Protocol
) -> None:
    rated_run(app_engine, small_run_uids, shipped_protocol, second_rater=True)
    freeze_calibration(app_run, version="pilot-2")
    loaded = load_calibration(app_run)
    assert loaded is not None
    calibration, _ = loaded
    assert calibration.version == "pilot-2"
    agreed = calibration.subject(small_run_uids["sub-000"])
    assert agreed is not None
    assert agreed.values == values(MINOR, NONE, MINOR)
    assert [(m.mark_class, m.tkr[0]) for m in agreed.marks] == [
        ("A", 1.5),
        ("C", -2.25),
        ("A", 3.5),
    ]


def test_freeze_refuses_to_overwrite(
    app_run: Path, app_engine: Engine, small_run_uids: dict[str, str], shipped_protocol: Protocol
) -> None:
    rated_run(app_engine, small_run_uids, shipped_protocol)
    path = freeze_calibration(app_run, subjects=[small_run_uids["sub-000"]])
    written, selected = path.read_bytes(), selection_rows(app_run)

    with pytest.raises(ConfigError) as caught:
        freeze_calibration(app_run)
    assert caught.value.reason == "invalid" and CALIBRATION_FILE in caught.value.detail
    assert path.read_bytes() == written
    assert selection_rows(app_run) == selected

    # removed by hand, the set can be frozen again, and its selection rows are replaced
    path.unlink()
    freeze_calibration(app_run, subjects=[small_run_uids["sub-002"]])
    assert [row[0] for row in selection_rows(app_run) if row[1] == CALIBRATION_REASON] == [
        small_run_uids["sub-002"]
    ]


def test_freeze_takes_the_named_subjects_only(
    app_run: Path, app_engine: Engine, small_run_uids: dict[str, str], shipped_protocol: Protocol
) -> None:
    rated_run(app_engine, small_run_uids, shipped_protocol)
    named = [small_run_uids["sub-002"], small_run_uids["sub-000"], small_run_uids["sub-002"]]
    freeze_calibration(app_run, subjects=named)
    loaded = load_calibration(app_run)
    assert loaded is not None
    assert [s.subject_uid for s in loaded[0].subjects] == sorted(set(named))
    assert [row[0] for row in selection_rows(app_run) if row[1] == CALIBRATION_REASON] == sorted(
        set(named)
    )


def test_no_name_at_all_means_every_subject(
    app_run: Path, app_engine: Engine, small_run_uids: dict[str, str], shipped_protocol: Protocol
) -> None:
    rated_run(app_engine, small_run_uids, shipped_protocol)
    freeze_calibration(app_run, subjects=[])
    loaded = load_calibration(app_run)
    assert loaded is not None and len(loaded[0].subjects) == 3


@pytest.mark.parametrize("name", ["sub-002", "ffffffffffffffff", "sub-000"])
def test_a_named_subject_without_a_final_label_is_refused_and_nothing_is_written(
    app_run: Path,
    app_engine: Engine,
    small_run_uids: dict[str, str],
    shipped_protocol: Protocol,
    name: str,
) -> None:
    """A subject nobody rated, a uid the run does not have, and a label in place of a uid."""
    with Session(app_engine) as session:
        ada = get_or_create_rater(session, "ada", role="lead")
        for label in ("sub-000", "sub-001"):
            rate(session, ada, small_run_uids[label], values(MINOR))
            record_single(session, small_run_uids[label], shipped_protocol)
        session.commit()
    unlabelled = small_run_uids["sub-002"] if name == "sub-002" else name
    before = selection_rows(app_run)

    with pytest.raises(ConfigError) as caught:
        freeze_calibration(app_run, subjects=[small_run_uids["sub-000"], unlabelled])
    assert caught.value.reason == "invalid"
    assert unlabelled in caught.value.detail and "no final label" in caught.value.detail
    assert not (app_run / CALIBRATION_FILE).exists()
    assert selection_rows(app_run) == before


def test_freeze_needs_a_final_label(
    app_run: Path, app_engine: Engine, small_run_uids: dict[str, str], shipped_protocol: Protocol
) -> None:
    before = selection_rows(app_run)
    with pytest.raises(ConfigError) as caught:
        freeze_calibration(app_run)
    assert caught.value.reason == "invalid" and "final label" in caught.value.detail

    # two raters and no finalize yet: there are ratings, and still no label
    with Session(app_engine) as session:
        for name in ("ada", "ben"):
            rater = get_or_create_rater(session, name, role="lead")
            rate(session, rater, small_run_uids["sub-000"], values(MINOR))
            record_single(session, small_run_uids["sub-000"], shipped_protocol)
        session.commit()
    with pytest.raises(ConfigError) as caught:
        freeze_calibration(app_run)
    assert caught.value.reason == "invalid" and "surfcensus finalize" in caught.value.detail
    assert not (app_run / CALIBRATION_FILE).exists()
    assert selection_rows(app_run) == before


def test_freeze_on_a_run_without_ratings_makes_no_database(app_run: Path) -> None:
    assert not app_db_path(app_run).exists()
    with pytest.raises(ConfigError) as caught:
        freeze_calibration(app_run)
    assert caught.value.reason == "invalid" and APP_DB_FILE in caught.value.detail
    assert not app_db_path(app_run).exists()
    assert not (app_run / CALIBRATION_FILE).exists()


def test_a_subject_that_could_not_be_rendered_is_no_calibration_subject(
    app_run: Path, app_engine: Engine, small_run_uids: dict[str, str], shipped_protocol: Protocol
) -> None:
    rated, never = small_run_uids["sub-000"], small_run_uids["sub-001"]
    with Session(app_engine) as session:
        ada = get_or_create_rater(session, "ada", role="lead")
        rate(session, ada, rated, values(MINOR))
        record_single(session, rated, shipped_protocol)
        sync_unratable(session, plan_of([rated], {never: "missing surface"}), shipped_protocol)
        assert set(labels(session)) == {rated, never}
        session.commit()

    with pytest.raises(ConfigError) as caught:
        freeze_calibration(app_run, subjects=[never])
    assert caught.value.reason == "invalid"
    assert never in caught.value.detail and "not rated" in caught.value.detail
    assert not (app_run / CALIBRATION_FILE).exists()

    freeze_calibration(app_run)
    loaded = load_calibration(app_run)
    assert loaded is not None
    assert [s.subject_uid for s in loaded[0].subjects] == [rated]


def test_only_subjects_that_could_not_be_rendered_is_no_calibration_set(
    app_run: Path, app_engine: Engine, small_run_uids: dict[str, str], shipped_protocol: Protocol
) -> None:
    with Session(app_engine) as session:
        failed = dict.fromkeys(small_run_uids.values(), "missing surface")
        assert sync_unratable(session, plan_of(failed=failed), shipped_protocol) == 3
        session.commit()
    with pytest.raises(ConfigError) as caught:
        freeze_calibration(app_run)
    assert caught.value.reason == "invalid" and "from a rating" in caught.value.detail
    assert not (app_run / CALIBRATION_FILE).exists()


def test_freeze_keeps_adjudicated_labels_and_marks(
    app_run: Path, app_engine: Engine, small_run_uids: dict[str, str]
) -> None:
    decided = small_run_uids["sub-001"]
    with Session(app_engine) as session:
        adjudicate(session, decided, MAJOR, 8.5)
        session.commit()
    freeze_calibration(app_run)
    loaded = load_calibration(app_run)
    assert loaded is not None
    (subject,) = loaded[0].subjects
    assert subject.subject_uid == decided and subject.values == values(MAJOR)
    assert [(m.mark_class, m.tkr) for m in subject.marks] == [("A", (8.5, 0.0, 0.0))]


def test_labels_that_do_not_fit_the_protocol_are_refused_before_anything_is_written(
    app_run: Path, app_engine: Engine, small_run_uids: dict[str, str], shipped_protocol: Protocol
) -> None:
    uid = small_run_uids["sub-000"]
    with Session(app_engine) as session:
        ada = get_or_create_rater(session, "ada", role="lead")
        rate(session, ada, uid, {"A": MINOR, "B": NONE})  # rated under a protocol without C
        record_single(session, uid, shipped_protocol)
        session.commit()
    before = selection_rows(app_run)
    with pytest.raises(ConfigError) as caught:
        freeze_calibration(app_run)
    assert caught.value.reason == "invalid" and uid in caught.value.detail
    assert not (app_run / CALIBRATION_FILE).exists()
    assert selection_rows(app_run) == before


def test_an_unratable_position_the_protocol_lacks_is_refused(
    app_run: Path, app_engine: Engine, small_run_uids: dict[str, str]
) -> None:
    uid = small_run_uids["sub-000"]
    with Session(app_engine) as session:
        session.add(
            FinalLabel(subject_uid=uid, criterion="unratable", value=9, method="adjudicated")
        )
        session.commit()
    with pytest.raises(ConfigError) as caught:
        freeze_calibration(app_run)
    assert caught.value.reason == "invalid"
    assert uid in caught.value.detail and PROTOCOL_FILE in caught.value.detail
    assert not (app_run / CALIBRATION_FILE).exists()


def test_freeze_needs_the_protocol_of_the_run(
    app_run: Path, app_engine: Engine, small_run_uids: dict[str, str], shipped_protocol: Protocol
) -> None:
    rated_run(app_engine, small_run_uids, shipped_protocol)
    (app_run / PROTOCOL_FILE).unlink()
    with pytest.raises(ConfigError) as caught:
        freeze_calibration(app_run)
    assert caught.value.reason == "missing"
    assert not (app_run / CALIBRATION_FILE).exists()


def test_freeze_leaves_the_database_as_it_was(
    app_run: Path, app_engine: Engine, small_run_uids: dict[str, str], shipped_protocol: Protocol
) -> None:
    rated_run(app_engine, small_run_uids, shipped_protocol)

    def content() -> tuple[Any, ...]:
        with Session(app_engine) as session:
            found = (labels(session), final_marks(session), final_mark_ids(session))
            submissions = len(session.exec(select(Submission)).all())
        return (*found, submissions)

    before = content()
    freeze_calibration(app_run)
    assert content() == before


def test_the_frozen_set_is_a_calibration_set(
    app_run: Path, app_engine: Engine, small_run_uids: dict[str, str], shipped_protocol: Protocol
) -> None:
    rated_run(app_engine, small_run_uids, shipped_protocol)
    freeze_calibration(app_run)
    loaded = load_calibration(app_run)
    assert loaded is not None and isinstance(loaded[0], CalibrationSet)
    assert len(loaded[1]) == 64
