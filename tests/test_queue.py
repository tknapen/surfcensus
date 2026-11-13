"""Tests of surfcensus/app/queue.py: the plan of a run and which subject a rater gets next.

Most tests give the queue a plan made by hand and a new ``app.sqlite``. The tests of
``build_plan`` read a catalog: that of a light synthetic tree whose ``renders`` rows are
written by hand (nothing is rendered), and that of ``small_run``.
"""

from __future__ import annotations

import dataclasses
import random
import shutil
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from sqlalchemy.engine import Engine
from sqlmodel import Session, col, select
from synth_tree import write_tree  # pyrefly: ignore[missing-import]

from surfcensus import store
from surfcensus.app.db import app_db_path, get_or_create_rater, open_app_db, store_submission
from surfcensus.app.models import Assignment, Rater
from surfcensus.app.queue import (
    PREFETCH,
    Next,
    QueuePlan,
    build_plan,
    has_work_left,
    next_assignment,
    previous_assignment,
    subjects_left,
)
from surfcensus.app.schemas import AppMode, RatingIn
from surfcensus.calibration import CalibrationSet, CalibrationSubject
from surfcensus.config import AppConfig, load_config
from surfcensus.records import RenderRow
from surfcensus.run import Run
from surfcensus.sources.local import local
from surfcensus.store import CATALOG_FILE, PublicCatalog

RUN_ID = "0f1e2d3c4b5a6978"
HASH = "17f83a0c9b6c"
"""The config hash that the plans of these tests are built for."""
OLD_HASH = "dff40fe66869"
LABELS = tuple(f"sub-{n:02d}" for n in range(8))
T = tuple(f"2026-11-13T10:00:{n:02d}.000Z" for n in range(60))


# --- a plan by hand, and a rater who rates ------------------------------------------------------


def key_of(uid: str) -> str:
    return f"key-{uid}"


def plan_of(
    order: Sequence[str],
    *,
    keys: Mapping[str, str] | None = None,
    failed: Mapping[str, str] | None = None,
    pending: int = 0,
    outdated: int = 0,
    **settings: Any,
) -> QueuePlan:
    """A local plan in which every subject of ``order`` is ready, unless ``keys`` says
    which are."""
    failed = {} if failed is None else failed
    if keys is None:
        keys = {uid: key_of(uid) for uid in order if uid not in failed}
    return QueuePlan("local", tuple(order), dict(keys), dict(failed), pending, outdated, **settings)


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
def rater(session: Session) -> Rater:
    return get_or_create_rater(session, "local", role="lead")


def complete(session: Session, assignment: Assignment, *, now: str | None = None) -> None:
    """Rate the assignment as the server does: one more submission."""
    assert assignment.assignment_id is not None
    rating = RatingIn(
        assignment_id=assignment.assignment_id,
        protocol_version="0.1-draft",
        values={"A": 0, "B": 0, "C": 0},
        ms_on_subject=1,
    )
    store_submission(
        session, assignment, rating, (), protocol_sha256="9c", app_version="0", now=now
    )


def assigned(session: Session, rater: Rater, plan: QueuePlan) -> Assignment:
    following = next_assignment(session, rater, plan)
    assert following.status == "assignment" and following.assignment is not None
    return following.assignment


def rate_all(
    session: Session, rater: Rater, plan: QueuePlan, *, limit: int = 2000
) -> list[Assignment]:
    """Rate whatever the queue gives until it gives nothing; the assignments in order."""
    rated: list[Assignment] = []
    while (following := next_assignment(session, rater, plan)).status == "assignment":
        assert following.assignment is not None and len(rated) < limit
        complete(session, following.assignment)
        rated.append(following.assignment)
    return rated


def all_assignments(session: Session) -> list[Assignment]:
    return list(session.exec(select(Assignment).order_by(col(Assignment.assignment_id))).all())


def subjects(assignments: Sequence[Assignment]) -> list[str]:
    return [assignment.subject_uid for assignment in assignments]


# --- a catalog whose renders are written by hand ------------------------------------------------


@pytest.fixture(scope="module")
def light_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The run directory of a local run on eight light subjects, inventoried and nothing
    else: no selection, no render row. Read-only; a test takes ``catalog_run``."""
    base = tmp_path_factory.mktemp("queue_light")
    root = write_tree(base / "subjects", "subjects_dir", LABELS)
    with Run(base / "run", local(root)) as run:
        run.inventory()
    return base / "run"


@pytest.fixture
def catalog_run(light_run: Path, tmp_path: Path) -> Path:
    run_dir = tmp_path / "catalog_run"
    run_dir.mkdir()
    shutil.copy(light_run / CATALOG_FILE, run_dir / CATALOG_FILE)
    return run_dir


@pytest.fixture
def uids(catalog_run: Path) -> list[str]:
    with PublicCatalog.open(catalog_run) as public:
        found = public.subject_uids()
    assert len(found) == len(LABELS) and found == sorted(found)
    return found


def select_subjects(run_dir: Path, uids: Sequence[str], reason: str = "all") -> None:
    rows = pd.DataFrame(
        {
            "subject_uid": list(uids),
            "reason": reason,
            "stratum": "all",
            "inclusion_prob": 1.0,
            "seed": 0,
        }
    )
    with PublicCatalog.open(run_dir, readonly=False) as public:
        public.replace_selection(rows, [reason])


def record_render(
    run_dir: Path,
    uid: str,
    *,
    status: str = "done",
    config_hash: str = HASH,
    render_key: str | None = None,
    reason: str | None = None,
    video: bool = False,
) -> None:
    done = status == "done"
    row = RenderRow(
        subject_uid=uid,
        config_hash=config_hash,
        input_signature="0" * 16,
        video=video,
        status="done" if done else "failed",
        tile_hash="c0ffee" if done else None,
        render_key=(render_key or key_of(uid)) if done else None,
        reason=reason,
        media_path=f"media/{uid}" if done else None,
    )
    with PublicCatalog.open(run_dir, readonly=False) as public:
        public.record_renders([row])


def forget_render(run_dir: Path, uid: str) -> None:
    connection = sqlite3.connect(run_dir / CATALOG_FILE)
    with connection:
        connection.execute("DELETE FROM renders WHERE subject_uid = ?", (uid,))
    connection.close()


def planned(
    run_dir: Path,
    scores: pd.DataFrame | None = None,
    *,
    app: AppConfig | None = None,
    mode: AppMode = "local",
    render_config_hash: str = HASH,
    calibration: CalibrationSet | None = None,
) -> QueuePlan:
    with PublicCatalog.open(run_dir) as public:
        return build_plan(
            public,
            scores,
            AppConfig(queue_seed=7) if app is None else app,
            mode,
            render_config_hash=render_config_hash,
            calibration=calibration,
        )


def scores_of(rows: Sequence[tuple[str, int | None, str]]) -> pd.DataFrame:
    """Primary rows of a scores table, as far as the order of the queue reads them."""
    return pd.DataFrame(
        {
            "subject_uid": pd.Series([row[0] for row in rows], dtype=object),
            "primary_rank": pd.array([row[1] for row in rows], dtype="Int32"),
            "flags": pd.Series([row[2] for row in rows], dtype=object),
        }
    )


# --- build_plan: what the catalog says of the renders -------------------------------------------


def test_every_selected_subject_is_ready_failed_pending_or_outdated(
    catalog_run: Path, uids: list[str]
) -> None:
    ready, re_encoded, failed, stale_done, stale_failed, no_row, not_selected, _ = uids
    select_subjects(catalog_run, uids[:6])
    record_render(catalog_run, ready)
    record_render(catalog_run, re_encoded, render_key="5d0c1f2a9b7e4c31", video=True)
    record_render(catalog_run, failed, status="failed", reason="missing surface")
    record_render(catalog_run, stale_done, config_hash=OLD_HASH)
    record_render(catalog_run, stale_failed, status="failed", config_hash=OLD_HASH, reason="x")
    record_render(catalog_run, not_selected)

    plan = planned(catalog_run)
    assert plan.mode == "local"
    assert plan.render_keys == {ready: key_of(ready), re_encoded: "5d0c1f2a9b7e4c31"}
    assert plan.failed == {failed: "missing surface"}
    # no row yet; and two rows of another config, one done and one failed, which a render
    # of the present config makes again
    assert (plan.pending, plan.outdated) == (1, 2)
    assert set(plan.order) == set(uids[:6]) and len(plan.order) == 6
    assert not_selected not in plan.order and not_selected not in plan.render_keys
    assert no_row not in plan.render_keys and no_row not in plan.failed

    # the same rows under the hash they were made with: what was outdated is ready or failed
    earlier = planned(catalog_run, render_config_hash=OLD_HASH)
    assert earlier.render_keys == {stale_done: key_of(stale_done)}
    assert earlier.failed == {stale_failed: "x"}
    assert (earlier.pending, earlier.outdated) == (1, 3)


def test_a_failure_without_a_reason_is_a_render_failure(catalog_run: Path, uids: list[str]) -> None:
    select_subjects(catalog_run, uids[:1])
    record_render(catalog_run, uids[0], status="failed", reason=None)
    assert planned(catalog_run).failed == {uids[0]: "render failure"}


def test_a_run_without_a_selection_has_an_empty_plan(catalog_run: Path, uids: list[str]) -> None:
    record_render(catalog_run, uids[0])
    plan = planned(catalog_run)
    assert plan.order == () and plan.render_keys == {} and plan.failed == {}
    assert (plan.pending, plan.outdated) == (0, 0)


def test_a_subject_selected_for_two_reasons_is_in_the_plan_once(
    catalog_run: Path, uids: list[str]
) -> None:
    select_subjects(catalog_run, uids[:3])
    select_subjects(catalog_run, uids[1:2], reason="calibration")
    plan = planned(catalog_run)
    assert sorted(plan.order) == uids[:3]
    assert plan.pending == 3


def test_the_plan_carries_the_settings_of_the_app_and_the_calibration_set(
    catalog_run: Path, uids: list[str]
) -> None:
    select_subjects(catalog_run, uids[:2])
    app = AppConfig(queue_seed=41, raters_per_subject=3, repeat_rate=0.05, repeat_gap=17)
    calibration = CalibrationSet(
        version="1",
        subjects=(CalibrationSubject(subject_uid=uids[1]), CalibrationSubject(subject_uid=uids[0])),
    )
    plan = planned(catalog_run, app=app, calibration=calibration)
    assert (plan.raters_per_subject, plan.repeat_rate, plan.repeat_gap, plan.seed) == (
        3,
        0.05,
        17,
        41,
    )
    assert plan.calibration == (uids[1], uids[0])
    assert planned(catalog_run).calibration == ()


# --- build_plan: the order ----------------------------------------------------------------------


def test_the_open_order_is_partial_scores_first_then_by_primary_rank_then_the_unscored(
    catalog_run: Path, uids: list[str]
) -> None:
    select_subjects(catalog_run, uids)
    a, b, c, d, e, f, g, h = uids
    scores = scores_of(
        [
            (a, 4, ""),
            (b, 1, "no_stats,partial_score"),
            (c, 3, ""),
            (d, 2, "sphere_flipped"),
            (e, 5, "partial_score"),
            (f, 1 + 5, ""),
            # g and h have no score; a subject that is not selected is not in the order
            ("f" * 16, 7, ""),
        ]
    )
    plan = planned(catalog_run, scores)
    assert plan.order == (b, e, d, c, a, f, g, h)
    # without scores: by subject_uid
    assert planned(catalog_run, None).order == tuple(uids)
    assert planned(catalog_run, scores.iloc[:0]).order == tuple(uids)


def test_the_blind_order_is_the_seeded_shuffle_whatever_the_scores(
    catalog_run: Path, uids: list[str]
) -> None:
    select_subjects(catalog_run, uids)
    scores = scores_of([(uid, rank, "") for rank, uid in enumerate(reversed(uids), start=1)])

    def blind(seed: int) -> tuple[str, ...]:
        return planned(catalog_run, scores, app=AppConfig(queue_seed=seed, blind=True)).order

    # random.Random(7).shuffle of eight sorted items, as positions
    assert blind(7) == tuple(uids[position] for position in (6, 7, 2, 4, 0, 3, 1, 5))
    assert blind(7) == planned(catalog_run, None, app=AppConfig(queue_seed=7, blind=True)).order
    assert sorted(blind(8)) == uids and blind(8) != blind(7)
    # the same seed without blind is the order of the scores
    assert planned(catalog_run, scores, app=AppConfig(queue_seed=7)).order == tuple(reversed(uids))


def test_the_order_of_the_plan_is_the_order_a_local_run_renders_in(
    catalog_run: Path, uids: list[str]
) -> None:
    select_subjects(catalog_run, uids)
    scores = scores_of([(uid, rank, "") for rank, uid in enumerate(reversed(uids), start=1)])
    for is_blind in (False, True):
        app = AppConfig(queue_seed=3, blind=is_blind)
        expected = store.queue_order(uids, scores, blind=is_blind, seed=3)
        assert list(planned(catalog_run, scores, app=app).order) == expected


def test_the_team_order_is_by_subject_uid(catalog_run: Path, uids: list[str]) -> None:
    select_subjects(catalog_run, uids)
    scores = scores_of([(uid, rank, "") for rank, uid in enumerate(reversed(uids), start=1)])
    plan = planned(catalog_run, scores, app=AppConfig(queue_seed=7, blind=True), mode="team")
    assert plan.mode == "team" and plan.order == tuple(uids)


def test_the_plan_of_a_rendered_run(
    app_run: Path, small_run_uids: dict[str, str], app_engine: Engine
) -> None:
    config = load_config(app_run)
    scores = store.primary_scores(app_run)
    assert len(scores) == 3
    with PublicCatalog.open(app_run) as public:
        plan = build_plan(
            public, scores, config.app, "local", render_config_hash=config.render.config_hash()
        )
        renders = public.renders()
        stale = build_plan(public, scores, config.app, "local", render_config_hash=OLD_HASH)
    by_rank = list(scores["subject_uid"])
    partial = [
        uid for uid, flags in zip(by_rank, scores["flags"], strict=True) if "partial" in flags
    ]
    assert by_rank[0] == small_run_uids["sub-001"]
    assert list(plan.order) == sorted(partial) + [uid for uid in by_rank if uid not in partial]
    assert plan.render_keys == dict(zip(renders["subject_uid"], renders["render_key"], strict=True))
    assert len(plan.render_keys) == 3 and not plan.failed
    assert (plan.pending, plan.outdated, plan.seed) == (0, 0, config.app.queue_seed)
    # a version of the package with another render config finds nothing ready
    assert stale.render_keys == {} and (stale.pending, stale.outdated) == (0, 3)

    # the three subjects have the same images, so one render key: each is still offered
    assert len(set(plan.render_keys.values())) == 1
    with Session(app_engine) as session:
        rater = get_or_create_rater(session, "local", role="lead")
        assert subjects(rate_all(session, rater, plan)) == list(plan.order)
        assert next_assignment(session, rater, stale).status == "rendering"


# --- next_assignment: the walk ------------------------------------------------------------------


def test_the_first_assignment_is_the_first_subject_of_the_order(
    session: Session, rater: Rater
) -> None:
    plan = plan_of(["c", "a", "b", "d"])
    following = next_assignment(session, rater, plan)
    assignment = following.assignment
    assert following.status == "assignment" and assignment is not None
    assert (assignment.subject_uid, assignment.kind, assignment.seq) == ("c", "main", 1)
    assert assignment.render_key == key_of("c") and assignment.rater_id == rater.rater_id
    assert assignment.assignment_id is not None  # flushed
    assert assignment.completed_at is None and assignment.consensus_served_at is None
    assert assignment.assigned_at.endswith("Z") and assignment.assigned_at.startswith("20")
    assert following.prefetch == (("a", key_of("a")), ("b", key_of("b")))
    assert len(following.prefetch) == PREFETCH


def test_a_new_assignment_is_not_committed(engine: Engine, tmp_path: Path) -> None:
    plan = plan_of(["a", "b"])
    with Session(engine) as session:
        rater = get_or_create_rater(session, "local", role="lead")
        session.commit()
        assigned(session, rater, plan)
        session.rollback()
        assert all_assignments(session) == []
        first = assigned(session, rater, plan)
        assert (first.subject_uid, first.seq) == ("a", 1)
        session.commit()
    connection = sqlite3.connect(app_db_path(tmp_path))
    assert connection.execute("SELECT subject_uid, seq FROM assignments").fetchall() == [("a", 1)]
    connection.close()


def test_an_open_assignment_is_returned_again(session: Session, rater: Rater) -> None:
    plan = plan_of(["a", "b", "c", "d"])
    first = next_assignment(session, rater, plan)
    for _ in range(3):
        again = next_assignment(session, rater, plan)
        assert again.status == "assignment" and again.assignment is first.assignment
        assert again.prefetch == first.prefetch == (("b", key_of("b")), ("c", key_of("c")))
    assert len(all_assignments(session)) == 1
    # also when the order changed meanwhile (scores arrived): the page keeps its subject
    reordered = plan_of(["d", "c", "b", "a"])
    again = next_assignment(session, rater, reordered)
    assert again.assignment is first.assignment
    assert again.prefetch == (("d", key_of("d")), ("c", key_of("c")))
    assert len(all_assignments(session)) == 1


def test_the_walk_gives_every_subject_once_in_order_and_then_done(
    session: Session, rater: Rater
) -> None:
    order = ["d", "b", "a", "c"]
    plan = plan_of(order)
    prefetched: list[tuple[tuple[str, str], ...]] = []
    for position, uid in enumerate(order):
        following = next_assignment(session, rater, plan)
        assert following.assignment is not None
        assert (following.assignment.subject_uid, following.assignment.seq) == (uid, position + 1)
        assert following.assignment.kind == "main"
        prefetched.append(following.prefetch)
        complete(session, following.assignment)
    assert [subjects_of for subjects_of in prefetched] == [
        (("b", key_of("b")), ("a", key_of("a"))),
        (("a", key_of("a")), ("c", key_of("c"))),
        (("c", key_of("c")),),
        (),
    ]
    assert next_assignment(session, rater, plan) == Next("done")
    assert next_assignment(session, rater, plan) == Next("done")
    assert len(all_assignments(session)) == 4


def test_a_subject_whose_render_failed_is_never_offered(session: Session, rater: Rater) -> None:
    plan = plan_of(["a", "b", "c"], failed={"a": "missing surface", "c": "render failure"})
    first = next_assignment(session, rater, plan)
    assert first.assignment is not None and first.assignment.subject_uid == "b"
    assert first.prefetch == ()
    complete(session, first.assignment)
    assert next_assignment(session, rater, plan) == Next("done")
    # a plan that names a subject as ready and as failed: failed wins
    both = plan_of(["a", "b"], keys={"a": "k", "b": key_of("b")}, failed={"a": "render failure"})
    assert next_assignment(session, rater, both) == Next("done")
    assert subjects(all_assignments(session)) == ["b"]


def test_a_failed_render_in_the_catalog_is_never_offered(
    catalog_run: Path, uids: list[str], session: Session, rater: Rater
) -> None:
    select_subjects(catalog_run, uids[:3])
    record_render(catalog_run, uids[0], status="failed", reason="missing surface")
    record_render(catalog_run, uids[1])
    record_render(catalog_run, uids[2], status="failed", reason="wrong orientation")
    plan = planned(catalog_run)
    assert subjects(rate_all(session, rater, plan)) == [uids[1]]
    assert next_assignment(session, rater, plan).status == "done"
    assert not has_work_left(session, rater, plan)


def test_rendering_while_nothing_is_ready(session: Session, rater: Rater) -> None:
    waiting = plan_of(["a", "b", "c"], keys={}, pending=3)
    assert next_assignment(session, rater, waiting) == Next("rendering")
    assert has_work_left(session, rater, waiting)
    assert all_assignments(session) == []

    # the second subject of the queue is rendered first: it is what the rater gets
    one = plan_of(["a", "b", "c"], keys={"b": key_of("b")}, pending=2)
    first = next_assignment(session, rater, one)
    assert first.assignment is not None and first.assignment.subject_uid == "b"
    assert first.prefetch == ()
    complete(session, first.assignment)
    assert next_assignment(session, rater, one) == Next("rendering")
    # outdated renders are awaited like pending ones
    assert next_assignment(session, rater, plan_of(["a", "b"], keys={}, outdated=2)).status == (
        "rendering"
    )
    # nothing pending and nothing ready: done, also for a plan without subjects
    assert next_assignment(session, rater, plan_of(["b"])) == Next("done")
    assert next_assignment(session, rater, plan_of([])) == Next("done")
    assert len(all_assignments(session)) == 1


def test_an_outdated_render_is_not_ready(
    catalog_run: Path, uids: list[str], session: Session, rater: Rater
) -> None:
    first, second = uids[:2]
    select_subjects(catalog_run, [first, second])
    record_render(catalog_run, first, config_hash=OLD_HASH)
    record_render(catalog_run, second)
    plan = planned(catalog_run)
    assert plan.outdated == 1 and first not in plan.render_keys
    assert subjects(rate_all(session, rater, plan)) == [second]
    assert next_assignment(session, rater, plan) == Next("rendering")

    # rendered again under the present config: now it is offered
    record_render(catalog_run, first)
    plan = planned(catalog_run)
    assert plan.outdated == 0
    assert assigned(session, rater, plan).subject_uid == first


def test_a_subject_whose_render_key_changed_comes_back_and_a_re_encoded_one_does_not(
    catalog_run: Path, uids: list[str], session: Session, rater: Rater
) -> None:
    changed, re_encoded, untouched = uids[:3]
    select_subjects(catalog_run, uids[:3])
    for uid in uids[:3]:
        record_render(catalog_run, uid)
    plan = planned(catalog_run)
    assert subjects(rate_all(session, rater, plan)) == uids[:3]
    assert next_assignment(session, rater, plan) == Next("done")

    # one subject got other surfaces (another key); one was encoded again with a video,
    # under a config of another hash, which leaves its tiles and its key
    new_hash = "0123456789ab"
    record_render(catalog_run, changed, config_hash=new_hash, render_key="another-key")
    record_render(catalog_run, re_encoded, config_hash=new_hash, video=True)
    # while the third still has the earlier config, the queue waits for it
    waiting = planned(catalog_run, render_config_hash=new_hash)
    assert waiting.outdated == 1 and untouched not in waiting.render_keys
    record_render(catalog_run, untouched, config_hash=new_hash)

    plan = planned(catalog_run, render_config_hash=new_hash)
    assert plan.render_keys[re_encoded] == key_of(re_encoded) and plan.outdated == 0
    again = rate_all(session, rater, plan)
    assert subjects(again) == [changed]
    assert (again[0].render_key, again[0].kind, again[0].seq) == ("another-key", "main", 4)
    assert next_assignment(session, rater, plan) == Next("done")
    # the earlier rating stays, with the key of the tiles it was made on
    keys = [(a.subject_uid, a.render_key) for a in all_assignments(session)]
    assert keys.count((changed, key_of(changed))) == 1 and len(keys) == 4


def test_a_subject_is_done_only_by_a_completed_main_assignment_of_its_present_render(
    session: Session, rater: Rater
) -> None:
    assert rater.rater_id is not None
    rows = [
        ("a", "repeat", key_of("a"), T[1]),  # a repeat is no rating of a new subject
        ("b", "calibration", key_of("b"), T[2]),
        ("c", "main", "an-earlier-key", T[3]),
        ("d", "main", key_of("d"), T[4]),  # done
    ]
    for seq, (uid, kind, render_key, completed_at) in enumerate(rows, start=1):
        session.add(
            Assignment(
                rater_id=rater.rater_id,
                subject_uid=uid,
                kind=kind,
                seq=seq,
                render_key=render_key,
                assigned_at=T[0],
                completed_at=completed_at,
            )
        )
    session.flush()
    plan = plan_of(["a", "b", "c", "d"])
    assert subjects_left(session, rater, plan) == 3
    rated = rate_all(session, rater, plan)
    assert subjects(rated) == ["a", "b", "c"]
    assert [a.seq for a in rated] == [5, 6, 7]


def test_two_local_raters_each_get_every_subject(session: Session) -> None:
    first = get_or_create_rater(session, "tk", role="lead")
    second = get_or_create_rater(session, "second", role="lead")
    order = ["c", "a", "d", "b"]
    plan = plan_of(order)
    assert subjects(rate_all(session, first, plan)) == order
    assert not has_work_left(session, first, plan)
    # what the first rated is not done for the second, and their numbering is their own
    assert has_work_left(session, second, plan) and subjects_left(session, second, plan) == 4
    rated = rate_all(session, second, plan)
    assert subjects(rated) == order and [a.seq for a in rated] == [1, 2, 3, 4]
    assert {a.rater_id for a in rated} == {second.rater_id}
    assert len(all_assignments(session)) == 8


def test_two_raters_who_take_turns_do_not_take_each_others_subjects(session: Session) -> None:
    first = get_or_create_rater(session, "tk", role="lead")
    second = get_or_create_rater(session, "second", role="lead")
    plan = plan_of(["a", "b", "c"])
    mine = assigned(session, first, plan)
    theirs = assigned(session, second, plan)
    assert mine is not theirs and (mine.subject_uid, theirs.subject_uid) == ("a", "a")
    complete(session, theirs)
    assert assigned(session, first, plan) is mine
    assert assigned(session, second, plan).subject_uid == "b"


# --- next_assignment: an open assignment whose render is no longer the subject's ----------------


def test_an_open_assignment_of_a_changed_render_is_left_and_the_subject_is_offered_anew(
    session: Session, rater: Rater
) -> None:
    plan = plan_of(["a", "b"])
    stale = assigned(session, rater, plan)
    assert (stale.subject_uid, stale.render_key) == ("a", key_of("a"))

    changed = plan_of(["a", "b"], keys={"a": "new-key", "b": key_of("b")})
    assert has_work_left(session, rater, changed)
    fresh = assigned(session, rater, changed)
    assert fresh is not stale
    assert (fresh.subject_uid, fresh.render_key, fresh.seq, fresh.kind) == (
        "a",
        "new-key",
        2,
        "main",
    )
    assert assigned(session, rater, changed) is fresh  # idempotent on the new one
    # nothing of the earlier assignment was changed: it stays open, with its key
    assert (stale.render_key, stale.completed_at, stale.seq) == (key_of("a"), None, 1)

    complete(session, fresh)
    assert assigned(session, rater, changed).subject_uid == "b"
    assert len(all_assignments(session)) == 3


def test_an_open_assignment_of_a_failed_or_awaited_render_is_not_returned(
    session: Session, rater: Rater
) -> None:
    plan = plan_of(["a", "b"])
    parked = assigned(session, rater, plan)
    assert parked.subject_uid == "a"

    # only that subject, and its render failed meanwhile: nothing to do, nothing closed
    failed = plan_of(["a"], failed={"a": "source changed"})
    assert next_assignment(session, rater, failed) == Next("done")
    assert not has_work_left(session, rater, failed) and subjects_left(session, rater, failed) == 0
    # its render is being made again: the rater waits
    awaited = plan_of(["a"], keys={}, outdated=1)
    assert next_assignment(session, rater, awaited) == Next("rendering")
    # no longer selected
    assert next_assignment(session, rater, plan_of(["b"], keys={})) == Next("done")
    assert len(all_assignments(session)) == 1

    # with the same render back, the same assignment is the rater's again
    assert assigned(session, rater, plan) is parked
    assert len(all_assignments(session)) == 1


def test_of_two_open_assignments_the_later_is_shown_and_the_earlier_follows(
    session: Session, rater: Rater
) -> None:
    plan = plan_of(["a", "b", "c"])
    parked = assigned(session, rater, plan)
    without_a = plan_of(["a", "b", "c"], keys={"b": key_of("b"), "c": key_of("c")}, outdated=1)
    shown = assigned(session, rater, without_a)
    assert (shown.subject_uid, shown.seq) == ("b", 2)

    # a is rendered again with the same tiles while the page shows b
    following = next_assignment(session, rater, plan)
    assert following.assignment is shown
    assert following.prefetch == (("a", key_of("a")), ("c", key_of("c")))
    complete(session, shown)
    assert assigned(session, rater, plan) is parked
    complete(session, parked)
    last = assigned(session, rater, plan)
    assert (last.subject_uid, last.seq) == ("c", 3)
    assert len(all_assignments(session)) == 3


# --- has_work_left ------------------------------------------------------------------------------


def test_has_work_left_agrees_with_next_and_creates_nothing(session: Session, rater: Rater) -> None:
    plan = plan_of(["a", "b"])
    assert has_work_left(session, rater, plan)  # next would give one
    assert all_assignments(session) == []
    first = assigned(session, rater, plan)
    assert has_work_left(session, rater, plan)  # an open assignment
    complete(session, first)
    assert has_work_left(session, rater, plan)
    assert len(all_assignments(session)) == 1
    complete(session, assigned(session, rater, plan))
    assert not has_work_left(session, rater, plan)
    assert next_assignment(session, rater, plan) == Next("done")
    # everything ready is rated, but a render is awaited: the unblinded screens stay closed
    assert has_work_left(session, rater, plan_of(["a", "b", "c"], keys=plan.render_keys, pending=1))
    assert len(all_assignments(session)) == 2


# --- previous_assignment ------------------------------------------------------------------------


def test_the_previous_assignment_is_the_most_recent_completed_one(
    session: Session, rater: Rater
) -> None:
    other = get_or_create_rater(session, "second")
    plan = plan_of(["a", "b", "c"])
    assert previous_assignment(session, rater) is None
    first = assigned(session, rater, plan)
    assert previous_assignment(session, rater) is None  # an open assignment is not it

    complete(session, first, now=T[1])
    assert previous_assignment(session, rater) is first
    second = assigned(session, rater, plan)
    assert previous_assignment(session, rater) is first
    complete(session, second, now=T[2])
    assert previous_assignment(session, rater) is second
    # a revision is a second submission of the same assignment
    complete(session, second, now=T[5])
    assert previous_assignment(session, rater) is second
    assert second.completed_at == T[2]

    # another rater's work is not this rater's to revise
    assert previous_assignment(session, other) is None
    theirs = assigned(session, other, plan)
    complete(session, theirs, now=T[9])
    assert previous_assignment(session, other) is theirs
    assert previous_assignment(session, rater) is second


def test_the_previous_assignment_goes_by_when_it_was_rated_not_when_it_was_made(
    session: Session, rater: Rater
) -> None:
    assert rater.rater_id is not None

    def made(seq: int, uid: str, completed_at: str | None) -> Assignment:
        assignment = Assignment(
            rater_id=rater.rater_id,
            subject_uid=uid,
            seq=seq,
            render_key=key_of(uid),
            assigned_at=T[0],
            completed_at=completed_at,
        )
        session.add(assignment)
        session.flush()
        return assignment

    # the first was parked and rated after the second
    parked = made(1, "a", T[8])
    made(2, "b", T[4])
    assert previous_assignment(session, rater) is parked
    # two rated in the same millisecond: the one made later
    made(3, "c", T[9])
    tie = made(4, "d", T[9])
    made(5, "e", None)
    assert previous_assignment(session, rater) is tie


# --- repeats ------------------------------------------------------------------------------------


def test_no_repeat_without_a_repeat_rate(session: Session, rater: Rater) -> None:
    order = [f"s{n:02d}" for n in range(30)]
    rated = rate_all(session, rater, plan_of(order, repeat_rate=0.0, repeat_gap=0, seed=1))
    assert subjects(rated) == order and {a.kind for a in rated} == {"main"}


def test_with_a_rate_of_one_every_subject_is_repeated_once_after_the_gap(
    session: Session, rater: Rater
) -> None:
    plan = plan_of(["a", "b", "c", "d"], repeat_rate=1.0, repeat_gap=2, seed=11)
    shown: list[tuple[str, str]] = []
    while (following := next_assignment(session, rater, plan)).status == "assignment":
        assert following.assignment is not None
        # a repeat is not among what comes next, and the walk goes on behind it
        assert all(key == key_of(uid) for uid, key in following.prefetch)
        assert has_work_left(session, rater, plan)
        shown.append((following.assignment.subject_uid, following.assignment.kind))
        complete(session, following.assignment)
    # at assignment n, a subject can be repeated that was rated at n - 2 or before
    assert shown == [
        ("a", "main"),
        ("b", "main"),
        ("a", "repeat"),
        ("b", "repeat"),
        ("c", "main"),
        ("d", "main"),
        ("c", "repeat"),
        ("d", "repeat"),
    ]
    assert next_assignment(session, rater, plan) == Next("done")
    assert not has_work_left(session, rater, plan)
    repeats = [a for a in all_assignments(session) if a.kind == "repeat"]
    assert [a.render_key for a in repeats] == [key_of(a.subject_uid) for a in repeats]
    assert subjects_left(session, rater, plan) == 0


def test_a_repeat_is_drawn_with_the_generator_of_its_assignment(
    session: Session, rater: Rater
) -> None:
    assert rater.rater_id is not None
    done = ["e", "b", "d", "a", "c"]
    for seq, uid in enumerate(done, start=1):
        session.add(
            Assignment(
                rater_id=rater.rater_id,
                subject_uid=uid,
                seq=seq,
                render_key=key_of(uid),
                assigned_at=T[0],
                completed_at=T[seq],
            )
        )
    session.flush()
    for seed in range(6):
        plan = plan_of([*done, "f"], repeat_rate=1.0, repeat_gap=1, seed=seed)
        rng = random.Random(f"{seed}:{rater.rater_id}:6")
        rng.random()
        expected = rng.choice(sorted(done))
        following = next_assignment(session, rater, plan)
        assert following.assignment is not None
        assert (following.assignment.subject_uid, following.assignment.kind) == (expected, "repeat")
        assert following.assignment.seq == 6
        assert following.prefetch == (("f", key_of("f")),)
        session.delete(following.assignment)
        session.flush()
    # the seeds of the loop do not all draw the same subject
    assert len({random.Random(f"{s}:{rater.rater_id}:6").random() for s in range(6)}) == 6


def test_repeats_come_at_about_the_rate_and_respect_the_gap(session: Session, rater: Rater) -> None:
    rate, gap, seed = 0.25, 10, 5
    order = [f"s{n:03d}" for n in range(300)]
    plan = plan_of(order, repeat_rate=rate, repeat_gap=gap, seed=seed)
    rated = rate_all(session, rater, plan)
    mains = [a for a in rated if a.kind == "main"]
    repeats = [a for a in rated if a.kind == "repeat"]
    assert subjects(mains) == order and len(mains) + len(repeats) == len(rated)
    assert [a.seq for a in rated] == list(range(1, len(rated) + 1))

    # 300 / (1 - 0.25) assignments in all, a quarter of them repeats
    assert 70 <= len(repeats) <= 130
    main_seq = {a.subject_uid: a.seq for a in mains}
    assert len(set(subjects(repeats))) == len(repeats)  # nobody is repeated twice
    for repeat in repeats:
        assert repeat.seq - main_seq[repeat.subject_uid] >= gap
        assert repeat.render_key == key_of(repeat.subject_uid)
    assert min(a.seq for a in repeats) > gap

    # the kind of every assignment follows from the database: the draw of its number
    for assignment in rated:
        if assignment.seq <= gap:
            assert assignment.kind == "main"
            continue
        draw = random.Random(f"{seed}:{rater.rater_id}:{assignment.seq}").random()
        assert (assignment.kind == "repeat") == (draw < rate), assignment.seq


def test_a_repeat_shows_the_render_that_was_rated(session: Session, rater: Rater) -> None:
    always = {"repeat_rate": 1.0, "repeat_gap": 1, "seed": 2}
    plan = plan_of(["a"], **always)
    complete(session, assigned(session, rater, plan))

    # the subject's surfaces changed: it comes back as a new subject, not as a repeat
    changed = plan_of(["a"], keys={"a": "new-key"}, **always)
    back = assigned(session, rater, changed)
    assert (back.kind, back.render_key) == ("main", "new-key")
    complete(session, back)
    # then the new render is repeated, once
    repeat = assigned(session, rater, changed)
    assert (repeat.kind, repeat.render_key, repeat.seq) == ("repeat", "new-key", 3)
    complete(session, repeat)
    assert next_assignment(session, rater, changed) == Next("done")
    # a render that failed since is not repeated
    assert next_assignment(session, rater, plan_of(["a"], failed={"a": "x"}, **always)) == Next(
        "done"
    )


def test_an_open_repeat_is_returned_again(session: Session, rater: Rater) -> None:
    plan = plan_of(["a", "b"], repeat_rate=1.0, repeat_gap=1, seed=0)
    complete(session, assigned(session, rater, plan))
    repeat = assigned(session, rater, plan)
    assert (repeat.subject_uid, repeat.kind) == ("a", "repeat")
    again = next_assignment(session, rater, plan)
    assert again.assignment is repeat and again.prefetch == (("b", key_of("b")),)
    assert len(all_assignments(session)) == 2


# --- what the local branch does not do ----------------------------------------------------------


def test_the_team_branch_is_not_written(session: Session, rater: Rater) -> None:
    team = dataclasses.replace(plan_of(["a"]), mode="team")
    for function in (next_assignment, has_work_left, subjects_left):
        with pytest.raises(NotImplementedError, match="milestone 5"):
            function(session, rater, team)
    assert all_assignments(session) == []


def test_a_rater_without_an_id_is_refused(session: Session) -> None:
    unsaved = Rater(name="nobody", created_at=T[0])
    plan = plan_of(["a"])
    with pytest.raises(ValueError, match="no id"):
        next_assignment(session, unsaved, plan)
    with pytest.raises(ValueError, match="no id"):
        has_work_left(session, unsaved, plan)
    with pytest.raises(ValueError, match="no id"):
        previous_assignment(session, unsaved)


def test_subjects_left_counts_what_is_ready_and_what_is_awaited(
    session: Session, rater: Rater
) -> None:
    plan = plan_of(
        ["a", "b", "c", "d", "e", "f"],
        keys={"a": key_of("a"), "b": key_of("b"), "c": key_of("c")},
        failed={"d": "missing surface"},
        pending=1,
        outdated=1,
    )
    assert subjects_left(session, rater, plan) == 5
    complete(session, assigned(session, rater, plan))
    assert subjects_left(session, rater, plan) == 4
    assigned(session, rater, plan)  # open: still to rate
    assert subjects_left(session, rater, plan) == 4
    assert len(all_assignments(session)) == 2
