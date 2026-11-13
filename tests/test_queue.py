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
from surfcensus.app.models import Assignment, Event, Rater
from surfcensus.app.queue import (
    PREFETCH,
    Next,
    QueuePlan,
    SkipRefused,
    build_plan,
    has_work_left,
    is_open_for,
    next_assignment,
    previous_assignment,
    skip_assignment,
    subjects_left,
    subjects_skipped,
)
from surfcensus.app.schemas import AppMode, RatingIn, SkipIn
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


def skip_in(assignment: Assignment, client_ts: str, ms: int = 2500) -> SkipIn:
    assert assignment.assignment_id is not None
    return SkipIn(assignment_id=assignment.assignment_id, client_ts=client_ts, ms_on_subject=ms)


class Skipper:
    """Skips as a page does: each skip with a client time of its own, a second later."""

    def __init__(self, session: Session, rater: Rater, plan: QueuePlan) -> None:
        self.session, self.rater, self.plan = session, rater, plan
        self.n = 0

    def __call__(self, assignment: Assignment, *, plan: QueuePlan | None = None) -> Next:
        self.n += 1
        return skip_assignment(
            self.session,
            self.rater,
            assignment,
            skip_in(assignment, f"client-{self.n:03d}"),
            self.plan if plan is None else plan,
            now=T[self.n],
        )

    def to(self, assignment: Assignment, **kw: Any) -> Assignment:
        """Skip, and the assignment that follows."""
        following = self(assignment, **kw)
        assert following.status == "assignment" and following.assignment is not None
        return following.assignment


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


def test_a_done_row_without_a_render_key_is_not_ready(catalog_run: Path, uids: list[str]) -> None:
    select_subjects(catalog_run, uids[:2])
    record_render(catalog_run, uids[0])
    keyless = RenderRow(
        subject_uid=uids[1], config_hash=HASH, input_signature="0" * 16, video=False, status="done"
    )
    with PublicCatalog.open(catalog_run, readonly=False) as public:
        public.record_renders([keyless])
    plan = planned(catalog_run)
    # no key, so no tiles that a rating could be tied to: to be rendered again
    assert plan.render_keys == {uids[0]: key_of(uids[0])} and not plan.failed
    assert (plan.pending, plan.outdated) == (0, 1)


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
    # also for a plan that still names the render it had
    both = plan_of(["a"], keys={"a": key_of("a")}, failed={"a": "source changed"})
    assert next_assignment(session, rater, both) == Next("done")
    assert not has_work_left(session, rater, both)
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
        assert following.assignment is not None and len(shown) < 20
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


def test_an_open_repeat_is_returned_again_and_is_work_left(session: Session, rater: Rater) -> None:
    # a seed whose draw makes the second assignment a repeat and the third none
    def draw(seed: int, seq: int) -> float:
        return random.Random(f"{seed}:{rater.rater_id}:{seq}").random()

    seed = next(seed for seed in range(100) if draw(seed, 2) < 0.5 <= draw(seed, 3))
    plan = plan_of(["a"], repeat_rate=0.5, repeat_gap=1, seed=seed)
    complete(session, assigned(session, rater, plan))
    assert has_work_left(session, rater, plan)
    repeat = assigned(session, rater, plan)
    assert (repeat.subject_uid, repeat.kind, repeat.seq) == ("a", "repeat", 2)
    for _ in range(2):
        again = next_assignment(session, rater, plan)
        assert again.assignment is repeat and again.prefetch == ()
        # nothing would follow it, and it is still to be rated
        assert has_work_left(session, rater, plan)
    assert len(all_assignments(session)) == 2
    complete(session, repeat)
    assert next_assignment(session, rater, plan) == Next("done")
    assert not has_work_left(session, rater, plan)


# --- what the local branch does not do ----------------------------------------------------------


def test_the_team_branch_is_not_written(session: Session, rater: Rater) -> None:
    team = dataclasses.replace(plan_of(["a"]), mode="team")
    for function in (next_assignment, has_work_left, subjects_left, subjects_skipped):
        with pytest.raises(NotImplementedError, match="milestone 5"):
            function(session, rater, team)
    assert all_assignments(session) == []
    mine = assigned(session, rater, plan_of(["a"]))
    with pytest.raises(NotImplementedError, match="milestone 5"):
        skip_assignment(session, rater, mine, skip_in(mine, "t"), team)
    assert (mine.skipped_at, mine.skip_count) == (None, 0)


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


# --- skips: a subject is deferred, never dropped ------------------------------------------------


def skip_events(session: Session) -> list[tuple[int | None, str]]:
    rows = session.exec(select(Event).where(Event.type == "skip").order_by(col(Event.event_id)))
    return [(event.assignment_id, event.ts) for event in rows]


def test_a_skipped_subject_comes_back_after_every_subject_not_yet_shown(
    session: Session, rater: Rater
) -> None:
    plan = plan_of(["a", "b", "c", "d", "e"])
    skip = Skipper(session, rater, plan)
    a = assigned(session, rater, plan)
    b = skip.to(a)
    assert (a.subject_uid, b.subject_uid, b.seq) == ("a", "b", 2)
    # a skip stores nothing of a rating and leaves the assignment open
    assert a.completed_at is None and (a.skipped_at, a.skip_count) == (T[1], 1)
    complete(session, b)
    c = assigned(session, rater, plan)
    d = skip.to(c)
    assert (c.subject_uid, d.subject_uid) == ("c", "d")
    complete(session, d)
    e = assigned(session, rater, plan)
    assert e.subject_uid == "e"
    complete(session, e)
    # everything was shown: the skipped ones come back in the order of skipping, as the
    # assignments they were
    assert assigned(session, rater, plan) is a
    complete(session, a)
    assert assigned(session, rater, plan) is c
    complete(session, c)
    assert next_assignment(session, rater, plan) == Next("done")
    assert subjects(all_assignments(session)) == ["a", "b", "c", "d", "e"]
    assert [x.seq for x in all_assignments(session)] == [1, 2, 3, 4, 5]
    assert skip_events(session) == [
        (a.assignment_id, "client-001"),
        (c.assignment_id, "client-002"),
    ]


def test_the_order_of_skipping_decides_and_not_the_order_of_the_queue(
    session: Session, rater: Rater
) -> None:
    plan = plan_of(["a", "b", "c"])
    skip = Skipper(session, rater, plan)
    a = assigned(session, rater, plan)
    b = skip.to(a)
    complete(session, b)
    c = assigned(session, rater, plan)
    # c is skipped, a comes back and is skipped again: now a was skipped last
    assert skip.to(c) is a
    assert skip.to(a) is c
    assert (a.skip_count, c.skip_count) == (2, 1) and a.skipped_at > c.skipped_at
    complete(session, c)
    assert assigned(session, rater, plan) is a


def test_when_only_skipped_subjects_are_left_skip_goes_round_them(
    session: Session, rater: Rater
) -> None:
    plan = plan_of(["a", "b", "c"])
    skip = Skipper(session, rater, plan)
    first = assigned(session, rater, plan)
    shown = [first]
    for _ in range(8):
        shown.append(skip.to(shown[-1]))
    assert subjects(shown) == ["a", "b", "c", "a", "b", "c", "a", "b", "c"]
    assert len(all_assignments(session)) == 3  # one assignment per subject, however often
    assert [x.skip_count for x in all_assignments(session)] == [3, 3, 2]
    assert len(skip_events(session)) == 8
    # the one on the page is returned again until it is rated or skipped
    assert assigned(session, rater, plan) is shown[-1]
    assert assigned(session, rater, plan) is shown[-1]
    assert has_work_left(session, rater, plan)


def test_the_only_subject_comes_straight_back(session: Session, rater: Rater) -> None:
    plan = plan_of(["a"])
    skip = Skipper(session, rater, plan)
    only = assigned(session, rater, plan)
    for count in (1, 2, 3):
        following = skip(only)
        assert following == Next("assignment", only, ())
        assert only.skip_count == count
    assert has_work_left(session, rater, plan) and subjects_left(session, rater, plan) == 1
    assert subjects_skipped(session, rater, plan, only) == 0


def test_the_queue_is_done_only_when_every_subject_has_a_submission(
    session: Session, rater: Rater
) -> None:
    plan = plan_of(["a", "b", "c"])
    skip = Skipper(session, rater, plan)
    a = assigned(session, rater, plan)
    b = skip.to(a)
    c = skip.to(b)
    complete(session, c)
    # two of three have no rating: nothing is done, nothing is dropped
    for _ in range(3):
        following = next_assignment(session, rater, plan)
        assert following.status == "assignment" and following.assignment is a
        assert has_work_left(session, rater, plan)
        assert subjects_left(session, rater, plan) == 2
    complete(session, a)
    assert has_work_left(session, rater, plan) and subjects_left(session, rater, plan) == 1
    assert assigned(session, rater, plan) is b
    complete(session, b)
    assert next_assignment(session, rater, plan) == Next("done")
    assert not has_work_left(session, rater, plan) and subjects_left(session, rater, plan) == 0
    # an unratable submission ends a subject as well (the queue asks for a submission)
    assert all(x.completed_at is not None for x in all_assignments(session))


def test_a_skip_is_no_rating_and_gives_the_subject_to_nobody_else(session: Session) -> None:
    first = get_or_create_rater(session, "first", role="lead")
    second = get_or_create_rater(session, "second")
    plan = plan_of(["a", "b"])
    mine = assigned(session, first, plan)
    following = skip_assignment(session, first, mine, skip_in(mine, "c1"), plan, now=T[1])
    assert following.assignment is not None and following.assignment.subject_uid == "b"
    complete(session, following.assignment)
    # the other rater walks the same order and is told nothing of the skip
    theirs = assigned(session, second, plan)
    assert theirs.subject_uid == "a" and theirs is not mine
    assert subjects_skipped(session, second, plan) == 0
    complete(session, theirs)
    complete(session, assigned(session, second, plan))
    assert not has_work_left(session, second, plan)
    # and their rating of it does not take it from the first, who still has to rate it
    assert has_work_left(session, first, plan)
    assert assigned(session, first, plan) is mine
    assert [(x.rater_id, x.subject_uid) for x in all_assignments(session)] == [
        (first.rater_id, "a"),
        (first.rater_id, "b"),
        (second.rater_id, "a"),
        (second.rater_id, "b"),
    ]


def test_backspace_after_a_skip_goes_back_to_the_skipped_subject(
    session: Session, rater: Rater
) -> None:
    plan = plan_of(["a", "b", "c", "d"])
    a = assigned(session, rater, plan)
    complete(session, a, now=T[1])
    b = assigned(session, rater, plan)
    assert previous_assignment(session, rater) is a
    c = skip_assignment(session, rater, b, skip_in(b, "c1"), plan, now=T[2]).assignment
    assert c is not None and c.subject_uid == "c"
    # the assignment the rater left last is the skipped one, open as it was
    assert previous_assignment(session, rater) is b
    assert b.completed_at is None and (b.skipped_at, b.skip_count) == (T[2], 1)
    # looking back changes nothing: the page of the server is still c, and b still waits
    assert assigned(session, rater, plan) is c
    assert subjects_skipped(session, rater, plan, c) == 1
    # rated from there, it is done and the skip stays recorded
    complete(session, b, now=T[3])
    assert (b.completed_at, b.skipped_at, b.skip_count) == (T[3], T[2], 1)
    assert previous_assignment(session, rater) is b
    assert assigned(session, rater, plan) is c
    complete(session, c, now=T[4])
    assert previous_assignment(session, rater) is c
    d = assigned(session, rater, plan)
    complete(session, d, now=T[5])
    assert next_assignment(session, rater, plan) == Next("done")
    assert len(all_assignments(session)) == 4


def test_the_previous_assignment_is_the_one_left_last_rated_or_skipped(
    session: Session, rater: Rater
) -> None:
    assert rater.rater_id is not None

    def made(seq: int, uid: str, completed_at: str | None, skipped_at: str | None) -> Assignment:
        assignment = Assignment(
            rater_id=rater.rater_id,
            subject_uid=uid,
            seq=seq,
            render_key=key_of(uid),
            assigned_at=T[0],
            completed_at=completed_at,
            skipped_at=skipped_at,
            skip_count=0 if skipped_at is None else 1,
        )
        session.add(assignment)
        session.flush()
        return assignment

    made(1, "a", None, None)  # never left
    assert previous_assignment(session, rater) is None
    rated = made(2, "b", T[4], None)
    skipped = made(3, "c", None, T[3])
    assert previous_assignment(session, rater) is rated
    later = made(4, "d", None, T[6])
    assert previous_assignment(session, rater) is later
    # skipped first and rated afterwards: left when it was rated
    both = made(5, "e", T[8], T[2])
    assert previous_assignment(session, rater) is both
    # two left in the same millisecond: the one made later
    tie = made(6, "f", None, T[8])
    assert previous_assignment(session, rater) is tie
    assert skipped.completed_at is None
    # another rater's are not this rater's to go back to
    assert previous_assignment(session, get_or_create_rater(session, "second")) is None


def test_a_skip_request_that_arrives_twice_is_one_skip(session: Session, rater: Rater) -> None:
    plan = plan_of(["a", "b", "c"])
    a = assigned(session, rater, plan)
    request = skip_in(a, "2026-11-13T09:59:59.250Z")
    first = skip_assignment(session, rater, a, request, plan, now=T[1])
    assert first.assignment is not None and first.assignment.subject_uid == "b"
    # the answer was lost and the page sends the same request again, twice
    for now in (T[2], T[3]):
        again = skip_assignment(session, rater, a, request, plan, now=now)
        assert again == first and again.assignment is first.assignment
    assert (a.skipped_at, a.skip_count) == (T[1], 1)
    assert skip_events(session) == [(a.assignment_id, "2026-11-13T09:59:59.250Z")]
    assert len(all_assignments(session)) == 2
    assert subjects_skipped(session, rater, plan, first.assignment) == 1


def test_a_skip_of_an_assignment_that_is_not_open_is_refused(session: Session) -> None:
    rater = get_or_create_rater(session, "first", role="lead")
    other = get_or_create_rater(session, "second")
    plan = plan_of(["a", "b", "c"])
    rated = assigned(session, rater, plan)
    complete(session, rated, now=T[1])
    mine = assigned(session, rater, plan)
    theirs = assigned(session, other, plan)
    changed = plan_of(["a", "b", "c"], keys={"a": key_of("a"), "b": "new-key", "c": key_of("c")})
    failed = plan_of(["a", "b", "c"], failed={"b": "source changed"})
    cases = [
        (rater, rated, plan, "has a rating"),
        (rater, theirs, plan, "another rater's"),
        (other, mine, plan, "another rater's"),
        (rater, mine, changed, "its render changed"),
        (rater, mine, failed, "its render failed"),
    ]
    for who, assignment, state, why in cases:
        assert not is_open_for(who, assignment, state), why
        with pytest.raises(SkipRefused, match="is not open for the rater"):
            skip_assignment(session, who, assignment, skip_in(assignment, why), state, now=T[5])
    assert issubclass(SkipRefused, ValueError)
    # a skip that names another assignment is the caller's mistake, not a refusal
    with pytest.raises(ValueError, match="not of") as caught:
        skip_assignment(session, rater, mine, skip_in(theirs, "x"), plan, now=T[5])
    assert not isinstance(caught.value, SkipRefused)
    # nothing was recorded, and nothing moved
    assert skip_events(session) == []
    assert [(x.skipped_at, x.skip_count) for x in all_assignments(session)] == [(None, 0)] * 3
    assert is_open_for(rater, mine, plan) and is_open_for(other, theirs, plan)
    assert assigned(session, rater, plan) is mine and len(all_assignments(session)) == 3


def test_a_skipped_subject_waits_for_a_render_that_is_awaited(
    session: Session, rater: Rater
) -> None:
    waiting = plan_of(["a", "b"], keys={"a": key_of("a")}, pending=1)
    a = assigned(session, rater, waiting)
    # b was not shown yet, since it is still being rendered: the rater waits for it
    following = skip_assignment(session, rater, a, skip_in(a, "c1"), waiting, now=T[1])
    assert following == Next("rendering")
    assert next_assignment(session, rater, waiting) == Next("rendering")
    assert has_work_left(session, rater, waiting)
    assert subjects_left(session, rater, waiting) == 2
    assert subjects_skipped(session, rater, waiting) == 1
    # rendered: b comes first, then the skipped one
    ready = plan_of(["a", "b"])
    b = assigned(session, rater, ready)
    assert b.subject_uid == "b" and next_assignment(session, rater, ready).prefetch == (
        ("a", key_of("a")),
    )
    complete(session, b)
    assert assigned(session, rater, ready) is a
    assert len(all_assignments(session)) == 2


def test_a_skipped_assignment_whose_render_changed_is_left_and_the_subject_is_new_again(
    session: Session, rater: Rater
) -> None:
    plan = plan_of(["a", "b", "c"])
    skip = Skipper(session, rater, plan)
    a = assigned(session, rater, plan)
    b = skip.to(a)
    changed = plan_of(["a", "b", "c"], keys={"a": "new-key", "b": key_of("b"), "c": key_of("c")})
    assert subjects_skipped(session, rater, changed, b) == 0  # its tiles are gone
    complete(session, b)
    # a has other tiles now: it is a subject the rater has not been shown, in its place
    fresh = assigned(session, rater, changed)
    assert fresh is not a and (fresh.subject_uid, fresh.render_key, fresh.seq) == (
        "a",
        "new-key",
        3,
    )
    complete(session, fresh)
    complete(session, assigned(session, rater, changed))
    assert next_assignment(session, rater, changed) == Next("done")
    assert (a.completed_at, a.skip_count) == (None, 1)
    # with the earlier render back, the skipped assignment is the rater's again
    assert assigned(session, rater, plan) is a


def test_what_comes_next_is_what_was_not_shown_and_then_what_was_skipped(
    session: Session, rater: Rater
) -> None:
    plan = plan_of(["a", "b", "c", "d"])
    skip = Skipper(session, rater, plan)
    a = assigned(session, rater, plan)
    after_a = skip(a)
    assert after_a.assignment is not None and after_a.assignment.subject_uid == "b"
    assert after_a.prefetch == (("c", key_of("c")), ("d", key_of("d")))
    after_b = skip(after_a.assignment)
    assert after_b.assignment is not None and after_b.assignment.subject_uid == "c"
    # one subject not shown, then the one skipped first
    assert after_b.prefetch == (("d", key_of("d")), ("a", key_of("a")))
    complete(session, after_b.assignment)
    last = next_assignment(session, rater, plan)
    assert last.assignment is not None and last.assignment.subject_uid == "d"
    assert last.prefetch == (("a", key_of("a")), ("b", key_of("b")))
    complete(session, last.assignment)
    back = next_assignment(session, rater, plan)
    assert back.assignment is a and back.prefetch == (("b", key_of("b")),)
    assert len(back.prefetch) < PREFETCH


def test_the_progress_line_counts_the_skipped_subjects_that_wait(
    session: Session, rater: Rater
) -> None:
    plan = plan_of(["a", "b", "c"])
    skip = Skipper(session, rater, plan)
    a = assigned(session, rater, plan)
    assert subjects_skipped(session, rater, plan, a) == 0
    b = skip.to(a)
    assert subjects_skipped(session, rater, plan, b) == 1
    c = skip.to(b)
    assert subjects_skipped(session, rater, plan, c) == 2
    # a skipped subject is still to rate
    assert subjects_left(session, rater, plan) == 3
    complete(session, c)
    assert assigned(session, rater, plan) is a
    # the one that is given back is on the page: it does not wait any more
    assert subjects_skipped(session, rater, plan, a) == 1
    assert subjects_skipped(session, rater, plan) == 2  # nothing on the page
    # shown again by Backspace while another is the open one
    assert subjects_skipped(session, rater, plan, b) == 1
    complete(session, a)
    complete(session, b)
    assert subjects_skipped(session, rater, plan) == 0
    assert all_assignments(session)[0].skip_count == 1  # the record stays
    assert len(all_assignments(session)) == 3  # and counting made no assignment


def test_two_skips_at_one_moment_come_back_in_the_order_they_were_made(
    session: Session, rater: Rater
) -> None:
    plan = plan_of(["a", "b", "c"])
    a = assigned(session, rater, plan)
    b = skip_assignment(session, rater, a, skip_in(a, "c1"), plan, now=T[1]).assignment
    assert b is not None
    c = skip_assignment(session, rater, b, skip_in(b, "c2"), plan, now=T[1]).assignment
    assert c is not None and c.subject_uid == "c"
    complete(session, c)
    assert assigned(session, rater, plan) is a


def test_a_skipped_repeat_waits_like_any_subject_and_is_not_drawn_twice(
    session: Session, rater: Rater
) -> None:
    plan = plan_of(["a", "b"], repeat_rate=1.0, repeat_gap=1, seed=3)
    skip = Skipper(session, rater, plan)
    complete(session, assigned(session, rater, plan))
    repeat = assigned(session, rater, plan)
    assert (repeat.subject_uid, repeat.kind, repeat.seq) == ("a", "repeat", 2)
    # skipped: no second repeat of a is made while this one waits
    b = skip.to(repeat)
    assert (b.subject_uid, b.kind, b.seq) == ("b", "main", 3)
    complete(session, b)
    again = assigned(session, rater, plan)
    assert (again.subject_uid, again.kind, again.seq) == ("b", "repeat", 4)
    complete(session, again)
    assert assigned(session, rater, plan) is repeat
    assert has_work_left(session, rater, plan)
    complete(session, repeat)
    assert next_assignment(session, rater, plan) == Next("done")
    assert [(x.subject_uid, x.kind) for x in all_assignments(session)] == [
        ("a", "main"),
        ("a", "repeat"),
        ("b", "main"),
        ("b", "repeat"),
    ]
