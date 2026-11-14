"""What the modules of milestone 4 share: the tables of ``app.sqlite``, the names that the
packages of the milestone import from each other and their signatures, and the helpers of
earlier modules that the app reads a run through.

The names were fixed before the functions were written, so that the packages could be
written side by side, and stay fixed now that they are.
"""

from __future__ import annotations

import dataclasses
import inspect
import pickle
import re
import sqlite3
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlmodel")

from browser_launch import BROWSER_ENV, launch_options  # pyrefly: ignore[missing-import]
from sqlmodel import Session, SQLModel, create_engine, select

from surfcensus.app import APP_DB_FILE, APP_TABLES, RATINGS_FLAT_VIEW, db, models
from surfcensus.app.marks import MARK_FIELDS, MarkError
from surfcensus.config import load_config
from surfcensus.protocol import Protocol, load_protocol
from surfcensus.records import SUBJECT_UID_PATTERN
from surfcensus.render.model import RENDER_JSON, RenderInfo, read_render
from surfcensus.run import Run
from surfcensus.store import (
    PublicCatalog,
    clusters_with_native,
    media_dir,
    primary_scores,
)

INTERFACE: dict[str, dict[str, tuple[str, ...] | None]] = {
    # module -> name -> parameter names of a function, or None for a class or a constant
    "surfcensus.protocol": {
        "Protocol": None,
        "ScaleLevel": None,
        "Criterion": None,
        "CRITERION_IDS": None,
        "MARK_CLASSES": None,
        "RENDER_REASONS": None,
        "FIXED_KEYS": None,
        "install_protocol": ("run_dir",),
        "load_protocol": ("run_dir",),
        "parse_protocol": ("data",),
        "marks_required": ("protocol", "app"),
        "render_reason": ("reason",),
    },
    "surfcensus.calibration": {
        "CalibrationMark": None,
        "CalibrationExample": None,
        "CalibrationSubject": None,
        "CalibrationSet": None,
        "load_calibration": ("run_dir",),
        "write_calibration": ("run_dir", "calibration"),
        "validate_calibration": ("calibration", "protocol", "subjects", "render"),
    },
    "surfcensus.app.schemas": {
        "RenderOut": None,
        "RENDER_PRIVATE_FIELDS": None,
        "ROUTES": None,
        "Route": None,
        "ReviewQuery": None,
        "MEDIA_FILE": None,
        "EVENT_TYPES": None,
        "SKIP_EVENT": None,
        "SERVER_EVENT_TYPES": None,
        "SkipIn": None,
        "render_out": ("info",),
        "media_base": ("subject_uid", "base_url"),
        "tile_name": ("pattern", "panel", "index"),
    },
    "surfcensus.store": {
        "primary_scores": ("run_dir",),
        "clusters_with_native": ("run_dir",),
        "queue_order": ("uids", "scores", "blind", "seed"),
        "media_dir": ("run_dir", "subject_uid"),
    },
    "surfcensus.render.model": {
        "RENDER_JSON": None,
        "read_render": ("out_dir",),
    },
    "surfcensus.app.marks": {
        "MarkError": None,
        "VertexIndex": None,
        "validate_mark": ("info", "mark"),
        "check_snap_radius": ("max_dist_mm", "vertex_map_margin_mm"),
        "resolve_mark": ("info", "index", "mark", "max_dist_mm"),
        "marked_vertices": ("index", "points", "tolerance_mm", "surface"),
    },
    "surfcensus.app.db": {
        "APP_SCHEMA_VERSION": None,
        "MIGRATIONS": None,
        "RATINGS_FLAT_SQL": None,
        "app_db_path": ("run_dir",),
        "utc_now": (),
        "open_app_db": ("run_dir", "run_id", "mode"),
        "app_db_mode": ("run_dir",),
        "get_or_create_rater": ("session", "name", "role"),
        "store_submission": (
            "session",
            "assignment",
            "rating",
            "marks",
            "protocol_sha256",
            "app_version",
            "calibration_sha256",
            "now",
        ),
        "record_skip": ("session", "assignment", "skip", "now"),
    },
    "surfcensus.app.queue": {
        "QueuePlan": None,
        "Next": None,
        "SkipRefused": None,
        "build_plan": (
            "public",
            "scores",
            "app",
            "mode",
            "render_config_hash",
            "calibration",
        ),
        "next_assignment": ("session", "rater", "plan"),
        "previous_assignment": ("session", "rater"),
        "has_work_left": ("session", "rater", "plan"),
        "is_open_for": ("rater", "assignment", "plan"),
        "skip_assignment": ("session", "rater", "assignment", "skip", "plan", "now"),
        "subjects_left": ("session", "rater", "plan"),
        "subjects_skipped": ("session", "rater", "plan", "shown"),
    },
    "surfcensus.app.finalize": {
        "FinalizeReport": None,
        "sync_unratable": ("session", "plan", "protocol"),
        "record_single": ("session", "subject_uid", "protocol"),
        "finalize": ("session", "plan", "protocol"),
        "freeze_calibration": ("run_dir", "subjects", "version"),
    },
    "surfcensus.app.events": {
        "store_events": ("session", "assignment", "events"),
        "record_run_event": ("session", "type", "payload", "now"),
        "record_setting": ("session", "setting", "value", "now"),
    },
    "surfcensus.app.server": {
        "ServerHandle": None,
        "DEFAULT_PORT": None,
        "LOCAL_HOSTS": None,
        "STATIC_DIR": None,
        "STATIC_URL": None,
        "TEMPLATES_DIR": None,
        "PAGES": None,
        "create_app": ("run_dir", "mode", "rater", "allowed_hosts"),
        "start_server": ("run_dir", "host", "port", "mode", "rater"),
        "serve": ("run_dir", "host", "port", "mode", "rater", "open_browser", "stop"),
    },
    "surfcensus.export.tables": {
        "EXPORT_TABLES": None,
        "EXPORT_DIRS": None,
        "MANIFEST": None,
        "RATINGS_COLUMNS": None,
        "SUBJECT_RATING_COLUMNS": None,
        "BEHAVIOUR_COLUMNS": None,
        "SKIPS_COLUMNS": None,
        "MARKS_COLUMNS": None,
        "FINAL_MARKS_COLUMNS": None,
        "DATASETS_COLUMNS": None,
        "ratings_frame": ("run_dir",),
        "marks_frame": ("run_dir",),
        "subjects_frame": ("run_dir",),
        "frame_csv": ("frame",),
        "export_run": ("run_dir", "out_dir", "fmt", "profile"),
    },
    "surfcensus.review": {
        "DEFAULT_RUN_DIR": None,
        "ReviewApp": None,
        "default_run_dir": ("path",),
        "rateable": ("run",),
        "review": (
            "path",
            "run_dir",
            "subjects",
            "rater",
            "blind",
            "require_marks",
            "field_strength",
            "jobs",
            "fsaverage",
            "video",
            "port",
            "open_browser",
            "serve",
            "serve_after",
            "relocate",
            "force",
        ),
        "open_review": (
            "path",
            "run_dir",
            "subjects",
            "rater",
            "blind",
            "require_marks",
            "field_strength",
            "jobs",
            "fsaverage",
            "video",
            "port",
            "serve",
            "serve_after",
            "relocate",
            "force",
        ),
        "serve_review": ("run", "app", "rater", "open_browser", "stop"),
    },
}

COLUMNS: dict[str, tuple[str, ...]] = {
    # docs/rater_app.md, "Database tables"
    "raters": ("rater_id", "name", "token_hash", "role", "active", "created_at"),
    "assignments": (
        "assignment_id",
        "rater_id",
        "subject_uid",
        "kind",
        "seq",
        "render_key",
        "assigned_at",
        "completed_at",
        "consensus_served_at",
        "skipped_at",
        "skip_count",
    ),
    "submissions": (
        "submission_id",
        "assignment_id",
        "revision",
        "current",
        "unratable_reason",
        "other",
        "note",
        "protocol_version",
        "protocol_sha256",
        "calibration_sha256",
        "app_version",
        "client_ts",
        "server_ts",
        "ms_on_subject",
    ),
    "ratings": ("rating_id", "submission_id", "assignment_id", "criterion", "value"),
    "marks": (
        "mark_id",
        "submission_id",
        "assignment_id",
        "class",
        "panel",
        "slice",
        "frame",
        "mode",
        "px",
        "py",
        "vox_i",
        "vox_j",
        "vox_k",
        "tkr_x",
        "tkr_y",
        "tkr_z",
        "ras_x",
        "ras_y",
        "ras_z",
        "hemi",
        "pial_vertex",
        "pial_dist_mm",
        "white_vertex",
        "white_dist_mm",
        "fsavg6_vertex",
        "vertex_surface",
        "box_dist_mm",
        "created_ts",
        "deleted_ts",
    ),
    "events": ("event_id", "assignment_id", "ts", "type", "payload_json"),
    "final_labels": ("subject_uid", "criterion", "value", "method"),
    "final_marks": (
        "final_mark_id",
        "subject_uid",
        "class",
        "tkr_x",
        "tkr_y",
        "tkr_z",
        "hemi",
        "pial_vertex",
        "fsavg6_vertex",
        "source_mark_ids",
    ),
    "app_meta": ("key", "value"),
}


# --- names and signatures -------------------------------------------------------------------


@pytest.mark.parametrize("module", sorted(INTERFACE))
def test_the_shared_names_exist_with_their_parameters(module: str) -> None:
    loaded = import_module(module)
    for name, parameters in INTERFACE[module].items():
        assert hasattr(loaded, name), f"{module}.{name}"
        if parameters is not None:
            found = tuple(inspect.signature(getattr(loaded, name)).parameters)
            assert found == parameters, f"{module}.{name}"


def test_the_records_of_the_queue_and_of_finalize_keep_their_fields() -> None:
    from surfcensus.app.finalize import FinalizeReport
    from surfcensus.app.queue import Next, QueuePlan
    from surfcensus.app.server import ServerHandle

    def fields(cls: type) -> tuple[str, ...]:
        return tuple(field.name for field in dataclasses.fields(cls))

    assert fields(QueuePlan) == (
        "mode",
        "order",
        "render_keys",
        "failed",
        "pending",
        "outdated",
        "calibration",
        "raters_per_subject",
        "repeat_rate",
        "repeat_gap",
        "seed",
    )
    assert fields(Next) == ("status", "assignment", "prefetch")
    assert fields(FinalizeReport) == ("consensus", "single", "unratable", "open")
    assert fields(ServerHandle)[:2] == ("url", "port")
    plan = QueuePlan(mode="local", order=("a",), render_keys={}, failed={}, pending=1, outdated=0)
    assert (plan.calibration, plan.raters_per_subject, plan.repeat_rate) == ((), 1, 0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.pending = 2  # pyrefly: ignore[read-only]
    assert Next("done") == Next(status="done", assignment=None, prefetch=())


def test_a_mark_error_names_its_field_and_survives_pickling() -> None:
    assert MARK_FIELDS == ("panel", "slice", "px", "py", "frame")
    error = MarkError("slice", "the sagittal panel has 7 slices, got 9")
    assert isinstance(error, ValueError)
    assert (error.field, error.message) == ("slice", "the sagittal panel has 7 slices, got 9")
    assert str(error) == "slice: the sagittal panel has 7 slices, got 9"
    copy = pickle.loads(pickle.dumps(error))
    assert type(copy) is MarkError and (copy.field, copy.message) == (error.field, error.message)


def test_the_time_of_the_tables_is_utc_in_milliseconds() -> None:
    first, second = db.utc_now(), db.utc_now()
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", first)
    assert first <= second  # text order is time order
    assert (
        db.app_db_path("/data/run")
        == Path("/data/run") / APP_DB_FILE
        == Path("/data/run/app.sqlite")
    )
    assert db.APP_SCHEMA_VERSION == 1 and db.MIGRATIONS == ()


def test_a_subject_uid_is_sixteen_hex_characters(small_run: Path) -> None:
    with PublicCatalog.open(small_run) as public:
        uids = public.subject_uids()
        keys = [key for key in public.renders()["render_key"]]
    assert len(uids) == 3
    for text in (*uids, *keys):
        assert re.fullmatch(SUBJECT_UID_PATTERN, text)
    for text in ("sub-001", uids[0].upper(), uids[0] + "0", uids[0][:-1]):
        assert not re.fullmatch(SUBJECT_UID_PATTERN, text)


# --- the tables -----------------------------------------------------------------------------


@pytest.fixture
def database(tmp_path: Path) -> Path:
    """A database with the tables of the models and the view, as ``create_all`` makes it."""
    path = tmp_path / APP_DB_FILE
    engine = create_engine(f"sqlite:///{path}")
    SQLModel.metadata.create_all(engine, tables=[model.__table__ for model in models.TABLES])
    with engine.begin() as connection:
        connection.exec_driver_sql(db.RATINGS_FLAT_SQL)
    engine.dispose()
    return path


def test_the_models_are_the_tables_of_the_docs(database: Path) -> None:
    assert tuple(model.__tablename__ for model in models.TABLES) == APP_TABLES == tuple(COLUMNS)
    connection = sqlite3.connect(database)
    try:
        found = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert found == set(APP_TABLES)
        for table, columns in COLUMNS.items():
            names = tuple(row[1] for row in connection.execute(f'PRAGMA table_info("{table}")'))
            assert names == columns, table
        keys = {
            table: tuple(
                row[1]
                for row in sorted(
                    (r for r in connection.execute(f'PRAGMA table_info("{table}")') if r[5]),
                    key=lambda r: r[5],
                )
            )
            for table in APP_TABLES
        }
        assert keys == {
            "raters": ("rater_id",),
            "assignments": ("assignment_id",),
            "submissions": ("submission_id",),
            "ratings": ("rating_id",),
            "marks": ("mark_id",),
            "events": ("event_id",),
            "final_labels": ("subject_uid", "criterion"),
            "final_marks": ("final_mark_id",),
            "app_meta": ("key",),
        }
    finally:
        connection.close()


def _filled(database: Path) -> tuple[Session, models.Assignment, models.Submission]:
    session = Session(create_engine(f"sqlite:///{database}"))
    rater = models.Rater(name="local", role="lead", created_at=db.utc_now())
    session.add(rater)
    session.flush()
    assert rater.rater_id is not None
    assignment = models.Assignment(
        rater_id=rater.rater_id,
        subject_uid="3f9a1c07d2e4b8a1",
        seq=1,
        render_key="5d0c1f2a9b7e4c31",
        assigned_at=db.utc_now(),
    )
    session.add(assignment)
    session.flush()
    assert assignment.assignment_id is not None
    submission = models.Submission(
        assignment_id=assignment.assignment_id,
        revision=1,
        protocol_version="0.1-draft",
        protocol_sha256="9c1f",
        app_version="0.0.1",
        server_ts=db.utc_now(),
        ms_on_subject=14200,
        note="a note",
    )
    session.add(submission)
    session.flush()
    return session, assignment, submission


def _mark(
    assignment: models.Assignment, submission: models.Submission, **fields: object
) -> models.Mark:
    assert assignment.assignment_id is not None and submission.submission_id is not None
    values: dict[str, object] = dict.fromkeys(
        ("px", "py", "vox_i", "vox_j", "vox_k", "tkr_x", "tkr_y", "tkr_z"), 1.5
    )
    values |= dict.fromkeys(("ras_x", "ras_y", "ras_z", "box_dist_mm"), 0.0)
    values |= {"mark_class": "A", "panel": "sagittal", "slice": 3, "created_ts": db.utc_now()}
    return models.Mark(
        submission_id=submission.submission_id,
        assignment_id=assignment.assignment_id,
        **(values | fields),  # pyrefly: ignore[bad-argument-type]
    )


def test_rows_go_in_and_come_out_with_their_defaults(database: Path) -> None:
    session, assignment, submission = _filled(database)
    with session:
        assert (assignment.kind, assignment.completed_at) == ("main", None)
        assert (submission.current, submission.other, submission.unratable_reason) == (
            True,
            False,
            None,
        )
        session.add(_mark(assignment, submission))
        session.add(models.Rating(submission_id=1, assignment_id=1, criterion="A", value=2))
        session.add(models.Event(assignment_id=None, ts=db.utc_now(), type="config_change"))
        session.add(
            models.FinalLabel(
                subject_uid="3f9a1c07d2e4b8a1", criterion="A", value=2, method="single"
            )
        )
        session.add(
            models.FinalLabel(
                subject_uid="3f9a1c07d2e4b8a1", criterion="unratable", method="single"
            )
        )
        session.add(
            models.FinalMark(
                subject_uid="3f9a1c07d2e4b8a1", mark_class="A", tkr_x=1, tkr_y=2, tkr_z=3
            )
        )
        session.add(models.AppMeta(key="schema_version", value="1"))
        session.commit()
        mark = session.exec(select(models.Mark)).one()
        assert (mark.mark_class, mark.hemi, mark.pial_vertex, mark.deleted_ts) == (
            "A",
            None,
            None,
            None,
        )
        assert session.exec(select(models.Event)).one().payload_json == "{}"
        assert session.exec(select(models.FinalMark)).one().source_mark_ids == "[]"
    connection = sqlite3.connect(database)
    try:
        # the column is `class`; code outside app/ reads it under that name
        assert connection.execute('SELECT "class", slice FROM marks').fetchall() == [("A", 3)]
        assert connection.execute('SELECT "class" FROM final_marks').fetchall() == [("A",)]
        flat = pd.read_sql_query(f"SELECT * FROM {RATINGS_FLAT_VIEW}", connection)
    finally:
        connection.close()
    assert list(flat.columns) == [
        "rating_id",
        "assignment_id",
        "criterion",
        "value",
        "note",
        "protocol_version",
        "app_version",
        "client_ts",
        "server_ts",
        "ms_on_subject",
    ]
    assert flat[["criterion", "value", "note", "ms_on_subject"]].to_dict("records") == [
        {"criterion": "A", "value": 2, "note": "a note", "ms_on_subject": 14200}
    ]


def test_the_view_shows_current_submissions_only(database: Path) -> None:
    session, assignment, submission = _filled(database)
    with session:
        assert assignment.assignment_id is not None and submission.submission_id is not None
        session.add(
            models.Rating(
                submission_id=submission.submission_id,
                assignment_id=assignment.assignment_id,
                criterion="A",
                value=2,
            )
        )
        submission.current = False
        revised = models.Submission(
            assignment_id=assignment.assignment_id,
            revision=2,
            protocol_version="0.1-draft",
            protocol_sha256="9c1f",
            app_version="0.0.1",
            server_ts=db.utc_now(),
        )
        session.add(revised)
        session.flush()
        assert revised.submission_id is not None
        session.add(
            models.Rating(
                submission_id=revised.submission_id,
                assignment_id=assignment.assignment_id,
                criterion="A",
                value=1,
            )
        )
        session.commit()
    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(f"SELECT criterion, value FROM {RATINGS_FLAT_VIEW}").fetchall()
        stored = connection.execute("SELECT COUNT(*) FROM ratings").fetchone()
    finally:
        connection.close()
    assert rows == [("A", 1)] and stored == (2,)


@pytest.mark.parametrize(
    "row",
    [
        lambda a, s: models.Rater(name="x", role="boss", created_at="t"),
        lambda a, s: models.Rater(name="local", role="rater", created_at="t"),  # the name is taken
        lambda a, s: models.Assignment(
            rater_id=1, subject_uid="u", seq=2, render_key="k", assigned_at="t", kind="extra"
        ),
        lambda a, s: models.Assignment(  # (rater, seq) is taken
            rater_id=1, subject_uid="u", seq=1, render_key="k", assigned_at="t"
        ),
        lambda a, s: models.Submission(  # (assignment, revision) is taken
            assignment_id=1,
            revision=1,
            protocol_version="v",
            protocol_sha256="h",
            app_version="0",
            server_ts="t",
        ),
        lambda a, s: models.Rating(submission_id=1, assignment_id=1, criterion="D", value=0),
        lambda a, s: _mark(a, s, mark_class="D"),
        lambda a, s: _mark(a, s, panel="oblique"),
        lambda a, s: _mark(a, s, mode="loop"),
        lambda a, s: _mark(a, s, hemi="both"),
        lambda a, s: _mark(a, s, vertex_surface="sphere"),
        lambda a, s: models.FinalLabel(subject_uid="u", criterion="D", value=0, method="single"),
        lambda a, s: models.FinalLabel(subject_uid="u", criterion="A", value=0, method="guess"),
        lambda a, s: models.FinalMark(subject_uid="u", mark_class="D", tkr_x=0, tkr_y=0, tkr_z=0),
    ],
)
def test_the_database_refuses_a_row_outside_its_vocabulary_or_its_keys(
    database: Path, row: object
) -> None:
    from sqlalchemy.exc import IntegrityError

    session, assignment, submission = _filled(database)
    with session:
        session.commit()
        session.add(row(assignment, submission))  # pyrefly: ignore[not-callable]
        with pytest.raises(IntegrityError):
            session.commit()


def test_a_rating_has_one_value_per_criterion_and_submission(database: Path) -> None:
    from sqlalchemy.exc import IntegrityError

    session, _, _ = _filled(database)
    with session:
        session.add(models.Rating(submission_id=1, assignment_id=1, criterion="A", value=2))
        session.commit()
        session.add(models.Rating(submission_id=1, assignment_id=1, criterion="A", value=1))
        with pytest.raises(IntegrityError):
            session.commit()


# --- reading a run without a Run --------------------------------------------------------------


def test_the_scores_and_clusters_of_a_run_directory_are_those_of_the_run(small_run: Path) -> None:
    scores = primary_scores(small_run)
    clusters = clusters_with_native(small_run)
    with Run(small_run) as run:
        by_run = run.scores()
        pd.testing.assert_frame_equal(clusters, run.clusters())
    # the run adds the labels of a local run, and nothing else
    assert list(by_run.columns) == [scores.columns[0], "subject_label", *scores.columns[1:]]
    pd.testing.assert_frame_equal(scores, by_run.drop(columns="subject_label"))
    assert scores["primary_rank"].tolist() == [1, 2, 3] and scores["primary"].all()
    assert len(clusters) > 0 and clusters["tkr_x"].notna().any()


def test_a_run_directory_without_tables_has_empty_scores_and_clusters(tmp_path: Path) -> None:
    scores = primary_scores(tmp_path)
    assert len(scores) == 0 and "primary_rank" in scores.columns
    clusters = clusters_with_native(tmp_path)
    assert len(clusters) == 0 and {"peak_z", "tkr_x"} <= set(clusters.columns)


def test_the_render_of_a_subject_is_read_from_its_media_directory(
    small_run: Path, small_run_uids: dict[str, str], tmp_path: Path
) -> None:
    for uid in small_run_uids.values():
        directory = media_dir(small_run, uid)
        info = read_render(directory)
        assert info is not None
        assert info.to_json() == (directory / RENDER_JSON).read_text()
        assert info.inputs and info.vertex_map is not None
        assert (directory / info.vertex_map.file).is_file()
    # no render, a render that is being written, a damaged file, an earlier schema: None
    assert read_render(tmp_path) is None
    assert read_render(tmp_path / "absent") is None
    text = (media_dir(small_run, uid) / RENDER_JSON).read_text()
    for damaged in ("", text[: len(text) // 2], "{}", text.replace('"box"', '"bax"', 1)):
        (tmp_path / RENDER_JSON).write_text(damaged)
        assert read_render(tmp_path) is None
    (tmp_path / RENDER_JSON).write_text(text)
    assert read_render(tmp_path) == info and read_render(str(tmp_path)) == info


def test_reading_a_render_does_not_load_the_renderer() -> None:
    # the server reads render.json for every assignment; it has no use for matplotlib
    code = (
        "import sys\n"
        "from surfcensus.render.model import read_render\n"
        "import surfcensus.app.schemas, surfcensus.protocol, surfcensus.calibration\n"
        "print('matplotlib' in sys.modules, 'surfcensus.render.api' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False"]


# --- the browser of the browser tests ---------------------------------------------------------


def test_the_browser_is_the_one_named_and_else_chrome(tmp_path: Path) -> None:
    assert BROWSER_ENV == "SURFCENSUS_BROWSER"
    assert launch_options({}) == {"channel": "chrome"}
    assert launch_options({BROWSER_ENV: ""}) == {"channel": "chrome"}
    executable = tmp_path / "chromium"
    executable.write_bytes(b"")
    assert launch_options({BROWSER_ENV: str(executable)}) == {"executable_path": str(executable)}
    # a name that is no file is an error, not a reason to use another browser or to skip
    for wrong in (tmp_path / "absent", tmp_path):
        with pytest.raises(FileNotFoundError, match="SURFCENSUS_BROWSER"):
            launch_options({BROWSER_ENV: str(wrong)})


# --- the fixtures the app tests share -------------------------------------------------------


def test_the_engine_fixture_opens_the_database_of_the_app_run(
    app_run: Path, app_engine: Any
) -> None:
    assert Path(app_engine.url.database) == app_run / APP_DB_FILE
    assert (app_run / APP_DB_FILE).is_file()


def test_the_fixtures_of_the_app_tests_are_what_they_say(
    app_run: Path,
    small_run: Path,
    small_run_uids: dict[str, str],
    small_run_renders: dict[str, RenderInfo],
    shipped_protocol: Protocol,
    api_examples: dict[str, object],
) -> None:
    assert sorted(small_run_uids) == ["sub-000", "sub-001", "sub-002"]
    assert sorted(small_run_renders) == sorted(small_run_uids)
    for label, info in small_run_renders.items():
        assert info == read_render(media_dir(small_run, small_run_uids[label]))
        assert {name: len(panel.slices) for name, panel in info.panels.items()} == {
            "sagittal": 7,
            "coronal": 6,
            "axial": 7,
        }
        assert info.vertex_map is not None and info.vertex_map.fsaverage == "fsaverage6"
    assert primary_scores(small_run)["subject_uid"][0] == small_run_uids["sub-001"]
    protocol, _ = load_protocol(app_run)
    assert protocol == shipped_protocol
    assert not (app_run / APP_DB_FILE).exists() and not (small_run / "protocol.yaml").exists()
    with PublicCatalog.open(app_run) as public:
        renders = public.renders()
        selection = public.selection()
        assert public.mode() == "local"
    assert sorted(renders["subject_uid"]) == sorted(small_run_uids.values())
    assert set(renders["status"]) == {"done"} and set(selection["reason"]) == {"all"}
    # the three subjects are the same anatomy, so one key names the tiles of all of them
    assert renders["render_key"].nunique() == 1
    config = load_config(app_run)
    assert (config.app.require_marks, config.app.blind, config.render_video) == (
        False,
        False,
        False,
    )
    assert config.app.mark_max_dist_mm <= config.render.vertex_map_margin_mm
    assert set(api_examples) >= {"session", "next_assignment", "rating_in", "rating_out"}
