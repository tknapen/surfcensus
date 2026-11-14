"""Tests of surfcensus/export/tables.py: the tables of ``surfcensus export`` and the two frames
behind the review screen, from the catalog, the metric tables and ``app.sqlite``.

The ratings are written with the models of the app (``app.db.store_submission``), as the
server writes them, and read back by the export with the standard library."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sqlalchemy import func
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, col, select
from synth_tree import write_tier_a, write_tree  # pyrefly: ignore[missing-import]

from surfcensus import __version__
from surfcensus.app import APP_DB_FILE, LABEL_REASONS_KEY
from surfcensus.app.db import (
    APP_SCHEMA_VERSION,
    get_or_create_rater,
    open_app_db,
    record_skip,
    store_submission,
)
from surfcensus.app.events import CONFIG_CHANGE, record_setting
from surfcensus.app.finalize import record_single, sync_unratable
from surfcensus.app.models import (
    AppMeta,
    Assignment,
    Event,
    FinalLabel,
    FinalMark,
    Mark,
    Submission,
)
from surfcensus.app.queue import QueuePlan
from surfcensus.app.schemas import (
    EVENT_TYPES,
    FINAL_METHODS,
    SKIP_EVENT,
    RatingIn,
    ResolvedMark,
    SkipIn,
)
from surfcensus.calibration import CalibrationSet, CalibrationSubject, write_calibration
from surfcensus.config import CONFIG_FILE, ConfigError, load_config, with_changes, write_config
from surfcensus.export import tables
from surfcensus.export.tables import (
    BEHAVIOUR_COLUMNS,
    DATASETS_COLUMNS,
    EXPORT_DIRS,
    EXPORT_TABLES,
    FINAL_MARKS_COLUMNS,
    MANIFEST,
    MARKS_COLUMNS,
    RATINGS_COLUMNS,
    RELEASE_LEFT_OUT,
    SKIPS_COLUMNS,
    SUBJECT_RATING_COLUMNS,
    Format,
    Profile,
    export_run,
    frame_csv,
    marks_frame,
    ratings_frame,
    subjects_frame,
)
from surfcensus.privacy import HashKey
from surfcensus.protocol import PROTOCOL_FILE, Protocol
from surfcensus.run import Run
from surfcensus.schemas import KEYS, SCHEMAS
from surfcensus.sources.local import local
from surfcensus.store import (
    CATALOG_FILE,
    Catalog,
    PublicCatalog,
    clusters_with_native,
    read_table,
    table_path,
    write_table,
)

DATASET_KEY = HashKey(bytes(range(1, 33)), "dataset-key")
FORMATS: tuple[Format, ...] = ("csv", "parquet")
PROFILES: tuple[Profile, ...] = ("analysis", "release")
T1, T2, T3 = (f"2026-11-13T10:00:0{n}.000Z" for n in (1, 2, 3))
VALUES = {"A": 2, "B": 0, "C": 1}
NONE = {"A": 0, "B": 0, "C": 0}
BLOCK_APP_PACKAGES = "import sys\n" + "".join(
    f"sys.modules[{name!r}] = None\n"
    for name in ("fastapi", "uvicorn", "sqlmodel", "sqlalchemy", "jinja2", "starlette")
)


# --- what the tests write -----------------------------------------------------------------------


def click(client_id: str, mark_class: str = "A", **fields: Any) -> dict[str, Any]:
    posted = {"client_id": client_id, "class": mark_class, "panel": "sagittal", "slice": 3}
    return posted | {"px": 30.5, "py": 28.25} | fields


def resolved(**fields: Any) -> ResolvedMark:
    values: dict[str, Any] = {
        "hemi": "lh",
        "pial_vertex": 211,
        "pial_dist_mm": 0.8,
        "white_vertex": 207,
        "white_dist_mm": 2.9,
        "fsavg6_vertex": 12950,
        "vertex_surface": "pial",
        "box_dist_mm": 0.0,
        "vox": (7.5, 8.25, 9.0),
        "tkr": (-1.5, -2.25, 3.0),
        "ras": (0.5, -4.25, 11.0),
    }
    return ResolvedMark.model_validate(values | fields)


def render_key(run_dir: Path, uid: str) -> str:
    with PublicCatalog.open(run_dir) as public:
        row = public.render(uid)
    assert row is not None and row.render_key is not None
    return row.render_key


def submit(
    engine: Engine,
    assignment_id: int,
    *,
    values: dict[str, int] | None = None,
    unratable: str | None = None,
    other: bool = False,
    note: str = "",
    marks: Sequence[dict[str, Any]] = (),
    found: Sequence[ResolvedMark] | None = None,
    ms: int = 1500,
    now: str = T1,
    protocol_version: str = "0.1-draft",
) -> None:
    """Store a rating of an assignment as the server does: the next revision of it."""
    with Session(engine) as session:
        assignment = session.get(Assignment, assignment_id)
        assert assignment is not None
        rating = RatingIn.model_validate(
            {
                "assignment_id": assignment_id,
                "protocol_version": protocol_version,
                "values": {} if unratable else (VALUES if values is None else values),
                "unratable": unratable,
                "other": other,
                "note": note,
                "marks": list(marks),
                "client_ts": "2026-11-13T09:59:59.000Z",
                "ms_on_subject": ms,
            }
        )
        found = [resolved() for _ in marks] if found is None else found
        store_submission(
            session, assignment, rating, found, protocol_sha256="9c1f", app_version="0.0.1", now=now
        )
        session.commit()


def assign(
    engine: Engine, run_dir: Path, uid: str, *, rater: str = "local", kind: str = "main"
) -> int:
    """An open assignment of ``uid`` to ``rater``, on the render the subject has now."""
    with Session(engine) as session:
        who = get_or_create_rater(session, rater, role="lead")
        last = session.exec(
            select(func.max(Assignment.seq)).where(Assignment.rater_id == who.rater_id)
        ).one()
        assignment = Assignment(
            rater_id=who.rater_id,
            subject_uid=uid,
            kind=kind,
            seq=(last or 0) + 1,
            render_key=render_key(run_dir, uid),
            assigned_at=T1,
        )
        session.add(assignment)
        session.commit()
        assert assignment.assignment_id is not None
        return assignment.assignment_id


def rate(
    engine: Engine,
    run_dir: Path,
    uid: str,
    *,
    rater: str = "local",
    kind: str = "main",
    **rating: Any,
) -> int:
    assignment_id = assign(engine, run_dir, uid, rater=rater, kind=kind)
    submit(engine, assignment_id, **rating)
    return assignment_id


def log(engine: Engine, assignment_id: int | None, *events: tuple[str, Any]) -> None:
    with Session(engine) as session:
        for kind, payload in events:
            text = payload if isinstance(payload, str) else json.dumps(payload)
            session.add(Event(assignment_id=assignment_id, ts=T1, type=kind, payload_json=text))
        session.commit()


def label(engine: Engine, uid: str, method: str = "single", **values: int | None) -> None:
    with Session(engine) as session:
        for criterion, value in values.items():
            session.add(
                FinalLabel(subject_uid=uid, criterion=criterion, value=value, method=method)
            )
        session.commit()


def uids_of(run_dir: Path) -> list[str]:
    with PublicCatalog.open(run_dir) as public:
        return list(public.subjects()["subject_uid"])


def private(run_dir: Path, sql: str, params: Sequence[object] = ()) -> None:
    """Change what only the inventory, the enrichment or a render writes."""
    with Catalog(run_dir / CATALOG_FILE) as catalog, catalog.write() as connection:
        connection.execute(sql, params)


def by_uid(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.set_index("subject_uid")


def listing(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir())}


def read_csv(path: Path) -> pd.DataFrame:
    """A CSV file as text, every field as it was written."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def manifest_of(directory: Path) -> dict[str, Any]:
    return json.loads((directory / MANIFEST).read_text())


@pytest.fixture
def run(app_run: Path) -> Path:
    """A local run of three rendered and selected subjects with a protocol, no ratings."""
    return app_run


@pytest.fixture
def engine(app_run: Path, app_engine: Engine) -> Engine:
    return app_engine


@pytest.fixture
def uids(small_run_uids: dict[str, str]) -> list[str]:
    """The three subjects in the order of their labels (``sub-000`` to ``sub-002``)."""
    return [small_run_uids[name] for name in sorted(small_run_uids)]


# --- the columns ----------------------------------------------------------------------------------


def test_every_table_has_the_documented_columns(run: Path, engine: Engine, uids: list[str]) -> None:
    rate(engine, run, uids[0], marks=[click("m1")])
    paths = export_run(run)
    assert [path.name for path in paths] == [f"{name}.csv" for name in EXPORT_TABLES] + [MANIFEST]
    assert {path.parent for path in paths} == {run / "export"}
    columns = {path.stem: list(read_csv(path).columns) for path in paths[:-1]}

    with PublicCatalog.open(run) as public:
        catalog = list(public.subjects().columns)
    scores = SCHEMAS["scores"].names
    assert catalog[:4] == ["subject_uid", "dataset", "subject_label", "participant"]
    assert columns["subjects"] == [
        *catalog,
        "render_status",
        "render_reason",
        *(name for name in scores if name not in catalog),
        *SUBJECT_RATING_COLUMNS,
    ]
    # what the scores have and the subjects table lacks is in the catalog's columns already
    assert set(scores) - set(columns["subjects"]) == set()
    assert [name for name in scores if name in catalog] == [
        "subject_uid",
        "dataset",
        "holes_lh",
        "holes_rh",
    ]
    assert columns["scores"] == scores
    assert columns["selection"] == ["subject_uid", "reason", "stratum", "inclusion_prob", "seed"]
    assert columns["ratings"] == list(RATINGS_COLUMNS)
    assert columns["marks"] == list(MARKS_COLUMNS)
    assert columns["final_marks"] == list(FINAL_MARKS_COLUMNS)
    native = [n for n in SCHEMAS["clusters_native"].names if n not in SCHEMAS["clusters"].names]
    assert columns["clusters"] == [*SCHEMAS["clusters"].names, *native]
    assert columns["datasets"] == list(DATASETS_COLUMNS)
    assert columns["behaviour"] == list(BEHAVIOUR_COLUMNS)
    assert columns["skips"] == list(SKIPS_COLUMNS)


def test_the_mark_tables_follow_the_tables_of_the_app() -> None:
    of_table = [column.name for column in SQLModel.metadata.tables[Mark.__tablename__].columns]
    assert MARKS_COLUMNS[:3] == ("subject_uid", "rater", "assignment_id")
    assert list(MARKS_COLUMNS[3:]) == [
        name for name in of_table if name not in ("assignment_id", "deleted_ts")
    ]
    final = SQLModel.metadata.tables[FinalMark.__tablename__].columns
    assert list(FINAL_MARKS_COLUMNS) == [column.name for column in final]


def test_the_export_reads_the_schema_that_the_app_writes() -> None:
    # a new schema of app.sqlite fails here until the export was looked at
    assert tables.APP_SCHEMA_READ == APP_SCHEMA_VERSION


# --- ratings ------------------------------------------------------------------------------------


def test_a_rating_is_a_row_with_its_values_and_what_it_was_made_on(
    run: Path, engine: Engine, uids: list[str]
) -> None:
    first = rate(engine, run, uids[1], other=True, note="skull, far from the sinus", ms=14200)
    second = rate(engine, run, uids[0], unratable="image too poor to judge", now=T2)
    frame = ratings_frame(run)
    assert list(frame.columns) == list(RATINGS_COLUMNS)
    assert list(frame["assignment_id"]) == [first, second]
    row = frame.iloc[0].to_dict()
    # nothing recorded how the queue was ordered before the rating was stored
    assert pd.isna(row.pop("blind"))
    assert row == {
        "assignment_id": first,
        "subject_uid": uids[1],
        "rater": "local",
        "kind": "main",
        "seq": 1,
        "render_key": render_key(run, uids[1]),
        "render_current": True,
        "A": 2,
        "B": 0,
        "C": 1,
        "unratable_reason": None,
        "other": True,
        "note": "skull, far from the sinus",
        "revision": 1,
        "protocol_version": "0.1-draft",
        "protocol_sha256": "9c1f",
        "app_version": "0.0.1",
        "client_ts": "2026-11-13T09:59:59.000Z",
        "server_ts": T1,
        "ms_on_subject": 14200,
    }
    unratable = frame.iloc[1]
    assert unratable["unratable_reason"] == "image too poor to judge"
    assert all(unratable[criterion] is pd.NA for criterion in "ABC")
    assert not unratable["other"] and unratable["server_ts"] == T2


def test_a_revised_rating_is_exported_as_it_stands_now(
    run: Path, engine: Engine, uids: list[str]
) -> None:
    assignment = rate(engine, run, uids[0], values=NONE, note="first look")
    submit(engine, assignment, values=VALUES, note="second look", now=T2, ms=900)
    frame = ratings_frame(run)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert (row["A"], row["B"], row["C"]) == (2, 0, 1)
    assert (row["revision"], row["note"], row["server_ts"]) == (2, "second look", T2)
    assert row["ms_on_subject"] == 900


def test_a_rating_says_where_it_stood_in_the_queue_and_whether_the_queue_was_blind(
    run: Path, engine: Engine, uids: list[str]
) -> None:
    """Whether a rater saw the subjects worst first or shuffled is an order effect, and the
    notebooks read the export only."""
    before = rate(engine, run, uids[2], rater="early", now="2026-11-13T09:00:00.000Z")
    with Session(engine) as session:
        record_setting(session, tables._BLIND_SETTING, False, now="2026-11-13T10:00:00.000Z")
        # another setting, and records that are not of the form the server writes
        record_setting(session, "require_marks", True, now="2026-11-13T10:00:00.500Z")
        session.commit()
    log(
        engine,
        None,
        (CONFIG_CHANGE, "not json"),
        (CONFIG_CHANGE, {"setting": "blind", "new": "yes"}),
        (CONFIG_CHANGE, ["blind", True]),
    )
    scored = rate(engine, run, uids[0], now=T1)
    with Session(engine) as session:
        record_setting(session, tables._BLIND_SETTING, True, now="2026-11-13T10:00:02.500Z")
        session.commit()
    shuffled = rate(engine, run, uids[1], now=T3)
    other = rate(engine, run, uids[0], rater="second", now=T3)
    # a revision keeps the place in the queue and is under the setting it was stored under
    submit(engine, scored, values=NONE, now=T3)
    again = rate(engine, run, uids[2], rater="second", now=T2)

    frame = ratings_frame(run).set_index("assignment_id")
    assert frame["seq"].to_dict() == {before: 1, scored: 1, shuffled: 2, other: 1, again: 2}
    assert pd.isna(frame.loc[before, "blind"])
    assert frame["blind"].drop(index=before).to_dict() == {
        scored: True,
        shuffled: True,
        other: True,
        again: False,
    }

    export_run(run)
    written = by_assignment(read_csv(run / "export" / "ratings.csv"))
    assert written["blind"].to_dict() == {
        str(before): "",
        str(scored): "True",
        str(shuffled): "True",
        str(other): "True",
        str(again): "False",
    }
    assert written["seq"].to_dict() == {
        str(before): "1",
        str(scored): "1",
        str(shuffled): "2",
        str(other): "1",
        str(again): "2",
    }


def test_a_rating_under_the_first_record_of_the_setting_is_not_blind(
    run: Path, engine: Engine, uids: list[str]
) -> None:
    with Session(engine) as session:
        record_setting(session, tables._BLIND_SETTING, False, now=T1)
        session.commit()
    # the record and the submission in one millisecond: the server records before it serves
    rate(engine, run, uids[0], now=T1)
    assert ratings_frame(run)["blind"].tolist() == [False]


def test_the_export_reads_the_setting_the_app_records() -> None:
    assert tables._CONFIG_CHANGE == CONFIG_CHANGE


def test_an_assignment_that_is_still_open_is_no_rating(
    run: Path, engine: Engine, uids: list[str]
) -> None:
    done = rate(engine, run, uids[0])
    assign(engine, run, uids[1])
    assert list(ratings_frame(run)["assignment_id"]) == [done]
    export_run(run)
    assert list(read_csv(run / "export" / "behaviour.csv")["assignment_id"]) == [str(done)]
    assert by_uid(subjects_frame(run))["n_ratings"].to_dict() == {
        uids[0]: 1,
        uids[1]: 0,
        uids[2]: 0,
    }


def test_the_kind_of_an_assignment_is_told_as_it_is(
    run: Path, engine: Engine, uids: list[str]
) -> None:
    rate(engine, run, uids[0])
    rate(engine, run, uids[0], kind="repeat", now=T2)
    rate(engine, run, uids[1], kind="calibration", now=T3)
    frame = ratings_frame(run)
    assert list(frame["kind"]) == ["main", "repeat", "calibration"]
    # every completed assignment counts for the subject, whatever its kind
    assert by_uid(subjects_frame(run))["n_ratings"].to_dict() == {
        uids[0]: 2,
        uids[1]: 1,
        uids[2]: 0,
    }


def test_a_rating_says_whether_it_was_made_on_the_tiles_the_subject_has_now(
    run: Path, engine: Engine, uids: list[str]
) -> None:
    for uid in uids:
        rate(engine, run, uid)
    assert list(ratings_frame(run)["render_current"]) == [True, True, True]

    with PublicCatalog.open(run, readonly=False) as public:
        rendered_again = public.render(uids[1])
        failed = public.render(uids[2])
        assert rendered_again is not None and failed is not None
        public.record_renders(
            [
                dataclasses.replace(rendered_again, render_key="0a1b2c3d4e5f6a7b"),
                # a later render of the subject failed: it has no tiles now
                dataclasses.replace(failed, status="failed", reason="render failure"),
            ]
        )
    frame = by_uid(ratings_frame(run))
    assert frame["render_current"].to_dict() == {uids[0]: True, uids[1]: False, uids[2]: False}
    # the key in the row is still the one the rating was made on
    assert set(frame["render_key"]) == {rendered_again.render_key} == {failed.render_key}

    private(run, "DELETE FROM renders WHERE subject_uid = ?", (uids[0],))
    assert not by_uid(ratings_frame(run))["render_current"].any()


def test_raters_keep_their_names_in_a_local_run(run: Path, engine: Engine, uids: list[str]) -> None:
    rate(engine, run, uids[0], rater="Tomas", marks=[click("m1")])
    rate(engine, run, uids[0], rater="Ada Lovelace", marks=[click("m1"), click("m2")])
    assert list(ratings_frame(run)["rater"]) == ["Tomas", "Ada Lovelace"]
    export_run(run)
    assert list(read_csv(run / "export" / "marks.csv")["rater"]) == ["Tomas"] + 2 * ["Ada Lovelace"]
    assert frame_csv(marks_frame(run)).encode() == (run / "export" / "marks.csv").read_bytes()
    assert list(read_csv(run / "export" / "behaviour.csv")["rater"]) == ["Tomas", "Ada Lovelace"]


def test_a_note_with_commas_quotes_and_line_ends_survives_the_csv(
    run: Path, engine: Engine, uids: list[str]
) -> None:
    note = 'dura, "enclosed"\nsecond line; =SUM(A1) and ünïcödé'
    rate(engine, run, uids[0], other=True, note=note)
    export_run(run)
    path = run / "export" / "ratings.csv"
    assert list(read_csv(path)["note"]) == [note]
    # what the server answers GET /api/export with is the file
    assert frame_csv(ratings_frame(run)).encode() == path.read_bytes()
    assert b"\r\n" not in path.read_bytes()


# --- marks ----------------------------------------------------------------------------------------


def test_marks_are_those_of_current_submissions_with_everything_the_table_holds(
    run: Path, engine: Engine, uids: list[str]
) -> None:
    assignment = rate(
        engine,
        run,
        uids[0],
        marks=[click("m1"), click("m2", "C", panel="axial", slice=2, frame=5, mode="seq")],
        found=[
            resolved(),
            resolved(
                hemi=None,
                pial_vertex=None,
                pial_dist_mm=None,
                white_vertex=None,
                white_dist_mm=None,
                fsavg6_vertex=None,
                vertex_surface=None,
                box_dist_mm=4.25,
                tkr=(10.5, -80.25, 3.125),
            ),
        ],
    )
    other = rate(engine, run, uids[1], rater="second", marks=[click("m1", "B")], now=T2)
    export_run(run, fmt="parquet")
    marks = read_table(run / "export" / "marks.parquet")
    assert list(marks["assignment_id"]) == [assignment, assignment, other]
    assert list(marks["subject_uid"]) == [uids[0], uids[0], uids[1]]
    assert list(marks["rater"]) == ["local", "local", "second"]
    assert list(marks["class"]) == ["A", "C", "B"]
    first, volumetric = marks.iloc[0], marks.iloc[1]
    assert (first["panel"], first["slice"], first["px"], first["py"]) == (
        "sagittal",
        3,
        30.5,
        28.25,
    )
    assert (first["vox_i"], first["vox_j"], first["vox_k"]) == (7.5, 8.25, 9.0)
    assert (first["tkr_x"], first["tkr_y"], first["tkr_z"]) == (-1.5, -2.25, 3.0)
    assert (first["ras_x"], first["ras_y"], first["ras_z"]) == (0.5, -4.25, 11.0)
    assert (first["hemi"], first["pial_vertex"], first["white_vertex"]) == ("lh", 211, 207)
    assert (first["pial_dist_mm"], first["white_dist_mm"]) == (0.8, 2.9)
    assert (first["fsavg6_vertex"], first["vertex_surface"]) == (12950, "pial")
    assert pd.isna(first["frame"]) and first["mode"] is None and first["created_ts"] == T1
    # a mark without a vertex in range is kept as a point
    assert (volumetric["panel"], volumetric["slice"], volumetric["frame"]) == ("axial", 2, 5)
    assert (volumetric["mode"], volumetric["box_dist_mm"]) == ("seq", 4.25)
    assert (volumetric["tkr_x"], volumetric["tkr_y"], volumetric["tkr_z"]) == (10.5, -80.25, 3.125)
    for name in ("hemi", "pial_vertex", "pial_dist_mm", "white_vertex", "fsavg6_vertex"):
        assert pd.isna(volumetric[name]), name

    # a revision: the marks of the superseded submission are no marks of the export
    submit(engine, assignment, marks=[click("m9", "A", px=11.0)], now=T3)
    export_run(run, fmt="parquet")
    marks = read_table(run / "export" / "marks.parquet")
    assert list(zip(marks["assignment_id"], marks["px"], strict=True)) == [
        (other, 30.5),
        (assignment, 11.0),
    ]
    assert by_uid(subjects_frame(run))["n_marks"].to_dict() == {uids[0]: 1, uids[1]: 1, uids[2]: 0}


def test_final_marks_are_exported_as_the_table_has_them(
    run: Path, engine: Engine, uids: list[str]
) -> None:
    with Session(engine) as session:
        session.add(
            FinalMark(
                subject_uid=uids[2],
                mark_class="B",
                tkr_x=1.5,
                tkr_y=-90.25,
                tkr_z=4.0,
                hemi="rh",
                pial_vertex=40,
                fsavg6_vertex=None,
                source_mark_ids="[3, 8]",
            )
        )
        session.add(FinalMark(subject_uid=uids[0], mark_class="X", tkr_x=0.0, tkr_y=0.0, tkr_z=0.0))
        session.commit()
    export_run(run)
    rows = read_csv(run / "export" / "final_marks.csv").to_dict("records")
    assert rows == [
        {
            "final_mark_id": "1",
            "subject_uid": uids[2],
            "class": "B",
            "tkr_x": "1.5",
            "tkr_y": "-90.25",
            "tkr_z": "4.0",
            "hemi": "rh",
            "pial_vertex": "40",
            "fsavg6_vertex": "",
            "source_mark_ids": "[3, 8]",
        },
        {
            "final_mark_id": "2",
            "subject_uid": uids[0],
            "class": "X",
            "tkr_x": "0.0",
            "tkr_y": "0.0",
            "tkr_z": "0.0",
            "hemi": "",
            "pial_vertex": "",
            "fsavg6_vertex": "",
            "source_mark_ids": "[]",
        },
    ]


# --- subjects -------------------------------------------------------------------------------------


def test_a_subject_has_its_provenance_its_render_and_its_primary_score(
    run: Path, small_run_uids: dict[str, str]
) -> None:
    frame = subjects_frame(run)
    assert list(frame["subject_uid"]) == sorted(small_run_uids.values())
    by_label = frame.set_index("subject_label")
    assert by_label["subject_uid"].to_dict() == small_run_uids
    assert set(frame["render_status"]) == {"done"} and frame["render_reason"].isna().all()
    assert frame["present"].all() and frame["complete"].all()
    assert set(frame["fs_version"]) == {"6.0.1"}

    with PublicCatalog.open(run) as public:
        catalog = by_uid(public.subjects())
    scores = read_table(table_path(run, "scores"), SCHEMAS["scores"])
    scores = by_uid(scores[scores["primary"]])
    frame = by_uid(frame)
    for uid in frame.index:
        for name in catalog.columns:
            assert _same(frame.loc[uid, name], catalog.loc[uid, name]), name
        for name in scores.columns:
            assert _same(frame.loc[uid, name], scores.loc[uid, name]), name
    # the one order of the subjects of a run, and the subject with the patch is first in it
    assert by_label.loc["sub-001", "primary_rank"] == 1
    assert sorted(by_label["primary_rank"]) == [1, 2, 3]
    assert str(frame["composite"].dtype) == "float32"


def _same(left: Any, right: Any) -> bool:
    if pd.isna(left) or pd.isna(right):
        return bool(pd.isna(left) and pd.isna(right))
    return bool(left == right)


def test_subjects_without_a_score_a_render_or_a_place_in_the_source_keep_their_row(
    run: Path, engine: Engine, uids: list[str], tmp_path: Path
) -> None:
    before = subjects_frame(run)
    # sub-000 lost its score row, sub-001 its render, sub-002 failed to render and is gone
    scores = read_table(table_path(run, "scores"), SCHEMAS["scores"])
    write_table(
        table_path(run, "scores"),
        scores[scores["subject_uid"] != uids[0]],
        SCHEMAS["scores"],
        sort_by=KEYS["scores"],
    )
    private(run, "DELETE FROM renders WHERE subject_uid = ?", (uids[1],))
    private(
        run,
        "UPDATE renders SET status = 'failed', reason = 'missing surface', render_key = NULL "
        "WHERE subject_uid = ?",
        (uids[2],),
    )
    private(run, "UPDATE subjects SET present = 0 WHERE subject_uid = ?", (uids[2],))

    frame = subjects_frame(run)
    assert list(frame["subject_uid"]) == list(before["subject_uid"])
    # the types of the columns do not depend on which rows have a value
    assert frame.dtypes.to_dict() == before.dtypes.to_dict()
    frame = by_uid(frame)
    with PublicCatalog.open(run) as public:
        catalog = set(public.subjects().columns)
    score_columns = [name for name in SCHEMAS["scores"].names if name not in catalog]
    assert frame.loc[uids[0], score_columns].isna().all()
    assert frame.loc[uids[0], "reference_id"] is None and frame.loc[uids[0], "flags"] is None
    assert frame.loc[uids[1], score_columns].notna().sum() > 20
    assert frame["render_status"].to_dict() == {uids[0]: "done", uids[1]: None, uids[2]: "failed"}
    assert frame["render_reason"].to_dict() == {
        uids[0]: None,
        uids[1]: None,
        uids[2]: "missing surface",
    }
    assert frame["present"].to_dict() == {uids[0]: True, uids[1]: True, uids[2]: False}

    export_run(run, tmp_path / "parquet", fmt="parquet")
    export_run(run)
    written = pq.read_table(tmp_path / "parquet" / "subjects.parquet")
    assert written.schema.field("composite").type == pa.float32()
    assert written.schema.field("primary_rank").type == pa.int32()
    assert written.column("subject_uid").to_pylist() == sorted(uids)
    assert written.column("primary").to_pylist() == [
        None if uid == uids[0] else True for uid in sorted(uids)
    ]
    assert read_csv(run / "export" / "subjects.csv")["primary"].tolist() == [
        "" if uid == uids[0] else "True" for uid in sorted(uids)
    ]


def test_final_labels_methods_and_counts_of_a_subject(
    run: Path, engine: Engine, uids: list[str]
) -> None:
    rate(engine, run, uids[0], marks=[click("m1"), click("m2")], other=True, note="see C")
    rate(engine, run, uids[0], rater="second", values=NONE, marks=[click("m1", "X")])
    rate(engine, run, uids[1], values=NONE)
    label(engine, uids[0], "consensus", A=0, B=0, C=1)
    label(engine, uids[1], "single", A=0, B=0)

    frame = by_uid(subjects_frame(run))
    rated = frame.loc[uids[0], list(SUBJECT_RATING_COLUMNS)].to_dict()
    assert rated == {
        "final_A": 0,
        "final_B": 0,
        "final_C": 1,
        "final_method": "consensus",
        "unratable_reason": None,
        "other": True,
        "n_ratings": 2,
        "n_marks": 3,
    }
    partly = frame.loc[uids[1]]
    assert (partly["final_A"], partly["final_B"], partly["final_method"]) == (0, 0, "single")
    assert partly["final_C"] is pd.NA  # a protocol without criterion C leaves it open
    assert (partly["other"], partly["n_ratings"], partly["n_marks"]) == (False, 1, 0)
    nobody = frame.loc[uids[2]]
    assert all(nobody[name] is pd.NA for name in ("final_A", "final_B", "final_C", "other"))
    assert nobody["final_method"] is None and nobody["unratable_reason"] is None
    assert (nobody["n_ratings"], nobody["n_marks"]) == (0, 0)


@pytest.mark.parametrize(
    ("methods", "expected"),
    [
        (("single", "single", "single"), "single"),
        (("consensus", "adjudicated", "consensus"), "adjudicated"),
        (("single", "consensus", "single"), "consensus"),
    ],
)
def test_the_method_of_a_subject_is_the_strongest_of_its_labels(
    run: Path, engine: Engine, uids: list[str], methods: tuple[str, ...], expected: str
) -> None:
    assert set(FINAL_METHODS) == {"adjudicated", "consensus", "single"}
    for criterion, method in zip("ABC", methods, strict=True):
        label(engine, uids[0], method, **{criterion: 1})
    assert by_uid(subjects_frame(run)).loc[uids[0], "final_method"] == expected


def test_an_unratable_subject_has_the_reason_in_words(
    run: Path, engine: Engine, uids: list[str], shipped_protocol: Protocol
) -> None:
    label(engine, uids[0], unratable=shipped_protocol.unratable_index("missing surface"))
    label(engine, uids[1], unratable=shipped_protocol.unratable_index("image too poor to judge"))
    frame = by_uid(subjects_frame(run))
    assert frame["unratable_reason"].to_dict() == {
        uids[0]: "missing surface",
        uids[1]: "image too poor to judge",
        uids[2]: None,
    }
    assert frame.loc[uids[0], "final_method"] == "single"
    assert frame.loc[uids[0], ["final_A", "final_B", "final_C"]].isna().all()


@pytest.mark.parametrize("value", [4, -1, None])
def test_an_unratable_label_that_the_protocol_cannot_name_stops_the_export(
    run: Path, engine: Engine, uids: list[str], value: int | None
) -> None:
    export_run(run)
    before = listing(run / "export")
    label(engine, uids[0], unratable=value)
    with pytest.raises(ConfigError, match="unratable reason") as raised:
        export_run(run)
    assert raised.value.reason == "invalid" and uids[0] in raised.value.detail
    assert "reasons 0 to 3" in raised.value.detail
    # refused before anything was written
    assert listing(run / "export") == before


def unratable(engine: Engine, run: Path, uid: str, reason: str, protocol: Protocol) -> None:
    """Rate a subject unratable and label it as the server does at a submission."""
    rate(engine, run, uid, unratable=reason)
    with Session(engine) as session:
        record_single(session, uid, protocol)
        session.commit()


@pytest.mark.parametrize("edit", ["reordered", "removed"])
def test_an_unratable_label_keeps_its_reason_when_the_protocol_is_edited(
    run: Path,
    engine: Engine,
    uids: list[str],
    shipped_protocol: Protocol,
    write_reasons: Callable[[Path, Sequence[str]], None],
    edit: str,
) -> None:
    unratable(engine, run, uids[0], "image too poor to judge", shipped_protocol)
    with Session(engine) as session:
        # a render that failed, labelled when the server started
        plan = QueuePlan("local", (uids[1],), {}, {uids[1]: "missing surface"}, 0, 0)
        sync_unratable(session, plan, shipped_protocol)
        session.commit()
    reasons = list(shipped_protocol.unratable_reasons)
    write_reasons(run, list(reversed(reasons)) if edit == "reordered" else reasons[:3])

    expected = {uids[0]: "image too poor to judge", uids[1]: "missing surface", uids[2]: None}
    assert by_uid(subjects_frame(run))["unratable_reason"].to_dict() == expected
    written = {path.name: read_csv(path) for path in export_run(run)[:-1]}
    subjects = written["subjects.csv"].set_index("subject_uid")
    assert subjects.loc[uids[0], "unratable_reason"] == "image too poor to judge"
    assert subjects.loc[uids[1], "unratable_reason"] == "missing surface"
    # the two tables of one export say the same of the rating
    ratings = written["ratings.csv"].set_index("subject_uid")
    assert ratings.loc[uids[0], "unratable_reason"] == "image too poor to judge"


def test_the_reasons_a_database_keeps_need_no_protocol(
    run: Path, engine: Engine, uids: list[str], shipped_protocol: Protocol
) -> None:
    unratable(engine, run, uids[0], "wrong orientation", shipped_protocol)
    (run / PROTOCOL_FILE).unlink()
    assert by_uid(subjects_frame(run)).loc[uids[0], "unratable_reason"] == "wrong orientation"


def test_kept_reasons_that_are_no_list_stop_the_export(
    run: Path, engine: Engine, uids: list[str], shipped_protocol: Protocol
) -> None:
    unratable(engine, run, uids[0], "wrong orientation", shipped_protocol)
    with Session(engine) as session:
        kept = session.get(AppMeta, LABEL_REASONS_KEY)
        assert kept is not None
        kept.value = "render failure"
        session.add(kept)
        session.commit()
    with pytest.raises(ConfigError, match=LABEL_REASONS_KEY) as raised:
        subjects_frame(run)
    assert raised.value.reason == "invalid"
    with Session(engine) as session:
        session.add(AppMeta(key="other", value="[]"))  # a row more is nothing to the export
        kept = session.get(AppMeta, LABEL_REASONS_KEY)
        assert kept is not None
        kept.value = json.dumps(["a", "wrong orientation"])
        session.add(kept)
        session.commit()
    # the label is at position 1 of the shipped list, and the kept list is what names it
    assert by_uid(subjects_frame(run)).loc[uids[0], "unratable_reason"] == "wrong orientation"
    with Session(engine) as session:
        kept = session.get(AppMeta, LABEL_REASONS_KEY)
        assert kept is not None
        kept.value = json.dumps(["a"])
        session.add(kept)
        session.commit()
    with pytest.raises(ConfigError, match="reasons 0 to 0") as raised:
        subjects_frame(run)
    assert APP_DB_FILE in raised.value.detail and PROTOCOL_FILE not in raised.value.detail


def test_unratable_labels_need_the_protocol_and_nothing_else_does(
    run: Path, engine: Engine, uids: list[str]
) -> None:
    rate(engine, run, uids[0])
    label(engine, uids[0], A=2, B=0, C=1)
    (run / PROTOCOL_FILE).unlink()
    assert by_uid(subjects_frame(run)).loc[uids[0], "final_A"] == 2
    assert manifest_of(export_run(run)[-1].parent)["protocol"] is None
    label(engine, uids[1], unratable=2)
    with pytest.raises(ConfigError) as raised:
        subjects_frame(run)
    assert raised.value.reason == "missing"


# --- a run without ratings ----------------------------------------------------------------------


def test_a_run_without_a_ratings_database_exports_with_empty_rating_columns(
    run_copy: Path, uids: list[str]
) -> None:
    assert not (run_copy / APP_DB_FILE).exists()
    frame = subjects_frame(run_copy)
    assert len(frame) == 3
    assert frame[list(SUBJECT_RATING_COLUMNS)].isna().all().all()
    ratings = ratings_frame(run_copy)
    assert list(ratings.columns) == list(RATINGS_COLUMNS) and ratings.empty
    marks = marks_frame(run_copy)
    assert list(marks.columns) == list(MARKS_COLUMNS) and marks.empty

    paths = export_run(run_copy)
    assert not (run_copy / APP_DB_FILE).exists()  # reading made no database
    rows = {path.stem: len(read_csv(path)) for path in paths[:-1]}
    assert rows == {
        "subjects": 3,
        "scores": 3,
        "selection": 3,
        "ratings": 0,
        "marks": 0,
        "final_marks": 0,
        "clusters": len(clusters_with_native(run_copy)),
        "datasets": 1,
        "behaviour": 0,
        "skips": 0,
    }
    assert rows["clusters"] > 0
    manifest = manifest_of(run_copy / "export")
    assert manifest["protocol"] is None and manifest["calibration"] is None


def test_a_run_with_a_database_and_no_rating_counts_zero(
    run: Path, engine: Engine, uids: list[str]
) -> None:
    frame = subjects_frame(run)
    assert list(frame["n_ratings"]) == [0, 0, 0] and list(frame["n_marks"]) == [0, 0, 0]
    names = [name for name in SUBJECT_RATING_COLUMNS if not name.startswith("n_")]
    assert frame[names].isna().all().all()
    # the same types as without a database, so the two cases write the same schema
    (run / APP_DB_FILE).rename(run / "aside.sqlite")
    assert subjects_frame(run).dtypes.to_dict() == frame.dtypes.to_dict()


def test_the_types_of_a_table_do_not_depend_on_its_rows(
    run: Path, engine: Engine, uids: list[str], tmp_path: Path
) -> None:
    export_run(run, tmp_path / "empty", fmt="parquet")
    assignment = rate(engine, run, uids[0], marks=[click("m1")], other=True, note="x")
    log(engine, assignment, ("play", {"frame": 0}))
    label(engine, uids[0], A=2, B=0, C=1)
    with Session(engine) as session:
        session.add(FinalMark(subject_uid=uids[0], mark_class="A", tkr_x=0.0, tkr_y=0.0, tkr_z=0.0))
        session.commit()
    export_run(run, tmp_path / "full", fmt="parquet")
    for name in EXPORT_TABLES:
        empty = pq.read_schema(tmp_path / "empty" / f"{name}.parquet")
        full = pq.read_schema(tmp_path / "full" / f"{name}.parquet")
        assert empty.equals(full), name
        assert full.metadata is None, name
    ratings = pq.read_schema(tmp_path / "full" / "ratings.parquet")
    kinds = {name: str(ratings.field(name).type) for name in ("A", "other", "note", "revision")}
    assert kinds == {"A": "int64", "other": "bool", "note": "string", "revision": "int64"}
    # a mark: text, whole numbers for what counts or indexes, and measurements
    marks = pq.read_schema(tmp_path / "full" / "marks.parquet")
    text = {
        "subject_uid",
        "rater",
        "class",
        "panel",
        "mode",
        "hemi",
        "vertex_surface",
        "created_ts",
    }
    whole = {"assignment_id", "mark_id", "submission_id", "slice", "frame"} | {
        "pial_vertex",
        "white_vertex",
        "fsavg6_vertex",
    }
    for field in marks:
        expected = "string" if field.name in text else "int64" if field.name in whole else "double"
        assert str(field.type) == expected, field.name
    final = pq.read_schema(tmp_path / "full" / "final_marks.parquet")
    assert [str(final.field(name).type) for name in FINAL_MARKS_COLUMNS] == [
        "int64",
        "string",
        "string",
        "double",
        "double",
        "double",
        "string",
        "int64",
        "int64",
        "string",
    ]


# --- scores, selection, clusters, datasets --------------------------------------------------------


def test_scores_and_clusters_are_the_tables_of_the_run(run: Path, tmp_path: Path) -> None:
    export_run(run, tmp_path / "out", fmt="parquet")
    # with the schema of the run's table, so what reads the one reads the other
    scores = read_table(tmp_path / "out" / "scores.parquet", SCHEMAS["scores"])
    expected = read_table(table_path(run, "scores"), SCHEMAS["scores"])
    pd.testing.assert_frame_equal(scores, expected)

    clusters = read_table(tmp_path / "out" / "clusters.parquet")
    joined = clusters_with_native(run)
    assert len(clusters) == len(joined) > 0
    key = list(KEYS["clusters"])
    assert clusters[key].to_dict("list") == joined[key].to_dict("list")
    placed = joined["native_vertex"].notna()
    assert 0 < placed.sum() < len(joined)
    assert clusters["native_vertex"].notna().tolist() == placed.tolist()
    assert clusters.loc[placed.to_numpy(), "native_vertex"].tolist() == [
        int(vertex) for vertex in joined.loc[placed, "native_vertex"]
    ]
    assert str(
        pq.read_schema(tmp_path / "out" / "clusters.parquet").field("native_vertex").type
    ) == ("int32")
    for name in ("tkr_x", "area_mm2", "peak_z"):
        np.testing.assert_array_equal(clusters[name].to_numpy(), joined[name].to_numpy())
    assert clusters.loc[~placed.to_numpy(), "template_id"].isna().all()


def test_a_subject_drawn_by_two_samples_has_two_selection_rows(run: Path, uids: list[str]) -> None:
    rows = pd.DataFrame(
        {
            "subject_uid": [uids[1], uids[1], uids[0]],
            "reason": ["prevalence", "detectability", "detectability"],
            "stratum": ["D-d44e977f", "decile-10", "decile-03"],
            "inclusion_prob": [0.25, 0.8125, 0.0625],
            "seed": [11, 12, 12],
        }
    )
    with PublicCatalog.open(run, readonly=False) as public:
        public.replace_selection(rows, ["prevalence", "detectability"])
    export_run(run)
    selection = read_csv(run / "export" / "selection.csv")
    twice = selection[selection["subject_uid"] == uids[1]]
    assert twice[["reason", "stratum", "inclusion_prob", "seed"]].to_dict("records") == [
        {"reason": "all", "stratum": "all", "inclusion_prob": "1.0", "seed": "0"},
        {
            "reason": "detectability",
            "stratum": "decile-10",
            "inclusion_prob": "0.8125",
            "seed": "12",
        },
        {"reason": "prevalence", "stratum": "D-d44e977f", "inclusion_prob": "0.25", "seed": "11"},
    ]
    assert len(selection) == 6
    assert manifest_of(run / "export")["files"]["selection.csv"]["rows"] == 6


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ((None, None, None), ""),
        ((3.0, None, None), "3.0"),
        ((2.89, 7.0, 7.0), "7.0"),
        ((3.0, 1.5, None), "1.5"),  # a tie: the lower one
        ((7.0, 3.0, 3.0), "3.0"),
    ],
)
def test_a_dataset_has_its_subject_count_and_its_most_frequent_field_strength(
    run_copy: Path, uids: list[str], fields: tuple[float | None, ...], expected: str
) -> None:
    for uid, field in zip(uids, fields, strict=True):
        private(
            run_copy, "UPDATE subjects SET field_strength = ? WHERE subject_uid = ?", (field, uid)
        )
    export_run(run_copy)
    (row,) = read_csv(run_copy / "export" / "datasets.csv").to_dict("records")
    with PublicCatalog.open(run_copy) as public:
        (pseudonym,) = public.datasets()["pseudonym"]
    assert row == {
        "dataset": pseudonym,
        "source": "local",
        "pipeline": "freesurfer",
        "pipeline_version": "",
        "n_subjects": "3",
        "modal_field_strength": expected,
    }


def test_a_subject_that_left_the_source_does_not_vote_on_the_field_strength(
    run_copy: Path, uids: list[str]
) -> None:
    for uid, field in zip(uids, (7.0, 7.0, 3.0), strict=True):
        private(
            run_copy, "UPDATE subjects SET field_strength = ? WHERE subject_uid = ?", (field, uid)
        )
    private(run_copy, "UPDATE subjects SET present = 0 WHERE subject_uid IN (?, ?)", uids[:2])
    export_run(run_copy)
    assert list(read_csv(run_copy / "export" / "datasets.csv")["modal_field_strength"]) == ["3.0"]


# --- behaviour ------------------------------------------------------------------------------------


def test_behaviour_sums_the_events_of_each_completed_assignment(
    run: Path, engine: Engine, uids: list[str]
) -> None:
    busy = rate(engine, run, uids[0], ms=20500)
    quiet = rate(engine, run, uids[1], rater="second", ms=900, now=T2)
    still_open = assign(engine, run, uids[2])
    log(
        engine,
        busy,
        ("load", {"n_tiles": 40}),
        ("play", {"frame": 0}),
        ("frames", {"mode": "sync", "visits": [[0, 80], [1, 80], [2, 85], [1, 400]]}),
        ("pause", {"frame": 1}),
        ("mode", {"mode": "seq"}),
        # frame 1 of the other mode is another frame; frame 2 of this one comes twice
        ("frames", {"mode": "seq", "visits": [[1, 80], [2, 0], [3, 90]]}),
        ("frames", {"mode": "sync", "visits": [[2, 30]]}),
        ("play", {"frame": 3}),
        ("pause", {"frame": 9}),
        ("play", {"frame": 9}),
        ("mode", {"mode": "sync"}),
        ("back", {}),
        ("zoom", {"zoom": 2}),
        ("something new", {"visits": [[7, 1]]}),
        # what a page may send in place of the documented form counts nothing
        ("frames", "not json"),
        ("frames", "[1, 2]"),
        ("frames", {"mode": "sync"}),
        ("frames", {"mode": "sync", "visits": "all"}),
        ("frames", {"mode": "odd", "visits": [None, [], "x", [True, 5], [2.5, 5], ["4", 1]]}),
    )
    log(engine, still_open, ("play", {"frame": 0}), ("back", {}))
    log(engine, None, ("config_change", {"setting": "blind"}), ("play", {}))
    assert {"play", "pause", "mode", "back", "frames"} <= set(EVENT_TYPES)

    export_run(run)
    rows = read_csv(run / "export" / "behaviour.csv").to_dict("records")
    assert rows == [
        {
            "assignment_id": str(busy),
            "subject_uid": uids[0],
            "rater": "local",
            "ms_on_subject": "20500",
            "n_frames_viewed": "6",
            "n_plays": "3",
            "n_pauses": "2",
            "n_mode_switches": "2",
            "n_back": "1",
            "n_skips": "0",
            "ms_skipped_visits": "0",
            "ms_revised_visits": "0",
        },
        {
            "assignment_id": str(quiet),
            "subject_uid": uids[1],
            "rater": "second",
            "ms_on_subject": "900",
            "n_frames_viewed": "0",
            "n_plays": "0",
            "n_pauses": "0",
            "n_mode_switches": "0",
            "n_back": "0",
            "n_skips": "0",
            "ms_skipped_visits": "0",
            "ms_revised_visits": "0",
        },
    ]


def skip(engine: Engine, assignment_id: int, *, ms: int, client_ts: str, now: str) -> bool:
    """Record a skip of an open assignment as the server does."""
    with Session(engine) as session:
        assignment = session.get(Assignment, assignment_id)
        assert assignment is not None
        body = SkipIn(assignment_id=assignment_id, client_ts=client_ts, ms_on_subject=ms)
        recorded = record_skip(session, assignment, body, now=now)
        session.commit()
        return recorded


def test_every_skip_is_a_row_and_the_skipped_visits_are_counted_with_the_rating(
    run: Path, engine: Engine, uids: list[str]
) -> None:
    """A rater skips one subject twice and rates it, skips a second and leaves it, and
    rates a third at once. The skips table has the three skips, whoever was rated in the
    end; the behaviour table says what came before each rating."""
    twice = assign(engine, run, uids[0])
    assert skip(engine, twice, ms=4000, client_ts="2026-11-13T09:00:01.000Z", now=T1)
    assert skip(engine, twice, ms=2500, client_ts="2026-11-13T09:00:09.000Z", now=T2)
    # the request of the second skip arrives again: it is one skip
    assert not skip(engine, twice, ms=2500, client_ts="2026-11-13T09:00:09.000Z", now=T3)
    left = assign(engine, run, uids[1], rater="second")
    assert skip(engine, left, ms=300, client_ts="2026-11-13T09:01:00.000Z", now=T2)
    submit(engine, twice, ms=7000, now=T3)
    at_once = rate(engine, run, uids[2], ms=1200, now=T3)
    # a skip event of a form the server does not write: counted, its time unknown
    spoiled = assign(engine, run, uids[1])
    log(engine, spoiled, ("skip", "not json"))
    submit(engine, spoiled, ms=10, now=T3)

    key = {uid: render_key(run, uid) for uid in uids}
    export_run(run)
    skips = read_csv(run / "export" / "skips.csv").to_dict("records")
    assert skips == [
        {
            "assignment_id": str(twice),
            "n": "1",
            "subject_uid": uids[0],
            "rater": "local",
            "kind": "main",
            "render_key": key[uids[0]],
            "client_ts": "2026-11-13T09:00:01.000Z",
            "server_ts": T1,
            "ms_on_subject": "4000",
            "rated": "True",
        },
        {
            "assignment_id": str(twice),
            "n": "2",
            "subject_uid": uids[0],
            "rater": "local",
            "kind": "main",
            "render_key": key[uids[0]],
            "client_ts": "2026-11-13T09:00:09.000Z",
            "server_ts": T2,
            "ms_on_subject": "2500",
            "rated": "True",
        },
        {
            "assignment_id": str(left),
            "n": "1",
            "subject_uid": uids[1],
            "rater": "second",
            "kind": "main",
            "render_key": key[uids[1]],
            "client_ts": "2026-11-13T09:01:00.000Z",
            "server_ts": T2,
            "ms_on_subject": "300",
            "rated": "False",
        },
        {
            "assignment_id": str(spoiled),
            "n": "1",
            "subject_uid": uids[1],
            "rater": "local",
            "kind": "main",
            "render_key": key[uids[1]],
            "client_ts": T1,
            "server_ts": "",
            "ms_on_subject": "",
            "rated": "True",
        },
    ]
    behaviour = by_assignment(read_csv(run / "export" / "behaviour.csv"))
    assert list(behaviour.index) == [str(twice), str(at_once), str(spoiled)]
    # the time of a rating is that of its last visit; the earlier visits are beside it
    assert behaviour.loc[
        str(twice), ["ms_on_subject", "n_skips", "ms_skipped_visits"]
    ].tolist() == [
        "7000",
        "2",
        "6500",
    ]
    assert behaviour.loc[str(at_once), ["n_skips", "ms_skipped_visits"]].tolist() == ["0", "0"]
    # the count is the assignment's own; the time of a visit nobody recorded is not a 0
    assert behaviour.loc[str(spoiled), ["n_skips", "ms_skipped_visits"]].tolist() == ["0", ""]
    # a skip is no rating: the subject that was only skipped has none from that rater
    ratings = read_csv(run / "export" / "ratings.csv")
    assert str(left) not in set(ratings["assignment_id"])

    export_run(run, fmt="parquet")
    typed = pd.read_parquet(run / "export" / "skips.parquet")
    assert typed["n"].tolist() == [1, 2, 1, 1] and typed["rated"].tolist() == [
        True,
        True,
        False,
        True,
    ]
    assert typed["ms_on_subject"].tolist()[:3] == [4000, 2500, 300] and pd.isna(
        typed["ms_on_subject"].iloc[3]
    )


def by_assignment(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.set_index("assignment_id")


def test_the_export_reads_the_skip_events_the_server_writes() -> None:
    assert tables._SKIP_EVENT == SKIP_EVENT


def test_a_run_without_ratings_has_an_empty_skips_table(run: Path) -> None:
    export_run(run)
    assert read_csv(run / "export" / "skips.csv").to_dict("records") == []
    assert list(read_csv(run / "export" / "skips.csv").columns) == list(SKIPS_COLUMNS)


# --- the two formats ------------------------------------------------------------------------------


def _parsed(text: str, kind: pa.DataType) -> object:
    if text == "":
        return None
    if pa.types.is_boolean(kind):
        return {"True": True, "False": False}[text]
    if pa.types.is_integer(kind):
        return int(text)
    if pa.types.is_float32(kind):
        return float(np.float32(text))
    if pa.types.is_floating(kind):
        return float(text)
    return text


def assert_formats_agree(csv_dir: Path, parquet_dir: Path) -> int:
    """Every value of every table is the same in the two formats, row by row; a text that
    is empty and one that is missing are one thing in a CSV file. Returns the number of
    values that were compared and are not missing."""
    compared = 0
    for name in EXPORT_TABLES:
        table = pq.read_table(parquet_dir / f"{name}.parquet")
        text = read_csv(csv_dir / f"{name}.csv")
        assert list(text.columns) == table.schema.names, name
        assert len(text) == table.num_rows, name
        for field in table.schema:
            typed = [
                None if value == "" else value for value in table.column(field.name).to_pylist()
            ]
            parsed = [_parsed(value, field.type) for value in text[field.name]]
            assert parsed == typed, (name, field.name)
            compared += sum(value is not None for value in typed)
    return compared


def fill(run: Path, engine: Engine, uids: list[str]) -> None:
    """Ratings of two raters with marks, a revision, an unratable rating, a skipped
    subject, events, final labels and final marks: a row in every table."""
    first = rate(engine, run, uids[0], marks=[click("m1"), click("m2", "C")], note='a, "b"\nc')
    submit(engine, first, marks=[click("m3", px=1 / 3)], found=[resolved(pial_dist_mm=0.1 + 0.2)])
    second = rate(engine, run, uids[0], rater="second", values=NONE, other=True, note="x", now=T2)
    rate(engine, run, uids[1], unratable="image too poor to judge", now=T3)
    log(engine, first, ("play", {}), ("frames", {"mode": "sync", "visits": [[0, 5], [1, 5]]}))
    log(engine, second, ("back", {}))
    waiting = assign(engine, run, uids[2], rater="second")
    skip(engine, waiting, ms=800, client_ts="2026-11-13T09:30:00.000Z", now=T2)
    label(engine, uids[0], "consensus", A=0, B=0, C=1)
    label(engine, uids[1], unratable=3)
    with Session(engine) as session:
        session.add(
            FinalMark(
                subject_uid=uids[0],
                mark_class="C",
                tkr_x=1 / 3,
                tkr_y=-2.5,
                tkr_z=1e-7,
                hemi="lh",
                pial_vertex=3,
                source_mark_ids="[1, 2]",
            )
        )
        session.commit()


def test_csv_and_parquet_hold_the_same_values(
    run: Path, engine: Engine, uids: list[str], tmp_path: Path
) -> None:
    fill(run, engine, uids)
    private(
        run,
        "UPDATE subjects SET field_strength = 2.89, age = 31.25 WHERE subject_uid = ?",
        uids[:1],
    )
    export_run(run, tmp_path / "csv", fmt="csv")
    export_run(run, tmp_path / "parquet", fmt="parquet")
    assert assert_formats_agree(tmp_path / "csv", tmp_path / "parquet") > 1000
    for name in EXPORT_TABLES:
        assert pq.read_table(tmp_path / "parquet" / f"{name}.parquet").num_rows > 0, name
    # a float that has no short decimal form is written so that it reads back as it was
    marks = read_csv(tmp_path / "csv" / "marks.csv")
    assert float(marks["px"][0]) == 1 / 3 and float(marks["pial_dist_mm"][0]) == 0.1 + 0.2
    # and a vertex is a whole number in the text too
    assert (marks["pial_vertex"][0], marks["fsavg6_vertex"][0]) == ("211", "12950")


def test_every_table_is_sorted_by_its_key(run: Path, engine: Engine, uids: list[str]) -> None:
    fill(run, engine, uids)
    export_run(run)
    keys = {
        "subjects": ["subject_uid"],
        "scores": ["subject_uid", "reference_id"],
        "selection": ["subject_uid", "reason"],
        "clusters": ["subject_uid", "reference_id", "hemi", "region", "rank"],
        "datasets": ["dataset", "source", "pipeline"],
    }
    for name, key in keys.items():
        frame = pd.read_csv(run / "export" / f"{name}.csv")
        rows = list(frame[key].itertuples(index=False, name=None))
        assert rows == sorted(rows) and len(set(rows)) == len(rows), name
    for name, key in (
        ("ratings", "assignment_id"),
        ("marks", "mark_id"),
        ("behaviour", "assignment_id"),
    ):
        ids = [int(value) for value in read_csv(run / "export" / f"{name}.csv")[key]]
        assert ids == sorted(set(ids)) and ids, name


# --- the manifest and the directory ---------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_the_manifest_names_the_run_and_every_file_with_its_hash(
    run: Path, engine: Engine, uids: list[str], shipped_protocol: Protocol, fmt: Format
) -> None:
    fill(run, engine, uids)
    paths = export_run(run, fmt=fmt)
    assert paths[-1] == run / "export" / MANIFEST
    manifest = manifest_of(run / "export")
    assert set(manifest["files"]) == {path.name for path in paths[:-1]}
    assert sorted(path.name for path in (run / "export").iterdir()) == sorted(
        [*manifest["files"], MANIFEST]
    )
    for path in paths[:-1]:
        entry = manifest["files"][path.name]
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest(), path.name
        rows = len(read_csv(path)) if fmt == "csv" else pq.read_table(path).num_rows
        assert entry["rows"] == rows, path.name
    assert manifest["files"][f"ratings.{fmt}"]["rows"] == 3
    assert manifest["files"][f"marks.{fmt}"]["rows"] == 1

    with PublicCatalog.open(run) as public:
        run_id, fingerprint = public.run_id(), public.meta("key_fingerprint")
        template_id = public.meta("template_id")
        render_hashes = set(public.renders()["config_hash"])
    scores = read_table(table_path(run, "scores"), SCHEMAS["scores"])
    del manifest["files"]
    stage_hashes = manifest.pop("stage_hashes")
    app = load_config(run).app
    assert (app.blind, app.repeat_rate) == (False, 0.0)
    assert manifest.pop("queue") == {
        "blind": False,
        "queue_seed": app.queue_seed,
        "repeat_rate": 0.0,
        "repeat_gap": app.repeat_gap,
    }
    assert manifest == {
        "manifest_schema_version": 1,
        "run_id": run_id,
        "mode": "local",
        "profile": "analysis",
        "format": fmt,
        "key_fingerprint": fingerprint,
        "surfcensus_version": __version__,
        "template_id": template_id,
        "reference_ids": sorted(set(scores["reference_id"])),
        "protocol": {
            "version": shipped_protocol.version,
            "sha256": hashlib.sha256((run / PROTOCOL_FILE).read_bytes()).hexdigest(),
        },
        "calibration": None,
    }
    assert template_id is not None and len(manifest["reference_ids"]) == 1
    # the identities the stages recorded when they ran
    assert sorted(stage_hashes) == sorted(
        ["inventory", "regional", "fsaverage", "reference", "native", "score", "render"]
    )
    assert {stage_hashes["score"]} == set(scores["score_config_hash"])
    assert {stage_hashes["render"]} == render_hashes


def test_the_manifest_names_the_calibration_set_of_the_run(run: Path, uids: list[str]) -> None:
    calibration = CalibrationSet(
        version="pilot-1",
        subjects=(CalibrationSubject(subject_uid=uids[0], values={"A": 2, "B": 0, "C": 0}),),
    )
    path = write_calibration(run, calibration)
    export_run(run)
    assert manifest_of(run / "export")["calibration"] == {
        "version": "pilot-1",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_a_run_without_a_reference_has_no_reference_stage_in_its_manifest(run_copy: Path) -> None:
    scores = read_table(table_path(run_copy, "scores"), SCHEMAS["scores"])
    scores["reference_id"] = "none"
    write_table(table_path(run_copy, "scores"), scores, SCHEMAS["scores"], sort_by=KEYS["scores"])
    export_run(run_copy)
    manifest = manifest_of(run_copy / "export")
    assert manifest["reference_ids"] == []
    assert sorted(manifest["stage_hashes"]) == [
        "fsaverage",
        "inventory",
        "regional",
        "render",
        "score",
    ]


def test_an_export_of_an_unchanged_run_gives_the_same_bytes(
    run: Path, engine: Engine, uids: list[str], tmp_path: Path
) -> None:
    fill(run, engine, uids)
    for fmt in FORMATS:
        export_run(run, tmp_path / "one", fmt=fmt)
        first = listing(tmp_path / "one")
        export_run(run, tmp_path / "one", fmt=fmt)
        assert listing(tmp_path / "one") == first
        export_run(run, tmp_path / fmt, fmt=fmt)
        assert listing(tmp_path / fmt) == first
    # nothing of the place or the moment is in the files
    text = b"".join(listing(tmp_path / "csv").values())
    assert str(tmp_path).encode() not in text and str(run).encode() not in text
    # and the export left the run as it was, but for its own directory
    assert not (run / "export").exists() and not (run / "release").exists()


def test_the_directory_is_the_one_given_or_that_of_the_profile(run: Path, tmp_path: Path) -> None:
    assert EXPORT_DIRS == {"analysis": "export", "release": "release"}
    assert {path.parent for path in export_run(run)} == {run / "export"}
    nested = tmp_path / "a" / "b"
    assert {path.parent for path in export_run(str(run), str(nested))} == {nested}
    assert manifest_of(nested)["files"].keys() == manifest_of(run / "export")["files"].keys()
    assert not list(nested.glob(".*")) and not list((run / "export").glob(".*"))


def test_an_export_in_the_other_format_replaces_the_files_of_the_earlier_one(
    run: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    out = tmp_path / "out"
    export_run(run, out, fmt="csv")
    # what no manifest lists is not the export's, whatever its name
    (out / "notes.csv").write_text("mine\n")
    (out / "ratings.parquet").write_text("mine, under a name of the export\n")
    (out / "subjects.csv.bak").write_text("mine\n")
    with caplog.at_level(logging.INFO, logger="surfcensus"):
        export_run(run, out, fmt="parquet")
    assert "removed subjects.csv of the earlier export" in caplog.text
    names = sorted(path.name for path in out.iterdir())
    assert names == sorted(
        [*(f"{name}.parquet" for name in EXPORT_TABLES), MANIFEST, "notes.csv", "subjects.csv.bak"]
    )
    assert (out / "notes.csv").read_text() == "mine\n"
    assert pq.read_table(out / "ratings.parquet").num_rows == 0  # written over, as any export
    assert set(manifest_of(out)["files"]) == {f"{name}.parquet" for name in EXPORT_TABLES}


def test_a_manifest_that_lists_other_names_takes_no_file_with_it(run: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.csv").write_text("mine\n")
    # a name of the export in the other format, which no manifest lists
    (out / "scores.parquet").write_text("mine\n")
    (out / "sub").mkdir()
    (out / "marks.parquet").mkdir()
    listed = {
        "keep.csv": {},
        "../outside.csv": {},
        "sub": {},
        "marks.parquet": {},
        "x/ratings.csv": {},
    }
    (tmp_path / "outside.csv").write_text("mine\n")
    (out / MANIFEST).write_text(json.dumps({"files": listed}))
    export_run(run, out)
    assert (out / "keep.csv").read_text() == "mine\n"
    assert (out / "scores.parquet").read_text() == "mine\n"
    assert (tmp_path / "outside.csv").read_text() == "mine\n"
    assert (out / "sub").is_dir() and (out / "marks.parquet").is_dir()
    assert set(manifest_of(out)["files"]) == {f"{name}.csv" for name in EXPORT_TABLES}
    for text in ("{", "[]", '{"files": 3}', '{"files": [1, 2]}', ""):
        (out / MANIFEST).write_text(text)
        export_run(run, out)
        assert (out / "keep.csv").exists()
        assert len(manifest_of(out)["files"]) == len(EXPORT_TABLES) == 10


def test_an_export_that_is_interrupted_leaves_no_manifest(
    run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    export_run(run, out)
    written = []

    def full_disk(path: Path, frame: pd.DataFrame, schema: pa.Schema | None) -> None:
        if len(written) == 3:
            raise OSError(28, "No space left on device")
        written.append(path.name)
        path.write_text("partly\n")

    monkeypatch.setitem(tables._WRITERS, "csv", full_disk)
    with pytest.raises(OSError, match="No space left"):
        export_run(run, out)
    assert written == ["subjects.csv", "scores.csv", "selection.csv"]
    # the files no longer fit a manifest, so there is none to say they do
    assert not (out / MANIFEST).exists()
    monkeypatch.undo()
    export_run(run, out)
    manifest = manifest_of(out)
    assert (
        manifest["files"]["subjects.csv"]["sha256"]
        == hashlib.sha256((out / "subjects.csv").read_bytes()).hexdigest()
    )


def test_a_format_or_a_profile_that_is_none_is_refused(run: Path) -> None:
    # in a fresh interpreter: here the type checks of the tests would refuse the call first
    code = (
        "from surfcensus.export.tables import export_run\n"
        "for wrong in ({'fmt': 'xlsx'}, {'profile': 'public'}):\n"
        "    try:\n"
        f"        export_run({str(run)!r}, **wrong)\n"
        "    except ValueError as exc:\n"
        "        print(type(exc).__name__, exc)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "ValueError an export is csv or parquet, not 'xlsx'",
        "ValueError the export profiles are analysis and release, not 'public'",
    ]
    assert not (run / "export").exists()


def test_a_directory_without_a_run_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as raised:
        export_run(tmp_path / "nowhere", tmp_path / "out")
    assert raised.value.reason == "missing" and not (tmp_path / "out").exists()
    for frame in (ratings_frame, subjects_frame):
        with pytest.raises(ConfigError) as raised:
            frame(tmp_path / "nowhere")
        assert raised.value.reason == "missing"


def test_a_local_run_has_no_release_profile(run: Path, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="analysis profile only") as raised:
        export_run(run, profile="release")
    assert raised.value.reason == "invalid"
    assert not (run / "release").exists() and not (run / "export").exists()
    with pytest.raises(ConfigError):
        export_run(run, tmp_path / "out", profile="release")
    assert not (tmp_path / "out").exists()


# --- without the packages of the app --------------------------------------------------------------


def test_the_export_works_without_the_packages_of_the_app(
    run: Path, engine: Engine, uids: list[str], tmp_path: Path
) -> None:
    fill(run, engine, uids)
    engine.dispose()
    export_run(run, tmp_path / "here")
    code = BLOCK_APP_PACKAGES + (
        "from surfcensus.export.tables import export_run, ratings_frame, subjects_frame\n"
        f"paths = export_run({str(run)!r}, {str(tmp_path / 'there')!r})\n"
        f"print(len(paths), len(ratings_frame({str(run)!r})), len(subjects_frame({str(run)!r})))\n"
        "loaded = [n for n in ('sqlmodel', 'sqlalchemy') if sys.modules[n] is not None]\n"
        "print(loaded, [n for n in sys.modules if n.startswith('surfcensus.app.')])\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["11 3 3", "[] []"]
    assert listing(tmp_path / "there") == listing(tmp_path / "here")


def test_the_module_imports_with_sqlmodel_blocked() -> None:
    code = (
        "import sys\nsys.modules['sqlmodel'] = None\n"
        "import surfcensus.export.tables as tables\nprint(len(tables.EXPORT_TABLES))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "10"


# --- a database that is not this run's ratings ---------------------------------------------------


def run_id_of(run_dir: Path) -> str:
    with PublicCatalog.open(run_dir) as public:
        return public.run_id()


def refused(run_dir: Path) -> ConfigError:
    errors = []
    for read in (ratings_frame, marks_frame, subjects_frame, export_run):
        with pytest.raises(ConfigError) as raised:
            read(run_dir)
        assert raised.value.reason == "invalid"
        errors.append(raised.value)
    assert len({error.detail for error in errors}) == 1
    assert not (run_dir / "export").exists()
    return errors[0]


def test_the_ratings_of_another_run_are_refused(run_copy: Path) -> None:
    open_app_db(run_copy, run_id="0f1e2d3c4b5a6978", mode="local").dispose()
    error = refused(run_copy)
    assert "another run" in error.detail and run_id_of(run_copy) in error.detail


def test_a_file_that_is_no_database_of_the_app_is_refused(run_copy: Path, tmp_path: Path) -> None:
    (run_copy / APP_DB_FILE).write_bytes(b"these are not the bytes of a database, " * 40)
    assert "not a SQLite database" in refused(run_copy).detail

    (run_copy / APP_DB_FILE).unlink()
    shutil.copyfile(run_copy / CATALOG_FILE, run_copy / APP_DB_FILE)
    assert "no table app_meta" in refused(run_copy).detail

    (run_copy / APP_DB_FILE).unlink()
    connection = sqlite3.connect(run_copy / APP_DB_FILE)
    connection.execute("CREATE TABLE app_meta (name TEXT, content TEXT)")
    connection.commit()
    connection.close()
    assert "other columns" in refused(run_copy).detail


def test_a_database_of_another_schema_or_without_a_table_is_refused(run_copy: Path) -> None:
    def change(sql: str) -> None:
        connection = sqlite3.connect(run_copy / APP_DB_FILE)
        connection.execute(sql)
        connection.commit()
        connection.close()

    open_app_db(run_copy, run_id=run_id_of(run_copy), mode="local").dispose()
    assert len(ratings_frame(run_copy)) == 0
    change(f"UPDATE app_meta SET value = '{APP_SCHEMA_VERSION + 1}' WHERE key = 'schema_version'")
    assert f"has schema {APP_SCHEMA_VERSION + 1}" in refused(run_copy).detail
    change("UPDATE app_meta SET value = '0' WHERE key = 'schema_version'")
    assert "earlier version" in refused(run_copy).detail
    change("UPDATE app_meta SET value = 'one' WHERE key = 'schema_version'")
    assert "no schema number" in refused(run_copy).detail
    change(f"UPDATE app_meta SET value = '{APP_SCHEMA_VERSION}' WHERE key = 'schema_version'")
    assert len(ratings_frame(run_copy)) == 0
    change("DROP TABLE events")
    assert "no table events" in refused(run_copy).detail


def test_a_database_whose_making_was_interrupted_counts_as_none(run_copy: Path) -> None:
    (run_copy / APP_DB_FILE).write_bytes(b"")
    assert ratings_frame(run_copy).empty
    assert subjects_frame(run_copy)[list(SUBJECT_RATING_COLUMNS)].isna().all().all()
    assert len(export_run(run_copy)) == len(EXPORT_TABLES) + 1
    assert (run_copy / APP_DB_FILE).read_bytes() == b""


def test_a_database_that_stays_locked_is_said_to_be_in_use(
    run: Path, engine: Engine, uids: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    rate(engine, run, uids[0])
    engine.dispose()
    monkeypatch.setattr(tables, "_BUSY_S", 0.2)
    holder = sqlite3.connect(run / APP_DB_FILE, isolation_level=None)
    try:
        holder.execute("BEGIN EXCLUSIVE")
        assert "in use by another process" in refused(run).detail
    finally:
        holder.close()
    assert len(ratings_frame(run)) == 1


def test_an_export_reads_what_is_committed_while_the_app_is_writing(
    run: Path, engine: Engine, uids: list[str]
) -> None:
    done = rate(engine, run, uids[0])
    pending = assign(engine, run, uids[1])
    with Session(engine) as session:
        # a request of the server that has stored its rating and not yet committed
        assignment = session.get(Assignment, pending)
        assert assignment is not None
        rating = RatingIn.model_validate(
            {"assignment_id": pending, "protocol_version": "0.1-draft", "values": VALUES}
            | {"ms_on_subject": 1}
        )
        store_submission(session, assignment, rating, [], protocol_sha256="9c1f", app_version="0")
        session.flush()
        assert list(ratings_frame(run)["assignment_id"]) == [done]
        assert len(export_run(run)) == len(EXPORT_TABLES) + 1
        session.commit()
    assert list(ratings_frame(run)["assignment_id"]) == [done, pending]


def test_reading_the_ratings_does_not_change_the_database(
    run: Path, engine: Engine, uids: list[str]
) -> None:
    fill(run, engine, uids)
    engine.dispose()
    before = (run / APP_DB_FILE).read_bytes()
    stamp = (run / APP_DB_FILE).stat().st_mtime_ns
    catalog = (run / CATALOG_FILE).read_bytes()
    export_run(run)
    export_run(run, fmt="parquet")
    assert (run / APP_DB_FILE).read_bytes() == before
    assert (run / APP_DB_FILE).stat().st_mtime_ns == stamp
    assert (run / CATALOG_FILE).read_bytes() == catalog
    assert sorted(path.name for path in run.iterdir() if path.name.startswith(APP_DB_FILE)) == [
        APP_DB_FILE
    ]


def test_a_metric_table_of_another_version_stops_the_export(run_copy: Path) -> None:
    scores = read_table(table_path(run_copy, "scores"), SCHEMAS["scores"])
    pq.write_table(
        pa.Table.from_pandas(scores.drop(columns=["primary_rank"])), table_path(run_copy, "scores")
    )
    for read in (subjects_frame, export_run):
        with pytest.raises(ConfigError, match=r"tables/scores\.parquet cannot be used") as raised:
            read(run_copy)
        assert raised.value.reason == "invalid" and str(run_copy) not in raised.value.detail
    assert not (run_copy / "export").exists()
    assert ratings_frame(run_copy).empty  # needs no metric table

    table_path(run_copy, "scores").write_bytes(b"PAR1 and then nothing of a table")
    with pytest.raises(ConfigError, match=r"tables/scores\.parquet cannot be used"):
        export_run(run_copy)
    table_path(run_copy, "scores").unlink()
    table_path(run_copy, "clusters_native").write_bytes(b"cut short")
    with pytest.raises(ConfigError, match=r"tables/clusters_native\.parquet cannot be used"):
        export_run(run_copy)
    assert len(subjects_frame(run_copy)) == 3  # needs no cluster table


def test_a_run_before_its_metrics_exports_without_scores(run_copy: Path) -> None:
    for name in ("scores", "clusters", "clusters_native"):
        table_path(run_copy, name).unlink()
    frame = subjects_frame(run_copy)
    assert len(frame) == 3 and frame["primary_rank"].isna().all()
    export_run(run_copy, fmt="parquet")
    assert pq.read_table(run_copy / "export" / "scores.parquet").num_rows == 0
    assert pq.read_schema(run_copy / "export" / "scores.parquet").names == SCHEMAS["scores"].names
    manifest = manifest_of(run_copy / "export")
    assert manifest["reference_ids"] == [] and manifest["files"]["clusters.parquet"]["rows"] == 0


# --- a census run: pseudonyms, and the release profile -------------------------------------------

LABELS_A = [f"sub-{n:02d}" for n in range(1, 5)]
LABELS_B = [f"sub-{n:02d}" for n in range(11, 23)]


@pytest.fixture(scope="module")
def census_base(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A census run over two local trees: dataset A, an fMRIPrep output of 4 subjects of
    FreeSurfer 7.3.2 at 2.89 T, and dataset B, a subjects directory of 12 at 7 T, one of
    them incomplete. Inventoried and scored (no template, so the ranking of D16); nothing
    is rendered."""
    root = tmp_path_factory.mktemp("census")
    a = write_tree(root / "study-a", "fmriprep", LABELS_A)
    b = write_tree(root / "study-b", "subjects_dir", LABELS_B, incomplete=["sub-13"])
    for subject in (a / "sourcedata" / "freesurfer").iterdir():
        write_tier_a(subject, "fs7")
        shutil.rmtree(subject / "stats")  # one-line stand-ins, which the metrics cannot read
    sources = [local(a, field_strength=2.89), local(b, field_strength=7.0)]
    with Run(root / "run", sources, key=DATASET_KEY) as census:
        assert census.inventory().done == 16
        report = census.metrics()
        assert report.failed == 0
    return root / "run"


@dataclasses.dataclass
class Census:
    run_dir: Path
    a: list[str]
    """The subjects of dataset A, sorted; likewise ``b``."""
    b: list[str]
    dataset_a: str
    dataset_b: str
    selected: list[str]
    names: tuple[str, str]


@pytest.fixture
def census(census_base: Path, tmp_path: Path) -> Iterator[Census]:
    """A copy of ``census_base`` with what a census has after rating: ages, a selection of
    two samples (two subjects of A, one of them in both, and three of B) and the ratings
    of two raters in a team database."""
    run_dir = Path(shutil.copytree(census_base, tmp_path / "census"))
    with PublicCatalog.open(run_dir) as public:
        subjects = public.subjects()
        datasets = public.datasets().set_index("layout")["pseudonym"].to_dict()
        run_id = public.run_id()
    dataset_a, dataset_b = datasets["fmriprep"], datasets["subjects_dir"]
    a = sorted(subjects.loc[subjects["dataset"] == dataset_a, "subject_uid"])
    b = sorted(subjects.loc[subjects["dataset"] == dataset_b, "subject_uid"])
    assert (len(a), len(b)) == (4, 12)
    private(run_dir, "UPDATE subjects SET age = 31.5, sex = 'F', manufacturer = 'Siemens'")
    rows = pd.DataFrame(
        {
            "subject_uid": [a[0], a[1], a[1], b[0], b[1], b[2]],
            "reason": ["prevalence", "prevalence", "detectability", *["prevalence"] * 3],
            "stratum": [dataset_a, dataset_a, "decile-10", dataset_b, dataset_b, dataset_b],
            "inclusion_prob": [0.5, 0.5, 0.123456, 0.2573, 0.2573, 0.2573],
            "seed": 7,
        }
    )
    with PublicCatalog.open(run_dir, readonly=False) as public:
        public.replace_selection(rows, ["prevalence", "detectability"])
    names = ("Tomas Knapen", "Ada Lovelace")
    engine = open_app_db(run_dir, run_id=run_id, mode="team")
    try:
        with Session(engine) as session:
            for name in (*names, "nobody yet"):
                get_or_create_rater(session, name)
            session.commit()
        for number, uid in enumerate([a[0], a[1], b[0]]):
            for rater in names[::-1]:
                with Session(engine) as session:
                    who = get_or_create_rater(session, rater)
                    assignment = Assignment(
                        rater_id=who.rater_id,
                        subject_uid=uid,
                        seq=number + 1,
                        render_key="5d0c1f2a9b7e4c31",
                        assigned_at=T1,
                    )
                    session.add(assignment)
                    session.commit()
                    assignment_id = assignment.assignment_id
                assert assignment_id is not None
                submit(engine, assignment_id, marks=[click("m1")], note=f"note {number}")
                log(engine, assignment_id, ("play", {}))
        label(engine, a[0], "consensus", A=2, B=0, C=1)
        with Session(engine) as session:
            session.add(
                FinalMark(subject_uid=a[0], mark_class="A", tkr_x=1.0, tkr_y=2.0, tkr_z=3.0)
            )
            session.commit()
    finally:
        engine.dispose()
    yield Census(run_dir, a, b, dataset_a, dataset_b, [a[0], a[1], b[0], b[1], b[2]], names)


def test_a_census_export_names_no_subject_no_rater_and_no_place(census: Census) -> None:
    for profile in PROFILES:
        paths = export_run(census.run_dir, profile=profile)
        assert {path.parent.name for path in paths} == {EXPORT_DIRS[profile]}
        text = "".join(path.read_text() for path in paths)
        for label_ in (*LABELS_A, *LABELS_B):
            assert label_ not in text, label_
        for word in ("study-a", "study-b", str(census.run_dir.parent), "Tomas", "Ada", "Lovelace"):
            assert word not in text, word
        subjects = read_csv(paths[0])
        assert "subject_label" not in subjects.columns and "participant" not in subjects.columns
        assert list(subjects.columns[:3]) == ["subject_uid", "dataset", "present"]
        assert set(subjects["dataset"]) <= {census.dataset_a, census.dataset_b}
        assert manifest_of(paths[-1].parent)["mode"] == "census"
        assert manifest_of(paths[-1].parent)["profile"] == profile
    config = (census.run_dir / CONFIG_FILE).read_text()
    assert manifest_of(census.run_dir / "export")["key_fingerprint"] == DATASET_KEY.fingerprint
    assert DATASET_KEY.fingerprint in config


def test_census_raters_are_numbered_in_the_order_they_were_made(census: Census) -> None:
    frame = ratings_frame(census.run_dir)
    # Tomas was made first and rated second
    assert list(frame["rater"]) == ["R-02", "R-01"] * 3
    assert list(marks_frame(census.run_dir)["rater"]) == ["R-02", "R-01"] * 3
    export_run(census.run_dir)
    for name in ("ratings", "marks", "behaviour"):
        assert (
            list(read_csv(census.run_dir / "export" / f"{name}.csv")["rater"])
            == [
                "R-02",
                "R-01",
            ]
            * 3
        ), name
    # the names stay where they were given
    connection = sqlite3.connect(census.run_dir / APP_DB_FILE)
    assert [
        name for (name,) in connection.execute("SELECT name FROM raters ORDER BY rater_id")
    ] == [
        *census.names,
        "nobody yet",
    ]
    connection.close()


def test_the_analysis_profile_of_a_census_is_whole(census: Census) -> None:
    export_run(census.run_dir)
    out = census.run_dir / "export"
    subjects = read_csv(out / "subjects.csv")
    assert len(subjects) == 16 and len(read_csv(out / "scores.csv")) == 16
    assert set(subjects["age"]) == {"31.5"} and set(subjects["sex"]) == {"F"}
    assert set(subjects["manufacturer"]) == {"Siemens"}
    assert subjects.set_index("subject_uid")["field_strength"].to_dict() == {
        **dict.fromkeys(census.a, "2.89"),
        **dict.fromkeys(census.b, "7.0"),
    }
    assert set(subjects.set_index("subject_uid").loc[census.a, "fs_version"]) == {"7.3.2"}
    datasets = read_csv(out / "datasets.csv").to_dict("records")
    assert datasets == sorted(
        [
            {
                "dataset": census.dataset_a,
                "source": "local",
                "pipeline": "fmriprep",
                "pipeline_version": "23.2.0",
                "n_subjects": "4",
                "modal_field_strength": "2.89",
            },
            {
                "dataset": census.dataset_b,
                "source": "local",
                "pipeline": "freesurfer",
                "pipeline_version": "",
                "n_subjects": "12",
                "modal_field_strength": "7.0",
            },
        ],
        key=lambda row: row["dataset"],
    )
    selection = read_csv(out / "selection.csv")
    assert sorted(selection["inclusion_prob"]) == sorted(
        ["0.5", "0.5", "0.123456", *["0.2573"] * 3]
    )
    ratings = read_csv(out / "ratings.csv")
    assert list(ratings["note"]) == ["note 0", "note 0", "note 1", "note 1", "note 2", "note 2"]


def test_the_release_profile_is_coarser_and_holds_selected_subjects_only(census: Census) -> None:
    export_run(census.run_dir)
    paths = export_run(census.run_dir, profile="release")
    out = census.run_dir / "release"
    assert {path.parent for path in paths} == {out}
    analysis = {name: read_csv(census.run_dir / "export" / f"{name}.csv") for name in EXPORT_TABLES}
    release = {name: read_csv(out / f"{name}.csv") for name in EXPORT_TABLES}

    # columns: what the release leaves out, and nothing else
    assert RELEASE_LEFT_OUT == {
        "subjects": ("age", "sex", "manufacturer"),
        "datasets": ("pipeline_version",),
        "ratings": ("note",),
    }
    for name in EXPORT_TABLES:
        left_out = RELEASE_LEFT_OUT.get(name, ())
        assert set(left_out) <= set(analysis[name].columns), name
        assert list(release[name].columns) == [
            column for column in analysis[name].columns if column not in left_out
        ], name

    # selected subjects only
    selected = sorted(census.selected)
    assert list(release["subjects"]["subject_uid"]) == selected
    assert list(release["scores"]["subject_uid"]) == selected
    assert sorted(set(release["selection"]["subject_uid"])) == selected
    assert len(release["selection"]) == 6  # the subject of both samples keeps its two rows
    assert len(release["ratings"]) == 6 and len(release["marks"]) == 6

    # field strength in bins, in the subjects and in the mode of a dataset
    fields = release["subjects"].set_index("subject_uid")["field_strength"].to_dict()
    assert fields == {uid: ("3.0" if uid in census.a else "7.0") for uid in selected}
    datasets = release["datasets"].set_index("dataset").to_dict("index")
    assert datasets == {
        census.dataset_a: {
            "source": "local",
            "pipeline": "fmriprep",
            "n_subjects": "1 to 9",
            "modal_field_strength": "3.0",
        },
        census.dataset_b: {
            "source": "local",
            "pipeline": "freesurfer",
            "n_subjects": "10 to 29",
            "modal_field_strength": "7.0",
        },
    }
    # the sampling probability to two digits
    assert sorted(release["selection"]["inclusion_prob"]) == sorted(
        ["0.5", "0.5", "0.12", "0.26", "0.26", "0.26"]
    )
    assert list(release["ratings"]["rater"]) == ["R-02", "R-01"] * 3
    # everything else is as in the analysis profile
    kept = analysis["subjects"][analysis["subjects"]["subject_uid"].isin(selected)]
    same = [name for name in release["subjects"].columns if name != "field_strength"]
    assert release["subjects"][same].to_dict("list") == kept[same].to_dict("list")
    assert release["behaviour"].equals(analysis["behaviour"])
    assert release["final_marks"].equals(analysis["final_marks"])

    manifest = manifest_of(out)
    assert manifest["profile"] == "release" and manifest["files"]["subjects.csv"]["rows"] == 5
    for path in paths[:-1]:
        assert (
            manifest["files"][path.name]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        )
    text = "".join(path.read_text() for path in paths)
    assert "31.5" not in text and "Siemens" not in text and "note " not in text
    assert "0.123456" not in text and "2.89" not in text and "23.2.0" not in text


def test_the_release_holds_the_rows_of_selected_subjects_in_every_table(census: Census) -> None:
    # a rating, a final mark and a cluster of a subject that the selection does not hold
    outside = census.b[5]
    with PublicCatalog.open(census.run_dir) as public:
        run_id = public.run_id()
    engine = open_app_db(census.run_dir, run_id=run_id, mode="team")
    try:
        with Session(engine) as session:
            who = get_or_create_rater(session, "nobody yet")
            assignment = Assignment(
                rater_id=who.rater_id,
                subject_uid=outside,
                seq=1,
                render_key="5d0c1f2a9b7e4c31",
                assigned_at=T1,
            )
            session.add(assignment)
            session.add(
                FinalMark(subject_uid=outside, mark_class="A", tkr_x=0.0, tkr_y=0.0, tkr_z=0.0)
            )
            session.commit()
            assignment_id = assignment.assignment_id
        assert assignment_id is not None
        submit(engine, assignment_id, marks=[click("m1")])
    finally:
        engine.dispose()
    clusters = pd.DataFrame(
        {
            "subject_uid": [outside, census.a[0]],
            "reference_id": "set-0123456789ab",
            "hemi": "lh",
            "region": "search",
            "rank": 1,
            "n_vertices": 4,
            "area_mm2": 2.5,
            "mass": 14.0,
            "excess_mass": 2.0,
            "peak_z": 3.5,
            "peak_vertex": 7,
            "centroid_vertex": 8,
        }
    )
    write_table(
        table_path(census.run_dir, "clusters"),
        clusters,
        SCHEMAS["clusters"],
        sort_by=KEYS["clusters"],
    )
    export_run(census.run_dir)
    export_run(census.run_dir, profile="release")
    for name in ("ratings", "marks", "final_marks", "clusters", "behaviour"):
        whole = read_csv(census.run_dir / "export" / f"{name}.csv")
        released = read_csv(census.run_dir / "release" / f"{name}.csv")
        assert outside in set(whole["subject_uid"]), name
        assert outside not in set(released["subject_uid"]), name
        assert len(released) == len(whole) - 1 > 0, name
    assert list(read_csv(census.run_dir / "export" / "ratings.csv")["rater"])[-1] == "R-03"


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (None, None),
        (0, "0"),
        (1, "1 to 9"),
        (9, "1 to 9"),
        (10, "10 to 29"),
        (29, "10 to 29"),
        (30, "30 to 99"),
        (99, "30 to 99"),
        (100, "100 or more"),
        (5000, "100 or more"),
    ],
)
def test_the_bins_of_a_subject_count(count: int | None, expected: str | None) -> None:
    assert tables._count_bin(count) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.123456, 0.12), (0.2573, 0.26), (1.0, 1.0), (0.5, 0.5), (0.0004567, 0.00046), (0.999, 1.0)],
)
def test_a_probability_is_released_to_two_digits(value: float, expected: float) -> None:
    assert tables._two_digits(value) == expected


def test_behaviour_keeps_the_time_of_a_revised_rating(
    run: Path, engine: Engine, uids: list[str]
) -> None:
    """A revision is a new submission: the visit that made the first rating is counted in
    the events of the row, so its time has to be in the row as well."""
    twice = rate(engine, run, uids[0], values=NONE, ms=3949)
    submit(engine, twice, values=VALUES, now=T2, ms=2529)
    submit(engine, twice, values=NONE, now=T3, ms=700)
    once = rate(engine, run, uids[1], ms=1100)

    export_run(run)
    behaviour = by_assignment(read_csv(run / "export" / "behaviour.csv"))
    assert behaviour.loc[str(twice), ["ms_on_subject", "ms_revised_visits"]].tolist() == [
        "700",
        "6478",
    ]
    assert behaviour.loc[str(once), ["ms_on_subject", "ms_revised_visits"]].tolist() == [
        "1100",
        "0",
    ]
    # the ratings table still has the visit of the current submission only
    ratings = by_assignment(read_csv(run / "export" / "ratings.csv"))
    assert ratings.loc[str(twice), "ms_on_subject"] == "700"


def test_a_superseded_submission_without_a_time_is_not_counted_as_none(
    run: Path, engine: Engine, uids: list[str]
) -> None:
    assignment = rate(engine, run, uids[0], values=NONE, ms=3949)
    submit(engine, assignment, values=VALUES, now=T2, ms=2529)
    with Session(engine) as session:
        first = session.exec(select(Submission).where(col(Submission.current).is_(False))).one()
        first.ms_on_subject = None
        session.add(first)
        session.commit()
    export_run(run)
    row = read_csv(run / "export" / "behaviour.csv").to_dict("records")[0]
    assert (row["ms_on_subject"], row["ms_revised_visits"]) == ("2529", "")


def test_the_manifest_says_how_the_queue_was_ordered(run: Path) -> None:
    config = load_config(run)
    changed = config.app.model_dump() | {"blind": True, "repeat_rate": 0.25, "repeat_gap": 7}
    write_config(run, with_changes(config, app=changed))
    export_run(run)
    assert manifest_of(run / "export")["queue"] == {
        "blind": True,
        "queue_seed": config.app.queue_seed,
        "repeat_rate": 0.25,
        "repeat_gap": 7,
    }
