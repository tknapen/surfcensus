"""Tests of surfcensus/app/db.py: opening ``app.sqlite``, its schema version, and the two
writes that more than one module makes (a rater, a submission)."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlmodel import Session, select

from surfcensus.app import APP_DB_FILE, APP_TABLES, RATINGS_FLAT_VIEW, db, models
from surfcensus.app.db import (
    app_db_mode,
    app_db_path,
    get_or_create_rater,
    open_app_db,
    store_submission,
)
from surfcensus.app.models import Assignment, Mark, Rater, Rating, Submission
from surfcensus.app.schemas import RatingIn, ResolvedMark
from surfcensus.config import ConfigError
from surfcensus.store import CATALOG_FILE, PublicCatalog

RUN_ID = "0f1e2d3c4b5a6978"
UID = "3f9a1c07d2e4b8a1"
KEY = "5d0c1f2a9b7e4c31"
SIDECARS = ("-journal", "-wal", "-shm")
T1, T2, T3 = (f"2026-11-13T10:00:0{n}.000Z" for n in (1, 2, 3))


def read(run_dir: Path, sql: str) -> list[tuple[Any, ...]]:
    """Rows as another program sees them: what is committed, read with the standard library."""
    connection = sqlite3.connect(app_db_path(run_dir))
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


def meta(run_dir: Path) -> dict[str, str]:
    return dict(read(run_dir, "SELECT key, value FROM app_meta"))


def journal_mode(run_dir: Path) -> str:
    return read(run_dir, "PRAGMA journal_mode")[0][0]


def sidecars(run_dir: Path) -> list[str]:
    return sorted(path.name for path in run_dir.iterdir() if path.name.endswith(SIDECARS))


def refused(run_dir: Path, *, run_id: str = RUN_ID, mode: str = "local") -> ConfigError:
    with pytest.raises(ConfigError) as caught:
        open_app_db(run_dir, run_id=run_id, mode=mode)
    assert caught.value.reason == "invalid"
    return caught.value


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    """An engine on a new local database in ``tmp_path``."""
    engine = open_app_db(tmp_path, run_id=RUN_ID, mode="local")
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    with Session(engine) as session:
        yield session


def new_assignment(session: Session, *, rater: str = "local", seq: int = 1) -> Assignment:
    assignment = Assignment(
        rater_id=get_or_create_rater(session, rater, role="lead").rater_id,
        subject_uid=UID,
        seq=seq,
        render_key=KEY,
        assigned_at=T1,
    )
    session.add(assignment)
    session.flush()
    return assignment


def rating_of(example: dict[str, Any], assignment: Assignment, **changes: Any) -> RatingIn:
    return RatingIn.model_validate(example | {"assignment_id": assignment.assignment_id} | changes)


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


def stored(
    session: Session, assignment: Assignment, rating: RatingIn, marks: Any = (), **kw: Any
) -> tuple[Submission, list[Mark]]:
    kw = {"protocol_sha256": "9c1f", "app_version": "0.0.1"} | kw
    return store_submission(session, assignment, rating, marks, **kw)


# --- a new database -----------------------------------------------------------------------------


def test_a_new_file_gets_every_table_the_view_and_what_ties_it_to_its_run(tmp_path: Path) -> None:
    assert not app_db_path(tmp_path).exists()
    engine = open_app_db(tmp_path, run_id=RUN_ID, mode="local")
    try:
        assert Path(str(engine.url.database)) == tmp_path / APP_DB_FILE
    finally:
        engine.dispose()
    made = dict(read(tmp_path, "SELECT name, type FROM sqlite_master WHERE type != 'index'"))
    assert made == dict.fromkeys(APP_TABLES, "table") | {RATINGS_FLAT_VIEW: "view"}
    for model in models.TABLES:
        table = model.__table__
        columns = [row[1] for row in read(tmp_path, f'PRAGMA table_info("{table.name}")')]
        assert columns == [column.name for column in table.columns], table.name
    assert meta(tmp_path) == {"schema_version": "1", "run_id": RUN_ID, "mode": "local"}
    for table in APP_TABLES:
        if table != "app_meta":
            assert read(tmp_path, f"SELECT COUNT(*) FROM {table}") == [(0,)]
    assert journal_mode(tmp_path) == "delete" and sidecars(tmp_path) == []
    assert sorted(path.name for path in tmp_path.iterdir()) == [APP_DB_FILE]


def test_a_team_database_uses_the_write_ahead_log(tmp_path: Path) -> None:
    engine = open_app_db(tmp_path, run_id=RUN_ID, mode="team")
    try:
        with Session(engine) as session:
            get_or_create_rater(session, "R-01")
            session.commit()
        assert journal_mode(tmp_path) == "wal"
    finally:
        engine.dispose()
    assert meta(tmp_path)["mode"] == "team"
    # the mode stays with the file, and a second opening finds it there
    again = open_app_db(tmp_path, run_id=RUN_ID, mode="team")
    again.dispose()
    assert journal_mode(tmp_path) == "wal"
    assert read(tmp_path, "SELECT name FROM raters") == [("R-01",)]


def test_a_team_database_that_missed_the_log_gets_it_at_the_next_opening(tmp_path: Path) -> None:
    # a process killed between the making and the switch leaves a team database like this
    open_app_db(tmp_path, run_id=RUN_ID, mode="team").dispose()
    connection = sqlite3.connect(app_db_path(tmp_path))
    connection.execute("PRAGMA journal_mode = DELETE")
    connection.close()
    assert journal_mode(tmp_path) == "delete"
    open_app_db(tmp_path, run_id=RUN_ID, mode="team").dispose()
    assert journal_mode(tmp_path) == "wal"


def test_opening_again_keeps_the_rows_and_makes_nothing_twice(tmp_path: Path) -> None:
    first = open_app_db(tmp_path, run_id=RUN_ID, mode="local")
    with Session(first) as session:
        assignment = new_assignment(session)
        session.commit()
        assert assignment.assignment_id == 1
    first.dispose()
    schema = read(tmp_path, "SELECT name, sql FROM sqlite_master ORDER BY name")
    second = open_app_db(str(tmp_path), run_id=RUN_ID, mode="local")
    try:
        with Session(second) as session:
            assert session.exec(select(Rater)).one().name == "local"
            assert session.exec(select(Assignment)).one().subject_uid == UID
    finally:
        second.dispose()
    assert read(tmp_path, "SELECT name, sql FROM sqlite_master ORDER BY name") == schema
    assert len(read(tmp_path, "SELECT * FROM app_meta")) == 3


def test_a_relative_run_directory_gives_an_absolute_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "run").mkdir()
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path)
    engine = open_app_db("run", run_id=RUN_ID, mode="local")
    try:
        assert Path(str(engine.url.database)) == tmp_path / "run" / APP_DB_FILE
        # a connection that the pool opens later is on the same file
        engine.dispose()
        monkeypatch.chdir(tmp_path / "elsewhere")
        with Session(engine) as session:
            get_or_create_rater(session, "local")
            session.commit()
    finally:
        engine.dispose()
    assert read(tmp_path / "run", "SELECT name FROM raters") == [("local",)]
    assert list((tmp_path / "elsewhere").iterdir()) == []


def test_the_engine_fixture_opens_the_database_of_the_run(app_run: Path, app_engine: Any) -> None:
    with PublicCatalog.open(app_run) as public:
        run_id = public.run_id()
    assert meta(app_run) == {"schema_version": "1", "run_id": run_id, "mode": "local"}
    assert app_db_mode(app_run) == "local"
    # the catalog of the run is another file and is no database of the app
    assert app_db_path(app_run).name != CATALOG_FILE
    with Session(app_engine) as session:
        assert session.exec(select(Rater)).all() == []


def test_no_database_without_a_directory_and_none_for_a_wrong_argument(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as caught:
        open_app_db(tmp_path / "absent", run_id=RUN_ID, mode="local")
    assert caught.value.reason == "missing" and not (tmp_path / "absent").exists()
    blocked = tmp_path / "file"
    blocked.write_text("")
    with pytest.raises(ConfigError) as caught:
        open_app_db(blocked, run_id=RUN_ID, mode="local")
    assert caught.value.reason == "missing"
    with pytest.raises(ValueError, match="id of the run"):
        open_app_db(tmp_path, run_id="", mode="local")
    assert list(tmp_path.iterdir()) == [blocked]


# --- the making is one transaction ----------------------------------------------------------


def test_an_interrupted_making_leaves_a_file_that_holds_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = db._create

    def interrupted(connection: Any, run_id: str, mode: Any) -> None:
        real(connection, run_id, mode)
        assert connection.exec_driver_sql("SELECT COUNT(*) FROM app_meta").scalar() == 3
        raise KeyboardInterrupt

    monkeypatch.setattr(db, "_create", interrupted)
    with pytest.raises(KeyboardInterrupt):
        open_app_db(tmp_path, run_id=RUN_ID, mode="local")
    # the tables, the view and the rows went with the transaction
    assert read(tmp_path, "SELECT name FROM sqlite_master") == [] and sidecars(tmp_path) == []
    assert app_db_mode(tmp_path) is None
    monkeypatch.setattr(db, "_create", real)
    open_app_db(tmp_path, run_id=RUN_ID, mode="local").dispose()
    assert meta(tmp_path)["run_id"] == RUN_ID
    assert {name for (name,) in read(tmp_path, "SELECT name FROM sqlite_master")} >= set(APP_TABLES)


KILLED_WHILE_MAKING = """
import os, sys
from surfcensus.app import db

real = db._create

def killed(connection, run_id, mode):
    real(connection, run_id, mode)
    os._exit(7)

db._create = killed
db.open_app_db(sys.argv[1], run_id=sys.argv[2], mode="local")
"""


def test_a_process_killed_while_it_makes_the_database_leaves_one_that_the_next_makes(
    tmp_path: Path,
) -> None:
    child = subprocess.run([sys.executable, "-c", KILLED_WHILE_MAKING, str(tmp_path), RUN_ID])
    assert child.returncode == 7 and app_db_path(tmp_path).exists()
    assert app_db_mode(tmp_path) is None
    engine = open_app_db(tmp_path, run_id=RUN_ID, mode="local")
    try:
        with Session(engine) as session:
            get_or_create_rater(session, "local")
            session.commit()
    finally:
        engine.dispose()
    assert meta(tmp_path) == {"schema_version": "1", "run_id": RUN_ID, "mode": "local"}
    assert sidecars(tmp_path) == []


def test_an_empty_file_is_made_into_the_database(tmp_path: Path) -> None:
    app_db_path(tmp_path).write_bytes(b"")
    assert app_db_mode(tmp_path) is None and app_db_path(tmp_path).stat().st_size == 0
    open_app_db(tmp_path, run_id=RUN_ID, mode="local").dispose()
    assert meta(tmp_path)["mode"] == "local"


def test_callers_that_open_a_new_database_together_all_get_the_one(tmp_path: Path) -> None:
    start = threading.Barrier(6)

    def opened(number: int) -> int:
        start.wait()
        engine = open_app_db(tmp_path, run_id=RUN_ID, mode="local")
        try:
            with Session(engine) as session:
                rater = get_or_create_rater(session, f"rater-{number}")
                session.commit()
                assert rater.rater_id is not None
                return rater.rater_id
        finally:
            engine.dispose()

    with ThreadPoolExecutor(6) as pool:
        ids = list(pool.map(opened, range(6)))
    assert sorted(ids) == [1, 2, 3, 4, 5, 6]
    assert len(read(tmp_path, "SELECT * FROM app_meta")) == 3
    assert len(read(tmp_path, "SELECT * FROM raters")) == 6


# --- what is refused --------------------------------------------------------------------------


def test_a_newer_schema_another_run_and_another_mode_are_refused(tmp_path: Path) -> None:
    open_app_db(tmp_path, run_id=RUN_ID, mode="local").dispose()
    before = app_db_path(tmp_path).read_bytes()

    error = refused(tmp_path, run_id="ffffffffffffffff")
    assert "another run" in error.detail and RUN_ID in error.detail
    error = refused(tmp_path, mode="team")
    assert "made in local mode" in error.detail and "team mode" in error.detail
    # nothing was changed, and a refused team opening did not switch the journal
    assert app_db_path(tmp_path).read_bytes() == before and sidecars(tmp_path) == []
    assert journal_mode(tmp_path) == "delete"

    connection = sqlite3.connect(app_db_path(tmp_path))
    connection.execute("UPDATE app_meta SET value = '2' WHERE key = 'schema_version'")
    connection.commit()
    connection.close()
    newer = app_db_path(tmp_path).read_bytes()
    error = refused(tmp_path)
    assert "has schema 2" in error.detail and "reads up to 1" in error.detail
    # the newer schema is what is said, also when the run or the mode differ as well
    assert "has schema 2" in refused(tmp_path, run_id="ffffffffffffffff", mode="team").detail
    assert app_db_path(tmp_path).read_bytes() == newer
    # the mode of a newer database can still be read
    assert app_db_mode(tmp_path) == "local"


def test_a_team_database_is_not_opened_in_local_mode(tmp_path: Path) -> None:
    open_app_db(tmp_path, run_id=RUN_ID, mode="team").dispose()
    error = refused(tmp_path, mode="local")
    assert "made in team mode" in error.detail
    assert meta(tmp_path) == {"schema_version": "1", "run_id": RUN_ID, "mode": "team"}
    assert journal_mode(tmp_path) == "wal" and app_db_mode(tmp_path) == "team"


def test_a_file_that_is_no_database_is_refused_and_left_alone(tmp_path: Path) -> None:
    text = b"these are my notes on the ratings\n" * 40
    app_db_path(tmp_path).write_bytes(text)
    for mode in ("local", "team"):
        error = refused(tmp_path, mode=mode)
        assert "is not a SQLite database" in error.detail and APP_DB_FILE in error.detail
        assert str(tmp_path) not in error.detail
    with pytest.raises(ConfigError, match="is not a SQLite database") as caught:
        app_db_mode(tmp_path)
    assert caught.value.reason == "invalid"
    assert app_db_path(tmp_path).read_bytes() == text and sidecars(tmp_path) == []


def other_database(run_dir: Path, *statements: str) -> bytes:
    connection = sqlite3.connect(app_db_path(run_dir))
    for statement in statements:
        connection.execute(statement)
    connection.commit()
    connection.close()
    return app_db_path(run_dir).read_bytes()


META = "CREATE TABLE app_meta (key TEXT PRIMARY KEY, value TEXT)"


@pytest.mark.parametrize(
    ("statements", "why"),
    [
        (("CREATE TABLE notes (text TEXT)",), "no table app_meta"),
        (("CREATE VIEW app_meta AS SELECT 1 AS key, 2 AS value",), "no table app_meta"),
        (("CREATE TABLE app_meta (name TEXT, value TEXT)",), "other columns"),
        ((META,), "no schema_version"),
        ((META, "INSERT INTO app_meta VALUES ('schema_version', '1')"), "no run_id"),
        (
            (META, "INSERT INTO app_meta VALUES ('schema_version', '1'), ('run_id', 'r')"),
            "no mode",
        ),
        (
            (
                META,
                "INSERT INTO app_meta VALUES ('schema_version', '1'), ('run_id', ''), "
                "('mode', 'local')",
            ),
            "no run_id",
        ),
        (
            (
                "CREATE TABLE app_meta (key, value)",  # a number stays a number here
                "INSERT INTO app_meta VALUES ('schema_version', 1), ('run_id', 'r'), "
                "('mode', 'local')",
            ),
            "no schema_version",
        ),
        (
            (
                META,
                "INSERT INTO app_meta VALUES ('schema_version', 'one'), ('run_id', 'r'), "
                "('mode', 'local')",
            ),
            "no schema number",
        ),
        (
            (
                META,
                "INSERT INTO app_meta VALUES ('schema_version', '0'), ('run_id', 'r'), "
                "('mode', 'local')",
            ),
            "no schema number",
        ),
        (
            (
                META,
                "INSERT INTO app_meta VALUES ('schema_version', '1'), ('run_id', 'r'), "
                "('mode', 'census')",
            ),
            "neither local nor team",
        ),
    ],
)
def test_a_database_of_something_else_is_refused_and_left_alone(
    tmp_path: Path, statements: tuple[str, ...], why: str
) -> None:
    before = other_database(tmp_path, *statements)
    error = refused(tmp_path, run_id="r")
    assert "is no database of the rater app" in error.detail and why in error.detail
    with pytest.raises(ConfigError, match="is no database of the rater app") as caught:
        app_db_mode(tmp_path)
    assert caught.value.reason == "invalid" and why in caught.value.detail
    assert app_db_path(tmp_path).read_bytes() == before and sidecars(tmp_path) == []


def test_tables_without_what_ties_them_to_a_run_are_not_adopted(tmp_path: Path) -> None:
    # what create_all alone makes: the making of open_app_db never leaves this
    from sqlalchemy import create_engine
    from sqlmodel import SQLModel

    plain = create_engine(f"sqlite:///{app_db_path(tmp_path)}")
    SQLModel.metadata.create_all(
        plain,
        tables=[model.__table__ for model in models.TABLES],
    )
    plain.dispose()
    assert "no schema_version" in refused(tmp_path).detail


def test_a_database_of_the_app_that_lacks_a_table_is_refused(tmp_path: Path) -> None:
    open_app_db(tmp_path, run_id=RUN_ID, mode="local").dispose()
    other_database(tmp_path, "DROP TABLE final_marks")
    error = refused(tmp_path)
    assert "no table final_marks" in error.detail
    assert app_db_mode(tmp_path) == "local"


def test_the_catalog_of_a_run_is_no_database_of_the_app(app_run: Path, tmp_path: Path) -> None:
    app_db_path(tmp_path).write_bytes((app_run / CATALOG_FILE).read_bytes())
    assert "no table app_meta" in refused(tmp_path).detail


def test_a_database_that_another_process_keeps_locked_is_in_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    open_app_db(tmp_path, run_id=RUN_ID, mode="local").dispose()
    monkeypatch.setitem(db._BUSY_MS, "local", 150)
    holder = sqlite3.connect(app_db_path(tmp_path), isolation_level=None)
    holder.execute("BEGIN EXCLUSIVE")
    try:
        started = time.monotonic()
        error = refused(tmp_path)
        assert time.monotonic() - started >= 0.1  # it waited before it gave up
        assert "in use by another process" in error.detail
        assert "not a SQLite database" not in error.detail
        with pytest.raises(ConfigError, match="in use by another process"):
            app_db_mode(tmp_path)
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    open_app_db(tmp_path, run_id=RUN_ID, mode="local").dispose()
    assert app_db_mode(tmp_path) == "local"


# --- migrations -----------------------------------------------------------------------------


def test_there_is_one_migration_for_each_step_of_the_schema() -> None:
    assert len(db.MIGRATIONS) == db.APP_SCHEMA_VERSION - 1
    assert list(
        db._statements("CREATE TABLE a (x);\n-- a note\nINSERT INTO a\nVALUES ('b; c');\n")
    ) == [
        "CREATE TABLE a (x);",
        "-- a note\nINSERT INTO a\nVALUES ('b; c');",
    ]


def a_database_with_a_rating(run_dir: Path, example: dict[str, Any]) -> list[list[tuple]]:
    engine = open_app_db(run_dir, run_id=RUN_ID, mode="local")
    with Session(engine) as session:
        assignment = new_assignment(session)
        rating = rating_of(example, assignment)
        stored(session, assignment, rating, [resolved()], now=T2)
        session.commit()
    engine.dispose()
    return rows_of(run_dir)


def rows_of(run_dir: Path) -> list[list[tuple]]:
    return [read(run_dir, f"SELECT * FROM {table} ORDER BY 1") for table in APP_TABLES[:-1]]


ADD_A_COLUMN_AND_A_TABLE = """
ALTER TABLE raters ADD COLUMN email TEXT;
-- a statement may hold a semicolon in a text
CREATE TABLE notes (note_id INTEGER PRIMARY KEY, text TEXT NOT NULL);
INSERT INTO notes (text)
VALUES ('first; second');
"""


def test_a_migration_brings_an_older_database_up_and_keeps_its_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api_examples: dict[str, Any]
) -> None:
    before = a_database_with_a_rating(tmp_path, api_examples["rating_in"])
    assert [len(rows) for rows in before] == [1, 1, 1, 3, 1, 0, 0, 0]
    monkeypatch.setattr(db, "APP_SCHEMA_VERSION", 2)
    monkeypatch.setattr(db, "MIGRATIONS", (ADD_A_COLUMN_AND_A_TABLE,))

    engine = open_app_db(tmp_path, run_id=RUN_ID, mode="local")
    try:
        assert meta(tmp_path) == {"schema_version": "2", "run_id": RUN_ID, "mode": "local"}
        after = rows_of(tmp_path)
        assert after[0] == [(*before[0][0], None)]  # the rater, with the new column empty
        assert after[1:] == before[1:]
        assert read(tmp_path, "SELECT text FROM notes") == [("first; second",)]
        # the models still read the tables
        with Session(engine) as session:
            assert session.exec(select(Mark)).one().pial_vertex == 211
            assert session.exec(select(Rater)).one().name == "local"
    finally:
        engine.dispose()
    # at the present schema nothing is applied again
    open_app_db(tmp_path, run_id=RUN_ID, mode="local").dispose()
    assert read(tmp_path, "SELECT COUNT(*) FROM notes") == [(1,)]
    assert rows_of(tmp_path) == after


def test_a_new_database_starts_at_the_present_schema_without_a_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(db, "APP_SCHEMA_VERSION", 2)
    monkeypatch.setattr(db, "MIGRATIONS", (ADD_A_COLUMN_AND_A_TABLE,))
    open_app_db(tmp_path, run_id=RUN_ID, mode="local").dispose()
    assert meta(tmp_path)["schema_version"] == "2"
    assert read(tmp_path, "SELECT name FROM sqlite_master WHERE name = 'notes'") == []


def test_migrations_are_applied_in_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    open_app_db(tmp_path, run_id=RUN_ID, mode="local").dispose()
    monkeypatch.setattr(db, "APP_SCHEMA_VERSION", 3)
    monkeypatch.setattr(
        db,
        "MIGRATIONS",
        (
            "ALTER TABLE raters ADD COLUMN mail TEXT;",
            "ALTER TABLE raters RENAME COLUMN mail TO email;",  # needs the first
        ),
    )
    open_app_db(tmp_path, run_id=RUN_ID, mode="local").dispose()
    assert meta(tmp_path)["schema_version"] == "3"
    assert [row[1] for row in read(tmp_path, 'PRAGMA table_info("raters")')][-1] == "email"


def test_a_migration_that_fails_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api_examples: dict[str, Any]
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    before = a_database_with_a_rating(tmp_path, api_examples["rating_in"])
    before_other = a_database_with_a_rating(other, api_examples["rating_in"])
    schema = read(other, "SELECT name, sql FROM sqlite_master ORDER BY name")
    monkeypatch.setattr(db, "APP_SCHEMA_VERSION", 3)
    monkeypatch.setattr(
        db,
        "MIGRATIONS",
        (
            "ALTER TABLE raters ADD COLUMN email TEXT;",
            "DELETE FROM marks;\nALTER TABLE nowhere ADD COLUMN x TEXT;",
        ),
    )
    error = refused(tmp_path)
    assert "from schema 2 to 3" in error.detail and "nowhere" in error.detail
    # the first step is in, since each is a transaction of its own; of the second, nothing
    assert meta(tmp_path)["schema_version"] == "2"
    assert rows_of(tmp_path)[1:] == before[1:] and len(rows_of(tmp_path)[4]) == 1

    monkeypatch.setattr(
        db, "MIGRATIONS", ("DELETE FROM marks;\nALTER TABLE raters ADD COLUMN email;\nnonsense;",)
    )
    monkeypatch.setattr(db, "APP_SCHEMA_VERSION", 2)
    error = refused(other)
    assert "from schema 1 to 2" in error.detail and "is as it was" in error.detail
    # also the statements before the one that failed, a change of a table among them
    assert meta(other)["schema_version"] == "1" and rows_of(other) == before_other
    assert read(other, "SELECT name, sql FROM sqlite_master ORDER BY name") == schema


# --- connections and transactions -------------------------------------------------------------


def test_foreign_keys_are_enforced_on_every_connection(engine: Engine) -> None:
    first, second = engine.raw_connection(), engine.raw_connection()
    try:
        assert first.driver_connection is not second.driver_connection
        for raw in (first, second):
            assert raw.cursor().execute("PRAGMA foreign_keys").fetchall() == [(1,)]
    finally:
        first.close()
        second.close()
    with Session(engine) as session:
        session.add(Assignment(rater_id=99, subject_uid=UID, seq=1, render_key=KEY, assigned_at=T1))
        with pytest.raises(IntegrityError, match="FOREIGN KEY"):
            session.commit()
    with Session(engine) as session:
        session.add(Rating(submission_id=5, assignment_id=7, criterion="A", value=1))
        with pytest.raises(IntegrityError, match="FOREIGN KEY"):
            session.commit()


def test_what_a_session_does_is_one_transaction(engine: Engine, tmp_path: Path) -> None:
    with Session(engine) as session:
        new_assignment(session)
        # a table made in a transaction goes with it as well
        session.connection().exec_driver_sql("CREATE TABLE scratch (x)")
        assert read(tmp_path, "SELECT COUNT(*) FROM raters") == [(0,)]  # not committed
        session.rollback()
    assert read(tmp_path, "SELECT COUNT(*) FROM raters") == [(0,)]
    assert read(tmp_path, "SELECT name FROM sqlite_master WHERE name = 'scratch'") == []
    with Session(engine) as session:
        new_assignment(session)
        session.commit()
    assert read(tmp_path, "SELECT COUNT(*) FROM assignments") == [(1,)]


def test_requests_that_read_and_then_write_wait_for_each_other(engine: Engine) -> None:
    with Session(engine) as session:
        rater_id = get_or_create_rater(session, "local").rater_id
        session.commit()
    assert rater_id is not None

    def next_assignment(_: int) -> None:
        with Session(engine) as session:
            last = session.exec(
                select(func.max(Assignment.seq)).where(Assignment.rater_id == rater_id)
            ).one()
            time.sleep(0.005)  # room for another request to read the same
            session.add(
                Assignment(
                    rater_id=rater_id,
                    subject_uid=UID,
                    seq=(last or 0) + 1,
                    render_key=KEY,
                    assigned_at=T1,
                )
            )
            session.commit()

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(next_assignment, range(40)))
    with Session(engine) as session:
        seqs = session.exec(select(Assignment.seq).order_by(Assignment.seq)).all()
    assert list(seqs) == list(range(1, 41))


def test_a_transaction_waits_for_the_one_before_it(
    engine: Engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    begun = threading.Event()

    def slow() -> None:
        with Session(engine) as session:
            get_or_create_rater(session, "first")
            begun.set()
            time.sleep(0.3)
            session.commit()

    thread = threading.Thread(target=slow)
    thread.start()
    try:
        assert begun.wait(10)
        started = time.monotonic()
        with Session(engine) as session:
            # also one that only reads
            names = session.exec(select(Rater.name)).all()
        assert time.monotonic() - started >= 0.2 and list(names) == ["first"]
    finally:
        thread.join()

    # and past the busy timeout it is refused, with the database's own error
    short = tmp_path / "short"
    short.mkdir()
    monkeypatch.setitem(db._BUSY_MS, "local", 100)
    quick = open_app_db(short, run_id=RUN_ID, mode="local")
    holder = sqlite3.connect(app_db_path(short), isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        with Session(quick) as session, pytest.raises(OperationalError, match="locked"):
            session.exec(select(Rater)).all()
    finally:
        holder.execute("ROLLBACK")
        holder.close()
        quick.dispose()


# --- the mode of a database as it is -----------------------------------------------------------


def test_the_mode_is_read_without_creating_or_changing_anything(tmp_path: Path) -> None:
    assert app_db_mode(tmp_path) is None and app_db_mode(tmp_path / "absent") is None
    assert list(tmp_path.iterdir()) == []
    for mode in ("local", "team"):
        run_dir = tmp_path / mode
        run_dir.mkdir()
        open_app_db(run_dir, run_id=RUN_ID, mode=mode).dispose()
        before = app_db_path(run_dir).read_bytes()
        stamp = app_db_path(run_dir).stat().st_mtime_ns
        assert app_db_mode(run_dir) == mode and app_db_mode(str(run_dir)) == mode
        assert app_db_path(run_dir).read_bytes() == before
        assert app_db_path(run_dir).stat().st_mtime_ns == stamp
        assert sidecars(run_dir) == []


KILLED_IN_A_COMMIT = """
import os, sqlite3, sys
connection = sqlite3.connect(sys.argv[1], isolation_level=None)
connection.execute("PRAGMA cache_size = 1")  # so that the change reaches the file
connection.execute("BEGIN IMMEDIATE")
connection.execute("UPDATE app_meta SET value = 'team' WHERE key = 'mode'")
connection.executemany(
    "INSERT INTO events (ts, type, payload_json) VALUES ('t', 'load', ?)",
    [("x" * 3000,) for _ in range(400)],
)
os._exit(7)
"""


def test_a_database_whose_writer_was_killed_reads_as_it_was_before(tmp_path: Path) -> None:
    open_app_db(tmp_path, run_id=RUN_ID, mode="local").dispose()
    size = app_db_path(tmp_path).stat().st_size
    child = subprocess.run([sys.executable, "-c", KILLED_IN_A_COMMIT, str(app_db_path(tmp_path))])
    assert child.returncode == 7
    # the half-written change is in the file, with the journal that takes it back
    assert sidecars(tmp_path) == [APP_DB_FILE + "-journal"]
    assert app_db_path(tmp_path).stat().st_size > size
    assert app_db_mode(tmp_path) == "local"
    assert sidecars(tmp_path) == [] and app_db_path(tmp_path).stat().st_size == size
    engine = open_app_db(tmp_path, run_id=RUN_ID, mode="local")
    engine.dispose()
    assert read(tmp_path, "SELECT COUNT(*) FROM events") == [(0,)]


# --- raters -----------------------------------------------------------------------------------


def test_a_rater_is_made_on_first_use_and_found_afterwards(
    session: Session, tmp_path: Path
) -> None:
    lead = get_or_create_rater(session, "local", role="lead")
    assert lead.rater_id == 1 and (lead.name, lead.role, lead.active) == ("local", "lead", True)
    assert lead.token_hash is None and lead.created_at.endswith("Z")
    assert read(tmp_path, "SELECT COUNT(*) FROM raters") == [(0,)]  # flushed, not committed
    # the role of a rater that exists is left as it is
    again = get_or_create_rater(session, "local")
    assert again is lead and again.role == "lead"
    second = get_or_create_rater(session, "Ada")
    assert second.rater_id == 2 and second.role == "rater"
    # names are compared as they are
    assert get_or_create_rater(session, "ada").rater_id == 3
    session.commit()
    assert read(tmp_path, "SELECT rater_id, name, role, active FROM raters ORDER BY 1") == [
        (1, "local", "lead", 1),
        (2, "Ada", "rater", 1),
        (3, "ada", "rater", 1),
    ]
    with Session(session.get_bind()) as later:
        assert get_or_create_rater(later, "Ada", role="adjudicator").role == "rater"
        assert later.exec(select(func.count()).select_from(Rater)).one() == 3


@pytest.mark.parametrize("name", ["", " ", "Ada ", " Ada", "Ada\n", "\tAda"])
def test_a_name_that_would_pass_for_another_rater_is_refused(session: Session, name: str) -> None:
    get_or_create_rater(session, "Ada")
    with pytest.raises(ValueError, match="name"):
        get_or_create_rater(session, name)
    assert session.exec(select(func.count()).select_from(Rater)).one() == 1


# --- submissions --------------------------------------------------------------------------------

MARK_COLUMNS = (
    "mark_id, submission_id, assignment_id, class, panel, slice, frame, mode, px, py, "
    "vox_i, vox_j, vox_k, tkr_x, tkr_y, tkr_z, ras_x, ras_y, ras_z, hemi, pial_vertex, "
    "pial_dist_mm, white_vertex, white_dist_mm, fsavg6_vertex, vertex_surface, box_dist_mm, "
    "created_ts, deleted_ts"
)


def test_a_first_submission_is_revision_one_with_its_values_and_marks(
    session: Session, tmp_path: Path, api_examples: dict[str, Any]
) -> None:
    assignment = new_assignment(session)
    assert assignment.completed_at is None
    rating = rating_of(
        api_examples["rating_in"],
        assignment,
        other=True,
        note="a fold",
        marks=[
            click("m1", "A", frame=3, mode="sync", ts="2026-11-12T14:03:20.004Z"),
            click("m2", "C", panel="axial", slice=5, px=1.25, py=63.5),
        ],
    )
    marks = [
        resolved(),
        resolved(
            hemi="rh",
            pial_vertex=None,
            pial_dist_mm=None,
            white_vertex=40,
            white_dist_mm=1.5,
            fsavg6_vertex=None,
            vertex_surface="white",
            box_dist_mm=2.5,
            vox=(1.0, 2.0, 3.0),
            tkr=(4.0, 5.0, 6.0),
            ras=(7.0, 8.0, 9.0),
        ),
    ]
    submission, saved = stored(
        session, assignment, rating, marks, protocol_sha256="9c1f", app_version="0.0.1", now=T2
    )
    # flushed, so the ids are there, and not committed
    assert submission.submission_id == 1 and [mark.mark_id for mark in saved] == [1, 2]
    assert read(tmp_path, "SELECT COUNT(*) FROM submissions") == [(0,)]
    session.commit()

    assert read(tmp_path, "SELECT * FROM submissions") == [
        (
            1,
            1,
            1,
            1,
            None,
            1,
            "a fold",
            "0.1-draft",
            "9c1f",
            None,
            "0.0.1",
            "2026-11-12T14:03:22.120Z",
            T2,
            14200,
        )
    ]
    assert read(tmp_path, "SELECT submission_id, assignment_id, criterion, value FROM ratings") == [
        (1, 1, "A", 2),
        (1, 1, "B", 0),
        (1, 1, "C", 0),
    ]
    columns = [name.strip() for name in MARK_COLUMNS.split(",")]
    first, second = (
        dict(zip(columns, row, strict=True))
        for row in read(tmp_path, f"SELECT {MARK_COLUMNS} FROM marks ORDER BY mark_id")
    )
    assert first == {
        "mark_id": 1,
        "submission_id": 1,
        "assignment_id": 1,
        "class": "A",
        "panel": "sagittal",
        "slice": 3,
        "frame": 3,
        "mode": "sync",
        "px": 30.5,
        "py": 28.25,
        "vox_i": 7.5,
        "vox_j": 8.25,
        "vox_k": 9.0,
        "tkr_x": -1.5,
        "tkr_y": -2.25,
        "tkr_z": 3.0,
        "ras_x": 0.5,
        "ras_y": -4.25,
        "ras_z": 11.0,
        "hemi": "lh",
        "pial_vertex": 211,
        "pial_dist_mm": 0.8,
        "white_vertex": 207,
        "white_dist_mm": 2.9,
        "fsavg6_vertex": 12950,
        "vertex_surface": "pial",
        "box_dist_mm": 0.0,
        "created_ts": "2026-11-12T14:03:20.004Z",  # the time of the click as the page sent it
        "deleted_ts": None,
    }
    assert second == {
        "mark_id": 2,
        "submission_id": 1,
        "assignment_id": 1,
        "class": "C",
        "panel": "axial",
        "slice": 5,
        "frame": None,
        "mode": None,
        "px": 1.25,
        "py": 63.5,
        "vox_i": 1.0,
        "vox_j": 2.0,
        "vox_k": 3.0,
        "tkr_x": 4.0,
        "tkr_y": 5.0,
        "tkr_z": 6.0,
        "ras_x": 7.0,
        "ras_y": 8.0,
        "ras_z": 9.0,
        "hemi": "rh",
        "pial_vertex": None,
        "pial_dist_mm": None,
        "white_vertex": 40,
        "white_dist_mm": 1.5,
        "fsavg6_vertex": None,
        "vertex_surface": "white",
        "box_dist_mm": 2.5,
        "created_ts": T2,  # no time from the page: that of the submission
        "deleted_ts": None,
    }
    # every column of the table was looked at
    assert columns == [row[1] for row in read(tmp_path, 'PRAGMA table_info("marks")')]
    assert read(tmp_path, "SELECT completed_at FROM assignments") == [(T2,)]
    assert read(tmp_path, f"SELECT criterion, value, note FROM {RATINGS_FLAT_VIEW}") == [
        ("A", 2, "a fold"),
        ("B", 0, "a fold"),
        ("C", 0, "a fold"),
    ]


def test_a_revision_is_a_new_row_and_nothing_is_overwritten(
    session: Session, tmp_path: Path, api_examples: dict[str, Any]
) -> None:
    assignment = new_assignment(session)
    example = api_examples["rating_in"]
    first, first_marks = stored(
        session,
        assignment,
        rating_of(example, assignment, marks=[click("m1"), click("m2", "X")]),
        [resolved(), resolved(pial_vertex=5)],
        now=T1,
    )
    session.commit()
    second, second_marks = stored(
        session,
        assignment,
        rating_of(example, assignment, values={"A": 1, "B": 0, "C": 0}, marks=[click("m9")]),
        [resolved(pial_vertex=6)],
        protocol_sha256="0000",
        now=T2,
    )
    session.commit()
    assert (first.revision, second.revision) == (1, 2)
    assert first.submission_id != second.submission_id
    assert read(tmp_path, "SELECT revision, current, protocol_sha256 FROM submissions") == [
        (1, 0, "9c1f"),
        (2, 1, "0000"),
    ]
    # the first revision keeps its values and its marks, which are marked as superseded
    assert read(tmp_path, "SELECT submission_id, criterion, value FROM ratings ORDER BY 1, 2") == [
        (1, "A", 2),
        (1, "B", 0),
        (1, "C", 0),
        (2, "A", 1),
        (2, "B", 0),
        (2, "C", 0),
    ]
    marks = "SELECT submission_id, class, pial_vertex, created_ts, deleted_ts FROM marks"
    assert read(tmp_path, marks) == [
        (1, "A", 211, T1, T2),
        (1, "X", 5, T1, T2),
        (2, "A", 6, T2, None),
    ]
    assert [mark.deleted_ts for mark in first_marks] == [T2, T2]
    assert [mark.mark_id for mark in second_marks] == [3]
    # the first time the assignment was completed stays
    assert read(tmp_path, "SELECT completed_at FROM assignments") == [(T1,)]
    assert read(tmp_path, f"SELECT criterion, value FROM {RATINGS_FLAT_VIEW}") == [
        ("A", 1),
        ("B", 0),
        ("C", 0),
    ]

    third, _ = stored(session, assignment, rating_of(example, assignment, marks=[]), [], now=T3)
    session.commit()
    assert third.revision == 3
    assert read(tmp_path, "SELECT revision, current FROM submissions") == [(1, 0), (2, 0), (3, 1)]
    # a mark keeps the time at which it was superseded
    assert read(tmp_path, "SELECT submission_id, deleted_ts FROM marks") == [
        (1, T2),
        (1, T2),
        (2, T3),
    ]


def test_an_unratable_rating_stores_no_value_and_no_mark(
    session: Session, tmp_path: Path, api_examples: dict[str, Any]
) -> None:
    assignment = new_assignment(session)
    unratable = rating_of(
        api_examples["rating_in_unratable"],
        assignment,
        # what a page may still send along is ignored
        values={"A": 2},
        marks=[click("m1")],
        note="motion",
    )
    submission, saved = stored(session, assignment, unratable, [], now=T1)
    session.commit()
    assert saved == [] and submission.unratable_reason == "image too poor to judge"
    assert read(tmp_path, "SELECT unratable_reason, note, ms_on_subject FROM submissions") == [
        ("image too poor to judge", "motion", 6100)
    ]
    assert read(tmp_path, "SELECT COUNT(*) FROM ratings") == [(0,)]
    assert read(tmp_path, "SELECT COUNT(*) FROM marks") == [(0,)]
    assert read(tmp_path, "SELECT completed_at FROM assignments") == [(T1,)]
    # also with resolved marks handed over
    stored(session, assignment, unratable, [resolved()], now=T2)
    session.commit()
    assert read(tmp_path, "SELECT COUNT(*) FROM marks") == [(0,)]

    # rated after all: the revision has values and marks, and the unratable ones stay
    rated = rating_of(api_examples["rating_in"], assignment)
    stored(session, assignment, rated, [resolved()], now=T3)
    session.commit()
    assert read(tmp_path, "SELECT revision, current, unratable_reason FROM submissions") == [
        (1, 0, "image too poor to judge"),
        (2, 0, "image too poor to judge"),
        (3, 1, None),
    ]
    assert read(tmp_path, "SELECT submission_id, deleted_ts FROM marks") == [(3, None)]
    # and unratable again: the marks of the rated revision are superseded
    stored(session, assignment, unratable, [], now="2026-11-13T10:00:04.000Z")
    session.commit()
    assert read(tmp_path, "SELECT submission_id, deleted_ts FROM marks") == [
        (3, "2026-11-13T10:00:04.000Z")
    ]
    assert read(tmp_path, f"SELECT COUNT(*) FROM {RATINGS_FLAT_VIEW}") == [(0,)]


def test_a_submission_touches_its_own_assignment_only(
    session: Session, tmp_path: Path, api_examples: dict[str, Any]
) -> None:
    example = api_examples["rating_in"]
    one, two = new_assignment(session, seq=1), new_assignment(session, seq=2)
    other = new_assignment(session, rater="Ada")
    for assignment in (one, two, other):
        stored(session, assignment, rating_of(example, assignment), [resolved()], now=T1)
    stored(session, two, rating_of(example, two, marks=[]), [], now=T2)
    session.commit()
    assert read(
        tmp_path, "SELECT assignment_id, revision, current FROM submissions ORDER BY 1, 2"
    ) == [(1, 1, 1), (2, 1, 0), (2, 2, 1), (3, 1, 1)]
    assert read(tmp_path, "SELECT assignment_id, deleted_ts FROM marks ORDER BY 1") == [
        (1, None),
        (2, T2),
        (3, None),
    ]
    assert read(tmp_path, "SELECT assignment_id, completed_at FROM assignments") == [
        (1, T1),
        (2, T1),
        (3, T1),
    ]


def test_the_time_of_a_submission_is_the_present_unless_given(
    session: Session, api_examples: dict[str, Any]
) -> None:
    assignment = new_assignment(session)
    rating = rating_of(api_examples["rating_in"], assignment, marks=[click("m1", ts="")])
    before = db.utc_now()
    submission, (mark,) = stored(
        session, assignment, rating, [resolved()], calibration_sha256="abcd"
    )
    assert before <= submission.server_ts <= db.utc_now()
    assert assignment.completed_at == submission.server_ts == mark.created_ts
    assert submission.calibration_sha256 == "abcd"


def test_a_rating_that_does_not_fit_its_assignment_stores_nothing(
    session: Session, tmp_path: Path, api_examples: dict[str, Any]
) -> None:
    example = api_examples["rating_in"]
    assignment = new_assignment(session)
    session.commit()
    unsaved = Assignment(rater_id=1, subject_uid=UID, seq=2, render_key=KEY, assigned_at=T1)
    cases = [
        (unsaved, rating_of(example, assignment), [resolved()], "no id yet"),
        (assignment, rating_of(example, assignment, assignment_id=99), [resolved()], "99"),
        (assignment, rating_of(example, assignment), [], "0 resolved marks for the 1"),
        (assignment, rating_of(example, assignment), [resolved()] * 2, "2 resolved marks"),
        (assignment, rating_of(example, assignment, values={}, marks=[]), [], "no value"),
    ]
    for target, rating, marks, message in cases:
        with pytest.raises(ValueError, match=message):
            stored(session, target, rating, marks)
    session.commit()
    for table in ("submissions", "ratings", "marks"):
        assert read(tmp_path, f"SELECT COUNT(*) FROM {table}") == [(0,)]
    assert read(tmp_path, "SELECT completed_at FROM assignments") == [(None,)]


def test_the_database_keeps_one_current_submission_per_assignment(
    session: Session, api_examples: dict[str, Any]
) -> None:
    assignment = new_assignment(session)
    stored(session, assignment, rating_of(api_examples["rating_in"], assignment), [resolved()])
    session.commit()
    assert assignment.assignment_id is not None
    session.add(
        Submission(
            assignment_id=assignment.assignment_id,
            revision=2,
            protocol_version="0.1-draft",
            protocol_sha256="9c1f",
            app_version="0.0.1",
            server_ts=T2,
        )
    )
    with pytest.raises(IntegrityError, match="UNIQUE"):
        session.commit()
    session.rollback()
    # one that is not current is no second one
    session.add(
        Submission(
            assignment_id=assignment.assignment_id,
            revision=2,
            current=False,
            protocol_version="0.1-draft",
            protocol_sha256="9c1f",
            app_version="0.0.1",
            server_ts=T2,
        )
    )
    session.commit()


def test_a_value_outside_the_tables_vocabulary_stores_nothing_of_the_rating(
    session: Session, tmp_path: Path, api_examples: dict[str, Any]
) -> None:
    # the rules of the server come first; what slips past them is refused by the tables,
    # and the transaction of the rating goes as a whole
    assignment = new_assignment(session)
    session.commit()
    rating = rating_of(api_examples["rating_in"], assignment, values={"A": 2, "D": 1})
    with pytest.raises(IntegrityError, match="CHECK"):
        stored(session, assignment, rating, [resolved()])
    session.rollback()
    for table in ("submissions", "ratings", "marks"):
        assert read(tmp_path, f"SELECT COUNT(*) FROM {table}") == [(0,)]
    assert read(tmp_path, "SELECT completed_at FROM assignments") == [(None,)]
