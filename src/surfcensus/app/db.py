"""``app.sqlite``: opening it, its schema version, and the writes that more than one module
makes. See docs/rater_app.md.

Transactions. Every transaction of an engine from :func:`open_app_db` starts with
``BEGIN IMMEDIATE``, also one that only reads, so the transactions of a database run one
after the other. A request reads before it writes (the last revision, then the next one;
a rater's last ``seq``, then the next one), and two requests can arrive together: the page
sends its events while a rating is on its way. A transaction that reads first and asks for
the write lock afterwards is refused by SQLite at once, without waiting, when another did
the same in the meantime. With the lock taken at the start the second request waits for the
first instead, up to the busy timeout. The price: a second session opened in a thread
while the first still has a transaction open waits for that long and then fails. So use
one session per request, pass it on, and close it: a session that was committed begins
its next transaction at its next read, and holds the lock again until it is closed.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple, cast, get_args

from sqlalchemy import event
from sqlalchemy.engine import URL, Connection, Engine, create_engine
from sqlalchemy.exc import DBAPIError
from sqlmodel import Session, SQLModel, col, select

from surfcensus.app import APP_DB_FILE, APP_TABLES, RATINGS_FLAT_VIEW
from surfcensus.app.models import Assignment, Mark, Rater, Rating, Submission
from surfcensus.app.schemas import AppMode, RaterRole, RatingIn, ResolvedMark
from surfcensus.config import ConfigError
from surfcensus.log import get_logger

APP_SCHEMA_VERSION = 1
"""``app_meta.schema_version`` of a database made by this version of the package."""

MIGRATIONS: tuple[str, ...] = ()
"""SQL script k takes schema k + 1 to k + 2. Applied in order, each in one transaction, to a
database of an older schema: ``create_all`` does not alter a table, so a database with
ratings is never left to it. Each statement of a script ends at the end of a line."""

RATINGS_FLAT_SQL = f"""
CREATE VIEW IF NOT EXISTS {RATINGS_FLAT_VIEW} AS
SELECT r.rating_id, r.assignment_id, r.criterion, r.value, s.note, s.protocol_version,
       s.app_version, s.client_ts, s.server_ts, s.ms_on_subject
FROM ratings AS r JOIN submissions AS s ON s.submission_id = r.submission_id
WHERE s.current = 1
"""
"""The ``ratings`` columns of docs/rater_app.md, for current submissions. Created with the
tables."""

_MODES: tuple[AppMode, ...] = get_args(AppMode)

# how long a transaction waits for another one before it is refused, as for the catalog
_BUSY_MS: dict[AppMode, int] = {"local": 30_000, "team": 60_000}

_Query = Callable[[str], Sequence[Sequence[Any]]]


class _Meta(NamedTuple):
    """What ``app_meta`` says the database is."""

    schema_version: int
    run_id: str
    mode: AppMode


def app_db_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / APP_DB_FILE


def utc_now() -> str:
    """The present time as the tables store it: UTC, ISO 8601, milliseconds, ``Z``."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# --- what a file is ---------------------------------------------------------------------------


def _foreign(why: str) -> ConfigError:
    return ConfigError("invalid", f"{APP_DB_FILE} is no database of the rater app: {why}")


def _look(query: _Query) -> _Meta | None:
    """What the database says it is, or None for one that holds nothing at all: a new
    file, or one whose making was interrupted (the tables, the view and the ``app_meta``
    rows are made in one transaction). Raises ``ConfigError("invalid")`` for a database of
    something else."""
    if not query("SELECT 1 FROM sqlite_master LIMIT 1"):
        return None
    if not query("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'app_meta'"):
        raise _foreign("it has no table app_meta")
    columns = {str(row[1]) for row in query('PRAGMA table_info("app_meta")')}
    if not {"key", "value"} <= columns:
        raise _foreign("its table app_meta has other columns")
    meta = {str(key): value for key, value in query("SELECT key, value FROM app_meta")}
    for key in _Meta._fields:
        if not isinstance(meta.get(key), str) or not meta[key]:
            raise _foreign(f"app_meta has no {key}")
    version, mode = meta["schema_version"], meta["mode"]
    if not (version.isascii() and version.isdecimal() and int(version) >= 1):
        raise _foreign("app_meta.schema_version is no schema number")
    if mode not in _MODES:
        raise _foreign("app_meta.mode is neither local nor team")
    return _Meta(int(version), meta["run_id"], cast(AppMode, mode))


def _in_use(exc: sqlite3.Error) -> bool:
    """Whether ``exc`` says that another connection kept the file locked for longer than
    the busy timeout."""
    code = getattr(exc, "sqlite_errorcode", None)
    # the low byte of an extended code (SQLITE_BUSY_TIMEOUT, ...) is the primary one
    return isinstance(code, int) and code & 0xFF in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)


def _unusable(exc: DBAPIError | sqlite3.Error) -> ConfigError:
    """The one-line error for a database that SQLite cannot open, read or write."""
    orig = exc.orig if isinstance(exc, DBAPIError) else exc
    if isinstance(orig, sqlite3.Error) and _in_use(orig):
        # a locked file is whole: saying anything else invites deleting the ratings
        return ConfigError(
            "invalid", f"{APP_DB_FILE} is in use by another process; try again when that is done"
        )
    code = getattr(orig, "sqlite_errorcode", None)
    if isinstance(code, int) and code & 0xFF in (sqlite3.SQLITE_NOTADB, sqlite3.SQLITE_CORRUPT):
        return ConfigError("invalid", f"{APP_DB_FILE} is not a SQLite database: {orig}")
    return ConfigError("invalid", f"{APP_DB_FILE} cannot be opened: {orig}")


# --- opening ------------------------------------------------------------------------------------


def _engine(path: Path, mode: AppMode) -> Engine:
    engine = create_engine(
        URL.create("sqlite", database=str(path)), connect_args={"timeout": _BUSY_MS[mode] / 1000}
    )

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_connection: Any, _record: object) -> None:
        dbapi_connection.execute("PRAGMA foreign_keys = ON")

    @event.listens_for(engine, "begin")
    def _on_begin(connection: Connection) -> None:
        # left to itself the driver begins a transaction before the first INSERT, UPDATE
        # or DELETE only: a SELECT or a CREATE TABLE before it would be outside of one
        connection.exec_driver_sql("BEGIN IMMEDIATE")

    return engine


def _query_of(connection: Connection) -> _Query:
    return lambda sql: connection.exec_driver_sql(sql).fetchall()


def _create(connection: Connection, run_id: str, mode: AppMode) -> None:
    # by name: the metadata of SQLModel also holds the tables of whoever else uses it
    tables = [SQLModel.metadata.tables[name] for name in APP_TABLES]
    SQLModel.metadata.create_all(connection, tables=tables)
    connection.exec_driver_sql(RATINGS_FLAT_SQL)
    rows = [("schema_version", str(APP_SCHEMA_VERSION)), ("run_id", run_id), ("mode", mode)]
    connection.exec_driver_sql("INSERT INTO app_meta (key, value) VALUES (?, ?)", rows)


def _use_wal(engine: Engine) -> None:
    """Put the database in WAL mode, which stays with the file. The change needs the file
    to itself for a moment and is refused while another connection reads or writes, so
    every opening of a team database asks again."""
    raw = engine.raw_connection()
    try:
        cursor = raw.cursor()
        try:
            cursor.execute("PRAGMA journal_mode = WAL")
        except sqlite3.OperationalError as exc:
            if not _in_use(exc):
                raise
        finally:
            cursor.close()
    finally:
        raw.close()


def _check_run(meta: _Meta, run_id: str, mode: AppMode) -> None:
    if meta.schema_version > APP_SCHEMA_VERSION:
        raise ConfigError(
            "invalid",
            f"{APP_DB_FILE} has schema {meta.schema_version}; this version of surfcensus "
            f"reads up to {APP_SCHEMA_VERSION}",
        )
    if meta.run_id != run_id:
        raise ConfigError(
            "invalid",
            f"{APP_DB_FILE} holds the ratings of another run ({meta.run_id}); this run is {run_id}",
        )
    if meta.mode != mode:
        raise ConfigError(
            "invalid",
            f"{APP_DB_FILE} was made in {meta.mode} mode and cannot be opened in {mode} mode",
        )


def _check_tables(query: _Query) -> None:
    found = {str(row[0]) for row in query("SELECT name FROM sqlite_master WHERE type = 'table'")}
    lacking = [name for name in APP_TABLES if name not in found]
    if lacking:
        raise _foreign(f"it has no table {lacking[0]}")


def _statements(script: str) -> Iterator[str]:
    """The statements of an SQL script, one at a time; each must end at the end of a line."""
    statement = ""
    for line in script.splitlines(keepends=True):
        statement += line
        if sqlite3.complete_statement(statement):
            yield statement.strip()
            statement = ""
    if any(line.strip() and not line.lstrip().startswith("--") for line in statement.splitlines()):
        yield statement.strip()  # unterminated; SQLite says what is wrong with it


def _step(connection: Connection, version: int) -> None:
    """Take the database from schema ``version`` to the next one, in the open transaction."""
    try:
        for statement in _statements(MIGRATIONS[version - 1]):
            connection.exec_driver_sql(statement)
        connection.exec_driver_sql(
            "UPDATE app_meta SET value = ? WHERE key = 'schema_version'", (str(version + 1),)
        )
    except (DBAPIError, sqlite3.Error) as exc:
        orig = exc.orig if isinstance(exc, DBAPIError) else exc
        raise ConfigError(
            "invalid",
            f"{APP_DB_FILE} could not be brought from schema {version} to {version + 1}: "
            f"{orig}; nothing of that step was applied",
        ) from exc


def _prepare(engine: Engine, run_id: str, mode: AppMode) -> None:
    """Make the database, or check that it is the one of this run and bring it up to the
    present schema. Each pass of the loop is one transaction under the write lock, so the
    look and what follows from it see one state of the file, whoever else opens it now."""
    while True:
        with engine.begin() as connection:
            query = _query_of(connection)
            meta = _look(query)
            if meta is None:
                _create(connection, run_id, mode)
                return
            _check_run(meta, run_id, mode)
            if meta.schema_version == APP_SCHEMA_VERSION:
                _check_tables(query)
                return
            _step(connection, meta.schema_version)
        get_logger().info(
            "%s brought from schema %d to %d",
            APP_DB_FILE,
            meta.schema_version,
            meta.schema_version + 1,
        )


def open_app_db(run_dir: Path | str, *, run_id: str, mode: AppMode) -> Engine:
    """An engine on ``app.sqlite`` of a run directory, made if it is not there.

    A new file gets every table of ``app.models``, the view :data:`RATINGS_FLAT_SQL` and the
    ``app_meta`` rows ``schema_version``, ``run_id`` and ``mode``, in one transaction: a
    process that is killed meanwhile leaves a file that holds nothing, which the next call
    fills. An existing file of an older schema is brought up by :data:`MIGRATIONS`. Foreign
    keys are enforced on every connection; a team database uses the write-ahead log, a local
    one the rollback journal. Every transaction of the engine is a write transaction (see
    the module). The caller disposes of the engine.

    Raises ``ConfigError("invalid", ...)`` for a file of a newer schema, of another run
    (``run_id``) or of the other mode, for a file that is no database of the app, for one
    that another process keeps locked and for one that cannot be opened or written; the
    file is left as it was. ``ConfigError("missing", ...)`` when there is no such directory.
    """
    if mode not in _MODES:
        raise ValueError(f"mode is local or team, not {mode!r}")
    if not run_id:
        raise ValueError("the id of the run is needed")
    # absolute, since a pooled connection is opened when it is first needed
    directory = Path(run_dir).absolute()
    if not directory.is_dir():
        raise ConfigError("missing", f"{directory} is no directory")
    engine = _engine(app_db_path(directory), mode)
    try:
        _prepare(engine, run_id, mode)
        if mode == "team":
            # after the checks, so that a database that is refused keeps its journal mode
            _use_wal(engine)
    except (DBAPIError, sqlite3.Error) as exc:
        engine.dispose()
        raise _unusable(exc) from exc
    except BaseException:
        engine.dispose()
        raise
    return engine


def _connect_as_it_is(path: Path) -> sqlite3.Connection:
    """A connection that creates no file. Read-write where that is possible: a writer that
    was killed in a commit leaves a journal, and SQLite refuses a read-only connection
    until a connection that may write has rolled it back."""
    uri = path.absolute().as_uri()
    timeout = _BUSY_MS["local"] / 1000
    try:
        connection = sqlite3.connect(f"{uri}?mode=rw", uri=True, timeout=timeout)
        try:
            connection.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchall()
        except BaseException:
            connection.close()
            raise
        return connection
    except sqlite3.OperationalError as exc:
        if _in_use(exc):
            raise
        return sqlite3.connect(f"{uri}?mode=ro", uri=True, timeout=timeout)


def app_db_mode(run_dir: Path | str) -> AppMode | None:
    """The mode of the ``app.sqlite`` of a run directory (``app_meta.mode``), or None for a
    run that has none yet. For what opens the database of a run as it is, without being
    told a mode: ``surfcensus finalize`` and ``freeze_calibration`` pass it to
    :func:`open_app_db`. It creates nothing, and changes nothing but for the journal of a
    writer that was killed, which SQLite rolls back at any opening. A file that holds
    nothing (a making that was interrupted) is no database yet: None.

    Raises ``ConfigError("invalid", ...)`` for a file that is no database of the app, and
    for one that another process keeps locked.
    """
    path = app_db_path(run_dir)
    if not path.exists():
        return None
    try:
        connection = _connect_as_it_is(path)
        try:
            meta = _look(lambda sql: connection.execute(sql).fetchall())
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise _unusable(exc) from exc
    return None if meta is None else meta.mode


# --- the writes that more than one module makes -------------------------------------------------


def get_or_create_rater(session: Session, name: str, *, role: RaterRole = "rater") -> Rater:
    """The rater of that name, made with ``role`` on first use (a local server makes its one
    rater with role ``lead``). The role of a rater that exists is left as it is. Flushed, so
    ``rater_id`` is set; not committed.

    Names are compared as they are. A name that is empty or has white space at either end
    is a ``ValueError``: ``"Ada "`` beside ``"Ada"`` would be a second rater, and a second
    rater changes who writes the final labels.
    """
    if not name or name != name.strip():
        raise ValueError(f"a rater's name is not empty and has no space at its ends: {name!r}")
    rater = session.exec(select(Rater).where(Rater.name == name)).first()
    if rater is None:
        rater = Rater(name=name, role=role, created_at=utc_now())
        session.add(rater)
        session.flush()
    return rater


def store_submission(
    session: Session,
    assignment: Assignment,
    rating: RatingIn,
    marks: Sequence[ResolvedMark],
    *,
    protocol_sha256: str,
    app_version: str,
    calibration_sha256: str | None = None,
    now: str | None = None,
) -> tuple[Submission, list[Mark]]:
    """Store a rating that passed every rule as the next revision of its assignment.

    ``marks`` are ``rating.marks`` resolved, in the same order; both are ignored for an
    unratable rating, which also stores no value. The new submission is ``current``; the
    earlier ones of the assignment are not, and their marks get ``deleted_ts``. The
    assignment's ``completed_at`` is set when it was open. A mark's ``created_ts`` is the
    client's time of the click, else ``now``. Returns the submission and its marks in the
    order given, flushed (ids set) and not committed: the caller's transaction also holds
    the final labels (``app.finalize.record_single``).

    Raises ``ValueError``, with nothing stored, for a rating of another assignment, for
    marks that are not one per mark of the rating, and for a rating that is neither
    unratable nor has a value.
    """
    assignment_id = assignment.assignment_id
    if assignment_id is None:
        raise ValueError("the assignment has no id yet: flush it first")
    if rating.assignment_id != assignment_id:
        raise ValueError(
            f"the rating is of assignment {rating.assignment_id}, not of {assignment_id}"
        )
    unratable = rating.unratable is not None
    if not unratable and len(marks) != len(rating.marks):
        raise ValueError(f"{len(marks)} resolved marks for the {len(rating.marks)} of the rating")
    if not unratable and not rating.values:
        raise ValueError("the rating is not unratable and has no value")
    now = utc_now() if now is None else now

    earlier = session.exec(
        select(Submission)
        .where(Submission.assignment_id == assignment_id)
        .order_by(col(Submission.revision))
    ).all()
    for superseded in earlier:
        if superseded.current:
            superseded.current = False
            session.add(superseded)
    live = session.exec(
        select(Mark).where(Mark.assignment_id == assignment_id, col(Mark.deleted_ts).is_(None))
    ).all()
    for mark in live:
        mark.deleted_ts = now
        session.add(mark)
    if assignment.completed_at is None:
        assignment.completed_at = now
    session.add(assignment)
    # before the new row: an assignment has one current submission at any moment
    session.flush()

    submission = Submission(
        assignment_id=assignment_id,
        revision=(earlier[-1].revision if earlier else 0) + 1,
        current=True,
        unratable_reason=rating.unratable,
        other=rating.other,
        note=rating.note,
        protocol_version=rating.protocol_version,
        protocol_sha256=protocol_sha256,
        calibration_sha256=calibration_sha256,
        app_version=app_version,
        client_ts=rating.client_ts,
        server_ts=now,
        ms_on_subject=rating.ms_on_subject,
    )
    session.add(submission)
    session.flush()
    submission_id = submission.submission_id
    assert submission_id is not None

    stored: list[Mark] = []
    if not unratable:
        for criterion in sorted(rating.values):
            session.add(
                Rating(
                    submission_id=submission_id,
                    assignment_id=assignment_id,
                    criterion=criterion,
                    value=rating.values[criterion],
                )
            )
        for posted, resolved in zip(rating.marks, marks, strict=True):
            stored.append(
                Mark(
                    submission_id=submission_id,
                    assignment_id=assignment_id,
                    mark_class=posted.mark_class,
                    panel=posted.panel,
                    slice=posted.slice,
                    frame=posted.frame,
                    mode=posted.mode,
                    px=posted.px,
                    py=posted.py,
                    vox_i=resolved.vox[0],
                    vox_j=resolved.vox[1],
                    vox_k=resolved.vox[2],
                    tkr_x=resolved.tkr[0],
                    tkr_y=resolved.tkr[1],
                    tkr_z=resolved.tkr[2],
                    ras_x=resolved.ras[0],
                    ras_y=resolved.ras[1],
                    ras_z=resolved.ras[2],
                    hemi=resolved.hemi,
                    pial_vertex=resolved.pial_vertex,
                    pial_dist_mm=resolved.pial_dist_mm,
                    white_vertex=resolved.white_vertex,
                    white_dist_mm=resolved.white_dist_mm,
                    fsavg6_vertex=resolved.fsavg6_vertex,
                    vertex_surface=resolved.vertex_surface,
                    box_dist_mm=resolved.box_dist_mm,
                    created_ts=posted.ts or now,
                )
            )
        session.add_all(stored)
    session.flush()
    return submission, stored
