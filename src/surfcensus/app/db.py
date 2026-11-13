"""``app.sqlite``: opening it, its schema version, and the writes that more than one module
makes. See docs/rater_app.md.

Interface of milestone 4. The constants and :func:`utc_now` are final; :func:`open_app_db`,
:func:`app_db_mode`, :func:`get_or_create_rater` and :func:`store_submission` are written by
package 4.B.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.engine import Engine
from sqlmodel import Session

from surfcensus.app import APP_DB_FILE, RATINGS_FLAT_VIEW
from surfcensus.app.models import Assignment, Mark, Rater, Submission
from surfcensus.app.schemas import AppMode, RaterRole, RatingIn, ResolvedMark

_OWNER = "package 4.B (models and database)"

APP_SCHEMA_VERSION = 1
"""``app_meta.schema_version`` of a database made by this version of the package."""

MIGRATIONS: tuple[str, ...] = ()
"""SQL script k takes schema k + 1 to k + 2. Applied in order, each in one transaction, to a
database of an older schema: ``create_all`` does not alter a table, so a database with
ratings is never left to it."""

RATINGS_FLAT_SQL = f"""
CREATE VIEW IF NOT EXISTS {RATINGS_FLAT_VIEW} AS
SELECT r.rating_id, r.assignment_id, r.criterion, r.value, s.note, s.protocol_version,
       s.app_version, s.client_ts, s.server_ts, s.ms_on_subject
FROM ratings AS r JOIN submissions AS s ON s.submission_id = r.submission_id
WHERE s.current = 1
"""
"""The ``ratings`` columns of docs/rater_app.md, for current submissions. Created with the
tables."""


def app_db_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / APP_DB_FILE


def utc_now() -> str:
    """The present time as the tables store it: UTC, ISO 8601, milliseconds, ``Z``."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def open_app_db(run_dir: Path | str, *, run_id: str, mode: AppMode) -> Engine:
    """An engine on ``app.sqlite`` of a run directory, made if it is not there.

    A new file gets every table of ``app.models``, the view :data:`RATINGS_FLAT_SQL` and the
    ``app_meta`` rows ``schema_version``, ``run_id`` and ``mode``. An existing file of an
    older schema is brought up by :data:`MIGRATIONS`. Foreign keys are enforced on every
    connection; a team database uses the write-ahead log, a local one the rollback journal.

    Raises ``ConfigError("invalid", ...)`` for a file of a newer schema, of another run
    (``run_id``) or of the other mode, and for a file that is no database of the app.
    """
    raise NotImplementedError(f"app.db.open_app_db: {_OWNER}")


def app_db_mode(run_dir: Path | str) -> AppMode | None:
    """The mode of the ``app.sqlite`` of a run directory (``app_meta.mode``), or None for a
    run that has none yet. For what opens the database of a run as it is, without being
    told a mode: ``surfcensus finalize`` and ``freeze_calibration`` pass it to
    :func:`open_app_db`. It creates and changes nothing.

    Raises ``ConfigError("invalid", ...)`` for a file that is no database of the app.
    """
    raise NotImplementedError(f"app.db.app_db_mode: {_OWNER}")


def get_or_create_rater(session: Session, name: str, *, role: RaterRole = "rater") -> Rater:
    """The rater of that name, made with ``role`` on first use (a local server makes its one
    rater with role ``lead``). The role of a rater that exists is left as it is. Flushed, so
    ``rater_id`` is set; not committed."""
    raise NotImplementedError(f"app.db.get_or_create_rater: {_OWNER}")


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
    """
    raise NotImplementedError(f"app.db.store_submission: {_OWNER}")
