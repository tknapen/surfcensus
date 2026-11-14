"""``surfcensus export``: one table per analysis unit, from the catalog, the metric tables
and ``app.sqlite``. See docs/phases.md, phase 8.

Datasets appear by pseudonym and subjects by ``subject_uid``; a local run also names its
subjects. Nothing here can reach what is private (decision D20): the catalog is opened as a
``PublicCatalog``, and ``app.sqlite`` is read with the standard library, by the names of
``surfcensus.app``, so the export loads no package of the rater app.

Missing values. In the frames returned here a text column holds ``None``, a count or a flag
``pd.NA`` and a measurement ``NaN``. A parquet file holds a null for each of them, a CSV
file nothing between two commas. CSV has no types: a reader that guesses them turns an
identifier that happens to consist of digits into a number, so read identifiers as text
(``dtype=str`` in pandas), or use parquet.

Raters are named as they were given in a local run. In a census run they are ``R-01``,
``R-02``, ... in the order they were made; their names stay in ``app.sqlite``.
"""

from __future__ import annotations

import io
import json
import re
import sqlite3
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from surfcensus import __version__
from surfcensus._io import atomic_path, sha256_file
from surfcensus.app import APP_DB_FILE, LABEL_REASONS_KEY, label_reasons
from surfcensus.calibration import load_calibration
from surfcensus.config import ConfigError, load_config, stage_hashes
from surfcensus.log import get_logger
from surfcensus.protocol import CRITERION_IDS, PROTOCOL_FILE, load_protocol
from surfcensus.reference.strata import field_bin
from surfcensus.schemas import KEYS, NO_REFERENCE, SCHEMAS
from surfcensus.sources.local import (
    freesurfer_template_places,
    refuse_in_sources,
    stored_subjects_dirs,
)
from surfcensus.store import (
    PublicCatalog,
    clusters_with_native,
    primary_scores,
    read_table,
    table_path,
)

Format = Literal["csv", "parquet"]
Profile = Literal["analysis", "release"]
"""``analysis``: everything, pseudonymous, for the team's notebooks; the only profile of a
local run. ``release``: what a census publishes (coarser, selected subjects only)."""

EXPORT_DIRS: dict[Profile, str] = {"analysis": "export", "release": "release"}
"""Where a profile is written under the run directory when no other place is given."""

EXPORT_TABLES: tuple[str, ...] = (
    "subjects",
    "scores",
    "selection",
    "ratings",
    "marks",
    "final_marks",
    "clusters",
    "datasets",
    "behaviour",
    "skips",
)
"""Each is written as ``<name>.csv`` or ``<name>.parquet``."""
MANIFEST = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1

RATINGS_COLUMNS: tuple[str, ...] = (
    "assignment_id",
    "subject_uid",
    "rater",
    "kind",
    "seq",
    "blind",
    "render_key",
    "render_current",
    "A",
    "B",
    "C",
    "unratable_reason",
    "other",
    "note",
    "revision",
    "protocol_version",
    "protocol_sha256",
    "app_version",
    "client_ts",
    "server_ts",
    "ms_on_subject",
)
"""``ratings``: one row per completed assignment, from its current submission.
``render_current`` says whether the rating was made on the tiles the subject has now.
``seq`` is the place of the assignment among those of its rater, from 1: the order in which
the queue offered the subjects, which is the order of rating but for a subject that was
skipped or revised (``server_ts`` is when the current submission was stored). ``blind``
is the setting of the local queue when that submission was stored: false for the order of
the scores, worst first, true for the seeded shuffle; missing when the database has no
record of the setting before the submission. The queue of a census run does not read it."""

SUBJECT_RATING_COLUMNS: tuple[str, ...] = (
    "final_A",
    "final_B",
    "final_C",
    "final_method",
    "unratable_reason",
    "other",
    "n_ratings",
    "n_marks",
)
"""The last columns of ``subjects``: the final labels, whether any current submission
flagged another problem, and how many completed assignments and current marks the subject
has."""

BEHAVIOUR_COLUMNS: tuple[str, ...] = (
    "assignment_id",
    "subject_uid",
    "rater",
    "ms_on_subject",
    "n_frames_viewed",
    "n_plays",
    "n_pauses",
    "n_mode_switches",
    "n_back",
    "n_skips",
    "ms_skipped_visits",
    "ms_revised_visits",
)
"""``behaviour``: one row per completed assignment. The counts of events (``n_frames_viewed``
to ``n_back``) are over every visit to the subject in that assignment. The time is in three
columns, one per kind of visit, and their sum covers the same visits as the counts:
``ms_on_subject`` is the time of the visit that ended with the current submission;
``ms_skipped_visits`` the time of the visits that ended with a skip, together (``n_skips``
of them; missing where a skip event holds no time); ``ms_revised_visits`` the time of the
visits that ended with a submission that a revision superseded, together (0 when the rating
was never revised; missing where such a submission holds no time)."""

SKIPS_COLUMNS: tuple[str, ...] = (
    "assignment_id",
    "n",
    "subject_uid",
    "rater",
    "kind",
    "render_key",
    "client_ts",
    "server_ts",
    "ms_on_subject",
    "rated",
)
"""``skips``: one row per skip a rater made, from the ``skip`` events that the server
writes. ``n`` numbers the skips of an assignment from 1, ``ms_on_subject`` is the time of
the visit that ended with the skip, and ``rated`` says whether the assignment has a
submission now. An assignment that was skipped and not rated yet has rows here and none in
``ratings``."""

MARKS_COLUMNS: tuple[str, ...] = (
    "subject_uid",
    "rater",
    "assignment_id",
    "mark_id",
    "submission_id",
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
)
"""``marks``: one row per mark of a current submission. After the subject, the rater and
the assignment, every column of the ``marks`` table but ``deleted_ts``, which no such mark
has."""

FINAL_MARKS_COLUMNS: tuple[str, ...] = (
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
)
"""``final_marks``: as the table of ``app.sqlite``."""

DATASETS_COLUMNS: tuple[str, ...] = (
    "dataset",
    "source",
    "pipeline",
    "pipeline_version",
    "n_subjects",
    "modal_field_strength",
)
"""``datasets``: one row per dataset and pipeline. ``n_subjects`` is the number of subject
directories the source listed; ``modal_field_strength`` the most frequent field strength
among the dataset's present subjects that have one, the lower one on a tie."""

SUBJECT_COUNT_BINS: tuple[tuple[int, str], ...] = (
    (100, "100 or more"),
    (30, "30 to 99"),
    (10, "10 to 29"),
    (1, "1 to 9"),
)
"""``datasets.n_subjects`` of the release profile: the lowest count of each bin and its
label, largest first. The labels have no hyphen, which a spreadsheet program reads as a
date (``1-9``, ``10-29``)."""

RELEASE_LEFT_OUT: dict[str, tuple[str, ...]] = {
    "subjects": ("age", "sex", "manufacturer"),
    "datasets": ("pipeline_version",),
    "ratings": ("note",),
}
"""Columns that the release profile does not have."""

APP_SCHEMA_READ = 1
"""The schema of ``app.sqlite`` (``app_meta.schema_version``) whose tables this module
reads. ``app.db.APP_SCHEMA_VERSION`` is not imported, since that module imports
``sqlmodel``; a test keeps the two equal."""

_BUSY_S = 30.0
_APP_TABLES_READ = (
    "raters",
    "assignments",
    "submissions",
    "ratings",
    "marks",
    "events",
    "final_labels",
    "final_marks",
)
# strongest first: what a subject's final labels are said to come from when its rows differ
_METHODS = ("adjudicated", "consensus", "single")
_UNRATABLE = "unratable"
_SORT_KEYS: dict[str, tuple[str, ...]] = {
    "subjects": ("subject_uid",),
    "scores": KEYS["scores"],
    "selection": ("subject_uid", "reason"),
    "ratings": ("assignment_id",),
    "marks": ("mark_id",),
    "final_marks": ("final_mark_id",),
    "clusters": KEYS["clusters"],
    "datasets": ("dataset", "source", "pipeline"),
    "behaviour": ("assignment_id",),
    "skips": ("assignment_id", "n"),
}
_SKIP_EVENT = "skip"  # app.schemas.SKIP_EVENT, not imported: this module stays off the app
_CONFIG_CHANGE = "config_change"  # app.events.CONFIG_CHANGE, likewise
_BLIND_SETTING = "blind"
_COUNTED_EVENTS = {
    "play": "n_plays",
    "pause": "n_pauses",
    "mode": "n_mode_switches",
    "back": "n_back",
}
_EXPORT_FILE = re.compile(rf"(?:{'|'.join(EXPORT_TABLES)})\.(?:csv|parquet)")

_Kind = Literal["str", "int", "float", "bool"]
_Spec = tuple[tuple[str, _Kind], ...]

_RATING_ROWS: _Spec = (
    ("assignment_id", "int"),
    ("subject_uid", "str"),
    ("rater_id", "int"),
    ("kind", "str"),
    ("seq", "int"),
    ("render_key", "str"),
    ("submission_id", "int"),
    ("unratable_reason", "str"),
    ("other", "bool"),
    ("note", "str"),
    ("revision", "int"),
    ("protocol_version", "str"),
    ("protocol_sha256", "str"),
    ("app_version", "str"),
    ("client_ts", "str"),
    ("server_ts", "str"),
    ("ms_on_subject", "int"),
)
_MARK_KINDS: dict[str, _Kind] = {
    name: "str" for name in ("class", "panel", "mode", "hemi", "vertex_surface", "created_ts")
} | {
    name: "int"
    for name in (
        "mark_id",
        "submission_id",
        "slice",
        "frame",
        "pial_vertex",
        "white_vertex",
        "fsavg6_vertex",
    )
}
_FINAL_MARK_KINDS: dict[str, _Kind] = {
    "final_mark_id": "int",
    "subject_uid": "str",
    "class": "str",
    "hemi": "str",
    "pial_vertex": "int",
    "fsavg6_vertex": "int",
    "source_mark_ids": "str",
}
_SKIP_KINDS: dict[str, _Kind] = {
    "assignment_id": "int",
    "n": "int",
    "subject_uid": "str",
    "rater": "str",
    "kind": "str",
    "render_key": "str",
    "client_ts": "str",
    "server_ts": "str",
    "ms_on_subject": "int",
    "rated": "bool",
}
_SUBJECT_RATING_KINDS: dict[str, _Kind] = {
    "final_A": "int",
    "final_B": "int",
    "final_C": "int",
    "final_method": "str",
    "unratable_reason": "str",
    "other": "bool",
    "n_ratings": "int",
    "n_marks": "int",
}


# --- frames with fixed column types -------------------------------------------------------------


def _column(values: Sequence[object], kind: _Kind) -> pd.Series:
    """A column whose type does not depend on its values, and that can hold a missing one."""
    if kind == "str":
        return pd.Series(list(values), dtype=object)
    if kind == "int":
        return pd.Series(pd.array(list(values), dtype="Int64"))
    if kind == "bool":
        flags = [None if value is None else bool(value) for value in values]
        return pd.Series(pd.array(flags, dtype="boolean"))
    return pd.Series([float("nan") if value is None else value for value in values], dtype=float)


def _frame(spec: _Spec, rows: Sequence[Sequence[object]]) -> pd.DataFrame:
    return pd.DataFrame(
        {name: _column([row[i] for row in rows], kind) for i, (name, kind) in enumerate(spec)}
    )


def _spec(names: Sequence[str], kinds: Mapping[str, _Kind], default: _Kind = "float") -> _Spec:
    return tuple((name, kinds.get(name, default)) for name in names)


def _missing(value: object) -> bool:
    return value is None or value is pd.NA or (isinstance(value, float) and value != value)


def _holding_missing(column: pd.Series) -> pd.Series:
    """``column`` in a type that can hold a missing value: a flag as ``boolean``, a whole
    number as the nullable type of its width, text as ``object`` with None."""
    dtype = column.dtype
    if pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.StringDtype):
        return _text(column)
    if pd.api.types.is_extension_array_dtype(dtype) or pd.api.types.is_float_dtype(dtype):
        return column
    if pd.api.types.is_bool_dtype(dtype):
        return column.astype("boolean")
    return column.astype(str(dtype).capitalize())


def _text(column: pd.Series) -> pd.Series:
    return pd.Series([None if _missing(value) else value for value in column], dtype=object)


def _rows_at(frame: pd.DataFrame, key: str, keys: Sequence[str]) -> pd.DataFrame:
    """The other columns of ``frame`` in the rows whose ``key`` is each of ``keys``, in that
    order, with missing values where ``frame`` has no such row."""
    position = {value: at for at, value in enumerate(frame[key])}
    take = [position.get(value, -1) for value in keys]
    data = {}
    for name in frame.columns:
        if name == key:
            continue
        column = _holding_missing(frame[name])
        if pd.api.types.is_object_dtype(column.dtype):
            values = column.tolist()
            data[name] = pd.Series([values[at] if at >= 0 else None for at in take], dtype=object)
        else:
            taken = pd.api.extensions.take(column.array, take, allow_fill=True)
            data[name] = pd.Series(taken)
    return pd.DataFrame(data, index=pd.RangeIndex(len(keys)))


def _looked_up(keys: Sequence[str], values: Mapping[str, object], kind: _Kind) -> pd.Series:
    return _column([values.get(key) for key in keys], kind)


def _arrow_type(dtype: object) -> pa.DataType:
    if isinstance(dtype, pd.StringDtype) or pd.api.types.is_object_dtype(dtype):
        return pa.string()
    if pd.api.types.is_bool_dtype(dtype):
        return pa.bool_()
    for name in ("int8", "int16", "int32", "int64", "float32", "float64"):
        if str(dtype).lower() == name:
            return getattr(pa, name)()
    raise TypeError(f"no export type for a column of type {dtype}")


def _arrow_table(frame: pd.DataFrame, schema: pa.Schema | None = None) -> pa.Table:
    """``frame`` as a table of ``schema``, or of types that follow from the column types
    alone, every column nullable: a table without rows, and one whose column holds nothing,
    has the schema of a full one."""
    if schema is None:
        schema = pa.schema([pa.field(str(name), _arrow_type(frame[name].dtype)) for name in frame])
    else:
        schema = pa.schema([schema.field(str(name)) for name in frame])
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
    return table.replace_schema_metadata(None)


def frame_csv(frame: pd.DataFrame) -> str:
    """The text of a CSV file of ``frame``, as :func:`export_run` writes it: a header line,
    ``\\n`` line ends, nothing for a missing value, a field quoted where it holds a comma, a
    quote or a line end. ``GET /api/export`` answers with this for :func:`ratings_frame`."""
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue()


def _write_csv(path: Path, frame: pd.DataFrame, schema: pa.Schema | None) -> None:
    with atomic_path(path) as tmp:
        tmp.write_text(frame_csv(frame), encoding="utf-8", newline="")


def _write_parquet(path: Path, frame: pd.DataFrame, schema: pa.Schema | None) -> None:
    table = _arrow_table(frame, schema)
    with atomic_path(path) as tmp:
        pq.write_table(table, tmp, compression="zstd")


_WRITERS: dict[str, Callable[[Path, pd.DataFrame, pa.Schema | None], None]] = {
    "csv": _write_csv,
    "parquet": _write_parquet,
}
# the tables of a run keep their schema; the others get one from their column types
_PARQUET_SCHEMAS: dict[str, pa.Schema] = {
    "scores": SCHEMAS["scores"],
    "clusters": pa.schema(
        [
            *SCHEMAS["clusters"],
            *(
                field.with_nullable(True)
                for field in SCHEMAS["clusters_native"]
                if field.name not in SCHEMAS["clusters"].names
            ),
        ]
    ),
}


# --- what is read -------------------------------------------------------------------------------


@dataclass
class _Sources:
    """The public side of a run, open for the length of one export or one frame."""

    run_dir: Path
    public: PublicCatalog
    app: sqlite3.Connection | None
    """``app.sqlite`` in one read transaction; None for a run that has no ratings database."""

    @property
    def local(self) -> bool:
        return self.public.mode() == "local"

    def query(self, sql: str) -> list[tuple[Any, ...]]:
        return [] if self.app is None else self.app.execute(sql).fetchall()


def _no_app_db(why: str) -> ConfigError:
    return ConfigError("invalid", f"{APP_DB_FILE} is no database of the rater app: {why}")


def _connect(path: Path) -> sqlite3.Connection:
    """A connection that creates no file. Read-write where that is possible: a writer that
    was killed in a commit leaves a journal, which only a connection that may write rolls
    back, and SQLite refuses to read until then."""
    uri = path.absolute().as_uri()
    for access in ("rw", "ro"):
        connection = sqlite3.connect(
            f"{uri}?mode={access}", uri=True, timeout=_BUSY_S, isolation_level=None
        )
        try:
            connection.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchall()
        except sqlite3.OperationalError as exc:
            connection.close()
            code = getattr(exc, "sqlite_errorcode", None)
            busy = isinstance(code, int) and code & 0xFF in (
                sqlite3.SQLITE_BUSY,
                sqlite3.SQLITE_LOCKED,
            )
            if busy or access == "ro":
                raise
        except BaseException:
            connection.close()
            raise
        else:
            return connection
    raise AssertionError("unreachable")


def _is_of_run(connection: sqlite3.Connection, run_id: str) -> bool:
    """Whether the database holds the ratings of this run. False for a file that holds
    nothing yet (its making was interrupted); ``ConfigError("invalid")`` for the database
    of something else, of another run, or of a schema this version does not read."""

    def query(sql: str) -> list[tuple[Any, ...]]:
        return connection.execute(sql).fetchall()

    if not query("SELECT 1 FROM sqlite_master LIMIT 1"):
        return False
    tables = {str(row[0]) for row in query("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if "app_meta" not in tables:
        raise _no_app_db("it has no table app_meta")
    columns = {str(row[1]) for row in query('PRAGMA table_info("app_meta")')}
    if not {"key", "value"} <= columns:
        raise _no_app_db("its table app_meta has other columns")
    meta = {str(key): value for key, value in query("SELECT key, value FROM app_meta")}
    version = meta.get("schema_version")
    if not (isinstance(version, str) and version.isascii() and version.isdecimal()):
        raise _no_app_db("app_meta has no schema number")
    if meta.get("run_id") != run_id:
        raise ConfigError(
            "invalid", f"{APP_DB_FILE} holds the ratings of another run; this run is {run_id}"
        )
    if int(version) > APP_SCHEMA_READ:
        raise ConfigError(
            "invalid",
            f"{APP_DB_FILE} has schema {version}; this version of surfcensus reads up to "
            f"{APP_SCHEMA_READ}",
        )
    if int(version) < APP_SCHEMA_READ:
        raise ConfigError(
            "invalid",
            f"{APP_DB_FILE} has schema {version}, of an earlier version of surfcensus; the "
            "rater app brings it up when it next opens the run",
        )
    lacking = [name for name in _APP_TABLES_READ if name not in tables]
    if lacking:
        raise _no_app_db(f"it has no table {lacking[0]}")
    return True


def _unreadable(exc: sqlite3.Error) -> ConfigError:
    code = getattr(exc, "sqlite_errorcode", None)
    primary = code & 0xFF if isinstance(code, int) else None
    if primary in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
        return ConfigError(
            "invalid", f"{APP_DB_FILE} is in use by another process; try again when that is done"
        )
    if primary in (sqlite3.SQLITE_NOTADB, sqlite3.SQLITE_CORRUPT):
        return ConfigError("invalid", f"{APP_DB_FILE} is not a SQLite database: {exc}")
    return ConfigError("invalid", f"{APP_DB_FILE} cannot be read: {exc}")


@contextmanager
def _sources(run_dir: Path | str) -> Iterator[_Sources]:
    """Open the catalog and, when the run has one, ``app.sqlite``. Every table of the app
    is then read in one transaction, so the tables of an export fit together while raters
    go on rating."""
    run_dir = Path(run_dir)
    with PublicCatalog.open(run_dir) as public:
        path = run_dir / APP_DB_FILE
        if not path.exists():
            yield _Sources(run_dir, public, None)
            return
        run_id = public.run_id()
        try:
            connection = _connect(path)
        except sqlite3.Error as exc:
            raise _unreadable(exc) from exc
        try:
            connection.execute("BEGIN")
            usable = _is_of_run(connection, run_id)
            yield _Sources(run_dir, public, connection if usable else None)
        except sqlite3.Error as exc:
            raise _unreadable(exc) from exc
        finally:
            connection.close()


def _metric_table(run_dir: Path, name: str) -> pd.DataFrame:
    """A table of the metrics as this version writes it; empty for a run that has none."""
    try:
        return read_table(table_path(run_dir, name), SCHEMAS[name])
    except (ValueError, OSError, pa.ArrowException) as exc:
        # the reason without the path, which the error of the system would name
        why = (exc.strerror if isinstance(exc, OSError) else None) or exc
        raise ConfigError("invalid", f"tables/{name}.parquet cannot be used: {why}") from exc


def _rater_names(src: _Sources) -> dict[int, str]:
    """``rater_id`` to the name a table shows: the rater's own in a local run, ``R-01``,
    ``R-02``, ... in the order of ``rater_id`` in a census run."""
    rows = src.query("SELECT rater_id, name FROM raters ORDER BY rater_id")
    if src.local:
        return {int(rater_id): str(name) for rater_id, name in rows}
    return {int(rater_id): f"R-{number:02d}" for number, (rater_id, _) in enumerate(rows, 1)}


# --- the tables ---------------------------------------------------------------------------------


def _ratings(src: _Sources) -> pd.DataFrame:
    rows = src.query(
        "SELECT a.assignment_id, a.subject_uid, a.rater_id, a.kind, a.seq, a.render_key, "
        "s.submission_id, s.unratable_reason, s.other, s.note, s.revision, "
        "s.protocol_version, s.protocol_sha256, s.app_version, s.client_ts, s.server_ts, "
        "s.ms_on_subject "
        "FROM assignments AS a JOIN submissions AS s "
        "ON s.assignment_id = a.assignment_id AND s.current = 1 "
        "WHERE a.completed_at IS NOT NULL ORDER BY a.assignment_id"
    )
    frame = _frame(_RATING_ROWS, rows)
    names = _rater_names(src)
    frame["rater"] = _column([names.get(rater_id) for rater_id in frame["rater_id"]], "str")
    recorded = _blind_records(src)
    frame["blind"] = _column([_blind_at(recorded, stored) for stored in frame["server_ts"]], "bool")
    renders = src.public.renders()
    done = renders[renders["status"] == "done"]
    shown_now = dict(zip(done["subject_uid"], done["render_key"], strict=True))
    frame["render_current"] = _column(
        [
            key is not None and shown_now.get(uid) == key
            for uid, key in zip(frame["subject_uid"], frame["render_key"], strict=True)
        ],
        "bool",
    )
    values = {
        (int(submission), str(criterion)): value
        for submission, criterion, value in src.query(
            "SELECT r.submission_id, r.criterion, r.value FROM ratings AS r "
            "JOIN submissions AS s ON s.submission_id = r.submission_id WHERE s.current = 1"
        )
    }
    for criterion in CRITERION_IDS:
        frame[criterion] = _column(
            [values.get((submission, criterion)) for submission in frame["submission_id"]], "int"
        )
    return frame[list(RATINGS_COLUMNS)]


def _blind_records(src: _Sources) -> list[tuple[str, bool]]:
    """``(server time, value)`` of every record of the setting ``blind`` among the events of
    the run, in the order they were written. The server writes one when it starts with
    another value than the last record, and ``review`` one when the setting is changed
    (``app.events.record_setting``). A payload of another form is no record."""
    records: list[tuple[str, bool]] = []
    for ts, text in src.query(
        "SELECT ts, payload_json FROM events "
        f"WHERE assignment_id IS NULL AND type = '{_CONFIG_CHANGE}' ORDER BY event_id"
    ):
        try:
            payload = json.loads(text) if isinstance(text, str) else None
        except ValueError:
            continue
        if not isinstance(payload, dict) or payload.get("setting") != _BLIND_SETTING:
            continue
        value = payload.get("new")
        if isinstance(ts, str) and type(value) is bool:
            records.append((ts, value))
    return records


def _blind_at(records: Sequence[tuple[str, bool]], server_ts: object) -> bool | None:
    """The last recorded value of ``blind`` that is no later than ``server_ts``, or None.
    Both times are the server's, in the one form the tables store, so they compare as text."""
    if not isinstance(server_ts, str):
        return None
    value = None
    for ts, recorded in records:
        if ts <= server_ts:
            value = recorded
    return value


def _marks(src: _Sources) -> pd.DataFrame:
    table_columns = MARKS_COLUMNS[3:]
    selected = ", ".join(f'm."{name}"' for name in table_columns)
    rows = src.query(
        f"SELECT a.subject_uid, a.rater_id, a.assignment_id, {selected} "
        "FROM marks AS m JOIN submissions AS s ON s.submission_id = m.submission_id "
        "JOIN assignments AS a ON a.assignment_id = s.assignment_id "
        "WHERE s.current = 1 AND a.completed_at IS NOT NULL ORDER BY m.mark_id"
    )
    spec: _Spec = (
        ("subject_uid", "str"),
        ("rater_id", "int"),
        ("assignment_id", "int"),
        *_spec(table_columns, _MARK_KINDS),
    )
    frame = _frame(spec, rows)
    names = _rater_names(src)
    frame["rater"] = _column([names.get(rater_id) for rater_id in frame["rater_id"]], "str")
    return frame[list(MARKS_COLUMNS)]


def _final_marks(src: _Sources) -> pd.DataFrame:
    selected = ", ".join(f'"{name}"' for name in FINAL_MARKS_COLUMNS)
    rows = src.query(f"SELECT {selected} FROM final_marks ORDER BY final_mark_id")
    return _frame(_spec(FINAL_MARKS_COLUMNS, _FINAL_MARK_KINDS), rows)


def _frames_viewed(payloads: Sequence[object]) -> int:
    """How many different frames the ``frames`` events of an assignment name, a frame of the
    synchronized mode and the frame of that number of the other mode counted apart. A
    payload is what a page sent: one that is not of the documented form counts nothing."""
    seen: set[tuple[str, int]] = set()
    for text in payloads:
        try:
            payload = json.loads(text) if isinstance(text, str) else None
        except ValueError:
            continue
        visits = payload.get("visits") if isinstance(payload, dict) else None
        if not isinstance(visits, list):
            continue
        mode = str(payload.get("mode"))
        for visit in visits:
            if isinstance(visit, list) and visit and type(visit[0]) is int:
                seen.add((mode, visit[0]))
    return len(seen)


def _behaviour(src: _Sources, ratings: pd.DataFrame) -> pd.DataFrame:
    counts: dict[tuple[int, str], int] = {
        (int(assignment), str(kind)): int(n)
        for assignment, kind, n in src.query(
            "SELECT assignment_id, type, COUNT(*) FROM events "
            "WHERE assignment_id IS NOT NULL GROUP BY assignment_id, type"
        )
    }
    payloads: dict[int, list[object]] = {}
    for assignment, payload in src.query(
        "SELECT assignment_id, payload_json FROM events "
        "WHERE assignment_id IS NOT NULL AND type = 'frames' ORDER BY event_id"
    ):
        payloads.setdefault(int(assignment), []).append(payload)
    assignments = [int(assignment) for assignment in ratings["assignment_id"]]
    frame = ratings[["assignment_id", "subject_uid", "rater", "ms_on_subject"]].reset_index(
        drop=True
    )
    frame["n_frames_viewed"] = _column(
        [_frames_viewed(payloads.get(assignment, ())) for assignment in assignments], "int"
    )
    for kind, name in _COUNTED_EVENTS.items():
        frame[name] = _column(
            [counts.get((assignment, kind), 0) for assignment in assignments], "int"
        )
    skipped = {
        int(assignment): int(count)
        for assignment, count in src.query("SELECT assignment_id, skip_count FROM assignments")
    }
    frame["n_skips"] = _column([skipped.get(assignment, 0) for assignment in assignments], "int")
    spent: dict[int, list[int | None]] = {}
    for assignment, _, ms, _ in _skip_events(src):
        spent.setdefault(assignment, []).append(ms)
    frame["ms_skipped_visits"] = _column(
        [_sum_or_none(spent.get(assignment, [])) for assignment in assignments], "int"
    )
    # a revision is a new submission, and the time of the visit that made the rating it
    # replaced is in no other table
    revised: dict[int, list[int | None]] = {}
    for assignment, ms in src.query(
        "SELECT assignment_id, ms_on_subject FROM submissions WHERE current = 0 "
        "ORDER BY submission_id"
    ):
        revised.setdefault(int(assignment), []).append(_whole(ms))
    frame["ms_revised_visits"] = _column(
        [_sum_or_none(revised.get(assignment, [])) for assignment in assignments], "int"
    )
    return frame[list(BEHAVIOUR_COLUMNS)]


def _sum_or_none(values: Sequence[int | None]) -> int | None:
    """The sum, or None when a value is not known; 0 for none."""
    known = [value for value in values if value is not None]
    return sum(known) if len(known) == len(values) else None


def _whole(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _skip_events(src: _Sources) -> list[tuple[int, str | None, int | None, dict[str, Any]]]:
    """``(assignment_id, client time, ms on the subject, payload)`` of every ``skip`` event,
    in the order they were written. A payload that is not of the form the server writes
    gives an empty one."""
    events = []
    for assignment, ts, text in src.query(
        "SELECT assignment_id, ts, payload_json FROM events "
        f"WHERE assignment_id IS NOT NULL AND type = '{_SKIP_EVENT}' ORDER BY event_id"
    ):
        try:
            payload = json.loads(text) if isinstance(text, str) else None
        except ValueError:
            payload = None
        payload = payload if isinstance(payload, dict) else {}
        events.append(
            (
                int(assignment),
                ts if isinstance(ts, str) else None,
                _whole(payload.get("ms_on_subject")),
                payload,
            )
        )
    return events


def _skips(src: _Sources) -> pd.DataFrame:
    assignments = {
        int(assignment): (str(uid), int(rater_id), str(kind), key, completed is not None)
        for assignment, uid, rater_id, kind, key, completed in src.query(
            "SELECT assignment_id, subject_uid, rater_id, kind, render_key, completed_at "
            "FROM assignments"
        )
    }
    names = _rater_names(src)
    rows = []
    position: dict[int, int] = {}
    for assignment, client_ts, ms, payload in _skip_events(src):
        if assignment not in assignments:
            continue
        uid, rater_id, kind, key, rated = assignments[assignment]
        position[assignment] = position.get(assignment, 0) + 1
        number = _whole(payload.get("n"))
        server_ts = payload.get("server_ts")
        rows.append(
            (
                assignment,
                position[assignment] if number is None else number,
                uid,
                names.get(rater_id),
                kind,
                key,
                client_ts,
                server_ts if isinstance(server_ts, str) else None,
                ms,
                rated,
            )
        )
    return _frame(_spec(SKIPS_COLUMNS, _SKIP_KINDS), rows)


def _label_reasons(src: _Sources) -> tuple[tuple[str, ...], str]:
    """The reasons that the values of the ``unratable`` final labels are positions in, and
    what holds them: the list that ``app.sqlite`` keeps with its labels, so that a label
    means what it was written as whatever ``protocol.yaml`` says now; for a database that
    keeps none, the list of the protocol."""
    kept = src.query(f"SELECT value FROM app_meta WHERE key = '{LABEL_REASONS_KEY}'")
    if not kept:
        return load_protocol(src.run_dir)[0].unratable_reasons, PROTOCOL_FILE
    try:
        return label_reasons(kept[0][0]), f"the list that {APP_DB_FILE} keeps"
    except ValueError as exc:
        raise ConfigError("invalid", f"{APP_DB_FILE}: {exc}") from exc


def _unratable_reasons(src: _Sources, labels: Sequence[tuple[Any, ...]]) -> dict[str, str]:
    """Subject to the reason it cannot be rated, for the final labels of criterion
    ``unratable``, whose value is the position of the reason in :func:`_label_reasons`."""
    positions = {str(uid): value for uid, criterion, value, _ in labels if criterion == _UNRATABLE}
    if not positions:
        return {}
    reasons, where = _label_reasons(src)
    named = {}
    for uid, position in positions.items():
        if type(position) is not int or not 0 <= position < len(reasons):
            raise ConfigError(
                "invalid",
                f"the final label of {uid} names unratable reason {position!r}, and "
                f"{where} has the reasons 0 to {len(reasons) - 1}",
            )
        named[uid] = reasons[position]
    return named


def _subject_ratings(src: _Sources, uids: Sequence[str]) -> pd.DataFrame:
    """:data:`SUBJECT_RATING_COLUMNS` for the subjects ``uids``; nothing in any of them for
    a run without ``app.sqlite``."""
    if src.app is None:
        return pd.DataFrame(
            {
                name: _column([None] * len(uids), kind)
                for name, kind in _SUBJECT_RATING_KINDS.items()
            }
        )
    labels = src.query(
        "SELECT subject_uid, criterion, value, method FROM final_labels "
        "ORDER BY subject_uid, criterion"
    )
    final: dict[str, dict[str, object]] = {name: {} for name in _SUBJECT_RATING_KINDS}
    methods: dict[str, set[str]] = {}
    for uid, criterion, value, method in labels:
        methods.setdefault(str(uid), set()).add(str(method))
        if criterion in CRITERION_IDS:
            final[f"final_{criterion}"][str(uid)] = value
    for uid, found in methods.items():
        known = [method for method in _METHODS if method in found]
        final["final_method"][uid] = known[0] if known else sorted(found)[0]
    final["unratable_reason"] = dict(_unratable_reasons(src, labels))
    n_ratings: dict[str, object] = {uid: 0 for uid in uids}
    for uid, n, other in src.query(
        "SELECT a.subject_uid, COUNT(*), MAX(s.other) FROM assignments AS a "
        "JOIN submissions AS s ON s.assignment_id = a.assignment_id AND s.current = 1 "
        "WHERE a.completed_at IS NOT NULL GROUP BY a.subject_uid"
    ):
        n_ratings[str(uid)] = n
        final["other"][str(uid)] = other
    n_marks: dict[str, object] = {uid: 0 for uid in uids}
    for uid, n in src.query(
        "SELECT a.subject_uid, COUNT(*) FROM marks AS m "
        "JOIN submissions AS s ON s.submission_id = m.submission_id "
        "JOIN assignments AS a ON a.assignment_id = s.assignment_id "
        "WHERE s.current = 1 AND a.completed_at IS NOT NULL GROUP BY a.subject_uid"
    ):
        n_marks[str(uid)] = n
    final["n_ratings"], final["n_marks"] = n_ratings, n_marks
    return pd.DataFrame(
        {name: _looked_up(uids, final[name], kind) for name, kind in _SUBJECT_RATING_KINDS.items()}
    )


def _subjects(src: _Sources) -> pd.DataFrame:
    subjects = src.public.subjects()
    if not src.local:
        # empty in a census run, whose labels stay in the catalog
        subjects = subjects.drop(columns=["subject_label", "participant"])
    uids = [str(uid) for uid in subjects["subject_uid"]]
    renders = src.public.renders()
    by_uid = {
        name: dict(zip(renders["subject_uid"], renders[name], strict=True))
        for name in ("status", "reason")
    }
    rendered = pd.DataFrame(
        {
            "render_status": _looked_up(uids, by_uid["status"], "str"),
            "render_reason": _looked_up(uids, by_uid["reason"], "str"),
        }
    )
    _metric_table(src.run_dir, "scores")
    scores = primary_scores(src.run_dir)
    # the dataset and the holes of the log are columns of the subject already
    new = [name for name in scores.columns if name == "subject_uid" or name not in subjects]
    scored = _rows_at(scores[new], "subject_uid", uids)
    parts = [subjects.reset_index(drop=True), rendered, scored, _subject_ratings(src, uids)]
    return pd.concat(parts, axis=1)


def _scores(src: _Sources) -> pd.DataFrame:
    return _metric_table(src.run_dir, "scores")


def _clusters(src: _Sources) -> pd.DataFrame:
    for name in ("clusters", "clusters_native"):
        _metric_table(src.run_dir, name)
    clusters = clusters_with_native(src.run_dir)
    # the join leaves the columns of a cluster without a native position in other types
    clusters["native_vertex"] = pd.Series(
        pd.array(
            [None if _missing(value) else int(value) for value in clusters["native_vertex"]],
            dtype="Int32",
        )
    )
    clusters["template_id"] = _text(clusters["template_id"])
    return clusters


def _mode_of(values: Sequence[float]) -> float | None:
    counts: dict[float, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return min(counts, key=lambda value: (-counts[value], value)) if counts else None


def _binned_field(field_strength: float | None) -> float | None:
    tesla = field_bin(field_strength)
    return None if tesla is None else float(tesla)


def _known(values: pd.Series) -> list[float | None]:
    return [None if _missing(value) else float(value) for value in values]


def _datasets(src: _Sources, *, release: bool) -> pd.DataFrame:
    datasets = src.public.datasets()
    subjects = src.public.subjects()
    fields: dict[str, list[float]] = {}
    for dataset, present, field in zip(
        subjects["dataset"], subjects["present"], _known(subjects["field_strength"]), strict=True
    ):
        value = _binned_field(field) if release else field
        if present and value is not None:
            fields.setdefault(str(dataset), []).append(value)
    listed = [None if _missing(n) else int(n) for n in datasets["n_subjects_listed"]]
    return pd.DataFrame(
        {
            "dataset": _text(datasets["pseudonym"]),
            "source": _text(datasets["source"]),
            "pipeline": _text(datasets["pipeline"]),
            "pipeline_version": _text(datasets["pipeline_version"]),
            "n_subjects": _column(listed, "int"),
            "modal_field_strength": _column(
                [_mode_of(fields.get(str(dataset), ())) for dataset in datasets["pseudonym"]],
                "float",
            ),
        }
    )[list(DATASETS_COLUMNS)]


def _count_bin(count: int | None) -> str | None:
    if count is None:
        return None
    for lowest, label in SUBJECT_COUNT_BINS:
        if count >= lowest:
            return label
    return "0"


def _two_digits(value: float) -> float:
    return float(f"{value:.2g}")


def _released(
    tables: dict[str, pd.DataFrame], selected: Collection[str]
) -> dict[str, pd.DataFrame]:
    """The tables of the release profile from those of the analysis profile: selected
    subjects only, coarser descriptors, and without the columns of
    :data:`RELEASE_LEFT_OUT`. The datasets table is made for the profile by
    :func:`_datasets`, since its field strength is the mode of binned values."""
    out = {}
    for name, frame in tables.items():
        if "subject_uid" in frame.columns:
            frame = frame[frame["subject_uid"].isin(list(selected))]
        frame = frame.drop(columns=list(RELEASE_LEFT_OUT.get(name, ()))).reset_index(drop=True)
        out[name] = frame
    subjects = out["subjects"]
    subjects["field_strength"] = _column(
        [_binned_field(value) for value in _known(subjects["field_strength"])], "float"
    )
    selection = out["selection"]
    selection["inclusion_prob"] = _column(
        [_two_digits(float(value)) for value in selection["inclusion_prob"]], "float"
    )
    datasets = out["datasets"]
    datasets["n_subjects"] = _column(
        [_count_bin(None if _missing(n) else int(n)) for n in datasets["n_subjects"]], "str"
    )
    return out


def _tables(src: _Sources, profile: Profile, scores: pd.DataFrame) -> dict[str, pd.DataFrame]:
    release = profile == "release"
    ratings = _ratings(src)
    tables = {
        "subjects": _subjects(src),
        "scores": scores,
        "selection": src.public.selection(),
        "ratings": ratings,
        "marks": _marks(src),
        "final_marks": _final_marks(src),
        "clusters": _clusters(src),
        "datasets": _datasets(src, release=release),
        "behaviour": _behaviour(src, ratings),
        "skips": _skips(src),
    }
    if release:
        tables = _released(tables, set(src.public.selection()["subject_uid"]))
    return {
        name: tables[name].sort_values(list(_SORT_KEYS[name]), kind="stable").reset_index(drop=True)
        for name in EXPORT_TABLES
    }


def ratings_frame(run_dir: Path | str) -> pd.DataFrame:
    """The ``ratings`` table (:data:`RATINGS_COLUMNS`), sorted by ``assignment_id``; also
    behind ``GET /api/export`` and the ratings of the review screen. Empty, with its
    columns, for a run without ``app.sqlite``.

    ``A``, ``B`` and ``C`` are missing in an unratable rating and for a criterion the
    protocol does not have. ``kind`` is that of the assignment, so a repeat says so here.
    The database is read through a connection of its own: what another connection has not
    committed is not in the frame.

    Raises ``ConfigError("invalid", ...)`` for an ``app.sqlite`` that cannot be read, is in
    use for longer than half a minute, or is not that of this run.
    """
    with _sources(run_dir) as src:
        return _ratings(src)


def marks_frame(run_dir: Path | str) -> pd.DataFrame:
    """The ``marks`` table (:data:`MARKS_COLUMNS`), sorted by ``mark_id``: the marks of the
    current submissions of completed assignments, with the rater named as in
    :func:`ratings_frame`. Empty, with its columns, for a run without ``app.sqlite``.
    Raises what :func:`ratings_frame` raises."""
    with _sources(run_dir) as src:
        return _marks(src)


def subjects_frame(run_dir: Path | str) -> pd.DataFrame:
    """The ``subjects`` table, sorted by ``subject_uid``; also behind ``GET /api/review``.

    ``subject_uid``, ``dataset``, in a local run ``subject_label`` and ``participant``, the
    provenance columns of ``PublicCatalog.subjects()``, ``render_status``,
    ``render_reason``, every column of the primary row of ``scores.parquet``, then
    :data:`SUBJECT_RATING_COLUMNS`. The rating columns are empty for a run without
    ``app.sqlite``.

    Every subject of the catalog has a row, also one that the source no longer lists
    (``present``). ``render_status`` is ``done`` or ``failed`` as the catalog has it, and
    missing for a subject without a render. ``dataset``, ``holes_lh`` and ``holes_rh`` are
    in the catalog and in the scores and appear once. In a run with ``app.sqlite``,
    ``n_ratings`` and ``n_marks`` are 0 for a subject nobody rated, and ``other`` is
    missing for it. ``n_ratings`` counts completed assignments of every kind.
    ``final_method`` is the one method of the subject's final labels, and when they differ
    the first of ``adjudicated``, ``consensus``, ``single``.

    Raises ``ConfigError``: as :func:`ratings_frame`; for a metric table that cannot be
    read as this version writes it; and when a final label says a subject is unratable and
    ``protocol.yaml`` cannot say for what reason.
    """
    with _sources(run_dir) as src:
        return _subjects(src)


def _manifest(src: _Sources, profile: Profile, fmt: Format, scores: pd.DataFrame) -> dict[str, Any]:
    template_id = src.public.meta("template_id")
    references = sorted({str(ref) for ref in scores["reference_id"]} - {NO_REFERENCE})
    try:
        protocol, protocol_sha256 = load_protocol(src.run_dir)
    except ConfigError as exc:
        if exc.reason != "missing":
            raise
        described = None
    else:
        described = {"version": protocol.version, "sha256": protocol_sha256}
    calibration = load_calibration(src.run_dir)
    config = load_config(src.run_dir)
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": src.public.run_id(),
        "mode": src.public.mode(),
        "profile": profile,
        "format": fmt,
        "key_fingerprint": src.public.meta("key_fingerprint"),
        "surfcensus_version": __version__,
        "stage_hashes": stage_hashes(
            config, template_id=template_id, with_reference=bool(references)
        ),
        # the settings as they are now; `ratings.blind` says under which each rating was made
        "queue": {
            "blind": config.app.blind,
            "queue_seed": config.app.queue_seed,
            "repeat_rate": config.app.repeat_rate,
            "repeat_gap": config.app.repeat_gap,
        },
        "template_id": template_id,
        "reference_ids": references,
        "protocol": described,
        "calibration": None
        if calibration is None
        else {"version": calibration[0].version, "sha256": calibration[1]},
    }


def _refuse_in_sources(run_dir: Path, target: Path) -> None:
    """Raise ``ConfigError("invalid", ...)`` when ``target`` is, or lies inside, a directory
    that a source of a local run reads (``sources.local.refuse_in_sources``): its subjects
    directory, what a subject or a template in it links to, and the fsaverage directory of
    a FreeSurfer install. Sources are only read. The roots are those of ``config.yaml``;
    where a root holds no subjects now (the data is not mounted), ``target`` is compared
    with the root itself. A census run keeps its roots in the private catalog, out of reach
    of this module; the command checks it through ``Run.refuse_output_in_sources``."""
    refuse_in_sources(
        target,
        stored_subjects_dirs(load_config(run_dir)),
        what="output",
        also=freesurfer_template_places(),
    )


def _refuse_in_run(run_dir: Path, target: Path) -> None:
    """Raise ``ConfigError("invalid", ...)`` when ``target`` is the run directory or lies
    inside it, other than in its ``export/`` or ``release/``: the tables of an export have
    names that the run's own tables have (``tables/clusters.parquet``), and an export over
    them leaves a run that no phase can read."""
    run, resolved = run_dir.resolve(), target.resolve()
    if not resolved.is_relative_to(run):
        return
    if any(resolved.is_relative_to(run / name) for name in EXPORT_DIRS.values()):
        return
    places = " or ".join(f"{name}/" for name in EXPORT_DIRS.values())
    raise ConfigError(
        "invalid",
        f"the output directory {target} lies inside the run directory {run_dir}, where "
        f"the run keeps its own files; choose its {places}, or a directory outside the run",
    )


def _forget_earlier(target: Path, writing: Collection[str]) -> None:
    """Remove the manifest of an earlier export in ``target`` and the table files it lists
    that this export does not write again (those of the other format). A file that no
    manifest lists is left alone, whatever its name: the directory may be one the user
    keeps other things in."""
    manifest = target / MANIFEST
    try:
        listed = json.loads(manifest.read_text(encoding="utf-8"))["files"]
        names = [name for name in listed if isinstance(name, str)]
    except (OSError, ValueError, KeyError, TypeError):
        names = []
    # first: from here until the new manifest is written, the directory holds no export
    manifest.unlink(missing_ok=True)
    for name in names:
        path = target / name
        if _EXPORT_FILE.fullmatch(name) and name not in writing:
            if path.is_file() or path.is_symlink():
                path.unlink()
                get_logger().info("removed %s of the earlier export", name)


def export_run(
    run_dir: Path | str,
    out_dir: Path | str | None = None,
    *,
    fmt: Format = "csv",
    profile: Profile = "analysis",
) -> list[Path]:
    """Write every table of :data:`EXPORT_TABLES` and ``manifest.json`` into ``out_dir``
    (default: ``export/`` or ``release/`` of the run directory), atomically and sorted;
    returns the paths written, the manifest last.

    Works on a run without ratings and on one without ``app.sqlite``. The manifest holds
    the run id, mode, profile, key fingerprint, package version, stage hashes, template id,
    reference ids, the versions and hashes of the protocol and the calibration set, how the
    rating queue is ordered (``queue``: ``blind``, ``queue_seed``, ``repeat_rate`` and
    ``repeat_gap`` of the configuration as it is now), and the row count and SHA-256 of
    every exported file. It has no reveal argument and cannot reach an accession.

    Every table is made before anything is written, so an export that is refused leaves the
    directory as it was. The manifest of an earlier export is removed first and the new one
    written last: a directory with a manifest holds the files it lists, with the hashes it
    gives. An export in the other format removes the table files that the earlier manifest
    lists. The manifest has no time in it, so a run that did not change gives the same
    bytes again.

    The stage hashes are those of the present configuration and template
    (``config.stage_hashes``), with ``reference`` and ``native`` when the scores name a
    reference.

    The release profile (a census run only): every table holds selected subjects only; the
    columns of :data:`RELEASE_LEFT_OUT` are left out; ``field_strength`` is the bin (1.5, 3
    or 7), ``datasets.n_subjects`` one of :data:`SUBJECT_COUNT_BINS` and
    ``selection.inclusion_prob`` rounded to two significant digits.

    Refused before anything is read or written: a target (``out_dir`` or the default) that
    is, or lies inside, a directory that a source of a local run reads, since a source is
    only read (:func:`_refuse_in_sources`); and an ``out_dir`` in the run directory that is
    not in its ``export/`` or ``release/``, where an export would replace files of the run.

    Raises ``ConfigError("invalid", ...)`` for the release profile of a local run, for such
    a target and for what :func:`subjects_frame` raises it for, ``ConfigError("missing",
    ...)`` for a directory without a run, ``ValueError`` for a format or profile there is
    none of, and ``OSError`` for a place that cannot be written.
    """
    if fmt not in _WRITERS:
        raise ValueError(f"an export is csv or parquet, not {fmt!r}")
    if profile not in EXPORT_DIRS:
        raise ValueError(f"the export profiles are analysis and release, not {profile!r}")
    run_dir = Path(run_dir)
    target = run_dir / EXPORT_DIRS[profile] if out_dir is None else Path(out_dir)
    # also the default: a run directory that was moved into its source gets no export
    _refuse_in_sources(run_dir, target)
    _refuse_in_run(run_dir, target)
    with _sources(run_dir) as src:
        if profile == "release" and src.local:
            raise ConfigError(
                "invalid",
                "a local run has the analysis profile only; the release profile is what a "
                "census publishes",
            )
        scores = _scores(src)
        tables = _tables(src, profile, scores)
        manifest = _manifest(src, profile, fmt, scores)
    target.mkdir(parents=True, exist_ok=True)
    names = {name: f"{name}.{fmt}" for name in EXPORT_TABLES}
    _forget_earlier(target, set(names.values()))
    paths = []
    files = {}
    for name in EXPORT_TABLES:
        path = target / names[name]
        _WRITERS[fmt](path, tables[name], _PARQUET_SCHEMAS.get(name))
        files[path.name] = {"rows": len(tables[name]), "sha256": sha256_file(path)}
        paths.append(path)
    manifest["files"] = files
    with atomic_path(target / MANIFEST) as tmp:
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return [*paths, target / MANIFEST]
