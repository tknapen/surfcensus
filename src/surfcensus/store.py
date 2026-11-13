"""The catalog of a run (``catalog.sqlite``) and the parquet tables under ``tables/``.

Two classes read the same file. :class:`Catalog` is the private side: accessions, roots,
file locations, input signatures, failure details. :class:`PublicCatalog` is what the code
downstream of inventory may hold; it returns none of those (decision D10).

The catalog is SQLite through the standard library, with ``PRAGMA user_version`` migrations
(decision D18). A local run keeps the default rollback journal, because run directories on
network file systems are common and WAL does not work there; a census run uses WAL. Every
write is one short ``BEGIN IMMEDIATE`` transaction, so several processes can write to one
catalog: each waits for the other.

Frames returned from the catalog and from :func:`read_table` have the same column types
whatever the rows hold, also when there are none: text is ``object`` with None for a
missing value, a flag that may be unknown is pandas ``boolean``, an integer that may be
missing a pandas nullable integer, and a float has NaN.
"""

from __future__ import annotations

import json
import os
import random
import re
import secrets
import sqlite3
import threading
import uuid
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal, cast

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from surfcensus import __version__
from surfcensus._io import atomic_path, canonical_json
from surfcensus.config import ConfigError
from surfcensus.log import get_logger
from surfcensus.privacy import HashKey
from surfcensus.records import Provenance, RenderRow, WorkRow
from surfcensus.schemas import KEYS, SCHEMAS, split_flags
from surfcensus.sources import base
from surfcensus.sources.base import (
    HEMIS,
    RENDER_ROLES,
    TIER_A_REQUIRED,
    TIER_B,
    TIER_B_REQUIRED,
    DatasetRecord,
    FileRef,
    FileRole,
    SubjectRecord,
)

SCHEMA_VERSION = 1
CATALOG_FILE = "catalog.sqlite"
MEDIA_DIR = "media"

Mode = Literal["local", "census"]

_MIGRATION_1 = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;

CREATE TABLE datasets (
  dataset_id        INTEGER PRIMARY KEY,
  pseudonym         TEXT NOT NULL,
  source            TEXT NOT NULL
                    CHECK (source IN ('local','openneuro-derivatives','openneuro-embedded')),
  identity          TEXT NOT NULL,
  accession         TEXT,
  root              TEXT NOT NULL,
  fs_root           TEXT NOT NULL,
  template_location TEXT,
  layout            TEXT NOT NULL,
  pipeline          TEXT NOT NULL,
  pipeline_version  TEXT,
  derivative_ref    TEXT,
  derivative_ref_source TEXT,
  snapshot_tag      TEXT,
  n_subjects_listed INTEGER,
  fs_subject_found  INTEGER,
  template_sphere_match INTEGER,
  discovered_at     TEXT NOT NULL,
  UNIQUE (source, identity, pipeline)
);
CREATE INDEX datasets_pseudonym ON datasets (pseudonym);

CREATE TABLE subjects (
  subject_uid    TEXT PRIMARY KEY,
  dataset_id     INTEGER NOT NULL REFERENCES datasets(dataset_id),
  subject_label  TEXT NOT NULL,
  participant    TEXT,
  fs_dir         TEXT NOT NULL,
  present        INTEGER NOT NULL DEFAULT 1,
  complete       INTEGER NOT NULL,
  tier_a         INTEGER NOT NULL,
  tier_b         INTEGER NOT NULL,
  missing_roles  TEXT NOT NULL DEFAULT '',
  problem        TEXT,
  longitudinal   INTEGER NOT NULL DEFAULT 0,
  fs_build_stamp TEXT,
  fs_version     TEXT,
  fs_major       INTEGER,
  recon_flags    TEXT,
  n_invocations  INTEGER,
  recon_finished INTEGER,
  t2pial INTEGER, flairpial INTEGER, hires INTEGER, expert INTEGER, xopts INTEGER, edited INTEGER,
  pial_variant   TEXT,
  holes_lh INTEGER, holes_rh INTEGER,
  trunc_lh_1 INTEGER, trunc_lh_2 INTEGER, trunc_rh_1 INTEGER, trunc_rh_2 INTEGER,
  age REAL, sex TEXT, field_strength REAL, manufacturer TEXT,
  inventory_hash TEXT NOT NULL,
  inventoried_at TEXT NOT NULL,
  UNIQUE (dataset_id, subject_label)
) WITHOUT ROWID;

CREATE TABLE files (
  subject_uid   TEXT NOT NULL REFERENCES subjects(subject_uid) ON DELETE CASCADE,
  role          TEXT NOT NULL,
  location      TEXT NOT NULL,
  variant       TEXT,
  size          INTEGER NOT NULL,
  etag_or_mtime TEXT NOT NULL,
  last_modified TEXT,
  version_id    TEXT,
  sha256        TEXT,
  PRIMARY KEY (subject_uid, role)
) WITHOUT ROWID;

CREATE TABLE dataset_files (
  dataset_id  INTEGER NOT NULL REFERENCES datasets(dataset_id),
  role        TEXT NOT NULL,
  bucket      TEXT,
  key         TEXT NOT NULL,
  size        INTEGER NOT NULL,
  etag        TEXT NOT NULL,
  version_id  TEXT,
  sha256      TEXT,
  fetched_at  TEXT NOT NULL,
  PRIMARY KEY (dataset_id, role, key)
) WITHOUT ROWID;

CREATE TABLE templates (
  template_id  TEXT PRIMARY KEY,
  mesh         TEXT NOT NULL,
  origin       TEXT NOT NULL,
  dataset_id   INTEGER REFERENCES datasets(dataset_id),
  build_stamp  TEXT,
  files_json   TEXT NOT NULL,
  created_at   TEXT NOT NULL
);

CREATE TABLE work (
  phase           TEXT NOT NULL,
  unit            TEXT NOT NULL,
  config_hash     TEXT NOT NULL,
  input_signature TEXT NOT NULL,
  status          TEXT NOT NULL CHECK (status IN ('done','failed')),
  reason          TEXT,
  detail          TEXT,
  output          TEXT,
  surfcensus_version TEXT NOT NULL,
  finished_at     TEXT NOT NULL,
  PRIMARY KEY (phase, unit)
) WITHOUT ROWID;

CREATE TABLE renders (
  subject_uid     TEXT PRIMARY KEY REFERENCES subjects(subject_uid),
  config_hash     TEXT NOT NULL,
  input_signature TEXT NOT NULL,
  tile_hash       TEXT,
  render_key      TEXT,
  video           INTEGER NOT NULL,
  status          TEXT NOT NULL CHECK (status IN ('done','failed')),
  reason          TEXT,
  detail          TEXT,
  media_path      TEXT,
  n_frames        INTEGER,
  template_id     TEXT,
  rendered_at     TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE selection (
  subject_uid    TEXT NOT NULL REFERENCES subjects(subject_uid),
  reason         TEXT NOT NULL,
  stratum        TEXT NOT NULL,
  inclusion_prob REAL NOT NULL,
  seed           INTEGER NOT NULL,
  PRIMARY KEY (subject_uid, reason)
) WITHOUT ROWID;

CREATE VIEW public_datasets AS
  SELECT pseudonym, source, layout, pipeline, pipeline_version, n_subjects_listed, fs_subject_found
  FROM datasets;

CREATE VIEW public_subjects AS
  SELECT s.subject_uid, d.pseudonym AS dataset,
         CASE WHEN (SELECT value FROM meta WHERE key = 'mode') = 'local' THEN s.subject_label END
           AS subject_label,
         CASE WHEN (SELECT value FROM meta WHERE key = 'mode') = 'local' THEN s.participant END
           AS participant,
         s.present, s.complete, s.tier_a, s.tier_b, s.missing_roles, s.problem, s.longitudinal,
         s.fs_version, s.fs_major, s.recon_flags, s.n_invocations, s.recon_finished,
         s.t2pial, s.flairpial, s.hires, s.expert, s.xopts, s.edited, s.pial_variant,
         s.holes_lh, s.holes_rh, s.trunc_lh_1, s.trunc_lh_2, s.trunc_rh_1, s.trunc_rh_2,
         s.age, s.sex, s.field_strength, s.manufacturer
  FROM subjects s JOIN datasets d USING (dataset_id);

CREATE VIEW public_files AS
  SELECT subject_uid, role, variant FROM files;
"""

MIGRATIONS: tuple[str, ...] = (_MIGRATION_1,)
"""Migration ``n`` (1-based) brings a catalog from ``user_version`` ``n - 1`` to ``n``. The
tables of milestones 3 and 5 (``templates``, ``dataset_files``) are in the first one, so
those milestones add none. In a migration every statement ends at the end of a line."""

PUBLIC_META: tuple[str, ...] = (
    "run_id",
    "mode",
    "key_fingerprint",
    "surfcensus_version",
    "template_id",
)
"""The ``meta`` keys that :meth:`PublicCatalog.meta` answers for."""

# written once by Catalog.create; a run whose secret or identity changed would be another run
_FIXED_META = frozenset(
    {"run_id", "mode", "key_kind", "key_fingerprint", "created_at", "run_secret"}
)

_BUSY_MS: dict[Mode, int] = {"local": 30_000, "census": 60_000}
_READONLY_CENSUS_BUSY_MS = 5_000

# file system types that Linux reports for network mounts (from memory of /proc/mounts on
# such machines, not from a list); SQLite's WAL needs shared memory, which they do not give
_NETWORK_FILESYSTEMS = frozenset(
    {
        "nfs",
        "nfs4",
        "cifs",
        "smb3",
        "smbfs",
        "9p",
        "afs",
        "ceph",
        "lustre",
        "gpfs",
        "beegfs",
        "glusterfs",
        "fuse.glusterfs",
        "fuse.sshfs",
    }
)

# what SQLite keeps beside a database file while it is written to
_SIDECARS = ("-journal", "-wal", "-shm")

_UID = re.compile(r"[0-9a-f]{16}")
_PSEUDONYM = re.compile(r"D-[0-9a-f]{1,64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")

# roles whose absence `subjects.missing_roles` names, in FileRole definition order
_NAMED_WHEN_MISSING = tuple(
    role for role in FileRole if role in {*TIER_A_REQUIRED, *TIER_B, *RENDER_ROLES}
)

# columns of the frames, with their kind: "str", "bool", "int", "float", and "bool?" or "int?"
# for a value that may be missing
_Columns = tuple[tuple[str, str], ...]


def _columns(kind: str, *names: str) -> _Columns:
    return tuple((name, kind) for name in names)


_DATASET_COLUMNS: _Columns = (
    *_columns("str", "pseudonym", "source", "layout", "pipeline", "pipeline_version"),
    ("n_subjects_listed", "int?"),
    ("fs_subject_found", "bool?"),
)
_SUBJECT_COLUMNS: _Columns = (
    *_columns("str", "subject_uid", "dataset", "subject_label", "participant"),
    *_columns("bool", "present", "complete", "tier_a", "tier_b"),
    *_columns("str", "missing_roles", "problem"),
    ("longitudinal", "bool"),
    ("fs_version", "str"),
    ("fs_major", "int?"),
    ("recon_flags", "str"),
    ("n_invocations", "int?"),
    *_columns("bool?", "recon_finished", "t2pial", "flairpial", "hires", "expert", "xopts"),
    ("edited", "bool?"),
    ("pial_variant", "str"),
    *_columns("int?", "holes_lh", "holes_rh"),
    *_columns("int?", "trunc_lh_1", "trunc_lh_2", "trunc_rh_1", "trunc_rh_2"),
    ("age", "float"),
    ("sex", "str"),
    ("field_strength", "float"),
    ("manufacturer", "str"),
)
_WORK_COLUMNS: _Columns = _columns("str", "unit", "status", "reason", "output", "finished_at")
_RENDER_COLUMNS: _Columns = (
    *_columns("str", "subject_uid", "config_hash", "tile_hash", "render_key"),
    ("video", "bool"),
    *_columns("str", "status", "reason", "media_path"),
    ("n_frames", "int?"),
    ("template_id", "str"),
)
_SELECTION_COLUMNS: _Columns = (
    *_columns("str", "subject_uid", "reason", "stratum"),
    ("inclusion_prob", "float"),
    ("seed", "int"),
)
_FAILURE_COLUMNS: _Columns = _columns("str", "unit", "phase", "reason", "detail")


def _frame(columns: _Columns, rows: Sequence[Sequence[object]]) -> pd.DataFrame:
    """Rows of a query as a frame whose column types do not depend on the rows."""
    data: dict[str, object] = {}
    for position, (name, kind) in enumerate(columns):
        values = [row[position] for row in rows]
        if kind == "str":
            data[name] = pd.Series(values, dtype=object)
        elif kind == "bool":
            data[name] = pd.Series([bool(value) for value in values], dtype=bool)
        elif kind == "bool?":
            flags = [None if value is None else bool(value) for value in values]
            data[name] = pd.Series(pd.array(flags, dtype="boolean"))
        elif kind == "int":
            data[name] = pd.Series(values, dtype="int64")
        elif kind == "int?":
            data[name] = pd.Series(pd.array(values, dtype="Int64"))
        else:
            data[name] = pd.Series(values, dtype="float64")
    return pd.DataFrame(data)


def _names(columns: _Columns) -> str:
    return ", ".join(name for name, _ in columns)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _flag(value: bool | None) -> int | None:
    return None if value is None else int(value)


def _check_digests(digests: Mapping[FileRole, str]) -> None:
    for role, digest in digests.items():
        if not _SHA256.fullmatch(digest):
            raise ValueError(f"not a SHA-256 digest for {role.value}: {digest!r}")


def _statements(script: str) -> Iterator[str]:
    """The statements of an SQL script, one at a time; each must end at the end of a line.

    ``sqlite3``'s ``executescript`` commits on its own, so a migration could not be one
    transaction with it.
    """
    statement = ""
    for line in script.splitlines(keepends=True):
        statement += line
        if sqlite3.complete_statement(statement):
            yield statement.strip()
            statement = ""
    if any(line.strip() and not line.lstrip().startswith("--") for line in statement.splitlines()):
        yield statement.strip()  # unterminated; SQLite says what is wrong with it


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply the migrations the file lacks, inside the transaction that is open."""
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    for number in range(version, SCHEMA_VERSION):
        for statement in _statements(MIGRATIONS[number]):
            conn.execute(statement)
        conn.execute(f"PRAGMA user_version = {number + 1}")


def _migrate(conn: sqlite3.Connection) -> None:
    """Apply the migrations the file lacks, all in one transaction.

    The version is read again under the write lock: another process may have brought the
    file up between this one's first look and now.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        _apply_migrations(conn)
        conn.execute("COMMIT")
    except BaseException:
        _roll_back(conn)
        raise


def _roll_back(conn: sqlite3.Connection) -> None:
    """End the open transaction, if there still is one. SQLite ends it by itself on some
    errors (a full disk), and a ROLLBACK would then raise and hide the error."""
    if conn.in_transaction:
        conn.execute("ROLLBACK")


def _in_use(exc: sqlite3.Error, name: str) -> ConfigError | None:
    """The error to raise when ``exc`` says that another connection holds the file locked
    for longer than the busy timeout, else None."""
    code = getattr(exc, "sqlite_errorcode", None)
    # the low byte of an extended code (SQLITE_BUSY_TIMEOUT, ...) is the primary one
    if isinstance(code, int) and code & 0xFF in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
        return ConfigError(
            "invalid", f"{name} is in use by another process; try again when that is done"
        )
    return None


class UnfinishedCatalog(ConfigError):
    """A catalog file in which no run was made yet: an empty file, or the tables of this
    package without a row.

    A process that makes a run on a file system without hard links leaves the first while it
    works, and for good when it is killed; a version before :meth:`Catalog.create` built
    the catalog under another name left either. Nothing is lost with such a file:
    :meth:`Catalog.create` with ``exist_ok`` makes the run in it. The reason is ``invalid``.
    """


def _unfinished(conn: sqlite3.Connection) -> bool:
    """Whether the database is an unfinished catalog (:class:`UnfinishedCatalog`).

    Something else made a file that has tables and no schema number, or a schema number
    and no ``meta`` table, or a row in any table. Raises ``sqlite3.DatabaseError`` when the
    file cannot be read as a database.
    """
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    tables = [
        name
        for (name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    ]
    if version > SCHEMA_VERSION or (version == 0) != (not tables):
        return False
    if tables and "meta" not in tables:
        return False
    for name in tables:
        quoted = name.replace('"', '""')
        if conn.execute(f'SELECT 1 FROM "{quoted}" LIMIT 1').fetchone() is not None:
            return False
    return True


def _use_wal(conn: sqlite3.Connection) -> bool:
    """Put the database in WAL mode, which stays with the file; says whether it is in it.

    The change needs the file to itself for a moment. It is refused while another connection
    reads or writes, which can happen where a run was made in the catalog file itself
    (``_fill``), so every opening of a census catalog for writing asks again.
    """
    try:
        journal = conn.execute("PRAGMA journal_mode = WAL").fetchone()
    except sqlite3.OperationalError:  # locked
        return False
    return journal is not None and str(journal[0]).lower() == "wal"


def _fill(path: Path, *, mode: Mode, key: HashKey) -> None:
    """Make a run in the unfinished catalog at ``path``: the schema, the identity of the
    run and its run secret, in one transaction.

    So the file holds a run or is unfinished, whenever the process ends. The file is looked
    at again under the write lock: when another process made the run meanwhile, or the
    file is not an unfinished catalog, nothing is written. A census catalog is put in WAL
    mode after the transaction, since the journal mode cannot change inside one.
    """
    conn = sqlite3.connect(path, timeout=_BUSY_MS["census"] / 1000, isolation_level=None)
    try:
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            busy = _in_use(exc, path.name)
            if busy is None:
                raise
            raise busy from exc
        try:
            if not _unfinished(conn):
                conn.execute("ROLLBACK")
                return
            _apply_migrations(conn)
            meta = {
                "run_id": uuid.uuid4().hex,
                "mode": mode,
                "key_kind": key.kind,
                "key_fingerprint": key.fingerprint,
                "created_at": _now(),
                "surfcensus_version": __version__,
                "run_secret": secrets.token_bytes(32).hex(),
            }
            conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta.items())
            conn.execute("COMMIT")
        except BaseException:
            _roll_back(conn)
            raise
        if mode == "census":
            # every row is in the file itself when the log begins, so a file that is given
            # another name after this is the whole catalog
            _use_wal(conn)
    finally:
        conn.close()


def _publish(tmp: Path, path: Path) -> bool:
    """Give the catalog ``tmp`` the name ``path``, unless the name is taken; says whether
    the name was free. ``tmp`` is left for the caller to remove.

    On a file system without hard links ``tmp`` cannot be given the name: a rename would
    replace a catalog that another process put there meanwhile. The name is then taken with
    an empty file, which every reader takes for an unfinished catalog, for the caller to
    make the run in.
    """
    try:
        os.link(tmp, path)  # atomic, and fails when the name is taken
    except FileExistsError:
        return False
    except OSError:
        try:
            path.open("x").close()
        except FileExistsError:
            return False
    return True


def filesystem_type(path: Path | str, mounts: str | None = None) -> str | None:
    """The type of the file system that holds ``path``, as Linux names it (``ext4``,
    ``nfs4``), or None when it cannot be told.

    Read from ``/proc/mounts``, or from ``mounts`` when given (the same format). Other
    platforms have no such file and give None: the check behind the warning of
    :meth:`Catalog.create` is made on Linux only.
    """
    if mounts is None:
        try:
            mounts = Path("/proc/mounts").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
    target = Path(path).resolve()
    best: tuple[int, str] | None = None
    for line in mounts.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        # the kernel writes a space, tab, newline or backslash in a mount point as \ooo
        point = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), fields[1])
        if target == Path(point) or Path(point) in target.parents:
            depth = len(Path(point).parts)
            # the deepest mount point holds the path; of two on one point, the later one
            if best is None or depth >= best[0]:
                best = (depth, fields[2])
    return best[1] if best is not None else None


class _Db:
    """One connection to a catalog file, shared by the private and the public side.

    A lock makes it usable from several threads (the rater app answers requests from a
    thread pool); one write transaction is open at a time.
    """

    def __init__(self, path: Path, *, readonly: bool) -> None:
        if not path.is_file():
            raise ConfigError("missing", f"{path.parent} has no {path.name}")
        self.path = path
        self.readonly = readonly
        self.mode: Mode = "local"  # until the file says
        self._lock = threading.RLock()
        target = f"{path.resolve().as_uri()}?mode=ro" if readonly else str(path)
        self._conn = sqlite3.connect(
            target,
            uri=readonly,
            timeout=_BUSY_MS["census"] / 1000,
            isolation_level=None,  # no implicit transactions: every write says BEGIN IMMEDIATE
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        try:
            self._prepare()
        except BaseException:
            self._conn.close()
            raise

    def _prepare(self) -> None:
        name = self.path.name
        try:
            self._conn.execute("PRAGMA foreign_keys = ON")
            version, mode, unfinished = self._look()
        except sqlite3.DatabaseError as exc:
            # a locked file is whole: saying otherwise invites deleting the catalog
            raise (
                _in_use(exc, name)
                or ConfigError("invalid", f"{name} is not a SQLite database: {exc}")
            ) from exc
        if version > SCHEMA_VERSION:
            raise ConfigError(
                "newer schema",
                f"{name} has schema {version}; this version of surfcensus reads up to "
                f"{SCHEMA_VERSION}",
            )
        if unfinished:
            # not "it was not made by surfcensus": that reads as a foreign file to delete
            raise UnfinishedCatalog(
                "invalid",
                f"{name} holds no run yet: the run is being made by another process, or "
                "its making was interrupted. `surfcensus status` on the run directory "
                "finishes it",
            )
        if mode not in _BUSY_MS:
            # also a file without a schema number: a catalog gets its number in the
            # transaction that makes the run, so no migration is applied to such a file
            raise ConfigError("invalid", f"{name} holds no run: it was not made by surfcensus")
        if version < SCHEMA_VERSION:
            if self.readonly:
                raise ConfigError(
                    "invalid",
                    f"catalog schema {version} is older than this version; run any phase "
                    "command once",
                )
            _migrate(self._conn)
        self.mode = mode
        busy = _BUSY_MS[self.mode]
        if self.readonly and self.mode == "census":
            busy = _READONLY_CENSUS_BUSY_MS  # in WAL mode a reader does not wait for a writer
        elif self.mode == "census":
            _use_wal(self._conn)  # nothing to do, but for a catalog that could not be switched
        self._conn.execute(f"PRAGMA busy_timeout = {busy}")

    def _look(self) -> tuple[int, Mode | None, bool]:
        """The schema number, the mode of the run, and whether the file is an unfinished
        catalog, all read from one state of the file: another process may be making the
        run in it right now."""
        self._conn.execute("BEGIN")
        try:
            version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            mode = None
            if 0 < version <= SCHEMA_VERSION:
                try:
                    row = self._conn.execute("SELECT value FROM meta WHERE key = 'mode'").fetchone()
                except sqlite3.OperationalError as exc:
                    if _in_use(exc, self.path.name) is not None:
                        raise
                    row = None  # no such table
                mode = row[0] if row is not None else None
            unfinished = mode is None and _unfinished(self._conn)
        finally:
            _roll_back(self._conn)
        return version, mode, unfinished

    def rows(self, sql: str, params: Sequence[object] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def one(self, sql: str, params: Sequence[object] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    @contextmanager
    def read(self) -> Iterator[None]:
        """Statements run inside see one state of the file."""
        with self._lock:
            if self._conn.in_transaction:
                yield
                return
            self._conn.execute("BEGIN")
            try:
                yield
            finally:
                self._conn.execute("COMMIT")

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        if self.readonly:
            raise ConfigError("invalid", f"{self.path.name} was opened read-only")
        with self._lock:
            if self._conn.in_transaction:
                raise RuntimeError("a transaction is already open on this catalog")
            try:
                self._conn.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as exc:
                busy = _in_use(exc, self.path.name)
                if busy is None:
                    raise
                raise busy from exc
            try:
                yield self._conn
                self._conn.execute("COMMIT")
            except BaseException:
                _roll_back(self._conn)
                raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class Catalog:
    """The private catalog.

    Imported only by ``remote/``, ``inventory/``, ``enrich/``, ``privacy.py``, ``run.py``,
    ``cli.py`` and (milestone 5) ``calibration_transfer.py``.
    """

    def __init__(self, path: Path | str, *, readonly: bool = False) -> None:
        """Open an existing catalog and apply pending migrations.

        Raises :class:`ConfigError`: ``missing`` when there is no file, ``newer schema``
        for one written by a later version, ``invalid`` for a file that is no catalog, and
        for an older one when ``readonly`` (nothing can then be migrated). A catalog in
        which no run was made yet raises :class:`UnfinishedCatalog` and is left as it is;
        :meth:`create` with ``exist_ok`` finishes it.
        """
        self._db = _Db(Path(path), readonly=readonly)

    @classmethod
    def create(
        cls, path: Path | str, *, mode: Mode, key: HashKey, exist_ok: bool = False
    ) -> Catalog:
        """A new catalog; also draws the run secret.

        ``key`` is the per-run salt of a local run or the dataset key of a census run; its
        fingerprint is stored, the key is not. A census catalog is put in WAL mode, with a
        warning when the directory is on a network file system (seen on Linux only), where
        WAL does not work.

        The catalog is built under a temporary name in the same directory and given its name
        when it is whole. So ``path`` holds a run or does not exist, also when this process
        is killed, and no other process opens a half-made catalog. On a file system without
        hard links the name is taken with an empty file and the run is made in it in one
        transaction; a process killed then leaves an unfinished catalog
        (:class:`UnfinishedCatalog`).

        An existing file is refused. With ``exist_ok`` it is opened in place of a new one
        when it is a catalog of the same mode and key, which another process made
        meanwhile, and an unfinished catalog is finished in place; a catalog of another
        run is refused.
        """
        path = Path(path)
        if (mode == "local") != (key.kind == "local-salt"):
            raise ConfigError("key", f"a {mode} run is not keyed with a {key.kind}")
        path.parent.mkdir(parents=True, exist_ok=True)
        catalog = cls._create(path, mode, key, exist_ok)
        if mode == "census":
            catalog._warn_about_wal()
        return catalog

    @classmethod
    def _create(cls, path: Path, mode: Mode, key: HashKey, exist_ok: bool) -> Catalog:
        free = False
        if not os.path.lexists(path):
            tmp = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
            try:
                tmp.touch(exist_ok=False)
                _fill(tmp, mode=mode, key=key)
                free = _publish(tmp, path)
            finally:
                for suffix in ("", *_SIDECARS):
                    tmp.with_name(tmp.name + suffix).unlink(missing_ok=True)
        if not free and not exist_ok:
            raise ConfigError("invalid", f"{path.parent} already has a {path.name}")
        try:
            catalog = cls(path)
        except UnfinishedCatalog:
            # The name was taken with an empty file, here or by a process that was killed.
            # A failure leaves the file: another process may have opened it by now.
            _fill(path, mode=mode, key=key)
            catalog = cls(path)
        # another process may have made the run, in a file that was unfinished or free
        if (catalog.meta("mode"), catalog.meta("key_fingerprint")) != (mode, key.fingerprint):
            catalog.close()
            raise ConfigError("invalid", f"{path.parent} already has a {path.name}, of another run")
        return catalog

    def _warn_about_wal(self) -> None:
        kind = filesystem_type(self._db.path.parent)
        if kind in _NETWORK_FILESYSTEMS:
            get_logger().warning(
                "the run directory is on a network file system (%s); the catalog of a census "
                "run uses SQLite's write-ahead log, which needs a local disk",
                kind,
            )
        journal = self._db.one("PRAGMA journal_mode")
        if journal is None or str(journal[0]).lower() != "wal":
            get_logger().warning(
                "the catalog could not be put in write-ahead mode (it is in %s mode); "
                "readers will wait while a phase command commits",
                journal[0] if journal is not None else "an unknown",
            )

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> Catalog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        """One ``BEGIN IMMEDIATE`` transaction: committed on success, rolled back otherwise.

        For the writes that have no method here (a column that one phase owns). Rows come
        back as ``sqlite3.Row``. The methods of this class open their own transaction and
        cannot be called inside this one.
        """
        with self._db.write() as conn:
            yield conn

    def query(self, sql: str, params: Sequence[object] = ()) -> list[sqlite3.Row]:
        """The rows of a read-only statement. Private like everything this class returns."""
        return self._db.rows(sql, params)

    def meta(self, key: str) -> str | None:
        row = self._db.one("SELECT value FROM meta WHERE key = ?", (key,))
        return None if row is None else row["value"]

    def set_meta(self, key: str, value: str) -> None:
        """Set a ``meta`` value. What :meth:`create` wrote about the run's identity (its
        id, mode, key and secret) cannot be changed: ValueError."""
        with self._db.write() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
            if key in _FIXED_META and row is not None and row["value"] != value:
                raise ValueError(f"meta.{key} is fixed when the run is created")
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def run_secret(self) -> bytes:
        """The 32 bytes that render keys and cache names are keyed with (``privacy.py``)."""
        value = self.meta("run_secret")
        if value is None or not _SHA256.fullmatch(value):
            raise ConfigError("invalid", f"{self._db.path.name} has no run secret")
        return bytes.fromhex(value)

    def upsert_dataset(
        self, record: DatasetRecord, pseudonym: str, n_subjects_listed: int | None
    ) -> int:
        """Insert the dataset, or update the one with the same source, identity and
        pipeline; returns the ``dataset_id``.

        The record is the truth for everything it holds, except that a ``derivative_ref``
        or ``snapshot_tag`` of None keeps the stored value (another phase finds those), as
        does an ``n_subjects_listed`` of None. A pseudonym never changes: a stored dataset
        given another one raises ``ConfigError("key", ...)``, and a pseudonym that another
        ``(source, identity)`` holds raises ``ConfigError("invalid", ...)``.
        """
        if not _PSEUDONYM.fullmatch(pseudonym):
            raise ValueError(f"not a pseudonym: {pseudonym!r}")
        with self._db.write() as conn:
            taken = conn.execute(
                "SELECT 1 FROM datasets WHERE pseudonym = ? AND NOT (source = ? AND identity = ?)",
                (pseudonym, record.source, record.identity),
            ).fetchone()
            if taken is not None:
                raise ConfigError("invalid", f"pseudonym {pseudonym} belongs to another dataset")
            row = conn.execute(
                "SELECT dataset_id, pseudonym FROM datasets "
                "WHERE source = ? AND identity = ? AND pipeline = ?",
                (record.source, record.identity, record.pipeline),
            ).fetchone()
            values = (
                record.accession,
                record.root,
                record.fs_root,
                record.template_location,
                record.layout,
                record.pipeline_version,
                record.derivative_ref,
                record.snapshot_tag,
                n_subjects_listed,
            )
            if row is None:
                cursor = conn.execute(
                    "INSERT INTO datasets (accession, root, fs_root, template_location, layout, "
                    "pipeline_version, derivative_ref, snapshot_tag, n_subjects_listed, "
                    "pseudonym, source, identity, pipeline, discovered_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (*values, pseudonym, record.source, record.identity, record.pipeline, _now()),
                )
                return cast(int, cursor.lastrowid)
            if row["pseudonym"] != pseudonym:
                raise ConfigError(
                    "key",
                    f"this dataset is {row['pseudonym']} in the catalog and {pseudonym} with "
                    "the key given: the run was made with another key",
                )
            conn.execute(
                "UPDATE datasets SET accession = ?, root = ?, fs_root = ?, template_location = ?, "
                "layout = ?, pipeline_version = ?, "
                "derivative_ref = COALESCE(?, derivative_ref), "
                "snapshot_tag = COALESCE(?, snapshot_tag), "
                "n_subjects_listed = COALESCE(?, n_subjects_listed) WHERE dataset_id = ?",
                (*values, row["dataset_id"]),
            )
            return row["dataset_id"]

    def upsert_subject(
        self,
        uid: str,
        dataset_id: int,
        record: SubjectRecord,
        provenance: Provenance,
        inventory_hash: str,
        *,
        digests: Mapping[FileRole, str] | None = None,
    ) -> None:
        """Store what the inventory found for one subject, and mark it present.

        Replaces the subject's ``files`` rows in the same transaction. ``digests`` are the
        SHA-256 digests of the files that this inventory read in full; a role the subject
        does not have is ignored. A file that is not among them keeps the digest it has
        while its location, variant, size and ETag or mtime are unchanged. The columns of
        phase 3 (age, sex, field strength, manufacturer) are not touched.

        Raises ``ConfigError("invalid", ...)`` when ``uid`` already names another subject
        (a collision of the keyed hash), and ``ConfigError("key", ...)`` when the subject
        is stored under another ``uid``, which happens when the key is not the run's.
        """
        if not _UID.fullmatch(uid):
            raise ValueError(f"a subject_uid is 16 hex characters, got {uid!r}")
        digests = {} if digests is None else digests
        _check_digests(digests)
        files = record.files
        truncations = [provenance.truncations.get(hemi) or (None, None) for hemi in HEMIS]
        holes = provenance.holes or (None, None)
        values: dict[str, object] = {
            "subject_uid": uid,
            "dataset_id": dataset_id,
            "subject_label": record.subject_label,
            "participant": provenance.participant,
            "fs_dir": record.fs_dir,
            "present": 1,
            "complete": int(all(role in files for role in RENDER_ROLES)),
            "tier_a": int(all(role in files for role in TIER_A_REQUIRED)),
            "tier_b": int(all(role in files for role in TIER_B_REQUIRED)),
            "missing_roles": ",".join(
                role.value for role in _NAMED_WHEN_MISSING if role not in files
            ),
            "problem": record.problem,
            "longitudinal": int(provenance.longitudinal),
            "fs_build_stamp": provenance.build_stamp,
            "fs_version": provenance.fs_version,
            "fs_major": provenance.fs_major,
            # no log read: the flags are unknown, which is not the same as none
            "recon_flags": (
                None
                if provenance.n_invocations is None
                else json.dumps(sorted(provenance.flags), separators=(",", ":"))
            ),
            "n_invocations": provenance.n_invocations,
            "recon_finished": _flag(provenance.finished),
            "t2pial": _flag(provenance.t2pial),
            "flairpial": _flag(provenance.flairpial),
            "hires": _flag(provenance.hires),
            "expert": _flag(provenance.expert),
            "xopts": _flag(provenance.xopts),
            "edited": _flag(provenance.edited),
            "pial_variant": provenance.pial_variant,
            "holes_lh": holes[0],
            "holes_rh": holes[1],
            "trunc_lh_1": truncations[0][0],
            "trunc_lh_2": truncations[0][1],
            "trunc_rh_1": truncations[1][0],
            "trunc_rh_2": truncations[1][1],
            "inventory_hash": inventory_hash,
            "inventoried_at": _now(),
        }
        names = list(values)
        updates = ", ".join(f"{name} = excluded.{name}" for name in names if name != "subject_uid")
        with self._db.write() as conn:
            known = conn.execute("SELECT 1 FROM datasets WHERE dataset_id = ?", (dataset_id,))
            if known.fetchone() is None:
                raise KeyError(f"no dataset {dataset_id} in the catalog")
            holder = conn.execute(
                "SELECT dataset_id, subject_label FROM subjects WHERE subject_uid = ?", (uid,)
            ).fetchone()
            if holder is not None and tuple(holder) != (dataset_id, record.subject_label):
                raise ConfigError(
                    "invalid",
                    f"subject_uid {uid} of {record.subject_label} already names "
                    f"{holder['subject_label']}: two subjects with one identifier",
                )
            stored = conn.execute(
                "SELECT subject_uid FROM subjects WHERE dataset_id = ? AND subject_label = ?",
                (dataset_id, record.subject_label),
            ).fetchone()
            if stored is not None and stored["subject_uid"] != uid:
                raise ConfigError(
                    "key",
                    f"{record.subject_label} is {stored['subject_uid']} in the catalog and "
                    f"{uid} with the key given: the run was made with another key",
                )
            conn.execute(
                f"INSERT INTO subjects ({', '.join(names)}) "
                f"VALUES ({', '.join('?' for _ in names)}) "
                f"ON CONFLICT(subject_uid) DO UPDATE SET {updates}",
                list(values.values()),
            )
            kept = {
                row["role"]: row
                for row in conn.execute(
                    "SELECT role, location, variant, size, etag_or_mtime, version_id, sha256 "
                    "FROM files WHERE subject_uid = ?",
                    (uid,),
                )
            }
            conn.execute("DELETE FROM files WHERE subject_uid = ?", (uid,))
            for role, ref in files.items():
                old = kept.get(role.value)
                # the variant too: a link that now leads to another file of the same size
                # and time has the location, size and time it had
                if old is not None and (
                    ref.location,
                    ref.variant,
                    ref.size,
                    ref.etag_or_mtime,
                ) != (old["location"], old["variant"], old["size"], old["etag_or_mtime"]):
                    old = None  # another file now: what was read of the old one does not hold
                conn.execute(
                    "INSERT INTO files (subject_uid, role, location, variant, size, etag_or_mtime, "
                    "last_modified, version_id, sha256) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        uid,
                        role.value,
                        ref.location,
                        ref.variant,
                        ref.size,
                        ref.etag_or_mtime,
                        ref.last_modified,
                        old["version_id"] if old is not None else None,
                        digests.get(role, old["sha256"] if old is not None else None),
                    ),
                )

    def record_dataset_file(
        self,
        dataset_id: int,
        role: str,
        bucket: str | None,
        key: str,
        *,
        size: int,
        etag: str,
        version_id: str | None,
        sha256: str | None,
    ) -> None:
        """Record a dataset-level file that a stored value was read from; a second record
        of the same file replaces the first."""
        with self._db.write() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO dataset_files (dataset_id, role, bucket, key, size, etag, "
                "version_id, sha256, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (dataset_id, role, bucket, key, size, etag, version_id, sha256, _now()),
            )

    def mark_absent(self, dataset_id: int, present_labels: Collection[str]) -> list[str]:
        """Bring ``present`` of the dataset's subjects in line with a listing.

        Subjects whose label is not in ``present_labels`` get ``present = 0``; their rows,
        renders and ratings stay. A subject that is listed again gets ``present = 1``.
        Returns the ``subject_uid``s that this call marked absent, sorted.
        """
        listed = set(present_labels)
        with self._db.write() as conn:
            rows = conn.execute(
                "SELECT subject_uid, subject_label, present FROM subjects WHERE dataset_id = ? "
                "ORDER BY subject_uid",
                (dataset_id,),
            ).fetchall()
            gone = [
                r["subject_uid"] for r in rows if r["present"] and r["subject_label"] not in listed
            ]
            back = [
                r["subject_uid"] for r in rows if not r["present"] and r["subject_label"] in listed
            ]
            conn.executemany(
                "UPDATE subjects SET present = 0 WHERE subject_uid = ?", [(uid,) for uid in gone]
            )
            conn.executemany(
                "UPDATE subjects SET present = 1 WHERE subject_uid = ?", [(uid,) for uid in back]
            )
        return gone

    def relocate(self, dataset_id: int, record: DatasetRecord) -> None:
        """The dataset now lives where ``record`` says: new ``root``, ``fs_root``,
        ``template_location``, ``layout`` and ``pipeline_version``.

        The stored ``identity`` stays, whatever ``record`` holds, so every ``subject_uid``
        stays. Source and pipeline are hashed into the uids as well, so a record of another
        source or pipeline is refused: ``ConfigError("invalid", ...)``.

        When ``fs_root`` changes (the data moved into another layout), the ``fs_dir`` of
        every subject of the dataset moves with it, listed or not: the inventory skips a
        subject whose files are unchanged, so nothing else would correct it.
        """
        with self._db.write() as conn:
            row = conn.execute(
                "SELECT source, pipeline, fs_root FROM datasets WHERE dataset_id = ?",
                (dataset_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"no dataset {dataset_id} in the catalog")
            if (row["source"], row["pipeline"]) != (record.source, record.pipeline):
                raise ConfigError(
                    "invalid",
                    f"the run holds a {row['source']} dataset of pipeline {row['pipeline']}; "
                    f"what was found at the new place is {record.source}, {record.pipeline}, "
                    "and its subjects would get other identifiers",
                )
            conn.execute(
                "UPDATE datasets SET root = ?, fs_root = ?, template_location = ?, layout = ?, "
                "pipeline_version = ? WHERE dataset_id = ?",
                (
                    record.root,
                    record.fs_root,
                    record.template_location,
                    record.layout,
                    record.pipeline_version,
                    dataset_id,
                ),
            )
            old, new = PurePosixPath(row["fs_root"]), PurePosixPath(record.fs_root)
            if old == new:
                return
            subjects = conn.execute(
                "SELECT subject_uid, subject_label, fs_dir FROM subjects WHERE dataset_id = ?",
                (dataset_id,),
            ).fetchall()
            conn.executemany(
                "UPDATE subjects SET fs_dir = ? WHERE subject_uid = ?",
                [
                    ((new / subject["subject_label"]).as_posix(), subject["subject_uid"])
                    for subject in subjects
                    if PurePosixPath(subject["fs_dir"]) == old / subject["subject_label"]
                ],
            )

    def _datasets_named(self, only: Collection[str] | None) -> list[sqlite3.Row]:
        rows = self._db.rows(
            "SELECT dataset_id, pseudonym, accession FROM datasets ORDER BY dataset_id"
        )
        if only is None:
            return rows
        unknown = sorted(
            name
            for name in set(only)
            if all(name not in (r["pseudonym"], r["accession"]) for r in rows)
        )
        if unknown:
            raise ConfigError("invalid", f"no dataset named {', '.join(unknown)} in this run")
        wanted = set(only)
        return [r for r in rows if r["pseudonym"] in wanted or r["accession"] in wanted]

    def dataset_ids(self, only: Collection[str] | None = None) -> list[int]:
        """Every ``dataset_id``, or those of the accessions or pseudonyms in ``only``, in
        ascending order. An unknown value raises ``ConfigError("invalid", ...)``."""
        return [row["dataset_id"] for row in self._datasets_named(only)]

    def pseudonyms(self, only: Collection[str] | None = None) -> list[str]:
        """Accessions or pseudonyms -> sorted pseudonyms; an unknown value raises
        ``ConfigError("invalid", ...)``."""
        return sorted({row["pseudonym"] for row in self._datasets_named(only)})

    def pseudonym(self, dataset_id: int) -> str:
        row = self._db.one("SELECT pseudonym FROM datasets WHERE dataset_id = ?", (dataset_id,))
        if row is None:
            raise KeyError(f"no dataset {dataset_id} in the catalog")
        return row["pseudonym"]

    def scrub_terms(self) -> tuple[dict[str, str], tuple[str, ...]]:
        """What ``privacy.scrub`` needs for this run: accession -> pseudonym, and every
        dataset root and identity, sorted. A local run has no accessions; its caller passes
        ``pseudonyms=None`` to ``scrub``, which then keeps subject labels."""
        rows = self._db.rows("SELECT pseudonym, accession, root, identity FROM datasets")
        pseudonyms = {r["accession"]: r["pseudonym"] for r in rows if r["accession"] is not None}
        roots = sorted({text for r in rows for text in (r["root"], r["identity"])})
        return pseudonyms, tuple(roots)

    @staticmethod
    def _dataset_record(row: sqlite3.Row) -> DatasetRecord:
        return DatasetRecord(
            source=row["source"],
            identity=row["identity"],
            accession=row["accession"],
            root=row["root"],
            fs_root=row["fs_root"],
            layout=row["layout"],
            pipeline=row["pipeline"],
            pipeline_version=row["pipeline_version"],
            template_location=row["template_location"],
            derivative_ref=row["derivative_ref"],
            snapshot_tag=row["snapshot_tag"],
        )

    def dataset_record(self, dataset_id: int) -> DatasetRecord:
        row = self._db.one("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,))
        if row is None:
            raise KeyError(f"no dataset {dataset_id} in the catalog")
        return self._dataset_record(row)

    def subject_record(self, uid: str) -> SubjectRecord:
        """Equal to the record the source yielded at inventory."""
        with self._db.read():  # the subject and its files are of one inventory
            subject = self._db.one(
                "SELECT dataset_id, subject_label, fs_dir, problem FROM subjects "
                "WHERE subject_uid = ?",
                (uid,),
            )
            if subject is None:
                raise KeyError(f"no subject {uid} in the catalog")
            dataset = self._db.one(
                "SELECT * FROM datasets WHERE dataset_id = ?", (subject["dataset_id"],)
            )
            files = _file_refs(self._db, uid)
        assert dataset is not None  # the foreign key
        return SubjectRecord(
            dataset=self._dataset_record(dataset),
            subject_label=subject["subject_label"],
            fs_dir=subject["fs_dir"],
            files=files,
            problem=subject["problem"],
        )

    def inventory_hash(self, uid: str) -> str | None:
        """The hash stored at the subject's last inventory; None for an unknown subject."""
        row = self._db.one("SELECT inventory_hash FROM subjects WHERE subject_uid = ?", (uid,))
        return None if row is None else row["inventory_hash"]

    def upsert_template(
        self,
        template_id: str,
        mesh: str,
        origin: str,
        dataset_id: int | None,
        build_stamp: str | None,
        files: Mapping[str, Mapping[str, object]],
    ) -> None:
        """``files``: relative file name -> ``{"size": ..., "sha256": ...}``. A template
        that is already recorded keeps its ``created_at``."""
        with self._db.write() as conn:
            conn.execute(
                "INSERT INTO templates (template_id, mesh, origin, dataset_id, build_stamp, "
                "files_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(template_id) DO UPDATE SET mesh = excluded.mesh, "
                "origin = excluded.origin, dataset_id = excluded.dataset_id, "
                "build_stamp = excluded.build_stamp, files_json = excluded.files_json",
                (
                    template_id,
                    mesh,
                    origin,
                    dataset_id,
                    build_stamp,
                    canonical_json({name: dict(entry) for name, entry in files.items()}),
                    _now(),
                ),
            )

    def failures(self, phase: str | None = None) -> pd.DataFrame:
        """Columns ``unit``, ``phase``, ``reason``, ``detail``; for ``status --failures``.

        The failed rows of the ``work`` ledger, the failed renders (phase ``render``) and
        the present subjects that the inventory could not list (phase ``inventory``, the
        text before the colon of ``problem`` as the reason). Ordered by phase, then unit.
        """
        rows = self._db.rows(
            "SELECT unit, phase, reason, detail FROM work WHERE status = 'failed' "
            "UNION ALL "
            "SELECT subject_uid, 'render', reason, detail FROM renders WHERE status = 'failed' "
            "UNION ALL "
            "SELECT subject_uid, 'inventory', "
            "  CASE WHEN instr(problem, ': ') > 0 "
            "       THEN substr(problem, 1, instr(problem, ': ') - 1) ELSE problem END, "
            "  CASE WHEN instr(problem, ': ') > 0 "
            "       THEN substr(problem, instr(problem, ': ') + 2) END "
            "FROM subjects WHERE problem IS NOT NULL AND present = 1 "
            "ORDER BY 2, 1"
        )
        return _frame(_FAILURE_COLUMNS, [row for row in rows if phase in (None, row["phase"])])

    def public(self) -> PublicCatalog:
        """The public side on this catalog's connection; closing it leaves this one open."""
        return PublicCatalog(self._db)


def _file_refs(db: _Db, uid: str) -> dict[FileRole, FileRef]:
    rows = db.rows(
        "SELECT role, location, size, etag_or_mtime, variant, last_modified FROM files "
        "WHERE subject_uid = ?",
        (uid,),
    )
    refs = {
        FileRole(row["role"]): FileRef(
            role=FileRole(row["role"]),
            location=row["location"],
            size=row["size"],
            etag_or_mtime=row["etag_or_mtime"],
            variant=row["variant"],
            last_modified=row["last_modified"],
        )
        for row in rows
    }
    return {role: refs[role] for role in FileRole if role in refs}


class PublicCatalog:
    """What ``metrics/``, ``render/``, ``app/``, ``export/``, ``select/``, ``reference/`` and
    ``calibration.py`` may hold: the same file, without anything private.

    Made by :meth:`open` or by :meth:`Catalog.public`, not by calling the class.
    """

    def __init__(self, db: _Db, *, owns_connection: bool = False) -> None:
        self._db = db
        self._owns_connection = owns_connection

    @classmethod
    def open(cls, run_dir: Path | str, *, readonly: bool = True) -> PublicCatalog:
        """Needs no key. Read-only by default: no migration is applied, and a catalog older
        than this version raises ``ConfigError("invalid", ...)``, a newer one
        ``ConfigError("newer schema", ...)``. The writing methods need ``readonly=False``."""
        return cls(_Db(Path(run_dir) / CATALOG_FILE, readonly=readonly), owns_connection=True)

    def close(self) -> None:
        if self._owns_connection:
            self._db.close()

    def __enter__(self) -> PublicCatalog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def mode(self) -> Mode:
        return self._db.mode

    def run_id(self) -> str:
        value = self.meta("run_id")
        if value is None:
            raise ConfigError("invalid", f"{self._db.path.name} has no run id")
        return value

    def meta(self, key: str) -> str | None:
        """Only the keys of :data:`PUBLIC_META` (``run_id``, ``mode``, ``key_fingerprint``,
        ``surfcensus_version``, ``template_id``); any other key raises KeyError."""
        if key not in PUBLIC_META:
            raise KeyError(
                f"meta.{key} is not public; the public keys are {', '.join(PUBLIC_META)}"
            )
        row = self._db.one("SELECT value FROM meta WHERE key = ?", (key,))
        return None if row is None else row["value"]

    def datasets(self) -> pd.DataFrame:
        """The view ``public_datasets``, ordered by pseudonym."""
        rows = self._db.rows(
            f"SELECT {_names(_DATASET_COLUMNS)} FROM public_datasets "
            "ORDER BY pseudonym, source, pipeline"
        )
        return _frame(_DATASET_COLUMNS, rows)

    def subjects(self) -> pd.DataFrame:
        """The view ``public_subjects``, ordered by ``subject_uid``. ``dataset`` is the
        pseudonym; ``subject_label`` and ``participant`` are empty in a census run;
        ``recon_flags`` is the stored JSON array of flag names."""
        rows = self._db.rows(
            f"SELECT {_names(_SUBJECT_COLUMNS)} FROM public_subjects ORDER BY subject_uid"
        )
        return _frame(_SUBJECT_COLUMNS, rows)

    def subject_uids(
        self,
        *,
        datasets: Collection[str] | None = None,
        present: bool | None = True,
        complete: bool | None = None,
        tier_a: bool | None = None,
        tier_b: bool | None = None,
    ) -> list[str]:
        """``datasets``: pseudonyms. Each of the four flags restricts to subjects that have
        it (True) or lack it (False); None does not restrict. Ordered by pseudonym, then
        ``subject_uid``."""
        wanted = None if datasets is None else set(datasets)
        flags = {"present": present, "complete": complete, "tier_a": tier_a, "tier_b": tier_b}
        rows = self._db.rows(
            "SELECT subject_uid, dataset, present, complete, tier_a, tier_b FROM public_subjects "
            "ORDER BY dataset, subject_uid"
        )
        return [
            row["subject_uid"]
            for row in rows
            if (wanted is None or row["dataset"] in wanted)
            and all(value is None or bool(row[name]) == value for name, value in flags.items())
        ]

    def dataset_of(self, uid: str) -> str:
        """The pseudonym of the subject's dataset; KeyError for an unknown subject."""
        row = self._db.one("SELECT dataset FROM public_subjects WHERE subject_uid = ?", (uid,))
        if row is None:
            raise KeyError(f"no subject {uid} in the catalog")
        return row["dataset"]

    def _known(self, uid: str) -> None:
        if self._db.one("SELECT 1 FROM subjects WHERE subject_uid = ?", (uid,)) is None:
            raise KeyError(f"no subject {uid} in the catalog")

    def roles(self, uid: str) -> dict[FileRole, str | None]:
        """Role -> variant, for the roles the subject has, in ``FileRole`` order."""
        self._known(uid)
        rows = self._db.rows("SELECT role, variant FROM public_files WHERE subject_uid = ?", (uid,))
        variants = {FileRole(row["role"]): row["variant"] for row in rows}
        return {role: variants[role] for role in FileRole if role in variants}

    def input_signature(
        self, uid: str, roles: Collection[FileRole], extra: Sequence[str] = ()
    ) -> str:
        """``sources.base.input_signature`` over the subject's stored files. The caller may
        pass it back to ``work_done``, ``record_work`` and ``record_renders``, and to
        nothing else."""
        self._known(uid)
        return base.input_signature(_file_refs(self._db, uid), roles, extra)

    def set_file_digests(self, uid: str, digests: Mapping[FileRole, str]) -> None:
        """Store SHA-256 digests of files that were read in full. Write-only. A role the
        subject does not have is ignored."""
        _check_digests(digests)
        with self._db.write() as conn:
            conn.executemany(
                "UPDATE files SET sha256 = ? WHERE subject_uid = ? AND role = ?",
                [(digest, uid, role.value) for role, digest in digests.items()],
            )

    def work(self, phase: str) -> pd.DataFrame:
        """Columns ``unit``, ``status``, ``reason``, ``output``, ``finished_at``; ordered by
        unit."""
        rows = self._db.rows(
            f"SELECT {_names(_WORK_COLUMNS)} FROM work WHERE phase = ? ORDER BY unit", (phase,)
        )
        return _frame(_WORK_COLUMNS, rows)

    def work_done(
        self, phase: str, unit: str, config_hash: str, input_signature: str
    ) -> bool | None:
        """True: done and matching. False: failed and matching. None: no row, or a row
        that does not match."""
        row = self._db.one(
            "SELECT status FROM work "
            "WHERE phase = ? AND unit = ? AND config_hash = ? AND input_signature = ?",
            (phase, unit, config_hash, input_signature),
        )
        return None if row is None else row["status"] == "done"

    def record_work(self, rows: Sequence[WorkRow]) -> None:
        """Record the outcome of units, replacing their earlier rows, in one transaction.
        A row without version or time gets the running version and the current time."""
        now = _now()
        with self._db.write() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO work (phase, unit, config_hash, input_signature, status, "
                "reason, detail, output, surfcensus_version, finished_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        row.phase,
                        row.unit,
                        row.config_hash,
                        row.input_signature,
                        row.status,
                        row.reason,
                        row.detail,
                        row.output,
                        row.surfcensus_version or __version__,
                        row.finished_at or now,
                    )
                    for row in rows
                ],
            )

    def forget_work(self, phase: str, units: Collection[str]) -> None:
        """Remove the ledger rows of ``units`` in ``phase``, in one transaction: for a
        subject that is no longer a unit of the phase, after its outputs were removed. A
        unit without a row is passed over."""
        with self._db.write() as conn:
            conn.executemany(
                "DELETE FROM work WHERE phase = ? AND unit = ?",
                [(phase, unit) for unit in sorted(units)],
            )

    def _local(self) -> bool:
        return self._db.mode == "local"

    def renders(self) -> pd.DataFrame:
        """Columns ``subject_uid``, ``config_hash``, ``tile_hash``, ``render_key``,
        ``video``, ``status``, ``reason``, ``media_path``, ``n_frames``, ``template_id``;
        and ``detail`` in a local run only. Ordered by ``subject_uid``."""
        columns = (*_RENDER_COLUMNS, ("detail", "str")) if self._local() else _RENDER_COLUMNS
        rows = self._db.rows(f"SELECT {_names(columns)} FROM renders ORDER BY subject_uid")
        return _frame(columns, rows)

    def render(self, uid: str) -> RenderRow | None:
        """The subject's row of ``renders``, with its input signature for the skip test and,
        in a census run, without its ``detail``."""
        row = self._db.one("SELECT * FROM renders WHERE subject_uid = ?", (uid,))
        if row is None:
            return None
        return RenderRow(
            subject_uid=row["subject_uid"],
            config_hash=row["config_hash"],
            input_signature=row["input_signature"],
            video=bool(row["video"]),
            status=row["status"],
            tile_hash=row["tile_hash"],
            render_key=row["render_key"],
            reason=row["reason"],
            detail=row["detail"] if self._local() else None,
            media_path=row["media_path"],
            n_frames=row["n_frames"],
            template_id=row["template_id"],
            rendered_at=row["rendered_at"],
        )

    def record_renders(self, rows: Sequence[RenderRow]) -> None:
        """Record renders, replacing the subjects' earlier rows, in one transaction. A row
        without a time gets the current one."""
        now = _now()
        with self._db.write() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO renders (subject_uid, config_hash, input_signature, "
                "tile_hash, render_key, video, status, reason, detail, media_path, n_frames, "
                "template_id, rendered_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        row.subject_uid,
                        row.config_hash,
                        row.input_signature,
                        row.tile_hash,
                        row.render_key,
                        int(row.video),
                        row.status,
                        row.reason,
                        row.detail,
                        row.media_path,
                        row.n_frames,
                        row.template_id,
                        row.rendered_at or now,
                    )
                    for row in rows
                ],
            )

    def selection(self) -> pd.DataFrame:
        """Columns ``subject_uid``, ``reason``, ``stratum``, ``inclusion_prob``, ``seed``;
        ordered by ``subject_uid``, then ``reason``."""
        rows = self._db.rows(
            f"SELECT {_names(_SELECTION_COLUMNS)} FROM selection ORDER BY subject_uid, reason"
        )
        return _frame(_SELECTION_COLUMNS, rows)

    def replace_selection(self, rows: pd.DataFrame, reasons: Collection[str]) -> None:
        """Replace the rows of those reasons by ``rows``, in one transaction.

        ``rows`` has exactly the columns of :meth:`selection`, every ``reason`` in it is
        one of ``reasons``, and no subject has a reason twice: ValueError otherwise.
        """
        names = [name for name, _ in _SELECTION_COLUMNS]
        if sorted(map(str, rows.columns)) != sorted(names):
            raise ValueError(f"selection rows have the columns {', '.join(names)}")
        wanted = set(reasons)
        other = sorted(set(map(str, rows["reason"])) - wanted)
        if other:
            raise ValueError(f"rows with a reason that is not being replaced: {', '.join(other)}")
        if rows.duplicated(["subject_uid", "reason"]).any():
            raise ValueError("a subject has the same selection reason twice")
        values = [
            (str(uid), str(reason), str(stratum), float(prob), int(seed))
            for uid, reason, stratum, prob, seed in rows[names].itertuples(index=False, name=None)
        ]
        with self._db.write() as conn:
            conn.executemany(
                "DELETE FROM selection WHERE reason = ?", [(r,) for r in sorted(wanted)]
            )
            conn.executemany(
                "INSERT INTO selection (subject_uid, reason, stratum, inclusion_prob, seed) "
                "VALUES (?, ?, ?, ?, ?)",
                values,
            )


# --- parquet tables -----------------------------------------------------------------------

_NULLABLE_DTYPES = {
    pa.int8(): pd.Int8Dtype(),
    pa.int16(): pd.Int16Dtype(),
    pa.int32(): pd.Int32Dtype(),
    pa.int64(): pd.Int64Dtype(),
    pa.bool_(): pd.BooleanDtype(),
}


def _to_arrow(frame: pd.DataFrame, schema: pa.Schema, what: str) -> pa.Table:
    """``frame`` as a table of exactly ``schema``, without index and metadata."""
    schema = schema.remove_metadata()
    columns = [str(column) for column in frame.columns]
    missing = [name for name in schema.names if name not in columns]
    extra = [name for name in columns if name not in schema.names]
    if missing or extra or len(set(columns)) != len(columns):
        problems = [
            f"{label} {', '.join(names)}"
            for label, names in (("missing", missing), ("unexpected", extra))
            if names
        ] or ["a column name twice"]
        raise ValueError(f"{what}: the columns do not fit the schema: {'; '.join(problems)}")
    try:
        table = pa.Table.from_pandas(frame[schema.names], schema=schema, preserve_index=False)
    except (pa.ArrowException, ValueError, TypeError) as exc:
        raise ValueError(f"{what}: the rows do not fit the schema: {exc}") from exc
    return table.replace_schema_metadata(None)


def _to_pandas(table: pa.Table) -> pd.DataFrame:
    data: dict[str, object] = {}
    for field in table.schema:
        column = table.column(field.name)
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
            data[field.name] = pd.Series(column.to_pylist(), dtype=object)
        elif field.nullable and field.type in _NULLABLE_DTYPES:
            data[field.name] = column.to_pandas(types_mapper=_NULLABLE_DTYPES.get)
        else:
            data[field.name] = column.to_pandas()
    return pd.DataFrame(data)


def _sorted(table: pa.Table, sort_by: Sequence[str], what: str) -> pa.Table:
    unknown = [name for name in sort_by if name not in table.schema.names]
    if unknown:
        raise ValueError(f"{what}: cannot sort by {', '.join(unknown)}: not in the schema")
    # the other columns break ties, so the order of the rows given never shows in the file
    keys = [*sort_by, *(name for name in table.schema.names if name not in sort_by)]
    return table.sort_by([(name, "ascending") for name in keys]).combine_chunks()


def _write(path: Path, table: pa.Table) -> None:
    with atomic_path(path) as tmp:
        pq.write_table(table, tmp, compression="zstd")


def _read(path: Path, schema: pa.Schema | None) -> pa.Table:
    table = pq.read_table(path).replace_schema_metadata(None)
    if schema is not None and not table.schema.equals(schema.remove_metadata()):
        raise ValueError(f"{path.name} was written with another schema than the one asked for")
    return table


def _unique(table: pa.Table, key: Sequence[str], what: str) -> None:
    counts = table.select(list(key)).group_by(list(key)).aggregate([([], "count_all")])
    if counts.num_rows != table.num_rows:
        twice = counts.filter(pc.greater(counts["count_all"], 1)).slice(0, 1).to_pylist()[0]
        twice.pop("count_all")
        raise ValueError(f"{what}: more than one row with the key {twice}")


def _in(table: pa.Table, uids: Collection[str]) -> pa.ChunkedArray:
    return pc.is_in(table["subject_uid"], value_set=pa.array(sorted(set(uids)), pa.string()))


def _table_schema(table: str, schema: pa.Schema | None) -> pa.Schema:
    if table not in KEYS:
        raise ValueError(f"no table {table!r}; the tables are {', '.join(KEYS)}")
    return (SCHEMAS[table] if schema is None else schema).remove_metadata()


def write_table(
    path: Path | str, frame: pd.DataFrame, schema: pa.Schema, *, sort_by: Sequence[str]
) -> None:
    """Cast ``frame`` to ``schema``, sort it and write zstd parquet atomically, without the
    pandas index and metadata.

    A missing or extra column, a value that does not fit its column's type and a missing
    value in a column that is not nullable are errors (ValueError). NaN counts as missing.
    Rows are sorted by ``sort_by`` and then by the remaining columns, so the bytes depend
    on the set of rows and the installed pyarrow only, not on the order they came in.
    """
    path = Path(path)
    _write(path, _sorted(_to_arrow(frame, schema, path.name), sort_by, path.name))


def read_table(path: Path | str, schema: pa.Schema | None = None) -> pd.DataFrame:
    """The rows of a parquet table.

    An absent file with a ``schema`` gives an empty frame of that schema; without one it is
    a FileNotFoundError. A file whose schema differs from ``schema`` is a ValueError.
    """
    path = Path(path)
    if not path.is_file():
        if schema is None:
            raise FileNotFoundError(path)
        return _to_pandas(schema.remove_metadata().empty_table())
    return _to_pandas(_read(path, schema))


def media_dir(run_dir: Path | str, subject_uid: str) -> Path:
    """``media/<subject_uid>`` of a run directory: the render of one subject."""
    return Path(run_dir) / MEDIA_DIR / subject_uid


def table_path(run_dir: Path | str, table: str) -> Path:
    """``tables/<table>.parquet`` of a run directory."""
    return Path(run_dir) / "tables" / f"{table}.parquet"


def primary_scores(run_dir: Path | str) -> pd.DataFrame:
    """The primary rows of ``tables/scores.parquet``, one per present subject, sorted by
    ``primary_rank``: the one order of the subjects of a run (``rank`` is within a
    reference). Empty before ``metrics`` has run. What ``Run.scores()`` returns, without the
    labels; the rater app, the export and the queue read the scores through this."""
    scores = read_table(table_path(run_dir, "scores"), SCHEMAS["scores"])
    scores = scores[scores["primary"].to_numpy(dtype=bool)]
    return scores.sort_values("primary_rank").reset_index(drop=True)


def queue_order(
    uids: Collection[str], scores: pd.DataFrame | None, *, blind: bool, seed: int
) -> list[str]:
    """The order in which a local run offers subjects for rating, and in which it renders
    them, so that the first renders are the first subjects of the queue.

    Not blind: first the subjects flagged ``partial_score``, by ``subject_uid`` (they could
    not be fully scored and would otherwise sink); then by ``primary_rank`` ascending, worst
    first; then the subjects without a score, by ``subject_uid``. ``scores`` are rows of
    ``scores.parquet`` (:func:`primary_scores`; a row that is not primary is ignored), or
    None for a run without scores. Blind: ``random.Random(seed).shuffle`` of the sorted
    uids, whatever the scores.
    """
    ordered = sorted(set(uids))
    if blind:
        random.Random(seed).shuffle(ordered)
        return ordered
    position: dict[str, tuple[int, int]] = {}
    if scores is not None:
        for uid, rank, flags in zip(
            scores["subject_uid"], scores["primary_rank"], scores["flags"], strict=True
        ):
            if not pd.isna(rank):
                partial = "partial_score" in split_flags(flags)
                position[uid] = (0, 0) if partial else (1, int(rank))
    return sorted(ordered, key=lambda uid: (*position.get(uid, (2, 0)), uid))


def clusters_with_native(run_dir: Path | str) -> pd.DataFrame:
    """``clusters.parquet`` left-joined with ``clusters_native.parquet`` on the key of the
    cluster table, which holds the reference: a native position of an earlier reference is
    never attached to a cluster of the present one. What ``Run.clusters()`` returns."""
    clusters = read_table(table_path(run_dir, "clusters"), SCHEMAS["clusters"])
    native = read_table(table_path(run_dir, "clusters_native"), SCHEMAS["clusters_native"])
    return clusters.merge(native, how="left", on=list(KEYS["clusters"]))


def part_path(run_dir: Path | str, table: str, dataset: str) -> Path:
    """``tables/_parts/<table>/<pseudonym>.parquet``: the rows of one dataset.

    ``dataset`` must be a pseudonym (ValueError), so that no accession becomes a file name.
    """
    if not _PSEUDONYM.fullmatch(dataset):
        raise ValueError(f"a part is named by the pseudonym of its dataset, got {dataset!r}")
    return Path(run_dir) / "tables" / "_parts" / table / f"{dataset}.parquet"


def upsert_part(
    run_dir: Path | str,
    table: str,
    dataset: str,
    frame: pd.DataFrame,
    *,
    replace_uids: Collection[str],
    schema: pa.Schema | None = None,
) -> Path:
    """Drop the rows of ``replace_uids`` from the dataset's part of ``table``, append
    ``frame``, write the part sorted by ``schemas.KEYS[table]``; returns its path.

    A subject without rows is replaced by nothing: name it in ``replace_uids`` and leave it
    out of ``frame``. The part is written even when it ends up empty. ``schema`` defaults
    to ``schemas.SCHEMAS[table]`` (the intensity table of a run with other depths has its
    own). Two rows with the same key are a ValueError.
    """
    wanted = _table_schema(table, schema)
    path = part_path(run_dir, table, dataset)
    what = f"{table} part of {dataset}"
    rows = _to_arrow(frame, wanted, what)
    if path.is_file():
        old = _read(path, wanted)
        rows = pa.concat_tables([old.filter(pc.invert(_in(old, replace_uids))), rows])
    _unique(rows, KEYS[table], what)
    _write(path, _sorted(rows, KEYS[table], what))
    return path


def merge_parts(
    run_dir: Path | str,
    table: str,
    *,
    present: Collection[str],
    schema: pa.Schema | None = None,
) -> Path:
    """Concatenate the parts of ``table``, drop rows whose ``subject_uid`` is not in
    ``present``, sort by ``schemas.KEYS[table]`` and write ``tables/<table>.parquet``;
    returns its path. Without parts the table is written empty."""
    wanted = _table_schema(table, schema)
    parts = sorted((Path(run_dir) / "tables" / "_parts" / table).glob("*.parquet"))
    rows = pa.concat_tables([wanted.empty_table(), *(_read(part, wanted) for part in parts)])
    rows = rows.filter(_in(rows, present))
    _unique(rows, KEYS[table], table)
    path = table_path(run_dir, table)
    _write(path, _sorted(rows, KEYS[table], table))
    return path
