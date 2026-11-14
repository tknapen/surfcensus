"""Tests of surfcensus/store.py: the catalog (private and public side) and the parquet tables."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import random
import re
import sqlite3
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import surfcensus
from surfcensus import privacy, store
from surfcensus.config import ConfigError
from surfcensus.privacy import HashKey
from surfcensus.records import Provenance, RenderRow, WorkRow
from surfcensus.schemas import KEYS, SCHEMAS, intensity_schema
from surfcensus.sources import base
from surfcensus.sources.base import (
    RENDER_ROLES,
    TIER_A_REQUIRED,
    TIER_B_REQUIRED,
    DatasetRecord,
    FileRef,
    FileRole,
    SubjectRecord,
)
from surfcensus.sources.patterns import ROLE_CANDIDATES
from surfcensus.store import CATALOG_FILE, Catalog, PublicCatalog

SALT = HashKey(bytes(range(32)), "local-salt")
KEY = HashKey(bytes(range(1, 33)), "dataset-key")
ISO_UTC = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00")
TABLES = {
    "meta",
    "datasets",
    "subjects",
    "files",
    "dataset_files",
    "templates",
    "work",
    "renders",
    "selection",
    # migration 2: the EPI interface in a run directory
    "func_sources",
    "func_participants",
    "func_sessions",
    "func_runs",
    "func_files",
    "packs",
}
VIEWS = {"public_datasets", "public_subjects", "public_files"}
# migration 2; what they hold is tested in tests/test_store_func.py
FUNC_VIEWS = {
    "public_func_sources",
    "public_func_participants",
    "public_func_sessions",
    "public_func_runs",
    "public_packs",
}
# item 1 of the contract's list of private values (section 5.1), as column names
PRIVATE_COLUMNS = {
    "dataset_id",
    "identity",
    "accession",
    "root",
    "fs_root",
    "template_location",
    "derivative_ref",
    "snapshot_tag",
    "fs_dir",
    "inventory_hash",
    "location",
    "size",
    "etag_or_mtime",
    "last_modified",
    "version_id",
    "sha256",
    "input_signature",
    "run_secret",
    "bucket",
    "key",
    "etag",
    "detail",
}
DIGEST = hashlib.sha256(b"canary").hexdigest()


# --- records ------------------------------------------------------------------------------


def make_dataset(**changes: object) -> DatasetRecord:
    record = DatasetRecord(
        source="local",
        identity="/CANARYROOT/study",
        accession=None,
        root="/CANARYROOT/study",
        fs_root="sourcedata/freesurfer",
        layout="fmriprep",
        pipeline="fmriprep",
        pipeline_version="23.2.0",
        template_location="sourcedata/freesurfer/fsaverage",
    )
    return dataclasses.replace(record, **changes)


def remote_dataset(accession: str, **changes: object) -> DatasetRecord:
    return make_dataset(
        source="openneuro-derivatives",
        identity=accession,
        accession=accession,
        root=f"s3://openneuro-derivatives/fmriprep/{accession}-fmriprep/",
        derivative_ref="refCANARY0123",
        snapshot_tag="tagCANARY1.0.0",
        **changes,
    )


def make_files(roles: Sequence[FileRole] = tuple(FileRole), *, stamp: str = "1") -> dict:
    """One ``FileRef`` per role, keyed in ``FileRole`` order, as a source yields them."""
    files = {}
    for number, role in enumerate(FileRole):
        if role not in roles:
            continue
        location = ROLE_CANDIDATES[role][0]
        files[role] = FileRef(
            role=role,
            location=location,
            size=1000 + number,
            etag_or_mtime=f"CANARYETAG{number:02d}{stamp}",
        )
    return files


def make_subject(
    dataset: DatasetRecord,
    label: str = "sub-CANARY7",
    roles: Sequence[FileRole] = tuple(FileRole),
    *,
    stamp: str = "1",
    problem: str | None = None,
) -> SubjectRecord:
    fs_dir = label if dataset.fs_root == "." else f"{dataset.fs_root}/{label}"
    return SubjectRecord(dataset, label, fs_dir, make_files(roles, stamp=stamp), problem)


def make_provenance(**changes: object) -> Provenance:
    record = Provenance(
        build_stamp="freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.1-f53a55a",
        fs_version="6.0.1",
        fs_major=6,
        flags=("-all", "-noT2pial", "-subjid"),
        n_invocations=4,
        finished=True,
        t2pial=False,
        flairpial=False,
        hires=False,
        expert=None,
        xopts=False,
        edited=True,
        pial_variant="pial",
        holes=(41, 26),
        truncations={"lh": (39, 225), "rh": (93, 434)},
        participant="sub-CANARY7",
        longitudinal=False,
    )
    return dataclasses.replace(record, **changes)


UNKNOWN = make_provenance(
    build_stamp=None,
    fs_version=None,
    fs_major=None,
    flags=(),
    n_invocations=None,
    finished=None,
    t2pial=None,
    flairpial=None,
    hires=None,
    expert=None,
    xopts=None,
    edited=None,
    pial_variant=None,
    holes=None,
    truncations={"lh": None, "rh": None},
    participant=None,
)


def uid_of(key: HashKey, dataset: DatasetRecord, label: str) -> str:
    return privacy.subject_uid(key, dataset.source, dataset.identity, dataset.pipeline, label)


def pseudonym_of(key: HashKey, dataset: DatasetRecord) -> str:
    return privacy.dataset_pseudonym(key, dataset.source, dataset.identity)


def add_subject(
    catalog: Catalog,
    key: HashKey,
    dataset: DatasetRecord,
    label: str = "sub-CANARY7",
    roles: Sequence[FileRole] = tuple(FileRole),
    **more: object,
) -> str:
    """Upsert the dataset and one subject of it; returns the ``subject_uid``."""
    dataset_id = catalog.upsert_dataset(dataset, pseudonym_of(key, dataset), None)
    record = make_subject(dataset, label, roles, **more)
    uid = uid_of(key, dataset, label)
    catalog.upsert_subject(uid, dataset_id, record, make_provenance(participant=label), "h" * 16)
    return uid


def row(catalog: Catalog, sql: str, *params: object) -> dict[str, object]:
    rows = catalog.query(sql, params)
    assert len(rows) == 1, rows
    return dict(rows[0])


def count(catalog: Catalog, table: str) -> int:
    return catalog.query(f"SELECT count(*) FROM {table}")[0][0]


@pytest.fixture
def local(tmp_path: Path) -> Iterator[Catalog]:
    with Catalog.create(tmp_path / "run" / CATALOG_FILE, mode="local", key=SALT) as catalog:
        yield catalog


@pytest.fixture
def census(tmp_path: Path) -> Iterator[Catalog]:
    with Catalog.create(tmp_path / "census" / CATALOG_FILE, mode="census", key=KEY) as catalog:
        yield catalog


def run_dir_of(catalog: Catalog) -> Path:
    return Path(catalog.query("PRAGMA database_list")[0]["file"]).parent


# --- create and reopen --------------------------------------------------------------------


def test_create_writes_the_identity_of_the_run_and_reopening_reads_it(tmp_path: Path) -> None:
    path = tmp_path / "new" / "run" / CATALOG_FILE
    with Catalog.create(path, mode="local", key=SALT) as catalog:
        meta = {r["key"]: r["value"] for r in catalog.query("SELECT key, value FROM meta")}
        assert set(meta) == {
            "run_id",
            "mode",
            "key_kind",
            "key_fingerprint",
            "created_at",
            "surfcensus_version",
            "run_secret",
        }
        assert re.fullmatch(r"[0-9a-f]{32}", meta["run_id"])
        assert meta["mode"] == "local" and meta["key_kind"] == "local-salt"
        assert meta["key_fingerprint"] == SALT.fingerprint
        assert meta["surfcensus_version"] == surfcensus.__version__
        assert ISO_UTC.fullmatch(meta["created_at"])
        secret = catalog.run_secret()
        assert len(secret) == 32 and secret.hex() == meta["run_secret"]
        assert catalog.meta("template_id") is None and catalog.meta("nothing") is None
        assert catalog.query("PRAGMA user_version")[0][0] == store.SCHEMA_VERSION == 2
    with Catalog(path) as again:
        assert {r["key"]: r["value"] for r in again.query("SELECT key, value FROM meta")} == meta
        assert again.run_secret() == secret
    with Catalog(path, readonly=True) as readonly:
        assert readonly.meta("run_id") == meta["run_id"]


def test_two_runs_have_different_ids_and_secrets_and_neither_holds_the_key(tmp_path: Path) -> None:
    one = Catalog.create(tmp_path / "a" / CATALOG_FILE, mode="census", key=KEY)
    two = Catalog.create(tmp_path / "b" / CATALOG_FILE, mode="census", key=KEY)
    assert one.meta("run_id") != two.meta("run_id")
    assert one.run_secret() != two.run_secret()
    assert one.meta("key_fingerprint") == two.meta("key_fingerprint") == KEY.fingerprint
    one.close()
    two.close()
    for path in sorted(tmp_path.rglob("catalog.sqlite*")):
        content = path.read_bytes()
        assert KEY.secret not in content and KEY.secret.hex().encode() not in content


def test_a_catalog_has_the_tables_and_views_of_the_contract(local: Catalog) -> None:
    found = {
        (r["type"], r["name"])
        for r in local.query("SELECT type, name FROM sqlite_master WHERE type IN ('table','view')")
    }
    assert found == {("table", name) for name in TABLES} | {
        ("view", name) for name in VIEWS | FUNC_VIEWS
    }
    assert local.query("PRAGMA foreign_keys")[0][0] == 1


def test_journal_and_busy_timeout_follow_the_mode(local: Catalog, census: Catalog) -> None:
    assert local.query("PRAGMA journal_mode")[0][0] == "delete"
    assert census.query("PRAGMA journal_mode")[0][0] == "wal"
    assert local.query("PRAGMA busy_timeout")[0][0] == 30_000
    assert census.query("PRAGMA busy_timeout")[0][0] == 60_000
    with PublicCatalog.open(run_dir_of(local)) as public:
        assert public._db.rows("PRAGMA busy_timeout")[0][0] == 30_000
    with PublicCatalog.open(run_dir_of(census)) as public:
        assert public._db.rows("PRAGMA busy_timeout")[0][0] == 5_000
    with Catalog(run_dir_of(census) / CATALOG_FILE) as again:  # WAL is a property of the file
        assert again.query("PRAGMA journal_mode")[0][0] == "wal"
        assert again.query("PRAGMA busy_timeout")[0][0] == 60_000


def test_create_refuses_a_key_of_the_other_kind_and_an_existing_file(tmp_path: Path) -> None:
    path = tmp_path / CATALOG_FILE
    for mode, key in (("local", KEY), ("census", SALT)):
        with pytest.raises(ConfigError) as caught:
            Catalog.create(path, mode=mode, key=key)
        assert caught.value.reason == "key"
        assert not path.exists()
    Catalog.create(path, mode="local", key=SALT).close()
    before = path.read_bytes()
    with pytest.raises(ConfigError, match=r"already has a catalog\.sqlite") as caught:
        Catalog.create(path, mode="local", key=SALT)
    assert caught.value.reason == "invalid"
    assert path.read_bytes() == before


def test_a_failed_create_leaves_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store, "MIGRATIONS", ("CREATE TABLE meta (key TEXT);\nCREATE TABLE (;",))
    with pytest.raises(sqlite3.OperationalError):
        Catalog.create(tmp_path / CATALOG_FILE, mode="census", key=KEY)
    assert list(tmp_path.iterdir()) == []


# --- a catalog holds a run or does not exist ----------------------------------------------

META_KEYS = {
    "run_id",
    "mode",
    "key_kind",
    "key_fingerprint",
    "created_at",
    "surfcensus_version",
    "run_secret",
}
KEY_OF = {"local": SALT, "census": KEY}
JOURNAL_OF = {"local": "delete", "census": "wal"}


def _looked_at(path: Path) -> dict[str, object]:
    """What a database file holds, read through a connection of its own."""
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        return {
            "version": conn.execute("PRAGMA user_version").fetchone()[0],
            "journal": conn.execute("PRAGMA journal_mode").fetchone()[0],
            "meta": dict(conn.execute("SELECT key, value FROM meta").fetchall()),
        }
    finally:
        conn.close()


def _tables_without_a_run(path: Path) -> None:
    """What a version before the catalog was built under another name left when it was
    killed between the schema and the identity of the run."""
    conn = sqlite3.connect(path, isolation_level=None)
    store._migrate(conn)
    conn.close()


UNFINISHED = {"an empty file": Path.touch, "tables without a row": _tables_without_a_run}


@pytest.mark.parametrize("mode", ["local", "census"])
def test_a_catalog_gets_its_name_when_it_is_whole(
    mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another process that opens the run, or a kill, must never find a ``catalog.sqlite``
    without a run in it."""
    path = tmp_path / CATALOG_FILE
    seen: list[dict[str, object]] = []
    link = os.link

    def linking(src: Path, dst: Path) -> None:
        assert Path(dst) == path and not path.exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == [Path(src).name]  # no journal, no log
        assert re.fullmatch(r"\.catalog\.sqlite\.[0-9a-f]{16}\.tmp", Path(src).name)
        seen.append(_looked_at(Path(src)))
        link(src, dst)

    monkeypatch.setattr(os, "link", linking)
    with Catalog.create(path, mode=mode, key=KEY_OF[mode]) as catalog:
        (whole,) = seen
        assert whole["version"] == store.SCHEMA_VERSION and whole["journal"] == JOURNAL_OF[mode]
        assert set(whole["meta"]) == META_KEYS and whole["meta"]["mode"] == mode
        assert catalog.run_secret().hex() == whole["meta"]["run_secret"]
        assert catalog.meta("run_id") == whole["meta"]["run_id"]
    assert [p.name for p in tmp_path.iterdir()] == [CATALOG_FILE]


def test_a_create_that_is_stopped_before_the_name_is_given_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(src: object, dst: object) -> None:
        raise KeyboardInterrupt

    with monkeypatch.context() as patched:
        patched.setattr(os, "link", interrupted)
        with pytest.raises(KeyboardInterrupt):
            Catalog.create(tmp_path / CATALOG_FILE, mode="census", key=KEY)
    assert list(tmp_path.iterdir()) == []
    Catalog.create(tmp_path / CATALOG_FILE, mode="census", key=KEY).close()


KILLED = """
import os, signal, sys
from surfcensus import store
from surfcensus.privacy import HashKey

where, mode, kind = sys.argv[1:4]


def killed(*args, **kwargs):
    os.kill(os.getpid(), signal.SIGKILL)


def no_hard_links(src, dst):
    raise PermissionError(1, "Operation not permitted")


if where == "before the name":
    store._publish = killed
else:  # in the transaction that makes the run in the file itself
    os.link = no_hard_links
    fill = store._fill

    def fill_in_place(path, **kwargs):
        if path.name == store.CATALOG_FILE:
            store._now = killed
        fill(path, **kwargs)

    store._fill = fill_in_place
key = HashKey(bytes.fromhex(sys.argv[5]), kind)
store.Catalog.create(sys.argv[4], mode=mode, key=key)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="needs SIGKILL")
@pytest.mark.parametrize("mode", ["local", "census"])
@pytest.mark.parametrize("where", ["before the name", "in the file itself"])
def test_a_create_that_is_killed_leaves_no_catalog_or_one_that_the_next_finishes(
    where: str, mode: str, tmp_path: Path
) -> None:
    path, key = tmp_path / CATALOG_FILE, KEY_OF[mode]
    child = subprocess.run(
        [sys.executable, "-c", KILLED, where, mode, key.kind, str(path), key.secret.hex()],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert child.returncode == -9, child.stderr
    if where == "before the name":
        assert not path.exists()
        with pytest.raises(ConfigError) as caught:
            Catalog(path)
        assert caught.value.reason == "missing"
    else:
        assert path.stat().st_size == 0
        with pytest.raises(store.UnfinishedCatalog):
            Catalog(path)
    with Catalog.create(path, mode=mode, key=key, exist_ok=True) as catalog:
        assert catalog.meta("mode") == mode and len(catalog.run_secret()) == 32
        assert catalog.query("PRAGMA journal_mode")[0][0] == JOURNAL_OF[mode]
        assert catalog.query("PRAGMA integrity_check")[0][0] == "ok"
        assert count(catalog, "meta") == 7


@pytest.mark.parametrize("mode", ["local", "census"])
@pytest.mark.parametrize("state", list(UNFINISHED))
def test_an_unfinished_catalog_is_left_alone_by_opening_and_finished_by_create(
    state: str, mode: str, tmp_path: Path
) -> None:
    path, key = tmp_path / CATALOG_FILE, KEY_OF[mode]
    UNFINISHED[state](path)
    before = path.read_bytes()
    openers = (
        lambda: Catalog(path),
        lambda: Catalog(path, readonly=True),
        lambda: PublicCatalog.open(tmp_path),
    )
    for opener in openers:
        with pytest.raises(store.UnfinishedCatalog, match="holds no run yet") as caught:
            opener()
        assert caught.value.reason == "invalid"
        assert "not made by surfcensus" not in str(caught.value)  # that would say: delete it
    with pytest.raises(ConfigError, match=r"already has a catalog\.sqlite"):
        Catalog.create(path, mode=mode, key=key)
    # nothing was migrated, nothing written: the file is as the killed process left it
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == [CATALOG_FILE]

    with Catalog.create(path, mode=mode, key=key, exist_ok=True) as catalog:
        meta = {r["key"]: r["value"] for r in catalog.query("SELECT key, value FROM meta")}
        assert set(meta) == META_KEYS
        assert (meta["mode"], meta["key_fingerprint"]) == (mode, key.fingerprint)
        assert catalog.query("PRAGMA user_version")[0][0] == store.SCHEMA_VERSION
        assert catalog.query("PRAGMA journal_mode")[0][0] == JOURNAL_OF[mode]
        names = {r[0] for r in catalog.query("SELECT name FROM sqlite_master WHERE type='table'")}
        assert names == TABLES
    with Catalog.create(path, mode=mode, key=key, exist_ok=True) as again:  # now it is whole
        assert again.meta("run_id") == meta["run_id"]
        assert again.run_secret().hex() == meta["run_secret"]


def test_an_existing_catalog_of_another_run_is_refused_also_with_exist_ok(tmp_path: Path) -> None:
    path = tmp_path / CATALOG_FILE
    Catalog.create(path, mode="local", key=SALT).close()
    before = path.read_bytes()
    other = HashKey(bytes(range(2, 34)), "local-salt")
    with pytest.raises(ConfigError, match="of another run") as caught:
        Catalog.create(path, mode="local", key=other, exist_ok=True)
    assert caught.value.reason == "invalid"
    assert path.read_bytes() == before


def _a_table_with_a_row(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE notes (text)")
    conn.execute("INSERT INTO notes VALUES ('mine')")


def _an_empty_table(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE notes (text)")


def _a_numbered_schema_of_its_own(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE notes (text)")
    conn.execute("PRAGMA user_version = 1")


def _a_row_in_the_tables_of_a_catalog(conn: sqlite3.Connection) -> None:
    store._migrate(conn)
    conn.execute("INSERT INTO meta (key, value) VALUES ('template_id', 'abc')")


@pytest.mark.parametrize(
    "fill",
    [
        _a_table_with_a_row,
        _an_empty_table,
        _a_numbered_schema_of_its_own,
        _a_row_in_the_tables_of_a_catalog,
    ],
)
def test_a_database_that_something_else_made_is_never_taken_for_unfinished(
    fill, tmp_path: Path
) -> None:
    """Only a file that holds nothing may be filled: here a run would be made in someone
    else's database, or over the rows of a catalog that lost its identity."""
    path = tmp_path / CATALOG_FILE
    conn = sqlite3.connect(path, isolation_level=None)
    fill(conn)
    conn.close()
    before = path.read_bytes()
    attempts = (
        lambda: Catalog(path),
        lambda: PublicCatalog.open(tmp_path),
        lambda: Catalog.create(path, mode="local", key=SALT, exist_ok=True),
    )
    for attempt in attempts:
        with pytest.raises(ConfigError, match="it was not made by surfcensus") as caught:
            attempt()
        assert not isinstance(caught.value, store.UnfinishedCatalog)
    store._fill(path, mode="local", key=SALT)  # looks again under the lock, and writes nothing
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == [CATALOG_FILE]


@pytest.mark.parametrize("mode", ["local", "census"])
def test_without_hard_links_the_run_is_made_in_the_file_itself(
    mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_hard_links(src: object, dst: object) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "link", no_hard_links)
    path, key = tmp_path / CATALOG_FILE, KEY_OF[mode]
    with Catalog.create(path, mode=mode, key=key) as catalog:
        assert catalog.meta("mode") == mode and count(catalog, "meta") == 7
        assert catalog.query("PRAGMA journal_mode")[0][0] == JOURNAL_OF[mode]
        run_id = catalog.meta("run_id")
    assert [p.name for p in tmp_path.iterdir()] == [CATALOG_FILE]
    before = path.read_bytes()
    with pytest.raises(ConfigError, match=r"already has a catalog\.sqlite"):
        Catalog.create(path, mode=mode, key=key)
    assert path.read_bytes() == before
    with Catalog.create(path, mode=mode, key=key, exist_ok=True) as again:
        assert again.meta("run_id") == run_id

    # the name was taken between the look and the claim: nothing of the other is replaced
    other = tmp_path / "other" / CATALOG_FILE
    lexists = os.path.lexists

    def taken_meanwhile(candidate: object) -> bool:
        if Path(str(candidate)) == other and not lexists(other):
            other.write_bytes(before)
            return False
        return lexists(candidate)

    monkeypatch.setattr(os.path, "lexists", taken_meanwhile)
    with Catalog.create(other, mode=mode, key=key, exist_ok=True) as adopted:
        assert adopted.meta("run_id") == run_id
    assert [p.name for p in other.parent.iterdir()] == [CATALOG_FILE]


CREATOR = """
import os, sys, time
from pathlib import Path
from surfcensus import store
from surfcensus.privacy import HashKey

path, tag, mode, kind = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
if sys.argv[6] == "no hard links":
    def no_hard_links(src, dst):
        raise PermissionError(1, "Operation not permitted")
    os.link = no_hard_links
key = HashKey(bytes.fromhex(sys.argv[5]), kind)
(path.parent / f"ready-{tag}").touch()
while not (path.parent / "go").exists():
    time.sleep(0.001)
with store.Catalog.create(path, mode=mode, key=key, exist_ok=True) as catalog:
    catalog.set_meta(f"seen-{tag}", catalog.meta("run_id"))
    print(catalog.meta("run_id"), catalog.run_secret().hex())
"""


@pytest.mark.parametrize("mode", ["local", "census"])
@pytest.mark.parametrize("start", ["no file", "an empty file", "no hard links"])
def test_processes_that_make_one_catalog_at_once_all_get_the_same_run(
    start: str, mode: str, tmp_path: Path
) -> None:
    """A run is opened while its catalog is being made, or several commands find a run whose
    making was killed. Each makes the catalog; one of them wins and the others take its."""
    path, key = tmp_path / CATALOG_FILE, KEY_OF[mode]
    if start == "an empty file":
        path.touch()
    tags = ("a", "b", "c", "d")
    makers = [
        subprocess.Popen(
            [sys.executable, "-c", CREATOR, path, tag, mode, key.kind, key.secret.hex(), start],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for tag in tags
    ]
    try:
        deadline = time.monotonic() + 120
        while not all((tmp_path / f"ready-{tag}").exists() for tag in tags):
            assert all(maker.poll() is None for maker in makers), "a maker ended early"
            assert time.monotonic() < deadline, "the makers did not start"
            time.sleep(0.02)
        (tmp_path / "go").touch()
        results = [maker.communicate(timeout=300) for maker in makers]
    finally:
        for maker in makers:
            maker.kill()
    assert [maker.returncode for maker in makers] == [0, 0, 0, 0], [err for _, err in results]
    assert len({out for out, _ in results}) == 1, results  # one run id, one run secret
    with Catalog(path) as catalog:
        run_id = catalog.meta("run_id")
        assert results[0][0].split() == [run_id, catalog.run_secret().hex()]
        # every process wrote to the catalog that has the name, none to a file it lost
        assert [catalog.meta(f"seen-{tag}") for tag in tags] == [run_id] * 4
        assert catalog.query("PRAGMA integrity_check")[0][0] == "ok"
    left = {p.name for p in tmp_path.iterdir()} - {"go", *(f"ready-{tag}" for tag in tags)}
    assert left - {f"{CATALOG_FILE}-wal", f"{CATALOG_FILE}-shm"} == {CATALOG_FILE}


def test_opening_what_is_not_a_catalog(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=f"has no {CATALOG_FILE}") as caught:
        Catalog(tmp_path / CATALOG_FILE)
    assert caught.value.reason == "missing"
    with pytest.raises(ConfigError) as caught:
        PublicCatalog.open(tmp_path)
    assert caught.value.reason == "missing"
    assert list(tmp_path.iterdir()) == []  # looking for a catalog does not create one

    (tmp_path / CATALOG_FILE).write_bytes(b"this is not a database, it is a sentence" * 30)
    for opener in (lambda: Catalog(tmp_path / CATALOG_FILE), lambda: PublicCatalog.open(tmp_path)):
        with pytest.raises(ConfigError, match="not a SQLite database") as caught:
            opener()
        assert caught.value.reason == "invalid"


def test_a_catalog_that_another_connection_holds_is_in_use_and_not_damaged(
    local: Catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = run_dir_of(local) / CATALOG_FILE
    monkeypatch.setitem(store._BUSY_MS, "census", 50)  # the timeout a connection starts with
    local.query("PRAGMA busy_timeout = 50")
    holder = sqlite3.connect(path, isolation_level=None)
    holder.execute("BEGIN EXCLUSIVE")
    try:
        for opener in (lambda: Catalog(path), lambda: PublicCatalog.open(path.parent)):
            with pytest.raises(ConfigError, match="in use by another process") as caught:
                opener()
            assert caught.value.reason == "invalid"
            assert "not a SQLite database" not in caught.value.detail
            assert isinstance(caught.value.__cause__, sqlite3.OperationalError)
        # the connection that is open already says the same when it comes to write
        with pytest.raises(ConfigError, match=r"catalog\.sqlite is in use by another process"):
            local.set_meta("template_id", "x")
    finally:
        holder.close()
    local.set_meta("template_id", "x")
    assert local.meta("template_id") == "x"


def test_a_full_disk_is_reported_as_what_it_is(local: Catalog, tmp_path: Path) -> None:
    """SQLite ends the transaction itself when the file cannot grow. A ROLLBACK then fails
    with "cannot rollback - no transaction is active", which would hide the cause."""
    pages = local.query("PRAGMA page_count")[0][0]
    local.query(f"PRAGMA max_page_count = {pages}")  # of this connection: no room to grow
    with pytest.raises(sqlite3.OperationalError, match="database or disk is full"):
        local.set_meta("template_id", "x" * 1_000_000)
    with pytest.raises(sqlite3.OperationalError, match="database or disk is full"):
        with local.write() as conn:
            conn.execute("INSERT INTO meta (key, value) VALUES ('big', ?)", ("x" * 1_000_000,))
    local.query("PRAGMA max_page_count = 1000000")
    local.set_meta("template_id", "x")  # the connection is usable again
    assert local.meta("template_id") == "x" and local.meta("big") is None

    path = tmp_path / "empty.sqlite"
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("PRAGMA max_page_count = 1")
    with pytest.raises(sqlite3.OperationalError, match="database or disk is full"):
        store._migrate(conn)
    conn.close()


def test_the_identity_of_a_run_cannot_be_overwritten(local: Catalog) -> None:
    secret = local.run_secret()
    for key in ("run_id", "mode", "key_kind", "key_fingerprint", "created_at", "run_secret"):
        with pytest.raises(ValueError, match="fixed"):
            local.set_meta(key, "0" * 64)
        local.set_meta(key, local.meta(key) or "")  # the same value is not a change
    assert local.run_secret() == secret
    local.set_meta("template_id", "abc123abc123")
    local.set_meta("template_id", "def456def456")
    assert local.meta("template_id") == "def456def456"
    assert count(local, "meta") == 8


def test_a_census_run_on_a_network_file_system_gets_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(store, "filesystem_type", lambda path: "nfs4")
    with caplog.at_level(logging.WARNING, logger="surfcensus"):
        Catalog.create(tmp_path / "local" / CATALOG_FILE, mode="local", key=SALT).close()
        assert caplog.records == []  # a local run keeps the rollback journal, which works there
        Catalog.create(tmp_path / "census" / CATALOG_FILE, mode="census", key=KEY).close()
    assert [r.levelname for r in caplog.records] == ["WARNING"]
    assert "network file system (nfs4)" in caplog.records[0].getMessage()

    caplog.clear()
    monkeypatch.setattr(store, "filesystem_type", lambda path: "ext4")
    with caplog.at_level(logging.WARNING, logger="surfcensus"):
        Catalog.create(tmp_path / "disk" / CATALOG_FILE, mode="census", key=KEY).close()
    assert caplog.records == []


def test_a_census_catalog_that_could_not_be_switched_is_switched_by_a_later_opening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The switch to the write-ahead log needs the file to itself, and where a run is made
    in the catalog file itself another process may have it open by then."""
    path = tmp_path / CATALOG_FILE
    with monkeypatch.context() as refused, caplog.at_level(logging.WARNING, logger="surfcensus"):
        refused.setattr(store, "_use_wal", lambda conn: False)
        with Catalog.create(path, mode="census", key=KEY) as catalog:
            assert catalog.query("PRAGMA journal_mode")[0][0] == "delete"
            catalog.set_meta("template_id", "abc")  # it works, with a rollback journal
    (warning,) = [r.getMessage() for r in caplog.records]
    assert "could not be put in write-ahead mode (it is in delete mode)" in warning

    with Catalog(path, readonly=True) as reader:  # a reader changes nothing
        assert reader.query("PRAGMA journal_mode")[0][0] == "delete"
    writer = sqlite3.connect(path, isolation_level=None)
    writer.execute("BEGIN IMMEDIATE")
    with Catalog(path) as meanwhile:  # the switch is refused: the opening goes on without
        assert meanwhile.query("PRAGMA journal_mode")[0][0] == "delete"
        assert meanwhile.meta("template_id") == "abc"
    writer.execute("ROLLBACK")
    writer.close()
    with Catalog(path) as alone:
        assert alone.query("PRAGMA journal_mode")[0][0] == "wal"
        assert alone.meta("template_id") == "abc"


def test_filesystem_type_reads_the_mount_that_holds_the_path(tmp_path: Path) -> None:
    inside = tmp_path / "data" / "my run" / "deeper"
    inside.mkdir(parents=True)
    spaced = str(tmp_path / "data" / "my run").replace(" ", "\\040")
    mounts = "\n".join(
        [
            "/dev/root / ext4 rw,relatime 0 0",
            f"server:/export {tmp_path / 'data'} nfs4 rw,vers=4.2 0 0",
            f"tmpfs {spaced} tmpfs rw 0 0",
            f"//nas/share {spaced} cifs rw 0 0",
            "malformed",
            "",
        ]
    )
    assert store.filesystem_type(tmp_path, mounts) == "ext4"
    assert store.filesystem_type(tmp_path / "data", mounts) == "nfs4"
    assert store.filesystem_type(tmp_path / "data" / "other" / "x.sqlite", mounts) == "nfs4"
    assert store.filesystem_type(inside, mounts) == "cifs"  # the later mount hides the tmpfs
    assert store.filesystem_type(tmp_path / "database", mounts) == "ext4"  # not under data/
    assert store.filesystem_type(tmp_path, "") is None
    real = store.filesystem_type(tmp_path)
    assert real is None or (isinstance(real, str) and real)


# --- migrations ---------------------------------------------------------------------------


def test_migrations_are_applied_to_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / CATALOG_FILE
    path.touch()
    conn = sqlite3.connect(path, isolation_level=None)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
    store._migrate(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == store.SCHEMA_VERSION
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert names == TABLES
    store._migrate(conn)  # nothing pending: nothing done, no error
    conn.close()
    # the tables are there, but no run was made in the file
    with pytest.raises(store.UnfinishedCatalog, match="holds no run yet") as caught:
        Catalog(path)
    assert caught.value.reason == "invalid"


def test_the_schema_version_counts_the_migrations() -> None:
    assert len(store.MIGRATIONS) == store.SCHEMA_VERSION
    statements = list(store._statements(store.MIGRATIONS[0]))
    assert len(statements) == 13  # nine tables, one index, three views
    assert all(sqlite3.complete_statement(statement) for statement in statements)
    first = "-- a comment\n\nCREATE TABLE a (\n  x TEXT DEFAULT ';' -- one; two\n);"
    assert list(store._statements(first + "\nCREATE TABLE c (y)\n")) == [
        first,
        "CREATE TABLE c (y)",  # unterminated, and still handed to SQLite
    ]
    assert list(store._statements("\n-- nothing but a comment\n\n")) == []


MIGRATION_3 = """
ALTER TABLE datasets ADD COLUMN note TEXT;
-- a second statement, with a semicolon in a string: ';'
CREATE TABLE extra (x TEXT DEFAULT ';');
"""


def test_a_test_migration_preserves_the_rows(
    local: Catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = make_dataset()
    uid = add_subject(local, SALT, dataset)
    record = local.subject_record(uid)
    meta = [tuple(r) for r in local.query("SELECT key, value FROM meta ORDER BY key")]
    path = run_dir_of(local) / CATALOG_FILE
    local.close()

    monkeypatch.setattr(store, "MIGRATIONS", (*store.MIGRATIONS, MIGRATION_3))
    monkeypatch.setattr(store, "SCHEMA_VERSION", 3)
    with Catalog(path) as migrated:
        assert migrated.query("PRAGMA user_version")[0][0] == 3
        assert migrated.subject_record(uid) == record
        assert [
            tuple(r) for r in migrated.query("SELECT key, value FROM meta ORDER BY key")
        ] == meta
        assert row(migrated, "SELECT note FROM datasets") == {"note": None}
        assert count(migrated, "extra") == 0
    with Catalog(path) as again:  # the second opening has nothing to apply
        assert again.query("PRAGMA user_version")[0][0] == 3
        assert again.subject_record(uid) == record


def test_a_newer_schema_is_refused(local: Catalog, monkeypatch: pytest.MonkeyPatch) -> None:
    path = run_dir_of(local) / CATALOG_FILE
    local.close()
    with monkeypatch.context() as patch:
        patch.setattr(store, "MIGRATIONS", (*store.MIGRATIONS, MIGRATION_3))
        patch.setattr(store, "SCHEMA_VERSION", 3)
        Catalog(path).close()
    before = path.read_bytes()
    openers = (
        lambda: Catalog(path),
        lambda: Catalog(path, readonly=True),
        lambda: PublicCatalog.open(path.parent),
        lambda: PublicCatalog.open(path.parent, readonly=False),
    )
    for opener in openers:
        with pytest.raises(ConfigError, match=r"has schema 3; this version .* reads up to 2") as c:
            opener()
        assert c.value.reason == "newer schema"
    assert path.read_bytes() == before


def test_a_read_only_opening_applies_no_migration_and_refuses_an_older_schema(
    local: Catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    uid = add_subject(local, SALT, make_dataset())
    path = run_dir_of(local) / CATALOG_FILE
    local.close()
    before = path.read_bytes()
    # a migration that is not additive raises the oldest schema a reader takes as it is (a
    # catalog of schema 1 is read as one without functional data: tests/test_store_func.py)
    monkeypatch.setattr(store, "MIGRATIONS", (*store.MIGRATIONS, MIGRATION_3))
    monkeypatch.setattr(store, "SCHEMA_VERSION", 3)
    monkeypatch.setattr(store, "READONLY_OLDEST", 3)

    for opener in (lambda: PublicCatalog.open(path.parent), lambda: Catalog(path, readonly=True)):
        with pytest.raises(ConfigError) as caught:
            opener()
        assert caught.value.reason == "invalid"
        assert caught.value.detail == (
            "catalog schema 2 is older than this version; run any phase command once"
        )
    assert path.read_bytes() == before
    assert sorted(p.name for p in path.parent.iterdir()) == [CATALOG_FILE]

    with PublicCatalog.open(path.parent, readonly=False) as public:  # a phase command's opening
        assert public.subject_uids() == [uid]
    with PublicCatalog.open(path.parent) as public:
        assert public.subject_uids() == [uid]
        assert public._db.rows("PRAGMA user_version")[0][0] == 3


def test_a_failing_migration_changes_nothing(
    local: Catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_subject(local, SALT, make_dataset())
    path = run_dir_of(local) / CATALOG_FILE
    local.close()
    broken = "CREATE TABLE extra (x TEXT);\nALTER TABLE datasets ADD COLUMN note TEXT;\nNONSENSE;"
    with monkeypatch.context() as patch:
        patch.setattr(store, "MIGRATIONS", (*store.MIGRATIONS, broken))
        patch.setattr(store, "SCHEMA_VERSION", 3)
        with pytest.raises(sqlite3.OperationalError):
            Catalog(path)
    with Catalog(path) as intact:
        assert intact.query("PRAGMA user_version")[0][0] == 2
        assert not intact.query("SELECT 1 FROM sqlite_master WHERE name = 'extra'")
        assert "note" not in {r["name"] for r in intact.query("PRAGMA table_info(datasets)")}
        assert count(intact, "subjects") == 1


# --- datasets -----------------------------------------------------------------------------


def later(monkeypatch: pytest.MonkeyPatch) -> None:
    """From now on the catalog's clock reads a later time, so a kept timestamp shows."""
    monkeypatch.setattr(store, "_now", lambda: "2031-01-01T00:00:00+00:00")


def test_upserting_a_dataset_twice_is_one_row(
    local: Catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = make_dataset()
    pseudonym = pseudonym_of(SALT, dataset)
    first = local.upsert_dataset(dataset, pseudonym, 3)
    stored = row(local, "SELECT * FROM datasets")
    assert stored["dataset_id"] == first and stored["pseudonym"] == pseudonym
    assert stored["n_subjects_listed"] == 3 and ISO_UTC.fullmatch(str(stored["discovered_at"]))
    assert local.dataset_record(first) == dataset

    later(monkeypatch)  # a second upsert must not move discovered_at
    assert local.upsert_dataset(dataset, pseudonym, 3) == first
    assert row(local, "SELECT * FROM datasets") == stored
    assert local.pseudonym(first) == pseudonym


def test_an_upsert_takes_the_record_and_keeps_what_the_record_does_not_know(
    census: Catalog,
) -> None:
    dataset = remote_dataset("ds999999")
    pseudonym = pseudonym_of(KEY, dataset)
    dataset_id = census.upsert_dataset(dataset, pseudonym, 12)
    later = dataclasses.replace(
        dataset,
        pipeline_version="25.1.4",
        template_location=None,
        derivative_ref=None,
        snapshot_tag=None,
        layout="fmriprep_legacy",
        fs_root="freesurfer",
    )
    assert census.upsert_dataset(later, pseudonym, None) == dataset_id
    stored = census.dataset_record(dataset_id)
    assert stored == dataclasses.replace(
        later, derivative_ref="refCANARY0123", snapshot_tag="tagCANARY1.0.0"
    )
    assert row(census, "SELECT n_subjects_listed FROM datasets") == {"n_subjects_listed": 12}
    newer = dataclasses.replace(later, derivative_ref="other-ref")
    census.upsert_dataset(newer, pseudonym, 14)
    assert census.dataset_record(dataset_id).derivative_ref == "other-ref"
    assert row(census, "SELECT n_subjects_listed FROM datasets") == {"n_subjects_listed": 14}
    assert count(census, "datasets") == 1


def test_a_dataset_is_one_source_identity_and_pipeline(census: Catalog) -> None:
    one = remote_dataset("ds000001")
    other_pipeline = dataclasses.replace(one, source="openneuro-embedded", pipeline="freesurfer")
    same_accession = dataclasses.replace(other_pipeline, pipeline="fmriprep")
    two = remote_dataset("ds000002")
    ids = [
        census.upsert_dataset(one, pseudonym_of(KEY, one), None),
        census.upsert_dataset(other_pipeline, pseudonym_of(KEY, other_pipeline), None),
        # two rows of one accession and source share their pseudonym
        census.upsert_dataset(same_accession, pseudonym_of(KEY, same_accession), None),
        census.upsert_dataset(two, pseudonym_of(KEY, two), None),
    ]
    assert ids == [1, 2, 3, 4] == census.dataset_ids()
    assert pseudonym_of(KEY, other_pipeline) == pseudonym_of(KEY, same_accession)
    assert len(census.pseudonyms()) == 3
    assert [census.dataset_record(i) for i in ids] == [one, other_pipeline, same_accession, two]


def test_a_pseudonym_never_changes_and_never_names_two_datasets(census: Catalog) -> None:
    one, two = remote_dataset("ds000001"), remote_dataset("ds000002")
    census.upsert_dataset(one, pseudonym_of(KEY, one), None)
    with pytest.raises(ConfigError, match="another key") as caught:
        census.upsert_dataset(one, "D-0123abcd", None)
    assert caught.value.reason == "key"
    with pytest.raises(ConfigError, match="belongs to another dataset") as caught:
        census.upsert_dataset(two, pseudonym_of(KEY, one), None)
    assert caught.value.reason == "invalid"
    for bad in ("ds000001", "D-", "D-XYZ", "d-0123abcd", "D-0123abcd/x", ""):
        with pytest.raises(ValueError, match="not a pseudonym"):
            census.upsert_dataset(two, bad, None)
    elsewhere = remote_dataset("ds000002")
    object.__setattr__(elsewhere, "source", "elsewhere")  # past the type check, to the CHECK
    with pytest.raises(sqlite3.IntegrityError):
        census.upsert_dataset(elsewhere, pseudonym_of(KEY, two), None)
    assert count(census, "datasets") == 1
    assert census.pseudonyms() == [pseudonym_of(KEY, one)]


def test_datasets_are_found_by_accession_or_pseudonym(census: Catalog) -> None:
    records = [remote_dataset(f"ds00000{n}") for n in (3, 1, 2)]
    local_one = make_dataset()
    names = [pseudonym_of(KEY, record) for record in (*records, local_one)]
    for record, name in zip((*records, local_one), names, strict=True):
        census.upsert_dataset(record, name, None)

    assert census.dataset_ids() == [1, 2, 3, 4]
    assert census.pseudonyms() == sorted(names)
    assert census.dataset_ids(["ds000001"]) == [2]
    assert census.dataset_ids([names[2], "ds000003", "ds000003"]) == [1, 3]
    assert census.dataset_ids([names[3]]) == [4]  # a local dataset has a pseudonym only
    assert census.pseudonyms(["ds000002", names[0]]) == sorted([names[2], names[0]])
    assert census.dataset_ids([]) == [] and census.pseudonyms([]) == []
    for unknown in (["ds123456"], ["ds000001", "D-ffffffff"], ["/CANARYROOT/study"]):
        with pytest.raises(ConfigError, match="no dataset named") as caught:
            census.pseudonyms(unknown)
        assert caught.value.reason == "invalid"
        with pytest.raises(ConfigError) as caught:
            census.dataset_ids(unknown)
        assert caught.value.reason == "invalid"
    with pytest.raises(KeyError):
        census.dataset_record(99)
    with pytest.raises(KeyError):
        census.pseudonym(99)

    pseudonyms, roots = census.scrub_terms()
    assert pseudonyms == {r.accession: n for r, n in zip(records, names[:3], strict=True)}
    assert roots == tuple(
        sorted({local_one.root, *(r.root for r in records), *(r.identity for r in records)})
    )
    text = privacy.scrub(
        f"Empty file: '{local_one.root}/sub-01/mri/T1.mgz' of ds000001",
        pseudonyms=pseudonyms,
        roots=roots,
    )
    assert text == f"Empty file: 'T1.mgz' of {names[1]}"


# --- subjects -----------------------------------------------------------------------------


def test_a_subject_record_comes_back_as_the_source_yielded_it(local: Catalog) -> None:
    dataset = make_dataset()
    dataset_id = local.upsert_dataset(dataset, pseudonym_of(SALT, dataset), 3)
    files = make_files()
    files[FileRole.LH_PIAL] = FileRef(
        FileRole.LH_PIAL, "surf/lh.pial.T1", 4_731_523, "1731400000000000000", variant="lh.pial.T1"
    )
    files[FileRole.T1] = dataclasses.replace(
        files[FileRole.T1], last_modified="2026-11-12T08:00:00+00:00"
    )
    shuffled = {role: files[role] for role in reversed(list(files))}
    records = {
        "a" * 16: SubjectRecord(dataset, "sub-01", "sourcedata/freesurfer/sub-01", shuffled),
        "b" * 16: SubjectRecord(dataset, "sub-02_ses-1", "sourcedata/freesurfer/sub-02_ses-1", {}),
        "c" * 16: SubjectRecord(
            dataset, "sub-03", "sourcedata/freesurfer/sub-03", {}, "unreadable: Permission denied"
        ),
    }
    for uid, record in records.items():
        local.upsert_subject(uid, dataset_id, record, UNKNOWN, uid)
    for uid, record in records.items():
        stored = local.subject_record(uid)
        assert stored == record
        assert stored.dataset == dataset
        assert list(stored.files) == [role for role in FileRole if role in record.files]
        assert local.inventory_hash(uid) == uid
    assert local.inventory_hash("d" * 16) is None
    with pytest.raises(KeyError):
        local.subject_record("d" * 16)
    with Catalog(run_dir_of(local) / CATALOG_FILE, readonly=True) as readonly:
        assert readonly.subject_record("a" * 16) == records["a" * 16]


def test_upserting_a_subject_twice_changes_nothing(local: Catalog) -> None:
    dataset = make_dataset()
    uid = add_subject(local, SALT, dataset)
    subject = row(local, "SELECT * FROM subjects")
    files = [dict(r) for r in local.query("SELECT * FROM files ORDER BY role")]
    assert len(files) == len(FileRole)
    assert ISO_UTC.fullmatch(str(subject.pop("inventoried_at")))

    assert add_subject(local, SALT, dataset) == uid
    again = row(local, "SELECT * FROM subjects")
    again.pop("inventoried_at")
    assert again == subject
    assert [dict(r) for r in local.query("SELECT * FROM files ORDER BY role")] == files
    assert count(local, "datasets") == 1


def test_the_columns_of_a_subject_are_what_the_inventory_found(local: Catalog) -> None:
    dataset = make_dataset()
    dataset_id = local.upsert_dataset(dataset, pseudonym_of(SALT, dataset), None)
    known = make_provenance(participant="sub-01", longitudinal=True, pial_variant="pial.T2")
    local.upsert_subject(
        "a" * 16, dataset_id, make_subject(dataset, "sub-01_ses-2.long.base"), known, "hash-a"
    )
    local.upsert_subject("b" * 16, dataset_id, make_subject(dataset, "odd name"), UNKNOWN, "hash-b")

    a = row(local, "SELECT * FROM subjects WHERE subject_uid = ?", "a" * 16)
    assert a | {"inventoried_at": None} == {
        "subject_uid": "a" * 16,
        "dataset_id": dataset_id,
        "subject_label": "sub-01_ses-2.long.base",
        "participant": "sub-01",
        "fs_dir": "sourcedata/freesurfer/sub-01_ses-2.long.base",
        "present": 1,
        "complete": 1,
        "tier_a": 1,
        "tier_b": 1,
        "missing_roles": "",
        "problem": None,
        "longitudinal": 1,
        "fs_build_stamp": known.build_stamp,
        "fs_version": "6.0.1",
        "fs_major": 6,
        "recon_flags": '["-all","-noT2pial","-subjid"]',
        "n_invocations": 4,
        "recon_finished": 1,
        "t2pial": 0,
        "flairpial": 0,
        "hires": 0,
        "expert": None,
        "xopts": 0,
        "edited": 1,
        "pial_variant": "pial.T2",
        "holes_lh": 41,
        "holes_rh": 26,
        "trunc_lh_1": 39,
        "trunc_lh_2": 225,
        "trunc_rh_1": 93,
        "trunc_rh_2": 434,
        "age": None,
        "sex": None,
        "field_strength": None,
        "manufacturer": None,
        "inventory_hash": "hash-a",
        "inventoried_at": None,
    }
    b = row(local, "SELECT * FROM subjects WHERE subject_uid = ?", "b" * 16)
    # nothing known: every provenance column is NULL, and no flag reads as False
    assert {name for name, value in b.items() if value is None} == {
        *("participant", "problem", "fs_build_stamp", "fs_version", "fs_major"),
        *("recon_flags", "n_invocations", "recon_finished", "pial_variant"),
        *("t2pial", "flairpial", "hires", "expert", "xopts", "edited"),
        *("holes_lh", "holes_rh", "trunc_lh_1", "trunc_lh_2", "trunc_rh_1", "trunc_rh_2"),
        *("age", "sex", "field_strength", "manufacturer"),
    }
    assert (b["subject_label"], b["longitudinal"], b["complete"]) == ("odd name", 0, 1)
    no_flags = make_provenance(flags=(), n_invocations=1)
    local.upsert_subject("b" * 16, dataset_id, make_subject(dataset, "odd name"), no_flags, "h")
    assert row(local, "SELECT recon_flags FROM subjects WHERE subject_uid = ?", "b" * 16) == {
        "recon_flags": "[]"
    }
    one_hemi = make_provenance(truncations={"rh": (7, 8)}, holes=None)
    local.upsert_subject("b" * 16, dataset_id, make_subject(dataset, "odd name"), one_hemi, "h")
    assert tuple(
        row(
            local,
            "SELECT trunc_lh_1, trunc_lh_2, trunc_rh_1, trunc_rh_2, holes_lh, holes_rh "
            "FROM subjects WHERE subject_uid = ?",
            "b" * 16,
        ).values()
    ) == (None, None, 7, 8, None, None)


@pytest.mark.parametrize(
    ("absent", "complete", "tier_a", "tier_b", "missing"),
    [
        ((), 1, 1, 1, ""),
        ((FileRole.LH_PIAL,), 0, 1, 1, "lh_pial"),
        ((FileRole.BRAINMASK, FileRole.TALAIRACH, FileRole.RECON_LOG), 1, 1, 1, ""),
        ((FileRole.ASEG_STATS, FileRole.BUILD_STAMP), 1, 1, 1, ""),
        ((FileRole.RH_STATS,), 1, 0, 1, "rh_stats"),
        ((FileRole.LH_ANNOT,), 1, 1, 1, "lh_annot"),  # the annotation is named but not required
        ((FileRole.RH_SPHERE_REG, FileRole.LH_THICKNESS), 1, 1, 0, "lh_thickness,rh_sphere_reg"),
        (
            (FileRole.T1, FileRole.RH_WHITE, FileRole.LH_STATS, FileRole.RH_THICKNESS),
            0,
            0,
            0,
            "t1,rh_white,rh_thickness,lh_stats",
        ),
    ],
)
def test_completeness_and_tiers_follow_the_roles(
    local: Catalog,
    absent: tuple[FileRole, ...],
    complete: int,
    tier_a: int,
    tier_b: int,
    missing: str,
) -> None:
    roles = [role for role in FileRole if role not in absent]
    add_subject(local, SALT, make_dataset(), "sub-01", roles)
    stored = row(local, "SELECT complete, tier_a, tier_b, missing_roles FROM subjects")
    assert tuple(stored.values()) == (complete, tier_a, tier_b, missing)
    # the definitions the flags stand for
    assert complete == all(role in roles for role in RENDER_ROLES)
    assert tier_a == all(role in roles for role in TIER_A_REQUIRED)
    assert tier_b == all(role in roles for role in TIER_B_REQUIRED)


def test_a_subject_without_files_is_incomplete_and_keeps_its_problem(local: Catalog) -> None:
    dataset = make_dataset()
    dataset_id = local.upsert_dataset(dataset, pseudonym_of(SALT, dataset), None)
    record = make_subject(dataset, "sub-01", (), problem="unreadable: Permission denied")
    local.upsert_subject("a" * 16, dataset_id, record, UNKNOWN, "h")
    stored = row(local, "SELECT complete, tier_a, tier_b, missing_roles, problem FROM subjects")
    assert stored == {
        "complete": 0,
        "tier_a": 0,
        "tier_b": 0,
        "missing_roles": (
            "t1,lh_white,rh_white,lh_pial,rh_pial,lh_thickness,rh_thickness,lh_sphere_reg,"
            "rh_sphere_reg,lh_annot,rh_annot,lh_stats,rh_stats"
        ),
        "problem": "unreadable: Permission denied",
    }
    assert count(local, "files") == 0
    # readable at the next inventory: the problem goes
    local.upsert_subject("a" * 16, dataset_id, make_subject(dataset, "sub-01"), UNKNOWN, "h2")
    assert row(local, "SELECT complete, problem FROM subjects") == {"complete": 1, "problem": None}
    assert count(local, "files") == len(FileRole)


def test_a_new_inventory_replaces_the_files_and_keeps_digests_of_unchanged_ones(
    local: Catalog,
) -> None:
    dataset = make_dataset()
    uid = add_subject(local, SALT, dataset)
    public = local.public()
    public.set_file_digests(uid, {FileRole.T1: DIGEST, FileRole.LH_PIAL: "1" * 64})
    with local.write() as conn:
        conn.execute(
            "UPDATE files SET version_id = 'v-CANARY' WHERE role IN ('t1', 'lh_pial', 'rh_pial')"
        )
        conn.execute("UPDATE subjects SET age = 31.5, sex = 'F', field_strength = 7.0")
        conn.execute("UPDATE subjects SET manufacturer = 'Philips', present = 0")

    record = make_subject(dataset, roles=[r for r in FileRole if r != FileRole.BRAINMASK])
    files = dict(record.files)
    files[FileRole.LH_PIAL] = dataclasses.replace(files[FileRole.LH_PIAL], etag_or_mtime="touched")
    record = dataclasses.replace(record, files=files)
    dataset_id = local.dataset_ids()[0]
    local.upsert_subject(uid, dataset_id, record, make_provenance(fs_version="7.3.2"), "h2")

    assert local.subject_record(uid) == record
    stored = {
        r["role"]: (r["sha256"], r["version_id"])
        for r in local.query("SELECT role, sha256, version_id FROM files")
    }
    assert "brainmask" not in stored and len(stored) == len(FileRole) - 1
    assert stored["t1"] == (DIGEST, "v-CANARY")  # unchanged: what was read of it still holds
    assert stored["lh_pial"] == (None, None)  # changed: its digest is of another file
    assert stored["rh_pial"] == (None, "v-CANARY")
    # phase 3 owns these columns; an inventory leaves them and marks the subject present
    kept = row(
        local, "SELECT age, sex, field_strength, manufacturer, present, fs_version FROM subjects"
    )
    assert tuple(kept.values()) == (31.5, "F", 7.0, "Philips", 1, "7.3.2")
    assert local.inventory_hash(uid) == "h2"


def test_a_link_that_leads_to_another_file_of_equal_size_and_time_loses_its_digest(
    local: Catalog,
) -> None:
    # lh.pial -> lh.pial.T1, then -> lh.pial.T2: the location stays surf/lh.pial, and the
    # variants of one subject have one size, so only the variant tells the two apart
    dataset = make_dataset()
    record = make_subject(dataset)
    files = dict(record.files)
    for role, name in ((FileRole.LH_PIAL, "lh.pial.T1"), (FileRole.RH_PIAL, "rh.pial.T1")):
        files[role] = dataclasses.replace(files[role], variant=name)
    record = dataclasses.replace(record, files=files)
    uid = uid_of(SALT, dataset, record.subject_label)
    dataset_id = local.upsert_dataset(dataset, pseudonym_of(SALT, dataset), None)
    local.upsert_subject(uid, dataset_id, record, UNKNOWN, "h")
    public = local.public()
    public.set_file_digests(uid, {FileRole.LH_PIAL: DIGEST, FileRole.RH_PIAL: "1" * 64})
    render_before = public.input_signature(uid, RENDER_ROLES)
    stats_before = public.input_signature(uid, TIER_A_REQUIRED)

    files[FileRole.LH_PIAL] = dataclasses.replace(files[FileRole.LH_PIAL], variant="lh.pial.T2")
    local.upsert_subject(uid, dataset_id, dataclasses.replace(record, files=files), UNKNOWN, "h2")
    # a render of the subject is not skipped either; a stage that reads no pial is
    assert public.input_signature(uid, RENDER_ROLES) != render_before
    assert public.input_signature(uid, TIER_A_REQUIRED) == stats_before
    stored = {
        r["role"]: (r["variant"], r["sha256"])
        for r in local.query("SELECT role, variant, sha256 FROM files WHERE role LIKE '%pial'")
    }
    assert stored == {"lh_pial": ("lh.pial.T2", None), "rh_pial": ("rh.pial.T1", "1" * 64)}


def test_an_inventory_stores_the_digests_of_what_it_read_with_the_subject(
    local: Catalog,
) -> None:
    dataset = make_dataset()
    record = make_subject(dataset, roles=[r for r in FileRole if r != FileRole.RECON_LOG_RH])
    uid = uid_of(SALT, dataset, record.subject_label)
    dataset_id = local.upsert_dataset(dataset, pseudonym_of(SALT, dataset), None)

    def digests() -> dict[str, str]:
        rows = local.query("SELECT role, sha256 FROM files WHERE sha256 IS NOT NULL")
        return {r["role"]: r["sha256"] for r in rows}

    # a role that the subject does not have is ignored, as by set_file_digests
    read = {
        FileRole.BUILD_STAMP: DIGEST,
        FileRole.RECON_LOG: "1" * 64,
        FileRole.RECON_LOG_RH: DIGEST,
    }
    local.upsert_subject(uid, dataset_id, record, UNKNOWN, "h", digests=read)
    assert digests() == {"build_stamp": DIGEST, "recon_log": "1" * 64}
    assert count(local, "files") == len(FileRole) - 1

    # the digest of this reading replaces the one kept; one that was not read now stays
    local.public().set_file_digests(uid, {FileRole.T1: "2" * 64})
    local.upsert_subject(
        uid, dataset_id, record, UNKNOWN, "h", digests={FileRole.RECON_LOG: DIGEST}
    )
    assert digests() == {"build_stamp": DIGEST, "recon_log": DIGEST, "t1": "2" * 64}

    # a changed file that was read gets the new digest, one that was not read gets none
    changed = make_subject(dataset, roles=list(record.files), stamp="2")
    local.upsert_subject(
        uid, dataset_id, changed, UNKNOWN, "h", digests={FileRole.RECON_LOG: "3" * 64}
    )
    assert digests() == {"recon_log": "3" * 64}

    # nothing is written with a digest that is none
    before = [tuple(r) for r in local.query("SELECT * FROM subjects")]
    for bad in ("", "abc", DIGEST.upper(), DIGEST + "0"):
        with pytest.raises(ValueError, match="not a SHA-256 digest"):
            local.upsert_subject(
                uid, dataset_id, record, UNKNOWN, "h3", digests={FileRole.BUILD_STAMP: bad}
            )
    assert [tuple(r) for r in local.query("SELECT * FROM subjects")] == before
    assert digests() == {"recon_log": "3" * 64}


def test_two_subjects_cannot_share_a_uid_and_a_subject_cannot_change_its_uid(
    local: Catalog,
) -> None:
    dataset, other = make_dataset(), make_dataset(identity="/elsewhere", root="/elsewhere")
    uid = add_subject(local, SALT, dataset, "sub-01")
    other_id = local.upsert_dataset(other, pseudonym_of(SALT, other), None)
    before = [tuple(r) for r in local.query("SELECT * FROM subjects")]

    with pytest.raises(ConfigError, match="two subjects with one identifier") as caught:
        local.upsert_subject(uid, 1, make_subject(dataset, "sub-02"), UNKNOWN, "h")
    assert caught.value.reason == "invalid"
    with pytest.raises(ConfigError) as caught:  # the same label in another dataset
        local.upsert_subject(uid, other_id, make_subject(other, "sub-01"), UNKNOWN, "h")
    assert caught.value.reason == "invalid"
    with pytest.raises(ConfigError, match="another key") as caught:
        local.upsert_subject("f" * 16, 1, make_subject(dataset, "sub-01"), UNKNOWN, "h")
    assert caught.value.reason == "key"
    with pytest.raises(KeyError):
        local.upsert_subject("e" * 16, 99, make_subject(dataset, "sub-09"), UNKNOWN, "h")
    for bad in ("", "abc", "A" * 16, "a" * 15, "a" * 17, "../" + "a" * 13):
        with pytest.raises(ValueError, match="16 hex"):
            local.upsert_subject(bad, 1, make_subject(dataset, "sub-09"), UNKNOWN, "h")
    assert [tuple(r) for r in local.query("SELECT * FROM subjects")] == before
    assert count(local, "files") == len(FileRole)


def test_mark_absent_follows_the_listing(local: Catalog) -> None:
    dataset, other = make_dataset(), make_dataset(identity="/elsewhere", root="/elsewhere")
    uids = {label: add_subject(local, SALT, dataset, label) for label in ("sub-01", "sub-02")}
    uids["sub-03"] = add_subject(local, SALT, dataset, "sub-03")
    elsewhere = add_subject(local, SALT, other, "sub-01")
    public = local.public()
    public.record_renders([RenderRow(uids["sub-02"], "c" * 12, "s" * 16, False, "done")])

    assert local.mark_absent(1, ["sub-01", "sub-02", "sub-03", "sub-99"]) == []
    gone = local.mark_absent(1, ["sub-01"])
    assert gone == sorted([uids["sub-02"], uids["sub-03"]])
    assert local.mark_absent(1, ["sub-01"]) == []  # a second run changes and reports nothing
    assert public.subject_uids() == sorted([uids["sub-01"], elsewhere])  # by pseudonym, then uid
    assert set(public.subject_uids(present=False)) == set(gone)
    assert len(public.subject_uids(present=None)) == 4
    # an absent subject keeps its row, its files and its render
    assert count(local, "subjects") == 4 and count(local, "files") == 4 * len(FileRole)
    assert public.render(uids["sub-02"]) is not None
    assert local.subject_record(uids["sub-02"]).subject_label == "sub-02"

    # listed again with unchanged files: present again, without a new upsert
    assert local.mark_absent(1, ["sub-01", "sub-02"]) == []
    assert set(public.subject_uids(present=False)) == {uids["sub-03"]}
    add_subject(local, SALT, dataset, "sub-03")  # or through its next upsert
    assert public.subject_uids(present=False) == []
    assert local.mark_absent(1, []) == sorted(uids.values())
    assert public.subject_uids() == [elsewhere]
    assert local.mark_absent(99, ["sub-01"]) == []


def test_relocate_keeps_identity_and_every_uid(local: Catalog) -> None:
    dataset = make_dataset()
    uids = sorted(add_subject(local, SALT, dataset, label) for label in ("sub-01", "sub-02"))
    public = local.public()
    public.record_renders([RenderRow(uids[0], "c" * 12, "s" * 16, False, "done", render_key="k")])
    signature = public.input_signature(uids[0], RENDER_ROLES)
    pseudonym = local.pseudonyms()

    # the source at the new place, built without the stored identity: its own is the new root
    moved = make_dataset(
        identity="/mnt/new/place",
        root="/mnt/new/place",
        fs_root="freesurfer",
        layout="fmriprep_legacy",
        template_location=None,
        pipeline_version="21.0.2",
    )
    local.relocate(1, moved)
    stored = local.dataset_record(1)
    assert stored == dataclasses.replace(moved, identity=dataset.identity)
    assert local.pseudonyms() == pseudonym
    assert public.subject_uids() == uids
    assert count(local, "datasets") == 1
    labels = sorted(["sub-01", "sub-02"], key=lambda name: uid_of(SALT, dataset, name))
    for uid, label in zip(uids, labels, strict=True):
        record = local.subject_record(uid)
        assert record.dataset.root == "/mnt/new/place" and record.subject_label == label
        # where the source at the new place lists the subject, so its files are found there
        assert record.fs_dir == f"freesurfer/{label}"
        # what the next inventory computes from the stored record is the uid already stored
        assert uid_of(SALT, record.dataset, label) == uid
    render = public.render(uids[0])
    assert render is not None and render.render_key == "k"
    assert public.input_signature(uids[0], RENDER_ROLES) == signature
    # and the next inventory upserts into the same row
    assert local.upsert_dataset(stored, pseudonym[0], 2) == 1

    for wrong in (
        dataclasses.replace(moved, pipeline="freesurfer", layout="subjects_dir"),
        remote_dataset("ds000001"),
    ):
        with pytest.raises(ConfigError, match="other identifiers") as caught:
            local.relocate(1, wrong)
        assert caught.value.reason == "invalid"
    with pytest.raises(KeyError):
        local.relocate(7, moved)
    assert local.dataset_record(1) == stored


def test_relocate_moves_fs_dir_with_fs_root_also_of_absent_subjects(local: Catalog) -> None:
    # a SUBJECTS_DIR that is later given as the derivatives root one level up
    dataset = make_dataset(
        fs_root=".", layout="subjects_dir", pipeline="freesurfer", template_location=None
    )
    uids = {label: add_subject(local, SALT, dataset, label) for label in ("sub-01", "sub-02")}
    assert local.subject_record(uids["sub-01"]).fs_dir == "sub-01"
    assert local.mark_absent(1, ["sub-01"]) == [uids["sub-02"]]
    hashes = {uid: local.inventory_hash(uid) for uid in uids.values()}

    up = dataclasses.replace(dataset, root="/CANARYROOT", fs_root="study")
    local.relocate(1, up)
    for label, uid in uids.items():
        assert local.subject_record(uid).fs_dir == f"study/{label}"
    assert {uid: local.inventory_hash(uid) for uid in uids.values()} == hashes
    assert local.public().subject_uids(present=False) == [uids["sub-02"]]

    # the same fs_root at another root leaves the subjects alone
    local.relocate(1, dataclasses.replace(up, root="/mnt/elsewhere"))
    assert local.subject_record(uids["sub-01"]).fs_dir == "study/sub-01"
    local.relocate(1, dataclasses.replace(up, fs_root="."))
    assert local.subject_record(uids["sub-02"]).fs_dir == "sub-02"


def test_dataset_files_and_templates_are_recorded(
    census: Catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = remote_dataset("ds999999")
    dataset_id = census.upsert_dataset(dataset, pseudonym_of(KEY, dataset), None)
    for size in (58, 61):
        census.record_dataset_file(
            dataset_id,
            "dataset_description",
            "openneuro-derivatives",
            "fmriprep/ds999999-fmriprep/dataset_description.json",
            size=size,
            etag="e" * 32,
            version_id=None,
            sha256=DIGEST,
        )
    census.record_dataset_file(
        dataset_id,
        "participants",
        None,
        "participants.tsv",
        size=1,
        etag="1",
        version_id="v",
        sha256=None,
    )
    rows = [dict(r) for r in census.query("SELECT * FROM dataset_files ORDER BY role")]
    assert [(r["role"], r["size"], r["bucket"]) for r in rows] == [
        ("dataset_description", 61, "openneuro-derivatives"),
        ("participants", 1, None),
    ]
    assert all(ISO_UTC.fullmatch(r["fetched_at"]) for r in rows)
    with pytest.raises(sqlite3.IntegrityError):
        census.record_dataset_file(
            99, "participants", None, "p.tsv", size=1, etag="1", version_id=None, sha256=None
        )

    files = {"surf/lh.sphere.reg": {"size": 5_900_000, "sha256": DIGEST}}
    census.upsert_template("abc123abc123", "fsaverage6", "adjacent", dataset_id, "stamp", files)
    first = row(census, "SELECT * FROM templates")
    assert json.loads(str(first["files_json"])) == files
    assert (
        first["files_json"] == '{"surf/lh.sphere.reg":{"sha256":"' + DIGEST + '","size":5900000}}'
    )
    later(monkeypatch)
    census.upsert_template("abc123abc123", "fsaverage6", "explicit", None, None, files)
    second = row(census, "SELECT * FROM templates")
    assert second == first | {"origin": "explicit", "dataset_id": None, "build_stamp": None}


def test_write_is_one_transaction_and_query_reads(local: Catalog) -> None:
    uid = add_subject(local, SALT, make_dataset())
    with local.write() as conn:
        conn.execute("UPDATE subjects SET field_strength = 3.0 WHERE subject_uid = ?", (uid,))
        assert conn.execute("SELECT field_strength FROM subjects").fetchone()["field_strength"] == 3
    with pytest.raises(RuntimeError, match="boom"), local.write() as conn:
        conn.execute("UPDATE subjects SET field_strength = 7.0")
        conn.execute("DELETE FROM files")
        raise RuntimeError("boom")
    assert row(local, "SELECT field_strength FROM subjects") == {"field_strength": 3.0}
    assert count(local, "files") == len(FileRole)
    with local.write(), pytest.raises(RuntimeError, match="already open"):
        local.set_meta("template_id", "x")
    assert local.meta("template_id") is None
    local.set_meta("template_id", "y")  # the refused nesting left the catalog usable
    assert local.query("SELECT value FROM meta WHERE key = ?", ("template_id",))[0]["value"] == "y"


# --- the public side ----------------------------------------------------------------------


def fill(catalog: Catalog, key: HashKey) -> dict[str, list[str]]:
    """Three bucket datasets and a local one, two subjects each, with ledger rows, renders
    and a selection. The datasets are inserted in descending order of their pseudonyms, so
    that discovery order and pseudonym order differ. Returns pseudonym -> uids."""
    datasets = [remote_dataset(f"ds99999{n}") for n in (9, 8, 7)]
    datasets.append(make_dataset())
    datasets.sort(key=lambda d: pseudonym_of(key, d), reverse=True)
    public = catalog.public()
    uids: dict[str, list[str]] = {}
    for dataset in datasets:
        pseudonym = pseudonym_of(key, dataset)
        uids[pseudonym] = sorted(
            add_subject(catalog, key, dataset, label, stamp=pseudonym)
            for label in ("sub-CANARY7", "sub-CANARY8")
        )
        done, failed = uids[pseudonym]
        public.set_file_digests(done, {FileRole.T1: DIGEST})
        public.record_work(
            [
                WorkRow("regional", done, "c" * 12, "sigCANARY0000001", "done", output="out.npz"),
                WorkRow(
                    "regional",
                    failed,
                    "c" * 12,
                    "sigCANARY0000002",
                    "failed",
                    reason="unreadable",
                    detail="DETAILCANARY: Empty file: 'T1.mgz'",
                ),
                WorkRow("regional", f"dataset:{pseudonym}", "c" * 12, "sigCANARY0000003", "done"),
            ]
        )
        public.record_renders(
            [
                RenderRow(
                    done,
                    "r" * 12,
                    "sigCANARY0000004",
                    False,
                    "done",
                    tile_hash="t" * 12,
                    render_key="k" * 16,
                    media_path=f"media/{done}",
                    n_frames=120,
                ),
                RenderRow(
                    failed,
                    "r" * 12,
                    "sigCANARY0000005",
                    True,
                    "failed",
                    reason="missing surface",
                    detail="DETAILCANARY: lh_pial",
                ),
            ]
        )
    everyone = [uid for group in uids.values() for uid in group]
    public.replace_selection(
        pd.DataFrame(
            {
                "subject_uid": everyone,
                "reason": "all",
                "stratum": "all",
                "inclusion_prob": 1.0,
                "seed": 7,
            }
        ),
        ["all"],
    )
    return uids


def public_frames(public: PublicCatalog) -> dict[str, pd.DataFrame]:
    return {
        "datasets": public.datasets(),
        "subjects": public.subjects(),
        "work": public.work("regional"),
        "renders": public.renders(),
        "selection": public.selection(),
    }


def private_values(catalog: Catalog) -> set[str]:
    """Every private value of the filled catalog, as text."""
    values = {"ds99999", "CANARYROOT", "refCANARY", "tagCANARY", "s3://", "sourcedata"}
    values |= {"CANARYETAG", "sigCANARY", DIGEST, catalog.run_secret().hex()}
    values |= {r["inventory_hash"] for r in catalog.query("SELECT inventory_hash FROM subjects")}
    values |= {r["location"] for r in catalog.query("SELECT location FROM files")}
    return values


@pytest.mark.parametrize("mode", ["local", "census"])
def test_nothing_private_leaves_through_the_public_side(
    mode: str, local: Catalog, census: Catalog
) -> None:
    catalog, key = (local, SALT) if mode == "local" else (census, KEY)
    fill(catalog, key)
    forbidden = private_values(catalog)
    allowed_columns = {"renders": {"detail"}} if mode == "local" else {}
    if mode == "census":
        forbidden |= {"CANARY7", "CANARY8", "DETAILCANARY"}

    def check(name: str, columns: Sequence[str], text: str) -> None:
        private = (PRIVATE_COLUMNS & set(columns)) - allowed_columns.get(name, set())
        assert not private, f"{name} has private columns {private}"
        found = sorted(value for value in forbidden if value in text)
        assert not found, f"{name} holds {found}"

    for name in sorted(VIEWS):
        rows = catalog.query(f"SELECT * FROM {name}")
        assert rows, name
        check(name, rows[0].keys(), repr([tuple(r) for r in rows]))
    with PublicCatalog.open(run_dir_of(catalog)) as public:
        frames = public_frames(public)
        for name, frame in frames.items():
            assert len(frame) >= 4, name
            check(name, list(frame.columns), frame.to_csv() + frame.to_json())
        for key_name in store.PUBLIC_META:
            assert public.meta(key_name) not in forbidden
    # the check can fail: the private side does hold these values
    everything = repr(
        [tuple(r) for t in sorted(TABLES) for r in catalog.query(f"SELECT * FROM {t}")]
    )
    assert all(value in everything for value in forbidden)
    if mode == "local":
        assert "sub-CANARY7" in frames["subjects"].to_csv()
        assert "DETAILCANARY" in frames["renders"].to_csv()


def test_the_frames_have_the_columns_of_the_views(census: Catalog) -> None:
    def view(name: str) -> list[str]:
        return [r["name"] for r in census.query(f"PRAGMA table_info({name})")]

    public = census.public()
    assert list(public.datasets().columns) == view("public_datasets")
    assert list(public.subjects().columns) == view("public_subjects")
    assert view("public_files") == ["subject_uid", "role", "variant"]
    assert list(public.work("regional").columns) == [
        "unit",
        "status",
        "reason",
        "output",
        "finished_at",
    ]
    assert list(public.renders().columns) == [
        "subject_uid",
        "config_hash",
        "tile_hash",
        "render_key",
        "video",
        "status",
        "reason",
        "media_path",
        "n_frames",
        "template_id",
    ]
    assert list(public.selection().columns) == [
        "subject_uid",
        "reason",
        "stratum",
        "inclusion_prob",
        "seed",
    ]


def test_subject_label_and_detail_are_withheld_in_a_census_run(
    local: Catalog, census: Catalog
) -> None:
    filled = {"local": fill(local, SALT), "census": fill(census, KEY)}
    for mode, catalog in (("local", local), ("census", census)):
        public = catalog.public()
        assert public.mode() == mode
        subjects, renders = public.subjects(), public.renders()
        failed = next(group[1] for group in filled[mode].values())
        row_of_failed = public.render(failed)
        assert row_of_failed is not None and row_of_failed.status == "failed"
        assert row_of_failed.input_signature == "sigCANARY0000005"  # for the skip test only
        view = catalog.query("SELECT subject_label, participant FROM public_subjects")
        if mode == "local":
            assert set(subjects["subject_label"]) == {"sub-CANARY7", "sub-CANARY8"}
            assert set(subjects["participant"]) == {"sub-CANARY7", "sub-CANARY8"}
            assert set(renders["detail"].dropna()) == {"DETAILCANARY: lh_pial"}
            assert row_of_failed.detail == "DETAILCANARY: lh_pial"
        else:
            assert subjects["subject_label"].isna().all() and subjects["participant"].isna().all()
            assert all(r["subject_label"] is None and r["participant"] is None for r in view)
            assert "detail" not in renders.columns
            assert row_of_failed.detail is None
        # the private side has both in either mode
        assert catalog.subject_record(failed).subject_label.startswith("sub-CANARY")
        assert set(catalog.failures("render")["detail"]) == {"DETAILCANARY: lh_pial"}


def test_order_is_by_pseudonym_and_uid_never_by_discovery(census: Catalog) -> None:
    uids = fill(census, KEY)
    public = census.public()
    pseudonyms = sorted(uids)
    discovery = [
        r["pseudonym"] for r in census.query("SELECT pseudonym FROM datasets ORDER BY dataset_id")
    ]
    assert discovery == pseudonyms[::-1]  # what `fill` set up: the two orders differ

    assert list(public.datasets()["pseudonym"]) == pseudonyms
    expected = [uid for pseudonym in pseudonyms for uid in uids[pseudonym]]
    assert public.subject_uids() == expected
    assert expected != sorted(expected)  # by pseudonym first, so not simply by uid
    by_discovery = [
        r["subject_uid"]
        for r in census.query("SELECT subject_uid FROM subjects ORDER BY dataset_id, subject_uid")
    ]
    assert public.subject_uids() != by_discovery
    assert list(public.subjects()["subject_uid"]) == sorted(expected)
    assert list(public.renders()["subject_uid"]) == sorted(expected)
    assert list(public.selection()["subject_uid"]) == sorted(expected)
    assert list(public.work("regional")["unit"]) == sorted(
        [*expected, *(f"dataset:{pseudonym}" for pseudonym in pseudonyms)]
    )
    for uid in expected:
        assert uid in uids[public.dataset_of(uid)]
    assert (
        list(public.subjects().sort_values(["dataset", "subject_uid"])["subject_uid"]) == expected
    )


def test_subject_uids_filters(local: Catalog) -> None:
    dataset, other = make_dataset(), make_dataset(identity="/elsewhere", root="/elsewhere")
    name, other_name = pseudonym_of(SALT, dataset), pseudonym_of(SALT, other)
    full = add_subject(local, SALT, dataset, "sub-01")
    no_pial = add_subject(
        local, SALT, dataset, "sub-02", [r for r in FileRole if r != FileRole.LH_PIAL]
    )
    no_stats = add_subject(
        local, SALT, dataset, "sub-03", [r for r in FileRole if r != FileRole.LH_STATS]
    )
    no_sphere = add_subject(
        local, SALT, other, "sub-01", [r for r in FileRole if r != FileRole.RH_SPHERE_REG]
    )
    local.mark_absent(1, ["sub-01", "sub-02"])
    public = local.public()

    def ordered(*uids: str) -> list[str]:
        owner = {full: name, no_pial: name, no_stats: name, no_sphere: other_name}
        return sorted(uids, key=lambda uid: (owner[uid], uid))

    assert public.subject_uids() == ordered(full, no_pial, no_sphere)
    assert public.subject_uids(present=None) == ordered(full, no_pial, no_stats, no_sphere)
    assert public.subject_uids(present=False) == [no_stats]
    assert public.subject_uids(complete=True) == ordered(full, no_sphere)
    assert public.subject_uids(complete=False) == [no_pial]
    assert public.subject_uids(present=None, tier_a=False) == [no_stats]
    assert public.subject_uids(tier_a=True, tier_b=True) == ordered(full, no_pial)
    assert public.subject_uids(tier_b=False) == [no_sphere]
    assert public.subject_uids(datasets=[name]) == ordered(full, no_pial)
    assert public.subject_uids(datasets=[other_name, "D-00000000"], complete=True) == [no_sphere]
    assert public.subject_uids(datasets=[]) == []
    assert public.subject_uids(datasets=[name], complete=True, tier_b=False) == []
    assert public.dataset_of(no_sphere) == other_name
    with pytest.raises(KeyError):
        public.dataset_of("0" * 16)


def test_public_meta_answers_for_five_keys_only(local: Catalog) -> None:
    local.set_meta("template_id", "abc123abc123")
    with PublicCatalog.open(run_dir_of(local)) as public:
        assert public.run_id() == local.meta("run_id")
        assert public.mode() == "local"
        assert {key: public.meta(key) for key in store.PUBLIC_META} == {
            "run_id": local.meta("run_id"),
            "mode": "local",
            "key_fingerprint": SALT.fingerprint,
            "surfcensus_version": surfcensus.__version__,
            "template_id": "abc123abc123",
        }
        for private in ("run_secret", "key_kind", "created_at", "anything"):
            with pytest.raises(KeyError, match="not public"):
                public.meta(private)


def test_roles_and_input_signature_come_from_the_stored_files(local: Catalog) -> None:
    dataset = make_dataset()
    dataset_id = local.upsert_dataset(dataset, pseudonym_of(SALT, dataset), None)
    roles = [r for r in FileRole if r not in (FileRole.BRAINMASK, FileRole.RH_PIAL)]
    record = make_subject(dataset, "sub-01", roles)
    files = dict(record.files)
    files[FileRole.LH_PIAL] = FileRef(FileRole.LH_PIAL, "surf/lh.pial.T1", 5, "9", "lh.pial.T1")
    record = dataclasses.replace(record, files=files)
    local.upsert_subject("a" * 16, dataset_id, record, UNKNOWN, "h")
    local.upsert_subject("b" * 16, dataset_id, make_subject(dataset, "sub-02", ()), UNKNOWN, "h")
    public = local.public()

    found = public.roles("a" * 16)
    assert list(found) == roles  # FileRole order
    assert found[FileRole.LH_PIAL] == "lh.pial.T1"
    assert all(variant is None for role, variant in found.items() if role != FileRole.LH_PIAL)
    assert public.roles("b" * 16) == {}
    with pytest.raises(KeyError):
        public.roles("c" * 16)

    for wanted, extra in (
        (tuple(FileRole), ()),
        (RENDER_ROLES, ()),
        ((FileRole.RH_PIAL, FileRole.BRAINMASK), ()),  # roles the subject lacks
        ((FileRole.T1, FileRole.LH_PIAL), ("template", "abc")),
        ((), ()),
    ):
        signature = public.input_signature("a" * 16, wanted, extra)
        assert signature == base.input_signature(record.files, wanted, extra)
        assert re.fullmatch(r"[0-9a-f]{16}", signature)
    assert public.input_signature("b" * 16, RENDER_ROLES) == base.input_signature({}, RENDER_ROLES)
    with pytest.raises(KeyError):
        public.input_signature("c" * 16, RENDER_ROLES)

    # a touched file changes the signature of the stages that read it, and of no other
    render_before = public.input_signature("a" * 16, RENDER_ROLES)
    stats_before = public.input_signature("a" * 16, TIER_A_REQUIRED)
    files[FileRole.T1] = dataclasses.replace(files[FileRole.T1], etag_or_mtime="later")
    local.upsert_subject(
        "a" * 16, dataset_id, dataclasses.replace(record, files=files), UNKNOWN, "h"
    )
    assert public.input_signature("a" * 16, RENDER_ROLES) != render_before
    assert public.input_signature("a" * 16, TIER_A_REQUIRED) == stats_before


def test_file_digests_are_stored_and_never_read_back(local: Catalog) -> None:
    uid = add_subject(local, SALT, make_dataset(), roles=RENDER_ROLES)
    public = local.public()
    public.set_file_digests(uid, {FileRole.T1: DIGEST, FileRole.BRAINMASK: "2" * 64})
    stored = {r["role"]: r["sha256"] for r in local.query("SELECT role, sha256 FROM files")}
    assert stored == {
        role.value: (DIGEST if role == FileRole.T1 else None) for role in RENDER_ROLES
    }
    public.set_file_digests(uid, {})
    public.set_file_digests("0" * 16, {FileRole.T1: DIGEST})  # an unknown subject: nothing to set
    for bad in ("", "abc", DIGEST.upper(), DIGEST + "0"):
        with pytest.raises(ValueError, match="not a SHA-256 digest"):
            public.set_file_digests(uid, {FileRole.LH_PIAL: "3" * 64, FileRole.T1: bad})
    assert {r["sha256"] for r in local.query("SELECT sha256 FROM files")} == {DIGEST, None}
    assert DIGEST not in "".join(frame.to_csv() for frame in public_frames(public).values())


def test_the_work_ledger(local: Catalog) -> None:
    public = local.public()
    assert public.work_done("regional", "u1", "c1", "s1") is None
    assert len(public.work("regional")) == 0
    public.record_work([])
    public.record_work(
        [
            WorkRow("regional", "u1", "c1", "s1", "done", output="tables/fsavg/u1.npz"),
            WorkRow("regional", "u2", "c1", "s2", "failed", reason="unreadable", detail="why"),
            WorkRow("fsaverage", "u1", "c9", "s9", "done"),
            WorkRow("score", "run", "c1", "s1", "done", finished_at="2026-01-02T03:04:05+00:00"),
        ]
    )
    assert public.work_done("regional", "u1", "c1", "s1") is True
    assert public.work_done("regional", "u2", "c1", "s2") is False
    for phase, unit, config, signature in (
        ("regional", "u1", "c2", "s1"),  # another config
        ("regional", "u1", "c1", "s2"),  # another input
        ("regional", "u2", "c1", "s1"),
        ("regional", "u3", "c1", "s1"),  # no row
        ("fsaverage", "u1", "c1", "s1"),  # the row of another phase
        ("native", "u1", "c1", "s1"),
    ):
        assert public.work_done(phase, unit, config, signature) is None

    frame = public.work("regional")
    assert frame.drop(columns="finished_at").to_dict("records") == [
        {"unit": "u1", "status": "done", "reason": None, "output": "tables/fsavg/u1.npz"},
        {"unit": "u2", "status": "failed", "reason": "unreadable", "output": None},
    ]
    assert all(ISO_UTC.fullmatch(stamp) for stamp in frame["finished_at"])
    assert list(public.work("score")["finished_at"]) == ["2026-01-02T03:04:05+00:00"]
    versions = {r["surfcensus_version"] for r in local.query("SELECT surfcensus_version FROM work")}
    assert versions == {surfcensus.__version__}

    # a later outcome replaces the row; the retried unit is done now
    public.record_work([WorkRow("regional", "u2", "c1", "s2", "done", surfcensus_version="9.9")])
    assert public.work_done("regional", "u2", "c1", "s2") is True
    assert row(local, "SELECT reason, detail, surfcensus_version FROM work WHERE unit = 'u2'") == {
        "reason": None,
        "detail": None,
        "surfcensus_version": "9.9",
    }
    assert count(local, "work") == 4


def test_forgetting_ledger_rows_removes_them_for_one_phase_only(local: Catalog) -> None:
    public = local.public()
    public.record_work(
        [
            WorkRow("regional", "u1", "c1", "s1", "done"),
            WorkRow("regional", "u2", "c1", "s2", "failed", reason="unreadable"),
            WorkRow("regional", "u3", "c1", "s3", "done"),
            WorkRow("fsaverage", "u1", "c1", "s1", "done"),
        ]
    )
    public.forget_work("regional", [])
    assert count(local, "work") == 4
    public.forget_work("regional", {"u1", "u2", "u9"})  # u9 has no row
    assert list(public.work("regional")["unit"]) == ["u3"]
    assert public.work_done("regional", "u1", "c1", "s1") is None
    assert public.work_done("fsaverage", "u1", "c1", "s1") is True
    assert local.failures("regional").empty


def test_a_batch_of_ledger_rows_is_recorded_whole_or_not_at_all(local: Catalog) -> None:
    public = local.public()
    public.record_work([WorkRow("regional", "u1", "c1", "s1", "done")])
    bad = WorkRow("regional", "u3", "c1", "s1", "done")
    object.__setattr__(bad, "status", "pending")  # past the type check, to the table's CHECK
    with pytest.raises(sqlite3.IntegrityError):
        public.record_work(
            [
                WorkRow("regional", "u1", "c2", "s1", "failed"),
                WorkRow("regional", "u2", "c1", "s1", "done"),
                bad,
            ]
        )
    assert public.work("regional").to_dict("records")[0]["status"] == "done"
    assert count(local, "work") == 1
    assert public.work_done("regional", "u1", "c1", "s1") is True


def test_renders_are_recorded_and_replaced(local: Catalog) -> None:
    uid = add_subject(local, SALT, make_dataset())
    other = add_subject(local, SALT, make_dataset(), "sub-02")
    public = local.public()
    assert public.render(uid) is None and len(public.renders()) == 0
    failed = RenderRow(
        uid, "r1", "s1", True, "failed", reason="source changed", detail="'T1.mgz' changed"
    )
    public.record_renders([failed])
    stored = public.render(uid)
    assert stored is not None and ISO_UTC.fullmatch(stored.rendered_at or "")
    assert stored == dataclasses.replace(failed, rendered_at=stored.rendered_at)
    assert stored.video is True  # a bool, as RenderRow declares it

    done = RenderRow(
        uid,
        "r1",
        "s2",
        False,
        "done",
        tile_hash="t" * 12,
        render_key="k" * 16,
        media_path=f"media/{uid}",
        n_frames=120,
        template_id="abc123abc123",
        rendered_at="2026-01-02T03:04:05+00:00",
    )
    public.record_renders([done, RenderRow(other, "r1", "s3", False, "done")])
    assert public.render(uid) == done
    assert count(local, "renders") == 2
    frame = public.renders().set_index("subject_uid")
    assert frame.loc[uid].to_dict() == {
        "config_hash": "r1",
        "tile_hash": "t" * 12,
        "render_key": "k" * 16,
        "video": False,
        "status": "done",
        "reason": None,
        "media_path": f"media/{uid}",
        "n_frames": 120,
        "template_id": "abc123abc123",
        "detail": None,
    }
    assert pd.isna(frame.loc[other, "n_frames"])

    with pytest.raises(sqlite3.IntegrityError):  # a render of a subject the catalog lacks
        public.record_renders(
            [
                RenderRow(other, "r2", "s3", False, "done"),
                RenderRow("0" * 16, "r1", "s", False, "done"),
            ]
        )
    started = RenderRow(uid, "r1", "s", False, "done")
    object.__setattr__(started, "status", "started")  # past the type check, to the CHECK
    with pytest.raises(sqlite3.IntegrityError):
        public.record_renders([started])
    assert public.render(uid) == done
    assert (public.render(other) or done).config_hash == "r1"  # the refused batch left no row


def selection_rows(uids: Sequence[str], reason: str, prob: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_uid": list(uids),
            "reason": reason,
            "stratum": f"stratum-{reason}",
            "inclusion_prob": prob,
            "seed": 11,
        }
    )


def test_replace_selection_replaces_the_rows_of_the_named_reasons(local: Catalog) -> None:
    dataset = make_dataset()
    a, b, c = sorted(add_subject(local, SALT, dataset, f"sub-0{n}") for n in (1, 2, 3))
    public = local.public()
    assert len(public.selection()) == 0

    public.replace_selection(selection_rows([c, a, b], "all"), ["all"])
    assert public.selection().to_dict("records") == [
        {
            "subject_uid": uid,
            "reason": "all",
            "stratum": "stratum-all",
            "inclusion_prob": 1.0,
            "seed": 11,
        }
        for uid in (a, b, c)
    ]
    both = pd.concat([selection_rows([a, b], "prevalence", 0.25), selection_rows([a], "extreme")])
    # columns in another order, and numpy scalars, as a sampler hands them over
    both = both[["seed", "inclusion_prob", "stratum", "reason", "subject_uid"]].astype(
        {"seed": np.int32, "inclusion_prob": np.float32}
    )
    public.replace_selection(both, ["prevalence", "detectability", "extreme"])
    found = public.selection()
    assert [tuple(r) for r in found[["subject_uid", "reason"]].itertuples(index=False)] == sorted(
        [(a, "all"), (b, "all"), (c, "all"), (a, "prevalence"), (b, "prevalence"), (a, "extreme")]
    )
    # a subject drawn by two samples has two rows, each with its own probability
    assert sorted(found[found["subject_uid"] == a]["inclusion_prob"]) == [0.25, 1.0, 1.0]

    public.replace_selection(selection_rows([b], "prevalence", 0.5), ["prevalence"])
    found = public.selection()
    assert list(found[found["reason"] == "prevalence"]["subject_uid"]) == [b]
    assert len(found) == 5  # "all" and "extreme" untouched
    public.replace_selection(selection_rows([], "all"), ["all", "extreme"])
    assert public.selection().to_dict("records") == [
        {
            "subject_uid": b,
            "reason": "prevalence",
            "stratum": "stratum-prevalence",
            "inclusion_prob": 0.5,
            "seed": 11,
        }
    ]


def test_selected_uids_are_none_until_a_selection_is_made(local: Catalog, census: Catalog) -> None:
    for catalog, key in ((local, SALT), (census, KEY)):
        a, b, c = sorted(add_subject(catalog, key, make_dataset(), f"sub-0{n}") for n in (1, 2, 3))
        public = catalog.public()
        assert public.selected_uids() is None
        public.replace_selection(selection_rows([c, a], "calibration"), ["calibration"])
        # a frozen calibration set names examples in a local run, and is no selection of
        # it; in a census run its subjects are selected, to be rendered and rated first
        assert public.selected_uids() == (None if catalog is local else [a, c])
        public.replace_selection(selection_rows([b, a], "prevalence", 0.5), ["prevalence"])
        assert public.selected_uids() == [a, b, c]  # each once, whatever its reasons
        public.replace_selection(selection_rows([], "calibration"), ["calibration"])
        assert public.selected_uids() == [a, b]
        public.replace_selection(selection_rows([], "prevalence"), ["prevalence"])
        assert public.selected_uids() is None


def test_replace_selection_refuses_rows_that_do_not_fit(local: Catalog) -> None:
    a, b = sorted(add_subject(local, SALT, make_dataset(), f"sub-0{n}") for n in (1, 2))
    public = local.public()
    public.replace_selection(selection_rows([a, b], "all"), ["all"])
    before = public.selection()

    with pytest.raises(ValueError, match="columns"):
        public.replace_selection(selection_rows([a], "all").drop(columns="seed"), ["all"])
    with pytest.raises(ValueError, match="columns"):
        public.replace_selection(selection_rows([a], "all").assign(rank=1), ["all"])
    with pytest.raises(ValueError, match="not being replaced: extreme"):
        public.replace_selection(selection_rows([a], "extreme"), ["all"])
    with pytest.raises(ValueError, match="twice"):
        public.replace_selection(selection_rows([a, a], "all"), ["all"])
    with pytest.raises(sqlite3.IntegrityError):  # a subject the catalog lacks: nothing replaced
        public.replace_selection(selection_rows([a, "0" * 16], "all"), ["all"])
    pd.testing.assert_frame_equal(public.selection(), before)


def test_failures_lists_every_failed_unit_with_its_detail(local: Catalog) -> None:
    dataset = make_dataset()
    dataset_id = local.upsert_dataset(dataset, pseudonym_of(SALT, dataset), None)
    assert len(local.failures()) == 0
    assert list(local.failures().columns) == ["unit", "phase", "reason", "detail"]
    for uid, label, problem in (
        ("a" * 16, "sub-01", None),
        ("b" * 16, "sub-02", "unreadable: Permission denied: x"),
        ("c" * 16, "sub-03", "unreadable"),
        ("d" * 16, "sub-04", "unreadable: gone"),
    ):
        record = make_subject(dataset, label, () if problem else tuple(FileRole), problem=problem)
        local.upsert_subject(uid, dataset_id, record, UNKNOWN, "h")
    local.mark_absent(dataset_id, ["sub-01", "sub-02", "sub-03"])
    public = local.public()
    public.record_work(
        [
            WorkRow("regional", "a" * 16, "c", "s", "failed", reason="unreadable", detail="stats"),
            WorkRow("regional", "b" * 16, "c", "s", "done"),
            WorkRow("fsaverage", "a" * 16, "c", "s", "failed", reason="low coverage"),
            WorkRow("score", "run", "c", "s", "failed", reason="unreadable", detail="scores"),
        ]
    )
    public.record_renders(
        [
            RenderRow("b" * 16, "r", "s", False, "failed", reason="missing surface", detail="t1"),
            RenderRow("a" * 16, "r", "s", False, "done"),
        ]
    )
    assert [tuple(r) for r in local.failures().itertuples(index=False)] == [
        ("a" * 16, "fsaverage", "low coverage", None),
        ("b" * 16, "inventory", "unreadable", "Permission denied: x"),
        ("c" * 16, "inventory", "unreadable", None),
        ("a" * 16, "regional", "unreadable", "stats"),
        ("b" * 16, "render", "missing surface", "t1"),
        ("run", "score", "unreadable", "scores"),
    ]
    assert [tuple(r) for r in local.failures("render").itertuples(index=False)] == [
        ("b" * 16, "render", "missing surface", "t1")
    ]
    assert list(local.failures("inventory")["unit"]) == ["b" * 16, "c" * 16]
    assert len(local.failures("native")) == 0


def test_frames_have_the_same_column_types_empty_and_filled(local: Catalog) -> None:
    public = local.public()
    empty = {
        name: frame.dtypes.astype(str).to_dict() for name, frame in public_frames(public).items()
    }
    assert all(len(frame) == 0 for frame in public_frames(public).values())
    fill(local, SALT)
    filled = public_frames(public)
    assert {name: frame.dtypes.astype(str).to_dict() for name, frame in filled.items()} == empty
    subjects = filled["subjects"].dtypes.astype(str).to_dict()
    assert subjects["subject_uid"] == subjects["problem"] == subjects["recon_flags"] == "object"
    assert subjects["present"] == subjects["complete"] == subjects["longitudinal"] == "bool"
    assert subjects["t2pial"] == subjects["expert"] == subjects["recon_finished"] == "boolean"
    assert subjects["fs_major"] == subjects["holes_lh"] == subjects["trunc_rh_2"] == "Int64"
    assert subjects["age"] == subjects["field_strength"] == "float64"
    assert empty["datasets"]["fs_subject_found"] == "boolean"
    assert empty["renders"]["video"] == "bool" and empty["renders"]["n_frames"] == "Int64"
    assert (
        empty["selection"]["seed"] == "int64" and empty["selection"]["inclusion_prob"] == "float64"
    )
    # an unknown stays unknown: None, not False
    first = filled["subjects"].iloc[0]
    assert first["expert"] is pd.NA and not first["t2pial"] and first["t2pial"] is not pd.NA
    assert first["problem"] is None and first["fs_major"] == 6 and np.isnan(first["age"])
    assert json.loads(first["recon_flags"]) == ["-all", "-noT2pial", "-subjid"]


def test_a_read_only_catalog_reads_and_refuses_to_write(local: Catalog) -> None:
    uid = add_subject(local, SALT, make_dataset())
    with PublicCatalog.open(run_dir_of(local)) as public:
        assert public.subject_uids() == [uid]
        calls = (
            lambda: public.record_work([WorkRow("regional", uid, "c", "s", "done")]),
            lambda: public.forget_work("regional", [uid]),
            lambda: public.record_renders([RenderRow(uid, "c", "s", False, "done")]),
            lambda: public.set_file_digests(uid, {FileRole.T1: DIGEST}),
            lambda: public.replace_selection(selection_rows([uid], "all"), ["all"]),
        )
        for call in calls:
            with pytest.raises(ConfigError, match="read-only") as caught:
                call()
            assert caught.value.reason == "invalid"
        # it sees what the writer commits meanwhile
        local.public().record_work([WorkRow("regional", uid, "c", "s", "done")])
        assert public.work_done("regional", uid, "c", "s") is True
    with Catalog(run_dir_of(local) / CATALOG_FILE, readonly=True) as readonly:
        with pytest.raises(ConfigError, match="read-only"):
            readonly.set_meta("template_id", "x")
        with pytest.raises(ConfigError, match="read-only"), readonly.write():
            pass
        assert readonly.public().subject_uids() == [uid]
    assert count(local, "renders") == 0 and count(local, "selection") == 0


def test_the_public_side_of_a_catalog_shares_its_connection(local: Catalog) -> None:
    uid = add_subject(local, SALT, make_dataset())
    with local.public() as public:
        public.record_work([WorkRow("regional", uid, "c", "s", "done")])
    assert count(local, "work") == 1  # closing the public side left the catalog open
    opened = PublicCatalog.open(run_dir_of(local), readonly=False)
    opened.record_work([WorkRow("regional", "run", "c", "s", "done")])
    opened.close()
    with pytest.raises(sqlite3.ProgrammingError):
        opened.subjects()
    assert count(local, "work") == 2


def test_a_catalog_can_be_used_from_several_threads(local: Catalog) -> None:
    uid = add_subject(local, SALT, make_dataset())
    public = local.public()

    def record(worker: int) -> int:
        for number in range(25):
            public.record_work([WorkRow("regional", f"{worker}-{number}", "c", "s", "done")])
            assert public.subject_uids() == [uid]
        return worker

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sorted(pool.map(record, range(4))) == [0, 1, 2, 3]
    assert count(local, "work") == 100


WRITER = """
import sys, time
from pathlib import Path
from surfcensus.records import WorkRow
from surfcensus.store import CATALOG_FILE, Catalog

run_dir, tag = Path(sys.argv[1]), sys.argv[2]
with Catalog(run_dir / CATALOG_FILE) as catalog:
    public = catalog.public()
    (run_dir / f"ready-{tag}").touch()
    while not (run_dir / "go").exists():
        time.sleep(0.01)
    for number in range(100):
        # a transaction that inserts, as every phase records its units
        public.record_work(
            [WorkRow("regional", f"{tag}-{number:03d}", "c" * 12, "s" * 16, "done")]
        )
        # and one that reads before it writes, slowly enough for the other process to get
        # in between if it could: an update lost to it would show in the count
        with catalog.write() as conn:
            value = conn.execute("SELECT value FROM meta WHERE key = 'counter'").fetchone()[0]
            time.sleep(0.002)
            conn.execute(
                "UPDATE meta SET value = ? WHERE key = 'counter'", (str(int(value) + 1),)
            )
"""


@pytest.mark.parametrize("mode", ["local", "census"])
def test_two_processes_writing_200_transactions_each_lose_nothing(
    mode: str, local: Catalog, census: Catalog
) -> None:
    catalog = local if mode == "local" else census
    run_dir = run_dir_of(catalog)
    catalog.set_meta("counter", "0")
    writers = [
        subprocess.Popen(
            [sys.executable, "-c", WRITER, str(run_dir), tag],
            stderr=subprocess.PIPE,
            text=True,
        )
        for tag in ("a", "b")
    ]
    try:
        deadline = time.monotonic() + 120
        while not all((run_dir / f"ready-{tag}").exists() for tag in ("a", "b")):
            assert all(writer.poll() is None for writer in writers), "a writer ended early"
            assert time.monotonic() < deadline, "the writers did not start"
            time.sleep(0.02)
        (run_dir / "go").touch()  # both hold an open catalog: they now write at the same time
        errors = [writer.communicate(timeout=300)[1] for writer in writers]
    finally:
        for writer in writers:
            writer.kill()
    assert [writer.returncode for writer in writers] == [0, 0], errors
    units = list(catalog.public().work("regional")["unit"])
    assert units == [f"{tag}-{number:03d}" for tag in ("a", "b") for number in range(100)]
    assert catalog.meta("counter") == "200"
    assert catalog.query("PRAGMA integrity_check")[0][0] == "ok"


# --- parquet tables -----------------------------------------------------------------------

SCHEMA = pa.schema(
    [
        pa.field("subject_uid", pa.string(), nullable=False),
        pa.field("hemi", pa.string(), nullable=False),
        pa.field("n", pa.int32(), nullable=False),
        pa.field("holes", pa.int32(), nullable=True),
        pa.field("ratio", pa.float32(), nullable=True),
        pa.field("flag", pa.bool_(), nullable=False),
        pa.field("note", pa.string(), nullable=True),
    ]
)


def table_rows(n_subjects: int = 6) -> pd.DataFrame:
    rows = [
        {
            "subject_uid": f"{subject:016x}",
            "hemi": hemi,
            "n": 100 * subject + (hemi == "rh"),
            "holes": None if subject % 3 == 0 else subject,
            "ratio": np.nan if subject % 4 == 0 else subject / 7,
            "flag": subject % 2 == 0,
            "note": None if hemi == "lh" else f"note {subject}",
        }
        for subject in range(n_subjects)
        for hemi in ("lh", "rh")
    ]
    return pd.DataFrame(rows).astype({"holes": "Int32"})


def test_write_table_is_byte_identical_for_shuffled_rows(tmp_path: Path) -> None:
    frame = table_rows()
    store.write_table(tmp_path / "a.parquet", frame, SCHEMA, sort_by=["subject_uid", "hemi"])
    reference = (tmp_path / "a.parquet").read_bytes()
    rng = np.random.default_rng(0)
    for trial in range(4):
        shuffled = frame.sample(frac=1.0, random_state=rng)  # keeps the shuffled index
        shuffled = shuffled[list(rng.permutation(frame.columns))]
        if trial % 2:
            shuffled = shuffled.set_index(
                pd.Index([f"row{i}" for i in range(len(frame))], name="x")
            )
        path = tmp_path / f"shuffled-{trial}.parquet"
        store.write_table(path, shuffled, SCHEMA, sort_by=["subject_uid", "hemi"])
        assert path.read_bytes() == reference
    # ties in the sort key are broken by the other columns, so they do not show either
    store.write_table(tmp_path / "t1.parquet", frame, SCHEMA, sort_by=["flag"])
    store.write_table(tmp_path / "t2.parquet", frame.iloc[::-1], SCHEMA, sort_by=["flag"])
    assert (tmp_path / "t1.parquet").read_bytes() == (tmp_path / "t2.parquet").read_bytes()
    assert (tmp_path / "t1.parquet").read_bytes() != reference
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        ["a.parquet", "t1.parquet", "t2.parquet", *(f"shuffled-{t}.parquet" for t in range(4))]
    )


def test_a_written_table_has_the_schema_sorted_rows_and_no_pandas_metadata(tmp_path: Path) -> None:
    frame = table_rows().iloc[::-1]
    path = tmp_path / "tables" / "deep" / "a.parquet"  # the directory is made
    store.write_table(path, frame, SCHEMA, sort_by=["subject_uid", "hemi"])
    file = pq.ParquetFile(path)
    assert file.schema_arrow.equals(SCHEMA)
    assert [f.nullable for f in file.schema_arrow] == [f.nullable for f in SCHEMA]
    assert file.schema_arrow.metadata is None  # no pandas metadata, no index description
    assert file.metadata.row_group(0).column(0).compression == "ZSTD"
    table = file.read()
    assert table.column_names == SCHEMA.names
    keys = list(zip(table["subject_uid"].to_pylist(), table["hemi"].to_pylist(), strict=True))
    assert keys == sorted(keys) and len(keys) == 12
    assert table["holes"].null_count == 4 and table["ratio"].null_count == 4  # NaN is missing

    back = store.read_table(path, SCHEMA)
    expected = frame.sort_values(["subject_uid", "hemi"]).reset_index(drop=True)
    assert list(back["subject_uid"]) == list(expected["subject_uid"])
    assert list(back["n"]) == list(expected["n"])
    np.testing.assert_array_equal(back["ratio"], expected["ratio"].astype(np.float32))
    assert back["holes"].tolist() == expected["holes"].tolist()
    assert back["note"].tolist() == [None if pd.isna(v) else v for v in expected["note"]]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda f: f.drop(columns="hemi"), "missing hemi"),
        (lambda f: f.assign(extra=1), "unexpected extra"),
        (
            lambda f: f.drop(columns=["n", "note"]).assign(z=1, y=2),
            "missing n, note; unexpected z, y",
        ),
        (lambda f: f.assign(n="many"), "do not fit the schema"),
        (lambda f: f.assign(n=2**40), "do not fit the schema"),
        (lambda f: f.assign(holes=1.5), "do not fit the schema"),
        (lambda f: f.assign(n=np.nan), "do not fit the schema"),
        (lambda f: f.assign(subject_uid=None), "do not fit the schema"),
        (lambda f: pd.concat([f, f[["hemi"]]], axis=1), "a column name twice"),
    ],
)
def test_write_table_rejects_rows_of_another_schema(tmp_path: Path, change, message: str) -> None:
    path = tmp_path / "a.parquet"
    good = table_rows()
    store.write_table(path, good, SCHEMA, sort_by=["subject_uid", "hemi"])
    before = path.read_bytes()
    with pytest.raises(ValueError, match=message):
        store.write_table(path, change(good), SCHEMA, sort_by=["subject_uid", "hemi"])
    with pytest.raises(ValueError, match="cannot sort by rank"):
        store.write_table(path, good, SCHEMA, sort_by=["subject_uid", "rank"])
    assert path.read_bytes() == before  # the table that was there is intact
    assert [p.name for p in tmp_path.iterdir()] == ["a.parquet"]  # and nothing is left behind


def test_read_table_of_an_absent_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        store.read_table(tmp_path / "none.parquet")
    for name, schema in SCHEMAS.items():
        empty = store.read_table(tmp_path / f"{name}.parquet", schema)
        assert list(empty.columns) == schema.names and len(empty) == 0
    assert list(tmp_path.iterdir()) == []


def test_read_table_gives_the_same_column_types_empty_and_filled(tmp_path: Path) -> None:
    empty = store.read_table(tmp_path / "none.parquet", SCHEMA)
    store.write_table(tmp_path / "a.parquet", table_rows(), SCHEMA, sort_by=["subject_uid"])
    filled = store.read_table(tmp_path / "a.parquet")
    expected = {
        "subject_uid": "object",
        "hemi": "object",
        "n": "int32",
        "holes": "Int32",
        "ratio": "float32",
        "flag": "bool",
        "note": "object",
    }
    assert empty.dtypes.astype(str).to_dict() == expected
    assert filled.dtypes.astype(str).to_dict() == expected
    assert filled["note"].iloc[0] is None and filled["holes"].iloc[0] is pd.NA
    # what was read can be written again, to the same bytes
    store.write_table(tmp_path / "b.parquet", filled, SCHEMA, sort_by=["subject_uid"])
    assert (tmp_path / "b.parquet").read_bytes() == (tmp_path / "a.parquet").read_bytes()

    other = pa.schema([SCHEMA.field(0), SCHEMA.field(1).with_nullable(True), *list(SCHEMA)[2:]])
    with pytest.raises(ValueError, match="another schema"):
        store.read_table(tmp_path / "a.parquet", other)
    with pytest.raises(ValueError, match="another schema"):
        store.read_table(tmp_path / "a.parquet", SCHEMAS["regional"])


def summary_rows(uids: Sequence[str], *, n_native: int = 1000) -> pd.DataFrame:
    """Rows of the ``vertex_summary`` table, two per subject."""
    rows = [
        {
            "subject_uid": uid,
            "hemi": hemi,
            "n_native": n_native + number,
            "thickness_max_mm": 5.0,
            "template_id": None,
            "n_fallback": None,
            "n_missing_cortex": None,
            "global_median_mm": np.nan,
            "region_kind": "annot",
            "ceiling_frac": 0.25,
            "control_ceiling_frac": np.nan,
            "ceiling_frac_annot": 0.5,
            "flags": "",
        }
        for number, uid in enumerate(uids)
        for hemi in ("rh", "lh")
    ]
    frame = pd.DataFrame(
        rows,
        columns=[name for name in SCHEMAS["vertex_summary"].names if rows == [] or name in rows[0]],
    )
    for name in SCHEMAS["vertex_summary"].names:
        if name not in frame.columns:
            frame[name] = pd.array([None] * len(frame), dtype="Int32")
    return frame


def keys_of(path: Path) -> list[tuple[str, str]]:
    frame = store.read_table(path, SCHEMAS["vertex_summary"])
    return list(zip(frame["subject_uid"], frame["hemi"], strict=True))


def test_upsert_part_replaces_exactly_the_named_subjects(tmp_path: Path) -> None:
    a, b, c, d = (f"{letter}" * 16 for letter in "abcd")
    path = store.upsert_part(
        tmp_path, "vertex_summary", "D-0123abcd", summary_rows([c, a, b]), replace_uids=[a, b, c]
    )
    assert path == tmp_path / "tables" / "_parts" / "vertex_summary" / "D-0123abcd.parquet"
    assert path == store.part_path(tmp_path, "vertex_summary", "D-0123abcd")
    assert keys_of(path) == [(uid, hemi) for uid in (a, b, c) for hemi in ("lh", "rh")]  # by key

    # b gets new rows, c is redone and has none, d is new, a is not named and stays
    again = store.upsert_part(
        tmp_path,
        "vertex_summary",
        "D-0123abcd",
        summary_rows([d, b], n_native=7000),
        replace_uids={b, c, d},
    )
    assert again == path
    frame = store.read_table(path, SCHEMAS["vertex_summary"])
    assert keys_of(path) == [(uid, hemi) for uid in (a, b, d) for hemi in ("lh", "rh")]
    by_subject = frame.groupby("subject_uid")["n_native"].min().to_dict()
    assert by_subject == {a: 1001, b: 7001, d: 7000}

    # the same call again changes nothing: not a byte
    before = path.read_bytes()
    store.upsert_part(
        tmp_path,
        "vertex_summary",
        "D-0123abcd",
        summary_rows([d, b], n_native=7000),
        replace_uids=[d, c, b],
    )
    assert path.read_bytes() == before
    # another dataset has its own part
    other = store.upsert_part(
        tmp_path, "vertex_summary", "D-ffff0000", summary_rows([c]), replace_uids=[c]
    )
    assert other != path and path.read_bytes() == before
    # a part can end up empty and is still a file: it is what a skip rule looks for
    store.upsert_part(tmp_path, "vertex_summary", "D-ffff0000", summary_rows([]), replace_uids=[c])
    assert other.is_file() and keys_of(other) == []
    assert sorted(p.name for p in path.parent.iterdir()) == [
        "D-0123abcd.parquet",
        "D-ffff0000.parquet",
    ]


def test_upsert_part_refuses_what_would_corrupt_a_part(tmp_path: Path) -> None:
    a, b = "a" * 16, "b" * 16
    path = store.upsert_part(
        tmp_path, "vertex_summary", "D-0123abcd", summary_rows([a, b]), replace_uids=[a, b]
    )
    before = path.read_bytes()
    with pytest.raises(ValueError, match="more than one row with the key"):  # a is not replaced
        store.upsert_part(
            tmp_path, "vertex_summary", "D-0123abcd", summary_rows([a]), replace_uids=[b]
        )
    with pytest.raises(ValueError, match="more than one row"):
        store.upsert_part(
            tmp_path, "vertex_summary", "D-0123abcd", summary_rows([a, a]), replace_uids=[a]
        )
    with pytest.raises(ValueError, match="missing hemi"):
        store.upsert_part(
            tmp_path,
            "vertex_summary",
            "D-0123abcd",
            summary_rows([a]).drop(columns="hemi"),
            replace_uids=[a],
        )
    for accession in ("ds000102", "D-", "D-XYZ", "../D-0123abcd", "D-0123abcd/..", ""):
        with pytest.raises(ValueError, match="pseudonym"):
            store.upsert_part(
                tmp_path, "vertex_summary", accession, summary_rows([a]), replace_uids=[a]
            )
    with pytest.raises(ValueError, match="no table 'ratings'"):
        store.upsert_part(tmp_path, "ratings", "D-0123abcd", summary_rows([a]), replace_uids=[a])
    with pytest.raises(
        ValueError, match="another schema"
    ):  # a part of another schema is not mixed in
        store.upsert_part(
            tmp_path, "vertex_summary", "D-0123abcd", table_rows(), replace_uids=[a], schema=SCHEMA
        )
    assert path.read_bytes() == before
    assert [p.name for p in (tmp_path / "tables" / "_parts" / "vertex_summary").iterdir()] == [
        path.name
    ]
    assert [p.name for p in (tmp_path / "tables" / "_parts").iterdir()] == ["vertex_summary"]


def test_merge_parts_equals_a_single_write_and_drops_absent_subjects(tmp_path: Path) -> None:
    uids = [f"{number:016x}" for number in (5, 3, 9, 1, 7, 2)]
    everything = summary_rows(uids)
    # three datasets; the parts are filled in several calls, in no particular order
    groups = {"D-00000003": uids[:2], "D-00000001": uids[2:5], "D-00000002": uids[5:]}
    for pseudonym, members in groups.items():
        for uid in members:
            rows = everything[everything["subject_uid"] == uid]
            store.upsert_part(tmp_path, "vertex_summary", pseudonym, rows, replace_uids=[uid])

    merged = store.merge_parts(tmp_path / "", "vertex_summary", present=uids)
    assert merged == tmp_path / "tables" / "vertex_summary.parquet"
    assert merged == store.table_path(tmp_path, "vertex_summary")
    schema, key = SCHEMAS["vertex_summary"], KEYS["vertex_summary"]
    store.write_table(tmp_path / "single.parquet", everything, schema, sort_by=key)
    assert merged.read_bytes() == (tmp_path / "single.parquet").read_bytes()
    assert keys_of(merged) == [(uid, hemi) for uid in sorted(uids) for hemi in ("lh", "rh")]

    # two subjects are no longer present: their rows stay in the parts and leave the table
    present = [uid for uid in uids if uid not in (uids[0], uids[3])] + ["f" * 16]
    store.merge_parts(tmp_path, "vertex_summary", present=present)
    kept = everything[everything["subject_uid"].isin(present)]
    store.write_table(tmp_path / "single.parquet", kept, schema, sort_by=key)
    assert merged.read_bytes() == (tmp_path / "single.parquet").read_bytes()
    assert len(keys_of(merged)) == 8
    assert len(keys_of(store.part_path(tmp_path, "vertex_summary", "D-00000003"))) == 4

    store.merge_parts(tmp_path, "vertex_summary", present=[])
    assert keys_of(merged) == []


def test_merge_parts_without_parts_and_with_a_subject_in_two_parts(tmp_path: Path) -> None:
    path = store.merge_parts(tmp_path, "regional", present=["a" * 16])
    assert path == tmp_path / "tables" / "regional.parquet"
    assert (
        pq.read_table(path).schema.equals(SCHEMAS["regional"]) and pq.read_table(path).num_rows == 0
    )
    with pytest.raises(ValueError, match="no table"):
        store.merge_parts(tmp_path, "nothing", present=[])

    a = "a" * 16
    store.upsert_part(tmp_path, "vertex_summary", "D-00000001", summary_rows([a]), replace_uids=[a])
    store.upsert_part(tmp_path, "vertex_summary", "D-00000002", summary_rows([a]), replace_uids=[a])
    with pytest.raises(ValueError, match="more than one row with the key"):
        store.merge_parts(tmp_path, "vertex_summary", present=[a])
    assert not store.table_path(tmp_path, "vertex_summary").exists()


def test_parts_of_the_intensity_table_take_the_schema_of_the_run(tmp_path: Path) -> None:
    schema = intensity_schema(2)
    assert schema != SCHEMAS["intensity"]
    uid = "a" * 16
    frame = pd.DataFrame(
        {
            "subject_uid": [uid, uid],
            "hemi": ["lh", "rh"],
            "region_kind": ["template", "none"],
            "search_n": pd.array([10, None], dtype="Int32"),
            "control_n": pd.array([20, None], dtype="Int32"),
            "outer_outlier_frac": [0.5, np.nan],
            "gap_frac": [0.0, np.nan],
            "search_median_d0": [70.0, np.nan],
            "search_median_d1": [80.0, np.nan],
            "control_median_d0": [71.0, np.nan],
            "control_median_d1": [81.0, np.nan],
        }
    )
    with pytest.raises(ValueError, match="missing search_median_d2"):  # the default: four depths
        store.upsert_part(tmp_path, "intensity", "D-00000001", frame, replace_uids=[uid])
    store.upsert_part(tmp_path, "intensity", "D-00000001", frame, replace_uids=[uid], schema=schema)
    with pytest.raises(ValueError, match="another schema"):
        store.merge_parts(tmp_path, "intensity", present=[uid])
    merged = store.merge_parts(tmp_path, "intensity", present=[uid], schema=schema)
    back = store.read_table(merged, schema)
    assert back["search_n"].tolist() == [10, pd.NA] and back["region_kind"].tolist() == [
        "template",
        "none",
    ]


# --- the order of the rating queue ------------------------------------------------------------


def _score_frame(rows: Sequence[tuple[str, int | None, str]]) -> pd.DataFrame:
    """``(subject_uid, primary_rank, flags)`` as the columns ``queue_order`` reads."""
    return pd.DataFrame(
        {
            "subject_uid": [uid for uid, _, _ in rows],
            "primary_rank": pd.array([rank for _, rank, _ in rows], dtype="Int32"),
            "flags": [flags for _, _, flags in rows],
        }
    )


def test_the_queue_goes_by_primary_rank_with_the_partial_scores_first() -> None:
    a, b, c, d, e, f = (letter * 16 for letter in "abcdef")
    scores = _score_frame(
        [
            (a, 2, "no_annot"),
            (c, 3, "no_stats,partial_score"),
            (d, 1, ""),
            (e, 4, "partial_score"),
            (f, None, "partial_score"),  # a row that is not primary says nothing
            (d, None, ""),
        ]
    )
    order = store.queue_order([f, e, d, c, b, a], scores, blind=False, seed=1)
    # could not be fully scored (by uid), then worst first, then without a primary row (by uid)
    assert order == [c, e, d, a, b, f]
    assert store.queue_order([a, a, d], scores, blind=False, seed=1) == [d, a]  # each once
    assert store.queue_order([b, a], None, blind=False, seed=1) == [a, b]
    assert store.queue_order([], scores, blind=False, seed=1) == []
    # a flag that only starts like the one that counts does not count
    lookalike = _score_frame([(a, 2, "partial_score_x"), (d, 1, "")])
    assert store.queue_order([a, d], lookalike, blind=False, seed=1) == [d, a]


def test_a_blind_queue_is_a_seeded_shuffle_whatever_the_scores() -> None:
    uids = [letter * 16 for letter in "abcdefgh"]
    scores = _score_frame([(uid, rank, "") for rank, uid in enumerate(reversed(uids), start=1)])
    expected = sorted(uids)
    random.Random(11).shuffle(expected)
    assert expected != sorted(uids) and expected != list(reversed(uids))
    assert store.queue_order(uids[::-1], scores, blind=True, seed=11) == expected
    assert store.queue_order(uids, None, blind=True, seed=11) == expected
    assert store.queue_order(uids, None, blind=True, seed=12) != expected
    assert store.queue_order(uids, scores, blind=False, seed=11) == list(reversed(uids))


def test_the_queue_order_of_the_small_run_is_that_of_its_scores(small_run: Path) -> None:
    scores = store.primary_scores(small_run)
    uids = sorted(scores["subject_uid"])
    assert not scores["flags"].str.contains("partial_score").any()
    assert store.queue_order(uids, scores, blind=False, seed=0) == scores["subject_uid"].tolist()
