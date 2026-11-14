"""Tests of the functional part of surfcensus/store.py (migration 2, work package E2.A of step 2
of the EPI interface; docs/epi_interface.md, "Step 2: the plan"): the migration of a catalog of
schema 1, the read-only rule for it, the methods of ``Catalog`` and ``PublicCatalog`` on the
tables ``func_sources``, ``func_participants``, ``func_sessions``, ``func_runs``, ``func_files``
and ``packs``, the two skip signatures, and the public views against the private columns."""

from __future__ import annotations

import dataclasses
import hashlib
import multiprocessing
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from test_store import (  # pyrefly: ignore[missing-import]
    KEY,
    SALT,
    add_subject,
    make_dataset,
    pseudonym_of,
    run_dir_of,
)

from surfcensus import privacy, store
from surfcensus.config import ConfigError
from surfcensus.records import (
    Anchor,
    ConnectorMatch,
    Corrections,
    Display,
    FrameCheck,
    FrameDisplacement,
    FuncFileRef,
    FuncRunRow,
    FuncSessionRow,
    FuncSourceRecord,
    Grid,
    PackRow,
    RunEntities,
    Statement,
    WorkRow,
    session_unit,
)
from surfcensus.sources.base import FuncRole
from surfcensus.store import CATALOG_FILE, Catalog, PublicCatalog

IDENTITY = "/CANARYROOT/study/derivatives/fmriprep"
DIGEST = hashlib.sha256(b"func canary").hexdigest()
GRID = Grid(
    (4, 5, 6),
    ((2.0, 0.0, 0.0, -4.0), (0.0, 2.0, 0.0, -5.0), (0.0, 0.0, 3.0, -9.0), (0.0, 0.0, 0.0, 1.0)),
)
# the starred columns of the plan, and what D20 keeps in the catalog
PRIVATE = {
    "deriv_root",
    "raw_root",
    "deriv_identity",
    "inventory_hash",
    "anchor_why",
    "entities",
    "location",
    "size",
    "etag_or_mtime",
    "sha256",
    "input_signature",
    "content_digest",
    "func_id",
    "dataset_id",
}


def source_record(**changes: object) -> FuncSourceRecord:
    record = FuncSourceRecord(
        connector="fmriprep",
        match=ConnectorMatch("fmriprep", "21.0.2", "scanner-T1w"),
        deriv_root="/CANARYROOT/study/derivatives/fmriprep",
        deriv_identity=IDENTITY,
        raw_root="/CANARYROOT/study",
        deriv_found="derivatives folder",
        raw_found="dataset root",
    )
    return dataclasses.replace(record, **changes)


def puid(participant: str = "sub-01", key: privacy.HashKey = SALT) -> str:
    return privacy.participant_uid(key, "local", IDENTITY, "fmriprep", participant)


def suid(participant: str = "sub-01", session: str | None = "1") -> str:
    return privacy.session_uid(SALT, "local", IDENTITY, "fmriprep", participant, session)


def ruid(name: str, participant: str = "sub-01", session: str | None = "1") -> str:
    return privacy.run_uid(SALT, "local", IDENTITY, "fmriprep", participant, session, name)


def ref(role: FuncRole, location: str, *, size: int = 100, mtime: str = "1") -> FuncFileRef:
    base = "raw" if role in (FuncRole.RAW_BOLD, FuncRole.RAW_SIDECAR) else "deriv"
    return FuncFileRef(role, base, location, size, mtime)


def run_files(name: str, *, mtime: str = "1") -> tuple[FuncFileRef, ...]:
    return (
        ref(FuncRole.RAW_BOLD, f"sub-01/ses-1/func/sub-01_{name}_bold.nii.gz", mtime=mtime),
        ref(FuncRole.PRE_BOLD, f"sub-01/ses-1/func/{name}_space-T1w_bold.nii.gz", mtime=mtime),
        ref(FuncRole.PRE_MASK, f"sub-01/ses-1/func/{name}_mask.nii.gz", mtime=mtime),
        ref(FuncRole.XFM_TO_ANAT, f"sub-01/ses-1/func/{name}_xfm.txt", mtime=mtime),
        ref(FuncRole.CONFOUNDS, f"sub-01/ses-1/func/{name}_confounds.tsv", mtime=mtime),
    )


CORRECTIONS = Corrections(
    motion=Statement("applied", None, None),
    sdc=Statement("not applied", "None", FuncRole.SUMMARY_REPORT),
    sdc_method=None,
    slice_timing=Statement("unknown", "n/a", FuncRole.SUMMARY_REPORT),
    registration=Statement("applied", "bbregister", FuncRole.SUMMARY_REPORT),
    dof=6,
    sdc_figure_listed=False,
)


def frame(mean: float = 0.0) -> FrameDisplacement:
    return FrameDisplacement(mean, mean * 2, 0.1, 0.2, (0.1, 0.0, 0.0), 0.0, 100, mean > 10)


def run_row(
    name: str,
    order: int,
    *,
    anchor: bool = False,
    packable: bool = True,
    files: tuple[FuncFileRef, ...] | None = None,
    session: str | None = "1",
) -> FuncRunRow:
    return FuncRunRow(
        run_uid=ruid(name, session=session),
        session_uid=suid(session=session),
        run_key=name,
        name=name,
        entities=RunEntities("01", session, task=name.split("_")[0], run=str(order + 1)),
        session_order=order,
        acquisition_order=order,
        anchor=anchor,
        selected=True,
        packable=packable,
        reason=None if packable else "no raw series",
        detail=None if packable else "CANARYDETAIL nothing in raw",
        missing_roles=() if packable else (FuncRole.RAW_BOLD,),
        n_volumes=10,
        tr_s=2.0,
        echoes=1,
        display=Display("anat", "T1w"),
        spaces=("T1w",),
        grid=GRID if packable else None,
        raw_affine="header" if packable else None,
        reference_image=True,
        rigid=True,
        dof=6,
        corrections=CORRECTIONS,
        frame=frame(float(order)) if packable else None,
        files=run_files(name) if files is None else files,
    )


def session_row(
    runs: tuple[FuncRunRow, ...] | None = None,
    *,
    subject_uid: str | None = None,
    inventory_hash: str = "a" * 16,
    session_files: tuple[FuncFileRef, ...] | None = None,
    anchor: str | None = None,
) -> FuncSessionRow:
    runs = (
        (
            run_row("rest_run-1", 0, anchor=True),
            run_row("rest_run-2", 1),
            run_row("task_run-1", 2, packable=False),
        )
        if runs is None
        else runs
    )
    anchor_uid = anchor or ruid("rest_run-1")
    return FuncSessionRow(
        session_uid=suid(),
        participant_uid=puid(),
        session="1",
        stream_order=0,
        subject_uid=subject_uid,
        fs_subject="sub-01" if subject_uid else None,
        anchor=Anchor(anchor_uid, "acquisition_time", "CANARYTIME 10:11:12"),
        anchor_note=None,
        frame=FrameCheck(anchor_uid, 10.0, {}, (), True, False),
        n_grids=1,
        problem=None,
        inventory_hash=inventory_hash,
        files=(
            (ref(FuncRole.XFM_ANAT_FSNATIVE, "sub-01/anat/sub-01_from-T1w_to-fsnative_xfm.txt"),)
            if session_files is None
            else session_files
        ),
        runs=runs,
    )


@pytest.fixture
def local(tmp_path: Path) -> Iterator[Catalog]:
    with Catalog.create(tmp_path / "run" / CATALOG_FILE, mode="local", key=SALT) as catalog:
        yield catalog


def stored(catalog: Catalog, *, subject: bool = True, row: FuncSessionRow | None = None) -> dict:
    """A dataset with one subject, its functional source, one participant and one session."""
    dataset = make_dataset(
        root="/CANARYROOT/study", fs_root="derivatives/fmriprep/sourcedata/freesurfer"
    )
    uid = add_subject(catalog, SALT, dataset, "sub-01") if subject else None
    dataset_id = catalog.upsert_dataset(dataset, pseudonym_of(SALT, dataset), None)
    func_id = catalog.upsert_func_source(dataset_id, source_record())
    catalog.upsert_func_participant(func_id, puid(), "sub-01", 1)
    catalog.upsert_func_session(row or session_row(subject_uid=uid))
    return {"dataset_id": dataset_id, "func_id": func_id, "subject_uid": uid}


def schema_1_catalog(path: Path, monkeypatch: pytest.MonkeyPatch, mode: str = "local") -> str:
    """A catalog made by the code of schema 1 (migration 1 alone), with one subject, a render
    and a ledger row; returns the subject's uid."""
    with monkeypatch.context() as patch:
        patch.setattr(store, "MIGRATIONS", (store._MIGRATION_1,))
        patch.setattr(store, "SCHEMA_VERSION", 1)
        key = SALT if mode == "local" else KEY
        with Catalog.create(path, mode=mode, key=key) as old:  # type: ignore[arg-type]
            uid = add_subject(old, key, make_dataset(), "sub-01")
            public = old.public()
            public.record_work([WorkRow("regional", uid, "c" * 12, "s" * 16, "done")])
            assert old.query("PRAGMA user_version")[0][0] == 1
    return uid


def dump(path: Path) -> dict[str, list[tuple]]:
    """Every row of every table of schema 1, by table."""
    conn = sqlite3.connect(path)
    try:
        names = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        ]
        return {n: sorted(conn.execute(f'SELECT * FROM "{n}"').fetchall()) for n in names}
    finally:
        conn.close()


@pytest.mark.parametrize("mode", ["local", "census"])
def test_a_read_only_opening_reads_schema_1_as_a_run_without_functional_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    path = tmp_path / "run" / CATALOG_FILE
    uid = schema_1_catalog(path, monkeypatch, mode)
    before = path.read_bytes()
    names = sorted(p.name for p in path.parent.iterdir())
    for opener in (
        lambda: PublicCatalog.open(path.parent),
        lambda: Catalog(path, readonly=True).public(),
    ):
        public = opener()
        assert public._db.version == 1
        assert public.subject_uids() == [uid]
        assert public.has_func() is False
        for frame in (
            public.func_sources(),
            public.func_participants(),
            public.func_sessions(),
            public.func_sessions(puid()),
            public.func_runs(),
            public.func_runs(suid()),
            public.packs(),
        ):
            assert frame.empty and len(frame.columns) > 0
        assert public.pack(ruid("rest_run-1")) is None
        with pytest.raises(KeyError):
            public.pack_signature(ruid("rest_run-1"))
        with pytest.raises(KeyError):
            public.epi_session_signature(suid())
        for writer, args in (
            (public.record_packs, ([],)),
            (public.forget_packs, ([],)),
            (public.set_func_digests, ("run", ruid("rest_run-1"), {})),
        ):
            with pytest.raises(ConfigError) as caught:
                writer(*args)  # type: ignore[operator]
            assert caught.value.reason == "invalid"
            assert "schema 1 has no functional tables" in caught.value.detail
        public._db.close()
    with Catalog(path, readonly=True) as private:
        assert private.func_source(1) is None
        assert private.func_inventory_hash(suid()) is None
        assert private.func_participant_uids() == {}
        with pytest.raises(KeyError):
            private.func_lease_facts(suid())
        with pytest.raises(ConfigError, match="no functional tables"):
            private.upsert_func_source(1, source_record())
    # not a byte changed; beside it only the files of SQLite's write-ahead log of a census
    # catalog, which every reader of one makes (architecture.md, as for an export)
    assert path.read_bytes() == before
    beside = {p.name for p in path.parent.iterdir()} - set(names)
    assert beside == (set() if mode == "local" else {f"{CATALOG_FILE}-wal", f"{CATALOG_FILE}-shm"})


@pytest.mark.parametrize("mode", ["local", "census"])
def test_a_writer_migrates_schema_1_and_keeps_every_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    path = tmp_path / "run" / CATALOG_FILE
    uid = schema_1_catalog(path, monkeypatch, mode)
    rows = dump(path)
    with Catalog(path) as migrated:
        assert migrated.query("PRAGMA user_version")[0][0] == 2 == store.SCHEMA_VERSION
        assert migrated.public().subject_uids() == [uid]
    after = dump(path)
    assert {name: after[name] for name in rows} == rows  # every earlier row as it was
    assert set(after) - set(rows) == {
        "func_sources",
        "func_participants",
        "func_sessions",
        "func_runs",
        "func_files",
        "packs",
    }
    assert all(not after[name] for name in set(after) - set(rows))
    with PublicCatalog.open(path.parent) as public:  # read-only, now at schema 2
        assert public._db.version == 2 and public.has_func() is False


def test_a_catalog_newer_than_this_code_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "run" / CATALOG_FILE
    Catalog.create(path, mode="local", key=SALT).close()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 3")
    conn.close()
    before = path.read_bytes()
    for opener in (
        lambda: Catalog(path),
        lambda: Catalog(path, readonly=True),
        lambda: PublicCatalog.open(path.parent),
    ):
        with pytest.raises(
            ConfigError, match="has schema 3; this version of surfcensus reads up to 2"
        ) as c:
            opener()
        assert c.value.reason == "newer schema"
    assert path.read_bytes() == before


# --- sources and participants ---------------------------------------------------------------


def test_a_functional_source_keeps_its_identity_and_takes_new_roots(local: Catalog) -> None:
    facts = stored(local)
    dataset_id, func_id = facts["dataset_id"], facts["func_id"]
    assert local.func_source(dataset_id) == source_record()
    moved = source_record(deriv_root="/elsewhere/fmriprep", raw_root=None, raw_found=None)
    assert local.upsert_func_source(dataset_id, moved) == func_id
    assert local.func_source(dataset_id) == moved
    with pytest.raises(ConfigError) as caught:
        local.upsert_func_source(dataset_id, source_record(deriv_identity="/other"))
    assert caught.value.reason == "key"
    assert local.func_source(dataset_id) == moved  # the refused call changed nothing
    with pytest.raises(KeyError):
        local.upsert_func_source(99, source_record())
    with pytest.raises(ValueError):
        local.upsert_func_source(dataset_id, source_record(connector="other"))
    assert local.func_source(99) is None


def test_relocate_func_replaces_the_roots_and_nothing_else(local: Catalog) -> None:
    facts = stored(local)
    before = local.query("SELECT * FROM func_sources")[0]
    signature = local.public().pack_signature(ruid("rest_run-2"))
    local.relocate_func(facts["func_id"], "/new/deriv", "/new/raw", "given")
    after = local.query("SELECT * FROM func_sources")[0]
    changed = {k for k in before.keys() if before[k] != after[k]}
    assert changed == {"deriv_root", "raw_root", "raw_found"}
    assert after["deriv_identity"] == IDENTITY
    assert local.func_source(facts["dataset_id"]) == source_record(
        deriv_root="/new/deriv", raw_root="/new/raw", raw_found="given"
    )
    # roots are in no signature: a moved source is not packed again
    assert local.public().pack_signature(ruid("rest_run-2")) == signature
    with pytest.raises(KeyError):
        local.relocate_func(99, "/x", None, None)


def test_participants_are_stored_and_listed(local: Catalog) -> None:
    facts = stored(local)
    local.upsert_func_participant(facts["func_id"], puid("sub-02"), "sub-02", 3)
    assert local.func_participant_uids() == {puid(): "sub-01", puid("sub-02"): "sub-02"}
    local.upsert_func_participant(facts["func_id"], puid("sub-02"), "sub-02", 4)  # an update
    frame = local.public().func_participants()
    assert list(frame["participant_uid"]) == sorted([puid(), puid("sub-02")])
    assert dict(zip(frame["participant"], frame["n_sessions"], strict=True)) == {
        "sub-01": 1,
        "sub-02": 4,
    }
    assert set(frame["session_order"]) == {store.SESSION_ORDER}
    with pytest.raises(ConfigError) as caught:  # another uid for a stored participant
        local.upsert_func_participant(facts["func_id"], puid("sub-03"), "sub-02", 1)
    assert caught.value.reason == "key"
    with pytest.raises(ConfigError) as caught:  # one uid for two participants
        local.upsert_func_participant(facts["func_id"], puid("sub-02"), "sub-09", 1)
    assert caught.value.reason == "invalid"
    with pytest.raises(KeyError):
        local.upsert_func_participant(99, puid("sub-04"), "sub-04", 1)
    with pytest.raises(ValueError):
        local.upsert_func_participant(facts["func_id"], "sub-04", "sub-04", 1)


# --- sessions and runs ------------------------------------------------------------------------


def test_a_session_is_stored_with_its_runs_and_files(local: Catalog) -> None:
    facts = stored(local)
    assert local.func_inventory_hash(suid()) == "a" * 16
    assert local.func_inventory_hash(suid(session="9")) is None
    session = dict(local.query("SELECT * FROM func_sessions")[0])
    assert session["anchor_run_uid"] == ruid("rest_run-1")
    assert session["anchor_rule"] == "acquisition_time"
    assert session["anchor_why"] == "CANARYTIME 10:11:12"
    assert (session["n_runs"], session["n_packable"], session["n_grids"]) == (3, 2, 1)
    assert session["subject_uid"] == facts["subject_uid"]
    runs = local.public().func_runs(suid())
    assert list(runs["run_uid"]) == [ruid(n) for n in ("rest_run-1", "rest_run-2", "task_run-1")]
    first = runs.iloc[0]
    assert bool(first["anchor"]) and bool(first["packable"]) and first["grid_key"] == GRID.key
    assert first["grid_shape"] == "[4, 5, 6]" and first["voxel_mm"] == "[2.0, 2.0, 3.0]"
    assert first["oblique_deg"] == 0.0
    assert (first["motion_state"], first["sdc"], first["slice_timing"]) == (
        "applied",
        "not applied",
        "unknown",
    )
    assert '"dof":6' in first["corrections"]
    last = runs.iloc[2]
    assert last["reason"] == "no raw series" and last["missing_roles"] == "raw_bold"
    assert last["grid_key"] is None and pd_na(last["flagged"])
    entities = local.query(
        "SELECT entities FROM func_runs WHERE run_uid = ?", (ruid("rest_run-1"),)
    )
    assert entities[0][0] == (
        '{"acq":null,"ce":null,"dir":null,"echo":null,"rec":null,"run":"1","ses":"1",'
        '"sub":"01","task":"rest"}'
    )
    files = local.query(
        "SELECT owner_kind, owner, role FROM func_files ORDER BY owner_kind, owner, role"
    )
    assert len(files) == 3 * 5 + 1
    with pytest.raises(KeyError):  # the participant first
        local.upsert_func_session(
            dataclasses.replace(session_row(), participant_uid=puid("sub-09"))
        )
    with pytest.raises(ValueError):
        local.upsert_func_session(session_row(runs=(run_row("x", 0, session="2"),)))


def pd_na(value: object) -> bool:
    import pandas as pd

    return bool(pd.isna(value))


def test_a_digest_stays_while_the_file_does(local: Catalog) -> None:
    stored(local)
    public = local.public()
    run = ruid("rest_run-2")
    public.set_func_digests("run", run, {FuncRole.PRE_BOLD: DIGEST, FuncRole.SDC_FIGURE: DIGEST})
    public.set_func_digests("session", suid(), {FuncRole.XFM_ANAT_FSNATIVE: DIGEST})

    def digests() -> dict[tuple[str, str], str]:
        rows = local.query("SELECT owner, role, sha256 FROM func_files WHERE sha256 IS NOT NULL")
        return {(r["owner"], r["role"]): r["sha256"] for r in rows}

    assert digests() == {(run, "pre_bold"): DIGEST, (suid(), "xfm_anat_fsnative"): DIGEST}
    local.upsert_func_session(session_row())  # the same listing: the digests stay
    assert digests() == {(run, "pre_bold"): DIGEST, (suid(), "xfm_anat_fsnative"): DIGEST}
    touched = tuple(
        dataclasses.replace(f, etag_or_mtime="2") if f.role == FuncRole.PRE_BOLD else f
        for f in run_files("rest_run-2")
    )
    runs = (
        run_row("rest_run-1", 0, anchor=True),
        run_row("rest_run-2", 1, files=touched),
        run_row("task_run-1", 2, packable=False),
    )
    local.upsert_func_session(session_row(runs, session_files=()))
    assert digests() == {}
    with pytest.raises(ValueError):
        public.set_func_digests("run", run, {FuncRole.PRE_BOLD: "nope"})


def test_a_run_that_left_keeps_its_row_until_its_pack_is_forgotten(local: Catalog) -> None:
    stored(local)
    public = local.public()
    gone, unpacked = ruid("rest_run-2"), ruid("task_run-1")
    public.record_packs([pack_row(gone)])
    local.upsert_func_session(session_row((run_row("rest_run-1", 0, anchor=True),)))
    rows = {
        r["run_uid"]: r["present"] for r in local.query("SELECT run_uid, present FROM func_runs")
    }
    assert rows == {ruid("rest_run-1"): 1, gone: 0}  # the one without a pack is gone at once
    assert not local.query("SELECT 1 FROM func_files WHERE owner IN (?, ?)", (gone, unpacked))
    assert public.pack(gone) is not None
    public.forget_packs([gone])
    assert public.pack(gone) is None
    assert {r[0] for r in local.query("SELECT run_uid FROM func_runs")} == {ruid("rest_run-1")}


def test_mark_func_absent_works_both_ways(local: Catalog) -> None:
    facts = stored(local)
    func_id = facts["func_id"]
    public = local.public()
    public.record_packs([pack_row(ruid("rest_run-2"))])
    # the session goes: its runs with it; rows and packs stay
    assert local.mark_func_absent(func_id, [puid()], []) == [suid()]
    assert not any(r[0] for r in local.query("SELECT present FROM func_runs"))
    assert not public.func_sessions()["present"].any()
    assert public.pack(ruid("rest_run-2")) is not None
    assert public.has_func()
    assert local.mark_func_absent(func_id, [puid()], []) == []  # nothing more becomes absent
    # it comes back
    assert local.mark_func_absent(func_id, [puid()], [suid()]) == []
    assert all(r[0] for r in local.query("SELECT present FROM func_runs"))
    # the participant goes
    local.mark_func_absent(func_id, [], [])
    assert local.func_participant_uids() == {}
    assert not public.has_func()
    local.mark_func_absent(func_id, [puid()], [suid()])
    assert local.func_participant_uids() == {puid(): "sub-01"}


def test_a_run_that_left_stays_absent_when_its_session_comes_back(local: Catalog) -> None:
    facts = stored(local)
    public = local.public()
    left = ruid("rest_run-2")
    public.record_packs([pack_row(left)])
    local.upsert_func_session(session_row((run_row("rest_run-1", 0, anchor=True),)))
    local.mark_func_absent(facts["func_id"], [puid()], [])
    local.mark_func_absent(facts["func_id"], [puid()], [suid()])
    present = {r[0]: r[1] for r in local.query("SELECT run_uid, present FROM func_runs")}
    assert present == {ruid("rest_run-1"): 1, left: 0}


def test_the_lease_facts_of_a_session(local: Catalog) -> None:
    facts = stored(local)
    lease = local.func_lease_facts(suid())
    assert lease.source == source_record()
    assert (lease.participant_uid, lease.session_uid) == (puid(), suid())
    assert (lease.participant, lease.session) == ("sub-01", "1")
    assert lease.run_ids == {n: ruid(n) for n in ("rest_run-1", "rest_run-2", "task_run-1")}
    assert lease.fs_subjects_dir == "/CANARYROOT/study/derivatives/fmriprep/sourcedata/freesurfer"
    assert (lease.fs_subject, lease.subject_uid) == ("sub-01", facts["subject_uid"])
    assert lease.inventory_hash == "a" * 16
    with pytest.raises(KeyError):
        local.func_lease_facts(suid(session="9"))


def test_the_lease_facts_of_a_session_without_contours(local: Catalog) -> None:
    stored(local, subject=False)
    lease = local.func_lease_facts(suid())
    assert (lease.fs_subjects_dir, lease.fs_subject, lease.subject_uid) == (None, None, None)


# --- packs --------------------------------------------------------------------------------------


def pack_row(run_uid: str, **changes: object) -> PackRow:
    row = PackRow(
        run_uid=run_uid,
        session_uid=suid(),
        config_hash="c" * 12,
        input_signature="s" * 16,
        status="done",
        pack_key="k" * 16,
        path=f"epi/{puid()}/{suid()}/{run_uid}",
        n_volumes=10,
        bytes=1234,
        content_digest=DIGEST,
        anchor_run_uid=ruid("rest_run-1"),
        grid_key=GRID.key,
    )
    return dataclasses.replace(row, **changes)


def test_packs_are_recorded_replaced_and_read_back(local: Catalog) -> None:
    stored(local)
    public = local.public()
    first, second = ruid("rest_run-1"), ruid("rest_run-2")
    public.record_packs(
        [
            pack_row(first),
            pack_row(
                second,
                status="failed",
                pack_key=None,
                reason="pack failure",
                detail="CANARYDETAIL x",
            ),
        ]
    )
    got = public.pack(first)
    assert got is not None and got.content_digest is None  # private
    assert got.input_signature == "s" * 16 and got.packed_at is not None
    assert dataclasses.replace(got, packed_at=None) == pack_row(first, content_digest=None)
    failed = public.pack(second)
    assert failed is not None and failed.detail == "CANARYDETAIL x"
    public.record_packs([pack_row(second)])  # a later row replaces the earlier
    assert public.pack(second) == dataclasses.replace(
        pack_row(second, content_digest=None),
        packed_at=public.pack(second).packed_at,  # type: ignore[union-attr]
    )
    frame = public.packs()
    assert list(frame["run_uid"]) == sorted([first, second])
    assert "detail" in frame.columns and "content_digest" not in frame.columns
    # one transaction: a batch with a row the catalog cannot hold stores nothing
    with pytest.raises(sqlite3.IntegrityError):
        public.record_packs([pack_row(first, n_volumes=99), pack_row(ruid("nothing"))])
    assert public.pack(first).n_volumes == 10  # type: ignore[union-attr]
    public.forget_packs([first, ruid("never packed")])
    assert public.pack(first) is None and public.pack(second) is not None
    assert {r[0] for r in local.query("SELECT run_uid FROM func_runs")} >= {first}  # it stays


def test_a_census_run_keeps_labels_and_details_from_the_public_side(tmp_path: Path) -> None:
    path = tmp_path / "census" / CATALOG_FILE
    with Catalog.create(path, mode="census", key=KEY) as census:
        dataset = make_dataset()
        dataset_id = census.upsert_dataset(dataset, pseudonym_of(KEY, dataset), None)
        func_id = census.upsert_func_source(dataset_id, source_record())
        census.upsert_func_participant(func_id, puid(), "sub-01", 1)
        census.upsert_func_session(session_row())
        public = census.public()
        public.record_packs([pack_row(ruid("rest_run-1"), detail="CANARYDETAIL y")])
        assert list(public.func_participants()["participant"]) == [None]
        sessions = public.func_sessions()
        assert list(sessions["session_label"]) == [None] and list(sessions["fs_subject"]) == [None]
        runs = public.func_runs()
        assert set(runs["name"]) == {None} and set(runs["detail"]) == {None}
        assert "detail" not in public.packs().columns
        assert public.pack(ruid("rest_run-1")).detail is None  # type: ignore[union-attr]
        assert public.has_func()


# --- the skip signatures ------------------------------------------------------------------------


def signatures(public: PublicCatalog) -> dict[str, str]:
    runs = {n: public.pack_signature(ruid(n)) for n in ("rest_run-1", "rest_run-2")}
    return {**runs, "session": public.epi_session_signature(suid())}


def changed_run(
    name: str, role: FuncRole, *, size: int | None = None, mtime: str | None = None
) -> FuncFileRef:
    (found,) = [f for f in run_files(name) if f.role == role]
    return dataclasses.replace(
        found, size=found.size if size is None else size, etag_or_mtime=mtime or found.etag_or_mtime
    )


def session_with(name: str, replaced: FuncFileRef, **row: object) -> FuncSessionRow:
    runs = []
    for run in session_row().runs:
        if run.run_key == name:
            files = tuple(replaced if f.role == replaced.role else f for f in run.files)
            run = dataclasses.replace(run, files=files)
        runs.append(run)
    return dataclasses.replace(session_row(tuple(runs)), **row)


def test_the_signatures_follow_the_listing_and_nothing_else(local: Catalog) -> None:
    facts = stored(local)
    public = local.public()
    subject = facts["subject_uid"]
    base = signatures(public)
    assert len(set(base.values())) == 3 and all(len(v) == 16 for v in base.values())
    with pytest.raises(KeyError):
        public.pack_signature(ruid("task_run-1"))  # cannot be packed
    with pytest.raises(KeyError):
        public.pack_signature(ruid("nothing"))
    with pytest.raises(KeyError):
        public.epi_session_signature(suid(session="9"))

    def after(change) -> dict[str, str]:  # type: ignore[no-untyped-def]
        change()
        now = signatures(public)
        local.upsert_func_session(dataclasses.replace(session_row(subject_uid=subject)))
        restore_fs()
        assert signatures(public) == base  # put back, it is the signature of before
        return {k for k, v in now.items() if v != base[k]}  # type: ignore[return-value]

    def restore_fs() -> None:
        with local.write() as conn:
            conn.execute(
                "UPDATE files SET size = 1000 WHERE subject_uid = ? AND role = 't1'", (subject,)
            )
            conn.execute(
                "UPDATE files SET etag_or_mtime = 'CANARYETAG161' "
                "WHERE subject_uid = ? AND role = 'lh_thickness'",
                (subject,),
            )

    def put(name: str, role: FuncRole, **change: object):  # type: ignore[no-untyped-def]
        return lambda: local.upsert_func_session(
            session_with(name, changed_run(name, role, **change), subject_uid=subject)  # type: ignore[arg-type]
        )

    # a file of a run that is not the anchor: its own pack and the session
    assert after(put("rest_run-2", FuncRole.PRE_BOLD, size=7)) == {"rest_run-2", "session"}
    assert after(put("rest_run-2", FuncRole.CONFOUNDS, mtime="9")) == {"rest_run-2", "session"}
    # a file of the anchor other than its transform: the anchor's pack only
    assert after(put("rest_run-1", FuncRole.RAW_BOLD, mtime="9")) == {"rest_run-1", "session"}
    # the anchor's transform: every pack of the session
    assert after(put("rest_run-1", FuncRole.XFM_TO_ANAT, size=7)) == {
        "rest_run-1",
        "rest_run-2",
        "session",
    }
    # a session-level file: every pack
    fsnative = ref(FuncRole.XFM_ANAT_FSNATIVE, "sub-01/anat/sub-01_from-T1w_to-fsnative_xfm.txt")
    for moved in (
        dataclasses.replace(fsnative, size=5),
        dataclasses.replace(fsnative, etag_or_mtime="5"),
    ):
        assert after(
            lambda m=moved: local.upsert_func_session(
                session_row(subject_uid=subject, session_files=(m,))
            )
        ) == {"rest_run-1", "rest_run-2", "session"}

    # a FreeSurfer file of the contours: every pack; one that the contours do not read: none
    def fs(sql: str):  # type: ignore[no-untyped-def]
        def change() -> None:
            with local.write() as conn:
                conn.execute(sql, (subject,))

        return change

    assert after(fs("UPDATE files SET size = 1 WHERE subject_uid = ? AND role = 't1'")) == {
        "rest_run-1",
        "rest_run-2",
        "session",
    }
    assert (
        after(
            fs(
                "UPDATE files SET etag_or_mtime = 'x' "
                "WHERE subject_uid = ? AND role = 'lh_thickness'"
            )
        )
        == set()
    )
    # the inventory hash alone: the session only
    assert after(
        lambda: local.upsert_func_session(session_row(subject_uid=subject, inventory_hash="b" * 16))
    ) == {"session"}
    # another anchor: every pack
    swapped = (
        dataclasses.replace(run_row("rest_run-1", 0), anchor=False),
        dataclasses.replace(run_row("rest_run-2", 1), anchor=True),
        run_row("task_run-1", 2, packable=False),
    )
    assert after(
        lambda: local.upsert_func_session(
            session_row(swapped, subject_uid=subject, anchor=ruid("rest_run-2"))
        )
    ) == {"rest_run-1", "rest_run-2", "session"}
    # no contours any more: every pack
    assert after(lambda: local.upsert_func_session(session_row())) == {
        "rest_run-1",
        "rest_run-2",
        "session",
    }
    # the roots, the reports of statements and the detail are in no signature
    local.relocate_func(facts["func_id"], "/x", None, None)
    assert signatures(public) == base


def test_a_run_that_cannot_be_packed_is_in_no_signature(local: Catalog) -> None:
    subject = stored(local)["subject_uid"]
    public = local.public()
    base = signatures(public)
    other = tuple(dataclasses.replace(f, size=f.size + 1) for f in run_files("task_run-1"))
    runs = (
        run_row("rest_run-1", 0, anchor=True),
        run_row("rest_run-2", 1),
        run_row("task_run-1", 2, packable=False, files=other),
    )
    local.upsert_func_session(session_row(runs, subject_uid=subject))
    assert signatures(public) == base


def test_a_signature_matches_records_func_signature(local: Catalog) -> None:
    from surfcensus.records import func_signature

    stored(local, subject=False)
    public = local.public()
    # for the anchor there is no anchor line, so the signature is func_signature of its refs,
    # then the session's refs, then the three lines
    order = {role: i for i, role in enumerate(FuncRole)}
    files = [
        *sorted(run_files("rest_run-1"), key=lambda f: order[f.role]),
        ref(FuncRole.XFM_ANAT_FSNATIVE, "sub-01/anat/sub-01_from-T1w_to-fsnative_xfm.txt"),
    ]
    lines = [f"anchor\t{ruid('rest_run-1')}", f"grid\t{GRID.key}", "fs\t-"]
    assert public.pack_signature(ruid("rest_run-1")) == func_signature(files, None, lines)


def test_an_unchanged_session_is_skipped_by_its_ledger_row(local: Catalog) -> None:
    subject = stored(local)["subject_uid"]
    public = local.public()
    unit = session_unit(suid())
    signature = public.epi_session_signature(suid())
    public.record_work([WorkRow("epi_session", unit, "h" * 12, signature, "done")])
    local.upsert_func_session(session_row(subject_uid=subject))  # inventoried again, unchanged
    assert public.work_done("epi_session", unit, "h" * 12, public.epi_session_signature(suid()))


# --- the public side --------------------------------------------------------------------------


def private_values(catalog: Catalog) -> set[str]:
    values: set[str] = {"CANARYROOT", "CANARYTIME", DIGEST, "s" * 16}
    for table, columns in (
        ("func_sources", ("deriv_root", "raw_root", "deriv_identity")),
        ("func_sessions", ("inventory_hash", "anchor_why")),
        ("func_runs", ("entities",)),
        ("func_files", ("location", "etag_or_mtime", "sha256")),
        ("packs", ("input_signature", "content_digest")),
    ):
        for row in catalog.query(f"SELECT {', '.join(columns)} FROM {table}"):
            values |= {str(v) for v in row if v is not None and len(str(v)) > 3}
    return values


@pytest.mark.parametrize("mode", ["local", "census"])
def test_nothing_private_leaves_through_the_functional_public_side(
    tmp_path: Path, mode: str
) -> None:
    key = SALT if mode == "local" else KEY
    path = tmp_path / mode / CATALOG_FILE
    with Catalog.create(path, mode=mode, key=key) as catalog:  # type: ignore[arg-type]
        dataset = make_dataset()
        uid = add_subject(catalog, key, dataset, "sub-01")
        dataset_id = catalog.upsert_dataset(dataset, pseudonym_of(key, dataset), None)
        func_id = catalog.upsert_func_source(dataset_id, source_record())
        catalog.upsert_func_participant(func_id, puid(), "sub-01", 1)
        catalog.upsert_func_session(session_row(subject_uid=uid, inventory_hash="CANARYHASHabcdef"))
        public = catalog.public()
        public.record_packs([pack_row(ruid("rest_run-1"))])
        public.set_func_digests("run", ruid("rest_run-1"), {FuncRole.PRE_BOLD: DIGEST})
        forbidden = private_values(catalog)
        if mode == "census":
            forbidden |= {"CANARYDETAIL"}
        frames = {
            "func_sources": public.func_sources(),
            "func_participants": public.func_participants(),
            "func_sessions": public.func_sessions(),
            "func_runs": public.func_runs(),
            "packs": public.packs(),
        }
        for name, frame in frames.items():
            assert len(frame), name
            assert not set(map(str, frame.columns)) & PRIVATE, name
            text = frame.to_csv()
            assert not sorted(v for v in forbidden if v in text), name
        for view in (
            "public_func_sources",
            "public_func_participants",
            "public_func_sessions",
            "public_func_runs",
            "public_packs",
        ):
            rows = catalog.query(f"SELECT * FROM {view}")
            assert rows and not set(rows[0].keys()) & PRIVATE, view
            text = repr([tuple(r) for r in rows])
            assert not sorted(v for v in forbidden if v in text), view
        pack = public.pack(ruid("rest_run-1"))
        assert pack is not None and pack.content_digest is None


def test_the_frames_have_fixed_column_types(local: Catalog) -> None:
    public = local.public()
    empty = {
        "sources": public.func_sources(),
        "participants": public.func_participants(),
        "sessions": public.func_sessions(),
        "runs": public.func_runs(),
        "packs": public.packs(),
    }
    stored(local)
    public.record_packs([pack_row(ruid("rest_run-1"))])
    full = {
        "sources": public.func_sources(),
        "participants": public.func_participants(),
        "sessions": public.func_sessions(),
        "runs": public.func_runs(),
        "packs": public.packs(),
    }
    for name in empty:
        assert empty[name].empty and len(full[name])
        assert dict(empty[name].dtypes) == dict(full[name].dtypes), name
    runs = full["runs"]
    assert str(runs["flagged"].dtype) == "boolean" and str(runs["n_volumes"].dtype) == "Int64"
    assert str(full["sessions"]["stream_order"].dtype) == "int64"
    assert list(public.func_sessions(puid())["session_uid"]) == [suid()]
    assert public.func_sessions(puid("sub-09")).empty
    assert public.func_runs(suid(session="9")).empty


# --- two processes ----------------------------------------------------------------------------


def _write_sessions(path: str, participant: str, count: int) -> None:
    with Catalog(path) as catalog:
        func_id = catalog.query("SELECT func_id FROM func_sources")[0][0]
        uid = puid(participant)
        catalog.upsert_func_participant(func_id, uid, participant, count)
        for number in range(count):
            label = str(number)
            sess = privacy.session_uid(SALT, "local", IDENTITY, "fmriprep", participant, label)
            run = privacy.run_uid(SALT, "local", IDENTITY, "fmriprep", participant, label, "r")
            row = dataclasses.replace(
                session_row(
                    (
                        dataclasses.replace(
                            run_row("r", 0, anchor=True), run_uid=run, session_uid=sess
                        ),
                    ),
                    anchor=run,
                ),
                session_uid=sess,
                participant_uid=uid,
                session=label,
            )
            catalog.upsert_func_session(row)
            catalog.public().record_packs([dataclasses.replace(pack_row(run), session_uid=sess)])


def test_two_processes_write_one_catalog_at_once(local: Catalog) -> None:
    stored(local, subject=False)
    path = str(run_dir_of(local) / CATALOG_FILE)
    context = multiprocessing.get_context("spawn")
    workers = [
        context.Process(target=_write_sessions, args=(path, name, 15))
        for name in ("sub-08", "sub-09")
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(120)
        assert worker.exitcode == 0
    public = local.public()
    assert len(public.func_sessions()) == 31
    assert len(public.packs()) == 30
    assert set(local.func_participant_uids().values()) == {"sub-01", "sub-08", "sub-09"}


def test_a_run_that_left_is_in_no_session_signature(local: Catalog) -> None:
    subject = stored(local)["subject_uid"]
    public = local.public()
    left = ruid("rest_run-2")
    public.record_packs([pack_row(left)])
    local.upsert_func_session(
        session_row((run_row("rest_run-1", 0, anchor=True),), subject_uid=subject)
    )
    kept = public.epi_session_signature(suid())  # the row of the run that left is still there
    public.forget_packs([left])
    assert public.epi_session_signature(suid()) == kept
