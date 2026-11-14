"""What ``surfcensus status`` says about the functional data of a run (step 3 of the EPI
interface, work package E3.B): ``func.status.status_lines``, ``failure_rows`` and
``unit_names``, on ``synth_func.write_review_folder`` through the real phases."""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
from pathlib import Path

import pytest
from synth_func import (  # pyrefly: ignore[missing-import]
    SyntheticFuncDataset,
    write_review_folder,
)
from typer.testing import CliRunner, Result

from surfcensus.cli import app
from surfcensus.func import status
from surfcensus.privacy import HashKey
from surfcensus.records import RunContext
from surfcensus.run import Run
from surfcensus.sources.local import local

REASONS = (
    "multi-echo 1, native space only 1, no brain mask 1, no preprocessed series 1, "
    "no raw series 1, no transform to the anatomy 1, standard space only 1, "
    "volume counts differ 1"
)
FIRST = (
    f"func_inventory: 3 sessions of 2 participants, 19 runs: 11 can be packed, 8 cannot ({REASONS})"
)
CHECKS = (
    "func checks: 3 runs flagged, 1 raw affine unchecked, 1 session with the anchor the "
    "outlier (statements about the raw data)"
)
CANNOT = {
    "sub-02 task-me_run-1": "multi-echo",
    "sub-02 task-min_run-1": "no preprocessed series",
    "sub-02 task-native_run-1": "native space only",
    "sub-02 task-rest_run-1": "no raw series",
    "sub-02 task-rest_run-2": "no brain mask",
    "sub-02 task-rest_run-3": "no transform to the anatomy",
    "sub-02 task-rest_run-4": "volume counts differ",
    "sub-02 task-std_run-1": "standard space only",
}


def cli(*args: object) -> Result:
    return CliRunner().invoke(app, [str(arg) for arg in args])


@pytest.fixture
def folder(tmp_path: Path) -> SyntheticFuncDataset:
    return write_review_folder(tmp_path / "data", extra_fs=("sub-03",))


@pytest.fixture
def inventoried(folder: SyntheticFuncDataset, tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    result = cli("inventory", "--run-dir", run_dir, "--source", folder.raw_root)
    assert result.exit_code == 0, result.output
    return run_dir


@pytest.fixture
def packed(inventoried: Path) -> Path:
    result = cli("pack", "--run-dir", inventoried)
    assert result.exit_code == 0, result.output
    return inventoried


def _context(run_dir: Path) -> tuple[Run, RunContext]:
    # as `status` opens a run: for writing (a killed writer is put right) and not migrated
    run = Run(run_dir, migrate=False)
    return run, run.context()


def lines(run_dir: Path) -> list[str]:
    run, ctx = _context(run_dir)
    with run:
        return status.status_lines(ctx)


def named_failures(run_dir: Path) -> list[tuple[str, str, str, str | None]]:
    run, ctx = _context(run_dir)
    with run:
        names = status.unit_names(run.public())
        return [
            (phase, names[unit], reason, detail)
            for unit, phase, reason, detail in status.failure_rows(ctx)
        ]


def test_the_three_lines_before_and_after_pack(inventoried: Path) -> None:
    assert lines(inventoried) == [FIRST, "pack: 0 done, 0 failed, 11 pending", CHECKS]
    assert cli("pack", "--run-dir", inventoried).exit_code == 0
    got = lines(inventoried)
    assert got == [FIRST, "pack: 11 done, 0 failed, 0 pending", CHECKS]
    assert tuple(line.split(" ", 1)[0] for line in got) == ("func_inventory:", "pack:", "func")
    assert all(
        line.startswith(prefix) for line, prefix in zip(got, status.STATUS_PREFIXES, strict=True)
    )


def test_status_lines_write_nothing(packed: Path) -> None:
    catalog = packed / "catalog.sqlite"
    before = catalog.read_bytes(), os.stat(catalog).st_mtime_ns
    lines(packed)
    named_failures(packed)
    assert (catalog.read_bytes(), os.stat(catalog).st_mtime_ns) == before


def test_the_runs_that_cannot_be_packed_are_listed_by_name(packed: Path) -> None:
    rows = named_failures(packed)
    assert {name: reason for phase, name, reason, _ in rows if phase == "func_inventory"} == CANNOT
    assert all(phase == "func_inventory" for phase, *_ in rows)  # nothing failed to pack
    # sorted by phase, then unit, as the listing sorts
    run, ctx = _context(packed)
    with run:
        raw = status.failure_rows(ctx)
    assert raw == sorted(raw, key=lambda row: (row[1], row[0]))


def test_a_failed_pack_is_counted_listed_and_said_how_to_retry(
    folder: SyntheticFuncDataset, inventoried: Path
) -> None:
    series = sorted(
        folder.root.rglob("fmriprep/sub-01/ses-2/func/*space-T1w_desc-preproc_bold.nii*")
    )
    blob = series[0].read_bytes()
    series[0].write_bytes(blob[: len(blob) // 2])  # the header reads, the volumes do not
    assert cli("inventory", "--run-dir", inventoried).exit_code == 0
    assert cli("pack", "--run-dir", inventoried).exit_code == 1
    got = lines(inventoried)
    assert got[1] == (
        "pack: 10 done, 1 failed, 0 pending (tried again with: surfcensus pack --run-dir "
        f"{inventoried} --retry-failed)"
    )
    failed = [row for row in named_failures(inventoried) if row[0] == "pack"]
    assert len(failed) == 1
    _, name, reason, detail = failed[0]
    assert name.startswith("sub-01 ses-2_task-rest_run-") and reason == "pack failure"
    assert isinstance(detail, str) and detail and "/" not in detail  # a file name, no path


def test_with_the_interface_off_the_pack_line_says_so(packed: Path) -> None:
    assert cli("inventory", "--run-dir", packed, "--no-epi").exit_code == 0
    assert lines(packed) == [FIRST, "pack: off (--no-epi, stored)", CHECKS]
    assert {phase for phase, *_ in named_failures(packed)} == {"func_inventory"}


def test_an_outdated_pack_is_pending_here(folder: SyntheticFuncDataset, packed: Path) -> None:
    # the pack is recorded (the pages and the export call it packed); `pack` would make it
    # again, so `status` counts it pending (tests/test_export_func.py has the other half)
    raw = sorted(folder.raw_root.rglob("sub-01/ses-1/func/*task-motor*_bold.nii*"))
    stamp = os.stat(raw[0]).st_mtime_ns + 10**9
    os.utime(raw[0], ns=(stamp, stamp))
    assert cli("inventory", "--run-dir", packed).exit_code == 0
    assert lines(packed)[1] == "pack: 10 done, 0 failed, 1 pending"


def test_a_session_that_could_not_be_planned_is_said_and_listed(packed: Path) -> None:
    with sqlite3.connect(packed / "catalog.sqlite") as db:
        uids = [row[0] for row in db.execute("SELECT session_uid FROM func_sessions ORDER BY 1")]
        db.execute(
            "UPDATE func_sessions SET problem = 'unreadable: sub-01_ses-1_scans.tsv (bad)' "
            "WHERE session_uid = ?",
            (uids[0],),
        )
        # a fetch problem is no failure (decision D35)
        db.execute(
            "UPDATE func_sessions SET problem = 'not fetched: 1 file is a git-annex link' "
            "WHERE session_uid = ?",
            (uids[1],),
        )
    assert lines(packed)[0] == f"{FIRST}; 1 session could not be planned"
    run, ctx = _context(packed)
    with run:
        names = status.unit_names(run.public())
        rows = [row for row in status.failure_rows(ctx) if row[0].startswith("session:")]
    assert rows == [
        (f"session:{uids[0]}", "func_inventory", "unreadable", "sub-01_ses-1_scans.tsv (bad)")
    ]
    assert names[rows[0][0]] in {"sub-01 ses-1", "sub-01 ses-2", "sub-02"}


def test_unit_names_name_runs_and_sessions_by_participant(packed: Path) -> None:
    run, _ = _context(packed)
    with run:
        public = run.public()
        names = status.unit_names(public)
        n_runs, n_sessions = len(public.func_runs()), len(public.func_sessions())
    assert len(names) == n_runs + n_sessions
    sessions = sorted(name for unit, name in names.items() if unit.startswith("session:"))
    assert sessions == ["sub-01 ses-1", "sub-01 ses-2", "sub-02"]
    assert "sub-01 ses-1_task-rest_run-1" in names.values()
    assert set(CANNOT) <= set(names.values())


def _as_schema_1(catalog: Path) -> None:
    """As in tests/test_cli_pack.py: migration 2 taken out, a catalog of before step 2."""
    with sqlite3.connect(catalog) as db:
        objects = db.execute(
            "SELECT type, name FROM sqlite_master WHERE type IN ('view', 'table') AND "
            "(name LIKE 'func\\_%' ESCAPE '\\' OR name LIKE 'public\\_func\\_%' ESCAPE '\\' "
            "OR name IN ('packs', 'public_packs')) ORDER BY type = 'table'"
        ).fetchall()
        for kind, name in objects:
            db.execute(f'DROP {kind.upper()} "{name}"')
        db.execute("DELETE FROM work WHERE phase = 'epi_session'")
        db.execute("PRAGMA user_version = 1")


def test_a_catalog_of_schema_1_gives_nothing_and_keeps_its_bytes(packed: Path) -> None:
    catalog = packed / "catalog.sqlite"
    _as_schema_1(catalog)
    before = catalog.read_bytes()
    run, ctx = _context(packed)
    with run:
        assert status.status_lines(ctx) == []
        assert status.failure_rows(ctx) == []
        assert status.unit_names(run.public()) == {}
    assert catalog.read_bytes() == before
    with sqlite3.connect(catalog) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 1


def test_a_run_without_functional_data_gives_nothing(
    folder: SyntheticFuncDataset, tmp_path: Path
) -> None:
    plain = tmp_path / "plain"
    shutil.copytree(folder.fs_subjects_dir, plain)
    run_dir = tmp_path / "plainrun"
    assert cli("inventory", "--run-dir", run_dir, "--source", plain).exit_code == 0
    run, ctx = _context(run_dir)
    with run:
        assert not run.public().has_func()
        assert status.status_lines(ctx) == []
        assert status.failure_rows(ctx) == []
        assert status.unit_names(run.public()) == {}


def test_a_census_run_gives_nothing(
    folder: SyntheticFuncDataset, packed: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "census"
    key = HashKey(bytes(range(1, 33)), "dataset-key")
    Run(run_dir, local(folder.raw_root), key=key).close()
    run, ctx = _context(run_dir)
    with run:
        assert status.status_lines(ctx) == []
        assert status.failure_rows(ctx) == []
    # a census run with functional rows (no command makes one): still nothing
    run, ctx = _context(packed)
    with run:
        ctx.public.mode = lambda: "census"  # pyrefly: ignore[bad-assignment]
        assert status.status_lines(ctx) == []
        assert status.failure_rows(ctx) == []


def test_runs_left_out_by_epi_runs_are_a_reason_and_no_failure(inventoried: Path) -> None:
    assert cli("inventory", "--run-dir", inventoried, "--epi-runs", "*task-rest*").exit_code == 0
    first = lines(inventoried)[0]
    found = re.search(r"(\d+) can be packed, (\d+) cannot \((.*)\)$", first)
    assert found is not None
    reasons = dict(part.rsplit(" ", 1) for part in found.group(3).split(", "))
    assert int(reasons["not selected"]) > 0
    assert int(found.group(1)) + int(found.group(2)) == 19
    assert int(found.group(1)) < 11
    rows = named_failures(inventoried)
    assert not [row for row in rows if row[2] == "not selected"]
