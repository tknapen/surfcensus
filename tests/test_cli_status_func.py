"""``surfcensus status`` and ``surfcensus export`` on a run with functional data, through the
command line (step 3 of the EPI interface, work package E3.Z). The lines themselves are tested
in ``test_func_status.py`` and the tables in ``test_export_func.py``; here: that the commands
print and write them, in their place, and that a run without functional data gets none."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from synth import write_synthetic_subject  # pyrefly: ignore[missing-import]
from synth_func import (  # pyrefly: ignore[missing-import]
    SyntheticFuncDataset,
    write_review_folder,
)
from typer.testing import CliRunner, Result

from surfcensus.cli import app
from surfcensus.func.status import STATUS_PREFIXES


def cli(*args: object) -> Result:
    return CliRunner().invoke(app, [str(arg) for arg in args])


@pytest.fixture
def folder(tmp_path: Path) -> SyntheticFuncDataset:
    return write_review_folder(tmp_path / "data")


@pytest.fixture
def inventoried(folder: SyntheticFuncDataset, tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    result = cli("inventory", "--run-dir", run_dir, "--source", folder.raw_root)
    assert result.exit_code == 0, result.output
    return run_dir


def _functional(output: str) -> list[str]:
    return [line for line in output.splitlines() if line.startswith(STATUS_PREFIXES)]


def test_status_says_where_the_functional_side_stands_after_the_render_line(
    inventoried: Path,
) -> None:
    result = cli("status", "--run-dir", inventoried)
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    at = next(i for i, line in enumerate(lines) if line.startswith("render: "))
    assert [line.split(":")[0] + ":" for line in lines[at + 1 :]] == list(STATUS_PREFIXES)
    first, second, third = lines[at + 1 :]
    assert first.startswith(
        "func_inventory: 3 sessions of 2 participants, 19 runs: 11 can be packed, 8 cannot ("
    )
    assert second == "pack: 0 done, 0 failed, 11 pending"
    assert re.match(r"func checks: \d+ runs? flagged, \d+ raw affines? unchecked", third)

    assert cli("pack", "--run-dir", inventoried).exit_code == 0
    result = cli("status", "--run-dir", inventoried)
    assert _functional(result.output)[1] == "pack: 11 done, 0 failed, 0 pending"


def test_status_failures_lists_the_runs_that_cannot_be_packed_by_name(inventoried: Path) -> None:
    result = cli("status", "--run-dir", inventoried, "--failures")
    assert result.exit_code == 0, result.output
    rows = [line for line in result.output.splitlines() if line.startswith("func_inventory ")]
    assert len(rows) == 8, result.output
    # a run is named by its participant and its BIDS name beside its id, as a subject is
    assert all(re.match(r"func_inventory [0-9a-f]{16} \(sub-\d+ \S+\): \S", row) for row in rows)
    assert any(": multi-echo" in row for row in rows)
    # the session unit of the pack phase is never listed: its runs are (decision D41)
    assert not any(line.startswith("epi_session ") for line in result.output.splitlines())


def test_a_run_without_functional_data_gets_no_functional_line(tmp_path: Path) -> None:
    subjects = tmp_path / "subjects"
    subjects.mkdir()
    write_synthetic_subject(subjects)
    run_dir = tmp_path / "run"
    assert cli("inventory", "--run-dir", run_dir, "--source", subjects).exit_code == 0
    result = cli("status", "--run-dir", run_dir, "--failures")
    assert result.exit_code == 0, result.output
    assert _functional(result.output) == []
    assert result.output.splitlines()[-1].startswith("render: ")


def test_export_writes_the_two_functional_tables_and_says_so_in_the_manifest(
    inventoried: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    result = cli("export", "--run-dir", inventoried, "--out", out)
    assert result.exit_code == 0, result.output
    assert (out / "func_sessions.csv").is_file()
    assert (out / "func_runs.csv").is_file()
    manifest = json.loads((out / "manifest.json").read_text())
    assert set(manifest["epi"]) == {"config_hashes", "sources"}
    assert len((out / "func_runs.csv").read_text().splitlines()) == 1 + 19


@pytest.mark.parametrize("how", ["no longer reads", "truncated"])
def test_a_failed_pack_keeps_its_detail_for_status_and_the_subject_page(
    folder: SyntheticFuncDataset, inventoried: Path, how: str
) -> None:
    """A preprocessed series that no longer reads at the pack (the same size and time as at the
    inventory, so the listing does not change) was stored with no detail: the lease lists such
    a run as ``pack failure`` with the reason in its plan, and the worker kept only the runs
    its plan still calls packable (seen in passing by the review of step 3). A series cut in
    half and inventoried again kept its detail already; both are checked."""
    import os
    import sqlite3

    from fastapi.testclient import TestClient

    from surfcensus.app.server import create_app

    (series,) = (folder.deriv_root / "sub-02" / "func").glob(
        "*task-nback_run-2_space-T1w_desc-preproc_bold.nii.gz"
    )
    data, stat = series.read_bytes(), series.stat()
    if how == "truncated":
        series.write_bytes(data[: len(data) // 2])
        assert cli("inventory", "--run-dir", inventoried).exit_code == 0
    else:
        series.write_bytes(bytes(b ^ 0x5A for b in data))
        os.utime(series, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert cli("pack", "--run-dir", inventoried).exit_code == 1
    with sqlite3.connect(inventoried / "catalog.sqlite") as db:
        ((reason, detail),) = db.execute("SELECT reason, detail FROM packs WHERE status = 'failed'")
    assert reason == "pack failure"
    assert detail and (series.name in detail or "pre_bold" in detail), detail
    assert str(folder.raw_root) not in detail and "/tmp" not in detail
    result = cli("status", "--run-dir", inventoried, "--failures")
    (line,) = [row for row in result.output.splitlines() if row.startswith("pack ")]
    assert line.endswith(f"(sub-02 task-nback_run-2): pack failure: {detail}"), line
    with TestClient(create_app(inventoried), base_url="http://127.0.0.1") as client:
        rows = client.get("/api/subjects").json()["subjects"]
        runs = [
            run
            for row in rows
            for session in client.get(f"/api/subjects/{row['key']}").json()["sessions"]
            for run in session["runs"]
            if run["state"] == "failed"
        ]
    assert [(r["name"], r["reason"], r["detail"]) for r in runs] == [
        ("task-nback_run-2", "pack failure", detail)
    ]
