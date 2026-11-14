"""``surfcensus pack``, the EPI options of ``inventory`` and ``review`` on the command line, and
``review`` stopped by Ctrl-C and SIGTERM while EPI packs are made behind the app (step 2 of the
EPI interface, work package E2.Z)."""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from synth_func import (  # pyrefly: ignore[missing-import]
    SyntheticFuncDataset,
    write_review_folder,
)
from typer.testing import CliRunner, Result

from surfcensus.cli import app
from surfcensus.config import CONFIG_FILE, load_config
from surfcensus.privacy import HashKey
from surfcensus.run import Run
from surfcensus.sources.local import local
from surfcensus.store import EPI_DIR

SECONDS = r"\d+\.\d s"


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


def _packs(run_dir: Path) -> list[tuple[str, str]]:
    with sqlite3.connect(run_dir / "catalog.sqlite") as db:
        return db.execute("SELECT run_uid, status FROM packs ORDER BY run_uid").fetchall()


def test_inventory_lists_the_functional_runs_after_the_subjects(
    folder: SyntheticFuncDataset, inventoried: Path
) -> None:
    result = cli("inventory", "--run-dir", inventoried)
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith("Functional: "))
    assert lines[first - 1].startswith("inventory: ")  # after the surface report
    # without --source, where the folders lie is said relative to the stored root, the
    # derivative that holds the subjects
    # the derivative and the raw dataset are named the same way: here both as they are,
    # since the raw dataset does not lie in the stored root (review of step 2, A-F5)
    assert (
        f"19 runs in 3 sessions of 2 participants in {folder.deriv_root} (fMRIPrep 21.0.2), "
        f"raw series in {folder.raw_root} (BIDS)"
    ) in lines[first]
    assert re.fullmatch(rf"func_inventory: 0 done, 3 skipped, 0 failed, {SECONDS}", lines[-1])


def test_pack_packs_what_is_pending_and_then_nothing(inventoried: Path) -> None:
    result = cli("pack", "--run-dir", inventoried)
    assert result.exit_code == 0, result.output
    assert re.match(rf"pack: 11 done, 0 skipped, 0 failed, {SECONDS}", result.output)
    assert "EPI packs: 11 of 11 runs made." in result.output
    result = cli("pack", "--run-dir", inventoried)
    assert result.exit_code == 0
    assert re.match(rf"pack: 0 done, 11 skipped, 0 failed, {SECONDS}", result.output)
    assert {status for _, status in _packs(inventoried)} == {"done"}


def test_pack_ends_with_1_on_a_failed_pack_and_names_retry_failed(inventoried: Path) -> None:
    # review of step 2, D-F3 and D-F6: a failed pack row is exit code 1, at every call until
    # it is tried again, and the line that counts it says how to try it again
    data = inventoried.parent / "data"
    series = sorted(data.rglob("fmriprep/sub-01/ses-2/func/*space-T1w_desc-preproc_bold.nii*"))
    assert series
    blob = series[0].read_bytes()
    series[0].write_bytes(blob[: len(blob) // 2])  # the header reads, the volumes do not
    assert cli("inventory", "--run-dir", inventoried).exit_code == 0
    for _ in range(2):
        result = cli("pack", "--run-dir", inventoried)
        assert result.exit_code == 1, result.output
        assert "1 failed" in result.output
        assert f"surfcensus pack --run-dir {inventoried} --retry-failed" in result.output
    result = cli("pack", "--run-dir", inventoried, "--retry-failed")
    assert result.exit_code == 1, result.output


def test_pack_with_the_data_unmounted_works_when_nothing_is_pending_and_stops_otherwise(
    folder: SyntheticFuncDataset, inventoried: Path, tmp_path: Path
) -> None:
    assert cli("pack", "--run-dir", inventoried).exit_code == 0
    away = tmp_path / "away"
    shutil.move(folder.root, away)
    result = cli("pack", "--run-dir", inventoried)
    assert result.exit_code == 0, result.output
    assert "0 done, 11 skipped" in result.output
    # a pack that lost a chunk is pending, and the data is not there to make it from
    chunk = next((inventoried / EPI_DIR).glob("*/*/*/raw/c0000.bin"))
    chunk.unlink()
    result = cli("pack", "--run-dir", inventoried)
    assert result.exit_code == 2
    assert result.output.startswith("not found: ")
    shutil.move(away, folder.root)
    result = cli("pack", "--run-dir", inventoried)
    assert result.exit_code == 0 and "1 done, 10 skipped" in result.output


def test_pack_epi_runs_is_stored_and_a_glob_that_matches_nothing_is_refused(
    inventoried: Path,
) -> None:
    before = (inventoried / CONFIG_FILE).read_bytes()
    result = cli("pack", "--run-dir", inventoried, "--epi-runs", "*task-nothing*")
    assert result.exit_code == 2 and "no functional run matches *task-nothing*" in result.output
    assert (inventoried / CONFIG_FILE).read_bytes() == before
    result = cli("pack", "--run-dir", inventoried, "--epi-runs", "*_run-1")
    assert result.exit_code == 0, result.output
    assert load_config(inventoried).epi.runs == ("*_run-1",)  # pyrefly: ignore
    assert len(_packs(inventoried)) < 11


def test_pack_refuses_a_census_run(folder: SyntheticFuncDataset, tmp_path: Path) -> None:
    run_dir = tmp_path / "census"
    key = HashKey(bytes(range(1, 33)), "dataset-key")
    Run(run_dir, local(folder.raw_root), key=key).close()
    result = cli("pack", "--run-dir", run_dir)
    assert result.exit_code == 2
    assert "census run has no EPI interface" in result.output


def test_inventory_stores_the_epi_options_and_refuses_a_glob_before_a_run_exists(
    folder: SyntheticFuncDataset, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    result = cli(
        "inventory", "--run-dir", run_dir, "--source", folder.raw_root, "--epi-runs", "*nothing*"
    )
    assert result.exit_code == 2 and not run_dir.exists()
    result = cli("inventory", "--run-dir", run_dir, "--source", folder.raw_root, "--no-epi")
    assert result.exit_code == 0, result.output
    assert "the EPI interface is off for this run" not in result.output  # nothing listed yet
    assert not load_config(run_dir).epi.enabled  # pyrefly: ignore[missing-attribute]
    result = cli("inventory", "--run-dir", run_dir, "--epi")
    assert result.exit_code == 0 and "Functional: 19 runs" in result.output
    result = cli("inventory", "--run-dir", run_dir, "--no-epi")
    assert "the EPI interface is off for this run (--no-epi, stored)" in result.output


def test_options_that_bring_the_epi_settings_back_to_the_defaults_remove_the_section(
    inventoried: Path,
) -> None:
    # review of step 2, A-F2: config.yaml goes back to the bytes it had
    before = (inventoried / CONFIG_FILE).read_bytes()
    assert b"epi:" not in before
    assert cli("inventory", "--run-dir", inventoried, "--no-epi").exit_code == 0
    assert b"epi:" in (inventoried / CONFIG_FILE).read_bytes()
    assert cli("inventory", "--run-dir", inventoried, "--epi").exit_code == 0
    assert (inventoried / CONFIG_FILE).read_bytes() == before
    assert cli("inventory", "--run-dir", inventoried, "--epi-runs", "*_run-1").exit_code == 0
    assert load_config(inventoried).epi.runs == ("*_run-1",)  # pyrefly: ignore
    # '*' selects every run, which is what no glob means: the globs are cleared
    assert cli("inventory", "--run-dir", inventoried, "--epi-runs", "*").exit_code == 0
    assert (inventoried / CONFIG_FILE).read_bytes() == before


def test_status_works_on_a_run_with_functional_tables(inventoried: Path) -> None:
    assert cli("pack", "--run-dir", inventoried).exit_code == 0
    result = cli("status", "--run-dir", inventoried, "--failures")
    assert result.exit_code == 0, result.output
    assert "render: 0 done" in result.output


def test_review_no_serve_ends_with_0_and_with_1_only_when_there_is_nothing_to_show(
    folder: SyntheticFuncDataset, tmp_path: Path
) -> None:
    for subject in folder.subjects:
        for surface in (folder.fs_subjects_dir / subject / "surf").glob("?h.pial*"):
            surface.unlink()
    result = cli("review", folder.raw_root, "--run-dir", tmp_path / "r", "--no-serve")
    assert result.exit_code == 0, result.output  # nothing to rate, the EPI movies to show
    result = cli("review", folder.raw_root, "--run-dir", tmp_path / "r", "--no-serve", "--no-epi")
    assert result.exit_code == 1


# --- review, stopped while packs are made behind the app -------------------------------------

_SLOW = (
    "import time\n"
    "import surfcensus.func.batch as b\n"
    "f = b.pack_session_unit\n"
    "def slow(*a, **k):\n"
    "    time.sleep(1.5)\n"
    "    return f(*a, **k)\n"
    "b.pack_session_unit = slow\n"
    "from surfcensus.cli import app\n"
    "app()\n"
)
"""``surfcensus`` with every session's pack 1.5 s slower, so that a signal finds packs being
made behind the app (one worker: ``--jobs 1`` works in the process)."""


def _default_sigint() -> None:
    signal.signal(signal.SIGINT, signal.SIG_DFL)


@pytest.fixture
def slow_review(folder: SyntheticFuncDataset) -> Iterator[subprocess.Popen[str]]:
    command = [sys.executable, "-c", _SLOW, "review", str(folder.raw_root)]
    command += ["--no-browser", "--port", "0", "--jobs", "1"]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ | {"PYTHONUNBUFFERED": "1"},
        preexec_fn=_default_sigint,
    )
    yield process
    if process.poll() is None:
        process.kill()
    process.wait()
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            stream.close()


@pytest.mark.skipif(os.name == "nt", reason="signals of a POSIX system")
@pytest.mark.parametrize("stop_with", [signal.SIGINT, signal.SIGTERM])
def test_review_stops_in_order_while_packs_are_made_and_the_next_call_goes_on(
    folder: SyntheticFuncDataset,
    slow_review: subprocess.Popen[str],
    stop_with: signal.Signals,
) -> None:
    assert slow_review.stderr is not None
    err = ""
    while "All renders are made" not in err:
        line = slow_review.stderr.readline()
        assert line, f"the command ended early:\n{err}"
        err += line
    assert "EPI movies: http://127.0.0.1:" in err
    time.sleep(0.5)  # the first session is being packed
    slow_review.send_signal(stop_with)
    if stop_with == signal.SIGTERM:
        for gap in (0.0, 0.001, 0.05):
            time.sleep(gap)
            slow_review.send_signal(stop_with)
    deadline = time.monotonic() + 120
    while slow_review.poll() is None:
        assert time.monotonic() < deadline, "the command did not end"
        time.sleep(0.05)
    err += slow_review.stderr.read()
    assert slow_review.returncode == 0, err
    assert "Traceback" not in err
    assert "Waiting for the EPI packs in progress; the others are made at the next call." in err
    found = re.search(
        r"^EPI packs: (\d+) of 11 runs made; (\d+) are made at the next call\.$", err, re.M
    )
    assert found, err
    made, left = int(found.group(1)), int(found.group(2))
    assert made + left == 11 and left > 0
    # the batch in progress was finished and recorded: its packs are whole and current
    run_dir = folder.raw_root.parent / "surfcensus-run"
    assert len(_packs(run_dir)) == made
    result = cli("review", folder.raw_root, "--no-serve")
    assert result.exit_code == 0, result.output
    assert "EPI packs: 11 of 11 runs made." in result.output
    with contextlib.suppress(ProcessLookupError):
        slow_review.kill()


def test_an_export_in_a_participant_folder_of_a_raw_dataset_given_with_raw_is_refused(
    tmp_path: Path,
) -> None:
    # review of step 2, A-F1: the participant folders of a raw dataset named with --raw are
    # read, so no export goes there, also not the default export/ of a run moved there
    from synth import snapshot_tree  # pyrefly: ignore[missing-import]
    from synth_func import write_func_dataset  # pyrefly: ignore[missing-import]

    from surfcensus.config import ConfigError
    from surfcensus.export.tables import export_run

    ds = write_func_dataset(tmp_path / "data", raw_inside=False)
    run_dir = tmp_path / "run"
    result = cli("inventory", "--run-dir", run_dir, "--source", ds.deriv_root, "--raw", ds.raw_root)
    assert result.exit_code == 0, result.output
    before = snapshot_tree(ds.raw_root)
    out = ds.raw_root / "sub-02" / "e"
    result = cli("export", "--run-dir", run_dir, "--out", out)
    assert result.exit_code == 2, result.output
    assert "only read" in result.output
    with pytest.raises(ConfigError, match="only read"):
        export_run(run_dir, out)
    with Run(run_dir) as run, pytest.raises(ConfigError, match="only read"):
        run.refuse_output_in_sources(out)
    assert snapshot_tree(ds.raw_root) == before
    moved = ds.raw_root / "sub-02" / "moved"
    shutil.copytree(run_dir, moved)
    with_moved = snapshot_tree(ds.raw_root)
    for command in (("export",), ("finalize",), ("calibration", "freeze")):
        result = cli(*command, "--run-dir", moved)
        assert result.exit_code == 2, (command, result.output)
        assert "only read" in result.output
    assert snapshot_tree(ds.raw_root) == with_moved


def _as_schema_1(catalog: Path) -> None:
    """Take migration 2 out of a catalog: it is additive, so what is left is a catalog of
    schema 1 as a version before step 2 wrote it."""
    with sqlite3.connect(catalog) as db:
        objects = db.execute(
            "SELECT type, name FROM sqlite_master WHERE type IN ('view', 'table') AND "
            "(name LIKE 'func\\_%' ESCAPE '\\' OR name LIKE 'public\\_func\\_%' ESCAPE '\\' "
            "OR name IN ('packs', 'public_packs')) ORDER BY type = 'table'"
        ).fetchall()
        assert len(objects) == 11
        for kind, name in objects:
            db.execute(f'DROP {kind.upper()} "{name}"')
        db.execute("DELETE FROM work WHERE phase = 'epi_session'")
        db.execute("PRAGMA user_version = 1")


def test_status_and_calibration_freeze_do_not_migrate_a_catalog_of_schema_1(
    inventoried: Path,
) -> None:
    # review of step 2, D-F2: an earlier version still opens the run afterwards
    catalog = inventoried / "catalog.sqlite"
    _as_schema_1(catalog)
    before = catalog.read_bytes()
    result = cli("status", "--run-dir", inventoried)
    assert result.exit_code == 0, result.output
    result = cli("status", "--run-dir", inventoried, "--failures")
    assert result.exit_code == 0, result.output
    result = cli("calibration", "freeze", "--run-dir", inventoried)
    assert result.exit_code == 2, result.output  # nothing is rated: refused
    with sqlite3.connect(catalog) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 1
    assert catalog.read_bytes() == before
    assert cli("pack", "--run-dir", inventoried).exit_code == 0  # a writer migrates
    with sqlite3.connect(catalog) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 2


def test_pack_says_what_is_there_to_pack(tmp_path: Path, inventoried: Path) -> None:
    # review of step 2, A-F5 (c) and (d)
    result = cli("pack", "--run-dir", tmp_path / "nothing")
    assert result.exit_code == 2
    assert "holds no run; make one with surfcensus review PATH" in result.output
    assert "give a source" not in result.output
    assert cli("inventory", "--run-dir", inventoried, "--no-epi").exit_code == 0
    result = cli("pack", "--run-dir", inventoried)
    assert result.exit_code == 0, result.output
    assert "EPI packs: none; the EPI interface is off for this run" in result.output
    from synth import write_synthetic_subject  # pyrefly: ignore[missing-import]

    subjects = tmp_path / "subjects"
    write_synthetic_subject(subjects, name="sub-01", shape=(64, 64, 64), subdivisions=2)
    plain = tmp_path / "plain"
    assert cli("inventory", "--run-dir", plain, "--source", subjects).exit_code == 0
    result = cli("pack", "--run-dir", plain)
    assert result.exit_code == 0, result.output
    assert "EPI packs: none; this run has no functional run that can be packed." in result.output
