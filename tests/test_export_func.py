"""The functional tables of ``surfcensus export`` (step 3 of the EPI interface, work package
E3.B): ``func_sessions``, ``func_runs`` and the manifest key ``epi``, on
``synth_func.write_review_folder`` through the real phases."""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import pytest
from synth_func import (  # pyrefly: ignore[missing-import]
    SyntheticFuncDataset,
    write_review_folder,
    write_session_dates,
)
from typer.testing import CliRunner, Result

from surfcensus.cli import app
from surfcensus.config import epi_settings, load_config
from surfcensus.export import tables
from surfcensus.func.batch import recorded_states
from surfcensus.run import Run
from surfcensus.store import EPI_DIR, PublicCatalog

TEN = [f"{name}.csv" for name in tables.EXPORT_TABLES]
TWELVE = [*TEN, "func_sessions.csv", "func_runs.csv"]


def cli(*args: object) -> Result:
    return CliRunner().invoke(app, [str(arg) for arg in args])


@pytest.fixture
def folder(tmp_path: Path) -> SyntheticFuncDataset:
    return write_review_folder(tmp_path / "data", extra_fs=("sub-03",))


@pytest.fixture
def packed(folder: SyntheticFuncDataset, tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    result = cli("inventory", "--run-dir", run_dir, "--source", folder.raw_root)
    assert result.exit_code == 0, result.output
    result = cli("pack", "--run-dir", run_dir)
    assert result.exit_code == 0, result.output
    return run_dir


def export(run_dir: Path, out: Path, fmt: tables.Format = "csv") -> list[str]:
    return [path.name for path in tables.export_run(run_dir, out, fmt=fmt)]


def read(out: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(out / f"{name}.csv", dtype=str, keep_default_na=False)


def manifest(out: Path) -> dict[str, Any]:
    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))


def test_the_two_tables_come_after_the_ten_with_their_columns_and_order(
    packed: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    assert export(packed, out) == [*TWELVE, "manifest.json"]
    sessions, runs = read(out, "func_sessions"), read(out, "func_runs")
    assert tuple(sessions.columns) == tables.FUNC_SESSIONS_COLUMNS
    assert tuple(runs.columns) == tables.FUNC_RUNS_COLUMNS
    assert len(sessions) == 3 and len(runs) == 19
    key = [(r.participant_uid, int(r.stream_order), r.session_uid) for r in sessions.itertuples()]
    assert key == sorted(key)
    key = [(r.session_uid, int(r.acquisition_order), r.run_uid) for r in runs.itertuples()]
    assert key == sorted(key)
    files = manifest(out)["files"]
    assert set(files) == set(TWELVE)
    assert files["func_runs.csv"]["rows"] == 19 and files["func_sessions.csv"]["rows"] == 3


def _value(value: object) -> object:
    return None if value is None or value is pd.NA or value != value else value


def test_the_tables_are_the_catalog_and_the_recorded_states(packed: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    export(packed, out, "parquet")
    sessions = pq.read_table(out / "func_sessions.parquet").to_pandas()
    runs = pq.read_table(out / "func_runs.parquet").to_pandas()
    with PublicCatalog.open(packed) as public:
        states = recorded_states(public, epi_settings(load_config(packed)).pack_hash())
        view = public.func_runs().set_index("run_uid")
        participants = public.func_participants().set_index("participant_uid")
        packs = public.packs().set_index("run_uid")
    assert set(runs["run_uid"]) == set(states)
    for row in runs.itertuples():
        state = states[row.run_uid]
        assert row.pack_state == state.state
        assert (None if pd.isna(row.pack_reason) else row.pack_reason) == state.reason
        for column in ("name", "raw_affine", "corrections", "grid_key", "reason"):
            assert _value(getattr(row, column)) == _value(view.loc[row.run_uid, column])
        if state.state == "packed":
            assert row.pack_bytes == packs.loc[row.run_uid, "bytes"]
            assert row.packed_at == packs.loc[row.run_uid, "packed_at"]
        else:
            assert pd.isna(row.pack_bytes) and pd.isna(row.packed_at)
    assert sorted(runs["pack_state"].value_counts().items()) == [
        ("cannot be packed", 8),
        ("packed", 11),
    ]
    for row in sessions.itertuples():
        mine = runs[runs["session_uid"] == row.session_uid]
        assert row.n_packed == int((mine["pack_state"] == "packed").sum())
        assert row.n_failed == 0 and row.n_pending == 0
        assert row.participant == participants.loc[row.participant_uid, "participant"]
        assert row.stream_rule == "session label"
        assert row.dataset == participants.loc[row.participant_uid, "dataset"]
        assert set(mine["participant_uid"]) == {row.participant_uid}
    assert sorted(sessions["participant"]) == ["sub-01", "sub-01", "sub-02"]  # not sub-03


def test_the_parquet_types_do_not_depend_on_the_rows(packed: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    export(packed, out, "parquet")
    for name in tables.FUNC_TABLES:
        schema = pq.read_schema(out / f"{name}.parquet")
        assert schema.remove_metadata() == tables._PARQUET_SCHEMAS[name]
    # a frame without rows has the same schema
    empty = tables._func_frame("func_runs", [])
    assert (
        tables._arrow_table(empty, tables._PARQUET_SCHEMAS["func_runs"]).schema
        == (tables._PARQUET_SCHEMAS["func_runs"])
    )
    types = {f.name: str(f.type) for f in tables._PARQUET_SCHEMAS["func_runs"]}
    assert types["flagged"] == "bool" and types["pack_bytes"] == "int64"
    assert types["corrections"] == "string" and types["frame_max_mm"] == "double"


def _private_values(run_dir: Path) -> set[str]:
    with sqlite3.connect(run_dir / "catalog.sqlite") as db:
        values = set()
        for sql in (
            "SELECT input_signature, content_digest, pack_key FROM packs",
            "SELECT inventory_hash FROM func_sessions",
            "SELECT location FROM func_files",
            "SELECT sha256 FROM func_files",
            "SELECT deriv_root, raw_root, deriv_identity FROM func_sources",
        ):
            for row in db.execute(sql):
                values |= {str(v) for v in row if v is not None}
    return values


def _trace_values(run_dir: Path) -> set[str]:
    """Framewise displacement and DVARS of the packs, as text of six decimals and more."""
    found = set()
    for path in (run_dir / EPI_DIR).rglob("traces.json"):
        for series in json.loads(path.read_text()).values():
            if isinstance(series, list):
                for value in series:
                    if isinstance(value, float) and value and len(repr(value)) > 8:
                        found.add(repr(value)[:8])
    return found


def test_nothing_private_and_no_trace_is_exported(
    folder: SyntheticFuncDataset, packed: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    export(packed, out)
    text = "\n".join((out / name).read_text() for name in ("func_sessions.csv", "func_runs.csv"))
    private = _private_values(packed)
    assert private
    assert not {value for value in private if value in text}
    traces = _trace_values(packed)
    assert traces, "the fixture's packs hold traces"
    assert not {value for value in traces if value in text}
    for root in (folder.root, folder.raw_root, folder.deriv_root, packed):
        assert str(root) not in text
    banned = {"input_signature", "content_digest", "pack_key", "fd", "dvars", "traces"}
    assert not banned & set(read(out, "func_runs").columns)


def test_no_acquisition_date_is_exported(
    folder: SyntheticFuncDataset, packed: Path, tmp_path: Path
) -> None:
    write_session_dates(
        folder.raw_root,
        "sub-01",
        sessions={"1": "1901-03-01T10:00:00", "2": "1900-01-01T10:00:00"},
    )
    assert cli("inventory", "--run-dir", packed).exit_code == 0
    out = tmp_path / "out"
    export(packed, out)
    sessions = read(out, "func_sessions")
    mine = sessions[sessions["participant"] == "sub-01"].sort_values("stream_order")
    assert list(mine["stream_rule"]) == ["acquisition time", "acquisition time"]
    assert list(mine["session_label"]) == ["2", "1"]  # the order, not the date
    for name in os.listdir(out):
        assert not re.search(r"190[01]-", (out / name).read_text()), name


def test_the_manifest_traces_the_tables_to_their_settings_and_sources(
    packed: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    export(packed, out)
    epi = epi_settings(load_config(packed))
    assert manifest(out)[tables.EPI_MANIFEST_KEY] == {
        "config_hashes": {
            "pack": epi.pack_hash(),
            "epi_session": epi.session_hash(),
            "func_inventory": epi.inventory_hash(),
        },
        "sources": [
            {
                "dataset": read(out, "func_sessions")["dataset"].iloc[0],
                "connector": "fmriprep",
                "pipeline_version": "21.0.2",
                "generation": "scanner-T1w",
            }
        ],
    }
    assert manifest(out)["manifest_schema_version"] == 1


def test_an_outdated_pack_is_packed_here_and_pending_in_status(
    folder: SyntheticFuncDataset, packed: Path, tmp_path: Path
) -> None:
    from surfcensus.func.batch import pack_status

    raw = sorted(folder.raw_root.rglob("sub-01/ses-1/func/*task-motor*_bold.nii*"))
    stamp = os.stat(raw[0]).st_mtime_ns + 10**9
    os.utime(raw[0], ns=(stamp, stamp))
    assert cli("inventory", "--run-dir", packed).exit_code == 0
    with Run(packed, migrate=False) as run:
        assert pack_status(run.context()) == (10, 0, 1)
    out = tmp_path / "out"
    export(packed, out)
    runs = read(out, "func_runs")
    assert (runs["pack_state"] == "packed").sum() == 11
    assert (runs["pack_state"] == "pending").sum() == 0


def test_a_failed_pack_is_a_row_with_its_reason(
    folder: SyntheticFuncDataset, packed: Path, tmp_path: Path
) -> None:
    series = sorted(
        folder.root.rglob("fmriprep/sub-01/ses-2/func/*space-T1w_desc-preproc_bold.nii*")
    )
    blob = series[0].read_bytes()
    series[0].write_bytes(blob[: len(blob) // 2])
    assert cli("inventory", "--run-dir", packed).exit_code == 0
    assert cli("pack", "--run-dir", packed).exit_code == 1
    out = tmp_path / "out"
    export(packed, out)
    runs = read(out, "func_runs")
    failed = runs[runs["pack_state"] == "failed"]
    assert len(failed) == 1
    assert failed["pack_reason"].iloc[0] == "pack failure" and failed["pack_detail"].iloc[0]
    assert failed["packed_at"].iloc[0] == "" and failed["pack_bytes"].iloc[0] == ""
    sessions = read(out, "func_sessions").set_index("session_uid")
    assert sessions.loc[failed["session_uid"].iloc[0], "n_failed"] == "1"


def test_a_later_export_with_the_interface_off_removes_the_functional_files(
    packed: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    export(packed, out)
    (out / "mine.csv").write_text("a user's file\n")
    assert cli("inventory", "--run-dir", packed, "--no-epi").exit_code == 0
    assert export(packed, out) == [*TEN, "manifest.json"]
    assert sorted(os.listdir(out)) == sorted([*TEN, "manifest.json", "mine.csv"])
    assert tables.EPI_MANIFEST_KEY not in manifest(out)
    # and in the other format: the csv files of an export with them go as well
    assert cli("inventory", "--run-dir", packed, "--epi").exit_code == 0
    export(packed, out)
    export(packed, out, "parquet")
    assert not [name for name in os.listdir(out) if name.endswith(".csv") and name != "mine.csv"]


def _plain_run(folder: SyntheticFuncDataset, tmp_path: Path) -> Path:
    plain = tmp_path / "plain"
    shutil.copytree(folder.fs_subjects_dir, plain)
    run_dir = tmp_path / "plainrun"
    assert cli("inventory", "--run-dir", run_dir, "--source", plain).exit_code == 0
    return run_dir


def test_a_run_without_functional_data_exports_the_ten_as_before(
    folder: SyntheticFuncDataset, tmp_path: Path
) -> None:
    run_dir = _plain_run(folder, tmp_path)
    out = tmp_path / "out"
    assert export(run_dir, out) == [*TEN, "manifest.json"]
    assert tables.EPI_MANIFEST_KEY not in manifest(out)
    assert set(manifest(out)["files"]) == set(TEN)


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


def test_a_catalog_of_schema_1_exports_the_ten_and_keeps_its_bytes(
    packed: Path, tmp_path: Path
) -> None:
    catalog = packed / "catalog.sqlite"
    _as_schema_1(catalog)
    before = catalog.read_bytes()
    out = tmp_path / "out"
    assert export(packed, out) == [*TEN, "manifest.json"]
    assert catalog.read_bytes() == before


def test_a_census_export_and_a_release_hold_nothing_functional(
    packed: Path, tmp_path: Path
) -> None:
    # no command makes a census run with functional rows: the mode of a copy is changed
    census = tmp_path / "census"
    shutil.copytree(packed, census)
    with sqlite3.connect(census / "catalog.sqlite") as db:
        db.execute("UPDATE meta SET value = 'census' WHERE key = 'mode'")
    with PublicCatalog.open(census) as public:
        assert public.mode() == "census" and public.has_func()
    out = tmp_path / "out"
    assert export(census, out) == [*TEN, "manifest.json"]
    assert tables.EPI_MANIFEST_KEY not in manifest(out)
    release = tmp_path / "release"
    names = [p.name for p in tables.export_run(census, release, profile="release")]
    assert names == [*TEN, "manifest.json"]
    assert tables.EPI_MANIFEST_KEY not in manifest(release)


def test_before_pack_the_packable_runs_are_pending(
    folder: SyntheticFuncDataset, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    assert cli("inventory", "--run-dir", run_dir, "--source", folder.raw_root).exit_code == 0
    out = tmp_path / "out"
    export(run_dir, out)
    runs = read(out, "func_runs")
    assert sorted(runs["pack_state"].value_counts().items()) == [
        ("cannot be packed", 8),
        ("pending", 11),
    ]
    sessions = read(out, "func_sessions")
    assert sorted(sessions["n_pending"].astype(int)) == [3, 4, 4]
    assert set(sessions["n_packed"]) == {"0"}
    assert set(runs["packed_at"]) == {""}


def test_a_packed_run_that_is_no_longer_selected_has_no_pack_facts(
    packed: Path, tmp_path: Path
) -> None:
    # its packs row stays until the next `pack` removes it; the export says what is recorded
    # under the present selection: not selected, and nothing of the earlier pack
    assert cli("inventory", "--run-dir", packed, "--epi-runs", "*task-rest*").exit_code == 0
    with sqlite3.connect(packed / "catalog.sqlite") as db:
        assert db.execute("SELECT count(*) FROM packs WHERE status = 'done'").fetchone()[0] == 11
    out = tmp_path / "out"
    export(packed, out)
    runs = read(out, "func_runs")
    left = runs[runs["pack_state"] == "not selected"]
    assert len(left) > 0
    assert set(left["pack_reason"]) == {"not selected"}
    assert set(left["packed_at"]) == {""} and set(left["pack_bytes"]) == {""}
