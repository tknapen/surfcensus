"""``func.inventory``: phase 2f of a run (the functional sessions and runs of a local source
listed, planned and stored), the lease of a session for the pack phase, and the fetch lines of
git-annex placeholders (docs/epi_interface.md, "Step 2: the plan"; decisions D21, D35, D36,
D38, D39).

Every test writes the folder of ``synth_func.write_review_folder`` (a BIDS root with raw
functional data, its fMRIPrep derivative and the FreeSurfer subjects ``sub-01`` and
``sub-02``), inventories its surfaces with the surface inventory and then runs phase 2f on a
catalog it made itself, as ``Run`` will (work package E2.Z).
"""

from __future__ import annotations

import builtins
import io
import os
import shutil
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import pytest
from synth_annex import snapshot_links  # pyrefly: ignore[missing-import]
from synth_func import (  # pyrefly: ignore[missing-import]
    SyntheticFuncDataset,
    func_counts,
    write_func_dataset,
    write_review_folder,
)

from surfcensus.config import EpiConfig, RunConfig, epi_settings, new_local_config, run_key
from surfcensus.func import inventory
from surfcensus.inventory.subjects import run_inventory
from surfcensus.privacy import HashKey
from surfcensus.records import NO_RAW_SERIES as NO_RAW
from surfcensus.records import NOT_FETCHED, NOT_SELECTED, PhaseReport
from surfcensus.sources.base import SourceError
from surfcensus.sources.local import LocalSource
from surfcensus.store import Catalog


@dataclass
class Bench:
    ds: SyntheticFuncDataset
    root: Path
    """The folder the run was made on."""
    config: RunConfig
    key: HashKey
    catalog: Catalog

    def sources(self) -> list[LocalSource]:
        return [LocalSource(self.root)]

    def surfaces(self) -> PhaseReport:
        return run_inventory(self.catalog, self.sources(), self.config, self.key)

    def func(self, *, force: bool = False, config: RunConfig | None = None) -> PhaseReport:
        return inventory.run_func_inventory(
            self.catalog, self.sources(), config or self.config, self.key, force=force
        )

    def with_epi(self, **settings: object) -> RunConfig:
        self.config = self.config.model_copy(update={"epi": EpiConfig(**settings)})
        return self.config

    def rows(self, sql: str, *args: object) -> list[tuple[object, ...]]:
        return [tuple(row) for row in self.catalog.query(sql, args)]


def _bench(tmp_path: Path, ds: SyntheticFuncDataset, root: Path | None = None) -> Iterator[Bench]:
    root = ds.raw_root if root is None else root
    config = new_local_config(root)
    key = run_key(config)
    with Catalog.create(tmp_path / "run" / "catalog.sqlite", mode="local", key=key) as catalog:
        bench = Bench(ds, root, config, key, catalog)
        bench.surfaces()
        yield bench


@pytest.fixture
def bench(tmp_path: Path) -> Iterator[Bench]:
    yield from _bench(tmp_path, write_review_folder(tmp_path / "data"))


def _runs(bench: Bench) -> dict[str, tuple[object, ...]]:
    """``<participant> <name>`` -> reason, packable, selected, anchor, session_order, present."""
    rows = bench.catalog.query(
        "SELECT p.participant, r.name, r.reason, r.packable, r.selected, r.anchor, "
        "r.session_order, r.present FROM func_runs r JOIN func_sessions s USING (session_uid) "
        "JOIN func_participants p USING (participant_uid)"
    )
    return {f"{row[0]} {row[1]}": tuple(row)[2:] for row in rows}


# --- what is stored -------------------------------------------------------------------------


def test_the_counts_of_the_design(bench: Bench) -> None:
    report = bench.func()
    counts = func_counts(bench.ds)
    assert (report.done, report.skipped, report.failed) == (len(counts.sessions), 0, 0)
    sessions = bench.rows(
        "SELECT p.participant, s.session_label FROM func_sessions s "
        "JOIN func_participants p USING (participant_uid) ORDER BY p.participant, s.stream_order"
    )
    assert tuple(sessions) == counts.sessions
    runs = _runs(bench)
    assert len(runs) == counts.runs
    assert sum(1 for r in runs.values() if r[1]) == counts.packable
    reasons: dict[str, int] = {}
    for reason, *_ in runs.values():
        if reason is not None:
            reasons[str(reason)] = reasons.get(str(reason), 0) + 1
    assert reasons == dict(counts.reasons)
    # one anchor per session, the run of the design, chosen among every run
    for session in bench.ds.sessions:
        label = session.session
        anchors = bench.rows(
            "SELECT r.name FROM func_runs r JOIN func_sessions s USING (session_uid) "
            "JOIN func_participants p USING (participant_uid) "
            "WHERE p.participant = ? AND s.session_label IS ? AND r.anchor = 1",
            session.participant,
            label,
        )
        stem = session.anchor.partition("_")[2]
        assert anchors == [(stem,)], session
        flagged = bench.rows(
            "SELECT r.name FROM func_runs r JOIN func_sessions s USING (session_uid) "
            "JOIN func_participants p USING (participant_uid) "
            "WHERE p.participant = ? AND s.session_label IS ? AND r.flagged = 1 ORDER BY r.name",
            session.participant,
            label,
        )
        assert [f for (f,) in flagged] == sorted(s.partition("_")[2] for s in session.flagged)
    # the FreeSurfer subject of each session, by its subject_uid in the surface catalog
    linked = bench.rows(
        "SELECT s.fs_subject, s.subject_uid = sub.subject_uid FROM func_sessions s "
        "JOIN subjects sub ON sub.subject_label = s.fs_subject"
    )
    assert sorted(linked) == [("sub-01", 1), ("sub-01", 1), ("sub-02", 1)]
    affines = dict(
        (str(row[0]), row[1])
        for row in bench.catalog.query("SELECT name, raw_affine FROM func_runs WHERE packable = 1")
    )
    expected = {r.name: r.affine_choice for r in bench.ds.runs if r.packable}
    assert affines == expected


def test_the_lease_of_a_session_gives_what_the_pack_phase_needs(bench: Bench) -> None:
    bench.func()
    settings = epi_settings(bench.config)
    packable = dict(
        (str(row[0]), int(row[1]))
        for row in bench.catalog.query(
            "SELECT session_uid, n_packable FROM func_sessions ORDER BY session_uid"
        )
    )
    for session_uid, n in packable.items():
        facts = bench.catalog.func_lease_facts(session_uid)
        work = inventory.session_work(facts, settings)
        assert work.session_uid == session_uid
        assert len(work.inputs) == n and not work.input_failures
        assert {r.run for r in work.plan.runs} <= set(facts.run_ids.values())
        assert work.contours is not None and work.contour_note.startswith("contours from")
    # a file that changed since the inventory: nothing is packed from it
    run = next(r for r in bench.ds.runs if r.design.participant == "sub-02" and r.packable)
    os.utime(run.files["confounds"], ns=(1, 1))
    changed = bench.rows("SELECT session_uid FROM func_sessions WHERE session_label IS NULL")[0][0]
    with pytest.raises(SourceError) as err:
        inventory.session_work(bench.catalog.func_lease_facts(str(changed)), settings)
    assert err.value.reason == "changed"


def test_nothing_is_written_into_the_three_folders(bench: Bench) -> None:
    before = snapshot_links(bench.ds.root)
    bench.func()
    for row in bench.catalog.query("SELECT session_uid FROM func_sessions"):
        inventory.session_work(bench.catalog.func_lease_facts(row[0]), EpiConfig())
    inventory.func_fetch_lines(bench.catalog)
    bench.func(force=True)
    assert snapshot_links(bench.ds.root) == before


def test_no_volume_of_a_series_is_read(bench: Bench, monkeypatch: pytest.MonkeyPatch) -> None:
    read: list[str] = []
    proxy = nib.arrayproxy.ArrayProxy

    def watch(name: str) -> object:
        real = getattr(proxy, name)

        def wrapper(self: object, *args: object, **kwargs: object) -> object:
            path = str(getattr(self, "file_like", ""))
            if path.endswith(("_bold.nii.gz", "_bold.nii")):
                read.append(path)
            return real(self, *args, **kwargs)

        return wrapper

    for name in ("__getitem__", "__array__", "get_unscaled"):
        monkeypatch.setattr(proxy, name, watch(name))
    bench.func()
    assert read == []
    # the watch sees a read of a series: one volume through the proxy
    series = next(r for r in bench.ds.runs if r.packable).files["pre_bold"]
    nib.load(str(series)).dataobj[..., 0]
    assert read == [str(series)]


# --- the skip -------------------------------------------------------------------------------


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every path opened through ``open``, ``io.open`` and ``os.open`` while the test runs."""
    seen: list[str] = []
    real_open, real_io_open, real_os_open = builtins.open, io.open, os.open

    def note(file: object) -> None:
        if isinstance(file, (str, bytes, os.PathLike)):
            seen.append(os.fsdecode(file))

    def b_open(file: object, *args: object, **kwargs: object) -> object:
        note(file)
        return real_open(file, *args, **kwargs)  # pyrefly: ignore[no-matching-overload]

    def i_open(file: object, *args: object, **kwargs: object) -> object:
        note(file)
        return real_io_open(file, *args, **kwargs)  # pyrefly: ignore[no-matching-overload]

    def o_open(file: object, *args: object, **kwargs: object) -> int:
        note(file)
        return real_os_open(file, *args, **kwargs)  # pyrefly: ignore[bad-argument-type]

    monkeypatch.setattr(builtins, "open", b_open)
    monkeypatch.setattr(io, "open", i_open)
    monkeypatch.setattr(os, "open", o_open)
    return seen


def _of_sessions(paths: list[str], ds: SyntheticFuncDataset) -> list[str]:
    """The opened paths that are files of a session: under ``sub-*`` of the raw dataset, the
    derivative or the FreeSurfer subjects directory."""
    roots = [str(ds.raw_root), str(ds.deriv_root), str(ds.fs_subjects_dir)]
    return [p for p in paths if any(p.startswith(f"{r}/sub-") for r in roots)]


def test_a_second_inventory_opens_no_file_of_an_unchanged_session(bench: Bench) -> None:
    bench.func()
    seen: list[str] = []
    real_open, real_io_open = builtins.open, io.open

    def b_open(file: object, *args: object, **kwargs: object) -> object:
        seen.append(os.fsdecode(file))  # pyrefly: ignore[bad-argument-type]
        return real_open(file, *args, **kwargs)  # pyrefly: ignore[no-matching-overload]

    def i_open(file: object, *args: object, **kwargs: object) -> object:
        seen.append(os.fsdecode(file))  # pyrefly: ignore[bad-argument-type]
        return real_io_open(file, *args, **kwargs)  # pyrefly: ignore[no-matching-overload]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builtins, "open", b_open)
        patch.setattr(io, "open", i_open)
        started = time.perf_counter()
        report = bench.func()
        seconds = time.perf_counter() - started
    assert (report.done, report.skipped) == (0, 3)
    assert _of_sessions(seen, bench.ds) == []
    assert seconds < 5


def test_the_watch_of_opened_files_sees_a_planned_session(bench: Bench, opened: list[str]) -> None:
    # the counterpart of the test above: a first inventory opens the files of its sessions
    bench.func()
    assert _of_sessions(opened, bench.ds)


def test_a_changed_file_inventories_its_session_again_and_no_other(bench: Bench) -> None:
    bench.func()
    run = next(r for r in bench.ds.runs if r.design.participant == "sub-02" and r.packable)
    os.utime(run.files["raw_sidecar"], ns=(10**9, 10**9))
    report = bench.func()
    assert (report.done, report.skipped) == (1, 2)
    # a FreeSurfer file of the contours is part of the sessions of its subject
    os.utime(bench.ds.fs_subjects_dir / "sub-01" / "surf" / "lh.white", ns=(10**9, 10**9))
    report = bench.func()
    assert (report.done, report.skipped) == (2, 1)
    assert (bench.func(force=True).done, bench.func().skipped) == (3, 3)


def test_a_session_that_leaves_is_marked_absent_and_comes_back(
    tmp_path: Path, bench: Bench
) -> None:
    bench.func()
    away = tmp_path / "away"
    away.mkdir()
    moved = [
        (bench.ds.raw_root / "sub-01" / "ses-2", away / "raw-ses-2"),
        (bench.ds.deriv_root / "sub-01" / "ses-2", away / "deriv-ses-2"),
    ]
    for here, there in moved:
        shutil.move(str(here), str(there))
    report = bench.func()
    assert (report.done, report.skipped) == (0, 2)
    assert any("no longer listed" in note for note in report.notes)
    present = dict(
        (str(row[0]), int(row[1]))
        for row in bench.catalog.query("SELECT session_label, present FROM func_sessions")
    )
    assert present == {"1": 1, "2": 0, "None": 1}
    runs = "SELECT r.present FROM func_runs r JOIN func_sessions s USING (session_uid) "
    assert set(bench.rows(runs + "WHERE s.session_label = '2'")) == {(0,)}
    for here, there in moved:
        shutil.move(str(there), str(here))
    report = bench.func()
    assert (report.done, report.skipped, report.notes) == (0, 3, ())
    assert set(bench.rows("SELECT present FROM func_sessions")) == {(1,)}
    assert set(bench.rows("SELECT present FROM func_runs")) == {(1,)}


# --- the settings ---------------------------------------------------------------------------


def test_runs_left_out_by_epi_runs_stay_listed_and_the_anchor_does_not_move(
    bench: Bench,
) -> None:
    bench.func()
    before = _runs(bench)
    anchors = set(bench.rows("SELECT session_uid, anchor_run_uid FROM func_sessions"))
    bench.with_epi(runs=("*task-motor*",))
    report = bench.func()
    assert report.done == 3  # the globs are part of the inventory hash
    after = _runs(bench)
    assert set(after) == set(before)
    for name, row in after.items():
        reason, packable, selected, anchor, order, present = row
        if "task-motor" in name:
            assert row == before[name], name
        else:
            assert (reason, packable, selected) == (NOT_SELECTED, 0, 0), name
        # the anchor and the order of a session follow every run, not the selection
        assert (anchor, order, present) == before[name][3:], name
    assert set(bench.rows("SELECT session_uid, anchor_run_uid FROM func_sessions")) == anchors
    detail = bench.rows("SELECT detail FROM func_runs WHERE reason = ?", NOT_SELECTED)[0][0]
    assert detail == "--epi-runs leaves it out (*task-motor*)"


def test_no_epi_and_a_census_run_do_nothing(bench: Bench) -> None:
    bench.with_epi(enabled=False)
    report = bench.func()
    assert (report.done, report.skipped, report.failed) == (0, 0, 0)
    assert bench.rows("SELECT COUNT(*) FROM func_sources") == [(0,)]
    # a census run does not run the phase even with the settings of a local run (model_copy
    # does not validate, so the enabled section a census config refuses gets through here)
    census = bench.config.model_copy(update={"mode": "census", "epi": EpiConfig()})
    assert bench.func(config=census).done == 0
    assert bench.rows("SELECT COUNT(*) FROM func_sessions") == [(0,)]


def test_raw_given_is_stored_and_found_without_it_is_named(tmp_path: Path) -> None:
    ds = write_func_dataset(tmp_path / "data", raw_inside=False)
    for bench in _bench(tmp_path, ds, ds.deriv_root):
        bench.func()
        (found,) = bench.rows("SELECT raw_root, raw_found FROM func_sources")
        assert found == (None, None)
        assert bench.rows("SELECT COUNT(*) FROM func_runs WHERE packable = 1") == [(0,)]
        details = bench.rows("SELECT DISTINCT detail FROM func_runs WHERE reason = ?", NO_RAW)
        hint = "no raw BIDS dataset was found for the derivative; give it with --raw DIR"
        assert details == [(hint,)]
        bench.with_epi(raw=str(ds.raw_root))
        report = bench.func()
        assert report.done == 3  # the raw files are in the listing of every session
        (found,) = bench.rows("SELECT raw_root, raw_found FROM func_sources")
        assert found == (str(ds.raw_root), "given")
        packable = bench.rows("SELECT COUNT(*) FROM func_runs WHERE packable = 1")
        assert packable == [(func_counts(ds).packable,)]


def test_a_derivative_that_cannot_be_packed_lists_its_reasons(tmp_path: Path) -> None:
    ds = write_review_folder(tmp_path / "data")
    for series in ds.deriv_root.glob("sub-*/**/func/*_desc-preproc_bold.nii.gz"):
        series.unlink()
    for bench in _bench(tmp_path, ds):
        report = bench.func()
        assert (report.done, report.failed) == (3, 0)
        assert bench.rows("SELECT COUNT(*) FROM func_runs WHERE packable = 1") == [(0,)]
        reasons = {str(r) for (r,) in bench.rows("SELECT DISTINCT reason FROM func_runs")}
        assert "no preprocessed series" in reasons and None not in reasons
        assert set(bench.rows("SELECT n_packable FROM func_sessions")) == {(0,)}


def test_a_session_that_cannot_be_planned_is_a_failed_unit(tmp_path: Path) -> None:
    ds = write_review_folder(tmp_path / "data")
    run = next(r for r in ds.runs if r.design.participant == "sub-02" and r.packable)
    run.files["pre_mask"].write_bytes(b"not a NIfTI file")
    for bench in _bench(tmp_path, ds):
        report = bench.func()
        assert (report.done, report.failed) == (3, 1)
        (unit, reason) = report.failures[0]
        assert unit.startswith("session:") and reason == "unreadable"
        (problem,) = bench.rows("SELECT problem FROM func_sessions WHERE session_label IS NULL")
        assert str(problem[0]).startswith("unreadable: the session cannot be planned:")
        assert str(tmp_path) not in str(problem[0])


# --- placeholders and the fetch lines -------------------------------------------------------


@pytest.mark.parametrize("kind", ["link", "pointer"])
def test_placeholders_and_the_command_that_fetches_them(tmp_path: Path, kind: str) -> None:
    ds = write_review_folder(tmp_path / "data", annex=kind)  # pyrefly: ignore[bad-argument-type]
    for bench in _bench(tmp_path, ds):
        bench.func()
        runs = _runs(bench)
        assert runs["sub-01 ses-2_task-rest_run-2"][:3] == (NOT_FETCHED, 0, 1)
        (problem,) = bench.rows("SELECT problem FROM func_sessions WHERE session_label = '2'")
        assert str(problem[0]).startswith("not fetched: ")
        lines = inventory.func_fetch_lines(bench.catalog)
        assert lines[0].startswith("sub-01 ses-2_task-rest_run-2: not fetched: ")
        commands = [line for line in lines if line.startswith("Fetch the functional files")]
        if kind == "link":
            # the raw series in the raw dataset, the rest in the derivative: two datasets
            assert len(commands) == 2
            assert f"datalad -C {ds.deriv_root} get " in commands[0]
            assert f"datalad -C {ds.raw_root} get sub-01/ses-2/func/" in commands[1]
        else:
            assert len(commands) == 1 and f"datalad -C {ds.deriv_root} get " in commands[0]
        named = {p.path.relative_to(ds.raw_root).as_posix() for p in ds.placeholders}
        for placeholder in named:
            assert any(placeholder.rsplit("/", 1)[1] in c for c in commands), placeholder


def test_no_fetch_line_without_placeholders(bench: Bench) -> None:
    bench.func()
    assert inventory.func_fetch_lines(bench.catalog) == []
