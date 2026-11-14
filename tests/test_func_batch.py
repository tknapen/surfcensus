"""The pack phase of a run (``func.batch``; docs/epi_interface.md, "Step 2: the plan", work
package E2.C): the skip rule from the catalog and the run directory, the order of the writes,
stale packs, failures, the machine, workers, and the stream behind the app.

The tests run on the fixture of one folder with both interfaces
(``synth_func.write_review_folder``: 3 sessions, 19 runs, 11 packable) through a real ``Run``,
its catalog (E2.A) and the functional inventory and lease of E2.B
(``func.inventory.run_func_inventory``, ``session_work``). What ``Run`` will bind into its
context (``RunContext.func_lease`` and ``pack_key``, work package E2.Z) is bound here by
:class:`Bench` in the same way: the lease is ``session_work`` on the catalog's
``func_lease_facts``, the pack key ``privacy.pack_key`` under the run secret.
"""

from __future__ import annotations

import dataclasses
import errno
import hashlib
import os
import shutil
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from synth_func import write_review_folder  # pyrefly: ignore[missing-import]

import surfcensus as sc
from surfcensus import privacy
from surfcensus.app.epi import Packs
from surfcensus.config import ConfigError, EpiConfig, RunConfig, epi_settings, run_key
from surfcensus.func import STREAM_JSON, packread
from surfcensus.func import batch as fb
from surfcensus.func import inventory as finv
from surfcensus.func import pack as pack_
from surfcensus.func.model import StreamInfo, read_info
from surfcensus.records import (
    EPI_SESSION_PHASE,
    PACK_FAILURE,
    SOURCE_CHANGED,
    PhaseReport,
    RunContext,
    SessionWork,
    session_unit,
)
from surfcensus.run import Run

# --- the bench ----------------------------------------------------------------------------------


@dataclass
class Bench:
    """A run of the fixture with its surfaces and functional data inventoried."""

    root: Path
    run: Run
    config: RunConfig
    leased: list[str] = field(default_factory=list)
    alter: Callable[[str, SessionWork], SessionWork] | None = None
    """Changes what the lease of a session hands the worker (a stand-in for a fault)."""

    @property
    def ds(self) -> Path:
        return self.root / "ds"

    @property
    def deriv(self) -> Path:
        return self.ds / "derivatives" / "fmriprep"

    @property
    def run_dir(self) -> Path:
        return self.run.run_dir

    def inventory(self, **epi: Any) -> PhaseReport:
        if epi:
            self.config = self.config.model_copy(
                update={"epi": epi_settings(self.config).model_copy(update=epi)}
            )
        catalog = self.run._catalog
        self.run.inventory()
        return finv.run_func_inventory(
            catalog, self.run._sources(), self.config, run_key(self.config)
        )

    @contextmanager
    def lease(self, session_uid: str) -> Iterator[SessionWork]:
        self.leased.append(session_uid)
        facts = self.run._catalog.func_lease_facts(session_uid)
        work = finv.session_work(facts, epi_settings(self.config))
        yield work if self.alter is None else self.alter(session_uid, work)

    def ctx(self) -> RunContext:
        catalog = self.run._catalog
        secret = catalog.run_secret()
        return dataclasses.replace(
            self.run.context(),
            config=self.config,
            func_lease=self.lease,
            pack_key=lambda pack_hash, digests: privacy.pack_key(secret, pack_hash, digests),
        )

    def pack(self, **kwargs: Any) -> PhaseReport:
        return fb.run_pack(self.ctx(), **kwargs)

    # reading what the phase made

    def runs(self) -> list[dict[str, Any]]:
        return self.ctx().public.func_runs().to_dict("records")

    def packable(self, session_uid: str | None = None) -> list[str]:
        return [
            r["run_uid"]
            for r in self.runs()
            if r["packable"] and r["present"] and session_uid in (None, r["session_uid"])
        ]

    def sessions(self) -> dict[str, dict[str, Any]]:
        frame = self.ctx().public.func_sessions()
        return {row["session_uid"]: row for row in frame.to_dict("records")}

    def session_of(self, participant: str, label: str | None) -> str:
        names = {r["participant_uid"]: r["participant"] for r in self.participants()}
        for uid, row in self.sessions().items():
            if names[row["participant_uid"]] == participant and row["session_label"] == label:
                return uid
        raise KeyError((participant, label))

    def participants(self) -> list[dict[str, Any]]:
        return self.ctx().public.func_participants().to_dict("records")

    def session_dir(self, session_uid: str) -> Path:
        return self.run_dir / "epi" / self.sessions()[session_uid]["participant_uid"] / session_uid

    def rows(self) -> dict[str, dict[str, Any]]:
        frame = self.ctx().public.packs()
        return {row["run_uid"]: row for row in frame.to_dict("records")}

    def files(self, run_uid: str, role: str) -> Path:
        (row,) = self.run._catalog.query(
            "SELECT base, location FROM func_files WHERE owner_kind = 'run' AND owner = ? "
            "AND role = ?",
            (run_uid, role),
        )
        return (self.ds if row["base"] == "raw" else self.deriv) / row["location"]

    def anchor_of(self, session_uid: str) -> str:
        return self.sessions()[session_uid]["anchor_run_uid"]


@pytest.fixture(scope="module")
def folder_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("folder")
    write_review_folder(root)
    return root


@pytest.fixture
def new_bench(tmp_path: Path, folder_template: Path) -> Iterator[Callable[..., Bench]]:
    benches: list[Bench] = []

    def make(name: str = "study", *, template: Path | None = None) -> Bench:
        root = tmp_path / name
        shutil.copytree(template or folder_template, root, symlinks=True)
        run = Run(root / "run", source=sc.sources.local(root / "ds"))
        bench = Bench(root, run, run.config)
        benches.append(bench)
        bench.inventory()
        return bench

    yield make
    for bench in benches:
        bench.run.close()


@pytest.fixture
def bench(new_bench: Callable[..., Bench]) -> Bench:
    return new_bench()


def tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def mtimes(root: Path) -> dict[str, int]:
    return {
        str(p.relative_to(root)): p.stat().st_mtime_ns
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def touch(path: Path) -> None:
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))


N_PACKABLE = 11


# --- the plan and the skip rule ---------------------------------------------------------------


def test_a_run_is_packed_and_a_second_call_does_nothing_and_opens_no_source(
    bench: Bench, tmp_path: Path
) -> None:
    ctx = bench.ctx()
    order = fb.session_order(ctx)
    assert fb.pack_pending(ctx) == order and len(order) == 3
    assert fb.pack_status(ctx) == (0, 0, N_PACKABLE)
    assert fb.pack_estimate(ctx) > 0

    report = bench.pack()
    assert (report.done, report.skipped, report.failed) == (N_PACKABLE, 0, 0)
    assert bench.leased == order
    rows = bench.rows()
    assert sorted(rows) == sorted(bench.packable())
    for uid, row in rows.items():
        session = row["session_uid"]
        participant = bench.sessions()[session]["participant_uid"]
        assert row["status"] == "done" and row["pack_key"] and row["n_volumes"] > 0
        assert row["path"] == f"epi/{participant}/{session}/{uid}"
        assert packread.pack_on_disk(bench.run_dir / "epi" / participant / session, uid)
    work = ctx.public.work(EPI_SESSION_PHASE)
    assert sorted(work["unit"]) == sorted(session_unit(s) for s in order)
    assert set(work["status"]) == {"done"}
    # the display cache holds no image volume: only what a pack and a stream write
    files = [p for p in (bench.run_dir / "epi").rglob("*") if p.is_file()]
    assert files and {p.suffix for p in files} == {".json", ".bin"}

    before = mtimes(bench.run_dir / "epi")
    again = bench.pack()
    assert (again.done, again.skipped, again.failed) == (0, N_PACKABLE, 0)
    assert bench.leased == order  # no lease: no source file opened
    assert mtimes(bench.run_dir / "epi") == before

    # with the data unmounted
    shutil.move(bench.ds, tmp_path / "away")
    ctx = bench.ctx()
    assert fb.pack_pending(ctx) == []
    assert fb.pack_status(ctx) == (N_PACKABLE, 0, 0)
    assert fb.pack_failures(ctx) == {} and fb.pack_estimate(ctx) == 0
    third = bench.pack()
    assert (third.done, third.skipped) == (0, N_PACKABLE)
    assert mtimes(bench.run_dir / "epi") == before


def test_the_display_cache_names_nothing_but_identifiers(bench: Bench) -> None:
    bench.pack()
    for path in (bench.run_dir / "epi").rglob("*"):
        assert "sub-" not in path.name and "task" not in path.name
    for participant in bench.participants():
        stream = read_info(
            bench.run_dir / "epi" / participant["participant_uid"] / STREAM_JSON, StreamInfo
        )
        assert stream is not None and stream.label == participant["participant"]
        sessions = [
            uid
            for uid, row in sorted(bench.sessions().items(), key=lambda kv: kv[1]["stream_order"])
            if row["participant_uid"] == participant["participant_uid"]
        ]
        assert stream.sessions == tuple(sessions) and stream.order == "session label"


def test_a_touched_input_repacks_its_run_only(bench: Bench) -> None:
    bench.pack()
    session = bench.session_of("sub-01", "1")
    anchor = bench.anchor_of(session)
    other = next(r for r in bench.packable(session) if r != anchor)
    digests = {
        r: pack_.content_digest(bench.session_dir(session), packread.read_pack(bench.session_dir(session), r))  # type: ignore[arg-type]
        for r in bench.packable(session)
    }
    before = mtimes(bench.run_dir / "epi")
    touch(bench.files(other, "confounds"))
    bench.inventory()
    plan = fb.pack_plan(bench.ctx())
    assert plan.pending == (session,)
    assert plan.keep[session] == frozenset(bench.packable(session)) - {other}

    report = bench.pack()
    assert (report.done, report.skipped) == (1, N_PACKABLE - 1)
    after = mtimes(bench.run_dir / "epi")
    changed = {name for name in after if before.get(name) != after[name]}
    # the run's pack and the session.json of its session; nothing of the other runs
    assert all(
        name.startswith(f"{bench.sessions()[session]['participant_uid']}/{session}/")
        for name in changed
    )
    assert {n.split("/")[2] for n in changed} <= {other, "session.json"}
    assert any(n.startswith(f"{bench.sessions()[session]['participant_uid']}/{session}/{other}/") for n in changed)
    # the same content: the same digest
    info = packread.read_pack(bench.session_dir(session), other)
    assert info is not None and pack_.content_digest(bench.session_dir(session), info) == digests[other]


def test_a_changed_anchor_transform_repacks_its_whole_session(bench: Bench) -> None:
    bench.pack()
    session = bench.session_of("sub-01", "1")
    anchor = bench.anchor_of(session)
    touch(bench.files(anchor, "xfm_to_anat"))
    bench.inventory()
    plan = fb.pack_plan(bench.ctx())
    assert plan.pending == (session,) and plan.keep[session] == frozenset()
    report = bench.pack()
    assert report.done == len(bench.packable(session))
    assert report.skipped == N_PACKABLE - report.done


def test_a_pack_that_lost_a_file_is_pending_and_is_made_again(bench: Bench) -> None:
    bench.pack()
    session = bench.session_of("sub-02", None)
    run = bench.packable(session)[0]
    (bench.session_dir(session) / run / "pre" / "c0000.bin").unlink()
    ctx = bench.ctx()
    assert fb.pack_pending(ctx) == [session]
    assert fb.pack_status(ctx) == (N_PACKABLE - 1, 0, 1)
    report = bench.pack()
    assert (report.done, report.skipped) == (1, N_PACKABLE - 1)
    assert packread.pack_on_disk(bench.session_dir(session), run)

    # a session.json that is gone: the session is written again, its packs are kept
    (bench.session_dir(session) / "session.json").unlink()
    assert fb.pack_pending(bench.ctx()) == [session]
    report = bench.pack()
    assert (report.done, report.skipped) == (0, N_PACKABLE)
    assert (bench.session_dir(session) / "session.json").is_file()


def test_a_run_that_is_no_longer_selected_loses_its_pack_and_then_its_row(
    bench: Bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench.pack()
    session = bench.session_of("sub-01", "1")
    names = {r["run_uid"]: r["name"] for r in bench.runs()}
    gone = next(r for r in bench.packable(session) if "motor" in names[r])
    kept = [r for r in bench.packable() if r != gone]
    bench.inventory(runs=("*rest*", "*sub-01_ses-2*", "*sub-02*"))
    assert gone not in bench.packable() and sorted(bench.packable()) == sorted(kept)

    public = bench.ctx().public
    forgotten: list[str] = []
    original = type(public).forget_packs

    def forget(self: Any, run_uids: Any) -> None:
        # outputs first, rows last: the pack is gone when its row goes
        assert not (bench.session_dir(session) / gone).exists()
        forgotten.extend(run_uids)
        original(self, run_uids)

    monkeypatch.setattr(type(public), "forget_packs", forget)
    report = bench.pack()
    assert (report.done, report.skipped) == (0, N_PACKABLE - 1)
    assert forgotten == [gone]
    assert gone not in bench.rows()
    info = packread.read_session(bench.session_dir(session))
    assert info is not None
    assert {r.run: r.reason for r in info.runs}[gone] == "not selected"
    for run in kept:
        assert bench.rows()[run]["status"] == "done"


def test_a_participant_that_leaves_loses_its_directory_and_its_rows(bench: Bench) -> None:
    bench.pack()
    (participant,) = [p["participant_uid"] for p in bench.participants() if p["participant"] == "sub-02"]
    session = bench.session_of("sub-02", None)
    assert (bench.run_dir / "epi" / participant).is_dir()
    shutil.rmtree(bench.deriv / "sub-02")
    shutil.rmtree(bench.ds / "sub-02")
    bench.inventory()
    plan = fb.pack_plan(bench.ctx())
    assert session_unit(session) in plan.stale and plan.pending == ()
    report = bench.pack()
    assert report.done == 0
    assert not (bench.run_dir / "epi" / participant).exists()
    assert all(row["session_uid"] != session for row in bench.rows().values())
    assert session_unit(session) not in set(bench.ctx().public.work(EPI_SESSION_PHASE)["unit"])
    # the other participant is untouched
    assert len(bench.rows()) == len(bench.packable())


# --- failures ---------------------------------------------------------------------------------


def test_a_failed_pack_is_recorded_and_not_tried_again_without_retry_failed(
    bench: Bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = bench.session_of("sub-01", "2")
    victim = bench.packable(session)[0]
    original = pack_.pack_run

    def breaks(session_dir: Path, run: Any, config: EpiConfig) -> Any:
        if run.run == victim:
            raise pack_.PackError(PACK_FAILURE, f"broken under {bench.root}/secret/place")
        return original(session_dir, run, config)

    with monkeypatch.context() as patched:
        patched.setattr(pack_, "pack_run", breaks)
        report = bench.pack()
    assert (report.done, report.failed) == (N_PACKABLE - 1, 1)
    assert report.failures == ((victim, PACK_FAILURE),)
    row = bench.rows()[victim]
    assert row["status"] == "failed" and row["reason"] == PACK_FAILURE
    assert str(bench.root) not in row["detail"] and "place" in row["detail"]
    assert fb.pack_failures(bench.ctx()) == {victim: (PACK_FAILURE, row["detail"])}
    assert fb.pack_status(bench.ctx()) == (N_PACKABLE - 1, 1, 0)
    info = packread.read_session(bench.session_dir(session))
    assert info is not None and {r.run: r.reason for r in info.runs}[victim] == PACK_FAILURE

    leased = len(bench.leased)
    again = bench.pack()
    assert (again.done, again.skipped, again.failed) == (0, N_PACKABLE - 1, 1)
    assert len(bench.leased) == leased  # not tried again

    retried = bench.pack(retry_failed=True)
    assert (retried.done, retried.failed) == (1, 0)
    assert bench.rows()[victim]["status"] == "done"

    forced = bench.pack(force=True)
    assert (forced.done, forced.skipped) == (N_PACKABLE, 0)


def test_files_that_changed_since_the_inventory_are_recorded_as_source_changed(
    bench: Bench,
) -> None:
    session = bench.session_of("sub-02", None)
    run = bench.packable(session)[0]
    touch(bench.files(run, "confounds"))  # no inventory: the lease finds another listing
    report = bench.pack()
    assert report.failed == len(bench.packable(session))
    assert {bench.rows()[r]["reason"] for r in bench.packable(session)} == {SOURCE_CHANGED}
    assert report.done == N_PACKABLE - report.failed
    # not tried again until the inventory saw the change
    assert bench.pack().failed == report.failed
    bench.inventory()
    final = bench.pack()
    assert (final.done, final.failed) == (len(bench.packable(session)), 0)


class _Exploding:
    """A stand-in for an input whose use in a worker meets a full disk."""

    def __init__(self, code: int, filename: str | None) -> None:
        self.code, self.filename = code, filename

    def __getattr__(self, name: str) -> Any:
        if name in ("code", "filename") or name.startswith("__"):
            raise AttributeError(name)
        raise OSError(self.code, os.strerror(self.code), self.filename)


class _Killer:
    """Ends the process that unpickles it: a worker that dies."""

    def __reduce__(self) -> tuple[Any, tuple[int]]:
        return os._exit, (3,)


@pytest.mark.parametrize("code", [errno.ENOSPC, errno.EROFS, errno.EDQUOT, errno.EACCES])
def test_a_display_cache_that_cannot_be_written_stops_the_phase_and_records_nothing_of_the_batch(
    bench: Bench, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    order = fb.session_order(bench.ctx())
    second = order[1]
    original = pack_.pack_run

    def no_room(session_dir: Path, run: Any, config: EpiConfig) -> Any:
        if Path(session_dir).name == second:
            raise OSError(code, os.strerror(code), str(session_dir / "x"))
        return original(session_dir, run, config)

    with monkeypatch.context() as full:
        full.setattr(pack_, "pack_run", no_room)
        with pytest.raises(ConfigError, match="the EPI packs cannot be written") as raised:
            bench.pack()
    assert raised.value.reason == "invalid" and isinstance(raised.value.__cause__, OSError)
    rows = bench.rows()
    assert {row["session_uid"] for row in rows.values()} == {order[0]}
    assert all(row["status"] == "done" for row in rows.values())
    units = set(bench.ctx().public.work(EPI_SESSION_PHASE)["unit"])
    assert units == {session_unit(order[0])}

    report = bench.pack()  # with room again the same call does the rest
    assert report.failed == 0 and report.done + report.skipped == N_PACKABLE
    assert report.skipped == len(bench.packable(order[0]))


def test_a_full_disk_in_a_worker_process_stops_the_phase(bench: Bench) -> None:
    order = fb.session_order(bench.ctx())

    def explode(session_uid: str, work: SessionWork) -> SessionWork:
        if session_uid != order[1]:
            return work
        return dataclasses.replace(work, inputs=(_Exploding(errno.ENOSPC, None),))  # type: ignore[arg-type]

    bench.alter = explode
    with pytest.raises(ConfigError, match="No space left on device"):
        bench.pack(jobs=2)
    assert bench.rows() == {}  # the first batch held both sessions: nothing recorded
    bench.alter = None
    report = bench.pack(jobs=2)
    assert (report.done, report.failed) == (N_PACKABLE, 0)


def test_a_worker_that_dies_fails_the_runs_of_its_session_only(bench: Bench) -> None:
    order = fb.session_order(bench.ctx())

    def kill(session_uid: str, work: SessionWork) -> SessionWork:
        if session_uid != order[0]:
            return work
        return dataclasses.replace(work, contour_note=_Killer())  # type: ignore[arg-type]

    bench.alter = kill
    report = bench.pack(jobs=2)
    dead = bench.packable(order[0])
    assert report.failed == len(dead)
    assert {bench.rows()[r]["reason"] for r in dead} == {PACK_FAILURE}
    assert all("BrokenProcessPool" in bench.rows()[r]["detail"] for r in dead)
    assert report.done == N_PACKABLE - len(dead)
    bench.alter = None
    assert bench.pack(jobs=2, retry_failed=True).done == len(dead)


def test_a_context_without_a_lease_is_refused_when_something_is_pending(bench: Bench) -> None:
    ctx = dataclasses.replace(bench.ctx(), func_lease=None)
    with pytest.raises(ConfigError, match="no lease"):
        fb.run_pack(ctx)
    assert bench.rows() == {}
    bench.pack()
    assert fb.run_pack(dataclasses.replace(bench.ctx(), func_lease=None)).skipped == N_PACKABLE


def test_without_the_epi_interface_nothing_is_packed_or_removed(bench: Bench) -> None:
    bench.pack()
    before = mtimes(bench.run_dir / "epi")
    bench.config = bench.config.model_copy(update={"epi": EpiConfig(enabled=False)})
    ctx = bench.ctx()
    assert fb.pack_pending(ctx) == [] and fb.pack_status(ctx) == (0, 0, 0)
    report = fb.run_pack(ctx)
    assert (report.done, report.skipped, report.failed) == (0, 0, 0)
    assert mtimes(bench.run_dir / "epi") == before


# --- order, stop, the stream behind the app -------------------------------------------------------


def test_sessions_go_in_the_order_of_the_queue_then_participants_without_surfaces(
    new_bench: Callable[..., Bench], tmp_path: Path
) -> None:
    template = tmp_path / "template"
    write_review_folder(template, without_fs=("sub-01",))
    bench = new_bench("alone", template=template)
    order = fb.session_order(bench.ctx())
    sessions = bench.sessions()
    assert [sessions[s]["subject_uid"] is None for s in order] == [False, True, True]
    sub01 = [s for s in order if sessions[s]["subject_uid"] is None]
    assert [sessions[s]["session_label"] for s in sub01] == ["1", "2"]


def test_stop_between_batches_and_the_stream_of_a_half_packed_participant(bench: Bench) -> None:
    order = fb.session_order(bench.ctx())
    sessions = bench.sessions()
    first = order[0]
    participant = sessions[first]["participant_uid"]
    asked = []

    def stop() -> bool:
        asked.append(1)
        return len(asked) > 1

    report = bench.pack(jobs=1, stop=stop)
    assert report.done == len(bench.packable(first)) and bench.leased == [first]
    stream = read_info(bench.run_dir / "epi" / participant / STREAM_JSON, StreamInfo)
    mine = [s for s in order if sessions[s]["participant_uid"] == participant]
    assert stream is not None and list(stream.sessions) == mine
    # the page shows the sessions that have a session.json, and grows as they come
    shape = Packs(bench.run_dir / "epi" / participant).shape()
    assert [s.session_uid for s in shape.sessions] == [first]
    assert fb.pack_pending(bench.ctx()) == order[1:]

    rest = bench.pack()
    assert (rest.done, rest.skipped) == (N_PACKABLE - report.done, report.done)
    shape = Packs(bench.run_dir / "epi" / participant).shape()
    assert [s.session_uid for s in shape.sessions] == mine


# --- workers, digests -----------------------------------------------------------------------------


def test_one_process_and_two_give_the_same_files_and_rows(
    new_bench: Callable[..., Bench], tmp_path: Path
) -> None:
    bench = new_bench()
    bench.run.close()
    shutil.copytree(bench.run_dir, tmp_path / "run2")
    bench.run = Run(bench.run_dir)
    twin = Bench(bench.root, Run(tmp_path / "run2"), bench.config)
    try:
        alone = bench.pack(jobs=1)
        pair = twin.pack(jobs=2)
        assert dataclasses.replace(pair, seconds=0.0) == dataclasses.replace(alone, seconds=0.0)
        assert tree(twin.run_dir / "epi") == tree(bench.run_dir / "epi")

        def comparable(b: Bench) -> list[tuple[Any, ...]]:
            return [
                tuple(r)
                for r in b.run._catalog.query(
                    "SELECT run_uid, session_uid, config_hash, input_signature, pack_key, "
                    "status, path, n_volumes, bytes, content_digest, anchor_run_uid, grid_key "
                    "FROM packs ORDER BY run_uid"
                )
            ]

        rows = comparable(bench)
        assert comparable(twin) == rows and len(rows) == N_PACKABLE
        assert all(r[9] for r in rows)
    finally:
        twin.run.close()


def test_the_digests_of_what_a_pack_read_end_up_in_the_catalog(bench: Bench) -> None:
    query = bench.run._catalog.query
    assert all(r["sha256"] is None for r in query("SELECT sha256 FROM func_files"))
    bench.pack()
    for run in bench.packable():
        row = bench.rows()[run]
        info = packread.read_pack(bench.run_dir / row["path"].rsplit("/", 1)[0], run)
        assert info is not None
        roles = {e.role for e in info.inputs if e.base != "freesurfer"} - {
            "anchor_xfm_to_anat",
            "xfm_anat_fsnative",
        }
        for role in roles:
            path = bench.files(run, role)
            (stored,) = query(
                "SELECT sha256 FROM func_files WHERE owner_kind = 'run' AND owner = ? "
                "AND role = ?",
                (run, role),
            )
            assert stored["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    for session, row in bench.sessions().items():
        stored = query(
            "SELECT sha256 FROM func_files WHERE owner_kind = 'session' AND owner = ?",
            (session,),
        )
        assert stored and all(r["sha256"] for r in stored)
        if row["subject_uid"] is not None:
            fs = query(
                "SELECT role, sha256 FROM files WHERE subject_uid = ? AND role IN "
                "('t1', 'lh_white', 'rh_white', 'lh_pial', 'rh_pial')",
                (row["subject_uid"],),
            )
            assert len(fs) == 5 and all(r["sha256"] for r in fs)


def test_pack_session_with_keep_takes_the_runs_named_from_disk(
    bench: Bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench.pack()
    session = bench.session_of("sub-01", "1")
    runs = bench.packable(session)
    calls: list[str] = []
    original = pack_.pack_run

    def counted(session_dir: Path, run: Any, config: EpiConfig) -> Any:
        calls.append(run.run)
        return original(session_dir, run, config)

    def hashed(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("current_pack is not asked when keep is given")

    monkeypatch.setattr(pack_, "pack_run", counted)
    monkeypatch.setattr(pack_, "current_pack", hashed)
    with bench.lease(session) as work:
        result = fb.pack_session_unit(
            session,
            work,
            epi_dir=bench.run_dir / "epi",
            config=epi_settings(bench.config),
            keep={session: frozenset(runs[1:])},
        )
    assert calls == [runs[0]]
    assert sorted(result.packs) == sorted(runs) and result.kept == tuple(sorted(runs[1:]))
    assert result.failures == {}


def _counts_by(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows.values():
        out[row["status"]] = out.get(row["status"], 0) + 1
    return out
