"""``surfcensus review`` on one folder with both kinds of data (step 2 of the EPI interface,
work package E2.Z; docs/epi_interface.md, "Step 2 as built"): the surfaces are inventoried,
scored and rendered as before, the functional runs are listed after the surface inventory
and packed behind the app, after the remaining renders, and the app offers the EPI index
beside the rating screen.

The folder is ``synth_func.write_review_folder``: a BIDS root with raw functional data, an
fMRIPrep derivative below ``derivatives/`` and the FreeSurfer subjects ``sub-01`` and
``sub-02`` (3 sessions, 19 runs, 11 packable).
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from synth import snapshot_tree  # pyrefly: ignore[missing-import]
from synth_func import (  # pyrefly: ignore[missing-import]
    SyntheticFuncDataset,
    func_counts,
    write_func_dataset,
    write_review_folder,
)

pytest.importorskip("fastapi")
pytest.importorskip("sqlmodel")

from surfcensus import review as reviewing
from surfcensus.config import CONFIG_FILE, ConfigError, load_config
from surfcensus.func import batch as fbatch
from surfcensus.func.packread import read_pack
from surfcensus.privacy import HashKey
from surfcensus.review import (
    ReviewApp,
    open_review,
    pack_counts,
    rateable,
    render_counts,
    review,
    viewable,
)
from surfcensus.run import Run
from surfcensus.sources.local import local
from surfcensus.store import EPI_DIR


@pytest.fixture
def folder(tmp_path: Path) -> SyntheticFuncDataset:
    return write_review_folder(tmp_path / "data")


@pytest.fixture
def opened() -> Iterator[list[tuple[Run, ReviewApp | None]]]:
    made: list[tuple[Run, ReviewApp | None]] = []
    yield made
    for run, app in made:
        if app is not None:
            app.stop()
        run.close()


def _rows(run_dir: Path, sql: str) -> list[tuple[Any, ...]]:
    with sqlite3.connect(run_dir / "catalog.sqlite") as db:
        return db.execute(sql).fetchall()


def _packed_dirs(run_dir: Path) -> list[Path]:
    return sorted(p.parent for p in (run_dir / EPI_DIR).glob("*/*/*/pack.json"))


# --- both interfaces --------------------------------------------------------------------------


def test_review_prepares_both_interfaces_and_the_app_serves_both(
    folder: SyntheticFuncDataset,
    tmp_path: Path,
    opened: list[tuple[Run, ReviewApp | None]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    counts = func_counts(folder)
    before = snapshot_tree(folder.raw_root)
    run_dir = tmp_path / "run"
    run, app = open_review(folder.raw_root, run_dir=run_dir, jobs=1, port=0)
    opened.append((run, app))
    said = capsys.readouterr().err
    assert app is not None and app.epi and app.epi_url == f"{app.url.rstrip('/')}/epi"
    # the functional block names where, how many and why the others cannot be packed
    assert (
        f"Functional: {counts.runs} runs in 3 sessions of 2 participants in "
        "derivatives/fmriprep (fMRIPrep 21.0.2), raw series in . (BIDS); "
        f"{counts.packable} can be packed (T1w space), {counts.runs - counts.packable} cannot: "
    ) in said
    assert "flagged (raw data)" in said and "is the outlier (raw data)" in said
    assert f"EPI packs: 0 of {counts.packable} runs made; {counts.packable} are made" in said
    # the app opened at the first surface render, before any pack
    assert said.index("Functional:") < said.index("Scores:") < said.index("Rendered:")
    assert app.wait_rendered(180) and app.render_problem is None
    assert render_counts(run) == (2, 0, 0)
    assert pack_counts(run) == (counts.packable, 0, 0)
    assert len(_packed_dirs(run_dir)) == counts.packable
    assert "All EPI packs are made: 11 of 11 runs can be viewed." in capsys.readouterr().err

    with httpx.Client(base_url=app.url, trust_env=False) as client:
        index = client.get("/api/epi")
        assert index.status_code == 200
        participants = index.json()["participants"]
        assert [p["label"] for p in participants] == ["sub-01", "sub-02"]
        for p in participants:
            stream = client.get(f"/api/epi/{p['participant_uid']}")
            assert stream.status_code == 200
        # the rating routes stay blind: no run name, participant or functional identifier
        shown = json.dumps([client.get("/api/session").json(), client.get("/api/next").json()])
        names = _rows(run_dir, "SELECT name FROM func_runs")
        uids = [r[0] for r in _rows(run_dir, "SELECT run_uid FROM func_runs")]
        uids += [p["participant_uid"] for p in participants]
        for word in [n for (n,) in names] + uids + ["task-", "sub-01", "sub-02"]:
            assert word not in shown, word
    # nothing was written into the data
    assert snapshot_tree(folder.raw_root) == before
    # every file under epi/ is a display cache file
    assert {p.suffix for p in (run_dir / EPI_DIR).rglob("*") if p.is_file()} <= {".json", ".bin"}


def test_the_same_command_again_does_nothing_and_is_back_at_the_app_quickly(
    folder: SyntheticFuncDataset,
    tmp_path: Path,
    opened: list[tuple[Run, ReviewApp | None]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "run"
    review(folder.raw_root, run_dir=run_dir, jobs=1, serve=False).close()
    stamps = {p: p.stat().st_mtime_ns for p in (run_dir / EPI_DIR).rglob("*") if p.is_file()}
    capsys.readouterr()
    started = time.perf_counter()
    run, app = open_review(folder.raw_root, run_dir=run_dir, jobs=1, port=0)
    opened.append((run, app))
    seconds = time.perf_counter() - started
    assert app is not None and not app.rendering  # nothing behind the app
    said = capsys.readouterr().err
    assert "EPI packs: 11 of 11 runs made." in said
    assert {p: p.stat().st_mtime_ns for p in stamps} == stamps
    assert seconds < 30  # measured: about 1 s on one machine
    report = run.pack()
    assert (report.done, report.skipped, report.failed) == (0, 11, 0)


def test_behind_the_app_the_surfaces_come_first_then_the_packs(
    folder: SyntheticFuncDataset,
    tmp_path: Path,
    opened: list[tuple[Run, ReviewApp | None]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    render, pack = Run.render, Run.pack

    def rendering(self: Run, **kwargs: Any) -> Any:
        events.append("render")
        return render(self, **kwargs)

    def packing(self: Run, **kwargs: Any) -> Any:
        events.append("pack")
        return pack(self, **kwargs)

    monkeypatch.setattr(Run, "render", rendering)
    monkeypatch.setattr(Run, "pack", packing)
    gate = threading.Event()
    original = reviewing._render_rest

    def held(*args: Any, **kwargs: Any) -> None:
        gate.wait(60)
        original(*args, **kwargs)

    monkeypatch.setattr(reviewing, "_render_rest", held)
    run, app = open_review(folder.raw_root, run_dir=tmp_path / "run", jobs=1, port=0)
    opened.append((run, app))
    assert app is not None
    assert "pack" not in events  # no pack before the app
    assert pack_counts(run) == (0, 0, 11) and render_counts(run) == (1, 0, 1)
    at_app = len(events)
    gate.set()
    assert app.wait_rendered(180)
    assert events[at_app:] == ["render", "pack"]
    assert pack_counts(run) == (11, 0, 0)


def test_the_stop_of_the_app_ends_the_packs_between_two_batches(
    folder: SyntheticFuncDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    review(folder.raw_root, run_dir=run_dir, jobs=1, serve=False, epi=False).close()
    with Run(run_dir) as run:
        run.set_epi(enabled=True)
        run.func_inventory()
        asked: list[int] = []

        def stop() -> bool:
            asked.append(1)
            return len(asked) > 1

        report = run.pack(stop=stop)  # one session, then the stop
        assert report.done in (3, 4) and pack_counts(run)[2] > 0
        report = run.pack()
        assert pack_counts(run) == (11, 0, 0)


# --- the options --------------------------------------------------------------------------------


def test_no_epi_makes_no_pack_and_is_stored_until_given_again(
    folder: SyntheticFuncDataset, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    run = review(folder.raw_root, run_dir=run_dir, jobs=1, serve=False, epi=False)
    assert not load_config(run_dir).epi.enabled  # pyrefly: ignore[missing-attribute]
    assert viewable(run) == 0 and rateable(run) == 2
    run.close()
    assert not (run_dir / EPI_DIR).exists()
    assert _rows(run_dir, "SELECT COUNT(*) FROM func_runs") == [(0,)]
    capsys.readouterr()
    review(folder.raw_root, run_dir=run_dir, jobs=1, serve=False).close()  # holds
    said = capsys.readouterr().err
    assert "EPI packs" not in said and not (run_dir / EPI_DIR).exists()
    run = review(folder.raw_root, run_dir=run_dir, jobs=1, serve=False, epi=True)
    assert viewable(run) == 11 and pack_counts(run) == (11, 0, 0)
    run.close()
    # back at the defaults: no section (review of step 2), so the interface is on by default
    assert load_config(run_dir).epi is None


def test_epi_runs_selects_runs_is_stored_and_a_glob_that_matches_none_is_refused(
    folder: SyntheticFuncDataset, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    with pytest.raises(ConfigError, match="no functional run matches"):
        review(folder.raw_root, run_dir=run_dir, serve=False, epi_runs=["*task-nothing*"])
    assert not run_dir.exists()  # refused before a run was made
    run = review(folder.raw_root, run_dir=run_dir, jobs=1, serve=False, epi_runs=["*_run-1"])
    selected = _rows(run_dir, "SELECT name FROM func_runs WHERE selected = 1 AND packable = 1")
    assert selected and all(name.endswith("_run-1") for (name,) in selected)
    assert pack_counts(run) == (len(selected), 0, 0)
    run.close()
    assert "not selected (--epi-runs)" in capsys.readouterr().err
    config = (run_dir / CONFIG_FILE).read_bytes()
    with pytest.raises(ConfigError, match="no functional run matches"):
        review(folder.raw_root, run_dir=run_dir, serve=False, epi_runs=["*task-nothing*"])
    assert (run_dir / CONFIG_FILE).read_bytes() == config  # not stored
    review(folder.raw_root, run_dir=run_dir, jobs=1, serve=False).close()  # holds
    assert load_config(run_dir).epi.runs == ("*_run-1",)  # pyrefly: ignore[missing-attribute]
    assert len(_packed_dirs(run_dir)) == len(selected)


def test_raw_names_a_raw_dataset_that_the_rules_do_not_find(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ds = write_func_dataset(tmp_path / "data", raw_inside=False)
    run_dir = tmp_path / "run"
    run = review(ds.deriv_root, run_dir=run_dir, jobs=1, serve=False)
    said = capsys.readouterr().err
    assert "no raw dataset found (give it with --raw DIR)" in said
    assert viewable(run) == 0
    run.close()
    run = review(ds.deriv_root, run_dir=run_dir, jobs=1, serve=False, raw=ds.raw_root)
    said = capsys.readouterr().err
    assert f"raw series in {ds.raw_root} (BIDS)" in said
    assert pack_counts(run) == (11, 0, 0)
    run.close()
    assert load_config(run_dir).epi.raw == str(ds.raw_root.resolve())  # pyrefly: ignore
    # a run directory in a participant folder of the raw dataset named is refused
    with pytest.raises(ConfigError, match="only read"):
        review(ds.deriv_root, run_dir=ds.raw_root / "sub-01" / "run", serve=False, raw=ds.raw_root)
    assert not (ds.raw_root / "sub-01" / "run").exists()


def test_a_run_directory_inside_the_dataset_works_and_one_in_a_participant_folder_not(
    folder: SyntheticFuncDataset, tmp_path: Path
) -> None:
    """Decision D35: a run directory inside a dataset may be asked for with --run-dir, and
    one that an earlier version made there is still used; the participant folders of the
    raw dataset and of the derivative are read and refuse it (E2.Z narrowed E2.B's rule)."""
    inside = folder.raw_root / "surfcensus-run"
    run = review(folder.raw_root, run_dir=inside, jobs=1, serve=False)
    assert pack_counts(run) == (11, 0, 0)
    run.close()
    for refused in (
        folder.raw_root / "sub-01" / "run",
        folder.deriv_root / "sub-01" / "run",
    ):
        with pytest.raises(ConfigError, match="only read"):
            review(folder.raw_root, run_dir=refused, serve=False)
        assert not refused.exists()


def test_a_folder_without_functional_data_is_told_and_stores_nothing_new(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = write_review_folder(tmp_path / "data")
    for func in [*folder.raw_root.glob("sub-*/**/func"), *folder.deriv_root.glob("sub-*/**/func")]:
        shutil.rmtree(func)
    run_dir = tmp_path / "run"
    run = review(folder.raw_root, run_dir=run_dir, jobs=1, serve=False)
    run.close()
    said = capsys.readouterr().err
    for word in ("Functional", "EPI", "epi"):
        assert word not in said
    text = (run_dir / CONFIG_FILE).read_text()
    assert "epi" not in text
    assert not (run_dir / EPI_DIR).exists()
    for table in ("func_sources", "func_participants", "func_sessions", "func_runs", "packs"):
        assert _rows(run_dir, f"SELECT COUNT(*) FROM {table}") == [(0,)]
    # the same command again writes the same configuration, also with --epi, which is the
    # default and so writes no section
    before = (run_dir / CONFIG_FILE).read_bytes()
    review(folder.raw_root, run_dir=run_dir, jobs=1, serve=False).close()
    assert (run_dir / CONFIG_FILE).read_bytes() == before
    review(folder.raw_root, run_dir=run_dir, jobs=1, serve=False, epi=True).close()
    assert (run_dir / CONFIG_FILE).read_bytes() == before


def test_relocate_keeps_every_functional_identifier_and_packs_nothing(
    folder: SyntheticFuncDataset, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    review(folder.raw_root, run_dir=run_dir, jobs=1, serve=False).close()
    before = sorted(_rows(run_dir, "SELECT run_uid, session_uid FROM func_runs"))
    moved = tmp_path / "moved"
    shutil.move(folder.root, moved)
    with pytest.raises(ConfigError):
        review(moved / "ds", run_dir=run_dir, serve=False)
    # the move itself takes the functional roots along, before any inventory, so that a
    # pack of the moved run reads the files at their new place
    with Run(run_dir, local(moved / "ds"), relocate=True) as run:
        roots = _rows(run_dir, "SELECT deriv_root, raw_root FROM func_sources")
        assert roots == [(str(moved / "ds" / "derivatives" / "fmriprep"), str(moved / "ds"))]
        shutil.rmtree(next((run_dir / EPI_DIR).glob("*/*/*/raw")))  # one pack to make again
        report = run.pack()
        assert (report.done, report.skipped, report.failed) == (1, 10, 0)
    run = review(moved / "ds", run_dir=run_dir, jobs=1, serve=False, relocate=True)
    assert sorted(_rows(run_dir, "SELECT run_uid, session_uid FROM func_runs")) == before
    report = run.pack()
    assert (report.done, report.skipped) == (0, 11)
    run.close()


def test_with_no_surface_to_render_the_app_opens_at_the_first_packed_session(
    tmp_path: Path, opened: list[tuple[Run, ReviewApp | None]], monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = write_review_folder(tmp_path / "data")
    for subject in folder.subjects:
        for surface in (folder.fs_subjects_dir / subject / "surf").glob("?h.pial*"):
            surface.unlink()
    monkeypatch.setattr(reviewing, "_render_rest", lambda *args, **kwargs: None)
    run, app = open_review(folder.raw_root, run_dir=tmp_path / "run", jobs=1, port=0)
    opened.append((run, app))
    assert app is not None and rateable(run) == 0
    order = fbatch.session_order(run.context())
    packed = {p.parent.name for p in _packed_dirs(tmp_path / "run")}
    assert packed == {order[0]}  # one session: --serve-after counts sessions here
    app.stop()
    run, app = open_review(folder.raw_root, run_dir=tmp_path / "run", jobs=1, port=0, serve_after=0)
    opened.append((run, app))
    assert app is not None and pack_counts(run) == (11, 0, 0) and not app.rendering


def test_nothing_to_render_and_nothing_to_pack_opens_no_app(tmp_path: Path) -> None:
    folder = write_review_folder(tmp_path / "data")
    for subject in folder.subjects:
        for surface in (folder.fs_subjects_dir / subject / "surf").glob("?h.pial*"):
            surface.unlink()
    run, app = open_review(folder.raw_root, run_dir=tmp_path / "run", port=0, epi=False)
    assert app is None and rateable(run) == 0 and viewable(run) == 0
    run.close()


# --- who may open the EPI pages ------------------------------------------------------------------


def test_a_blind_run_keeps_the_epi_pages_closed_to_a_rater_with_work_left(
    folder: SyntheticFuncDataset,
    tmp_path: Path,
    opened: list[tuple[Run, ReviewApp | None]],
) -> None:
    run, app = open_review(
        folder.raw_root, run_dir=tmp_path / "run", jobs=1, port=0, blind=True, serve_after=0
    )
    opened.append((run, app))
    assert app is not None
    with httpx.Client(base_url=app.url, trust_env=False) as client:
        assert client.get("/api/epi").status_code == 403
        assert client.get("/api/next").status_code == 200


def test_a_second_rater_has_the_epi_pages_closed_until_done(
    folder: SyntheticFuncDataset,
    tmp_path: Path,
    opened: list[tuple[Run, ReviewApp | None]],
) -> None:
    run_dir = tmp_path / "run"
    review(folder.raw_root, run_dir=run_dir, jobs=1, serve=False).close()
    for rater in ("ada", "ben"):
        run, app = open_review(folder.raw_root, run_dir=run_dir, jobs=1, port=0, rater=rater)
        opened.append((run, app))
        assert app is not None
        with httpx.Client(base_url=app.url, trust_env=False) as client:
            code = client.get("/api/epi").status_code
        app.stop()
        assert code == (200 if rater == "ada" else 403), rater


# --- a census run ---------------------------------------------------------------------------------


def test_a_census_run_does_nothing_functional(folder: SyntheticFuncDataset, tmp_path: Path) -> None:
    key = HashKey(bytes(range(1, 33)), "dataset-key")
    run = Run(tmp_path / "census", local(folder.raw_root), key=key)
    try:
        run.inventory()
        report = run.func_inventory()
        assert (report.done, report.skipped, report.failed) == (0, 0, 0)
        assert run.context().func_lease is None
        with pytest.raises(ConfigError, match="census"):
            run.pack()
        assert not run.public().has_func() and run.func_folders() == []
        assert run.func_fetch_lines() == []
    finally:
        run.close()


def test_the_pack_of_a_run_equals_that_of_epi_session_on_the_same_folder(
    folder: SyntheticFuncDataset, tmp_path: Path
) -> None:
    """The decoded streams of a run's pack are those that ``epi-session`` makes of the same
    session: one implementation of planning and packing (D40)."""
    from surfcensus.func.api import epi_subject

    run_dir = tmp_path / "run"
    review(folder.raw_root, run_dir=run_dir, jobs=1, serve=False).close()
    out = tmp_path / "out"
    stream = epi_subject(folder.raw_root, "sub-01", out)
    by_run: dict[str, str] = {}
    for sid in stream.sessions:
        for pack_json in (out / sid).glob("*/pack.json"):
            info = read_pack(out / sid, pack_json.parent.name)
            assert info is not None
            by_run[info.run] = info.content_digest
    digests = {d for (d,) in _rows(run_dir, "SELECT content_digest FROM packs")}
    assert set(by_run.values()) <= digests and len(by_run) == 7


def test_the_session_order_of_the_store_is_that_of_the_stream() -> None:
    """``func_participants.session_order`` and step 1's ``StreamInfo.order`` name one rule."""
    from surfcensus import store
    from surfcensus.func import api, inventory

    assert store.SESSION_ORDER == inventory.SESSION_ORDER == api.SESSION_ORDER


def test_a_session_whose_contours_cannot_be_cut_has_no_contour_files_among_its_inputs(
    folder: SyntheticFuncDataset, tmp_path: Path
) -> None:
    """As ``epi-session`` does: a pack without contours does not depend on the FreeSurfer
    files (E2.Z, for the open point of E2.B and E2.C)."""
    (folder.fs_subjects_dir / "sub-01" / "surf" / "lh.white").write_bytes(b"not a surface")
    run_dir = tmp_path / "run"
    run = review(folder.raw_root, run_dir=run_dir, jobs=1, serve=False, epi=True)
    try:
        assert pack_counts(run)[0] == 11
    finally:
        run.close()
    for session_dir in (run_dir / EPI_DIR).glob("*/*"):
        if not session_dir.is_dir():
            continue
        for pack_json in session_dir.glob("*/pack.json"):
            info = read_pack(session_dir, pack_json.parent.name)
            assert info is not None
            bases = {item.base for item in info.inputs}
            contours = list(session_dir.glob("contours/*/*.bin"))
            assert ("freesurfer" in bases) == bool(contours), pack_json


# --- what the commands say (review of step 2, A-F5 and A-F6) -------------------------------------


def test_a_session_without_contours_is_named_by_its_labels(
    folder: SyntheticFuncDataset, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (folder.fs_subjects_dir / "sub-01" / "surf" / "lh.white").write_bytes(b"not a surface")
    with caplog.at_level("WARNING", logger="surfcensus"):
        review(folder.raw_root, run_dir=tmp_path / "run", jobs=1, serve=False).close()
    said = [r.getMessage() for r in caplog.records if "without contours" in r.getMessage()]
    assert said and all(line.startswith("sub-01 ses-") for line in said), said
    assert not any("Error:" in line for line in said), said


def _no_surfaces(tmp_path: Path) -> SyntheticFuncDataset:
    folder = write_review_folder(tmp_path / "data")
    for subject in folder.subjects:
        for surface in (folder.fs_subjects_dir / subject / "surf").glob("?h.pial*"):
            surface.unlink()
    return folder


def test_with_nothing_to_rate_the_browser_opens_the_epi_movies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from surfcensus.stopping import StopFlag

    folder = _no_surfaces(tmp_path)
    opened_urls: list[str] = []
    monkeypatch.setattr(reviewing.webbrowser, "open", lambda url: opened_urls.append(url))
    monkeypatch.setattr(reviewing, "_render_rest", lambda *args, **kwargs: None)
    run, app = open_review(folder.raw_root, run_dir=tmp_path / "run", jobs=1, port=0)
    assert app is not None and app.epi_url is not None
    stop = StopFlag()
    stop.set()
    reviewing.serve_review(run, app, open_browser=True, stop=stop)
    run.close()
    assert opened_urls == [app.epi_url]
    said = capsys.readouterr().err
    assert f"Rating app: {app.url}" in said
    assert "No subject can be rated; the browser opens the EPI movies." in said

    # serve on the same run: the EPI address is printed, and the browser goes there too
    from surfcensus.app import server

    monkeypatch.setattr(server.webbrowser, "open", lambda url: opened_urls.append(url))
    stop = StopFlag()
    stop.set()
    server.serve(tmp_path / "run", port=0, open_browser=True, stop=stop)
    said = capsys.readouterr().err
    (address,) = [line for line in said.splitlines() if line.startswith("EPI movies: ")]
    assert opened_urls[-1] == address.removeprefix("EPI movies: ")


def test_review_no_serve_with_nothing_to_rate_does_not_say_that_movies_are_shown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = _no_surfaces(tmp_path)
    review(folder.raw_root, run_dir=tmp_path / "run", jobs=1, serve=False).close()
    said = capsys.readouterr().err
    assert "are shown" not in said
    assert "Its functional runs are packed for the EPI movies." in said


def test_serve_names_the_epi_movies_when_they_are_offered(
    folder: SyntheticFuncDataset, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from surfcensus.app import server
    from surfcensus.stopping import StopFlag

    review(folder.raw_root, run_dir=tmp_path / "run", jobs=1, serve=False).close()
    capsys.readouterr()
    stop = StopFlag()
    stop.set()
    server.serve(tmp_path / "run", port=0, open_browser=False, stop=stop)
    said = capsys.readouterr().err
    assert re.search(r"EPI movies: http://127\.0\.0\.1:\d+/epi", said)
    assert "the browser opens the EPI movies" not in said  # there are subjects to rate
    review(folder.raw_root, run_dir=tmp_path / "off", jobs=1, serve=False, epi=False).close()
    capsys.readouterr()
    server.serve(tmp_path / "off", port=0, open_browser=False, stop=stop)
    assert "EPI movies:" not in capsys.readouterr().err
