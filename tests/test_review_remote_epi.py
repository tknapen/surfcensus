"""The EPI interface of a dataset of the bucket: ``review openneuro:dsXXXXXX --epi``
(decision D50; docs/epi_interface.md, "From a bucket dataset").

The synthetic folder of ``tests/synth_func.py`` (``write_review_folder``) laid out as two fake
buckets: its raw BIDS dataset as ``s3://openneuro.org/ds999999/`` and its fMRIPrep derivative,
with the FreeSurfer subjects in its ``sourcedata/freesurfer``, as
``s3://openneuro-derivatives/fmriprep/ds999999-fmriprep/``. A remote review fetches the
functional files of the selected participants into a tree of links in the run's cache, and the
path-based planner and pack phase of ``func/`` run on it; the packs are those of a local review
of the same folder. Nothing here touches the network.
"""

from __future__ import annotations

import _thread
import fnmatch
import json
import os
import shutil
import signal
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fakes3 import FAKE_ACCESSION, FakeS3  # pyrefly: ignore[missing-import]
from synth_func import SyntheticFuncDataset, write_review_folder  # pyrefly: ignore[missing-import]

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from surfcensus import remote
from surfcensus import review as reviewing
from surfcensus import run as run_module
from surfcensus.app.server import create_app
from surfcensus.config import CONFIG_FILE, ConfigError, load_config
from surfcensus.export.tables import export_run
from surfcensus.func.status import status_lines
from surfcensus.remote import RAW_BUCKET
from surfcensus.remote.cache import CACHE_DIR, LEASES, RETAINED
from surfcensus.remote.s3 import S3Client
from surfcensus.review import review
from surfcensus.run import Run
from surfcensus.sources.base import SourceError
from surfcensus.sources.openneuro_derivatives import (
    FUNC_TREE_OBJECTS,
    OpenNeuroDerivatives,
)

DERIV_BUCKET = "openneuro-derivatives"
TEXT = f"openneuro:{FAKE_ACCESSION}"


@pytest.fixture(autouse=True)
def no_freesurfer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FREESURFER_HOME", raising=False)


def lay_out(dataset: SyntheticFuncDataset, root: Path) -> dict[str, Path]:
    """The two buckets of ``dataset`` under ``root``: links to its raw dataset and its
    derivative, so that a change to a file of the folder is a change of the bucket."""
    raw = root / RAW_BUCKET
    raw.mkdir(parents=True)
    (raw / FAKE_ACCESSION).symlink_to(dataset.raw_root.resolve(), target_is_directory=True)
    deriv = root / DERIV_BUCKET / "fmriprep"
    deriv.mkdir(parents=True)
    (deriv / f"{FAKE_ACCESSION}-fmriprep").symlink_to(
        dataset.deriv_root.resolve(), target_is_directory=True
    )
    return {RAW_BUCKET: raw, DERIV_BUCKET: root / DERIV_BUCKET}


def remote_review(fake: FakeS3, run_dir: Path, **options: Any) -> Run:
    options = {"serve": False, "jobs": 1} | options
    return review(TEXT, run_dir=run_dir, transport=fake.transport(), **options)


@pytest.fixture(scope="module")
def folder(tmp_path_factory: pytest.TempPathFactory) -> SyntheticFuncDataset:
    """The synthetic folder with both kinds of data. Read-only."""
    return write_review_folder(tmp_path_factory.mktemp("func_folder"))


@pytest.fixture(scope="module")
def buckets(tmp_path_factory: pytest.TempPathFactory, folder: SyntheticFuncDataset) -> dict:
    return lay_out(folder, tmp_path_factory.mktemp("func_buckets"))


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory, buckets: dict) -> Path:
    """One ``review(openneuro:ds999999, epi=True, serve=False)``: every packable run packed.
    Read-only."""
    run_dir = tmp_path_factory.mktemp("remote_epi") / f"{FAKE_ACCESSION}-surfcensus-run"
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("FREESURFER_HOME", raising=False)
        remote_review(FakeS3(buckets), run_dir, epi=True).close()
    return run_dir


@pytest.fixture(scope="module")
def local_run(tmp_path_factory: pytest.TempPathFactory, folder: SyntheticFuncDataset) -> Path:
    """A local review of the same folder. Read-only."""
    run_dir = tmp_path_factory.mktemp("local_epi") / "run"
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("FREESURFER_HOME", raising=False)
        review(folder.raw_root, run_dir=run_dir, serve=False, jobs=1).close()
    return run_dir


@pytest.fixture
def run_dir(prepared: Path, tmp_path: Path) -> Path:
    return Path(shutil.copytree(prepared, tmp_path / f"{FAKE_ACCESSION}-surfcensus-run"))


def runs_by_name(run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Every present run of a run directory by (participant, BIDS name), with its pack."""
    with Run(run_dir) as run:
        rows = run._catalog.query(
            "SELECT p.participant, r.*, k.status AS pack_status, k.content_digest, k.packed_at "
            "FROM func_runs r JOIN func_sessions s USING (session_uid) "
            "JOIN func_participants p USING (participant_uid) "
            "LEFT JOIN packs k USING (run_uid) WHERE r.present = 1"
        )
    return {(row["participant"], row["name"]): dict(row) for row in rows}


def own_tree(run_dir: Path) -> Path:
    """The functional tree in the cache of ``run_dir`` itself: a copy of a run directory keeps
    the stored roots of the original until a call of its own follows them."""
    (tree,) = (run_dir / CACHE_DIR / RETAINED).glob("*/*/func")
    return tree


def gets(fake: FakeS3, bucket: str | None = None) -> list[str]:
    return [
        entry.key or ""
        for entry in fake.log
        if entry.method == "GET" and (bucket is None or entry.bucket == bucket)
    ]


def raw_requests(fake: FakeS3) -> list[Any]:
    return [entry for entry in fake.log if entry.bucket == RAW_BUCKET]


# --- what a review makes -----------------------------------------------------------------------


def test_the_packs_of_a_remote_review_are_those_of_a_local_review(
    prepared: Path, local_run: Path
) -> None:
    remote_runs, local_runs = runs_by_name(prepared), runs_by_name(local_run)
    assert set(remote_runs) == set(local_runs)
    packable = [key for key, row in local_runs.items() if row["packable"]]
    assert len(packable) == 11
    for key in packable:
        assert remote_runs[key]["pack_status"] == "done", key
        assert remote_runs[key]["content_digest"] == local_runs[key]["content_digest"], key
    # the facts of every run agree but for the roots and the keyed identifiers, and for the
    # two runs whose only preprocessed series lies in another space: the tree holds no such
    # series (it is not shown), so the planner says that the run has none (decision D50)
    other_space = {("sub-02", "task-native_run-1"), ("sub-02", "task-std_run-1")}
    columns = ("reason", "packable", "selected", "anchor", "n_volumes", "grid_key", "raw_affine")
    columns += ("corrections", "frame_max_mm", "flagged", "acquisition_order", "dof")
    for key, row in local_runs.items():
        theirs = remote_runs[key]
        for column in columns:
            if key in other_space and column == "reason":
                assert row[column] in ("native space only", "standard space only")
                assert theirs[column] == "no preprocessed series"
                continue
            assert theirs[column] == row[column], (key, column)


def test_the_tree_is_in_the_retained_cache_and_named_by_keyed_names(prepared: Path) -> None:
    with Run(prepared) as run:
        (row,) = run._catalog.query("SELECT deriv_root, raw_root, deriv_identity FROM func_sources")
    deriv, raw = Path(row["deriv_root"]), Path(row["raw_root"])
    tree = deriv.parent
    assert deriv.name == "deriv" and raw == tree / "raw" and tree.name == "func"
    assert tree.parent.parent.parent == (prepared / CACHE_DIR / RETAINED).resolve() or (
        tree.parent.parent.parent == prepared / CACHE_DIR / RETAINED
    )
    # the identity names the dataset in the bucket, no path of this machine
    assert row["deriv_identity"].startswith(f"s3://{DERIV_BUCKET}/fmriprep/{FAKE_ACCESSION}")
    record = json.loads((tree / FUNC_TREE_OBJECTS).read_text())
    assert set(record) == {"objects"} and len(record["objects"]) > 20
    # the tree's files are hard links of retained objects
    series = next(deriv.rglob("*_space-T1w_desc-preproc_bold.nii.gz"))
    assert series.stat().st_nlink == 2
    # nothing of the functional files is outside the tree but their retained objects
    assert not (prepared / CACHE_DIR / LEASES).exists()


def test_without_epi_the_raw_bucket_is_never_asked(tmp_path: Path, buckets: dict) -> None:
    fake = FakeS3(buckets)
    run_dir = tmp_path / "run"
    remote_review(fake, run_dir).close()
    assert raw_requests(fake) == []
    text = (run_dir / CONFIG_FILE).read_text()
    assert "epi" not in text  # the bytes of config.yaml of a remote run before D50
    with Run(run_dir) as run:
        assert run._catalog.query("SELECT count(*) AS n FROM func_sources")[0]["n"] == 0
        assert run.func_inventory().done == 0
    # --epi, then --no-epi, leaves config.yaml as it was
    remote_review(FakeS3(buckets), run_dir, epi=True).close()
    assert "epi" in (run_dir / CONFIG_FILE).read_text()
    remote_review(FakeS3(buckets), run_dir, epi=False).close()
    assert (run_dir / CONFIG_FILE).read_text() == text


def test_a_second_call_lists_and_fetches_plans_and_packs_nothing(
    run_dir: Path, buckets: dict
) -> None:
    fake = FakeS3(buckets)
    with remote_review(fake, run_dir) as run:  # --epi is stored
        report = run.func_inventory()
        assert report.done == 0 and report.skipped == 3
        assert run.pack().done == 0
    # what is asked for an object is the derivative's description that the source reads
    # for the dataset's version, as before D50; no functional file is fetched again
    assert gets(fake) == [f"fmriprep/{FAKE_ACCESSION}-fmriprep/dataset_description.json"]
    assert {entry.method for entry in raw_requests(fake)} == {"LIST"}


def test_a_changed_functional_file_is_fetched_again_and_only_its_run_packed(
    tmp_path: Path,
) -> None:
    dataset = write_review_folder(tmp_path / "data")
    buckets = lay_out(dataset, tmp_path / "buckets")
    run_dir = tmp_path / "run"
    remote_review(FakeS3(buckets), run_dir, epi=True).close()
    before = runs_by_name(run_dir)
    confounds = next(
        (dataset.deriv_root / "sub-01" / "ses-1" / "func").glob(
            "*task-rest_run-1_desc-confounds_timeseries.tsv"
        )
    )
    confounds.write_text(confounds.read_text() + "\n")  # another ETag, the same table
    fake = FakeS3(buckets)
    with Run(run_dir, transport=fake.transport()) as run:
        run.inventory()
        func = run.func_inventory()
        packed = run.pack()
    fetched = [k for k in gets(fake) if "/func/" in k]
    assert fetched == [
        f"fmriprep/{FAKE_ACCESSION}-fmriprep/{confounds.relative_to(dataset.deriv_root)}"
    ]
    assert func.done == 1 and func.skipped == 2  # its session planned again, no other
    assert packed.done == 1  # its run packed again; the others of the session are kept
    after = runs_by_name(run_dir)
    changed = [
        key for key in after if after[key]["content_digest"] != before[key]["content_digest"]
    ]
    assert changed == [("sub-01", "ses-1_task-rest_run-1")]
    replanned = {k for k, v in after.items() if v["inventoried_at"] != before[k]["inventoried_at"]}
    assert replanned and all(k[0] == "sub-01" and k[1].startswith("ses-1_") for k in replanned)


def test_another_subject_set_moves_the_tree_and_the_first_participant_is_absent(
    run_dir: Path, buckets: dict
) -> None:
    with Run(run_dir) as run:
        (old,) = run._catalog.query("SELECT deriv_root FROM func_sources")
    query = (
        "SELECT p.participant FROM packs k JOIN func_runs r USING (run_uid) "
        "JOIN func_sessions s USING (session_uid) "
        "JOIN func_participants p USING (participant_uid)"
    )
    with remote_review(FakeS3(buckets), run_dir, subjects=["sub-02"]) as run:
        (row,) = run._catalog.query("SELECT deriv_root FROM func_sources")
        present = run._catalog.query("SELECT participant FROM func_participants WHERE present = 1")
        assert row["deriv_root"] != old["deriv_root"]
        assert [r["participant"] for r in present] == ["sub-02"]
        # as for a local folder whose participant left: the inventory marks it absent, and
        # its packs stay until the pack phase runs, which removes them (review of step 2)
        assert {r["participant"] for r in run._catalog.query(query)} == {"sub-01", "sub-02"}
        run.pack()
        assert {r["participant"] for r in run._catalog.query(query)} == {"sub-02"}
    participant_dirs = sorted(p.name for p in (run_dir / "epi").iterdir())
    assert len(participant_dirs) == 1


# --- the estimate, the guard, the disk ---------------------------------------------------------


def test_the_estimate_counts_and_names_the_functional_files(
    tmp_path: Path, buckets: dict, capsys: pytest.CaptureFixture[str]
) -> None:
    remote_review(FakeS3(buckets), tmp_path / "run", epi=True).close()
    err = capsys.readouterr().err
    line = next(line for line in err.splitlines() if line.startswith("To fetch:"))
    assert "the functional files of 2 subjects for the EPI movies" in line
    functional = next(line for line in err.splitlines() if line.startswith("Functional:"))
    assert f"in {TEXT} (fMRIPrep 21.0.2), raw series from s3://{RAW_BUCKET}" in functional
    assert str(tmp_path) not in functional


def test_the_guard_counts_the_functional_bytes(
    tmp_path: Path, buckets: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeS3(buckets)
    surface = reviewing.remote_estimate
    found: dict[str, int] = {}

    def spy(*args: Any, **kwargs: Any) -> Any:
        estimate = surface(*args, **kwargs)
        found["func" if kwargs.get("epi") else "plain"] = estimate.before_app
        return estimate

    monkeypatch.setattr(reviewing, "remote_estimate", spy)
    remote_review(fake, tmp_path / "plain").close()
    remote_review(FakeS3(buckets), tmp_path / "func", epi=True).close()
    assert found["func"] > found["plain"] + 1_000_000
    monkeypatch.setattr(reviewing, "REMOTE_GUARD_BYTES", (found["plain"] + found["func"]) // 2)
    remote_review(FakeS3(buckets), tmp_path / "plain2").close()  # under the guard
    with pytest.raises(ConfigError, match="--subjects"):
        remote_review(FakeS3(buckets), tmp_path / "func2", epi=True)
    assert not (tmp_path / "func2").exists()


def test_a_disk_without_room_for_the_functional_files_is_refused_before_they_are_fetched(
    tmp_path: Path, buckets: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    remote_review(FakeS3(buckets), run_dir).close()
    monkeypatch.setattr(run_module, "free_bytes", lambda path: 1_000)
    fake = FakeS3(buckets)
    with pytest.raises(ConfigError, match="functional files of the EPI interface"):
        remote_review(fake, run_dir, epi=True)
    assert [entry for entry in raw_requests(fake) if entry.method == "GET"] == []


# --- faults and stops ----------------------------------------------------------------------------


def test_an_unreachable_raw_bucket_records_nothing_and_the_next_call_goes_on(
    tmp_path: Path, buckets: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    from surfcensus.remote import s3

    monkeypatch.setattr(s3, "backoff_delay", lambda attempt, after=None: 0.0)
    run_dir = tmp_path / "run"
    remote_review(FakeS3(buckets), run_dir).close()
    fake = FakeS3(buckets)
    fake.fault(f"{FAKE_ACCESSION}/sub-*/*", "503", times=None, methods=("GET",))
    with pytest.raises(SourceError) as caught:
        remote_review(fake, run_dir, epi=True)
    assert caught.value.reason == "unreachable"
    assert "nothing was recorded" in str(caught.value)
    with Run(run_dir) as run:
        assert run._catalog.query("SELECT count(*) AS n FROM func_sessions")[0]["n"] == 0
        assert run._catalog.query("SELECT count(*) AS n FROM packs")[0]["n"] == 0
    assert not any((run_dir / CACHE_DIR / RETAINED).rglob(FUNC_TREE_OBJECTS))
    with remote_review(FakeS3(buckets), run_dir, epi=True) as run:
        assert run._catalog.query("SELECT count(*) AS n FROM packs")[0]["n"] == 11


FAULTS = {
    "raw series": ("*/sub-01/ses-1/func/*task-rest_run-1_bold.nii.gz", RAW_BUCKET),
    "derivative file": ("fmriprep/*/sub-01/ses-1/func/*task-rest_run-2*confounds*", DERIV_BUCKET),
    "contour file": ("fmriprep/*/sourcedata/freesurfer/sub-01/surf/lh.white", DERIV_BUCKET),
}


def functional_rows(run_dir: Path) -> int:
    with Run(run_dir) as run:
        return sum(
            run._catalog.query(f"SELECT count(*) AS n FROM {table}")[0]["n"]
            for table in ("func_sources", "func_sessions", "func_runs", "packs")
        )


@pytest.mark.parametrize("what", sorted(FAULTS))
def test_a_file_refused_once_records_nothing_and_the_next_call_fetches_only_it(
    tmp_path: Path, buckets: dict, capsys: pytest.CaptureFixture[str], what: str
) -> None:
    """A 403 on one functional file (decision D50, review): no tree is built, nothing
    functional is recorded in that call, the surfaces are rendered, one line says so; the next
    clean call fetches that file alone (and the listings) and packs every run with contours."""
    pattern, bucket = FAULTS[what]
    run_dir = tmp_path / "run"
    fake = FakeS3(buckets)
    fake.fault(pattern, "403", times=1, methods=("GET",))
    with remote_review(fake, run_dir, epi=True) as run:
        assert reviewing.render_counts(run)[0] == 2  # the surfaces went on
    err = capsys.readouterr().err
    assert f"Functional: the functional files of {TEXT} could not all be fetched" in err
    assert "1 file was refused or changed since they were listed" in err
    assert "the same command tries again" in err
    assert functional_rows(run_dir) == 0
    assert not any((run_dir / CACHE_DIR / RETAINED).rglob(FUNC_TREE_OBJECTS))
    clean = FakeS3(buckets)
    with remote_review(clean, run_dir) as run:
        sessions = run._catalog.query(
            "SELECT count(*) AS n FROM func_sessions WHERE fs_subject IS NOT NULL"
        )
        assert sessions[0]["n"] == 3
    fetched = [k for k in gets(clean) if not k.endswith("dataset_description.json")]
    assert len(fetched) == 1 and fnmatch.fnmatchcase(fetched[0], pattern)
    assert fetched[0] in gets(clean, bucket)
    rows = runs_by_name(run_dir)
    assert sum(1 for row in rows.values() if row["pack_status"] == "done") == 11
    session_files = list((run_dir / "epi").glob("*/*/session.json"))
    assert len(session_files) == 3
    for session_json in session_files:  # every session with its contours
        assert json.loads(session_json.read_text())["contours_from"] is not None, session_json


def test_a_file_refused_always_changes_nothing_of_an_earlier_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file whose new version is refused at every call: the earlier tree, its rows and its
    packs stay as they were at every call, and no false reason is recorded."""
    dataset = write_review_folder(tmp_path / "data")
    buckets = lay_out(dataset, tmp_path / "buckets")
    run_dir = tmp_path / "run"
    remote_review(FakeS3(buckets), run_dir, epi=True).close()
    before = runs_by_name(run_dir)
    with Run(run_dir) as run:
        (tree,) = run._catalog.query("SELECT deriv_root FROM func_sources")
    confounds = next(
        (dataset.deriv_root / "sub-01" / "ses-1" / "func").glob("*run-1_desc-confounds*")
    )
    confounds.write_text(confounds.read_text() + "\n")  # a new version, which is refused
    for _ in range(2):
        fake = FakeS3(buckets)
        fake.fault("*/sub-01/ses-1/func/*run-1_desc-confounds*", "403", times=None)
        remote_review(fake, run_dir).close()
        after = runs_by_name(run_dir)
        assert after == before
        with Run(run_dir) as run:
            assert run._catalog.query("SELECT deriv_root FROM func_sources") == [tree]
        assert Path(tree["deriv_root"]).is_dir()


def test_a_refused_listing_of_a_participant_leaves_the_surfaces_going(
    tmp_path: Path, buckets: dict, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeS3(buckets)
    fake.fault(f"{FAKE_ACCESSION}/sub-01/", "403", times=None, methods=("LIST",))
    run_dir = tmp_path / "run"
    with remote_review(fake, run_dir, epi=True) as run:
        assert reviewing.render_counts(run)[0] == 2
    err = capsys.readouterr().err
    line = next(line for line in err.splitlines() if line.startswith("To fetch:"))
    assert "the functional files could not be listed" in line
    assert "could not all be fetched (listing: HTTP 403" in err
    assert functional_rows(run_dir) == 0
    with remote_review(FakeS3(buckets), run_dir) as run:
        assert run._catalog.query("SELECT count(*) AS n FROM packs")[0]["n"] == 11


def test_an_interrupt_during_the_functional_downloads_leaves_nothing_partial(
    tmp_path: Path, buckets: dict
) -> None:
    """Ctrl-C while the raw series are downloaded: both clients are stopped, nothing partial
    is under ``retained/``, no tree has a name, and the next call fetches what is missing."""
    run_dir = tmp_path / "run"
    remote_review(FakeS3(buckets), run_dir).close()
    fake = FakeS3(buckets)
    with Run(run_dir, transport=fake.transport()) as run:
        run.set_epi(enabled=True)
        (source,) = run._sources()
        assert isinstance(source, OpenNeuroDerivatives) and source.raw_client is not None
        raw = source.raw_client
        download = raw.download
        release = threading.Event()
        started = threading.Semaphore(0)

        def held(key: str, dest: Path, **kwargs: Any) -> Any:
            started.release()
            assert release.wait(60)
            return download(key, dest, **kwargs)

        raw.download = held  # type: ignore[method-assign]

        def interrupt() -> None:
            assert started.acquire(timeout=60)
            time.sleep(0.2)
            _thread.interrupt_main()
            deadline = time.monotonic() + 20
            while not raw.stopped.is_set() and time.monotonic() < deadline:
                time.sleep(0.01)
            release.set()

        before = signal.signal(signal.SIGINT, signal.default_int_handler)
        thread = threading.Thread(target=interrupt)
        thread.start()
        try:
            with pytest.raises(KeyboardInterrupt):
                run.func_inventory()
        finally:
            signal.signal(signal.SIGINT, before)
            thread.join(60)
        raw.download = download  # type: ignore[method-assign]
        assert not raw.stopped.is_set() and not source.client.stopped.is_set()
        assert run._catalog.query("SELECT count(*) AS n FROM func_sessions")[0]["n"] == 0
    retained = run_dir / CACHE_DIR / RETAINED
    assert not any(retained.rglob(FUNC_TREE_OBJECTS))
    assert not [p for p in retained.rglob("*") if p.name.endswith((".tmp", ".retain", ".part"))]
    for path in retained.rglob("*.nii.gz"):  # every retained series is whole
        assert path.stat().st_size > 0
    with remote_review(FakeS3(buckets), run_dir) as run:
        assert run._catalog.query("SELECT count(*) AS n FROM packs")[0]["n"] == 11


# --- the commands that make no request ---------------------------------------------------------


def refusing() -> tuple[httpx.MockTransport, list[httpx.Request]]:
    asked: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(request)
        raise httpx.ConnectError("no request may be made", request=request)

    return httpx.MockTransport(handler), asked


def test_pack_status_serve_and_export_of_a_finished_run_make_no_request(run_dir: Path) -> None:
    transport, asked = refusing()
    start = remote.stats()
    with Run(run_dir, transport=transport) as run:
        report = run.pack(force=True)
        assert report.done == 11 and report.failed == 0
        # the copied run directory reads the tree of its own cache, not that of the original
        (row,) = run._catalog.query("SELECT deriv_root FROM func_sources")
        assert Path(row["deriv_root"]).is_relative_to(run_dir)
        lines = status_lines(run.context())
        assert lines[1].startswith("pack: 11 done")
    export_run(run_dir)
    assert (run_dir / "export" / "func_runs.csv").is_file()
    client = TestClient(create_app(run_dir), base_url="http://127.0.0.1")
    answer = client.get("/api/epi")
    assert answer.status_code == 200 and answer.json()["participants"]
    assert asked == [] and remote.stats().requests == start.requests


# --- the cache ---------------------------------------------------------------------------------


def test_prune_cache_keeps_the_tree_and_its_objects_and_removes_an_old_tree(
    run_dir: Path, buckets: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    from surfcensus import templates

    retained = run_dir / CACHE_DIR / RETAINED
    with Run(run_dir) as run:
        tree = own_tree(run_dir)
        old = retained / "ab" / ("ab" + "0" * 30) / "func"  # an earlier tree of the run
        shutil.copytree(tree, old)
        stray = retained / "cd" / ("cd" + "1" * 30)
        stray.mkdir(parents=True)
        (stray / "sub-01_task-rest_run-1_bold.nii.gz").write_bytes(b"old")
        run._catalog.set_meta("template_id", "x" * 12)
        monkeypatch.setattr(
            templates, "read_template", lambda path: SimpleNamespace(template_id="x" * 12)
        )
        removed = run.prune_cache()
    assert removed > 0
    assert not old.exists() and not stray.exists()
    assert (tree / FUNC_TREE_OBJECTS).is_file()
    record = json.loads((tree / FUNC_TREE_OBJECTS).read_text())
    for name in record["objects"]:
        assert (retained / name[:2] / name).is_dir(), name
    monkeypatch.undo()
    # the tree still reads: the next functional inventory fetches and plans nothing
    fake = FakeS3(buckets)
    with Run(run_dir, transport=fake.transport()) as run:
        report = run.func_inventory()
        assert (report.done, report.skipped) == (0, 3)
        assert run.pack().done == 0
    assert [key for key in gets(fake) if "/func/" in key] == []


# --- the guard and privacy -----------------------------------------------------------------------


def test_the_raw_client_is_behind_the_read_only_guard(buckets: dict) -> None:
    from surfcensus.config import RemoteConfig

    fake = FakeS3(buckets)
    client = S3Client(RAW_BUCKET, config=RemoteConfig(), transport=fake.transport())
    assert client.addressing == "path"
    with pytest.raises(PermissionError):
        client.http.get(f"{client.endpoint}/{FAKE_ACCESSION}/", headers={"Authorization": "x"})
    with pytest.raises(PermissionError):
        client.http.request("PUT", f"{client.endpoint}/{FAKE_ACCESSION}/x")
    assert fake.signed == [] and fake.log == []


def test_the_accession_is_only_where_it_is_allowed(prepared: Path, buckets: dict) -> None:
    """Outside ``catalog.sqlite`` and the bucket's files kept under ``cache/`` as they are,
    no file of the run names the accession or a bucket key: the files of ``cache/`` that are
    not a bucket file byte for byte (``.objects.json`` among them) are searched as well."""
    import hashlib

    fake = FakeS3(buckets)
    as_is = {
        hashlib.md5(fake.buckets[bucket].joinpath(*key.split("/")).read_bytes()).hexdigest()
        for bucket in buckets
        for key in fake.keys(bucket)
    }
    searched_in_cache = 0
    for path in sorted(prepared.rglob("*")):
        relative = path.relative_to(prepared)
        if path.name.startswith("catalog.sqlite"):
            continue
        assert FAKE_ACCESSION not in relative.as_posix(), relative
        if not path.is_file():
            continue
        content = path.read_bytes()
        if relative.parts[0] == CACHE_DIR:
            if hashlib.md5(content).hexdigest() in as_is:
                continue  # a file of the bucket as it is
            searched_in_cache += 1
        for word in (FAKE_ACCESSION, "s3://", "fmriprep/ds"):
            assert word.encode() not in content, (relative, word)
    assert searched_in_cache >= 1  # the tree's .objects.json at least
    assert "epi" in load_config(prepared).model_dump()


# --- the commands --------------------------------------------------------------------------------


def test_the_commands_on_a_remote_run_with_the_epi_interface(
    tmp_path: Path, buckets: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``review openneuro:dsXXXXXX --epi --no-serve`` through the command line, then
    ``status``, ``pack`` and ``export``, which make no request, and ``inventory --run-dir``,
    which refuses ``--raw`` and takes ``--epi-runs``."""
    from typer.testing import CliRunner

    from surfcensus.cli import app

    def cli(*args: object) -> Any:
        return CliRunner().invoke(app, [str(arg) for arg in args])

    fake = FakeS3(buckets, page_size=7)
    run_dir = tmp_path / "run"
    with fake.serve() as endpoint:
        monkeypatch.setenv(remote.ENDPOINT_ENV, endpoint)
        result = cli("review", TEXT, "--epi", "--no-serve", "--jobs", "1", "--run-dir", run_dir)
        assert result.exit_code == 0, result.output
        assert f"Functional: 19 runs in 3 sessions of 2 participants in {TEXT}" in result.output
        assert "EPI packs: 11 of 11 runs made." in result.output
        fake.clear()
        for args in (("status",), ("pack",), ("export",)):
            result = cli(*args, "--run-dir", run_dir)
            assert result.exit_code == 0, (args, result.output)
        assert fake.log == []
        result = cli("inventory", "--run-dir", run_dir, "--raw", tmp_path)
        assert result.exit_code == 2 and "--raw" in result.output
        result = cli("inventory", "--run-dir", run_dir, "--epi-runs", "*task-rest*")
        assert result.exit_code == 0, result.output
        assert "not selected (--epi-runs)" in result.output
        result = cli("inventory", "--run-dir", run_dir, "--epi-runs", "*no-such*")
        assert result.exit_code == 2 and "--epi-runs" in result.output


# --- the review of D50: guard, times, copies, and what the earlier tests let pass ------------


def test_the_guard_holds_for_the_epi_interface_turned_on_for_an_existing_run(
    tmp_path: Path, buckets: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    remote_review(FakeS3(buckets), run_dir).close()
    text = (run_dir / CONFIG_FILE).read_text()
    monkeypatch.setattr(reviewing, "REMOTE_GUARD_BYTES", 1_000_000)  # the functions: 7.5 MB
    fake = FakeS3(buckets)
    with pytest.raises(ConfigError, match=r"functional files.*--subjects"):
        remote_review(fake, run_dir, epi=True)
    assert gets(fake, RAW_BUCKET) == []
    assert (run_dir / CONFIG_FILE).read_text() == text  # nothing stored
    with remote_review(FakeS3(buckets), run_dir, epi=True, subjects=["*"]) as run:
        assert run._catalog.query("SELECT count(*) AS n FROM packs")[0]["n"] == 11


def test_the_times_of_the_tree_are_those_of_the_bucket(prepared: Path, buckets: dict) -> None:
    """The planner hashes size and modification time; the tree's are the listing's
    ``LastModified`` (the fake's file time, to the millisecond), not the download time."""
    with Run(prepared) as run:
        (row,) = run._catalog.query("SELECT raw_root FROM func_sources")
    raw = Path(row["raw_root"])
    series = next(raw.rglob("*_bold.nii.gz"))
    origin = buckets[RAW_BUCKET] / FAKE_ACCESSION / series.relative_to(raw)
    listed = origin.stat().st_mtime_ns // 1_000_000 * 1_000_000
    assert series.stat().st_mtime_ns == listed


def test_a_cache_fetched_again_packs_nothing_again(run_dir: Path, buckets: dict) -> None:
    before = runs_by_name(run_dir)
    shutil.rmtree(run_dir / CACHE_DIR / RETAINED)
    fake = FakeS3(buckets)
    with Run(run_dir, transport=fake.transport()) as run:
        run.inventory()
        report = run.func_inventory()
        assert (report.done, report.skipped) == (0, 3)
        assert run.pack().done == 0
    assert len(gets(fake)) > 100  # every file fetched again
    after = runs_by_name(run_dir)
    for key, row in after.items():
        assert row["content_digest"] == before[key]["content_digest"], key
        assert row["packed_at"] == before[key]["packed_at"], key


def test_a_copied_run_without_its_tree_reads_no_other_run_and_names_the_remedy(
    run_dir: Path, tmp_path: Path
) -> None:
    from typer.testing import CliRunner

    from surfcensus.cli import app

    copy = Path(shutil.copytree(run_dir, tmp_path / "copy"))
    shutil.rmtree(copy / CACHE_DIR / RETAINED)
    next((copy / "epi").glob("*/*/session.json")).unlink()  # a session of the copy pending
    shutil.move(run_dir, tmp_path / "elsewhere")
    result = CliRunner().invoke(app, ["pack", "--run-dir", str(copy)])
    assert result.exit_code == 2, result.output
    assert "not in its cache" in result.output
    assert f"surfcensus review {TEXT} --epi" in result.output
    with Run(copy) as run:
        (row,) = run._catalog.query("SELECT deriv_root FROM func_sources")
    assert Path(row["deriv_root"]).is_relative_to(copy)


def test_a_tree_file_that_lost_bytes_is_built_again(run_dir: Path, buckets: dict) -> None:
    raw = own_tree(run_dir) / "raw"
    series = next(raw.rglob("*_bold.nii.gz"))
    place, size = series.relative_to(raw), series.stat().st_size
    with series.open("r+b") as stream:  # a copy of a run: the tree holds copies, not links
        stream.truncate(size // 2)
    fake = FakeS3(buckets)
    with Run(run_dir, transport=fake.transport()) as run:
        run.inventory()
        run.func_inventory()
        (row,) = run._catalog.query("SELECT raw_root FROM func_sources")
    # the tree is not whole and is built again from the retained object, which is
    assert [key for key in gets(fake) if "/func/" in key] == []
    assert (Path(row["raw_root"]) / place).stat().st_size == size


def test_the_estimate_does_not_count_the_contour_files_twice(
    tmp_path: Path, buckets: dict, folder: SyntheticFuncDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    found: dict[str, Any] = {}
    surface = reviewing.remote_estimate

    def spy(*args: Any, **kwargs: Any) -> Any:
        estimate = surface(*args, **kwargs)
        found["func" if kwargs.get("epi") else "plain"] = estimate
        return estimate

    monkeypatch.setattr(reviewing, "remote_estimate", spy)
    remote_review(FakeS3(buckets), tmp_path / "plain").close()
    remote_review(FakeS3(buckets), tmp_path / "func", epi=True).close()
    places = ("mri/T1.mgz", "surf/lh.white", "surf/rh.white", "surf/lh.pial", "surf/rh.pial")
    shared = sum(
        (folder.fs_subjects_dir / subject / place).stat().st_size
        for subject in ("sub-01", "sub-02")
        for place in places
    )
    assert found["plain"].renders - found["func"].renders == shared


def test_a_continued_run_counts_the_functional_files_and_not_the_contours_twice(
    tmp_path: Path,
    buckets: dict,
    folder: SyntheticFuncDataset,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    remote_review(FakeS3(buckets), run_dir, subjects=["sub-01"]).close()
    capsys.readouterr()
    seen: dict[str, int] = {}
    line_of = reviewing.continued_line

    def spy(run: Run, **kwargs: Any) -> str:
        seen["renders"] = sum(run.remote_pending()[1].values())
        return line_of(run, **kwargs)

    monkeypatch.setattr(reviewing, "continued_line", spy)
    remote_review(FakeS3(buckets), run_dir, subjects=["*"], epi=True).close()
    line = next(line for line in capsys.readouterr().err.splitlines() if "To fetch" in line)
    assert "of it the functional files of 2 subjects" in line
    # a render leases the brainmask alone: the tree holds the T1 and the surfaces
    # without a pinned template the plan is an upper bound over every complete subject
    masks = sum(
        (folder.fs_subjects_dir / subject / "mri" / "brainmask.mgz").stat().st_size
        for subject in ("sub-01", "sub-02")
    )
    assert seen["renders"] == masks


def test_without_hard_links_a_tree_built_again_keeps_every_hash(
    run_dir: Path, buckets: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_link(*args: Any, **kwargs: Any) -> None:
        raise OSError(1, "no hard links here")

    shutil.rmtree(own_tree(run_dir))  # the tree alone; its objects stay
    monkeypatch.setattr(os, "link", no_link)
    fake = FakeS3(buckets)
    with Run(run_dir, transport=fake.transport()) as run:
        run.inventory()
        report = run.func_inventory()
        assert (report.done, report.skipped) == (0, 3)
    assert [k for k in gets(fake) if "/func/" in k] == []


def test_the_sessions_table_is_fetched_and_orders_the_stream(tmp_path: Path) -> None:
    from synth_func import write_session_dates  # pyrefly: ignore[missing-import]

    dataset = write_review_folder(tmp_path / "data")
    dates = {"1": "1900-01-03T10:00:00", "2": "1900-01-02T10:00:00"}
    write_session_dates(dataset.raw_root, "sub-01", sessions=dates)
    buckets = lay_out(dataset, tmp_path / "buckets")
    with remote_review(FakeS3(buckets), tmp_path / "run", epi=True) as run:
        (row,) = run._catalog.query(
            "SELECT session_order FROM func_participants WHERE participant = 'sub-01'"
        )
        (root,) = run._catalog.query("SELECT raw_root FROM func_sources")
    assert row["session_order"] == "acquisition time"
    assert (Path(root["raw_root"]) / "sub-01" / "sub-01_sessions.tsv").is_file()


def test_the_disk_keeps_a_margin_beside_the_functional_files(
    tmp_path: Path, buckets: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    remote_review(FakeS3(buckets), run_dir).close()
    need = 7_600_000  # a little more than the functional files of the folder (7.5 MB)
    monkeypatch.setattr(run_module, "free_bytes", lambda path: need + 100_000)
    with pytest.raises(ConfigError, match="beside room for what the phases write"):
        remote_review(FakeS3(buckets), run_dir, epi=True)
