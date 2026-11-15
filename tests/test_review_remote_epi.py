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
import json
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
            "SELECT p.participant, r.*, k.status AS pack_status, k.content_digest "
            "FROM func_runs r JOIN func_sessions s USING (session_uid) "
            "JOIN func_participants p USING (participant_uid) "
            "LEFT JOIN packs k USING (run_uid) WHERE r.present = 1"
        )
    return {(row["participant"], row["name"]): dict(row) for row in rows}


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
    assert record["left_out"] == [] and len(record["objects"]) > 20
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


def test_a_file_refused_by_the_bucket_is_left_out_and_its_run_says_why(
    tmp_path: Path, buckets: dict, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeS3(buckets)
    fake.fault("*/sub-01/ses-1/func/*task-rest_run-1_bold.nii.gz", "403", times=None)
    run_dir = tmp_path / "run"
    remote_review(fake, run_dir, epi=True).close()
    assert "1 functional file(s) of the bucket could not be fetched" in capsys.readouterr().err
    rows = runs_by_name(run_dir)
    assert rows[("sub-01", "ses-1_task-rest_run-1")]["reason"] == "no raw series"
    assert sum(1 for row in rows.values() if row["pack_status"] == "done") == 10


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
        (row,) = run._catalog.query("SELECT deriv_root FROM func_sources")
        tree = Path(row["deriv_root"]).parent
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


def test_the_accession_is_only_where_it_is_allowed(prepared: Path) -> None:
    """Outside ``catalog.sqlite`` and the bucket's files kept under ``cache/`` (whose names
    alone are searched), no file of the run names the accession or a bucket key."""
    for path in sorted(prepared.rglob("*")):
        relative = path.relative_to(prepared)
        if path.name.startswith("catalog.sqlite"):
            continue
        assert FAKE_ACCESSION not in relative.as_posix(), relative
        if relative.parts[0] == CACHE_DIR or not path.is_file():
            continue
        content = path.read_bytes()
        for word in (FAKE_ACCESSION, "s3://", "fmriprep/ds"):
            assert word.encode() not in content, (relative, word)
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
