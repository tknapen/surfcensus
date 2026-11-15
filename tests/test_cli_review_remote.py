"""``surfcensus review openneuro:dsXXXXXX`` and the commands on its run, as a user runs them,
against the fake bucket served on a loopback port (``FakeS3.serve()``, reached through
``SURFCENSUS_S3_ENDPOINT``; docs/dev/remote_plan.md, section 7).

The process tests stop the command while it fetches the Tier B files of the metrics: the
fake holds the first request for ``lh.thickness`` until the test has sent its signal, so
nothing depends on how fast the machine is. They then check the exit code, that nothing is
left under ``cache/leases/``, that the next call finishes, and that what it made equals a
review that was never stopped, matched by subject label. The request log of the fake holds
GET requests (listings among them) only, at most 8 at once.
"""

from __future__ import annotations

import fnmatch
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import httpx
import pandas as pd
import pytest
from fakes3 import FAKE_ACCESSION, FakeS3, bucket_of  # pyrefly: ignore[missing-import]
from synth_tierb import SyntheticCohort  # pyrefly: ignore[missing-import]
from typer.testing import CliRunner, Result

pytest.importorskip("fastapi")
pytest.importorskip("sqlmodel")

from fastapi.testclient import TestClient

from surfcensus import remote
from surfcensus import review as reviewing
from surfcensus.app.server import create_app
from surfcensus.cli import app
from surfcensus.remote.cache import CACHE_DIR, LEASES
from surfcensus.review import review
from surfcensus.store import PublicCatalog, read_table, table_path

BUCKET = "openneuro-derivatives"
TEXT = f"openneuro:{FAKE_ACCESSION}"
BASE = "http://127.0.0.1"
COMMAND = [sys.executable, "-c", "from surfcensus.cli import app; app()"]


class HeldFake(FakeS3):
    """A fake bucket that holds the first GET of a key matching ``pattern`` until
    :attr:`gate` is set, and says when it holds one (:attr:`entered`)."""

    def __init__(self, buckets: Mapping[str, Path], pattern: str, **options: Any) -> None:
        super().__init__(buckets, **options)
        self.pattern = pattern
        self.entered = threading.Event()
        self.gate = threading.Event()

    def respond(
        self, method: str, host: str, path: str, query: str, headers: Mapping[str, str]
    ) -> Any:
        if (
            method == "GET"
            and not self.gate.is_set()
            and fnmatch.fnmatch(unquote(path), self.pattern)
        ):
            self.entered.set()
            self.gate.wait(60)
        return super().respond(method, host, path, query, headers)


@pytest.fixture(autouse=True)
def no_freesurfer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FREESURFER_HOME", raising=False)


@pytest.fixture(scope="module")
def bucket(tmp_path_factory: pytest.TempPathFactory, small_run_cohort: SyntheticCohort) -> Path:
    return bucket_of(small_run_cohort, root=tmp_path_factory.mktemp("bucket"))


@pytest.fixture(scope="module")
def clean(tmp_path_factory: pytest.TempPathFactory, bucket: Path) -> Path:
    """A review of the bucket that was never stopped, made in this process."""
    run_dir = tmp_path_factory.mktemp("clean") / "run"
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("FREESURFER_HOME", raising=False)
        fake = FakeS3({BUCKET: bucket})
        review(TEXT, run_dir=run_dir, serve=False, jobs=1, transport=fake.transport()).close()
    return run_dir


@pytest.fixture
def served(bucket: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[FakeS3, str]]:
    """The fake on a loopback port, its endpoint in the environment of this process (for
    ``CliRunner``) and of the processes it starts."""
    fake = FakeS3({BUCKET: bucket}, page_size=7)
    with fake.serve() as endpoint:
        monkeypatch.setenv(remote.ENDPOINT_ENV, endpoint)
        yield fake, endpoint


def cli(*args: object) -> Result:
    return CliRunner().invoke(app, [str(arg) for arg in args])


def _default_sigint() -> None:
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def start(args: list[str], cwd: Path, endpoint: str) -> subprocess.Popen[str]:
    environment = {k: v for k, v in os.environ.items() if k != "FREESURFER_HOME"}
    return subprocess.Popen(
        [*COMMAND, *args],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment | {"PYTHONUNBUFFERED": "1", remote.ENDPOINT_ENV: endpoint},
        preexec_fn=_default_sigint,
    )


def finish(process: subprocess.Popen[str], timeout: float = 120) -> tuple[int, str, str]:
    out, err = process.communicate(timeout=timeout)
    return process.returncode, out, err


def leased(run_dir: Path) -> list[Path]:
    return sorted(path for path in (run_dir / CACHE_DIR / LEASES).rglob("*") if path.is_file())


def by_label(run_dir: Path) -> dict[str, str]:
    with PublicCatalog.open(run_dir) as public:
        subjects = public.subjects()
    return dict(zip(subjects["subject_label"], subjects["subject_uid"], strict=True))


def assert_same_outputs(run_dir: Path, clean: Path) -> None:
    """Maps, tiles, posters, vertex maps and the rows of two tables equal those of
    ``clean``, matched by label."""
    here, there = by_label(run_dir), by_label(clean)
    assert sorted(here) == sorted(there)
    for label in here:
        mine = run_dir / "media" / here[label]
        theirs = clean / "media" / there[label]
        names = sorted(
            p.relative_to(theirs)
            for p in theirs.rglob("*")
            if p.is_file() and p.name != "render.json"
        )
        assert names
        for name in names:
            assert (mine / name).read_bytes() == (theirs / name).read_bytes(), (label, name)
        assert (run_dir / "tables" / "fsavg" / f"{here[label]}.npz").read_bytes() == (
            clean / "tables" / "fsavg" / f"{there[label]}.npz"
        ).read_bytes()
    for name in ("regional", "vertex_summary"):
        frames = []
        for directory, uids in ((run_dir, here), (clean, there)):
            frame = read_table(table_path(directory, name))
            label = {uid: lab for lab, uid in uids.items()}
            frame = frame.assign(subject_uid=frame["subject_uid"].map(label))
            keys = [c for c in ("subject_uid", "hemi", "parcel") if c in frame.columns]
            frames.append(frame.sort_values(keys).reset_index(drop=True))
        pd.testing.assert_frame_equal(*frames)


def assert_only_reads(fake: FakeS3) -> None:
    assert fake.methods <= {"GET", "HEAD", "LIST"} and fake.signed == []
    assert 1 <= fake.high_water <= remote.MAX_CONCURRENCY


# --- the command as a process ---------------------------------------------------------------------


@pytest.mark.skipif(os.name == "nt", reason="signals of a POSIX system")
@pytest.mark.parametrize(("signum", "code"), [(signal.SIGINT, 130), (signal.SIGTERM, 143)])
def test_a_signal_during_tier_b_ends_the_review_and_the_next_call_finishes(
    tmp_path: Path, bucket: Path, clean: Path, signum: signal.Signals, code: int
) -> None:
    fake = HeldFake({BUCKET: bucket}, "*/surf/lh.thickness", page_size=7, latency_s=0.01)
    run_dir = tmp_path / "run"
    args = ["review", TEXT, "--no-browser", "--port", "0", "--jobs", "1", "--run-dir", str(run_dir)]
    with fake.serve() as endpoint:
        process = start(args, tmp_path, endpoint)
        try:
            assert fake.entered.wait(120), "no Tier B file was asked for"
            process.send_signal(signum)
            time.sleep(0.2)  # the signal reaches the main thread while the lease waits
            fake.gate.set()
            returned, out, err = finish(process)
        finally:
            fake.gate.set()
            if process.poll() is None:
                process.kill()
                process.wait()
        assert returned == code, err
        word = "terminated" if signum == signal.SIGTERM else "interrupted"
        assert err.splitlines()[-1] == f"{word}; what was recorded stays, and the next call goes on"
        assert "Traceback" not in err and out == ""
        assert leased(run_dir) == []
        assert_only_reads(fake)
        # the next call finishes, with the outputs of a review that was never stopped
        returned, out, err = finish(start([*args, "--no-serve"], tmp_path, endpoint))
        assert returned == 0, err
        assert "Rendered: 3 of 3 subjects." in err
        assert leased(run_dir) == []
    assert_same_outputs(run_dir, clean)
    assert_only_reads(fake)


@pytest.mark.skipif(os.name == "nt", reason="signals of a POSIX system")
def test_after_a_kill_the_next_call_prunes_what_was_left_and_finishes(
    tmp_path: Path, bucket: Path, clean: Path
) -> None:
    fake = HeldFake({BUCKET: bucket}, "*/surf/lh.thickness", page_size=7)
    run_dir = tmp_path / "run"
    args = ["review", TEXT, "--no-serve", "--jobs", "1", "--run-dir", str(run_dir)]
    with fake.serve() as endpoint:
        process = start(args, tmp_path, endpoint)
        try:
            assert fake.entered.wait(120), "no Tier B file was asked for"
            process.kill()
            process.wait()
        finally:
            fake.gate.set()
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        # the killed process left its lease area: its lock, and what it had fetched
        areas = sorted((run_dir / CACHE_DIR / LEASES).iterdir())
        assert len(areas) == 1 and areas[0].name.endswith(f"-{process.pid}")
        returned, _, err = finish(start(args, tmp_path, endpoint))
        assert returned == 0, err
        assert not areas[0].exists() and leased(run_dir) == []
    assert_same_outputs(run_dir, clean)
    assert_only_reads(fake)


@pytest.mark.skipif(os.name == "nt", reason="signals of a POSIX system")
def test_the_app_of_a_remote_review_names_what_was_fetched_and_stops_at_sigterm(
    tmp_path: Path, served: tuple[FakeS3, str]
) -> None:
    fake, endpoint = served
    process = start(
        ["review", TEXT, "--no-browser", "--port", "0", "--jobs", "1"], tmp_path, endpoint
    )
    try:
        assert process.stderr is not None
        lines: list[str] = []
        while True:
            line = process.stderr.readline()
            assert line, "\n".join(lines)
            lines.append(line.rstrip("\n"))
            found = re.fullmatch(
                r"Rating app: (http://127\.0\.0\.1:\d+/) \(Ctrl-C to stop\)", lines[-1]
            )
            if found:
                url = found.group(1)
                break
        run_dir = tmp_path / f"{FAKE_ACCESSION}-surfcensus-run"
        assert lines[0] == f"Listing {TEXT} in s3://{BUCKET}, anonymously and read-only ..."
        assert lines[1] == f"Run directory: {run_dir} (new)"
        with httpx.Client(base_url=url, trust_env=False) as client:
            assert client.get("/api/next").json()["status"] == "assignment"
            deadline = time.monotonic() + 120
            while client.get("/api/next").json()["progress"]["rendered"] < 3:
                assert time.monotonic() < deadline, "the others were never rendered"
                time.sleep(0.1)
        process.send_signal(signal.SIGTERM)
        # read on from the stream the first lines came from: it has read ahead
        err = "\n".join(lines) + "\n" + process.stderr.read()
        returned = process.wait(timeout=120)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    assert returned == 0, err
    assert re.search(
        r"^  \d+\.\d s from the start of the command to the app; .+ fetched in \d+ requests\.$",
        err,
        re.M,
    ), err
    assert re.search(
        r"^Fetched .+ in \d+ requests in all; .+ of source files left in the cache\.$", err, re.M
    ), err
    assert "0 rated, 3 to go." in err
    assert leased(run_dir) == []
    assert_only_reads(fake)


# --- the other commands on the run of a remote review ---------------------------------------------


def test_the_phase_commands_status_export_and_finalize_work_on_a_remote_run(
    tmp_path: Path, served: tuple[FakeS3, str]
) -> None:
    fake, _ = served
    run_dir = tmp_path / "run"
    result = cli("review", TEXT, "--no-serve", "--jobs", 1, "--run-dir", run_dir)
    assert result.exit_code == 0, result.output
    for args in (
        ("inventory", "--run-dir", run_dir),
        ("metrics", "--run-dir", run_dir),
        ("render", "--run-dir", run_dir),
        ("status", "--run-dir", run_dir, "--failures"),
    ):
        result = cli(*args)
        assert result.exit_code == 0, (args, result.output)
        assert FAKE_ACCESSION not in result.output or args[0] == "status", result.output
    # the phases with nothing to do made no request for a file
    status = cli("status", "--run-dir", run_dir).output
    assert "render: 3 done, 0 failed, 0 pending" in status
    with TestClient(create_app(run_dir), base_url=BASE) as client:
        shown = client.get("/api/next").json()["assignment"]
        body = {
            "assignment_id": shown["assignment_id"],
            "protocol_version": client.get("/api/session").json()["protocol"]["version"],
            "values": {"A": 0, "B": 0, "C": 0},
            "client_ts": "2026-11-15T12:00:00.000Z",
            "ms_on_subject": 1000,
        }
        assert client.post("/api/rating", json=body).status_code == 200
        assert FAKE_ACCESSION not in client.get("/api/session").text
    result = cli("finalize", "--run-dir", run_dir)
    assert result.exit_code == 0 and "1 from one rating" in result.output, result.output
    result = cli("export", "--run-dir", run_dir)
    assert result.exit_code == 0, result.output
    for path in (run_dir / "export").iterdir():
        assert FAKE_ACCESSION.encode() not in path.read_bytes(), path.name
    assert_only_reads(fake)


@pytest.mark.skipif(os.name == "nt", reason="signals of a POSIX system")
def test_serve_on_a_remote_run_needs_no_bucket(tmp_path: Path, served: tuple[FakeS3, str]) -> None:
    fake, endpoint = served
    run_dir = tmp_path / "run"
    assert cli("review", TEXT, "--no-serve", "--jobs", 1, "--run-dir", run_dir).exit_code == 0
    asked = len(fake.log)
    process = start(
        ["serve", "--run-dir", str(run_dir), "--no-browser", "--port", "0"], tmp_path, endpoint
    )
    try:
        assert process.stderr is not None
        url = None
        while url is None:
            line = process.stderr.readline()
            assert line, "serve ended early"
            found = re.match(r"Rating app: (http://127\.0\.0\.1:\d+/)", line)
            url = found.group(1) if found else None
        with httpx.Client(base_url=url, trust_env=False) as client:
            assert client.get("/api/next").json()["status"] == "assignment"
        process.send_signal(signal.SIGTERM)
        returned, _, err = finish(process)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    assert returned == 0, err
    assert len(fake.log) == asked  # serving reads the run directory only


# --- what the command refuses ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "option",
    [
        ("--relocate",),
        ("--raw", "/tmp"),
        ("--field-strength", "3"),
    ],
    ids=lambda option: option[0],
)
def test_options_for_local_data_are_refused_with_one_line(
    tmp_path: Path, served: tuple[FakeS3, str], option: tuple[str, ...]
) -> None:
    fake, _ = served
    result = cli("review", TEXT, "--no-serve", "--run-dir", tmp_path / "run", *option)
    assert result.exit_code == 2, result.output
    assert len(result.output.strip().splitlines()) == 1 and result.output.startswith("invalid: ")
    assert fake.log == [] and not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ("render-subject", TEXT, "--out", "media"),
        ("epi-session", TEXT, "--subject", "sub-001", "--out", "packs"),
        ("inventory", "--run-dir", "run", "--source", TEXT),
    ],
)
def test_the_commands_that_read_folders_say_so_for_a_bucket_dataset(
    tmp_path: Path,
    served: tuple[FakeS3, str],
    arguments: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review of the remote stage: they answered "not a directory" or "missing surface"."""
    fake, _ = served
    monkeypatch.chdir(tmp_path)
    result = cli(*arguments)
    said = " ".join(result.output.split())
    assert result.exit_code == 2, result.output
    assert "does not read" in said and f"surfcensus review {TEXT}" in said
    assert fake.log == [] and list(tmp_path.iterdir()) == []
    # a folder of that name is a folder: the command goes on to read it
    (tmp_path / TEXT).mkdir()
    result = cli(*arguments)
    assert "does not read" not in " ".join(result.output.split())


def test_a_text_that_names_no_dataset_is_refused(
    tmp_path: Path, served: tuple[FakeS3, str]
) -> None:
    result = cli("review", "openneuro:102", "--no-serve", "--run-dir", tmp_path / "run")
    assert result.exit_code == 2 and result.output.startswith("invalid: "), result.output


def test_a_review_that_would_fetch_too_much_asks_for_subjects(
    tmp_path: Path, served: tuple[FakeS3, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reviewing, "REMOTE_GUARD_BYTES", 1_000_000)
    run_dir = tmp_path / "run"
    result = cli("review", TEXT, "--no-serve", "--jobs", 1, "--run-dir", run_dir)
    assert result.exit_code == 2, result.output
    last = result.output.strip().splitlines()[-1]
    assert last.startswith("invalid: this dataset lists 3 subjects") and "--subjects" in last
    assert not run_dir.exists()
    result = cli(
        "review", TEXT, "--no-serve", "--jobs", 1, "--run-dir", run_dir, "--subjects", "sub-00[01]"
    )
    assert result.exit_code == 0, result.output
    assert "Rendered: 2 of 2 subjects." in result.output


def test_the_help_of_review_names_the_remote_form() -> None:
    result = cli("review", "--help")
    assert result.exit_code == 0
    assert "openneuro:dsXXXXXX" in " ".join(result.output.split())
