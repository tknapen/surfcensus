"""Tests of ``surfcensus review``, ``serve``, ``finalize``, ``calibration freeze`` and
``export``: what they print, their exit codes, and that the app of a review stops at Ctrl-C
and at SIGTERM.

The runs are made by ``review`` on light subjects (16^3 voxels, 42 vertices per surface, no
template); the test that needs scores against a reference has its own.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
import yaml
from synth import snapshot_tree  # pyrefly: ignore[missing-import]
from synth_tree import (  # pyrefly: ignore[missing-import]
    write_light_template,
    write_linked_tree,
    write_tree,
)
from typer.testing import CliRunner, Result

pytest.importorskip("fastapi")
pytest.importorskip("sqlmodel")

from fastapi.testclient import TestClient

from surfcensus.app.server import create_app
from surfcensus.calibration import CALIBRATION_FILE
from surfcensus.cli import app
from surfcensus.config import ConfigError, load_config, with_changes, write_config
from surfcensus.export.tables import EXPORT_TABLES, MANIFEST, export_run
from surfcensus.privacy import HashKey
from surfcensus.review import DEFAULT_RUN_DIR
from surfcensus.run import Run
from surfcensus.sources.local import local
from surfcensus.store import PublicCatalog

BASE = "http://127.0.0.1"
LABELS = ("sub-01", "sub-02", "sub-03")


def cli(*args: object) -> Result:
    return CliRunner().invoke(app, [str(arg) for arg in args])


def said(output: str) -> str:
    """The words of a usage error, whatever the terminal: without escape codes, without the
    frame Rich draws, with every run of white space as one space."""
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output)
    return " ".join(re.sub(r"[─-╿]", " ", text).split())


@pytest.fixture
def study(tmp_path: Path) -> Path:
    """An fMRIPrep output with three light subjects; ``sub-03`` lacks ``surf/lh.pial``."""
    return write_tree(tmp_path / "data" / "study", "fmriprep", LABELS, incomplete=["sub-03"])


@pytest.fixture
def run_dir(study: Path) -> Path:
    """The run that ``surfcensus review study --no-serve`` makes beside the data."""
    result = cli("review", study, "--no-serve", "--jobs", 1)
    assert result.exit_code == 0, result.output
    return study.parent / DEFAULT_RUN_DIR


def uids(run_dir: Path) -> dict[str, str]:
    with PublicCatalog.open(run_dir) as public:
        subjects = public.subjects()
    return dict(zip(subjects["subject_label"], subjects["subject_uid"], strict=True))


def rate_all(run_dir: Path, rater: str, values: dict[str, dict[str, int] | str]) -> None:
    """Rate every subject the app offers as ``rater``: ``values`` gives, per label, the
    three values or the reason the subject is unratable."""
    labels = {uid: label for label, uid in uids(run_dir).items()}
    with TestClient(create_app(run_dir, rater=rater), base_url=BASE) as client:
        version = client.get("/api/session").json()["protocol"]["version"]
        answer = client.get("/api/next").json()
        clock = 0
        while answer["status"] == "assignment":
            shown = answer["assignment"]
            given = values[labels[shown["subject_uid"]]]
            clock += 1
            body: dict[str, Any] = {
                "assignment_id": shown["assignment_id"],
                "protocol_version": version,
                "client_ts": f"2026-11-13T12:00:{clock:02d}.000Z",
                "ms_on_subject": 1000 * clock,
            }
            body |= {"unratable": given} if isinstance(given, str) else {"values": given}
            posted = client.post("/api/rating", json=body)
            assert posted.status_code == 200, posted.text
            answer = posted.json()["next"]
        assert answer["status"] == "done"


def app_rows(run_dir: Path, sql: str) -> list[tuple[Any, ...]]:
    connection = sqlite3.connect(run_dir / "app.sqlite")
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


# --- review ---------------------------------------------------------------------------------------


def test_the_command_table_has_one_line_per_command() -> None:
    """``surfcensus --help`` lists each command with the first paragraph of its help and keeps
    the line breaks of a docstring, so a summary of two source lines came out ragged
    (``... inventory, metrics,`` / ``render and ...``). Each has a summary of one line."""
    import inspect

    from typer.main import get_command

    def summaries(group: Any, prefix: str = "") -> Iterator[tuple[str, str]]:
        for name, command in group.commands.items():
            text = command.short_help or inspect.cleandoc(command.help or "").split("\n\n")[0]
            yield prefix + name, text
            if hasattr(command, "commands"):
                yield from summaries(command, f"{prefix}{name} ")

    listed = dict(summaries(get_command(app)))
    assert {"review", "serve", "finalize", "export", "calibration freeze"} <= set(listed)
    assert all(text for text in listed.values())
    assert {name: text for name, text in listed.items() if "\n" in text} == {}


def test_review_without_the_app_prepares_the_run_and_says_what_it_did(study: Path) -> None:
    before = snapshot_tree(study)
    result = cli("review", study, "--no-serve", "--jobs", 1)
    run_dir = study.parent / DEFAULT_RUN_DIR
    assert result.exit_code == 0, result.output
    assert result.stdout == f"Run directory: {run_dir}\n"
    lines = result.stderr.splitlines()
    assert lines[0] == f"Run directory: {run_dir} (new)"
    assert lines[1] == (
        "Found 3 subjects in sourcedata/freesurfer (fMRIPrep layout). 1 cannot be rendered: "
        "sub-03 (no surf/lh.pial)."
    )
    assert "Scores: " in result.stderr
    assert "Rendered: 2 of 3 subjects, 1 cannot be rated." in lines
    assert "  not rendered, missing surface: 1 (sub-03)" in lines
    assert not any(line.startswith("Rating app") for line in lines)
    assert snapshot_tree(study) == before

    again = cli("review", study, "--no-serve", "--jobs", 1)
    assert again.exit_code == 0
    assert again.stderr.splitlines()[0] == f"Run directory: {run_dir} (continued)"


def test_review_ends_with_1_when_no_subject_can_be_rendered(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "subjects", "subjects_dir", LABELS, incomplete=LABELS)
    for options in (["--no-serve"], ["--no-browser", "--port", "0"]):
        result = cli("review", tree, "--run-dir", tmp_path / "run", *options)
        assert result.exit_code == 1, result.output
        assert result.stdout == ""
        assert "3 cannot be rendered: sub-01 (no surf/lh.pial)" in result.stderr
        assert result.stderr.splitlines()[-1] == (
            "No subject of this run can be rendered, so there is nothing to rate."
        )


def test_review_stops_with_exit_2_for_what_is_wrong_with_the_call(
    study: Path, run_dir: Path, tmp_path: Path
) -> None:
    def refused(*args: object) -> str:
        result = cli("review", *args, "--no-serve")
        assert result.exit_code == 2, result.output
        assert result.stdout == ""
        return said(result.stderr)

    assert refused(tmp_path / "nowhere").startswith("not found: ")
    assert refused(study, "--subjects", "sub-9*").startswith("not found: no subject matches")
    assert "a rater's name is not empty" in refused(study, "--rater", " ada")
    assert "a rater's name is not empty" in refused(study, "--rater", "")
    assert refused(study, "--field-strength", "-3").startswith("invalid: field_strength")
    assert "--jobs" in refused(study, "--jobs", 0)
    assert "--serve-after" in refused(study, "--serve-after", -1)
    assert "--port" in refused(study, "--port", 70000)
    moved = study.rename(study.parent / "moved")
    assert refused(moved, "--run-dir", run_dir).startswith("other root: ")
    result = cli("review", moved, "--run-dir", run_dir, "--relocate", "--no-serve")
    assert result.exit_code == 0, result.output


def test_review_refuses_to_change_blind_under_ratings_without_force(
    study: Path, run_dir: Path
) -> None:
    rate_all(run_dir, "local", {"sub-01": {"A": 0, "B": 0, "C": 0}, "sub-02": "render failure"})
    result = cli("review", study, "--no-serve", "--blind")
    assert result.exit_code == 2
    assert result.stderr.startswith("invalid: this run is not blind and has 2 submissions")
    assert result.stderr.count("\n") == 1 and "--force" in result.stderr
    assert not load_config(run_dir).app.blind
    forced = cli("review", study, "--no-serve", "--blind", "--force")
    assert forced.exit_code == 0 and load_config(run_dir).app.blind
    # and it holds until it is given again
    assert cli("review", study, "--no-serve").exit_code == 0 and load_config(run_dir).app.blind


def lacks(package: str) -> str:
    """The line for an installation that lacks ``package``. The app's packages are
    dependencies of the package itself, so the line names no extra."""
    return (
        f"missing: the rater app needs the package {package}, which this installation "
        "lacks; `pip install surfcensus` installs what is missing"
    )


def without(monkeypatch: pytest.MonkeyPatch, package: str) -> None:
    """Make ``import package`` fail as it does where the package is not installed, and the
    server module import its packages anew."""
    for name in [name for name in sys.modules if name.startswith("surfcensus.app.server")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, package, None)


def test_review_without_the_packages_of_the_app_says_how_to_get_them(
    study: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    without(monkeypatch, "fastapi")
    result = cli("review", study, "--no-browser")
    assert result.exit_code == 2
    assert result.stderr.strip() == lacks("fastapi")
    # nothing was done before the refusal, and without the app the command works
    assert not (study.parent / DEFAULT_RUN_DIR).exists()
    assert cli("review", study, "--no-serve", "--jobs", 1).exit_code == 0


@pytest.mark.parametrize("package", ["uvicorn", "sqlmodel", "jinja2"])
def test_review_names_whichever_package_of_the_app_is_missing(
    study: Path, monkeypatch: pytest.MonkeyPatch, package: str
) -> None:
    # every one of them is a dependency of the package (pyproject.toml), so this is an
    # installation that was stripped or is broken; no extra would repair it
    without(monkeypatch, package)
    result = cli("review", study, "--no-browser")
    assert result.exit_code == 2
    assert result.stderr.strip() == lacks(package)
    assert "[app]" not in result.stderr and "extra" not in result.stderr
    assert not (study.parent / DEFAULT_RUN_DIR).exists()


@pytest.mark.parametrize(
    ("command", "package"),
    [(("serve",), "uvicorn"), (("finalize",), "sqlmodel"), (("serve", "--no-browser"), "fastapi")],
)
def test_serve_and_finalize_say_the_same_of_a_missing_package(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch, command: tuple[str, ...], package: str
) -> None:
    before = snapshot_tree(run_dir)
    without(monkeypatch, package)
    result = cli(*command, "--run-dir", run_dir)
    assert result.exit_code == 2
    assert result.stderr.strip() == lacks(package)
    assert snapshot_tree(run_dir) == before
    # the export reads the database of the app without any of these packages
    assert cli("export", "--run-dir", run_dir).exit_code == 0


# --- review with the app, as a process ------------------------------------------------------------


def _default_sigint() -> None:
    # a test session started with SIGINT ignored (a job of a shell without job control)
    # would hand that on, and Python then installs no handler for Ctrl-C
    signal.signal(signal.SIGINT, signal.SIG_DFL)


@pytest.fixture
def review_process(study: Path) -> Iterator[subprocess.Popen[str]]:
    command = [
        sys.executable,
        "-c",
        "from surfcensus.cli import app; app()",
        "review",
        str(study),
        "--no-browser",
        "--port",
        "0",
        "--jobs",
        "1",
    ]
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


def _until_serving(process: subprocess.Popen[str]) -> tuple[str, list[str]]:
    """The URL of the app and the lines of stderr up to the one that names it."""
    assert process.stderr is not None
    lines = []
    while True:
        line = process.stderr.readline()
        assert line, f"the command ended early:\n{''.join(lines)}"
        lines.append(line.rstrip("\n"))
        found = re.fullmatch(
            r"Rating app: (http://127\.0\.0\.1:\d+/) \(Ctrl-C to stop\)", lines[-1]
        )
        if found:
            return found.group(1), lines


@pytest.mark.skipif(os.name == "nt", reason="signals of a POSIX system")
@pytest.mark.parametrize("stop_with", [signal.SIGINT, signal.SIGTERM])
def test_the_app_of_a_review_stops_at_ctrl_c_and_at_sigterm(
    study: Path, review_process: subprocess.Popen[str], stop_with: signal.Signals
) -> None:
    url, lines = _until_serving(review_process)
    run_dir = study.parent / DEFAULT_RUN_DIR
    assert lines[0] == f"Run directory: {run_dir} (new)"
    # the app opens at the first render; the second subject is rendered behind it
    assert "Rendered: 1 of 3 subjects, 1 to render while the app runs, 1 cannot be rated." in lines
    with httpx.Client(base_url=url, trust_env=False) as client:
        assert client.get("/healthz").status_code == 200
        page = client.get("/")
        assert page.status_code == 200 and "<title>" in page.text
        deadline = time.monotonic() + 60
        while client.get("/api/next").json()["progress"]["rendered"] < 2:
            assert time.monotonic() < deadline, "the second subject was never rendered"
            time.sleep(0.1)
    review_process.send_signal(stop_with)
    if stop_with == signal.SIGTERM:
        # a process manager may signal a whole group, and a wrapper may pass the signal on:
        # more arrive at once and while the command is stopping, and change nothing. With
        # a handler that took a lock (threading.Event.set) a burst could hang the command
        # for good (review of the quick start, F1)
        for gap in (0.0, 0.0, 0.001, 0.05):
            time.sleep(gap)
            review_process.send_signal(stop_with)
    # read on from the stream the first lines came from: it has read ahead
    assert review_process.stderr is not None and review_process.stdout is not None
    err = ""
    while "to go." not in err:
        line = review_process.stderr.readline()
        assert line, f"no closing line:\n{err}"
        err += line
    # and more come late, when the closing line is out and the command is about to end:
    # with a Python handler in place, which CPython takes away while the interpreter
    # finalizes, one of them killed the command (F2). SIGINT as well: the closing line
    # is out, and a Ctrl-C now has nothing left to stop
    deadline = time.monotonic() + 60
    while review_process.poll() is None:
        assert time.monotonic() < deadline, "the command did not end"
        for late in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(ProcessLookupError):
                review_process.send_signal(late)
        time.sleep(0.01)
    err += review_process.stderr.read()
    out = review_process.stdout.read()
    assert review_process.wait(timeout=60) == 0, err
    assert out == ""
    assert re.search(r"^  \d+\.\d s from the start of the command to the app\.$", err, re.M)
    assert err.splitlines()[-1] == (
        "0 rated, 2 to go. Run the same command to continue. "
        f"Export: surfcensus export --run-dir {run_dir}"
    )
    assert "Traceback" not in err
    with pytest.raises(httpx.TransportError), httpx.Client(trust_env=False) as client:
        client.get(url + "healthz")


def _session_processes(session: int) -> list[str]:
    """The processes of a session that are not zombies, by their command lines."""
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            if os.getsid(int(entry)) != session:
                continue
            with open(f"/proc/{entry}/stat") as stat:
                if stat.read().rsplit(")", 1)[1].split()[0] == "Z":
                    continue
            with open(f"/proc/{entry}/cmdline") as cmdline:
                found.append(cmdline.read().replace("\0", " "))
        except OSError:
            continue
    return found


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
@pytest.mark.parametrize(
    "order", [(signal.SIGTERM, signal.SIGINT), (signal.SIGINT, signal.SIGTERM)], ids=str
)
def test_signals_while_a_review_prepares_with_workers_end_it_cleanly(
    study: Path, order: tuple[signal.Signals, signal.Signals]
) -> None:
    """F3 of the review of the quick start: SIGTERM and SIGINT in either order while the
    preparation runs in worker processes. The first decides the code (143 or 130); no
    traceback, no hang, and no worker is left behind."""
    command = [sys.executable, "-c", "from surfcensus.cli import app; app()", "review"]
    # every render in the pool of two workers, before the app
    options = ["--no-browser", "--port", "0", "--jobs", "2", "--video", "--serve-after", "0"]
    process = subprocess.Popen(
        [*command, str(study), *options],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ | {"PYTHONUNBUFFERED": "1"},
        preexec_fn=_default_sigint,
        start_new_session=True,
    )
    try:
        assert process.stderr is not None
        err = ""
        while not err.endswith("\n") or not err.splitlines()[-1].startswith("Scores: "):
            line = process.stderr.readline()  # the scores are out: the renders come next
            assert line, err
            err += line
        time.sleep(0.3)  # while the workers start and render
        for signum in order:
            process.send_signal(signum)
            time.sleep(0.05)
        err += process.stderr.read()
        code = process.wait(timeout=60)
        time.sleep(0.5)
        left = _session_processes(process.pid)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        if process.stderr is not None:
            process.stderr.close()
    assert "Traceback" not in err and "Exception ignored" not in err, err
    assert left == [], left
    if "Rating app:" in err:  # the preparation was over: the first signal stopped the app
        assert code == 0, err
    else:
        word, expected = ("terminated", 143) if order[0] == signal.SIGTERM else ("interrupted", 130)
        assert code == expected, err
        assert err.splitlines()[-1] == (
            f"{word}; what was recorded stays, and the next call goes on"
        )


# The handlers of the commands, without a process: the functions behind the commands are
# replaced by ones that send this process a signal where a real one arrives. A burst that
# would make a handler wait for a lock cannot be sent into the test process safely (it
# would hang the tests); what is tested instead is the cause, deterministically: a handler
# that sets the stop calls nothing of the threading module (rule 1 of ``stopping``).


@pytest.fixture
def handlers_restored() -> Iterator[None]:
    """A command that took a signal, or served the app, ignores SIGTERM and SIGINT from
    then on, for a process that is about to end. The test process goes on, and gets back
    the handlers it had; SIGINT is Python's own during the test."""
    earlier_term = signal.getsignal(signal.SIGTERM)
    earlier_int = signal.signal(signal.SIGINT, signal.default_int_handler)
    yield
    signal.signal(signal.SIGTERM, earlier_term)
    signal.signal(signal.SIGINT, earlier_int)


class _NoRun:
    run_dir = Path("nowhere")

    def __enter__(self) -> _NoRun:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _signal_to_self(signum: int = signal.SIGTERM) -> None:
    os.kill(os.getpid(), signum)
    time.sleep(0.01)  # the handler runs in this thread, at the next instruction


def threading_calls(handler: Any, *args: object) -> list[str]:
    """The functions of the threading module that ``handler(*args)`` calls, as a profile of
    this thread records them: ``Event.set`` and every lock of a ``Condition`` are there."""
    called: list[str] = []

    def profile(frame: Any, event: str, arg: object) -> None:
        if event == "call" and frame.f_code.co_filename == threading.__file__:
            called.append(frame.f_code.co_name)

    sys.setprofile(profile)
    try:
        handler(*args)
    finally:
        sys.setprofile(None)
    return called


@pytest.mark.skipif(os.name == "nt", reason="signals of a POSIX system")
def test_a_sigterm_while_serving_stops_the_app_without_a_lock_and_is_ignored_after(
    monkeypatch: pytest.MonkeyPatch, handlers_restored: None
) -> None:
    from surfcensus import review as reviewing

    seen: list[object] = []

    def served(run: object, app: object, *, stop: Any, **options: object) -> None:
        assert not stop.is_set()
        handler = signal.getsignal(signal.SIGTERM)
        # F1: the handler takes no lock, so a second signal that arrives while it runs, or
        # while the main thread waits on the stop, cannot wait for it for good
        assert threading_calls(handler, signal.SIGTERM, None) == []
        assert stop.is_set()
        # F2: from its first statement on, every later SIGTERM is ignored by the process
        seen.append(signal.getsignal(signal.SIGTERM))
        _signal_to_self()  # ignored: the test goes on
        # Ctrl-C keeps its meaning while the app stops
        seen.append(signal.getsignal(signal.SIGINT))

    monkeypatch.setattr(reviewing, "open_review", lambda *a, **k: (_NoRun(), object()))
    monkeypatch.setattr(reviewing, "serve_review", served)
    monkeypatch.setattr(reviewing, "rateable", lambda run: 1)
    result = cli("review", "nowhere", "--no-browser")
    assert result.exit_code == 0, result.output
    assert seen[0] == signal.SIG_IGN and callable(seen[1])
    # the closing line is out: neither signal ends the process now, also not once the
    # interpreter finalizes, when CPython takes Python's handlers away and not SIG_IGN
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
    assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
    _signal_to_self(signal.SIGTERM)
    _signal_to_self(signal.SIGINT)


@pytest.mark.skipif(os.name == "nt", reason="signals of a POSIX system")
def test_a_sigterm_while_a_review_prepares_ends_it_as_ctrl_c_does(
    monkeypatch: pytest.MonkeyPatch, handlers_restored: None
) -> None:
    """F3: the preparation is written for KeyboardInterrupt (the batches wait for running
    work and release their leases); a SystemExit raised in pool or lock code is not what
    it expects. Once taken, SIGTERM and SIGINT are ignored, so nothing interrupts the
    unwinding a second time."""
    from surfcensus import review as reviewing

    raised: list[type[BaseException]] = []

    def preparing(*args: object, **options: object) -> None:
        try:
            _signal_to_self()
            raise AssertionError("the first SIGTERM did not end the preparation")
        except BaseException as exc:
            raised.append(type(exc))
            raise
        finally:
            _signal_to_self(signal.SIGTERM)  # more, while the first is being acted on
            _signal_to_self(signal.SIGINT)
            raised.append(GeneratorExit)  # the unwinding was not interrupted

    monkeypatch.setattr(reviewing, "open_review", preparing)
    result = cli("review", "nowhere", "--no-browser")
    assert raised == [KeyboardInterrupt, GeneratorExit]
    assert result.exit_code == 143
    assert result.stderr == "terminated; what was recorded stays, and the next call goes on\n"
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
    assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN


@pytest.mark.skipif(os.name == "nt", reason="signals of a POSIX system")
def test_ctrl_c_while_a_review_prepares_takes_a_later_sigterm_quietly(
    monkeypatch: pytest.MonkeyPatch, handlers_restored: None
) -> None:
    from surfcensus import review as reviewing

    raised: list[type[BaseException]] = []

    def preparing(*args: object, **options: object) -> None:
        try:
            _signal_to_self(signal.SIGINT)
        except BaseException as exc:
            raised.append(type(exc))
            # Ctrl-C is ending the command: a SIGTERM now raises nothing of its own
            _signal_to_self(signal.SIGTERM)
            raised.append(GeneratorExit)
            raise

    monkeypatch.setattr(reviewing, "open_review", preparing)
    result = cli("review", "nowhere", "--no-browser")
    assert raised == [KeyboardInterrupt, GeneratorExit]
    assert result.exit_code == 130
    assert result.stderr == "interrupted; what was recorded stays, and the next call goes on\n"
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN


@pytest.mark.skipif(os.name == "nt", reason="signals of a POSIX system")
def test_a_phase_command_ends_at_sigterm_as_at_ctrl_c(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch, handlers_restored: None
) -> None:
    from surfcensus.run import Run

    raised: list[type[BaseException]] = []

    def rendering(self: object, **options: object) -> None:
        try:
            _signal_to_self()
        except BaseException as exc:
            raised.append(type(exc))
            raise

    monkeypatch.setattr(Run, "render", rendering)
    result = cli("render", "--run-dir", run_dir)
    assert raised == [KeyboardInterrupt]
    assert result.exit_code == 143
    assert result.stderr == "terminated; what was recorded stays, and the next call goes on\n"
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN


def test_a_command_that_takes_no_signal_puts_the_handlers_back(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch, handlers_restored: None
) -> None:
    from surfcensus import review as reviewing

    earlier = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
    during: list[object] = []

    def nothing_served(*args: object, **options: object) -> tuple[_NoRun, None]:
        during.append(signal.getsignal(signal.SIGTERM))
        return _NoRun(), None

    monkeypatch.setattr(reviewing, "open_review", nothing_served)
    monkeypatch.setattr(reviewing, "rateable", lambda run: 1)
    result = cli("review", "nowhere", "--no-serve")
    assert result.exit_code == 0, result.output
    assert during[0] not in earlier
    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)) == earlier

    def refused(*args: object, **options: object) -> None:
        raise ConfigError("invalid", "no")

    monkeypatch.setattr(reviewing, "open_review", refused)
    assert cli("review", "nowhere", "--no-serve").exit_code == 2
    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)) == earlier
    # a phase command that does its work (1: sub-03 lacks lh.pial, a failed unit)
    assert cli("render", "--run-dir", run_dir).exit_code == 1
    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)) == earlier


@pytest.mark.skipif(os.name == "nt", reason="signals of a POSIX system")
def test_serve_stops_at_sigterm_without_a_lock_and_ignores_both_signals_after(
    monkeypatch: pytest.MonkeyPatch, handlers_restored: None
) -> None:
    import surfcensus.app.server as server

    earlier = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)

    def served(run_dir: object, *, stop: Any, **options: object) -> None:
        assert threading_calls(signal.getsignal(signal.SIGTERM), signal.SIGTERM, None) == []
        assert stop.is_set()
        _signal_to_self()  # ignored from the first one on

    monkeypatch.setattr(server, "serve", served)
    result = cli("serve", "--run-dir", "nowhere", "--mode", "local", "--no-browser")
    assert result.exit_code == 0, result.output
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
    assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
    _signal_to_self(signal.SIGTERM)
    _signal_to_self(signal.SIGINT)

    # a serve that is refused has served nothing, and the handlers are put back
    signal.signal(signal.SIGTERM, earlier[0])
    signal.signal(signal.SIGINT, earlier[1])

    def refused(run_dir: object, **options: object) -> None:
        raise ConfigError("invalid", "no")

    monkeypatch.setattr(server, "serve", refused)
    result = cli("serve", "--run-dir", "nowhere", "--mode", "local", "--no-browser")
    assert result.exit_code == 2
    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)) == earlier


def test_review_as_a_function_stops_at_sigterm_without_a_lock_and_puts_it_back(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch, handlers_restored: None
) -> None:
    from surfcensus import review as reviewing

    earlier = signal.getsignal(signal.SIGTERM)
    stops: list[Any] = []

    def served(run: object, app: object, *, stop: Any, **options: object) -> None:
        assert threading_calls(signal.getsignal(signal.SIGTERM), signal.SIGTERM, None) == []
        stops.append(stop)

    monkeypatch.setattr(reviewing, "open_review", lambda *a, **k: (Run(run_dir), object()))
    monkeypatch.setattr(reviewing, "serve_review", served)
    reviewing.review("nowhere", open_browser=False).close()
    assert stops[0].is_set()
    assert signal.getsignal(signal.SIGTERM) is earlier


# --- serve ----------------------------------------------------------------------------------------


def test_serve_refuses_what_it_cannot_serve(run_dir: Path, tmp_path: Path) -> None:
    def refused(*args: object) -> str:
        result = cli("serve", "--run-dir", run_dir, "--no-browser", *args)
        assert result.exit_code == 2, result.output
        return said(result.stderr)

    assert "is local or team" in refused("--mode", "census")
    assert refused("--mode", "team").startswith("invalid: team mode")
    assert refused("--host", "0.0.0.0").startswith("invalid: a local server listens on")
    assert "a rater's name is not empty" in refused("--rater", "ada ")
    missing = cli("serve", "--run-dir", tmp_path / "none", "--no-browser")
    assert missing.exit_code == 2 and missing.stderr.startswith("missing: ")
    # renders of another render config, the recorded failure of sub-03 among them: the app
    # would show nothing of them
    config = load_config(run_dir)
    write_config(
        run_dir, with_changes(config, render=config.render.model_dump() | {"panel_px": 128})
    )
    assert refused() == (
        f"invalid: 3 renders are out of date; run surfcensus render --run-dir {run_dir}"
    )
    assert not (run_dir / "app.sqlite").exists()


# --- finalize -------------------------------------------------------------------------------------


def test_finalize_needs_ratings(run_dir: Path, tmp_path: Path) -> None:
    result = cli("finalize", "--run-dir", run_dir)
    assert result.exit_code == 2
    assert (
        result.stderr == f"missing: {run_dir} has no ratings yet: there is no app.sqlite to read\n"
    )
    assert cli("finalize", "--run-dir", tmp_path / "none").exit_code == 2


def test_finalize_of_two_raters_prints_the_counts_and_the_open_subjects(
    tmp_path: Path,
) -> None:
    tree = write_tree(
        tmp_path / "subjects", "subjects_dir", ["sub-01", "sub-02", "sub-03", "sub-04"]
    )
    run_dir = tmp_path / "run"
    (tree / "sub-04" / "surf" / "lh.pial").unlink()
    assert cli("review", tree, "--run-dir", run_dir, "--no-serve", "--blind").exit_code == 0
    none = {"A": 0, "B": 0, "C": 0}
    rate_all(
        run_dir,
        "ada",
        {"sub-01": {"A": 1, "B": 0, "C": 0}, "sub-02": {"A": 2, "B": 0, "C": 1}, "sub-03": none},
    )
    one = cli("finalize", "--run-dir", run_dir)
    assert one.exit_code == 0, one.output
    assert one.stdout.splitlines() == [
        "final labels: 0 by consensus, 3 from one rating, 1 unratable (render failed), 0 open"
    ]
    rate_all(
        run_dir,
        "ben",
        {
            "sub-01": {"A": 0, "B": 0, "C": 0},
            "sub-02": {"A": 0, "B": 0, "C": 1},
            "sub-03": "image too poor to judge",
        },
    )
    two = cli("finalize", "--run-dir", run_dir)
    assert two.exit_code == 0, two.output
    assert two.stdout.splitlines() == [
        "final labels: 1 by consensus, 0 from one rating, 1 unratable (render failed), 2 open",
        "open (raters two or more steps apart, or unratable for one of them): decide "
        "together, then rate again or adjudicate",
        "  sub-02",
        "    ada: A 2, B 0, C 1",
        "    ben: A 0, B 0, C 1",
        "  sub-03",
        "    ada: A 0, B 0, C 0",
        "    ben: unratable (image too poor to judge)",
    ]
    by_label = uids(run_dir)
    labels = app_rows(
        run_dir, "SELECT subject_uid, criterion, value, method FROM final_labels ORDER BY 1, 2"
    )
    assert [row for row in labels if row[0] == by_label["sub-01"]] == [
        (by_label["sub-01"], "A", 0, "consensus"),
        (by_label["sub-01"], "B", 0, "consensus"),
        (by_label["sub-01"], "C", 0, "consensus"),
    ]
    assert {row[0] for row in labels} == {by_label["sub-01"], by_label["sub-04"]}
    # a second call writes the same
    assert cli("finalize", "--run-dir", run_dir).stdout == two.stdout


def test_a_run_of_the_phase_commands_is_rated_and_finalized_like_one_of_review(
    study: Path, tmp_path: Path
) -> None:
    # `inventory` and `render` write no selection. The selection of a local run is then
    # every present subject, for the app and for `finalize` as for `render`
    run_dir = tmp_path / "run"
    assert cli("inventory", "--run-dir", run_dir, "--source", study).exit_code == 0
    rendered = cli("render", "--run-dir", run_dir, "--jobs", 1)
    assert rendered.exit_code == 1 and "render: 2 done, 0 skipped, 1 failed" in rendered.output
    with PublicCatalog.open(run_dir) as public:
        assert len(public.selection()) == 0
    none = {"A": 0, "B": 0, "C": 0}
    rate_all(run_dir, "ada", {"sub-01": {"A": 1, "B": 0, "C": 0}, "sub-02": none})
    rate_all(run_dir, "ben", {"sub-01": none, "sub-02": none})
    result = cli("finalize", "--run-dir", run_dir)
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        "final labels: 2 by consensus, 0 from one rating, 1 unratable (render failed), 0 open"
    ]
    by_label = uids(run_dir)
    labels = app_rows(run_dir, "SELECT subject_uid, criterion, method FROM final_labels")
    assert sorted(labels) == sorted(
        [(by_label[label], criterion, "consensus") for label in LABELS[:2] for criterion in "ABC"]
        + [(by_label["sub-03"], "unratable", "single")]
    )


def test_finalize_names_a_rater_who_has_subjects_left(run_dir: Path) -> None:
    with TestClient(create_app(run_dir, rater="ada"), base_url=BASE) as client:
        version = client.get("/api/session").json()["protocol"]["version"]
        first = client.get("/api/next").json()["assignment"]
        skip = {
            "assignment_id": first["assignment_id"],
            "client_ts": "2026-11-13T12:00:00.000Z",
            "ms_on_subject": 700,
        }
        second = client.post("/api/skip", json=skip).json()["assignment"]
        rating = {
            "assignment_id": second["assignment_id"],
            "protocol_version": version,
            "values": {"A": 0, "B": 0, "C": 0},
            "client_ts": "2026-11-13T12:00:05.000Z",
            "ms_on_subject": 900,
        }
        assert client.post("/api/rating", json=rating).status_code == 200
    result = cli("finalize", "--run-dir", run_dir)
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        "final labels: 0 by consensus, 1 from one rating, 1 unratable (render failed), 0 open",
        "  ada still has 1 subject to rate (1 skipped); labels of those are from the other "
        "ratings only",
    ]


# --- calibration freeze ---------------------------------------------------------------------------


def test_calibration_freeze_writes_the_set_once(run_dir: Path) -> None:
    none = {"A": 0, "B": 0, "C": 0}
    refused = cli("calibration", "freeze", "--run-dir", run_dir)
    assert refused.exit_code == 2 and "no app.sqlite yet" in refused.stderr
    rate_all(run_dir, "local", {"sub-01": {"A": 2, "B": 0, "C": 0}, "sub-02": none})
    by_label = uids(run_dir)
    result = cli("calibration", "freeze", "--run-dir", run_dir, "--version", "pilot-1")
    assert result.exit_code == 0, result.output
    assert result.stdout == f"{run_dir / CALIBRATION_FILE}\n"
    written = yaml.safe_load((run_dir / CALIBRATION_FILE).read_text(encoding="utf-8"))
    assert written["version"] == "pilot-1"
    # the two rated subjects; the one whose render failed is not a calibration subject
    assert sorted(subject["subject_uid"] for subject in written["subjects"]) == sorted(
        [by_label["sub-01"], by_label["sub-02"]]
    )
    again = cli("calibration", "freeze", "--run-dir", run_dir)
    assert again.exit_code == 2 and again.stderr.startswith("invalid: calibration.yaml exists")
    unknown = cli("calibration", "freeze", "--run-dir", run_dir.parent / "none")
    assert unknown.exit_code == 2


def test_calibration_freeze_of_named_subjects(run_dir: Path) -> None:
    none = {"A": 0, "B": 0, "C": 0}
    rate_all(run_dir, "local", {"sub-01": none, "sub-02": none})
    by_label = uids(run_dir)
    result = cli("calibration", "freeze", "--run-dir", run_dir, "--subject", by_label["sub-02"])
    assert result.exit_code == 0, result.output
    written = yaml.safe_load((run_dir / CALIBRATION_FILE).read_text(encoding="utf-8"))
    assert [subject["subject_uid"] for subject in written["subjects"]] == [by_label["sub-02"]]


# --- export ---------------------------------------------------------------------------------------


def test_export_prints_the_files_it_wrote(run_dir: Path, tmp_path: Path) -> None:
    result = cli("export", "--run-dir", run_dir)
    assert result.exit_code == 0, result.output
    names = [f"{name}.csv" for name in EXPORT_TABLES] + [MANIFEST]
    assert result.stdout.splitlines() == [str(run_dir / "export" / name) for name in names]
    assert result.stderr == ""
    subjects = pd.read_csv(run_dir / "export" / "subjects.csv", dtype=str)
    assert sorted(subjects["subject_label"]) == list(LABELS)
    assert subjects.set_index("subject_label").loc["sub-03", "render_reason"] == "missing surface"
    manifest = json.loads((run_dir / "export" / MANIFEST).read_text(encoding="utf-8"))
    assert manifest["profile"] == "analysis" and manifest["format"] == "csv"

    out = tmp_path / "elsewhere"
    parquet = cli("export", "--run-dir", run_dir, "--out", out, "--format", "parquet")
    assert parquet.exit_code == 0, parquet.output
    assert sorted(path.name for path in out.iterdir()) == sorted(
        [f"{name}.parquet" for name in EXPORT_TABLES] + [MANIFEST]
    )
    assert len(pd.read_parquet(out / "subjects.parquet")) == 3


def test_export_stops_with_exit_2_for_what_it_cannot_do(run_dir: Path, tmp_path: Path) -> None:
    wrong = cli("export", "--run-dir", run_dir, "--format", "xlsx")
    assert wrong.exit_code == 2 and "is csv or parquet" in said(wrong.stderr)
    release = cli("export", "--run-dir", run_dir, "--release")
    assert release.exit_code == 2 and release.stderr.startswith("invalid: ")
    assert not (run_dir / "release").exists()
    missing = cli("export", "--run-dir", tmp_path / "none")
    assert missing.exit_code == 2 and missing.stderr.startswith("missing: ")
    taken = tmp_path / "a-file"
    taken.write_text("")
    blocked = cli("export", "--run-dir", run_dir, "--out", taken)
    assert blocked.exit_code == 2
    assert blocked.stderr.startswith("invalid: the run directory cannot be used: ")


@pytest.mark.parametrize("where", ["subjects directory", "subject", "below a subject", "link"])
def test_export_refuses_an_output_in_the_subjects_directory(
    run_dir: Path, study: Path, tmp_path: Path, where: str
) -> None:
    subjects_dir = study / "sourcedata" / "freesurfer"
    out = {
        "subjects directory": subjects_dir,
        "subject": subjects_dir / "sub-01",
        "below a subject": subjects_dir / "sub-01" / "tables",
        "link": tmp_path / "link",
    }[where]
    if where == "link":
        out.symlink_to(subjects_dir / "sub-02")
    before = snapshot_tree(study)
    result = cli("export", "--run-dir", run_dir, "--out", out)
    assert result.exit_code == 2, result.output
    assert result.stdout == "" and len(result.stderr.splitlines()) == 1
    assert result.stderr.startswith("invalid: ")
    assert "lies inside the subjects directory" in result.stderr
    # the same from Python
    with pytest.raises(ConfigError, match="lies inside the subjects directory") as raised:
        export_run(run_dir, out)
    assert raised.value.reason == "invalid"
    assert snapshot_tree(study) == before
    # beside the subjects directory, inside the fMRIPrep output, is a place like any other
    beside = cli("export", "--run-dir", run_dir, "--out", study / "tables")
    assert beside.exit_code == 0, beside.output
    assert (study / "tables" / MANIFEST).is_file()


@pytest.fixture
def linked(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A subjects directory whose two subjects and whose template are links into a store,
    and the run that ``review`` makes of it: ``(subjects directory, store, run)``."""
    store = tmp_path / "store"
    root = write_linked_tree(tmp_path / "linked" / "study", store, ["sub-01", "sub-02"])
    result = cli("review", root, "--no-serve", "--jobs", 1)
    assert result.exit_code == 0, result.output
    # the template comes after the review: a light one has too few vertices to score with
    write_light_template(store / "fsaverage")
    (root / "fsaverage").symlink_to(store / "fsaverage", target_is_directory=True)
    return root, store, tmp_path / "linked" / DEFAULT_RUN_DIR


@pytest.mark.parametrize(
    "where", ["link", "below the link", "target", "below the target", "template", "alias"]
)
def test_export_refuses_an_output_in_what_a_subject_links_to(
    linked: tuple[Path, Path, Path], tmp_path: Path, where: str
) -> None:
    """A subject directory that is a link resolves outside the subjects directory. What it
    leads to is read as a source all the same, and so is what the template link leads to."""
    root, store, run_dir = linked
    out = {
        "link": root / "sub-02",
        "below the link": root / "sub-01" / "exp",
        "target": store / "sub-02",
        "below the target": store / "sub-01" / "exp",
        "template": store / "fsaverage",
        "alias": tmp_path / "alias",
    }[where]
    if where == "alias":
        out.symlink_to(store / "sub-01" / "surf")
    before = snapshot_tree(store), snapshot_tree(root)
    result = cli("export", "--run-dir", run_dir, "--out", out)
    assert result.exit_code == 2, result.output
    assert result.stdout == "" and len(result.stderr.splitlines()) == 1
    assert result.stderr.startswith("invalid: the output directory ")
    assert "which is only read" in result.stderr
    with pytest.raises(ConfigError, match="which is only read") as raised:
        export_run(run_dir, out)
    assert raised.value.reason == "invalid"
    assert (snapshot_tree(store), snapshot_tree(root)) == before
    # beside what the links lead to is a place like any other
    beside = cli("export", "--run-dir", run_dir, "--out", store / "tables")
    assert beside.exit_code == 0, beside.output


def test_a_run_directory_in_what_a_subject_links_to_is_refused_before_it_is_made(
    linked: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    root, store, _run_dir = linked
    before = snapshot_tree(store), snapshot_tree(root)
    for place in (store / "sub-01" / "run", root / "sub-02" / "run", store / "fsaverage" / "r"):
        for command in (
            ("review", root, "--run-dir", place, "--no-serve", "--jobs", 1),
            ("inventory", "--run-dir", place, "--source", root),
        ):
            result = cli(*command)
            assert result.exit_code == 2, result.output
            assert result.stderr.splitlines()[-1].startswith("invalid: the run directory ")
            assert "which is only read" in result.stderr
    # a run directory in the fsaverage directory that the call names
    given = tmp_path / "templates" / "fsaverage"
    result = cli("review", root, "--run-dir", given / "run", "--fsaverage", given, "--no-serve")
    assert result.exit_code == 2 and "lies inside the fsaverage directory" in result.stderr
    assert not given.exists()
    assert (snapshot_tree(store), snapshot_tree(root)) == before


def test_a_run_that_was_moved_into_its_source_is_not_written_to(run_dir: Path, study: Path) -> None:
    """``export`` without ``--out``, ``serve``, ``finalize`` and ``calibration freeze`` open a
    run without building its sources. Nothing the package writes goes into a source."""
    rate_all(run_dir, "ada", {"sub-01": dict(A=0, B=0, C=0), "sub-02": dict(A=1, B=0, C=0)})
    moved = study / "sourcedata" / "freesurfer" / "run-moved"
    run_dir.rename(moved)
    before = snapshot_tree(study)
    for command in (
        ("export", "--run-dir", moved),
        ("export", "--run-dir", moved, "--format", "parquet"),
        ("serve", "--run-dir", moved, "--no-browser", "--port", 0),
        ("finalize", "--run-dir", moved),
        ("calibration", "freeze", "--run-dir", moved),
        ("review", study, "--run-dir", moved, "--no-serve"),
    ):
        result = cli(*command)
        assert result.exit_code == 2, (command, result.output)
        assert result.stdout == "" and len(result.stderr.splitlines()) == 1, command
        assert "lies inside the subjects directory" in result.stderr, command
    with pytest.raises(ConfigError, match="lies inside the subjects directory"):
        create_app(moved)
    with pytest.raises(ConfigError, match="lies inside the subjects directory"):
        export_run(moved)
    assert snapshot_tree(study) == before
    assert cli("status", "--run-dir", moved).exit_code == 0  # reading is no writing


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_export_refuses_an_output_among_the_files_of_the_run(
    run_dir: Path, study: Path, fmt: str
) -> None:
    """``clusters.parquet`` and ``scores.parquet`` are names of the run's own tables and of
    an export: an export into ``tables/`` used to leave a run that no phase could read."""
    before = snapshot_tree(run_dir)
    for out in (run_dir / "tables", run_dir, run_dir / "media" / "x", run_dir / "mine"):
        result = cli("export", "--run-dir", run_dir, "--format", fmt, "--out", out)
        assert result.exit_code == 2, result.output
        assert result.stdout == "" and len(result.stderr.splitlines()) == 1
        assert "lies inside the run directory" in result.stderr
        assert "export/ or release/" in result.stderr
        with pytest.raises(ConfigError, match="lies inside the run directory"):
            export_run(run_dir, out, fmt="csv" if fmt == "csv" else "parquet")
    assert snapshot_tree(run_dir) == before
    # the two directories of an export, also when they are named, and places below them
    for out in (run_dir / "export", run_dir / "release", run_dir / "export" / "second"):
        result = cli("export", "--run-dir", run_dir, "--format", fmt, "--out", out)
        assert result.exit_code == 0, result.output
        assert (out / MANIFEST).is_file()
    # the run is as usable as before
    again = cli("review", study, "--no-serve", "--jobs", 1)
    assert again.exit_code == 0, again.output


def test_export_refuses_an_output_where_the_subjects_were(
    run_dir: Path, study: Path, tmp_path: Path
) -> None:
    study.rename(tmp_path / "moved")  # the data is not mounted: the root is what is known
    result = cli("export", "--run-dir", run_dir, "--out", study / "sourcedata" / "tables")
    assert result.exit_code == 2 and "lies inside the subjects directory" in result.stderr
    assert not study.exists()
    assert cli("export", "--run-dir", run_dir).exit_code == 0


def test_export_of_a_census_run_refuses_an_output_in_a_local_source(
    study: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "census"
    Run(run_dir, local(study), key=HashKey(bytes(range(1, 33)), "dataset-key")).close()
    assert load_config(run_dir).sources[0].root is None  # the root is in the catalog only
    before = snapshot_tree(study)
    out = study / "sourcedata" / "freesurfer" / "sub-01"
    result = cli("export", "--run-dir", run_dir, "--out", out)
    assert result.exit_code == 2, result.output
    assert result.stderr.startswith("invalid: ")
    assert "lies inside the subjects directory" in result.stderr
    assert snapshot_tree(study) == before
    elsewhere = cli("export", "--run-dir", run_dir, "--out", tmp_path / "out")
    assert elsewhere.exit_code == 0, elsewhere.output
    # the run itself moved into the source: its own export/ is an output there
    moved = study / "sourcedata" / "freesurfer" / "census"
    run_dir.rename(moved)
    before = snapshot_tree(study)
    result = cli("export", "--run-dir", moved)
    assert result.exit_code == 2, result.output
    assert "lies inside the subjects directory" in result.stderr
    # the roots of a census run are read from its catalog, whose write-ahead log is opened
    assert not (moved / "export").exists()
    assert {name: at for name, at in snapshot_tree(study).items() if "/census" not in name} == {
        name: at for name, at in before.items() if "/census" not in name
    }


def test_the_commands_are_listed_in_the_help() -> None:
    result = cli("--help")
    listed = said(result.output)
    for command in ("review", "serve", "finalize", "calibration", "export"):
        assert f" {command} " in listed
    assert "freeze" in said(cli("calibration", "--help").output)


def test_serve_review_ignores_the_signals_before_its_closing_line(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``stopped`` (the command line's ``CommandSignals.served``) is called once the app
    has stopped and before the closing line, so that a Ctrl-C that comes as the line is
    printed has nothing to interrupt (review of the quick start: one came between the
    line and the end of the call, and the command ended with 130 after its closing line)."""
    from surfcensus import review as reviewing
    from surfcensus.app.server import ServerHandle

    order: list[str] = []

    class App(reviewing.ReviewApp):
        def stop(self) -> None:
            order.append("app stopped")

    def progress(run_dir: Path, rater: str | None) -> tuple[int, int]:
        order.append("closing line")
        return 0, 1

    monkeypatch.setattr(reviewing, "_progress_of", progress)
    with Run(run_dir) as run:  # a handle without a server: it is not running
        app = App(ServerHandle(url="http://127.0.0.1:1/", port=1), run_dir, 0.0)
        reviewing.serve_review(
            run, app, open_browser=False, stopped=lambda: order.append("signals ignored")
        )
    assert order == ["app stopped", "signals ignored", "closing line"]
    assert capsys.readouterr().err.splitlines()[-1].startswith("0 rated, 1 to go.")
