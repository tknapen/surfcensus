"""Tests of surfcensus/batches.py: order, failures that stay with their subject, leases that
are always released, and the same results in this process and in worker processes.

The functions that run in workers are at module level, because a spawned process imports
this module to find them. The leases write their files on entry and delete them on exit,
as a remote lease does, so work that outlives its lease fails.
"""

from __future__ import annotations

import io
import logging
import os
import signal
import sys
import time
from collections.abc import Collection, Iterator, Mapping, Sequence
from concurrent.futures.process import BrokenProcessPool
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from types import MappingProxyType

import pytest

from surfcensus.batches import RemoteTraceback, WorkerError, run_batches
from surfcensus.log import get_logger
from surfcensus.sources.base import FileRole, SourceError

Outcome = tuple[str, object]


class Leases:
    """A lease function that keeps a log, and fails or interrupts for chosen subjects."""

    def __init__(
        self,
        root: Path,
        *,
        refused: Collection[str] = (),
        changed: Collection[str] = (),
        interrupted: Collection[str] = (),
        events: list[tuple[str, str]] | None = None,
    ) -> None:
        self.root = root
        self.refused = refused
        self.changed = changed
        self.interrupted = interrupted
        self.events: list[tuple[str, str]] = [] if events is None else events
        self.open: set[str] = set()
        self.peak = 0
        self.seen_at_release: dict[str, list[str]] = {}

    def paths(self, unit: str) -> dict[FileRole, Path]:
        return {
            FileRole.T1: self.root / unit / "mri" / "T1.mgz",
            FileRole.LH_PIAL: self.root / unit / "surf" / "lh.pial",
        }

    def __call__(self, unit: str) -> AbstractContextManager[Mapping[FileRole, Path]]:
        if unit in self.refused:
            raise SourceError("not found", f"{unit} is not in the catalog")
        return self._lease(unit)

    @contextmanager
    def _lease(self, unit: str) -> Iterator[Mapping[FileRole, Path]]:
        if unit in self.changed:
            raise SourceError("changed", f"{unit}/mri/T1.mgz")
        if unit in self.interrupted:
            raise KeyboardInterrupt
        paths = self.paths(unit)
        for role, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{unit} {role.value}")
        self.events.append(("lease", unit))
        self.open.add(unit)
        self.peak = max(self.peak, len(self.open))
        try:
            # a mapping that pickle refuses: run_batches may not send it to a worker as it is
            yield MappingProxyType(paths)
        finally:
            self.seen_at_release[unit] = sorted(p.name for p in self.root.glob("finished-*"))
            for path in paths.values():
                path.unlink()
            self.open.discard(unit)
            self.events.append(("release", unit))

    def leased(self) -> list[str]:
        return [unit for event, unit in self.events if event == "lease"]

    def released(self) -> list[str]:
        return [unit for event, unit in self.events if event == "release"]


class Commits:
    """A commit function that keeps what it was given, batch by batch."""

    def __init__(
        self, leases: Leases | None = None, *, fail_at: int | None = None, error: type = OSError
    ) -> None:
        self.batches: list[list[Outcome]] = []
        self.open_leases: list[set[str]] = []
        self.leases = leases
        self.fail_at = fail_at
        self.error = error

    def __call__(self, results: Sequence[Outcome]) -> None:
        if self.fail_at == len(self.batches):
            raise self.error("commit failed")
        self.batches.append(list(results))
        if self.leases is not None:
            self.open_leases.append(set(self.leases.open))
            self.leases.events.append(("commit", ",".join(unit for unit, _ in results)))

    def flat(self) -> list[Outcome]:
        return [result for batch in self.batches for result in batch]


def plain(outcome: object) -> object:
    """An outcome in a form that compares by value: exceptions do not."""
    if isinstance(outcome, BaseException):
        return (type(outcome), outcome.args, dict(vars(outcome)), str(outcome))
    return outcome


def plain_batches(commits: Commits) -> list[list[Outcome]]:
    return [[(unit, plain(outcome)) for unit, outcome in batch] for batch in commits.batches]


class Tagged(Exception):
    """An error that pickle reads back with another message: ``Tagged("a", "b")`` is stored
    as ``args == ("a: b",)`` and rebuilt as ``Tagged("a: b")``, which reads ``a: b: ``."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


def describe(unit: str, files: Mapping[FileRole, Path]) -> tuple[str, list[tuple[str, str]]]:
    """Read every leased file. A unit named ``slow...`` finishes after the ones behind it."""
    if unit.startswith("slow"):
        time.sleep(0.4)
    return unit, [(role.value, path.read_text()) for role, path in files.items()]


def fragile(unit: str, files: Mapping[FileRole, Path]) -> object:
    """Fail in a way that the name of the unit selects."""
    if unit.startswith("value"):
        raise ValueError(unit, 3)
    if unit.startswith("source"):
        raise SourceError("unreadable", f"{unit}: not a volume")
    if unit.startswith("tagged"):
        raise Tagged("low coverage", unit)
    if unit.startswith("os"):
        raise FileNotFoundError(2, "No such file or directory", f"{unit}/surf/lh.white")
    if unit.startswith("local"):

        class Local(Exception):
            pass

        raise Local(f"{unit} has a problem")
    if unit.startswith("closure"):
        error = RuntimeError(f"{unit} failed")
        error.hook = lambda: None
        raise error
    if unit.startswith("interrupt"):
        raise KeyboardInterrupt
    if unit.startswith("die"):
        os._exit(7)
    if unit.startswith("lambda"):
        return lambda: unit
    return describe(unit, files)


def counted(unit: str, files: Mapping[FileRole, Path]) -> str:
    """Leave one file per call. ``die-after-a-b`` kills its process once ``a`` and ``b`` have
    run and their results have had time to reach the parent."""
    root = files[FileRole.T1].parents[2]
    (root / f"ran-{unit}-{os.getpid()}-{time.time_ns()}").touch()
    if unit.startswith("die-after-"):
        others = unit.removeprefix("die-after-").split("-")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if all(list(root.glob(f"ran-{other}-*")) for other in others):
                break
            time.sleep(0.01)
        time.sleep(0.5)
        os._exit(7)
    return unit


def process_state(unit: str, files: Mapping[FileRole, Path]) -> tuple[int, bool]:
    return os.getpid(), signal.getsignal(signal.SIGINT) == signal.SIG_IGN


def interrupt_parent(unit: str, files: Mapping[FileRole, Path]) -> str:
    """Send SIGINT to the parent from the unit named ``signal``, then go on reading the leased
    files for a while and leave a marker next to the subject directories."""
    if unit == "signal":
        os.kill(os.getppid(), signal.SIGINT)
        time.sleep(0.6)
    text = files[FileRole.T1].read_text()
    (files[FileRole.T1].parents[2] / f"finished-{unit}").write_text(text)
    return text


def chatty(unit: str, files: Mapping[FileRole, Path]) -> str:
    """Log to the package logger at two levels; a unit named ``value...`` fails after that."""
    log = get_logger()

    class Count:  # an argument that pickle refuses: the message has to be merged in the worker
        def __str__(self) -> str:
            return f"{len(files)} files"

    log.info("%s: read %s", unit, Count())
    log.warning("%s: no brainmask under %s", unit, "/data/study/derivatives")
    if unit.startswith("value"):
        try:
            raise ValueError(unit)
        except ValueError:
            log.warning("%s: giving up", unit, exc_info=True)
            raise
    return unit


class Redact(logging.Filter):
    """Rewrites a record on the package logger, as the redacting filter of a census run does."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = record.getMessage().replace("/data/study", "<root>")
        record.args = None
        return True


def expected(leases: Leases, unit: str) -> tuple[str, list[tuple[str, str]]]:
    return unit, [(role.value, f"{unit} {role.value}") for role in leases.paths(unit)]


@pytest.mark.parametrize("jobs", [1, 2])
def test_order_is_preserved_within_and_across_batches(tmp_path: Path, jobs: int) -> None:
    units = ["slow-g", "f", "e", "slow-d", "c", "b", "a"]
    leases = Leases(tmp_path)
    commits = Commits(leases)

    run_batches(units, leases, describe, commits, batch_subjects=3, jobs=jobs)

    assert [[unit for unit, _ in batch] for batch in commits.batches] == [
        ["slow-g", "f", "e"],
        ["slow-d", "c", "b"],
        ["a"],
    ]
    # each result under its own unit, although the slow units finish last in their batch
    assert commits.flat() == [(unit, expected(leases, unit)) for unit in units]
    assert leases.leased() == units


def test_a_batch_is_leased_worked_committed_and_released_in_that_order(tmp_path: Path) -> None:
    events: list[tuple[str, str]] = []
    leases = Leases(tmp_path, events=events)
    commits = Commits(leases)

    def work(unit: str, files: Mapping[FileRole, Path]) -> str:
        events.append(("work", unit))
        return unit

    run_batches(["a", "b", "c", "d", "e"], leases, work, commits, batch_subjects=2, jobs=1)

    def batch(first: str, second: str | None = None) -> list[tuple[str, str]]:
        units = [first] if second is None else [first, second]
        return [
            *[("lease", unit) for unit in units],
            *[("work", unit) for unit in units],
            ("commit", ",".join(units)),
            *[("release", unit) for unit in reversed(units)],
        ]

    assert events == [*batch("a", "b"), *batch("c", "d"), *batch("e")]
    # the leases of a batch are still open when it is committed, and no others
    assert commits.open_leases == [{"a", "b"}, {"c", "d"}, {"e"}]


@pytest.mark.parametrize("jobs", [1, 2])
def test_no_more_leases_are_open_than_one_batch(tmp_path: Path, jobs: int) -> None:
    leases = Leases(tmp_path)
    units = [f"u{i}" for i in range(7)]

    run_batches(units, leases, describe, Commits(), batch_subjects=3, jobs=jobs)

    assert leases.peak == 3
    assert leases.open == set()
    assert sorted(leases.released()) == sorted(units)
    assert not list(tmp_path.rglob("*.mgz"))


@pytest.mark.parametrize("jobs", [1, 2])
def test_a_failing_unit_is_handed_to_commit_and_the_batch_goes_on(
    tmp_path: Path, jobs: int
) -> None:
    units = ["a", "value-b", "c", "changed-d", "refused-e", "f", "source-g"]
    leases = Leases(tmp_path, changed=["changed-d"], refused=["refused-e"])
    commits = Commits()

    run_batches(units, leases, fragile, commits, batch_subjects=4, jobs=jobs)

    assert [len(batch) for batch in commits.batches] == [4, 3]
    outcomes = dict(commits.flat())
    assert list(outcomes) == units
    for unit in ("a", "c", "f"):
        assert outcomes[unit] == expected(leases, unit)

    value = outcomes["value-b"]
    assert type(value) is ValueError
    assert value.args == ("value-b", 3)

    changed = outcomes["changed-d"]
    assert type(changed) is SourceError
    assert (changed.reason, changed.detail) == ("changed", "changed-d/mri/T1.mgz")
    refused = outcomes["refused-e"]
    assert type(refused) is SourceError
    assert refused.reason == "not found"

    source = outcomes["source-g"]
    assert type(source) is SourceError
    assert (source.reason, source.detail) == ("unreadable", "source-g: not a volume")
    assert str(source) == "unreadable: source-g: not a volume"

    # a subject whose lease failed is never worked on, and every lease that was taken is released
    assert leases.leased() == ["a", "value-b", "c", "f", "source-g"]
    assert sorted(leases.released()) == sorted(leases.leased())
    assert leases.open == set()


def test_a_batch_whose_leases_all_fail_is_still_committed(tmp_path: Path) -> None:
    leases = Leases(tmp_path, changed=["a", "b"])
    commits = Commits()

    run_batches(["a", "b", "c"], leases, describe, commits, batch_subjects=2, jobs=1)

    assert [type(outcome) for _, outcome in commits.batches[0]] == [SourceError, SourceError]
    assert commits.batches[1] == [("c", expected(leases, "c"))]


def test_one_job_and_two_jobs_commit_the_same_results(tmp_path: Path) -> None:
    units = [
        "a",
        "value-1",
        "slow-b",
        "source-2",
        "changed-3",
        "os-4",
        "c",
        "tagged-5",
        "refused-6",
        "d",
        "value-7",
    ]
    results = []
    for jobs in (1, 2):
        leases = Leases(tmp_path / str(jobs), changed=["changed-3"], refused=["refused-6"])
        commits = Commits()
        run_batches(units, leases, fragile, commits, batch_subjects=4, jobs=jobs)
        results.append(plain_batches(commits))
        assert leases.open == set()

    assert results[0] == results[1]
    kinds = [type(outcome) for _, outcome in commits.flat() if isinstance(outcome, Exception)]
    assert kinds == [
        ValueError,
        SourceError,
        SourceError,
        FileNotFoundError,
        Tagged,
        SourceError,
        ValueError,
    ]


def test_one_job_works_in_this_process_and_more_jobs_in_reused_workers(tmp_path: Path) -> None:
    units = [f"u{i}" for i in range(6)]
    before = signal.getsignal(signal.SIGINT)

    here = Commits()
    run_batches(units, Leases(tmp_path / "1"), process_state, here, batch_subjects=2, jobs=1)
    assert [outcome for _, outcome in here.flat()] == [(os.getpid(), False)] * 6
    assert signal.getsignal(signal.SIGINT) is before

    there = Commits()
    run_batches(units, Leases(tmp_path / "2"), process_state, there, batch_subjects=2, jobs=2)
    pids = {outcome[0] for _, outcome in there.flat()}
    assert os.getpid() not in pids
    # three batches, two processes: the workers outlive a batch
    assert 1 <= len(pids) <= 2
    # the workers leave Ctrl-C to the parent
    assert all(outcome[1] for _, outcome in there.flat())
    assert signal.getsignal(signal.SIGINT) is before


def test_an_exception_of_a_worker_arrives_as_its_own_class(tmp_path: Path) -> None:
    units = ["source-a", "tagged-b", "os-c", "value-d"]
    commits = Commits()

    run_batches(units, Leases(tmp_path), fragile, commits, batch_subjects=8, jobs=2)
    outcomes = dict(commits.flat())

    # the error classes of the package take (reason, detail) and keep one string in args,
    # so pickle alone cannot read them back
    source = outcomes["source-a"]
    assert type(source) is SourceError
    assert source.args == ("unreadable: source-a: not a volume",)
    assert vars(source) == {"reason": "unreadable", "detail": "source-a: not a volume"}

    tagged = outcomes["tagged-b"]
    assert type(tagged) is Tagged
    assert str(tagged) == "low coverage: tagged-b"
    assert tagged.reason == "low coverage"

    missing = outcomes["os-c"]
    assert type(missing) is FileNotFoundError
    assert (missing.errno, missing.filename) == (2, "os-c/surf/lh.white")
    assert str(missing) == "[Errno 2] No such file or directory: 'os-c/surf/lh.white'"

    for unit, outcome in outcomes.items():
        cause = outcome.__cause__
        assert isinstance(cause, RemoteTraceback), unit
        assert "in fragile" in str(cause)
        assert type(outcome).__name__ in str(cause)


def test_an_exception_that_cannot_leave_its_worker_arrives_as_a_stand_in(tmp_path: Path) -> None:
    units = ["a", "local-b", "closure-c", "d"]
    leases = Leases(tmp_path)
    commits = Commits()

    run_batches(units, leases, fragile, commits, batch_subjects=8, jobs=2)
    outcomes = dict(commits.flat())

    local = outcomes["local-b"]
    assert type(local) is WorkerError
    assert local.type_name == "fragile.<locals>.Local"
    assert local.message == "local-b has a problem"
    assert str(local) == "fragile.<locals>.Local: local-b has a problem"
    assert "in fragile" in str(local.__cause__)

    closure = outcomes["closure-c"]
    assert type(closure) is WorkerError
    assert (closure.type_name, closure.message) == ("RuntimeError", "closure-c failed")

    # and the pool is intact for the others
    assert outcomes["a"] == expected(leases, "a")
    assert outcomes["d"] == expected(leases, "d")

    # in this process nothing has to be sent, and the exception arrives as it was raised
    here = Commits()
    run_batches(["local-b"], Leases(tmp_path / "1"), fragile, here, batch_subjects=8, jobs=1)
    raised = here.flat()[0][1]
    assert type(raised).__name__ == "Local"
    assert raised.__cause__ is None


def test_a_result_that_cannot_be_pickled_fails_its_own_unit_only(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    leases = Leases(tmp_path)
    commits = Commits()

    with caplog.at_level(logging.WARNING, logger="surfcensus"):
        run_batches(["a", "lambda-b", "c"], leases, fragile, commits, batch_subjects=8, jobs=2)
    outcomes = dict(commits.flat())

    assert isinstance(outcomes["lambda-b"], Exception)
    assert not isinstance(outcomes["lambda-b"], BrokenProcessPool)
    assert "lambda" in str(outcomes["lambda-b"])
    assert outcomes["a"] == expected(leases, "a")
    assert outcomes["c"] == expected(leases, "c")
    assert not caplog.records


def test_a_worker_that_dies_is_charged_to_its_own_unit_only(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # the slow units are running or waiting when the worker of die-b dies, and the pool
    # fails them all with it
    units = ["slow-a", "die-b", "slow-c", "slow-d", "e", "f", "g"]
    leases = Leases(tmp_path)
    commits = Commits(leases)

    with caplog.at_level(logging.WARNING, logger="surfcensus"):
        run_batches(units, leases, fragile, commits, batch_subjects=5, jobs=2)
    outcomes = dict(commits.flat())

    assert list(outcomes) == units
    assert type(outcomes["die-b"]) is BrokenProcessPool
    for unit in ("slow-a", "slow-c", "slow-d", "e", "f", "g"):
        assert outcomes[unit] == expected(leases, unit), unit
    assert commits.open_leases == [{"slow-a", "die-b", "slow-c", "slow-d", "e"}, {"f", "g"}]
    assert leases.open == set()

    (record,) = caplog.records
    assert "a worker process died" in record.getMessage()
    assert isinstance(record.args, tuple)
    lost = record.args[0]
    assert isinstance(lost, int)
    assert 2 <= lost <= 5


def test_only_the_lost_subjects_run_again(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    units = ["a", "b", "die-after-a-b"]
    leases = Leases(tmp_path)
    commits = Commits(leases)

    with caplog.at_level(logging.WARNING, logger="surfcensus"):
        run_batches(units, leases, counted, commits, batch_subjects=3, jobs=2)
    outcomes = dict(commits.flat())
    runs = {unit: len(list(tmp_path.glob(f"ran-{unit}-*"))) for unit in units}

    assert (outcomes["a"], outcomes["b"]) == ("a", "b")
    assert type(outcomes["die-after-a-b"]) is BrokenProcessPool
    (record,) = caplog.records
    assert isinstance(record.args, tuple)
    lost = record.args[0]
    # a and b have normally reached the parent when the worker dies (lost == 1); if one had
    # not, it is lost with the pool and runs a second time, and that is all that runs again
    assert isinstance(lost, int)
    assert runs["die-after-a-b"] == 2
    assert sum(runs.values()) == len(units) + lost, runs
    assert all(count in (1, 2) for count in runs.values()), runs


@pytest.mark.parametrize("jobs", [1, 2])
def test_what_work_logs_passes_the_filters_of_the_package_logger(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, jobs: int
) -> None:
    units = ["a", "value-b", "c"]
    commits = Commits()
    logger = get_logger()
    redact = Redact()
    logger.addFilter(redact)
    try:
        with caplog.at_level(logging.WARNING, logger="surfcensus"):
            run_batches(units, Leases(tmp_path), chatty, commits, batch_subjects=2, jobs=jobs)
    finally:
        logger.removeFilter(redact)

    # every warning of every subject, the failed one included, and nothing below the level
    # that the logger has in this process
    assert sorted(record.getMessage() for record in caplog.records) == [
        "a: no brainmask under <root>/derivatives",
        "c: no brainmask under <root>/derivatives",
        "value-b: giving up",
        "value-b: no brainmask under <root>/derivatives",
    ]
    assert {(record.name, record.levelname) for record in caplog.records} == {
        ("surfcensus", "WARNING")
    }
    assert "/data/study" not in caplog.text
    assert "ValueError: value-b" in caplog.text
    assert [type(outcome) for _, outcome in commits.flat()] == [str, ValueError, str]


@pytest.mark.parametrize("jobs", [1, 2])
def test_work_logs_at_the_level_of_the_package_logger(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, jobs: int
) -> None:
    with caplog.at_level(logging.INFO, logger="surfcensus"):
        run_batches(["a", "b"], Leases(tmp_path), chatty, Commits(), batch_subjects=2, jobs=jobs)

    assert sorted((record.levelname, record.getMessage()) for record in caplog.records) == [
        ("INFO", "a: read 2 files"),
        ("INFO", "b: read 2 files"),
        ("WARNING", "a: no brainmask under /data/study/derivatives"),
        ("WARNING", "b: no brainmask under /data/study/derivatives"),
    ]


def run_interrupted(
    tmp_path: Path, where: str, jobs: int
) -> tuple[Leases, Commits, list[tuple[str, str]]]:
    """Seven units in batches of three; the second batch is interrupted at ``where``."""
    stop = "interrupt-4" if where == "work" else "u4"
    units = ["u0", "u1", "u2", "u3", stop, "u5", "u6"]
    events: list[tuple[str, str]] = []
    leases = Leases(tmp_path, interrupted=[stop] if where == "lease" else [], events=events)
    commits = Commits(leases, fail_at=1 if where == "commit" else None, error=KeyboardInterrupt)

    def work(unit: str, files: Mapping[FileRole, Path]) -> object:
        events.append(("work", unit))
        return fragile(unit, files)

    with pytest.raises(KeyboardInterrupt):
        run_batches(
            units, leases, work if jobs == 1 else fragile, commits, batch_subjects=3, jobs=jobs
        )
    return leases, commits, events


@pytest.mark.parametrize(
    ("where", "jobs"), [("lease", 1), ("work", 1), ("work", 2), ("commit", 1), ("commit", 2)]
)
def test_keyboard_interrupt_releases_the_leases_and_loses_one_batch(
    tmp_path: Path, where: str, jobs: int
) -> None:
    leases, commits, events = run_interrupted(tmp_path, where, jobs)

    # the batch before is committed, the interrupted one is not, the one after is not started
    assert [[unit for unit, _ in batch] for batch in commits.batches] == [["u0", "u1", "u2"]]
    assert commits.flat() == [(unit, expected(leases, unit)) for unit in ("u0", "u1", "u2")]
    second = (
        ["u3"] if where == "lease" else ["u3", "interrupt-4" if where == "work" else "u4", "u5"]
    )
    assert leases.leased() == ["u0", "u1", "u2", *second]
    assert sorted(leases.released()) == sorted(leases.leased())
    assert leases.open == set()
    assert not list(tmp_path.rglob("*.mgz"))

    if jobs == 1:
        worked = [unit for event, unit in events if event == "work"]
        assert (
            worked
            == {
                "lease": ["u0", "u1", "u2"],
                "work": ["u0", "u1", "u2", "u3", "interrupt-4"],
                "commit": ["u0", "u1", "u2", "u3", "u4", "u5"],
            }[where]
        )


@pytest.mark.skipif(sys.platform == "win32", reason="sends SIGINT with os.kill")
def test_ctrl_c_waits_for_running_work_before_the_leases_are_released(tmp_path: Path) -> None:
    units = ["a", "signal", "b", "c", "d", "e"]
    leases = Leases(tmp_path)
    commits = Commits()

    with pytest.raises(KeyboardInterrupt):
        run_batches(units, leases, interrupt_parent, commits, batch_subjects=4, jobs=2)

    assert commits.batches == []
    assert leases.leased() == ["a", "signal", "b", "c"]
    assert sorted(leases.released()) == ["a", "b", "c", "signal"]
    # the worker went on reading for 0.6 s after the signal; its lease, and every other one,
    # was released only after it had finished
    for unit in leases.released():
        assert "finished-signal" in leases.seen_at_release[unit], unit
    assert not list(tmp_path.rglob("*.mgz"))


@pytest.mark.parametrize("jobs", [1, 2])
def test_an_exception_of_commit_ends_the_run_and_releases_the_leases(
    tmp_path: Path, jobs: int
) -> None:
    leases = Leases(tmp_path)
    commits = Commits(fail_at=1)

    with pytest.raises(OSError, match="commit failed"):
        run_batches(
            ["a", "b", "c", "d", "e"], leases, describe, commits, batch_subjects=2, jobs=jobs
        )

    assert [[unit for unit, _ in batch] for batch in commits.batches] == [["a", "b"]]
    assert leases.leased() == ["a", "b", "c", "d"]
    assert leases.open == set()


def test_no_units_leases_and_commits_nothing(tmp_path: Path) -> None:
    leases = Leases(tmp_path)
    commits = Commits()

    for jobs in (1, 2):
        run_batches([], leases, describe, commits, batch_subjects=4, jobs=jobs)

    assert leases.events == []
    assert commits.batches == []


def test_units_may_be_any_sequence(tmp_path: Path) -> None:
    leases = Leases(tmp_path)
    commits = Commits()

    run_batches(("a", "b", "c"), leases, describe, commits, batch_subjects=2, jobs=1)

    assert commits.flat() == [(unit, expected(leases, unit)) for unit in ("a", "b", "c")]


@pytest.mark.parametrize(
    ("batch_subjects", "jobs", "message"),
    [(0, 1, "batch_subjects must be at least 1"), (4, 0, "jobs must be at least 1")],
)
def test_sizes_below_one_are_refused(
    tmp_path: Path, batch_subjects: int, jobs: int, message: str
) -> None:
    leases = Leases(tmp_path)
    with pytest.raises(ValueError, match=message):
        run_batches(["a"], leases, describe, Commits(), batch_subjects=batch_subjects, jobs=jobs)
    assert leases.events == []


def test_work_that_cannot_be_sent_to_a_worker_is_refused_before_any_lease(tmp_path: Path) -> None:
    leases = Leases(tmp_path)
    commits = Commits()

    def local_work(unit: str, files: Mapping[FileRole, Path]) -> str:
        return unit

    with pytest.raises(TypeError, match="picklable top-level function"):
        run_batches(["a", "b"], leases, local_work, commits, batch_subjects=2, jobs=2)
    assert leases.events == []
    assert commits.batches == []

    # in this process any callable will do
    run_batches(["a", "b"], leases, local_work, commits, batch_subjects=2, jobs=1)
    assert commits.flat() == [("a", "a"), ("b", "b")]


class Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.mark.parametrize("jobs", [1, 2])
def test_the_progress_bar_counts_every_subject_on_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: int
) -> None:
    terminal = Terminal()
    monkeypatch.setattr(sys, "stderr", terminal)
    units = ["a", "value-b", "changed-c", "d", "e"]
    leases = Leases(tmp_path, changed=["changed-c"])

    run_batches(units, leases, fragile, Commits(), batch_subjects=2, jobs=jobs, progress="render")

    shown = terminal.getvalue()
    assert "render:" in shown
    assert "0/5" in shown
    # failed subjects count too
    assert "5/5" in shown
    assert "subject" in shown


def test_no_progress_bar_without_a_description_or_a_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    terminal = Terminal()
    monkeypatch.setattr(sys, "stderr", terminal)
    run_batches(["a", "b"], Leases(tmp_path / "1"), describe, Commits(), batch_subjects=2, jobs=1)
    assert terminal.getvalue() == ""

    redirected = io.StringIO()
    monkeypatch.setattr(sys, "stderr", redirected)
    run_batches(
        ["a", "b"],
        Leases(tmp_path / "2"),
        describe,
        Commits(),
        batch_subjects=2,
        jobs=1,
        progress="render",
    )
    assert redirected.getvalue() == ""
