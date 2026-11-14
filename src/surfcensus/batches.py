"""Run a per-subject function over leased files, batch by batch.

Phases that read large source files (the fsaverage maps, rendering) go through
:func:`run_batches`. A batch is the unit of everything that bounds a census run: the source
files on disk at one time (a remote lease downloads on entry and deletes on exit), and the
work lost when the process is killed.

Workers return what they have to say: results, exceptions and the log records of the
package logger all come back to the parent, which alone stores, prints and logs. So one
filter on the package logger sees every record, whatever ``jobs`` is.

Standard library only, apart from ``tqdm`` when a progress bar is asked for.
"""

from __future__ import annotations

import copy
import logging
import multiprocessing
import pickle
import signal
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from surfcensus.log import get_logger
from surfcensus.sources.base import FileRole

R = TypeVar("R")
L = TypeVar("L")
"""What a lease yields for a unit: the local paths of its files (``Files``) for the phases of
the surfaces, a ``records.SessionWork`` for the pack phase of the EPI interface."""

Files = dict[FileRole, Path]


class WorkerError(Exception):
    """Stands in for an exception of a worker process that cannot be sent to the parent.

    Only an exception that neither pickle nor :class:`_Raised` can carry arrives as one, for
    example an instance of a class defined inside a function.
    """

    def __init__(self, type_name: str, message: str) -> None:
        super().__init__(type_name, message)

    @property
    def type_name(self) -> str:
        """The name of the class that was raised."""
        return self.args[0]

    @property
    def message(self) -> str:
        """``str()`` of the exception that was raised."""
        return self.args[1]

    def __str__(self) -> str:
        return f"{self.type_name}: {self.message}"


class RemoteTraceback(Exception):
    """The traceback of an exception raised in a worker process, as text.

    It is the ``__cause__`` of every exception that :func:`run_batches` brings back from a
    worker, so the place where it was raised is not lost. The text names source files by
    absolute path and is not scrubbed.
    """

    def __str__(self) -> str:
        return f"\n{self.args[0]}"


@dataclass(frozen=True)
class _Raised:
    """An exception of ``work`` on its way from a worker to the parent.

    The error classes of this package take ``(reason, detail)`` and pass one string to
    ``Exception.__init__``. pickle rebuilds an exception by calling its class with ``args``,
    so it cannot read such an exception back, and a process pool that receives one is broken
    for every subject still waiting (seen with ``SourceError`` on Python 3.13). ``parts``
    carries such an exception as class, ``args`` and instance attributes, and it is rebuilt
    without a call of ``__init__``.
    """

    exception: Exception | None
    """The exception itself, when pickle can write it and read it back."""
    parts: tuple[type[Exception], tuple[object, ...], dict[str, object]] | None
    remote_traceback: str

    @classmethod
    def of(cls, exc: Exception) -> _Raised:
        """The first form that comes back as the same class with the same message. It is
        tried here, in the worker: a form that failed in the parent would break the pool."""
        text = "".join(traceback.format_exception(exc))
        try:
            message = str(exc)
        except Exception:
            message = "<exception str() failed>"
        parts = (type(exc), exc.args, dict(vars(exc)))
        for candidate in (cls(exc, None, text), cls(None, parts, text)):
            try:
                back = pickle.loads(pickle.dumps(candidate)).rebuild()
                if type(back) is type(exc) and str(back) == message:
                    return candidate
            except Exception:
                continue
        return cls(WorkerError(type(exc).__qualname__, message), None, text)

    def rebuild(self) -> Exception:
        if self.exception is not None:
            exc = self.exception
        else:
            assert self.parts is not None
            kind, args, state = self.parts
            exc = kind.__new__(kind, *args)
            vars(exc).update(state)
        exc.__cause__ = RemoteTraceback(self.remote_traceback)
        return exc


class _Collector(logging.Handler):
    """Keeps the records of the package logger in a worker until its task hands them over."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # the message is merged and the traceback formatted here, as
            # logging.handlers.QueueHandler does: arguments and tracebacks may not pickle
            sent = copy.copy(record)
            sent.msg = record.getMessage()
            sent.args = None
            if record.exc_info:
                sent.exc_text = logging.Formatter().formatException(record.exc_info)
            sent.exc_info = None
            pickle.dumps(sent)
            self.records.append(sent)
        except Exception:
            self.handleError(record)

    def take(self) -> tuple[logging.LogRecord, ...]:
        records, self.records = self.records, []
        return tuple(records)


_collector: _Collector | None = None
"""Set in a worker process, None in the parent."""


def _start_worker(level: int) -> None:
    """Initializer of the workers.

    Ctrl-C reaches every process of the terminal's group; the parent alone acts on it, and
    a worker ends the subject it is on, so that no lease is released under a worker that
    still reads its files. The package logger gets the level it has in the parent and
    collects its records for the parent.

    SIGTERM gets its default: a command that has taken its SIGTERM ignores the signal
    (``stopping``), and a process spawned after that would inherit the ignoring, so that
    neither a process manager nor the pool itself (which ends broken workers with SIGTERM)
    could end it.
    """
    global _collector
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    _collector = _Collector()
    logger = get_logger()
    logger.handlers[:] = [_collector]
    logger.setLevel(level)
    logger.propagate = False


@dataclass(frozen=True)
class _Finished(Generic[R]):
    """What a worker sends back for one subject."""

    outcome: R | _Raised
    records: tuple[logging.LogRecord, ...]


def _in_worker(work: Callable[[str, L], R], unit: str, files: L) -> _Finished[R]:
    outcome: R | _Raised
    try:
        outcome = work(unit, files)
    except Exception as exc:
        outcome = _Raised.of(exc)
    return _Finished(outcome, _collector.take() if _collector is not None else ())


class _Workers(Generic[R]):
    """A pool of spawned processes, started at the first task and replaceable when broken."""

    def __init__(self, processes: int) -> None:
        self._processes = processes
        self._executor: ProcessPoolExecutor | None = None

    def submit(self, work: Callable[[str, L], R], unit: str, files: L) -> Future[_Finished[R]]:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                self._processes,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_start_worker,
                initargs=(get_logger().getEffectiveLevel(),),
            )
        return self._executor.submit(_in_worker, work, unit, files)

    def stop(self) -> None:
        """Drop the tasks that have not started and wait for those that run. The next task
        starts new processes."""
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)


def _unwrap(finished: _Finished[R]) -> R | Exception:
    """The outcome of a worker's task, after its log records went to the package logger."""
    logger = get_logger()
    for record in finished.records:
        if logger.isEnabledFor(record.levelno):
            logger.handle(record)
    outcome = finished.outcome
    return outcome.rebuild() if isinstance(outcome, _Raised) else outcome


def _in_pool(
    workers: _Workers[R],
    work: Callable[[str, L], R],
    leased: Sequence[tuple[int, str, L]],
    tick: Callable[[], None],
) -> dict[int, R | Exception]:
    """The outcome of ``work`` for each leased subject, keyed by its position in the batch."""
    outcomes: dict[int, R | Exception] = {}
    lost: set[int] = set()
    futures: dict[Future[_Finished[R]], int] = {}
    for position, unit, files in leased:
        try:
            futures[workers.submit(work, unit, files)] = position
        except BrokenProcessPool:
            lost.add(position)
    for future in as_completed(futures):
        position = futures[future]
        try:
            outcomes[position] = _unwrap(future.result())
        except BrokenProcessPool:
            lost.add(position)
            continue
        except Exception as exc:  # a result that cannot be pickled
            outcomes[position] = exc
        tick()
    if lost:
        # A process that dies takes the pool with it, and the pool does not say which task
        # it was on. Each task that was lost runs again alone in new processes, so only
        # the subject that kills its worker is charged with it.
        get_logger().warning(
            "a worker process died; running %d subjects of the batch again, one at a time",
            len(lost),
        )
        for position, unit, files in leased:
            if position not in lost:
                continue
            workers.stop()
            try:
                outcomes[position] = _unwrap(workers.submit(work, unit, files).result())
            except Exception as exc:
                outcomes[position] = exc
            tick()
    return outcomes


def _in_process(
    work: Callable[[str, L], R],
    leased: Sequence[tuple[int, str, L]],
    tick: Callable[[], None],
) -> dict[int, R | Exception]:
    outcomes: dict[int, R | Exception] = {}
    for position, unit, files in leased:
        try:
            outcomes[position] = work(unit, files)
        except Exception as exc:
            outcomes[position] = exc
        tick()
    return outcomes


def run_batches(
    units: Sequence[str],
    lease: Callable[[str], AbstractContextManager[L]],
    work: Callable[[str, L], R],
    commit: Callable[[Sequence[tuple[str, R | BaseException]]], None],
    *,
    batch_subjects: int,
    jobs: int,
    progress: str | None = None,
    stop: Callable[[], bool] | None = None,
) -> None:
    """Per batch of ``units`` (pending ``subject_uid``s, in order; for the pack phase
    ``session_uid``s): lease every subject, run ``work`` on each with what its lease yields
    (the local paths of its files, or a ``records.SessionWork``), call ``commit`` once in the
    parent, release the leases.

    ``commit`` receives one ``(unit, outcome)`` per subject of the batch, in the order of
    ``units``; an outcome is what ``work`` returned, or the exception that ``lease`` or
    ``work`` raised for that subject, and the batch goes on. At most ``batch_subjects``
    leases are open at any time.

    With ``jobs`` of 1, ``work`` runs in this process. Above that it runs in ``jobs``
    spawned processes, which live for the whole call; ``work`` must then be a picklable
    top-level function (TypeError before anything is leased otherwise), its results must be
    picklable, and a script that calls this needs the ``if __name__ == "__main__":`` guard
    of any program that spawns. An exception comes back from a worker as an instance of its
    own class with the same ``args`` and attributes and a :class:`RemoteTraceback` as its
    cause, or as a :class:`WorkerError` when it cannot be sent. What ``work`` logs to the
    package logger is logged in this process when its subject is finished. A subject whose
    worker dies gets a ``BrokenProcessPool``; subjects that were lost with it are run again,
    so ``work`` may run more than once for a subject.

    ``KeyboardInterrupt`` ends the call, and so does an exception from ``commit`` or from
    the release of a lease: work that has not started is dropped, work that runs is waited
    for, the leases are released, and the exception is raised again. The batch it interrupts
    is not committed.

    ``progress`` is the description of a progress bar on stderr that counts subjects; None
    shows none, and none is drawn when stderr is not a terminal.

    ``stop`` is asked before each batch is leased: once it answers true, no further batch is
    started and the call returns. The batches before it are committed, so this is how a
    caller in another thread ends the work between two batches without losing any.
    """
    if batch_subjects < 1:
        raise ValueError(f"batch_subjects must be at least 1, not {batch_subjects}")
    if jobs < 1:
        raise ValueError(f"jobs must be at least 1, not {jobs}")
    workers: _Workers[R] | None = None
    if jobs > 1:
        try:
            pickle.dumps(work)
        except Exception as exc:
            raise TypeError(
                f"with jobs > 1, work must be a picklable top-level function: {exc}"
            ) from exc
        workers = _Workers(min(jobs, batch_subjects, max(len(units), 1)))

    with ExitStack() as cleanup:
        tick = _progress_bar(progress, len(units), cleanup)
        if workers is not None:
            cleanup.callback(workers.stop)
        for start in range(0, len(units), batch_subjects):
            if stop is not None and stop():
                break
            batch = units[start : start + batch_subjects]
            with ExitStack() as leases:
                outcomes: dict[int, R | Exception] = {}
                leased: list[tuple[int, str, L]] = []
                for position, unit in enumerate(batch):
                    try:
                        files = _held(leases.enter_context(lease(unit)))
                    except Exception as exc:
                        outcomes[position] = exc
                        tick()
                    else:
                        leased.append((position, unit, files))
                if workers is None:
                    outcomes.update(_in_process(work, leased, tick))
                else:
                    try:
                        outcomes.update(_in_pool(workers, work, leased, tick))
                    except BaseException:
                        # before the leases go: a worker may still be reading their files
                        workers.stop()
                        raise
                commit([(unit, outcomes[position]) for position, unit in enumerate(batch)])


def _held(value: L) -> L:
    """What a lease yields, as it is handed to ``work``: a mapping of files as a plain
    ``dict`` (as before step 2 of the EPI interface; a mapping proxy would not pickle), any
    other value as it is."""
    if isinstance(value, Mapping):
        return dict(value)
    return value


def _progress_bar(description: str | None, total: int, cleanup: ExitStack) -> Callable[[], None]:
    """A function that counts one subject on a bar that ``cleanup`` closes."""
    if description is None:
        return lambda: None
    from tqdm import tqdm

    bar = cleanup.enter_context(
        tqdm(total=total, desc=description, unit="subject", file=sys.stderr, disable=None)
    )

    def tick() -> None:
        bar.update(1)

    return tick
