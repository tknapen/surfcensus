"""Run a per-subject function over leased files, batch by batch.

Stub of milestone 2: the name and signature are final, the body is written by work
package 2.D.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TypeVar

from surfcensus.sources.base import FileRole

R = TypeVar("R")

_PACKAGE = "work package 2.D (batches.py)"


def run_batches(
    units: Sequence[str],
    lease: Callable[[str], AbstractContextManager[Mapping[FileRole, Path]]],
    work: Callable[[str, Mapping[FileRole, Path]], R],
    commit: Callable[[Sequence[tuple[str, R | BaseException]]], None],
    *,
    batch_subjects: int,
    jobs: int,
    progress: str | None = None,
) -> None:
    """Per batch of ``units`` (pending ``subject_uid``s, in order): lease every subject, run
    ``work`` (a picklable top-level function; in a worker process when ``jobs`` > 1), call
    ``commit`` once in the parent, release the leases.

    An exception from ``lease`` or ``work`` for one subject is handed to ``commit`` for that
    subject and the batch goes on. ``progress`` is the description of a progress bar on
    stderr; None shows none.
    """
    raise NotImplementedError(_PACKAGE)
