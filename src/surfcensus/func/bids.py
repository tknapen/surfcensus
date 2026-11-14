"""BIDS names of functional runs: entities, run keys, name order, sidecar times, and the
listing of the raw series of a participant. Work package E1.A.

A run is keyed by ``sub``, ``ses``, ``task``, ``acq``, ``ce``, ``rec``, ``dir``, ``run`` and
``echo`` (docs/epi_interface.md, The unit), the run index compared as a number, so that
``run-01`` and ``run-1`` are one run (stage 1 found that necessary in 3,102 of 9,462 runs).
A dataset without ``ses-`` folders has one session per participant. Everything here works from
names and the listing; only :func:`sidecar_time` opens a file.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from surfcensus.records import DerivRun, RawRun, RunEntities

ENTITIES: tuple[str, ...] = ("sub", "ses", "task", "acq", "ce", "rec", "dir", "run", "echo")
"""The entities of :class:`~surfcensus.records.RunEntities`, in BIDS order."""
ORDER_ENTITIES: tuple[str, ...] = ("task", "acq", "ce", "rec", "dir")
"""Compared by name in the name-order rule, before the run index and the echo index."""
RAW_SUFFIXES: tuple[str, ...] = ("_bold.nii.gz", "_bold.nii")
SIDECAR_SUFFIX = "_bold.json"


def parse_entities(name: str) -> RunEntities:
    """The entities of a BIDS file name or stem (a path is allowed; its last part is read).
    Raises ``ValueError`` for a name without ``sub-``. Unknown entities are ignored; of an
    entity named twice the first is kept."""
    raise NotImplementedError("work package E1.A: func/bids.py")


def run_index(label: str | None) -> int | None:
    """A run or echo index as a number; None for a missing or unreadable one."""
    raise NotImplementedError("work package E1.A: func/bids.py")


def run_key(entities: RunEntities) -> str:
    """A run within its participant, the same for the raw series and its derivative files:
    ``ses``, ``task``, ``acq``, ``ce``, ``rec``, ``dir`` as written and the run index as a
    number (``ses-1_task-rest_run-1`` for ``run-01``), no subject and no echo, so that the
    echoes of a multi-echo raw run share the key of the run the derivative has."""
    raise NotImplementedError("work package E1.A: func/bids.py")


def name_order(entities: RunEntities) -> tuple[str | int, ...]:
    """Sort key of the name-order rule: task by name, then ``acq``, ``ce``, ``rec`` and
    ``dir`` by name (absent first), then the run index as a number (absent first), then the
    echo index."""
    raise NotImplementedError("work package E1.A: func/bids.py")


def run_name(entities: RunEntities) -> str:
    """The BIDS name of a run without the subject, as the pages show it:
    ``ses-1_task-rest_run-1`` (labels as written)."""
    raise NotImplementedError("work package E1.A: func/bids.py")


def acquisition_seconds(value: object) -> tuple[str, float] | None:
    """``AcquisitionTime`` of a sidecar as ``(kind, seconds)``: kind ``time`` for a time of day
    ``H:M:S[.f]`` (sidecars write it without leading zeros, as in ``17:30:2.685000``), kind
    ``datetime`` for an ISO date and time; None when it cannot be read."""
    raise NotImplementedError("work package E1.A: func/bids.py")


def sidecar_time(path: Path) -> str | None:
    """``AcquisitionTime`` of a JSON sidecar as written, None when the file cannot be read or
    the field is absent or not a string."""
    raise NotImplementedError("work package E1.A: func/bids.py")


def is_raw_dataset(folder: Path) -> bool:
    """Whether ``folder`` holds ``sub-*/[ses-*/]func/*_bold.nii[.gz]``."""
    raise NotImplementedError("work package E1.A: func/bids.py")


def list_raw_runs(raw_root: Path, participant: str) -> list[RawRun]:
    """The raw series of ``participant`` (``sub-01``) in a raw BIDS dataset, from the listing,
    in :func:`name_order`, each with its sidecar beside it or inherited from a higher level
    (BIDS inheritance) when the listing has one. Locations are relative to ``raw_root``."""
    raise NotImplementedError("work package E1.A: func/bids.py")


def session_labels(runs: Sequence[RawRun | DerivRun]) -> list[str | None]:
    """The sessions the runs belong to, sorted, None (no ``ses-``) first."""
    raise NotImplementedError("work package E1.A: func/bids.py")
