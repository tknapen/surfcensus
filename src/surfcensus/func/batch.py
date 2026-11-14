"""The pack phase of a run: the sessions whose packs are pending, packed a batch at a time into
the display cache ``epi/<participant_uid>/<session_uid>/`` and recorded in ``packs`` and the
``work`` ledger (docs/epi_interface.md, "Step 2: the plan"; work package E2.C). The
counterpart of ``render/batch.py``.

Downstream of the inventory (decision D20): it holds a ``RunContext`` and its
``PublicCatalog``, never the ``Run``; it names subjects by ``subject_uid``, participants,
sessions and runs by their keyed identifiers; files reach it as local paths inside the lease
of a session (``RunContext.func_lease``), and input signatures only as values to pass back.

The units. The page's unit is a participant's stream (decision D38); the unit of work is a
session, which is the unit of the one registration (decision D36) and of ``session.json``
(``func.pack.pack_session``); the runs of a session have rows of their own in ``packs``.

The skip rule (decision D21), from the catalog and the run directory alone, opening no source
file. A session is skipped when its ``epi_session`` row in ``work`` is ``done`` for
``EpiConfig.session_hash()`` and ``PublicCatalog.epi_session_signature``, its ``session.json``
is there, and every packable run has a ``packs`` row ``done`` for ``EpiConfig.pack_hash()``
and ``PublicCatalog.pack_signature`` whose pack is whole on disk
(``func.packread.pack_on_disk``, by directory listings). Otherwise the session is pending; its
runs whose rows are current are kept (``func.pack.pack_session(..., keep=...)``: taken from
disk without hashing their inputs) and the others are packed. A ``failed`` row that matches is
not tried again unless ``retry_failed``; ``force`` ignores every row. A new anchor changes the
signature of every run of its session, so the whole session is packed again. Outputs are
written first and rows last; a run that is no longer packable or selected, and a session or a
participant that left, lose their pack directories first and their rows after.

What lies in the machine (a full or read-only disk, a run directory that cannot be written)
stops the phase with ``ConfigError`` and exit code 2, recorded against no run (decision D29);
anything else of one run is a ``pack failure`` of that run.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from surfcensus.records import PhaseReport, RunContext, SessionResult, SessionWork

if TYPE_CHECKING:
    from surfcensus.config import EpiConfig


@dataclass(frozen=True)
class PackPlan:
    """What a call of :func:`run_pack` would do now, decided from the catalog and the run
    directory (:func:`pack_plan`)."""

    pack_hash: str
    """``EpiConfig.pack_hash()``."""
    session_hash: str
    """``EpiConfig.session_hash()``."""
    pending: tuple[str, ...]
    """``session_uid``s to pack, in the order of :func:`session_order`."""
    keep: Mapping[str, frozenset[str]]
    """``session_uid`` -> the ``run_uid``s of a pending session whose packs are current."""
    signatures: Mapping[str, str] = field(repr=False)
    """``run_uid`` -> its pack signature, and ``session:<session_uid>`` -> the session's.
    Values to pass back to the catalog and nothing else."""
    skipped: tuple[str, ...]
    """Sessions that are current."""
    failed: Mapping[str, tuple[str, str | None]]
    """``run_uid`` -> reason and detail of a matching ``failed`` row that is not tried again."""
    stale: tuple[str, ...]
    """``run_uid``s whose packs are to be removed: runs that left, or that are no longer
    packable or selected; and ``session:<session_uid>`` of sessions that left."""


def session_order(ctx: RunContext) -> list[str]:
    """The present sessions with a packable run, in the order in which the phase packs them:
    participants with a FreeSurfer subject in the order of the rating queue of their subject
    (``store.queue_order``), then participants without one by label (``participant_uid`` in a
    census-like run); within a participant its sessions in stream order. So behind the app the
    first packs are those of the first subjects a rater is given."""
    raise NotImplementedError("work package E2.C: func/batch.py")


def pack_plan(
    ctx: RunContext,
    *,
    sessions: Collection[str] | None = None,
    force: bool = False,
    retry_failed: bool = False,
) -> PackPlan:
    """The plan of a call (module docstring, the skip rule). ``sessions`` restricts it to those
    ``session_uid``s. Reads the catalog and lists the run directory; opens no source file."""
    raise NotImplementedError("work package E2.C: func/batch.py")


def pack_pending(
    ctx: RunContext,
    *,
    sessions: Collection[str] | None = None,
    force: bool = False,
    retry_failed: bool = False,
) -> list[str]:
    """The sessions a call would pack now, so that ``Run.pack`` builds its sources only when
    there is something to do (a finished run is checked with its data unmounted)."""
    raise NotImplementedError("work package E2.C: func/batch.py")


def pack_status(ctx: RunContext) -> tuple[int, int, int]:
    """Packable runs that are packed, failed and pending now, for ``surfcensus status``: what
    :func:`run_pack` would find."""
    raise NotImplementedError("work package E2.C: func/batch.py")


def pack_failures(ctx: RunContext) -> dict[str, tuple[str, str | None]]:
    """``run_uid`` -> reason and detail of the runs counted as failed by :func:`pack_status`."""
    raise NotImplementedError("work package E2.C: func/batch.py")


def pack_estimate(ctx: RunContext) -> float:
    """Seconds that packing the pending runs takes in one process, an estimate: the rate of
    ``func.api.RATE_S_PER_VOLUME`` scaled by the volumes and voxels the catalog has for them."""
    raise NotImplementedError("work package E2.C: func/batch.py")


def run_pack(
    ctx: RunContext,
    *,
    jobs: int = 1,
    sessions: Collection[str] | None = None,
    force: bool = False,
    retry_failed: bool = False,
    stop: Callable[[], bool] | None = None,
) -> PhaseReport:
    """Pack the pending sessions (:func:`pack_plan`) in batches of ``jobs`` sessions, each in a
    worker process when ``jobs`` is above 1 (``batches.run_batches`` with ``ctx.func_lease``
    as the lease), record every run and session, write each touched participant's
    ``stream.json`` (:func:`write_stream`) and remove what is stale. ``stop`` is asked before
    each batch, as for ``Run.render``. The report is of phase ``pack``; its units are runs:
    done, skipped (kept and current), failed (``pack failure``, ``source changed``). Runs that
    cannot be packed are no units: their reason is the inventory's. Raises ``ConfigError`` for
    what lies in the machine (module docstring) and for a context without ``func_lease``."""
    raise NotImplementedError("work package E2.C: func/batch.py")


def pack_session_unit(
    session_uid: str,
    work: SessionWork,
    *,
    epi_dir: Path,
    config: EpiConfig,
    keep: Mapping[str, frozenset[str]],
) -> SessionResult:
    """The work of one session in a worker: the contours of each grid from ``work.contours``
    (``func.contours``), then ``func.pack.pack_session`` into
    ``<epi_dir>/<participant_uid>/<session_uid>/`` with ``keep[session_uid]`` kept. A top-level
    function, so that it can be sent to a worker process (``functools.partial`` with
    ``epi_dir``, ``config`` and ``keep``)."""
    raise NotImplementedError("work package E2.C: func/batch.py")


def write_stream(ctx: RunContext, participant_uid: str) -> None:
    """Write ``epi/<participant_uid>/stream.json`` (``func.model.StreamInfo``) from the catalog:
    the participant's present sessions that have a packable run, in stream order, with the
    participant's label in a local run. Written after each batch that touched the participant,
    atomically. In a run directory it lists the sessions in order, packed or not yet: the
    server shows those whose ``session.json`` is there (``app.epi.Packs``), so the page grows
    as the packs are made behind the app."""
    raise NotImplementedError("work package E2.C: func/batch.py")
