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

import errno
import os
import re
import shutil
import time
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from surfcensus.func import ID_PATTERN, SESSION_JSON, STREAM_JSON
from surfcensus.log import get_logger
from surfcensus.records import (
    EPI_SESSION_PHASE,
    NOT_SELECTED,
    PACK_FAILURE,
    PACK_PHASE,
    SOURCE_CHANGED,
    PackRow,
    PhaseReport,
    RunContext,
    SessionResult,
    SessionWork,
    WorkRow,
    session_unit,
)

if TYPE_CHECKING:
    import pandas as pd

    from surfcensus.config import EpiConfig
    from surfcensus.func.model import PackInfo
    from surfcensus.store import PublicCatalog

# Nothing here imports numpy, pandas or nibabel at import time (the module is imported by the
# commands that only plan): the pack, the contours, the catalog's helpers and the
# configuration are imported where they are used.

_log = get_logger()

_SESSION_PREFIX = "session:"


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


# --- reading the public catalog -------------------------------------------------------------------


def _true(value: object) -> bool:
    """A flag of a public frame (0 or 1, a bool, pandas' boolean or a missing value)."""
    if value is None:
        return False
    try:
        if value != value:  # NaN, and pandas' NA raises in bool()
            return False
        return bool(value)
    except TypeError:
        return False


def _text(value: object) -> str | None:
    if value is None or not isinstance(value, str):
        return None
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [] if frame is None or len(frame) == 0 else frame.to_dict("records")


@dataclass(frozen=True)
class _Session:
    session_uid: str
    participant_uid: str
    stream_order: int
    subject_uid: str | None
    present: bool
    runs: tuple[str, ...]
    """The packable runs, present and selected, in acquisition order."""


@dataclass(frozen=True)
class _Catalog:
    """The functional rows of the public catalog that the pack phase reads, once per call."""

    participants: Mapping[str, dict[str, Any]]
    sessions: Mapping[str, _Session]
    packs: Mapping[str, dict[str, Any]]
    """``run_uid`` -> its row of ``public_packs``."""
    runs: Mapping[str, dict[str, Any]]
    """``run_uid`` -> its row of ``public_func_runs``."""


def _read(ctx: RunContext) -> _Catalog:
    public = ctx.public
    participants = {row["participant_uid"]: row for row in _records(public.func_participants())}
    runs = {row["run_uid"]: row for row in _records(public.func_runs())}
    by_session: dict[str, list[dict[str, Any]]] = {}
    for row in runs.values():
        if _true(row.get("present", 1)) and _true(row.get("packable")):
            if "selected" in row and not _true(row["selected"]):
                continue
            by_session.setdefault(row["session_uid"], []).append(row)
    sessions = {}
    for row in _records(public.func_sessions()):
        uid = row["session_uid"]
        members = sorted(
            by_session.get(uid, []),
            key=lambda r: (_order(r.get("acquisition_order")), r["run_uid"]),
        )
        participant = participants.get(row["participant_uid"], {})
        present = _true(row.get("present", 1)) and _true(participant.get("present", 1))
        sessions[uid] = _Session(
            session_uid=uid,
            participant_uid=row["participant_uid"],
            stream_order=_order(row.get("stream_order")),
            subject_uid=_text(row.get("subject_uid")),
            present=present,
            runs=tuple(r["run_uid"] for r in members) if present else (),
        )
    packs = {row["run_uid"]: row for row in _records(public.packs())}
    return _Catalog(participants, sessions, packs, runs)


def _order(value: object) -> int:
    try:
        if value is None or value != value:
            return 1 << 30
        return int(value)  # pyrefly: ignore[bad-argument-type]
    except (TypeError, ValueError):
        return 1 << 30


def _label(row: Mapping[str, Any], participant_uid: str) -> str:
    label = _text(row.get("participant"))
    return label if label else participant_uid


# --- the order of the sessions --------------------------------------------------------------------


def _ordered(ctx: RunContext, cat: _Catalog) -> list[str]:
    from surfcensus.store import queue_order, queue_scores

    packable = [s for s in cat.sessions.values() if s.present and s.runs]
    subjects = sorted({s.subject_uid for s in packable if s.subject_uid is not None})
    if ctx.public.mode() == "local" and subjects:
        app = ctx.config.app
        queue = queue_order(
            subjects, queue_scores(ctx.run_dir), blind=app.blind, seed=app.queue_seed
        )
    else:
        queue = subjects
    place = {uid: n for n, uid in enumerate(queue)}
    first: dict[str, tuple[int, int, str]] = {}
    for s in packable:
        if s.subject_uid is not None:
            key = (0, place[s.subject_uid], "")
        else:
            row = cat.participants.get(s.participant_uid, {})
            key = (1, 0, _label(row, s.participant_uid))
        if s.participant_uid not in first or key < first[s.participant_uid]:
            first[s.participant_uid] = key
    return [
        s.session_uid
        for s in sorted(
            packable,
            key=lambda s: (
                *first[s.participant_uid],
                s.participant_uid,
                s.stream_order,
                s.session_uid,
            ),
        )
    ]


def session_order(ctx: RunContext) -> list[str]:
    """The present sessions with a packable run, in the order in which the phase packs them:
    participants with a FreeSurfer subject in the order of the rating queue of their subject
    (``store.queue_order``), then participants without one by label (``participant_uid`` in a
    census-like run); within a participant its sessions in stream order. So behind the app the
    first packs are those of the first subjects a rater is given."""
    return _ordered(ctx, _read(ctx))


# --- the plan -------------------------------------------------------------------------------------


def _epi(ctx: RunContext) -> EpiConfig:
    from surfcensus.config import epi_settings

    return epi_settings(ctx.config)


def _session_dir(run_dir: Path, participant_uid: str, session_uid: str) -> Path:
    from surfcensus.store import epi_session_dir

    return epi_session_dir(run_dir, participant_uid, session_uid)


@dataclass(frozen=True)
class _Planned:
    plan: PackPlan
    current: frozenset[str]
    """Runs whose pack is current, in pending and in skipped sessions."""
    order: tuple[str, ...]
    """Every present session with a packable run, in order (also those not asked for)."""
    catalog: _Catalog


def _plan(
    ctx: RunContext,
    *,
    sessions: Collection[str] | None,
    force: bool,
    retry_failed: bool,
) -> _Planned:
    from surfcensus.config import ConfigError
    from surfcensus.func.packread import pack_on_disk, session_on_disk

    epi = _epi(ctx)
    pack_hash, session_hash = epi.pack_hash(), epi.session_hash()
    public = ctx.public
    if not epi.enabled:
        empty = PackPlan(pack_hash, session_hash, (), {}, {}, (), {}, ())
        return _Planned(empty, frozenset(), (), _Catalog({}, {}, {}, {}))
    cat = _read(ctx)
    order = _ordered(ctx, cat)
    wanted: set[str] | None = None
    if sessions is not None:
        wanted = {sessions} if isinstance(sessions, str) else set(sessions)
        unknown = sorted(uid for uid in wanted if uid not in cat.sessions)
        if unknown:
            raise ConfigError(
                "invalid", f"the run has no functional session {', '.join(unknown[:3])}"
            )
    asked = [uid for uid in order if wanted is None or uid in wanted]

    signatures: dict[str, str] = {}
    pending: list[str] = []
    skipped: list[str] = []
    keep: dict[str, frozenset[str]] = {}
    failed: dict[str, tuple[str, str | None]] = {}
    current: set[str] = set()
    statuses = {str(r["unit"]): r["status"] for r in _records(public.work(EPI_SESSION_PHASE))}
    for uid in asked:
        item = cat.sessions[uid]
        unit = session_unit(uid)
        session_dir = _session_dir(ctx.run_dir, item.participant_uid, uid)
        sig = public.epi_session_signature(uid)
        signatures[unit] = sig
        outcome = None if force else public.work_done(EPI_SESSION_PHASE, unit, session_hash, sig)
        # the last outcome of the session was a failure of the whole session (its lease or
        # its worker, recorded on every run it tried): the session's row decides, so that a
        # change anywhere in the session (its signature) tries every such run again
        whole = statuses.get(unit) == "failed"
        kept: set[str] = set()
        rows: dict[str, PackRow] = {}
        settled = True
        for run in item.runs:
            run_sig = public.pack_signature(run)
            signatures[run] = run_sig
            row = None if force else public.pack(run)
            same = (
                row is not None and row.config_hash == pack_hash and row.input_signature == run_sig
            )
            if row is not None and same:
                rows[run] = row
            if (
                same
                and row is not None
                and row.status == "failed"
                and not retry_failed
                and not whole
            ):
                failed[run] = (row.reason or PACK_FAILURE, row.detail)
            elif (
                same and row is not None and row.status == "done" and pack_on_disk(session_dir, run)
            ):
                kept.add(run)
            else:
                settled = False
        current |= kept
        # session.json and every contour file it lists, by name (review of step 2)
        done = outcome is True and session_on_disk(session_dir)
        # a packs row of a run that is no longer a packable run of this session: the session
        # is written again, so that its new session.json no longer lists the run before the
        # run's directory goes (pack_session removes it after that)
        leaving = any(
            row.get("session_uid") == uid and run not in item.runs for run, row in cat.packs.items()
        )
        if whole and outcome is False and not retry_failed:
            # the whole session failed on these files and is not tried again
            for run in item.runs:
                if run not in kept:
                    row = rows.get(run)
                    failed[run] = (
                        PACK_FAILURE if row is None else row.reason or PACK_FAILURE,
                        None if row is None else row.detail,
                    )
            skipped.append(uid)
        elif not whole and settled and not leaving and (done or not kept):
            # current; or every run failed on its own and is not tried again
            skipped.append(uid)
        else:
            pending.append(uid)
            keep[uid] = frozenset(kept)

    stale = _stale(ctx, cat, set(order), wanted)
    plan = PackPlan(
        pack_hash=pack_hash,
        session_hash=session_hash,
        pending=tuple(pending),
        keep=keep,
        signatures=signatures,
        skipped=tuple(skipped),
        failed=failed,
        stale=stale,
    )
    return _Planned(plan, frozenset(current), tuple(order), cat)


def _stale(
    ctx: RunContext, cat: _Catalog, live: set[str], wanted: set[str] | None
) -> tuple[str, ...]:
    """Runs with a ``packs`` row that are no longer packable runs of a live session, and
    sessions with an ``epi_session`` row that are no longer live: present, with a packable
    run. With ``wanted``, only those of the sessions asked for."""
    packable = {run for uid in live for run in cat.sessions[uid].runs}
    out: list[str] = []
    for run, row in sorted(cat.packs.items()):
        if run in packable:
            continue
        if wanted is not None and row.get("session_uid") not in wanted:
            continue
        out.append(run)
    work = ctx.public.work(EPI_SESSION_PHASE)
    for row in _records(work):
        unit = str(row["unit"])
        if not unit.startswith(_SESSION_PREFIX):
            continue
        uid = unit[len(_SESSION_PREFIX) :]
        if uid in live or (wanted is not None and uid not in wanted):
            continue
        out.append(unit)
    return tuple(out)


def pack_plan(
    ctx: RunContext,
    *,
    sessions: Collection[str] | None = None,
    force: bool = False,
    retry_failed: bool = False,
) -> PackPlan:
    """The plan of a call (module docstring, the skip rule). ``sessions`` restricts it to those
    ``session_uid``s. Reads the catalog and lists the run directory; opens no source file."""
    return _plan(ctx, sessions=sessions, force=force, retry_failed=retry_failed).plan


def pack_pending(
    ctx: RunContext,
    *,
    sessions: Collection[str] | None = None,
    force: bool = False,
    retry_failed: bool = False,
) -> list[str]:
    """The sessions a call would pack now, so that ``Run.pack`` builds its sources only when
    there is something to do (a finished run is checked with its data unmounted)."""
    return list(pack_plan(ctx, sessions=sessions, force=force, retry_failed=retry_failed).pending)


def _counts(planned: _Planned) -> tuple[int, int, int]:
    plan, cat = planned.plan, planned.catalog
    runs = [run for uid in planned.order for run in cat.sessions[uid].runs]
    done = sum(run in planned.current for run in runs)
    failed = sum(run in plan.failed for run in runs)
    return done, failed, len(runs) - done - failed


def pack_status(ctx: RunContext) -> tuple[int, int, int]:
    """Packable runs that are packed, failed and pending now, for ``surfcensus status``: what
    :func:`run_pack` would find."""
    return _counts(_plan(ctx, sessions=None, force=False, retry_failed=False))


def pack_failures(ctx: RunContext) -> dict[str, tuple[str, str | None]]:
    """``run_uid`` -> reason and detail of the runs counted as failed by :func:`pack_status`."""
    return dict(pack_plan(ctx).failed)


# --- the recorded state of a run (step 3) ----------------------------------------------------

RecordedStateName = Literal["packed", "pending", "failed", "cannot be packed", "not selected"]
"""``schemas.PACK_STATES``, as :class:`RecordedState` holds them."""


@dataclass(frozen=True)
class RecordedState:
    """What the catalog records of one present run under the present pack settings, as the
    subject pages, the index of the EPI pages and the export say it (decision D44)."""

    state: RecordedStateName
    reason: str | None
    """``func_runs.reason`` of a run that cannot be packed or is not selected, ``packs.reason``
    of a failed pack; None otherwise."""
    detail: str | None
    """The detail beside that reason (in a local run only, as the public views give it)."""


def recorded_states(public: PublicCatalog, pack_hash: str) -> dict[str, RecordedState]:
    """``run_uid`` -> :class:`RecordedState` of every present run, in the order of
    ``PublicCatalog.func_runs`` (session, then ``acquisition_order``, then ``run_uid``).

    A run that is not selected is ``not selected``, one that is not packable ``cannot be
    packed``, each with the reason and detail of ``func_runs``; of the rest, a ``packs`` row
    ``done`` under ``pack_hash`` (``EpiConfig.pack_hash()``) makes it ``packed``, a ``failed``
    row ``failed`` with the reason and detail of the row, and no such row ``pending``. This is
    what is recorded, decided from the catalog alone; :func:`pack_status` says what ``pack``
    would do now, which also asks the pack signature and the disk, so an outdated pack is
    ``packed`` here and pending there. Empty for a catalog without functional data."""
    recorded: dict[str, dict[str, Any]] = {}
    for row in _records(public.packs()):
        if row.get("config_hash") == pack_hash and row.get("status") in ("done", "failed"):
            recorded[str(row["run_uid"])] = row
    out: dict[str, RecordedState] = {}
    for row in _records(public.func_runs()):
        if not _true(row.get("present")):
            continue
        uid = str(row["run_uid"])
        reason, detail = _text(row.get("reason")), _text(row.get("detail"))
        if not _true(row.get("selected")):
            out[uid] = RecordedState("not selected", reason or NOT_SELECTED, detail)
        elif not _true(row.get("packable")):
            out[uid] = RecordedState("cannot be packed", reason, detail)
        elif uid in recorded and recorded[uid].get("status") == "done":
            out[uid] = RecordedState("packed", None, None)
        elif uid in recorded:
            pack = recorded[uid]
            out[uid] = RecordedState("failed", _text(pack.get("reason")), _text(pack.get("detail")))
        else:
            out[uid] = RecordedState("pending", None, None)
    return out


def _voxels(shape: object) -> int | None:
    """The number of voxels of a grid as the catalog writes its shape (``[49, 62, 36]``,
    ``49x62x36`` or a sequence); None when it says none."""
    if shape is None:
        return None
    if isinstance(shape, str):
        numbers = [int(x) for x in re.findall(r"\d+", shape)]
    else:
        try:
            numbers = [int(x) for x in shape]  # pyrefly: ignore[not-iterable]
        except (TypeError, ValueError):
            return None
    if len(numbers) != 3:
        return None
    return numbers[0] * numbers[1] * numbers[2]


def pack_estimate(ctx: RunContext) -> float:
    """Seconds that packing the pending runs takes in one process, an estimate: the rate of
    ``func.api.RATE_S_PER_VOLUME`` scaled by the volumes and voxels the catalog has for them."""
    from surfcensus.func.api import RATE_S_PER_VOLUME, RATE_VOXELS

    planned = _plan(ctx, sessions=None, force=False, retry_failed=False)
    plan, cat = planned.plan, planned.catalog
    work = 0
    for uid in plan.pending:
        for run in cat.sessions[uid].runs:
            if run in plan.keep.get(uid, frozenset()) or run in plan.failed:
                continue
            row = cat.runs.get(run, {})
            voxels = _voxels(row.get("grid_shape"))
            volumes = _order(row.get("n_volumes"))
            if voxels is not None and volumes < (1 << 30):
                work += volumes * voxels
    return RATE_S_PER_VOLUME * work / RATE_VOXELS


# --- the work of one session ----------------------------------------------------------------------


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
    ``epi_dir``, ``config`` and ``keep``).

    Contours that cannot be cut (a FreeSurfer file that cannot be read) leave the session
    without contours, with a warning, and the files of the contours out of the inputs of its
    packs, as ``surfcensus epi-session`` does; the runs are packed.
    A run whose inputs could not be read (``work.input_failures``) is listed as ``pack
    failure`` with its detail. An error of writing the display cache is raised as it is, for
    the parent to tell a full disk from a failure of the session."""
    from surfcensus.func import pack as pack_
    from surfcensus.func.inventory import cut_contours

    started = time.perf_counter()
    if work.session_uid != session_uid:
        raise ValueError(f"the work of session {work.session_uid} was handed in for {session_uid}")
    plan = work.plan
    contours = None
    inputs = work.inputs
    if work.contours is not None and plan.grids:
        contours, note = cut_contours(work.contours, plan)
        if contours is None:
            # named by its labels: the EPI phases run in a local run only (review of step 2)
            from surfcensus.func.inventory import session_text

            _log.warning("%s is packed %s", session_text(plan.participant, plan.session), note)
            # as ``epi-session`` does: a pack without contours does not depend on the files
            # of the contours, so they are not among its inputs (E2.Z)
            inputs = tuple(
                replace(
                    item,
                    session_inputs=tuple(
                        entry
                        for entry in item.session_inputs
                        if entry[1] != "freesurfer" and entry[0] != "xfm_anat_fsnative"
                    ),
                )
                for item in inputs
            )
    if work.input_failures:
        plan = replace(plan, details={**plan.details, **work.input_failures})
    packable = {r.run for r in plan.runs if r.reason is None}
    info = pack_.pack_session(
        Path(epi_dir) / work.participant_uid,
        session_uid,
        plan,
        inputs,
        contours,
        config,
        keep=keep.get(session_uid, frozenset()),
    )
    from surfcensus.func.packread import read_pack

    session_dir = Path(epi_dir) / work.participant_uid / session_uid
    kept = set(keep.get(session_uid, frozenset()))
    packs: dict[str, PackInfo] = {}
    failures: dict[str, tuple[str, str | None]] = {}
    for item in info.runs:
        if item.run not in packable:
            continue
        if item.reason is None:
            pack = read_pack(session_dir, item.run)
            if pack is None:  # cannot happen: pack_session wrote or kept it
                failures[item.run] = (PACK_FAILURE, "its pack.json does not read back")
            else:
                packs[item.run] = pack
        else:
            failures[item.run] = (item.reason, item.detail)
    return SessionResult(
        session_uid=session_uid,
        info=info,
        packs=packs,
        kept=tuple(sorted(run for run in packs if run in kept)),
        failures=failures,
        seconds=time.perf_counter() - started,
    )


# --- the stream of a participant ------------------------------------------------------------------


def _stream_text(
    ctx: RunContext, cat: _Catalog, participant_uid: str, order: Sequence[str]
) -> str | None:
    from surfcensus import __version__
    from surfcensus.func.model import StreamInfo, info_text

    members = [uid for uid in order if cat.sessions[uid].participant_uid == participant_uid]
    if not members:
        return None
    row = cat.participants.get(participant_uid, {})
    local = ctx.public.mode() == "local"
    stream = StreamInfo(
        surfcensus_version=__version__,
        label=_label(row, participant_uid) if local else participant_uid,
        sessions=tuple(members),
        order=_text(row.get("session_order")) or "session label",
    )
    return info_text(stream)


def _write_text(path: Path, text: str) -> bool:
    """Write ``text`` to ``path`` atomically unless the file holds it already; whether it
    was written."""
    from surfcensus._io import atomic_path

    try:
        if path.is_file() and not path.is_symlink() and path.read_text() == text:
            return False
    except (OSError, UnicodeDecodeError):
        pass
    with atomic_path(path) as tmp:
        tmp.write_text(text)
    return True


def _write_stream(
    ctx: RunContext, cat: _Catalog, participant_uid: str, order: Sequence[str]
) -> None:
    from surfcensus.store import epi_participant_dir

    text = _stream_text(ctx, cat, participant_uid, order)
    path = epi_participant_dir(ctx.run_dir, participant_uid) / STREAM_JSON
    if text is None:
        if path.is_file() or path.is_symlink():
            path.unlink()
        return
    _write_text(path, text)


def write_stream(ctx: RunContext, participant_uid: str) -> None:
    """Write ``epi/<participant_uid>/stream.json`` (``func.model.StreamInfo``) from the catalog:
    the participant's present sessions that have a packable run, in stream order, with the
    participant's label in a local run. Written after each batch that touched the participant,
    atomically. In a run directory it lists the sessions in order, packed or not yet: the
    server shows those whose ``session.json`` is there (``app.epi.Packs``), so the page grows
    as the packs are made behind the app.

    A file that holds the text already is not written again; a participant without such a
    session loses its ``stream.json``."""
    cat = _read(ctx)
    _write_stream(ctx, cat, participant_uid, _ordered(ctx, cat))


# --- what the machine says ------------------------------------------------------------------------

_NO_ROOM = (errno.ENOSPC, errno.EDQUOT, errno.EROFS)
_NOT_ALLOWED = (errno.EACCES, errno.EPERM)


def _is_unwritable(exc: BaseException, epi_root: Path) -> bool:
    """Whether a failure says that the display cache cannot be written, whatever the session:
    the disk is full, over quota or read-only, or a path under ``epi_root`` may not be
    written (the rule of ``render.batch._is_unwritable`` for ``media/``). The inputs are no
    cause of these: ``func.pack`` turns every error of reading them into a ``PackError``."""
    if not isinstance(exc, OSError):
        return False
    if exc.errno in _NO_ROOM:
        return True
    name = exc.filename
    if exc.errno in _NOT_ALLOWED and isinstance(name, str | bytes | os.PathLike):
        return Path(os.fsdecode(name)).is_relative_to(epi_root)
    return False


def _unwritable(err: OSError) -> Exception:
    from surfcensus.config import ConfigError

    error = ConfigError("invalid", f"the EPI packs cannot be written: {err}")
    error.__cause__ = err
    return error


def _failure(exc: BaseException) -> tuple[str, str]:
    """Reason and unscrubbed detail of the ``packs`` rows of a session whose lease or work
    raised."""
    from surfcensus.batches import WorkerError
    from surfcensus.sources.base import SourceError

    if isinstance(exc, SourceError) and exc.reason == "changed":
        return SOURCE_CHANGED, exc.detail
    if isinstance(exc, WorkerError):
        return PACK_FAILURE, str(exc)
    return PACK_FAILURE, f"{type(exc).__name__}: {exc}"


# --- removing what is stale -----------------------------------------------------------------------


def _remove_tree(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _remove_stale(
    ctx: RunContext, cat: _Catalog, stale: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Remove the pack directories of the stale runs and the session directories of the stale
    sessions, by the names the catalog gives, never by what a listing of ``epi/`` finds;
    returns the runs and the session units whose rows are to be forgotten (after this)."""
    from surfcensus.func import pack as pack_
    from surfcensus.store import EPI_DIR, epi_participant_dir

    runs, units = [], []
    epi_root = Path(ctx.run_dir) / EPI_DIR
    for name in stale:
        if name.startswith(_SESSION_PREFIX):
            uid = name[len(_SESSION_PREFIX) :]
            session = cat.sessions.get(uid)
            if session is not None:
                _remove_tree(_session_dir(ctx.run_dir, session.participant_uid, uid))
            units.append(name)
            continue
        row = cat.packs.get(name, {})
        session_uid = _text(row.get("session_uid"))
        path = _text(row.get("path"))
        where: Path | None = None
        if session_uid is not None and session_uid in cat.sessions:
            where = _session_dir(
                ctx.run_dir, cat.sessions[session_uid].participant_uid, session_uid
            )
        elif path is not None:
            parts = Path(path).parts
            if (
                len(parts) == 4
                and parts[0] == EPI_DIR
                and all(re.fullmatch(ID_PATTERN, p) for p in parts[1:])
                and parts[3] == name
            ):
                where = epi_root / parts[1] / parts[2]
        if where is not None and re.fullmatch(ID_PATTERN, name) and where.is_dir():
            pack_._remove_run_dir(where, name)
        runs.append(name)
    # participants that left, or have no session with a packable run any more
    live = {s.participant_uid for s in cat.sessions.values() if s.present and s.runs}
    for participant_uid in sorted(cat.participants):
        if participant_uid in live:
            continue
        folder = epi_participant_dir(ctx.run_dir, participant_uid)
        if folder.exists() or folder.is_symlink():
            _remove_tree(folder)
    return runs, units


# --- the phase ------------------------------------------------------------------------------------


@contextmanager
def _withheld(
    lease: Callable[[str], Any], uid: str, skip: Mapping[str, tuple[str, str | None]]
) -> Iterator[SessionWork]:
    """The lease of a session with the runs that are not tried again (failed rows that match)
    taken out of its inputs: they are listed as ``pack failure`` and keep their rows."""
    with lease(uid) as work:
        runs = {i.run for i in work.inputs} | set(work.input_failures)
        mine = {run: skip[run] for run in runs if run in skip}
        if mine:
            work = replace(
                work,
                inputs=tuple(i for i in work.inputs if i.run not in mine),
                input_failures={
                    **work.input_failures,
                    **{run: detail or reason for run, (reason, detail) in mine.items()},
                },
            )
        yield work


def _digests(info: PackInfo) -> dict[str, str]:
    """``<base>:<role>`` -> SHA-256 of every input of a pack, for ``pack_key``."""
    return {f"{entry.base}:{entry.role}": entry.sha256 for entry in info.inputs}


def _pack_bytes(info: PackInfo) -> int:
    files = [info.traces]
    for stream in info.streams.values():
        files += [stream.ref, stream.mean, stream.sd, *stream.chunks]
    return sum(entry.bytes for entry in files)


def _store_digests(
    ctx: RunContext, cat: _Catalog, session_uid: str, packs: Mapping[str, PackInfo]
) -> None:
    """The SHA-256 of every file a pack read into the catalog: the run's files, the anchor's
    transform under the anchor run, the session's transform under the session, the
    FreeSurfer files under the session's subject (decision D20: digests stay private)."""
    from surfcensus.sources.base import FUNC_SESSION_ROLES, FileRole, FuncRole

    func_roles = {role.value: role for role in FuncRole}
    session_files: dict[FuncRole, str] = {}
    subject_files: dict[FileRole, str] = {}
    anchors: dict[str, dict[FuncRole, str]] = {}
    for run, info in sorted(packs.items()):
        own: dict[FuncRole, str] = {}
        for entry in info.inputs:
            if entry.base == "freesurfer":
                subject_files[FileRole(entry.role)] = entry.sha256
            elif entry.role == "anchor_xfm_to_anat":
                anchors.setdefault(info.anchor_run, {})[FuncRole.XFM_TO_ANAT] = entry.sha256
            elif entry.role in func_roles:
                role = func_roles[entry.role]
                if role in FUNC_SESSION_ROLES:
                    session_files[role] = entry.sha256
                else:
                    own[role] = entry.sha256
        if own:
            ctx.public.set_func_digests("run", run, own)
    for run, digests in sorted(anchors.items()):
        ctx.public.set_func_digests("run", run, digests)
    if session_files:
        ctx.public.set_func_digests("session", session_uid, session_files)
    subject = cat.sessions[session_uid].subject_uid if session_uid in cat.sessions else None
    if subject_files and subject is not None:
        ctx.public.set_file_digests(subject, subject_files)


def run_pack(
    ctx: RunContext,
    *,
    jobs: int = 1,
    sessions: Collection[str] | None = None,
    force: bool = False,
    retry_failed: bool = False,
    stop: Callable[[], bool] | None = None,
    waiting: Callable[[], None] | None = None,
) -> PhaseReport:
    """Pack the pending sessions (:func:`pack_plan`) in batches of ``jobs`` sessions, each in a
    worker process when ``jobs`` is above 1 (``batches.run_batches`` with ``ctx.func_lease``
    as the lease), record every run and session, write each touched participant's
    ``stream.json`` (:func:`write_stream`) and remove what is stale. ``stop`` is asked before
    each batch, as for ``Run.render``. The report is of phase ``pack``; its units are runs:
    done, skipped (kept and current), failed (``pack failure``, ``source changed``). Runs that
    cannot be packed are no units: their reason is the inventory's. Raises ``ConfigError`` for
    what lies in the machine (module docstring) and for a context without ``func_lease``.

    The order of the writes: per batch the packs and ``session.json`` (in the worker), then
    the digests of their inputs, the ``packs`` rows, the ``epi_session`` rows, then the
    ``stream.json`` of the participants the batch touched; at the end the directories of what
    is stale, then their rows. Nothing of a batch in which the display cache could not be
    written is recorded: the packs of it that were made are whole on disk and are made again
    by the next call (their rows are what decides, not the files)."""
    from surfcensus.batches import run_batches
    from surfcensus.config import ConfigError
    from surfcensus.store import EPI_DIR

    started = time.perf_counter()
    if jobs < 1:
        raise ValueError(f"jobs must be at least 1, not {jobs}")
    planned = _plan(ctx, sessions=sessions, force=force, retry_failed=retry_failed)
    plan, cat = planned.plan, planned.catalog
    public = ctx.public
    epi = _epi(ctx)
    epi_root = Path(ctx.run_dir) / EPI_DIR
    if plan.pending and (ctx.func_lease is None or ctx.pack_key is None):
        raise ConfigError(
            "invalid", "the run has functional sessions to pack and no lease to read them with"
        )
    func_lease = ctx.func_lease
    pack_key = ctx.pack_key

    failures: dict[str, str] = {run: reason for run, (reason, _) in plan.failed.items()}
    done = 0
    skipped = sum(
        len(cat.sessions[uid].runs) - sum(r in plan.failed for r in cat.sessions[uid].runs)
        for uid in plan.skipped
    )
    touched: set[str] = set()
    processed: set[str] = set()
    # sessions whose pack_session ran to its end in this call

    def failed_row(session_uid: str, run: str, reason: str, detail: str | None) -> PackRow:
        return PackRow(
            run_uid=run,
            session_uid=session_uid,
            config_hash=plan.pack_hash,
            input_signature=plan.signatures[run],
            status="failed",
            reason=reason,
            detail=None if detail is None else ctx.scrub(detail),
        )

    def commit(outcomes: Sequence[tuple[str, SessionResult | BaseException]]) -> None:
        nonlocal done, skipped
        unwritable = [
            o for _, o in outcomes if isinstance(o, OSError) and _is_unwritable(o, epi_root)
        ]
        if unwritable:
            # A full or read-only disk fails every session alike. Nothing of the batch is
            # recorded, so the next call does the work once there is room.
            raise _unwritable(unwritable[0])
        rows: list[PackRow] = []
        work_rows: list[WorkRow] = []
        for uid, outcome in outcomes:
            session = cat.sessions[uid]
            touched.add(session.participant_uid)
            tried = [run for run in session.runs if run not in plan.failed]
            if isinstance(outcome, BaseException):
                # the session's runs are recorded as failed, but for those whose packs are
                # current: their rows and their packs stay as they are; and the session's row
                # says that the whole session failed (pack_plan: not tried again until its
                # signature changes, or with retry_failed)
                reason, detail = _failure(outcome)
                tried = [run for run in tried if run not in plan.keep.get(uid, frozenset())]
                rows += [failed_row(uid, run, reason, detail) for run in tried]
                failures.update(dict.fromkeys(tried, reason))
                work_rows.append(
                    WorkRow(
                        phase=EPI_SESSION_PHASE,
                        unit=session_unit(uid),
                        config_hash=plan.session_hash,
                        input_signature=plan.signatures[session_unit(uid)],
                        status="failed",
                        reason=reason,
                        detail=ctx.scrub(detail),
                    )
                )
                continue
            processed.add(uid)
            _store_digests(ctx, cat, uid, outcome.packs)
            kept = set(outcome.kept)
            for run in tried:
                if run in kept:
                    skipped += 1
                    continue
                info = outcome.packs.get(run)
                if info is None:
                    reason, detail = outcome.failures.get(run, (PACK_FAILURE, None))
                    rows.append(failed_row(uid, run, reason, detail))
                    failures[run] = reason
                    continue
                assert pack_key is not None
                rows.append(
                    PackRow(
                        run_uid=run,
                        session_uid=uid,
                        config_hash=plan.pack_hash,
                        input_signature=plan.signatures[run],
                        status="done",
                        pack_key=pack_key(plan.pack_hash, _digests(info)),
                        path=f"{EPI_DIR}/{session.participant_uid}/{uid}/{run}",
                        n_volumes=info.n_volumes,
                        bytes=_pack_bytes(info),
                        content_digest=info.content_digest,
                        anchor_run_uid=info.anchor_run,
                        grid_key=info.grid_key,
                    )
                )
                failures.pop(run, None)
                done += 1
            work_rows.append(
                WorkRow(
                    phase=EPI_SESSION_PHASE,
                    unit=session_unit(uid),
                    config_hash=plan.session_hash,
                    input_signature=plan.signatures[session_unit(uid)],
                    status="done",
                    output=f"{EPI_DIR}/{session.participant_uid}/{uid}/{SESSION_JSON}",
                )
            )
        if rows:
            public.record_packs(rows)
        if work_rows:
            public.record_work(work_rows)  # last: a session row stands for its packs
        for participant_uid in sorted(touched):
            try:
                _write_stream(ctx, cat, participant_uid, planned.order)
            except OSError as err:
                if _is_unwritable(err, epi_root):
                    raise _unwritable(err) from err
                raise

    if plan.pending:
        assert func_lease is not None
        work = partial(pack_session_unit, epi_dir=epi_root, config=epi, keep=plan.keep)
        run_batches(
            list(plan.pending),
            lambda uid: _withheld(func_lease, uid, plan.failed),
            work,
            commit,
            batch_subjects=jobs,
            jobs=jobs,
            progress=PACK_PHASE,
            stop=stop,
            waiting=waiting,
        )

    # every participant's stream.json, also of those no batch touched (contents that are
    # there already are not written again), then what is stale: the directories first, the
    # rows after. A stale run of a live session goes with the session's new session.json
    # (``pack_session`` removes its directory after writing it), so it waits for the call that
    # packs its session.
    live = set(planned.order)
    stale = [
        name
        for name in plan.stale
        if name.startswith(_SESSION_PREFIX)
        or cat.packs.get(name, {}).get("session_uid") not in live
        or cat.packs[name]["session_uid"] in processed
    ]
    try:
        for participant_uid in sorted({cat.sessions[uid].participant_uid for uid in live}):
            _write_stream(ctx, cat, participant_uid, planned.order)
        runs, units = _remove_stale(ctx, cat, stale)
    except OSError as err:
        if _is_unwritable(err, epi_root):
            raise _unwritable(err) from err
        raise
    if runs:
        public.forget_packs(runs)
    if units:
        public.forget_work(EPI_SESSION_PHASE, units)
    return PhaseReport(
        phase=PACK_PHASE,
        done=done,
        skipped=skipped,
        failed=len(failures),
        failures=tuple(sorted(failures.items())),
        seconds=time.perf_counter() - started,
    )
