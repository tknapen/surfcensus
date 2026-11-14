"""``epi_subject`` and ``epi_session``: the functional sessions of one subject of a local folder
packed into a directory, without a run directory or a catalog; the functions behind
``surfcensus epi-session`` (docs/epi_interface.md, Step 1), the counterpart of
``render.three_plane``. Work package E1.Z.

Both uses of the interface are served by what is packed: quality control of the raw data (the
raw BIDS series, placed once per session by the anchor run's stored transform and nothing
else, decision D36) and quality control of the preprocessing (the series as the pipeline
wrote it, and what the pipeline says it did to it).

What is written, and only there (decision D38: "the display cache is perfect, just no new
nifti files")::

    <out>/stream.json            the sessions of the subject in the order of its one stream
    <out>/<session_id>/...       one pack directory per session (``func.pack``)

No NIfTI file and no other image volume is written anywhere, nothing is resampled to a grid
common to runs or sessions (runs on different grids are packed on their own grids), nothing is
fetched, and nothing is written into the folders that are read: the raw dataset, the derivative
and the FreeSurfer subjects directory are sources, and an ``out`` in any of them, or in what a
link at their top level leads to, is refused before anything is written.

Ids. A session and a run are named by ``func.func_id`` under ``key``. With ``key=None`` a
random key is drawn, except that a second call into an ``out`` whose ``stream.json`` is that of
the same participant takes the ids recorded there for the sessions (by label) and runs (by
name) that are still there, so that the page's addresses stay. No key is written: ids of
step 1 say nothing outside their ``out`` (the session and run ids of a run directory are keyed
in step 2 by the run's own key, ``privacy.py``).
"""

from __future__ import annotations

import fnmatch
import re
import secrets
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from surfcensus import __version__
from surfcensus._io import atomic_path
from surfcensus.config import ConfigError, EpiConfig
from surfcensus.func import ID_PATTERN, SESSION_JSON, STREAM_JSON, bids, func_id
from surfcensus.func import connectors as connectors_
from surfcensus.func import pack as pack_
from surfcensus.func.connectors.base import ConnectorError
from surfcensus.func.discover import (
    FuncFolders,
    _absolute,
    _shown,
    find_folders,
    refuse_out,
)
from surfcensus.func.inventory import (
    SESSION_ORDER,
    contour_job,
    cut_contours,
    link_targets,
    list_participant,
    pack_inputs,
    plan_one,
    probe_participant,
    run_name,
    session_key,
    session_text,
    withheld_of,
)
from surfcensus.func.model import SessionInfo, StreamInfo, info_text, read_info
from surfcensus.records import (
    NOT_SELECTED,
    PlaneContours,
    RunPackInput,
    SessionPlan,
)
from surfcensus.sources.base import SourceError

__all__ = [
    "RATE_S_PER_VOLUME",
    "RATE_VOXELS",
    "SESSION_ORDER",
    "FuncFolders",
    "epi_session",
    "epi_subject",
    "find_folders",
    "refuse_in_sources_of_links",
    "refuse_out",
    "session_key",
]

RATE_S_PER_VOLUME = 0.114
"""Seconds to pack one volume of both streams on :data:`RATE_VOXELS` voxels, one process:
measured on the worked example (``ds000102`` sub-01 run 1, 146 volumes, portable voxel map;
16.6 s on one shared 4-CPU Linux machine, 2026-11-14). The estimate scales it by voxels."""
RATE_VOXELS = 109_368
"""The voxels of the grid :data:`RATE_S_PER_VOLUME` was measured on (49 x 62 x 36, 3 mm)."""

Progress = Callable[[str], None]


def _participant(subject: str) -> str:
    participant = subject if subject.startswith("sub-") else f"sub-{subject}"
    if not bids.valid_participant(participant):
        raise ConfigError(
            "invalid", f"{subject!r} is no BIDS participant (sub-<letters and digits>)"
        )
    return participant


@dataclass(frozen=True)
class _Previous:
    sessions: Mapping[str, str]
    """Session label -> id."""
    runs: Mapping[tuple[str, str], str]
    """(session label, run name) -> id."""
    listed: tuple[str, ...]


def _previous(out: Path, participant: str) -> _Previous:
    """The ids of an earlier call into ``out``; refuses an ``out`` that holds the stream of
    another participant."""
    stream = read_info(out / STREAM_JSON, StreamInfo)
    if stream is None:
        return _Previous({}, {}, ())
    if stream.label != participant:
        raise ConfigError(
            "invalid",
            f"{out} holds the packs of {stream.label}; choose another output directory for "
            f"{participant}",
        )
    sessions: dict[str, str] = {}
    runs: dict[tuple[str, str], str] = {}
    for sid in stream.sessions:
        info = read_info(out / sid / SESSION_JSON, SessionInfo)
        if info is None or not re.fullmatch(ID_PATTERN, sid):
            continue
        sessions[info.label] = sid
        for run in info.runs:
            runs[(info.label, run.name)] = run.run
    return _Previous(sessions, runs, tuple(stream.sessions))


@dataclass
class _Session:
    label: str | None
    sid: str
    plan: SessionPlan
    inputs: list[RunPackInput]
    contours: dict[str, Sequence[PlaneContours]] | None
    contour_note: str
    work: dict[str, int]
    """Run id -> volumes times voxels, for the estimate."""


# --- links among the inputs, and what the registration says ----------------------------------


def refuse_in_sources_of_links(
    out: Path, planned: Sequence[_Session], folders: FuncFolders
) -> None:
    """Raise ``ConfigError("invalid", ...)`` when ``out`` is, or lies inside, a directory that
    holds the target of a linked input file (``func.inventory.link_targets``), as the
    renderer refuses an output beside the target of a linked input. Per-file links are the
    layout of DataLad
    and git-annex clones, whose object store holds the targets."""
    from surfcensus.sources.local import refuse_in_sources

    also = link_targets([item.plan for item in planned], folders)
    refuse_in_sources(out, [], what="output", also=also)


def _registration_lines(planned: Sequence[_Session], participant: str) -> list[str]:
    """Lines that say, per session, what the one registration of the session finds in the raw
    data: an anchor that is not packed or not the first run by the rule, each run beyond the
    frame limit, and an anchor that is the outlier."""
    lines = []
    for item in planned:
        plan = item.plan
        if plan.anchor is None:
            continue
        text = session_text(participant, item.label)
        names = {r.run: r.name for r in plan.runs}
        anchor = names.get(plan.anchor.run, plan.anchor.run)
        reasons = {r.run: r.reason for r in plan.runs}
        if plan.anchor_note:
            lines.append(f"{text}: {plan.anchor_note}.")
        if reasons.get(plan.anchor.run) is not None and any(v is None for v in reasons.values()):
            lines.append(
                f"{text}: the anchor {anchor} is not packed ({reasons[plan.anchor.run]}); the "
                "raw stream of every packed run is drawn through its stored registration."
            )
        if plan.frame is None:
            continue
        for run in plan.frame.flagged:
            d = plan.frame.runs[run]
            lines.append(
                f"{text}: flagged (raw data): {names.get(run, run)}, drawn through the "
                f"registration of the anchor {anchor}, lies up to {d.max_mm:.1f} mm (mean "
                f"{d.mean_mm:.1f} mm) from where its own stored registration puts it, beyond "
                f"the limit of {plan.frame.limit_mm:g} mm."
            )
        if plan.frame.anchor_is_outlier:
            lines.append(
                f"{text}: the anchor {anchor} is the outlier (raw data): every other run is "
                "flagged against it and they agree with each other, so the anchor, or its "
                "stored registration, is the odd one."
            )
    return lines


# --- packing --------------------------------------------------------------------------------


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _remove_stale(out: Path, previous: Sequence[str], kept: Sequence[str]) -> None:
    for sid in previous:
        if sid in kept or not re.fullmatch(ID_PATTERN, sid):
            continue
        shutil.rmtree(out / sid, ignore_errors=True)


def epi_subject(
    path: Path,
    subject: str,
    out: Path,
    sessions: Sequence[str] | None = None,
    raw: Path | None = None,
    config: EpiConfig | None = None,
    key: bytes | None = None,
    runs: Sequence[str] | None = None,
    progress: Progress | None = None,
    force: bool = False,
) -> StreamInfo:
    """Pack every session of ``subject`` (``sub-01`` or ``01``) found under ``path``, or the
    sessions named (``1`` or ``ses-1``), into ``<out>/<session_id>/``, and write
    ``<out>/stream.json`` last: the sessions in the order of the subject's one continuous
    stream (decision D38), by :func:`session_key` (no session first, then the labels with
    numbers compared as numbers). Within a session the runs follow in acquisition order.

    ``runs`` restricts the runs packed to those whose derivative stem matches one of the
    shell patterns (``*task-rest*``); the other runs of the sessions that hold a selected run
    are listed as ``not selected`` and are not packed, but they stay runs of their session: one
    of them can be its anchor, and the session's acquisition order is the same whatever is
    selected (``session.plan_session``). A run whose pack in ``out`` is current is kept
    (``pack.current_pack``); ``force`` packs every run again. ``key`` keys the ids
    (module docstring). ``progress`` is given one line at a time: the listing, an estimate
    of the time, each run with its volumes and seconds, each session.

    A participant without a packable run gives a ``stream.json`` whose sessions list their
    runs with reasons. Raises ``SourceError`` as :func:`find_folders` does and ``not found``
    for a participant without functional runs, ``ConfigError("invalid")`` for an ``out`` in a
    folder that is read, an ``out`` that holds the packs of another participant, a session
    that is not there, or a participant that is no BIDS label. An error of writing ``out``
    (a full disk) is raised as it is.
    """
    say = progress if progress is not None else (lambda line: None)
    config = config if config is not None else EpiConfig()
    participant = _participant(subject)
    given = _absolute(path)
    folders = find_folders(given, raw)
    out = _absolute(out)
    refuse_out(out, folders)
    connector = connectors_.connector(folders.match.connector)
    deriv, raw_root = folders.deriv_root, folders.raw_root
    listing = list_participant(folders, connector, participant)
    druns_all = listing.deriv_runs
    if not druns_all:
        raise SourceError(
            "not found", f"{participant} has no functional run in {_shown(deriv, given)}"
        )
    patterns_ = tuple(runs or ())
    selected = [r for r in druns_all if any(fnmatch.fnmatchcase(r.stem, p) for p in patterns_)]
    if patterns_ and not selected:
        raise ConfigError("invalid", f"no run of {participant} matches {', '.join(patterns_)}")
    if not patterns_:
        selected = list(druns_all)
    chosen = {(r.entities.ses, r.key) for r in selected}
    # the sessions of the runs selected; within them every run is planned, the others as
    # ``not selected``, so that the anchor and the order of a session do not follow the filter
    labels = sorted({r.entities.ses for r in selected}, key=session_key)
    if sessions:
        wanted = [s.removeprefix("ses-") for s in sessions]
        unknown = [s for s in wanted if s not in labels]
        if unknown:
            have = ", ".join(lab for lab in labels if lab is not None) or "none"
            raise ConfigError(
                "invalid", f"{participant} has no session {', '.join(unknown)}; sessions: {have}"
            )
        labels = [lab for lab in labels if lab in wanted]

    previous = _previous(out, participant)
    secret = key if key else secrets.token_bytes(32)
    probes = probe_participant(folders, participant)

    planned: list[_Session] = []
    for label in labels:
        text = session_text(participant, label)
        sid = (
            previous.sessions.get(text)
            if key is None and text in previous.sessions
            else func_id(secret, "session", participant, label or "")
        )
        assert sid is not None
        druns, rruns = listing.of(label)
        run_ids = {}
        for r in druns:
            reused = previous.runs.get((text, run_name(r))) if key is None else None
            run_ids[r.key] = reused or func_id(secret, "run", participant, label or "", r.key)
        withheld = withheld_of(label, druns, probes, patterns_, chosen)
        plan = plan_one(
            folders,
            connector,
            participant,
            label,
            druns,
            rruns,
            run_ids,
            withheld,
            config.frame_limit_mm,
            probes,
        )
        job, note, session_inputs = contour_job(plan, folders, connector)
        contours = None
        if job is not None:
            contours, note = cut_contours(job, plan, folders.deriv_root)
            if contours is None:
                session_inputs = []
        inputs, failed, work = pack_inputs(plan, folders, connector, session_inputs)
        if failed:
            plan = replace(plan, details={**plan.details, **failed})
        planned.append(_Session(label, sid, plan, inputs, contours, note, work))

    # nothing is written into a directory that holds the target of a linked input
    refuse_in_sources_of_links(out, planned, folders)

    # what was found, and an estimate
    n_runs = sum(len(s.plan.runs) for s in planned)
    n_ready = sum(len(s.inputs) for s in planned)
    cannot = [(s, r) for s in planned for r in s.plan.runs if r.reason not in (None, NOT_SELECTED)]
    n_left_out = sum(r.reason == NOT_SELECTED for s in planned for r in s.plan.runs)
    if raw_root is None:
        raw_said = "no raw dataset found (give it with --raw DIR)"
    else:
        named = raw if raw is not None else (path if raw_root == given else raw_root)
        raw_said = f"raw series in {named} (BIDS)"
    version = folders.match.pipeline_version
    pipeline = f"fMRIPrep {version}" if version else "fMRIPrep"
    line = (
        f"Functional: {_plural(n_runs, 'run')} in {_plural(len(planned), 'session')} of "
        f"{participant} in {_shown(deriv, given)} ({pipeline}), {raw_said}; "
        f"{n_ready} can be packed (T1w space)"
    )
    if cannot:
        shown = ", ".join(f"{r.name} ({r.reason})" for _, r in cannot[:6]) + (
            f" and {len(cannot) - 6} more" if len(cannot) > 6 else ""
        )
        line += f", {len(cannot)} cannot: {shown}"
    if n_left_out:
        line += f", {n_left_out} not selected (--epi-runs)"
    say(line + ".")
    for line in _registration_lines(planned, participant):
        say(line)
    kept = (
        set()
        if force
        else {
            run.run
            for item in planned
            for run in item.inputs
            if pack_.current_pack(out / item.sid, run, config) is not None
        }
    )
    work = sum(n for item in planned for run, n in item.work.items() if run not in kept)
    to_pack = n_ready - len(kept)
    if n_ready:
        parts = [f"EPI packs: {_plural(to_pack, 'run')} to pack"]
        if kept:
            parts.append(f"{len(kept)} kept from an earlier call into {out}")
        if work:
            seconds = RATE_S_PER_VOLUME * work / RATE_VOXELS
            parts.append(
                f"about {_duration(seconds)} in one process (an estimate: "
                f"{RATE_S_PER_VOLUME} s per volume of both streams on {RATE_VOXELS:,} voxels, "
                "scaled by voxels)"
            )
        say(", ".join(parts) + ".")

    # pack, session by session, and the stream last; an earlier stream.json stays until the
    # new one is renamed over it, so a page served from ``out`` keeps a whole stream
    out.mkdir(parents=True, exist_ok=True)
    infos: list[SessionInfo] = []
    for item in planned:
        text = session_text(participant, item.label)
        names = {r.run: r.name for r in item.plan.runs}

        kept_here: list[str] = []

        def report(
            run: str,
            info: object,
            reason: str | None,
            seconds: float,
            kept: bool,
            text: str = text,
            names: Mapping[str, str] = names,
            kept_here: list[str] = kept_here,
        ) -> None:
            name = names.get(run, run)
            if reason is None and info is not None:
                n = getattr(info, "n_volumes", 0)
                if kept:
                    kept_here.append(run)
                    say(f"{text}: {name}: {n} volumes, kept (its pack is current), {seconds:.1f} s")
                else:
                    say(f"{text}: {name}: {n} volumes, packed, {seconds:.1f} s")
            else:
                say(f"{text}: {name}: not packed ({reason}), {seconds:.1f} s")

        started = time.perf_counter()
        info = pack_.pack_session(
            out, item.sid, item.plan, item.inputs, item.contours, config, report, force=force
        )
        infos.append(info)
        packed = sum(1 for r in info.runs if r.reason is None)
        said = f"; {item.contour_note}" if item.contour_note else ""
        again = (
            f" ({len(kept_here)} kept, {packed - len(kept_here)} packed now)" if kept_here else ""
        )
        say(
            f"{text}: {packed} of {_plural(len(info.runs), 'run')} packed{again}, "
            f"{time.perf_counter() - started:.1f} s{said}"
        )
    stream = StreamInfo(
        surfcensus_version=__version__,
        label=participant,
        sessions=tuple(item.sid for item in planned),
        order=SESSION_ORDER,
    )
    with atomic_path(out / STREAM_JSON) as tmp:
        tmp.write_text(info_text(stream))
    _remove_stale(out, previous.listed, stream.sessions)
    return stream


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{max(1, round(seconds))} s"
    return f"{round(seconds / 60)} min"


def epi_session(
    path: Path,
    subject: str,
    out: Path,
    session: str | None = None,
    raw: Path | None = None,
    config: EpiConfig | None = None,
    key: bytes | None = None,
) -> SessionInfo:
    """Pack one session of ``subject`` (``sub-01`` or ``01``) found under ``path`` into
    ``out``: :func:`epi_subject` with that one session, so ``out`` gets a ``stream.json`` of
    one session. ``session`` names one when the participant has several; without it, a
    participant with several sessions is ``ConfigError("invalid")``. ``key`` keys the session
    and run ids (``func.func_id``); a random one when None."""
    if session is None:
        folders = find_folders(_absolute(path), raw)
        participant = _participant(subject)
        made = connectors_.connector(folders.match.connector)
        try:
            listed = made.list_runs(folders.deriv_root, participant, folders.match)
        except ConnectorError as err:
            raise SourceError("unreadable", err.detail) from None
        labels = sorted({r.entities.ses for r in listed}, key=session_key)
        if len(labels) > 1:
            named = ", ".join(lab or "none" for lab in labels)
            raise ConfigError("invalid", f"{participant} has several sessions ({named}); name one")
    stream = epi_subject(
        path,
        subject,
        out,
        sessions=None if session is None else [session],
        raw=raw,
        config=config,
        key=key,
    )
    info = read_info(_absolute(out) / stream.sessions[0] / SESSION_JSON, SessionInfo)
    if info is None:  # written by epi_subject just now
        raise ConfigError("invalid", "the session was packed and its session.json cannot be read")
    return info
