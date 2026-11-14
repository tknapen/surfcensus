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
import os
import re
import secrets
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import nibabel as nib

from surfcensus import __version__
from surfcensus._io import atomic_path
from surfcensus.config import ConfigError, EpiConfig
from surfcensus.func import ID_PATTERN, SESSION_JSON, STREAM_JSON, bids, func_id, geometry
from surfcensus.func import connectors as connectors_
from surfcensus.func import contours as contours_
from surfcensus.func import pack as pack_
from surfcensus.func import session as session_
from surfcensus.func.connectors.base import Connector, ConnectorError
from surfcensus.func.model import SessionInfo, StreamInfo, info_text, read_info
from surfcensus.records import (
    NO_PREPROCESSED_SERIES,
    NO_RAW_SERIES,
    NOT_FETCHED,
    NOT_SELECTED,
    PACK_FAILURE,
    ConnectorMatch,
    DerivRun,
    PlaneContours,
    RawRun,
    RunPackInput,
    SessionPlan,
    SessionRun,
)
from surfcensus.sources import patterns
from surfcensus.sources.base import FUNC_READ_ROLES, FileRole, FuncRole, SourceError

RATE_S_PER_VOLUME = 0.114
"""Seconds to pack one volume of both streams on :data:`RATE_VOXELS` voxels, one process:
measured on the worked example (``ds000102`` sub-01 run 1, 146 volumes, portable voxel map;
16.6 s on one shared 4-CPU Linux machine, 2026-11-14). The estimate scales it by voxels."""
RATE_VOXELS = 109_368
"""The voxels of the grid :data:`RATE_S_PER_VOLUME` was measured on (49 x 62 x 36, 3 mm)."""

SESSION_ORDER = "session label"
"""The rule of the order of a subject's sessions in its stream (``StreamInfo.order``)."""

_FS_FILES: tuple[FileRole, ...] = (
    FileRole.T1,
    FileRole.LH_WHITE,
    FileRole.RH_WHITE,
    FileRole.LH_PIAL,
    FileRole.RH_PIAL,
)
"""The FreeSurfer files the contours are cut from, inputs of every pack with contours."""

Progress = Callable[[str], None]


@dataclass(frozen=True)
class FuncFolders:
    """The three folders a session needs, as :func:`find_folders` found them."""

    deriv_root: Path
    match: ConnectorMatch
    raw_root: Path | None
    """None: the runs are listed with ``no raw series`` and the message names ``--raw``."""
    fs_subjects_dir: Path | None
    """None: the sessions are packed without contours."""


# --- the folders ----------------------------------------------------------------------------


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(path))


def _detect(folder: Path) -> tuple[Connector, ConnectorMatch] | None:
    try:
        return connectors_.detect(folder)
    except ConnectorError as err:
        raise SourceError("unreadable", f"{folder}: {err.detail}") from None


def _subjects_dir_of(folder: Path) -> Path | None:
    """The FreeSurfer subjects directory that the surface source finds for an fMRIPrep
    output ``folder`` (``sourcedata/freesurfer/``, or ``freesurfer/`` beside it), or None."""
    from surfcensus.sources.local import detect_layout

    try:
        record, _ = detect_layout(folder, nested=False, quiet=True)
    except SourceError:
        return None
    if record.layout not in ("fmriprep", "fmriprep_legacy"):
        return None
    return Path(record.root) / record.fs_root


def _deriv_candidates(record_root: Path, fs_root: str, layout: str) -> list[Path]:
    """Where the fMRIPrep output lies for a surface layout: the folder above
    ``sourcedata/freesurfer``, or the ``fmriprep/`` folder beside ``freesurfer/``."""
    subjects = record_root / fs_root
    if layout == "fmriprep":
        return [record_root]
    if layout == "fmriprep_legacy":
        return [record_root / "fmriprep"]
    out = []
    if subjects.name == "freesurfer" and subjects.parent.name == "sourcedata":
        out.append(subjects.parent.parent)
    if subjects.name == "freesurfer":
        out.append(subjects.parent / "fmriprep")
    return out


def _derivative(given: Path) -> tuple[Path, ConnectorMatch, Path | None]:
    from surfcensus.sources.local import detect_layout

    looked: list[Path] = []
    # 1. what the surface source finds, then the fMRIPrep output that holds those subjects
    try:
        record, _ = detect_layout(given, quiet=True)
    except SourceError as err:
        if err.reason in ("not found", "invalid"):
            raise
        record = None
    if record is not None:
        subjects = Path(record.root) / record.fs_root
        for candidate in _deriv_candidates(Path(record.root), record.fs_root, record.layout):
            looked.append(candidate)
            if candidate.is_dir():
                found = _detect(candidate)
                if found is not None:
                    return candidate, found[1], subjects
    # 2. the folder itself is a derivative (one that holds no FreeSurfer subjects, say)
    looked.append(given)
    found = _detect(given)
    if found is not None:
        return given, found[1], _subjects_dir_of(given)
    # 3. the folders of derivatives/, in name order
    derivatives = given if given.name == "derivatives" else given / "derivatives"
    if derivatives.is_dir():
        try:
            names = sorted(e.name for e in os.scandir(derivatives) if e.is_dir())
        except OSError as err:
            raise SourceError("unreadable", f"{derivatives}: {err.strerror or err}") from None
        for name in names:
            candidate = derivatives / name
            looked.append(candidate)
            found = _detect(candidate)
            if found is not None:
                return candidate, found[1], _subjects_dir_of(candidate)
    shown = ", ".join(_shown(p, given) for p in dict.fromkeys(looked))
    raise SourceError(
        "layout",
        f"no derivative of a known pipeline ({', '.join(connectors_.CONNECTORS)}) with "
        f"functional runs found in {given}: looked at {shown}. A derivative has a "
        "dataset_description.json that names the pipeline and sub-*/[ses-*/]func/ folders",
    )


def _is_raw(folder: Path) -> bool:
    try:
        return bids.is_raw_dataset(folder)
    except OSError as err:
        raise SourceError("unreadable", f"{folder}: {err.strerror or err}") from None


def _raw_dataset(deriv: Path, raw: Path | None) -> Path | None:
    if raw is not None:
        raw = _absolute(raw)
        if not raw.is_dir():
            raise SourceError("not found", f"--raw {raw} is not a directory")
        if not _is_raw(raw):
            raise SourceError(
                "layout",
                f"--raw {raw} holds no raw BIDS functional series "
                "(sub-*/[ses-*/]func/*_bold.nii[.gz])",
            )
        return raw
    for chain in (_absolute(deriv), deriv.resolve()):
        for folder in chain.parents:
            if folder.name == "derivatives" and _is_raw(folder.parent):
                return folder.parent
    inside = deriv / "sourcedata" / "raw"
    if inside.is_dir() and _is_raw(inside):
        return inside
    return None


def find_folders(path: Path, raw: Path | None = None) -> FuncFolders:
    """The derivative, the raw dataset and the FreeSurfer subjects directory of ``path`` by
    the rules of docs/epi_interface.md, "Which folders are found by rule", for a local folder.

    The derivative: the fMRIPrep output that holds the FreeSurfer subjects that the surface
    source finds for ``path`` (``sources.local.detect_layout``: the folder above
    ``sourcedata/freesurfer/``, or ``fmriprep/`` beside ``freesurfer/``); else ``path`` itself
    when a connector recognises it (a derivative without FreeSurfer subjects); else the first
    folder of ``path/derivatives/`` (or of ``path`` when it is ``derivatives/``), in name
    order, that a connector recognises. The raw dataset: ``raw``; else the BIDS dataset whose
    ``derivatives/`` holds the derivative; else ``sourcedata/raw/`` inside the derivative;
    else None. The FreeSurfer subjects directory: the one the surface source finds, or None.

    Raises ``SourceError``: ``not found`` for a path that is no directory, ``layout`` when no
    derivative of a known pipeline is found (the message says where it looked) or ``raw``
    holds no raw series, ``unreadable`` for a folder that cannot be listed.
    """
    given = _absolute(path)
    if not given.is_dir():
        raise SourceError("not found", f"{given} is not a directory")
    deriv, match, subjects = _derivative(given)
    return FuncFolders(
        deriv_root=deriv,
        match=match,
        raw_root=_raw_dataset(deriv, raw),
        fs_subjects_dir=subjects,
    )


# --- places that are only read ----------------------------------------------------------------


def _read_places(folders: FuncFolders) -> list[tuple[str, Path]]:
    """The derivative and the raw dataset, and what a link at their top level leads to; the
    places of the FreeSurfer subjects directory come from ``sources.local.source_places``."""
    places: list[tuple[str, Path]] = []
    for what, root in (
        ("the derivative", folders.deriv_root),
        ("the raw dataset", folders.raw_root),
    ):
        if root is None:
            continue
        places.append((what, root))
        try:
            with os.scandir(root) as entries:
                links = sorted(e.name for e in entries if e.is_symlink())
        except OSError:
            continue
        for name in links:
            try:
                target = (root / name).resolve()
            except (OSError, RuntimeError):
                continue
            if target.is_dir():
                places.append((f"the directory that {name} in {what} links to,", target))
    return places


def refuse_out(out: Path, folders: FuncFolders) -> None:
    """Raise ``ConfigError("invalid", ...)`` when ``out`` is, or lies inside, a folder that is
    read (the raw dataset, the derivative, the FreeSurfer subjects directory) or what a link
    at the top level of one of them leads to."""
    from surfcensus.sources.local import refuse_in_sources

    subjects = [folders.fs_subjects_dir] if folders.fs_subjects_dir is not None else []
    refuse_in_sources(out, subjects, what="output", also=_read_places(folders))


# --- files without content ------------------------------------------------------------------


def _stem_key(name: str) -> str | None:
    try:
        return bids.run_key(bids.parse_entities(name.split(".", 1)[0]))
    except ValueError:
        return None


def _placeholders(
    root: Path, participant: str, extra: Sequence[Path] = ()
) -> dict[tuple[str | None, str], list[tuple[str, int | None]]]:
    """The git-annex placeholders of a participant's functional files under ``root``: for
    each ``(session, run key)`` the files (relative to ``root``) and their sizes from the
    keys. Links without content are found in the listing (``readlink``); a regular file of at
    most 32 kB is read at its head (``sources.patterns.annex_pointer``)."""
    found: dict[tuple[str | None, str], list[tuple[str, int | None]]] = {}
    try:
        places = list(bids.func_dirs(root, participant))
    except OSError:
        return found
    folders = [(session, func) for session, func in places]
    folders += [(None, p) for p in extra]
    for session, folder in folders:
        try:
            entries = list(os.scandir(folder))
        except OSError:
            continue
        for entry in sorted(entries, key=lambda e: e.name):
            key = _stem_key(entry.name)
            if key is None:
                continue
            path = Path(entry.path)
            size: int | None = None
            try:
                link = patterns.annex_link(path)
                if link is not None:
                    size = patterns.annex_key_size(link[1])
                elif entry.is_file():
                    st = entry.stat()
                    if st.st_size > patterns.POINTER_MAX_BYTES:
                        continue
                    with path.open("rb") as stream:
                        head = stream.read(patterns.POINTER_PROBE_BYTES)
                    pointer = patterns.annex_pointer(head, st.st_size)
                    if pointer is None:
                        continue
                    size = patterns.annex_key_size(pointer)
                else:
                    continue
            except OSError:
                continue
            ses = session
            if ses is None:
                try:
                    ses = bids.parse_entities(entry.name.split(".", 1)[0]).ses
                except ValueError:
                    continue
            location = path.relative_to(root).as_posix()
            found.setdefault((ses, key), []).append((location, size))
    return found


def _dangling(root: Path, participant: str) -> dict[tuple[str | None, str], list[str]]:
    """The links of a participant's ``func/`` folders under ``root`` whose target is missing
    and that are no git-annex links (those are placeholders, :func:`_placeholders`): for each
    ``(session, run key)`` the file names. From the listing and ``readlink`` alone."""
    found: dict[tuple[str | None, str], list[str]] = {}
    try:
        places = list(bids.func_dirs(root, participant))
    except OSError:
        return found
    for session, folder in places:
        try:
            entries = sorted(os.scandir(folder), key=lambda e: e.name)
        except OSError:
            continue
        for entry in entries:
            key = _stem_key(entry.name)
            if key is None:
                continue
            try:
                if not entry.is_symlink() or os.path.exists(entry.path):
                    continue
                if patterns.annex_link(Path(entry.path)) is not None:
                    continue
            except OSError:
                continue
            found.setdefault((session, key), []).append(entry.name)
    return found


_SERIES_REASONS = {
    NO_RAW_SERIES: ("raw", ("_bold.nii.gz", "_bold.nii")),
    NO_PREPROCESSED_SERIES: ("preprocessed", ("_desc-preproc_bold.nii.gz", "_desc-preproc_bold.nii")),
}
"""A missing series, what it is called, and the ends of the names of such a series."""


def _dangling_details(
    plan: SessionPlan,
    raw_links: Mapping[tuple[str | None, str], list[str]],
    deriv_links: Mapping[tuple[str | None, str], list[str]],
) -> SessionPlan:
    """The detail of a run whose series is missing because the file there is a link whose
    target is missing: that it is, so that ``no preprocessed series`` (or ``no raw series``)
    does not read as "the pipeline wrote none"."""
    details = dict(plan.details)
    for item in plan.runs:
        if item.reason not in _SERIES_REASONS or item.run in details:
            continue
        what, ends = _SERIES_REASONS[item.reason]
        links = (raw_links if what == "raw" else deriv_links).get(
            (plan.session, item.deriv.key), []
        )
        names = [n for n in links if n.endswith(ends) and (what == "raw" or "_space-" in n)]
        if names:
            words = "is a link" if len(names) == 1 else "are links"
            details[item.run] = (
                f"the {what} series {', '.join(names)} {words} whose target is missing"
            )
    return replace(plan, details=details)


def _not_fetched(files: Sequence[tuple[str, int | None]]) -> str:
    names = ", ".join(name for name, _ in files)
    known = [size for _, size in files if size is not None]
    amount = f"; {patterns.size_words(sum(known))}" if known else ""
    verb = (
        "file is a git-annex placeholder" if len(files) == 1 else "files are git-annex placeholders"
    )
    return f"{len(files)} {verb} whose content is not here: {names}{amount}"


# --- ordering and ids -----------------------------------------------------------------------


def _natural(label: str) -> tuple[tuple[int, int, str], ...]:
    """Numbers in a label compared as numbers: ``2`` before ``10``, ``pre`` after ``2``."""
    return tuple(
        (0, int(part), "") if part.isascii() and part.isdigit() else (1, 0, part)
        for part in re.split(r"([0-9]+)", label)
        if part
    )


def session_key(label: str | None) -> tuple[int, tuple[tuple[int, int, str], ...]]:
    """The order of a subject's sessions in its stream: no ``ses-`` (a dataset without
    sessions) first, then the labels with numbers compared as numbers."""
    return (0, ()) if label is None else (1, _natural(label))


def _participant(subject: str) -> str:
    participant = subject if subject.startswith("sub-") else f"sub-{subject}"
    if not bids.valid_participant(participant):
        raise ConfigError(
            "invalid", f"{subject!r} is no BIDS participant (sub-<letters and digits>)"
        )
    return participant


def _session_label(participant: str, label: str | None) -> str:
    return participant if label is None else f"{participant} ses-{label}"


def _run_name(run: DerivRun) -> str:
    first, _, rest = run.stem.partition("_")
    return rest if first.startswith("sub-") and rest else run.stem


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


# --- the plan and the inputs of one session ---------------------------------------------------


def _paths(files: Mapping[FuncRole, object], deriv: Path, raw: Path | None) -> dict[FuncRole, Path]:
    out: dict[FuncRole, Path] = {}
    for role, ref in files.items():
        base = getattr(ref, "base", "deriv")
        location = getattr(ref, "location", "")
        if base == "raw":
            if raw is not None:
                out[role] = raw / location
        else:
            out[role] = deriv / location
    return out


def _fs_subject(subjects: Path | None, participant: str, label: str | None) -> str | None:
    """``sub-X_ses-Y`` when the subjects directory has it, else ``sub-X``, else None."""
    if subjects is None:
        return None
    names = ([f"{participant}_ses-{label}"] if label is not None else []) + [participant]
    for name in names:
        try:
            if patterns.is_subject_candidate(subjects / name):
                return name
        except OSError:
            continue
    return None


def _failed_plan(
    participant: str,
    label: str | None,
    druns: Sequence[DerivRun],
    run_ids: Mapping[str, str],
    connector: Connector,
    detail: str,
) -> SessionPlan:
    """A session whose plan could not be made (a file that cannot be read): every run is
    listed as ``pack failure`` with the detail."""
    runs = tuple(
        SessionRun(
            run=run_ids[r.key],
            name=_run_name(r),
            deriv=r,
            raw=None,
            display=connector.display(r),
            reason=PACK_FAILURE,
        )
        for r in sorted(druns, key=lambda r: (r.order, r.stem))
    )
    return SessionPlan(
        participant=participant,
        session=label,
        fs_subject=None,
        runs=runs,
        anchor=None,
        frame=None,
        grids={},
        grid_of={},
        details={r.run: detail for r in runs},
    )


def _with_withheld(plan: SessionPlan, withheld: Mapping[str, tuple[str, str | None]]) -> SessionPlan:
    """A failed plan whose withheld runs keep their own reason and detail."""
    if not withheld:
        return plan
    key_of = {r.run: r.deriv.key for r in plan.runs}
    runs = tuple(
        replace(r, reason=withheld[key_of[r.run]][0]) if key_of[r.run] in withheld else r
        for r in plan.runs
    )
    details = dict(plan.details)
    for r in runs:
        if r.deriv.key in withheld:
            detail = withheld[r.deriv.key][1]
            if detail:
                details[r.run] = detail
            else:
                details.pop(r.run, None)
    return replace(plan, runs=runs, details=details)


_READ_ERRORS = (OSError, ValueError, EOFError, ConnectorError, nib.filebasedimages.ImageFileError)


def _no_path(err: BaseException, *roots: Path | None) -> str:
    text = str(getattr(err, "detail", None) or getattr(err, "strerror", None) or err)
    for root in roots:
        if root is not None:
            text = text.replace(f"{root}{os.sep}", "").replace(str(root), ".")
    return f"{type(err).__name__}: {text}" if not isinstance(err, ConnectorError) else text


@dataclass
class _Session:
    label: str | None
    sid: str
    plan: SessionPlan
    inputs: list[RunPackInput]
    contours: dict[str, Sequence[PlaneContours]] | None
    contour_note: str
    voxels_volumes: int


def _contours(
    plan: SessionPlan,
    folders: FuncFolders,
    connector: Connector,
) -> tuple[dict[str, Sequence[PlaneContours]] | None, str, list[tuple[str, str, Path]]]:
    """The contours of every grid of a session, what to say about them, and the session
    inputs they add to each pack (the fsnative transform and the FreeSurfer files)."""
    if not plan.grids:
        return None, "", []
    if plan.fs_subject is None or folders.fs_subjects_dir is None:
        return None, "without contours: no FreeSurfer subject of this session", []
    fs_dir = folders.fs_subjects_dir / plan.fs_subject
    from surfcensus.sources.local import unfetched_render_files

    unfetched = unfetched_render_files(fs_dir)
    if unfetched:
        return None, f"without contours: {plan.fs_subject}: {unfetched[0]}", []
    xfm = plan.files.get(FuncRole.XFM_ANAT_FSNATIVE)
    if xfm is None:
        return None, "without contours: the derivative has no transform to fsnative", []
    xfm_path = folders.deriv_root / xfm.location
    try:
        q = connector.anat_to_fsnative({FuncRole.XFM_ANAT_FSNATIVE: xfm_path})
        files = patterns.resolve_subject_files(fs_dir)
        out: dict[str, Sequence[PlaneContours]] = {}
        for key, grid in plan.grids.items():
            surfaces = contours_.surfaces_in_display(fs_dir, q, grid)
            out[key] = contours_.plane_contours(surfaces, grid)
    except _READ_ERRORS as err:
        return None, f"without contours: {_no_path(err, fs_dir, folders.deriv_root)}", []
    inputs: list[tuple[str, str, Path]] = [(FuncRole.XFM_ANAT_FSNATIVE.value, "deriv", xfm_path)]
    inputs += [(role.value, "freesurfer", files[role]) for role in _FS_FILES if role in files]
    return out, f"contours from {plan.fs_subject}", inputs


def _inputs(
    plan: SessionPlan,
    folders: FuncFolders,
    connector: Connector,
    session_inputs: Sequence[tuple[str, str, Path]],
) -> tuple[list[RunPackInput], dict[str, str], int]:
    """The inputs of every packable run of a plan, the details of those whose inputs could
    not be read, and the volumes times voxels of the packable runs (for the estimate)."""
    items = {r.run: r for r in plan.runs if r.reason is None}
    if not items or plan.anchor is None:
        return [], {}, 0

    def paths_of(item: SessionRun) -> dict[FuncRole, Path]:
        files: dict[FuncRole, object] = dict(item.deriv.files)
        if item.raw is not None:
            files.update({k: v for k, v in item.raw.files.items() if k not in files})
        found = _paths(files, folders.deriv_root, folders.raw_root)
        return {role: p for role, p in found.items() if role in FUNC_READ_ROLES}

    # the anchor is a run of the session, packed or not (``session.plan_session``)
    anchor = next(r for r in plan.runs if r.run == plan.anchor.run)
    anchor_paths = paths_of(anchor)
    failed: dict[str, str] = {}
    try:
        anchor_rigid = connector.raw_to_anat(anchor.deriv, anchor_paths)
    except _READ_ERRORS as err:
        detail = f"the anchor's transform cannot be read: {_no_path(err, folders.deriv_root)}"
        return [], {run: detail for run in items}, 0
    out: list[RunPackInput] = []
    work = 0
    for run, item in items.items():
        paths = paths_of(item)
        try:
            rigid = connector.raw_to_anat(item.deriv, paths)
            raw_affine = session_.raw_affine_choice(
                paths[FuncRole.RAW_BOLD], paths.get(FuncRole.REFERENCE)
            )
            motion = connector.motion(item.deriv, paths)
            corrections = connector.corrections(item.deriv, paths)
            tr_s = connector.repetition_time(item.deriv, paths)
            n = int(nib.load(str(paths[FuncRole.PRE_BOLD])).shape[3])
        except _READ_ERRORS as err:
            failed[run] = _no_path(err, folders.deriv_root, folders.raw_root)
            continue
        grid = plan.grids[plan.grid_of[run]]
        extra = list(session_inputs)
        if run != plan.anchor.run:
            extra.insert(0, ("anchor_xfm_to_anat", "deriv", anchor_paths[FuncRole.XFM_TO_ANAT]))
        out.append(
            RunPackInput(
                run=run,
                name=item.name,
                files=paths,
                grid=grid,
                display_to_anat=geometry.compose(),
                anchor_run=plan.anchor.run,
                anchor_matrix=anchor_rigid.matrix,
                raw_affine=raw_affine,
                motion=motion,
                corrections=corrections,
                rigid=rigid,
                tr_s=tr_s,
                session_inputs=tuple(extra),  # pyrefly: ignore[bad-argument-type]
            )
        )
        work += n * grid.shape[0] * grid.shape[1] * grid.shape[2]
    return out, failed, work


# --- packing --------------------------------------------------------------------------------


def _shown(path: Path, base: Path) -> str:
    """``path`` relative to ``base`` when it lies inside it (``.`` for ``base`` itself), else
    as it is."""
    if path == base:
        return "."
    if path.is_relative_to(base):
        return path.relative_to(base).as_posix()
    return str(path)


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
) -> StreamInfo:
    """Pack every session of ``subject`` (``sub-01`` or ``01``) found under ``path``, or the
    sessions named (``1`` or ``ses-1``), into ``<out>/<session_id>/``, and write
    ``<out>/stream.json`` last: the sessions in the order of the subject's one continuous
    stream (decision D38), by :func:`session_key` (no session first, then the labels with
    numbers compared as numbers). Within a session the runs follow in acquisition order.

    ``runs`` restricts the runs packed to those whose derivative stem matches one of the
    shell patterns (``*task-rest*``); the others are not listed. ``key`` keys the ids
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
    try:
        druns_all = connector.list_runs(deriv, participant, folders.match)
        rruns_all = bids.list_raw_runs(raw_root, participant) if raw_root is not None else []
    except ConnectorError as err:
        raise SourceError("unreadable", f"{deriv}: {err.detail}") from None
    except OSError as err:
        raise SourceError("unreadable", str(err)) from None
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
    unfetched_deriv = _placeholders(deriv, participant, [deriv / participant / "figures"])
    unfetched_raw = _placeholders(raw_root, participant) if raw_root is not None else {}
    dangling_deriv = _dangling(deriv, participant)
    dangling_raw = _dangling(raw_root, participant) if raw_root is not None else {}

    planned: list[_Session] = []
    for label in labels:
        text = _session_label(participant, label)
        sid = (
            previous.sessions.get(text)
            if key is None and text in previous.sessions
            else func_id(secret, "session", participant, label or "")
        )
        assert sid is not None
        druns = [r for r in druns_all if r.entities.ses == label]
        rruns = [r for r in rruns_all if r.entities.ses == label]
        run_ids = {}
        for r in druns:
            reused = previous.runs.get((text, _run_name(r))) if key is None else None
            run_ids[r.key] = reused or func_id(secret, "run", participant, label or "", r.key)
        withheld: dict[str, tuple[str, str | None]] = {}
        for r in druns:
            files = unfetched_deriv.get((label, r.key), []) + unfetched_raw.get((label, r.key), [])
            if (label, r.key) not in chosen:
                withheld[r.key] = (NOT_SELECTED, f"--epi-runs leaves it out ({', '.join(patterns_)})")
            elif files:
                withheld[r.key] = (NOT_FETCHED, _not_fetched(files))
        fs_subject = _fs_subject(folders.fs_subjects_dir, participant, label)
        try:
            plan = session_.plan_session(
                participant,
                label,
                druns,
                rruns,
                connector,
                deriv,
                raw_root,
                run_ids,
                fs_subject,
                config.frame_limit_mm,
                withheld=withheld,
            )
        except _READ_ERRORS as err:
            detail = f"the session cannot be planned: {_no_path(err, deriv, raw_root)}"
            plan = _failed_plan(participant, label, druns, run_ids, connector, detail)
            plan = _with_withheld(plan, withheld)
        plan = _dangling_details(plan, dangling_raw, dangling_deriv)
        if raw_root is None:
            hint = "no raw BIDS dataset was found for the derivative; give it with --raw DIR"
            details = dict(plan.details)
            for r in plan.runs:
                if r.reason == NO_RAW_SERIES and r.run not in details:
                    details[r.run] = hint
            plan = replace(plan, details=details)
        contours, note, session_inputs = _contours(plan, folders, connector)
        inputs, failed, work = _inputs(plan, folders, connector, session_inputs)
        if failed:
            plan = replace(plan, details={**plan.details, **failed})
        planned.append(_Session(label, sid, plan, inputs, contours, note, work))

    # what was found, and an estimate
    n_runs = sum(len(s.plan.runs) for s in planned)
    n_ready = sum(len(s.inputs) for s in planned)
    cannot = [(s, r) for s in planned for r in s.plan.runs if r.reason is not None]
    raw_said = (
        f"raw series in {_shown(raw_root, given)} (BIDS)"
        if raw_root is not None
        else "no raw dataset found (give it with --raw DIR)"
    )
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
    say(line + ".")
    work = sum(s.voxels_volumes for s in planned)
    if work:
        seconds = RATE_S_PER_VOLUME * work / RATE_VOXELS
        say(
            f"EPI packs: {_plural(n_ready, 'run')} to pack, about {_duration(seconds)} in one "
            f"process (an estimate: {RATE_S_PER_VOLUME} s per volume of both streams on "
            f"{RATE_VOXELS:,} voxels, scaled by voxels)."
        )

    # pack, session by session, and the stream last
    out.mkdir(parents=True, exist_ok=True)
    (out / STREAM_JSON).unlink(missing_ok=True)
    infos: list[SessionInfo] = []
    for item in planned:
        text = _session_label(participant, item.label)
        names = {r.run: r.name for r in item.plan.runs}

        def report(
            run: str,
            info: object,
            reason: str | None,
            seconds: float,
            text: str = text,
            names: Mapping[str, str] = names,
        ) -> None:
            if reason is None and info is not None:
                n = getattr(info, "n_volumes", 0)
                say(f"{text}: {names.get(run, run)}: {n} volumes, {seconds:.1f} s")
            else:
                say(f"{text}: {names.get(run, run)}: not packed ({reason}), {seconds:.1f} s")

        started = time.perf_counter()
        info = pack_.pack_session(
            out, item.sid, item.plan, item.inputs, item.contours, config, report
        )
        infos.append(info)
        packed = sum(1 for r in info.runs if r.reason is None)
        said = f"; {item.contour_note}" if item.contour_note else ""
        say(
            f"{text}: {packed} of {_plural(len(info.runs), 'run')} packed, "
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
