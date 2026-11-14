"""Phase 2f, the functional inventory of a run, and the lease of a session for the pack phase
(docs/epi_interface.md, "Step 2: the plan"; work package E2.B).

Private side of a run, like ``inventory/subjects.py``: it takes the private ``Catalog``, roots
and the run's key, and hands the pack phase nothing but a ``records.SessionWork`` inside a
lease. It reads listings, the raw sidecars, the summary and validation reports, the stored
transforms, the brain masks, the confounds tables (their row count) and the headers of the
series (``func.session.plan_session`` and the connector's ``raw_to_anat``, ``corrections``,
``repetition_time``); it reads no volume of a series, fetches nothing and writes nothing
outside ``catalog.sqlite``.

What :func:`run_func_inventory` does, per local dataset of the run with a functional source
(``func.discover.func_source_of``):

* stores the source (``Catalog.upsert_func_source``) and, per participant of the derivative,
  its ``participant_uid`` (``privacy.participant_uid`` under the run's key, over the source
  kind, the stored ``deriv_identity``, the connector and the label);
* per session, its ``session_uid`` and the ``run_uid`` of every run (``privacy.session_uid``,
  ``privacy.run_uid``); its ``inventory_hash`` (:func:`session_hash`); a session whose hash is
  the stored one is skipped without opening a file (decision D21), otherwise it is planned
  (:func:`plan_one`: the anchor among every run, the frame check, the grids, ``not selected``
  from ``EpiConfig.runs``, ``not fetched`` from the git-annex placeholders) and stored with
  its runs (``Catalog.upsert_func_session``);
* participants, sessions and runs that the listing no longer has are marked absent
  (``Catalog.mark_func_absent``); their packs stay until the pack phase removes them.

The planning is that of step 1's ``func.api.epi_subject`` and lives here (the listing of a
participant, the placeholders, the details of dangling links, the plan of a session, the
contour job, the inputs of every packable run): ``func.api`` calls it, so one implementation
serves ``surfcensus epi-session`` and a run directory.
"""

from __future__ import annotations

import fnmatch
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from surfcensus.func import bids
from surfcensus.func import connectors as connectors_
from surfcensus.func.connectors.base import Connector, ConnectorError
from surfcensus.func.discover import FuncFolders, folders_of, func_source_of
from surfcensus.records import (
    FUNC_INVENTORY_PHASE,
    NO_PREPROCESSED_SERIES,
    NO_RAW_SERIES,
    NOT_FETCHED,
    NOT_SELECTED,
    PACK_FAILURE,
    ContourJob,
    Corrections,
    DerivRun,
    FuncFileRef,
    FuncLeaseFacts,
    FuncRunRow,
    FuncSessionRow,
    FuncSourceRecord,
    PhaseReport,
    PlaneContours,
    RawRun,
    RunPackInput,
    SessionPlan,
    SessionRun,
    SessionWork,
    func_signature,
    session_unit,
)
from surfcensus.sources import patterns
from surfcensus.sources.base import (
    FUNC_READ_ROLES,
    FileRef,
    FileRole,
    FuncRole,
    SourceError,
)

if TYPE_CHECKING:
    from surfcensus.config import EpiConfig, RunConfig
    from surfcensus.privacy import HashKey
    from surfcensus.records import AffineChoice
    from surfcensus.sources.base import DatasetRecord, Source
    from surfcensus.store import Catalog


FS_FILES: tuple[FileRole, ...] = (
    FileRole.T1,
    FileRole.LH_WHITE,
    FileRole.RH_WHITE,
    FileRole.LH_PIAL,
    FileRole.RH_PIAL,
)
"""The FreeSurfer files the contours are cut from, inputs of every pack with contours."""

SESSION_ORDER = "session label"
"""The rule of the order of a subject's sessions in its stream (``StreamInfo.order``) when
not every session has an acquisition date: the labels, numbers compared as numbers."""

ACQUISITION_ORDER = "acquisition time"
"""The rule when every session of the participant has an acquisition date in the raw dataset
(decision D42): by date, ties by the label order."""


def _read_errors() -> tuple[type[BaseException], ...]:
    """What reading a functional file can raise. nibabel is imported here and not with the
    module, which loads no numpy (``tests/test_func_step2_interfaces.py``)."""
    import nibabel as nib

    return (OSError, ValueError, EOFError, ConnectorError, nib.filebasedimages.ImageFileError)


# --- files without content ------------------------------------------------------------------


def _stem_key(name: str) -> str | None:
    try:
        return bids.run_key(bids.parse_entities(name.split(".", 1)[0]))
    except ValueError:
        return None


Placeholders = dict[tuple[str | None, str], list[tuple[str, int | None]]]
"""``(session, run key)`` -> the placeholders of the run: location relative to the root
that was scanned, and the size from the git-annex key (None when the key has none)."""


def placeholders(root: Path, participant: str, extra: Sequence[Path] = ()) -> Placeholders:
    """The git-annex placeholders of a participant's functional files under ``root``: for
    each ``(session, run key)`` the files (relative to ``root``) and their sizes from the
    keys. Links without content are found in the listing (``readlink``); a regular file of at
    most 32 kB is read at its head (``sources.patterns.annex_pointer``)."""
    found: Placeholders = {}
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


def dangling(root: Path, participant: str) -> dict[tuple[str | None, str], list[str]]:
    """The links of a participant's ``func/`` folders under ``root`` whose target is missing
    and that are no git-annex links (those are placeholders, :func:`placeholders`): for each
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
    NO_PREPROCESSED_SERIES: (
        "preprocessed",
        ("_desc-preproc_bold.nii.gz", "_desc-preproc_bold.nii"),
    ),
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


def not_fetched_words(files: Sequence[tuple[str, int | None]]) -> str:
    """``2 files are git-annex placeholders whose content is not here: a, b; 1.2 MB``."""
    names = ", ".join(name for name, _ in files)
    known = [size for _, size in files if size is not None]
    amount = f"; {patterns.size_words(sum(known))}" if known else ""
    verb = (
        "file is a git-annex placeholder" if len(files) == 1 else "files are git-annex placeholders"
    )
    return f"{len(files)} {verb} whose content is not here: {names}{amount}"


# --- ordering and names ---------------------------------------------------------------------


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


@dataclass(frozen=True)
class SessionOrder:
    """How the sessions of a participant's stream are ordered (decision D42)."""

    rule: str
    """:data:`ACQUISITION_ORDER` or :data:`SESSION_ORDER`."""
    labels: tuple[str | None, ...]
    """The sessions in stream order."""
    undated: tuple[str, ...]
    """The labels of the sessions without an acquisition date, in label order: empty when
    every session has one (and for a participant without sessions)."""


def order_sessions(
    labels: Sequence[str | None], dates: Mapping[str | None, object] | None = None
) -> SessionOrder:
    """The stream order of ``labels`` (decision D42): when every one is a session (not None)
    with a date in ``dates``, by date, ties by :func:`session_key`; otherwise by
    :func:`session_key` for all of them, with no mixing. Dates are compared as given (the
    naive ``datetime`` of ``bids.session_dates``) and are not kept in the result."""
    by_label = sorted(dict.fromkeys(labels), key=session_key)
    found = dates or {}
    undated = tuple(lab for lab in by_label if lab is not None and found.get(lab) is None)
    named = [lab for lab in by_label if lab is not None]
    if named and len(named) == len(by_label) and not undated:
        ordered = sorted(by_label, key=lambda lab: (found[lab], session_key(lab)))
        return SessionOrder(ACQUISITION_ORDER, tuple(ordered), ())
    return SessionOrder(SESSION_ORDER, tuple(by_label), undated)


def session_order_of(
    raw_root: Path | None, participant: str, labels: Sequence[str | None]
) -> SessionOrder:
    """:func:`order_sessions` with the dates of the raw dataset (``bids.session_dates``);
    label order without a raw dataset or without a labelled session. Reads the participant's
    ``sessions.tsv`` and the ``scans.tsv`` of its sessions."""
    if raw_root is None or not any(lab is not None for lab in labels):
        return order_sessions(labels)
    return order_sessions(labels, bids.session_dates(raw_root, participant, labels))


def order_words(order: SessionOrder) -> str:
    """``in the order of their acquisition times``, or ``in the order of their labels (no
    acquisition date for ses-2 in the raw dataset)``: what the commands say of a participant
    with more than one session. Names no date."""
    if order.rule == ACQUISITION_ORDER:
        return "in the order of their acquisition times"
    named = [lab for lab in order.labels if lab is not None]
    if not named or len(order.undated) == len(named):
        return "in the order of their labels (no acquisition date in the raw dataset)"
    lacking = ", ".join(f"ses-{lab}" for lab in order.undated)
    return (
        f"in the order of their labels, not of their acquisition times (no acquisition date "
        f"for {lacking} in the raw dataset)"
    )


def run_name(run: DerivRun) -> str:
    """The BIDS name of a run without the subject (``ses-1_task-rest_run-1``)."""
    first, _, rest = run.stem.partition("_")
    return rest if first.startswith("sub-") and rest else run.stem


def session_text(participant: str, label: str | None) -> str:
    """``sub-01 ses-1``, or ``sub-01`` in a dataset without sessions."""
    return participant if label is None else f"{participant} ses-{label}"


# --- the listing of a participant -----------------------------------------------------------


@dataclass(frozen=True)
class Listing:
    """What the listing of a participant gives, before any file is opened."""

    participant: str
    deriv_runs: tuple[DerivRun, ...]
    raw_runs: tuple[RawRun, ...]

    def labels(self) -> list[str | None]:
        """The sessions of the participant's derivative runs, in label order
        (:func:`session_key`); the stream order is :func:`session_order_of`."""
        return sorted({r.entities.ses for r in self.deriv_runs}, key=session_key)

    def of(self, label: str | None) -> tuple[list[DerivRun], list[RawRun]]:
        """The derivative and the raw runs of one session."""
        return (
            [r for r in self.deriv_runs if r.entities.ses == label],
            [r for r in self.raw_runs if r.entities.ses == label],
        )


def list_participant(folders: FuncFolders, connector: Connector, participant: str) -> Listing:
    """The derivative's and the raw dataset's runs of ``participant``, from the listings.
    Raises ``SourceError("unreadable", ...)`` for a folder that cannot be listed."""
    deriv, raw_root = folders.deriv_root, folders.raw_root
    try:
        druns = connector.list_runs(deriv, participant, folders.match)
        rruns = bids.list_raw_runs(raw_root, participant) if raw_root is not None else []
    except ConnectorError as err:
        raise SourceError("unreadable", f"{deriv}: {err.detail}") from None
    except OSError as err:
        raise SourceError("unreadable", str(err)) from None
    return Listing(participant, tuple(druns), tuple(rruns))


@dataclass(frozen=True)
class Probes:
    """What the planning of a participant reads besides the listing: the git-annex
    placeholders (heads of files of at most 32 kB) and the dangling links."""

    unfetched_deriv: Placeholders
    unfetched_raw: Placeholders
    dangling_deriv: Mapping[tuple[str | None, str], list[str]]
    dangling_raw: Mapping[tuple[str | None, str], list[str]]


def probe_participant(folders: FuncFolders, participant: str) -> Probes:
    """:class:`Probes` of a participant: the placeholders of its ``func/`` folders in both
    datasets and of its ``figures/`` in the derivative, and the dangling links."""
    deriv, raw_root = folders.deriv_root, folders.raw_root
    return Probes(
        unfetched_deriv=placeholders(deriv, participant, [deriv / participant / "figures"]),
        unfetched_raw=placeholders(raw_root, participant) if raw_root is not None else {},
        dangling_deriv=dangling(deriv, participant),
        dangling_raw=dangling(raw_root, participant) if raw_root is not None else {},
    )


def withheld_of(
    label: str | None,
    deriv_runs: Sequence[DerivRun],
    probes: Probes,
    globs: Sequence[str],
    chosen: set[tuple[str | None, str]] | None = None,
) -> dict[str, tuple[str, str | None]]:
    """The runs of a session that are listed and not packed whatever their files say: those
    that ``globs`` (``--epi-runs``) leave out, on the derivative stem, as ``not selected``,
    and those with git-annex placeholders as ``not fetched``. ``chosen`` (``(session, run
    key)``) overrides which runs the globs select."""
    if chosen is None:
        chosen = {
            (r.entities.ses, r.key)
            for r in deriv_runs
            if not globs or any(fnmatch.fnmatchcase(r.stem, p) for p in globs)
        }
    withheld: dict[str, tuple[str, str | None]] = {}
    for r in deriv_runs:
        files = probes.unfetched_deriv.get((label, r.key), []) + probes.unfetched_raw.get(
            (label, r.key), []
        )
        if (label, r.key) not in chosen:
            withheld[r.key] = (NOT_SELECTED, f"--epi-runs leaves it out ({', '.join(globs)})")
        elif files:
            withheld[r.key] = (NOT_FETCHED, not_fetched_words(files))
    return withheld


def fs_subject_of(subjects: Path | None, participant: str, label: str | None) -> str | None:
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


# --- the plan of one session ----------------------------------------------------------------


def _no_path(err: BaseException, *roots: Path | None, named: bool = True) -> str:
    """The text of ``err`` without the paths of ``roots``, after the name of its class
    unless ``named`` is false (a line for a person, which a class name does not help)."""
    text = str(getattr(err, "detail", None) or getattr(err, "strerror", None) or err)
    for root in roots:
        if root is not None:
            text = text.replace(f"{root}{os.sep}", "").replace(str(root), ".")
    if not named or isinstance(err, ConnectorError):
        return text
    return f"{type(err).__name__}: {text}"


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
            name=run_name(r),
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


def _with_withheld(
    plan: SessionPlan, withheld: Mapping[str, tuple[str, str | None]]
) -> SessionPlan:
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


UNPLANNED = "the session cannot be planned: "
"""The start of the detail of every run of a session whose plan could not be made."""


def plan_one(
    folders: FuncFolders,
    connector: Connector,
    participant: str,
    label: str | None,
    deriv_runs: Sequence[DerivRun],
    raw_runs: Sequence[RawRun],
    run_ids: Mapping[str, str],
    withheld: Mapping[str, tuple[str, str | None]],
    frame_limit_mm: float,
    probes: Probes,
) -> SessionPlan:
    """The plan of one session (``session.plan_session``), with what step 1 adds to it: a
    session whose plan cannot be made lists every run as ``pack failure`` with the reason
    (withheld runs keep theirs; the details start with :data:`UNPLANNED`), a missing series
    that is a dangling link says so, and a run without raw series in a source without raw
    dataset names ``--raw``."""
    from surfcensus.func import session as session_

    deriv, raw_root = folders.deriv_root, folders.raw_root
    fs_subject = fs_subject_of(folders.fs_subjects_dir, participant, label)
    try:
        plan = session_.plan_session(
            participant,
            label,
            deriv_runs,
            raw_runs,
            connector,
            deriv,
            raw_root,
            run_ids,
            fs_subject,
            frame_limit_mm,
            withheld=withheld,
        )
    except _read_errors() as err:
        detail = f"{UNPLANNED}{_no_path(err, deriv, raw_root)}"
        plan = _failed_plan(participant, label, deriv_runs, run_ids, connector, detail)
        plan = _with_withheld(plan, withheld)
    plan = _dangling_details(plan, probes.dangling_raw, probes.dangling_deriv)
    if raw_root is None:
        hint = "no raw BIDS dataset was found for the derivative; give it with --raw DIR"
        details = dict(plan.details)
        for r in plan.runs:
            if r.reason == NO_RAW_SERIES and r.run not in details:
                details[r.run] = hint
        plan = replace(plan, details=details)
    return plan


# --- the contours and the inputs of the packable runs ---------------------------------------


def _paths(
    files: Mapping[FuncRole, FuncFileRef], deriv: Path, raw: Path | None
) -> dict[FuncRole, Path]:
    out: dict[FuncRole, Path] = {}
    for role, ref in files.items():
        if ref.base == "raw":
            if raw is not None:
                out[role] = raw / ref.location
        else:
            out[role] = deriv / ref.location
    return out


def run_files(item: SessionRun) -> dict[FuncRole, FuncFileRef]:
    """The listed files of a run of a plan: the derivative's, then those of its raw run that
    the derivative does not have."""
    files: dict[FuncRole, FuncFileRef] = dict(item.deriv.files)
    if item.raw is not None:
        files.update({k: v for k, v in item.raw.files.items() if k not in files})
    return files


SessionInput = tuple[str, str, Path]
"""``(role, base, path)`` of a file outside a run that its pack depends on."""


def contour_job(
    plan: SessionPlan, folders: FuncFolders, connector: Connector
) -> tuple[ContourJob | None, str, list[SessionInput]]:
    """What the contours of a session are cut from, what to say about them, and the session
    inputs they add to each pack (the fsnative transform and the FreeSurfer files). Reads the
    fsnative transform and the listing of the FreeSurfer subject; cuts nothing
    (:func:`cut_contours`)."""
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
    except _read_errors() as err:
        words = _no_path(err, fs_dir, folders.deriv_root, named=False)
        return None, f"without contours: {words}", []
    inputs: list[SessionInput] = [(FuncRole.XFM_ANAT_FSNATIVE.value, "deriv", xfm_path)]
    inputs += [(role.value, "freesurfer", files[role]) for role in FS_FILES if role in files]
    job = ContourJob(fs_dir=fs_dir, fs_subject=plan.fs_subject, anat_to_fsnative=q)
    return job, f"contours from {plan.fs_subject}", inputs


def cut_contours(
    job: ContourJob, plan: SessionPlan, *roots: Path | None
) -> tuple[dict[str, Sequence[PlaneContours]] | None, str]:
    """The contours of every grid of ``plan`` cut from ``job``, with what to say about them:
    ``contours from <subject>``, or ``without contours: <error>`` (the error named without
    the paths of ``roots`` and of the subject directory) and None when a surface cannot be
    read. Reads the T1 header and the four surfaces."""
    from surfcensus.func import contours as contours_

    try:
        out: dict[str, Sequence[PlaneContours]] = {}
        for key, grid in plan.grids.items():
            surfaces = contours_.surfaces_in_display(job.fs_dir, job.anat_to_fsnative, grid)
            out[key] = contours_.plane_contours(surfaces, grid)
    except _read_errors() as err:
        return None, f"without contours: {_no_path(err, job.fs_dir, *roots, named=False)}"
    return out, f"contours from {job.fs_subject}"


def pack_inputs(
    plan: SessionPlan,
    folders: FuncFolders,
    connector: Connector,
    session_inputs: Sequence[SessionInput],
) -> tuple[list[RunPackInput], dict[str, str], dict[str, int]]:
    """The inputs of every packable run of a plan, the details of those whose inputs could
    not be read, and the volumes times voxels of each packable run (for the estimate)."""
    import nibabel as nib

    from surfcensus.func import geometry
    from surfcensus.func import session as session_

    items = {r.run: r for r in plan.runs if r.reason is None}
    if not items or plan.anchor is None:
        return [], {}, {}

    def paths_of(item: SessionRun) -> dict[FuncRole, Path]:
        found = _paths(run_files(item), folders.deriv_root, folders.raw_root)
        return {role: p for role, p in found.items() if role in FUNC_READ_ROLES}

    # the anchor is a run of the session, packed or not (``session.plan_session``)
    anchor = next(r for r in plan.runs if r.run == plan.anchor.run)
    anchor_paths = paths_of(anchor)
    failed: dict[str, str] = {}
    try:
        anchor_rigid = connector.raw_to_anat(anchor.deriv, anchor_paths)
    except _read_errors() as err:
        detail = f"the anchor's transform cannot be read: {_no_path(err, folders.deriv_root)}"
        return [], {run: detail for run in items}, {}
    out: list[RunPackInput] = []
    work: dict[str, int] = {}
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
        except _read_errors() as err:
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
        work[run] = n * grid.shape[0] * grid.shape[1] * grid.shape[2]
    return out, failed, work


def link_targets(plans: Sequence[SessionPlan], folders: FuncFolders) -> list[tuple[str, Path]]:
    """For every listed input file of the plans that is a link, or is reached through one,
    the directory that holds its target: the raw series and sidecars, the derivative's files
    of every run (selected or not) and of the session, and the FreeSurfer files of the
    contours. From the listing and ``realpath`` alone; no file is opened."""
    paths: list[Path] = []
    for plan in plans:
        for run in plan.runs:
            paths += _paths(run_files(run), folders.deriv_root, folders.raw_root).values()
        paths += _paths(dict(plan.files), folders.deriv_root, folders.raw_root).values()
        if plan.fs_subject is not None and folders.fs_subjects_dir is not None:
            try:
                found = patterns.resolve_subject_files(folders.fs_subjects_dir / plan.fs_subject)
            except OSError:
                found = {}
            paths += [found[role] for role in FS_FILES if role in found]
    places: dict[Path, str] = {}
    for path in paths:
        written = Path(os.path.abspath(path))
        real = Path(os.path.realpath(path))
        if real != written and real.parent not in places:
            places[real.parent] = f"the directory that the link {path.name} leads into,"
    return [(what, place) for place, what in places.items()]


# --- the inventory hash ---------------------------------------------------------------------


def _fs_refs(folders: FuncFolders, fs_subject: str | None) -> dict[FileRole, FileRef]:
    """The listed FreeSurfer files of the contours of a session (stat only)."""
    if fs_subject is None or folders.fs_subjects_dir is None:
        return {}
    from surfcensus.sources.local import subject_file_refs

    try:
        found = subject_file_refs(folders.fs_subjects_dir / fs_subject)
    except OSError:
        return {}
    return {role: found[role] for role in FS_FILES if role in found}


def session_hash(
    folders: FuncFolders,
    connector: Connector,
    listing: Listing,
    label: str | None,
    run_ids: Mapping[str, str],
    config: EpiConfig,
) -> str:
    """``func_sessions.inventory_hash`` of a session, from the listings alone (decision D21):
    ``records.func_signature`` over the listed files of every derivative run of the session in
    name order (its own files in ``FuncRole`` order, then those of every raw run of its key,
    the echoes included), the session's own files, the FreeSurfer files of the contours, and
    the extra lines ``EpiConfig.inventory_hash()``, the FreeSurfer subject's name, the raw
    dataset's presence and the run ids. For a participant with sessions also the list of its
    session labels and the listing facts of the files its sessions' acquisition dates are read
    from (``bids.session_date_lines``: ``sessions.tsv`` and every ``scans.tsv``), so that a
    change of either plans every session of the participant again and its stored
    ``stream_order`` follows (decision D42). Opens no file: ``scandir``, ``stat`` and
    ``readlink`` only."""
    druns, rruns = listing.of(label)
    refs: list[FuncFileRef] = []
    for run in sorted(druns, key=lambda r: (r.order, r.stem)):
        refs += [run.files[role] for role in FuncRole if role in run.files]
        for raw in sorted((r for r in rruns if r.key == run.key), key=lambda r: (r.order, r.stem)):
            refs += [raw.files[role] for role in FuncRole if role in raw.files]
    session_files = connector.session_files(
        folders.deriv_root, listing.participant, label, folders.match
    )
    refs += [session_files[role] for role in FuncRole if role in session_files]
    fs_subject = fs_subject_of(folders.fs_subjects_dir, listing.participant, label)
    extra = [
        f"inventory\t{config.inventory_hash()}",
        f"fs_subject\t{fs_subject or ''}",
        f"raw\t{'yes' if folders.raw_root is not None else 'no'}",
        *(f"run\t{key}\t{uid}" for key, uid in sorted(run_ids.items())),
    ]
    labels = listing.labels()
    if any(lab is not None for lab in labels):
        extra.append("sessions\t" + "\t".join(lab or "" for lab in labels))
        if folders.raw_root is not None:
            extra += bids.session_date_lines(folders.raw_root, listing.participant, labels)
    return func_signature(refs, _fs_refs(folders, fs_subject), extra)


# --- what phase 2f stores of a run ----------------------------------------------------------


@dataclass(frozen=True)
class _Facts:
    n_volumes: int | None = None
    tr_s: float | None = None
    raw_affine: AffineChoice | None = None
    reference_image: bool | None = None
    rigid: bool | None = None
    dof: int | None = None
    corrections: Corrections | None = None


def _run_facts(
    item: SessionRun, folders: FuncFolders, connector: Connector
) -> tuple[_Facts, str | None]:
    """The statements of a run from its headers, sidecars, reports and transform, each read
    on its own (a file that is missing or cannot be read leaves its fact empty), and the
    error of the first fact of a packable run that could not be read. Reads no volume; of a
    run whose files are git-annex placeholders nothing is read."""
    import nibabel as nib

    from surfcensus.func import session as session_

    if item.reason == NOT_FETCHED:
        return _Facts(), None
    found = _paths(run_files(item), folders.deriv_root, folders.raw_root)
    paths = {role: p for role, p in found.items() if role in FUNC_READ_ROLES}
    errors: list[str] = []
    caught = _read_errors()

    def failed(err: BaseException) -> None:
        errors.append(_no_path(err, folders.deriv_root, folders.raw_root))

    facts = _Facts()
    series = paths.get(FuncRole.PRE_BOLD, paths.get(FuncRole.RAW_BOLD))
    if series is not None:
        try:
            facts = replace(facts, n_volumes=int(nib.load(str(series)).shape[3]))
        except caught as err:
            failed(err)
    try:
        facts = replace(facts, tr_s=connector.repetition_time(item.deriv, paths))
    except caught as err:
        failed(err)
    if FuncRole.XFM_TO_ANAT in paths:
        try:
            rigid = connector.raw_to_anat(item.deriv, paths)
            facts = replace(facts, rigid=rigid.rigid, dof=rigid.dof)
        except caught as err:
            failed(err)
    if FuncRole.RAW_BOLD in paths:
        try:
            choice = session_.raw_affine_choice(
                paths[FuncRole.RAW_BOLD], paths.get(FuncRole.REFERENCE)
            )
            facts = replace(facts, raw_affine=choice.choice, reference_image=choice.reference_image)
        except caught as err:
            failed(err)
    try:
        facts = replace(facts, corrections=connector.corrections(item.deriv, paths))
    except caught as err:
        failed(err)
    return facts, errors[0] if errors and item.reason is None else None


def _echoes(item: SessionRun, raw_runs: Sequence[RawRun]) -> int:
    labels = {r.entities.echo for r in raw_runs if r.key == item.deriv.key and r.entities.echo}
    return max(item.deriv.echoes, len(labels), 1)


def session_row(
    plan: SessionPlan,
    *,
    folders: FuncFolders,
    connector: Connector,
    listing: Listing,
    session_uid: str,
    participant_uid: str,
    stream_order: int,
    subject_uid: str | None,
    inventory_hash: str,
    probes: Probes,
) -> FuncSessionRow:
    """What phase 2f stores of a planned session and its runs."""
    _, rruns = listing.of(plan.session)
    order = {run: n for n, run in enumerate(plan.order)}
    details = dict(plan.details)
    runs: list[FuncRunRow] = []
    for n, item in enumerate(plan.runs):
        facts, error = _run_facts(item, folders, connector)
        if error is not None and item.run not in details:
            details[item.run] = error
        grid = plan.grids.get(plan.grid_of[item.run]) if item.run in plan.grid_of else None
        frame = plan.frame.runs.get(item.run) if plan.frame is not None else None
        files = run_files(item)
        runs.append(
            FuncRunRow(
                run_uid=item.run,
                session_uid=session_uid,
                run_key=item.deriv.key,
                name=item.name,
                entities=item.deriv.entities,
                session_order=order.get(item.run),
                acquisition_order=n,
                anchor=plan.anchor is not None and plan.anchor.run == item.run,
                selected=item.reason != NOT_SELECTED,
                packable=item.reason is None,
                reason=item.reason,
                detail=details.get(item.run),
                missing_roles=item.missing,
                n_volumes=facts.n_volumes,
                tr_s=facts.tr_s,
                echoes=_echoes(item, rruns),
                display=item.display,
                spaces=item.deriv.spaces,
                grid=grid,
                raw_affine=facts.raw_affine,
                reference_image=facts.reference_image,
                rigid=facts.rigid,
                dof=facts.dof,
                corrections=facts.corrections,
                frame=frame,
                files=tuple(files[role] for role in FuncRole if role in files),
            )
        )
    problem: str | None = None
    unplanned = [d for d in details.values() if d.startswith(UNPLANNED)]
    if unplanned:
        problem = f"unreadable: {unplanned[0]}"
    else:
        files: list[tuple[str, int | None]] = []
        for item in plan.runs:
            if item.reason != NOT_FETCHED:
                continue
            key = (plan.session, item.deriv.key)
            files += probes.unfetched_deriv.get(key, []) + probes.unfetched_raw.get(key, [])
        if files:
            problem = f"{NOT_FETCHED}: {not_fetched_words(files)}"
    session_files = connector.session_files(
        folders.deriv_root, listing.participant, plan.session, folders.match
    )
    return FuncSessionRow(
        session_uid=session_uid,
        participant_uid=participant_uid,
        session=plan.session,
        stream_order=stream_order,
        subject_uid=subject_uid,
        fs_subject=plan.fs_subject,
        anchor=plan.anchor,
        anchor_note=plan.anchor_note,
        frame=plan.frame,
        n_grids=len(plan.grids),
        problem=problem,
        inventory_hash=inventory_hash,
        files=tuple(session_files[role] for role in FuncRole if role in session_files),
        runs=tuple(runs),
    )


# --- phase 2f -------------------------------------------------------------------------------


def participants_of(folders: FuncFolders) -> list[str]:
    """The participants of the derivative that have a ``func/`` folder, by label. Raises
    ``SourceError("unreadable", ...)`` for a derivative that cannot be listed."""
    root = folders.deriv_root
    try:
        _, dirs = bids.list_dir(root, root)
    except OSError as err:
        raise SourceError("unreadable", f"{root}: {err.strerror or err}") from None
    out = []
    for name in dirs:
        if not bids.valid_participant(name):
            continue
        try:
            if bids.func_dirs(root, name):
                out.append(name)
        except OSError as err:
            raise SourceError("unreadable", f"{root}: {err.strerror or err}") from None
    return out


@dataclass
class _Dataset:
    dataset_id: int
    record: DatasetRecord
    source: FuncSourceRecord
    folders: FuncFolders
    connector: Connector
    listings: list[Listing]


def _dataset_id(catalog: Catalog, record: DatasetRecord) -> int | None:
    rows = catalog.query(
        "SELECT dataset_id FROM datasets WHERE source = ? AND identity = ? AND pipeline = ?",
        (record.source, record.identity, record.pipeline),
    )
    return None if not rows else int(rows[0]["dataset_id"])


def _subject_uid(catalog: Catalog, dataset_id: int, fs_subject: str | None) -> str | None:
    if fs_subject is None:
        return None
    rows = catalog.query(
        "SELECT subject_uid FROM subjects WHERE dataset_id = ? AND subject_label = ? "
        "AND present = 1",
        (dataset_id, fs_subject),
    )
    return None if not rows else str(rows[0]["subject_uid"])


def run_func_inventory(
    catalog: Catalog,
    sources: Sequence[Source],
    config: RunConfig,
    key: HashKey,
    *,
    force: bool = False,
) -> PhaseReport:
    """Phase 2f for the local sources of a run (module docstring). Returns a report of phase
    ``func_inventory`` whose units are sessions: done (stored now), skipped (hash unchanged),
    failed (a session whose plan could not be made: ``unreadable``). A dataset without a
    functional source, a run whose ``EpiConfig.enabled`` is false and a census run give an
    empty report and no row. ``force`` plans every session again. Raises ``SourceError`` for
    a derivative or a raw dataset that cannot be listed, before anything is recorded.

    Datasets are those the surface inventory stored (a dataset of a source that the catalog
    does not hold is passed over). Everything is listed first, so that a listing that fails
    records nothing; then each session is skipped or planned and stored, one transaction per
    session; then what is no longer listed is marked absent, per functional source."""
    from surfcensus.config import epi_settings

    start = time.perf_counter()
    settings = epi_settings(config)
    empty = PhaseReport(phase=FUNC_INVENTORY_PHASE, done=0, skipped=0, failed=0, failures=())
    if config.mode != "local" or not settings.enabled:
        return empty
    raw = None if settings.raw is None else Path(settings.raw)

    # 1. every listing, before anything is recorded
    found: list[_Dataset] = []
    for source in sources:
        if source.kind != "local":
            continue
        for record in source.datasets():
            dataset_id = _dataset_id(catalog, record)
            if dataset_id is None:
                continue
            func = func_source_of(record, raw)
            if func is None:
                continue
            stored = catalog.func_source(dataset_id)
            if stored is not None and stored.connector == func.connector:
                func = replace(func, deriv_identity=stored.deriv_identity)
            connector = connectors_.connector(func.connector)
            folders = folders_of(func, Path(record.root) / record.fs_root)
            listings = [
                list_participant(folders, connector, participant)
                for participant in participants_of(folders)
            ]
            found.append(_Dataset(dataset_id, record, func, folders, connector, listings))

    # 2. each session: skipped by its hash, or planned and stored
    done = skipped = 0
    failures: list[tuple[str, str]] = []
    notes: list[str] = []
    for item in found:
        func_id = catalog.upsert_func_source(item.dataset_id, item.source)
        uid_of = _Uids(key, item.record.source, item.source)
        present_participants: list[str] = []
        present_sessions: list[str] = []
        for listing in item.listings:
            participant = listing.participant
            puid = uid_of.participant(participant)
            labels = listing.labels()
            present_participants.append(puid)
            probes: Probes | None = None
            # the hashes first, from the listing; the dates are read only when a session of
            # the participant is planned (decisions D21, D42)
            pending: list[tuple[str | None, str, dict[str, str], str]] = []
            for label in labels:
                suid = uid_of.session(participant, label)
                present_sessions.append(suid)
                druns, _ = listing.of(label)
                run_ids = {r.key: uid_of.run(participant, label, r.key) for r in druns}
                digest = session_hash(
                    item.folders, item.connector, listing, label, run_ids, settings
                )
                if not force and catalog.func_inventory_hash(suid) == digest:
                    skipped += 1
                    continue
                pending.append((label, suid, run_ids, digest))
            order: SessionOrder | None = None
            if pending:
                order = session_order_of(item.folders.raw_root, participant, labels)
            catalog.upsert_func_participant(
                func_id,
                puid,
                participant,
                len(labels),
                session_order=None if order is None else order.rule,
            )
            for label, suid, run_ids, digest in pending:
                assert order is not None
                stream_order = order.labels.index(label)
                druns, rruns = listing.of(label)
                if probes is None:
                    probes = probe_participant(item.folders, participant)
                withheld = withheld_of(label, druns, probes, settings.runs)
                plan = plan_one(
                    item.folders,
                    item.connector,
                    participant,
                    label,
                    druns,
                    rruns,
                    run_ids,
                    withheld,
                    settings.frame_limit_mm,
                    probes,
                )
                row = session_row(
                    plan,
                    folders=item.folders,
                    connector=item.connector,
                    listing=listing,
                    session_uid=suid,
                    participant_uid=puid,
                    stream_order=stream_order,
                    subject_uid=_subject_uid(catalog, item.dataset_id, plan.fs_subject),
                    inventory_hash=digest,
                    probes=probes,
                )
                catalog.upsert_func_session(row)
                done += 1
                if row.problem is not None and row.problem.startswith("unreadable"):
                    failures.append((session_unit(suid), "unreadable"))
        gone = catalog.mark_func_absent(func_id, present_participants, present_sessions)
        if gone:
            pseudonym = catalog.pseudonym(item.dataset_id)
            notes.append(
                f"{pseudonym}: {len(gone)} functional session(s) are no longer listed and were "
                "marked absent; their packs are removed by the pack phase"
            )
    failed = tuple(sorted(failures))
    return PhaseReport(
        phase=FUNC_INVENTORY_PHASE,
        done=done,
        skipped=skipped,
        failed=len(failed),
        failures=failed,
        notes=tuple(notes),
        seconds=time.perf_counter() - start,
    )


@dataclass(frozen=True)
class _Uids:
    """The keyed identifiers of a functional source (``privacy.py``)."""

    key: HashKey
    kind: str
    source: FuncSourceRecord

    def participant(self, participant: str) -> str:
        from surfcensus import privacy

        return privacy.participant_uid(
            self.key, self.kind, self.source.deriv_identity, self.source.connector, participant
        )

    def session(self, participant: str, label: str | None) -> str:
        from surfcensus import privacy

        return privacy.session_uid(
            self.key,
            self.kind,
            self.source.deriv_identity,
            self.source.connector,
            participant,
            label,
        )

    def run(self, participant: str, label: str | None, run_key: str) -> str:
        from surfcensus import privacy

        return privacy.run_uid(
            self.key,
            self.kind,
            self.source.deriv_identity,
            self.source.connector,
            participant,
            label,
            run_key,
        )


# --- the lease of a session -----------------------------------------------------------------


def session_work(facts: FuncLeaseFacts, config: EpiConfig) -> SessionWork:
    """The session of ``facts`` made again from its files, for the lease of the pack phase
    (``Run``'s ``func_lease``): the derivative and the raw dataset listed again for the
    participant, the listing compared with ``facts.inventory_hash`` (``SourceError("changed",
    ...)`` when it differs, so nothing is packed from files other than those inventoried),
    the plan (:func:`plan_one` with the stored ``run_uid``s), the inputs of every packable run
    (``records.RunPackInput``, with the anchor's transform and the session's inputs) and the
    contour job. The paths are those of the files in place: a local lease.

    The contours are not cut here: the worker cuts them from ``SessionWork.contours`` with
    :func:`cut_contours`. A session whose listing gives other runs than the stored ones (a
    run that came or went since the inventory) is ``changed`` as well, since its hash covers
    the run ids."""
    source = facts.source
    if not Path(source.deriv_root).is_dir():
        raise SourceError("not found", "the derivative of this session is not there")
    folders = folders_of(source, facts.fs_subjects_dir)
    connector = connectors_.connector(source.connector)
    listing = list_participant(folders, connector, facts.participant)
    druns, rruns = listing.of(facts.session)
    unknown = [r for r in druns if r.key not in facts.run_ids]
    run_ids = {r.key: facts.run_ids[r.key] for r in druns if r.key in facts.run_ids}
    digest = session_hash(folders, connector, listing, facts.session, run_ids, config)
    if unknown or not druns or digest != facts.inventory_hash:
        raise SourceError(
            "changed",
            f"the files of {session_text(facts.participant, facts.session)} differ from the "
            "inventory; run the inventory again",
        )
    probes = probe_participant(folders, facts.participant)
    withheld = withheld_of(facts.session, druns, probes, config.runs)
    plan = plan_one(
        folders,
        connector,
        facts.participant,
        facts.session,
        druns,
        rruns,
        run_ids,
        withheld,
        config.frame_limit_mm,
        probes,
    )
    job, note, extra = contour_job(plan, folders, connector)
    inputs, failed, _ = pack_inputs(plan, folders, connector, extra)
    return SessionWork(
        session_uid=facts.session_uid,
        participant_uid=facts.participant_uid,
        plan=plan,
        inputs=tuple(inputs),
        input_failures=failed,
        contours=job,
        contour_note=note,
    )


# --- the order lines ------------------------------------------------------------------------

GROUP_ABOVE = 3
"""Participants whose order lines would read alike are named in one line when there are more
than this many of them."""


def func_order_lines(catalog: Catalog) -> list[str]:
    """What ``review`` and ``inventory`` say of the order of the sessions of each present
    participant with more than one present session (decision D42): ``sub-01: 2 sessions in
    the order of their acquisition times``, or ``... in the order of their labels (no
    acquisition date in the raw dataset)``; more than :data:`GROUP_ABOVE` participants with
    the same words in one line (``12 participants: their sessions in the order of ...``). A
    participant whose sessions fell back to labels because some, not all, have a date is
    named on its own with the sessions that lack one.

    The rule is the stored one (``func_participants.session_order``); which sessions lack a
    date is read again from the raw dataset for the participants ordered by label (their
    ``sessions.tsv`` and ``scans.tsv``), as the fetch lines read the placeholders again. No
    date is printed."""
    rows = catalog.query(
        "SELECT p.participant, p.session_order, f.raw_root, s.session_label "
        "FROM func_sessions s JOIN func_participants p USING (participant_uid) "
        "JOIN func_sources f USING (func_id) "
        "WHERE s.present = 1 AND p.present = 1 AND f.present = 1 "
        "ORDER BY p.participant, s.stream_order"
    )
    by: dict[str, tuple[str, str | None, list[str | None]]] = {}
    for row in rows:
        entry = by.setdefault(row["participant"], (str(row["session_order"]), row["raw_root"], []))
        entry[2].append(row["session_label"])
    alike: dict[str, list[tuple[str, int]]] = {}
    single: list[str] = []
    for participant, (rule, raw_root, labels) in by.items():
        if len(labels) < 2:
            continue
        if rule == ACQUISITION_ORDER:
            words = order_words(SessionOrder(rule, tuple(labels), ()))
        else:
            raw = None if raw_root is None else Path(raw_root)
            found = session_order_of(raw, participant, labels)
            if found.undated:
                words = order_words(SessionOrder(SESSION_ORDER, tuple(labels), found.undated))
            else:  # the raw dataset changed since the inventory: say only the rule
                words = "in the order of their labels"
            named = [lab for lab in labels if lab is not None]
            if found.undated and len(found.undated) < len(named):
                single.append(f"{participant}: {len(labels)} sessions {words}")
                continue
        alike.setdefault(words, []).append((participant, len(labels)))
    lines: list[str] = []
    for words, members in alike.items():
        if len(members) > GROUP_ABOVE:
            lines.append(f"{len(members)} participants: their sessions {words}")
        else:
            lines += [f"{participant}: {n} sessions {words}" for participant, n in members]
    return [f"{line}." for line in sorted(lines + single)]


# --- the fetch lines ------------------------------------------------------------------------


def func_fetch_lines(catalog: Catalog) -> list[str]:
    """The lines that name the functional runs with git-annex placeholders and the one
    command that fetches their files (``datalad -C <dataset> get ...``, else ``git -C <repo>
    annex get ...``), printed after the surface fetch line by ``review`` and ``inventory``
    (decision D35). A raw dataset and a derivative are often two DataLad datasets; the command
    names each. Empty when no present run has placeholders.

    One line per such run (``sub-01 ses-1_task-rest_run-1: not fetched: ...``), then one
    ``Fetch the functional files (<size>) with: <command>`` per dataset (the innermost DataLad
    dataset, else git repository, that holds the files; ``sources.local.fetch_commands``).
    The files are found again from the listing and the heads of the small files of the
    participants named (as the inventory found them); a run fetched since the inventory is
    still named until the next inventory, and its files are left out of the command."""
    from surfcensus.sources.local import fetch_commands

    rows = catalog.query(
        "SELECT f.deriv_root, f.raw_root, p.participant, s.session_label, r.run_key, r.name, "
        "r.detail FROM func_runs r JOIN func_sessions s USING (session_uid) "
        "JOIN func_participants p USING (participant_uid) JOIN func_sources f USING (func_id) "
        "WHERE r.present = 1 AND s.present = 1 AND p.present = 1 AND r.reason = ? "
        "ORDER BY p.participant, s.stream_order, r.acquisition_order",
        (NOT_FETCHED,),
    )
    if not rows:
        return []
    lines: list[str] = []
    files: list[tuple[Path, int | None]] = []
    scanned: dict[tuple[str, str | None, str], tuple[Placeholders, Placeholders]] = {}
    for row in rows:
        participant, label = row["participant"], row["session_label"]
        lines.append(f"{participant} {row['name']}: {NOT_FETCHED}: {row['detail']}")
        deriv = Path(row["deriv_root"])
        raw = None if row["raw_root"] is None else Path(row["raw_root"])
        cache = (str(deriv), row["raw_root"], participant)
        if cache not in scanned:
            scanned[cache] = (
                placeholders(deriv, participant, [deriv / participant / "figures"]),
                placeholders(raw, participant) if raw is not None else {},
            )
        in_deriv, in_raw = scanned[cache]
        for root, found in ((deriv, in_deriv), (raw, in_raw)):
            if root is None:
                continue
            for location, size in found.get((label, row["run_key"]), []):
                files.append((root / location, size))
    for command, size in fetch_commands(list(dict.fromkeys(files))):
        amount = f" ({patterns.size_words(size)})" if size else ""
        lines.append(f"Fetch the functional files{amount} with: {command}")
    return lines
