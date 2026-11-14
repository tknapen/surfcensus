"""Plain records that more than one module needs.

They live here so that no module has to import another one's module to name a type. This
module imports the standard library and ``sources/base.py`` only; the types of other
modules that its fields name are imported for the type checker alone.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Collection, Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from surfcensus.sources.base import FileRef, FileRole, FuncRole, Hemi

if TYPE_CHECKING:
    import pandas as pd

    from surfcensus.config import RunConfig
    from surfcensus.func.model import PackInfo, SessionInfo
    from surfcensus.render.model import RenderInfo
    from surfcensus.store import PublicCatalog

Status = Literal["done", "failed"]

SUBJECT_UID_PATTERN = r"[0-9a-f]{16}"
"""What a ``subject_uid`` looks like (``privacy.subject_uid``): the name of a subject in every
path, table and URL. A ``render_key`` has the same form."""

METRIC_STAGES: tuple[str, ...] = ("regional", "fsaverage", "reference", "score")
"""The stages of phase 4 that ``surfcensus metrics`` runs when none is named, in order."""

ALL_METRIC_STAGES: tuple[str, ...] = (*METRIC_STAGES, "native", "merge")
"""Every stage that can be named: ``native`` gives cluster centroids their native position
from the vertex maps of the renders, ``merge`` joins the per-dataset parts of each table."""


@dataclass(frozen=True)
class Provenance:
    """What the inventory learns about a subject from its build stamp, logs and file names.

    Built by ``inventory.provenance.subject_provenance`` and stored by
    ``Catalog.upsert_subject``. A None says that the fact could not be established, which is
    different from False.
    """

    build_stamp: str | None
    fs_version: str | None
    """For example ``6.0.1``; None for a development build."""
    fs_major: int | None
    flags: tuple[str, ...]
    """Names of the recon-all flags of all invocations of all logs, sorted, without values."""
    n_invocations: int | None
    finished: bool | None
    t2pial: bool | None
    """None: no log was read and the pial variant does not say; likewise ``flairpial``."""
    flairpial: bool | None
    hires: bool | None
    expert: bool | None
    xopts: bool | None
    edited: bool | None
    pial_variant: str | None
    """``pial``, ``pial.T1``, ``pial.T2`` or ``pial.FLAIR``."""
    holes: tuple[int, int] | None
    """Surface holes before topology fixing, left and right."""
    truncations: Mapping[Hemi, tuple[int, int] | None]
    """The two truncation counts of the thickness measurement, in the order of the log."""
    participant: str | None
    """``sub-01`` of ``sub-01_ses-2``."""
    longitudinal: bool


@dataclass(frozen=True)
class PhaseReport:
    """What a phase command did, for the closing line and the exit code."""

    phase: str
    done: int
    skipped: int
    failed: int
    """Units in the failed state when the command ends, retried or not."""
    failures: tuple[tuple[str, str], ...]
    """``(unit, reason)``; a unit is a ``subject_uid``, ``dataset:<pseudonym>`` or ``run``."""
    notes: tuple[str, ...] = ()
    """Sentences for the user."""
    seconds: float = 0.0


@dataclass(frozen=True)
class WorkRow:
    """A row of the ``work`` ledger: one unit of one phase, done or failed."""

    phase: str
    unit: str
    """A ``subject_uid``, ``dataset:<pseudonym>`` or ``run``."""
    config_hash: str
    input_signature: str
    """Private: it never leaves the catalog (see ``sources.base.input_signature``)."""
    status: Status
    reason: str | None = None
    detail: str | None = None
    """Private; the scrubbed text of the failure."""
    output: str | None = None
    """Path relative to the run directory, when the unit has one output file."""
    surfcensus_version: str | None = None
    """None in a row to be recorded: the catalog writes the running version."""
    finished_at: str | None = None
    """UTC, ISO 8601. None in a row to be recorded: the catalog writes the current time."""


@dataclass(frozen=True)
class RenderRow:
    """A row of the ``renders`` table: the render of one subject, done or failed."""

    subject_uid: str
    config_hash: str
    """``RenderConfig.config_hash()``."""
    input_signature: str
    """Private, like ``WorkRow.input_signature``."""
    video: bool
    status: Status
    tile_hash: str | None = None
    """``RenderConfig.tile_hash()``; None for a failed render, likewise ``render_key``."""
    render_key: str | None = None
    reason: str | None = None
    """A ``RenderError`` reason, or ``source changed``."""
    detail: str | None = None
    media_path: str | None = None
    """``media/<subject_uid>``."""
    n_frames: int | None = None
    template_id: str | None = None
    rendered_at: str | None = None
    """UTC, ISO 8601. None in a row to be recorded: the catalog writes the current time."""


@dataclass(frozen=True)
class RenderResult:
    """What a render worker returns to the parent process."""

    info: RenderInfo
    extras: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    """Table name -> the rows of that table for this subject. Empty until the intensity
    metric is computed inside the render (milestone 5)."""


@dataclass(frozen=True)
class RunContext:
    """What a phase function may touch: the public side of a run.

    Phase code receives this and never a ``Run``, so it cannot reach the private catalog
    (accessions, roots, input signatures) even by attribute access.
    """

    run_dir: Path
    config: RunConfig
    public: PublicCatalog
    lease: Callable[[str, Collection[FileRole]], AbstractContextManager[Mapping[FileRole, Path]]]
    """``lease(subject_uid, roles)``: local paths of the roles the subject has, while open."""
    render_key: Callable[[str, Mapping[str, str]], str]
    """``render_key(tile_hash, digests)``: ``privacy.render_key`` with the run secret bound."""
    scrub: Callable[[str], str]
    """``privacy.scrub`` with this run's pseudonyms and roots."""
    func_lease: Callable[[str], AbstractContextManager[SessionWork]] | None = None
    """``func_lease(session_uid)``: the session made again from its files, with local paths,
    while open (step 2 of the EPI interface; ``Run`` binds it). None where the run has no
    functional source, and in a context made before step 2."""
    pack_key: Callable[[str, Mapping[str, str]], str] | None = None
    """``pack_key(pack_hash, digests)``: ``privacy.pack_key`` with the run secret bound."""


# --- the EPI movie interface (docs/epi_interface.md) ----------------------------------------
#
# The records below are what the connector, the session stage, the pack and the page of step 1
# hand each other. They hold Python values only (no numpy array), so that they compare, hash
# and pickle as plain data: a 4x4 matrix is a :data:`Matrix4` of Python floats, which is also
# the form in which the portable voxel map is computed (docs/epi_interface.md, Determinism).

Row4 = tuple[float, float, float, float]
Matrix4 = tuple[Row4, Row4, Row4, Row4]
"""A 4x4 affine as four rows of Python floats."""

IDENTITY4: Matrix4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def to_matrix4(rows: Iterable[Iterable[float]]) -> Matrix4:
    """A :data:`Matrix4` from four rows of four numbers (a list, a tuple or a numpy array).

    Raises ``ValueError`` for another shape, a value that is not finite, or a last row that is
    not ``0 0 0 1``: every matrix of the EPI interface is an affine point map.
    """
    values = [[float(x) for x in row] for row in rows]
    if len(values) != 4 or any(len(row) != 4 for row in values):
        raise ValueError("an affine has 4 rows of 4 numbers")
    if not all(math.isfinite(x) for row in values for x in row):
        raise ValueError("an affine has finite numbers only")
    if values[3] != [0.0, 0.0, 0.0, 1.0]:
        raise ValueError("the last row of an affine is 0 0 0 1")

    def row(r: list[float]) -> Row4:
        return (r[0], r[1], r[2], r[3])

    return (row(values[0]), row(values[1]), row(values[2]), row(values[3]))


FuncBase = Literal["raw", "deriv"]
"""Which root a functional file lies under: the raw BIDS dataset or the derivative."""

DisplayLayout = Literal["anat", "native"]
"""``anat``: the preprocessed series lies in the anatomy's world (``space-T1w``);
``native``: in the run's own reference frame (docs/epi_interface.md, The display geometry)."""

Generation = Literal["scanner-T1w", "boldref-T1w"]
"""The layout of an fMRIPrep derivative: below 23.2, and 23.2 and later."""

CorrectionState = Literal["applied", "not applied", "unknown"]

AffineChoice = Literal["header", "qform", "reference", "header, unchecked"]
"""Which affine the raw stream of a run used (``session.raw_affine_choice``): the header's
``img.affine`` that the reference image agrees with; the qform, which it agrees with when
``img.affine`` does not; the reference image's own affine when neither does; ``img.affine``
without a usable reference image."""

AnchorRule = Literal["acquisition_time", "name_order"]

# Why a run is listed and not packed (docs/epi_interface.md, "The pack as a stage"). The first
# nine are decided from the listing and the headers, the last two when a pack is made.
NO_RAW_SERIES = "no raw series"
NO_PREPROCESSED_SERIES = "no preprocessed series"
STANDARD_SPACE_ONLY = "standard space only"
NATIVE_SPACE_ONLY = "native space only"
"""Not in the list of the spec: a run whose only preprocessed series is in its own frame, which
step 1 does not pack (decision 1 for Tomas)."""
NO_TRANSFORM = "no transform to the anatomy"
NO_BRAIN_MASK = "no brain mask"
VOLUME_COUNTS_DIFFER = "volume counts differ"
RUNS_ON_DIFFERENT_GRIDS = "runs on different grids"
MULTI_ECHO = "multi-echo"
SOURCE_CHANGED = "source changed"
PACK_FAILURE = "pack failure"
NO_CONFOUNDS = "no confounds table"
"""Not in the list of the spec: a run with its series and transform whose confounds table is
missing (fMRIPrep writes it at level ``full`` only); the pack needs it for the volume count
and the stills. Before this reason existed such a run was given ``no preprocessed series``."""
NOT_FETCHED = "not fetched"
"""Not in the list of the spec: a file the pack would read is a git-annex placeholder (a link
without its content, or a pointer file) in a DataLad or git-annex clone that was not fetched
(decision D35). ``SessionPlan.details`` names the files; the fetch command is step 2."""
NOT_SELECTED = "not selected"
"""Not in the list of the spec: a run of the session that ``--epi-runs`` leaves out. It is still
a run of the session: it is listed, it can be the session's anchor, and it takes its place in
the session's acquisition order, so that the anchor and the order do not depend on the
filter; it is not packed."""

PACK_REASONS: tuple[str, ...] = (
    NO_RAW_SERIES,
    NO_PREPROCESSED_SERIES,
    STANDARD_SPACE_ONLY,
    NATIVE_SPACE_ONLY,
    NO_TRANSFORM,
    NO_BRAIN_MASK,
    VOLUME_COUNTS_DIFFER,
    RUNS_ON_DIFFERENT_GRIDS,
    MULTI_ECHO,
    SOURCE_CHANGED,
    PACK_FAILURE,
    NO_CONFOUNDS,
    NOT_FETCHED,
    NOT_SELECTED,
)
"""Every reason a run is listed and not packed, or its pack failed. ``runs on different
grids`` is kept so that the list stays as pinned, and is given to no run: runs on different
grids are each packed on their own grid (decision D38)."""

DISPLAY_REASONS: tuple[str, ...] = (
    NO_PREPROCESSED_SERIES,
    STANDARD_SPACE_ONLY,
    NATIVE_SPACE_ONLY,
    MULTI_ECHO,
)
"""The reasons a connector's ``display`` can give for a run without a display grid."""


@dataclass(frozen=True)
class RunEntities:
    """The BIDS entities that key a functional run (docs/epi_interface.md, The unit).

    Labels are kept as written (``run`` is ``"01"`` for ``run-01``); ``func.bids.run_key``
    compares the run index as a number. A missing entity is None.
    """

    sub: str
    ses: str | None = None
    task: str | None = None
    acq: str | None = None
    ce: str | None = None
    rec: str | None = None
    dir: str | None = None
    run: str | None = None
    echo: str | None = None


@dataclass(frozen=True)
class FuncFileRef:
    """One file of a functional run or session as a listing describes it. Nothing was opened."""

    role: FuncRole
    base: FuncBase
    location: str
    """POSIX path relative to the root of ``base`` (the raw dataset or the derivative)."""
    size: int
    etag_or_mtime: str
    """Local: ``st_mtime_ns`` in decimal, as ``FileRef.etag_or_mtime``."""
    variant: str | None = None
    """For a role with several candidate names: the one that was listed, for example
    ``desc-coreg_boldref`` among the reference images of fMRIPrep 23.2 and later."""


@dataclass(frozen=True)
class RawRun:
    """A raw BIDS series and its sidecar, from the listing of the raw dataset (``func.bids``)."""

    entities: RunEntities
    key: str
    """``func.bids.run_key``: equal for the raw series and its derivative files."""
    order: tuple[str | int, ...]
    """``func.bids.name_order``, the sort key of the name-order rule."""
    stem: str
    """The BIDS stem without ``_bold``: ``sub-01_ses-1_task-rest_run-1``."""
    files: Mapping[FuncRole, FuncFileRef]
    """``raw_bold`` and, when there is one, ``raw_sidecar``."""


@dataclass(frozen=True)
class DerivRun:
    """A functional run as a connector lists it in a derivative (``Connector.list_runs``)."""

    entities: RunEntities
    key: str
    order: tuple[str | int, ...]
    stem: str
    """The derivative's BIDS stem of the run, without space, desc and suffix."""
    spaces: tuple[str, ...]
    """The spaces of the preprocessed series the derivative has for the run, sorted; ``""``
    for the run's own frame. Empty when it has none."""
    files: Mapping[FuncRole, FuncFileRef]
    """The files of every role the run has, for the display layout the connector chose:
    ``pre_bold`` and ``pre_mask`` are those of that layout."""
    echoes: int = 1
    """Echoes of the raw acquisition when the connector can tell from the derivative's names;
    more than one is ``multi-echo``. fMRIPrep's default multi-echo output carries no echo
    entity, so this is 1 there, and a multi-echo run is known from the raw listing instead:
    several :class:`RawRun` of one ``key`` with distinct ``entities.echo``
    (``func.session.plan_session``)."""


@dataclass(frozen=True)
class ConnectorMatch:
    """What ``Connector.detect`` says about a folder that is a derivative of its pipeline."""

    connector: str
    pipeline_version: str | None
    generation: Generation


@dataclass(frozen=True)
class Display:
    """Which preprocessed series of a run is its display grid (``Connector.display``)."""

    layout: DisplayLayout | None
    """None: the run has no display grid, and ``reason`` says why."""
    space: str | None = None
    """The space entity of the series, ``""`` for the run's own frame."""
    reason: str | None = None
    """One of :data:`DISPLAY_REASONS` when ``layout`` is None."""


@dataclass(frozen=True)
class RigidPath:
    """The stored transform of a run, read as a point map (``Connector.raw_to_anat``)."""

    matrix: Matrix4
    """From the anatomy's world to the scanner world of the run's reference image, RAS mm."""
    dof: int | None
    """The degrees of freedom of the registration when a file states them."""
    shear_scale: float
    """The largest deviation of a singular value of the 3x3 part from 1; 0 for a rigid one."""
    rigid: bool
    """``shear_scale`` within 1e-3. A transform that is not rigid is used as stored."""
    source: FuncRole
    reference: FuncRole | None
    """The role of the image that arbitrates the raw affine, or None when the run has none."""


@dataclass(frozen=True)
class Motion:
    """The pipeline's motion estimates of a run (``Connector.motion``)."""

    fd: tuple[float | None, ...]
    """Framewise displacement per volume in mm; None where the table has no number."""
    dvars: tuple[float | None, ...]
    """Standardised DVARS per volume."""
    non_steady: tuple[int, ...]
    """Indices of the volumes marked as non-steady-state, ascending."""
    source: FuncRole


@dataclass(frozen=True)
class Statement:
    """One statement about a correction, with where it was read."""

    state: CorrectionState
    words: str | None
    """The words read, tags removed (``FLB ("fieldmap-less", SyN-based)``, ``Skipped (too few
    volumes)``); None when nothing was read."""
    source: FuncRole | None
    """The role of the file the words were read from; None for a statement by the pipeline's
    design (head-motion correction in fMRIPrep) or when nothing was read."""
    note: str | None = None
    """One sentence for the page, for example why a state is ``unknown``."""


@dataclass(frozen=True)
class Corrections:
    """What the pipeline did to the preprocessed series of a run (``Connector.corrections``)."""

    motion: Statement
    sdc: Statement
    sdc_method: str | None
    slice_timing: Statement
    registration: Statement
    """``applied`` with the report's words (``FreeSurfer <code>bbregister</code> ...`` with
    the tags removed), or ``unknown``."""
    dof: int | None
    """Degrees of freedom as the registration line states them."""
    sdc_figure_listed: bool | None
    """Whether the listing has the distortion figure of the run, kept as a cross-check."""


@dataclass(frozen=True)
class RawAffine:
    """The affine the raw stream of a run used (``session.raw_affine_choice``)."""

    matrix: Matrix4
    choice: AffineChoice
    reference_image: bool
    """Whether a reference image of the run was there at all."""


@dataclass(frozen=True)
class AnchorCandidate:
    """A run that can be the anchor of its session: one with a stored transform."""

    run: str
    """The id of the run (a ``run_uid`` from step 2)."""
    order: tuple[str | int, ...]
    """``func.bids.name_order`` of the run."""
    acquisition_time: str | None
    """``AcquisitionTime`` of the raw sidecar as written, None when it has none."""


@dataclass(frozen=True)
class Anchor:
    run: str
    rule: AnchorRule
    why: str
    """One line: what the rule saw. Under the time rule it holds an acquisition time, so it is
    shown on the local machine only and is not written into a pack."""


@dataclass(frozen=True)
class FrameDisplacement:
    """How far a run, drawn through the anchor's transform, lies from where its own puts it."""

    mean_mm: float
    max_mm: float
    rotation_deg: float
    translation_mm: float
    translation_ras: tuple[float, float, float]
    shear_scale: float
    n_voxels: int
    flagged: bool


@dataclass(frozen=True)
class FrameCheck:
    """The frame check of one session (``session.check_frame``)."""

    anchor: str
    limit_mm: float
    runs: Mapping[str, FrameDisplacement]
    """Run id -> its displacement; the anchor's entry is all zeros."""
    flagged: tuple[str, ...]
    shared_frame_plausible: bool
    anchor_is_outlier: bool


@dataclass(frozen=True)
class Grid:
    """A display grid: the shape and the voxel-to-world affine of a preprocessed series."""

    shape: tuple[int, int, int]
    affine: Matrix4

    @property
    def key(self) -> str:
        """12 hex characters over shape and affine: the name of the grid's contour folder.

        The SHA-256 of the canonical JSON of ``{"affine": rows, "shape": list}`` with every
        number written by ``repr`` (``json.dumps``), so equal grids give equal keys on every
        machine. Two grids whose affines differ in the last bit get two keys; whether two runs
        lie on one grid is decided with a tolerance elsewhere (``geometry.same_grid``).
        """
        text = json.dumps(
            {"affine": [list(row) for row in self.affine], "shape": list(self.shape)},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(text.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class PlaneContours:
    """The contours of one plane of one grid, as ``func.contours`` makes them."""

    plane: str
    """``sagittal``, ``coronal`` or ``axial``: the slice axis is voxel axis 0, 1 or 2."""
    data: bytes
    """The file of the plane, uncompressed (the layout of stage 1's ``pack_contours.py``)."""
    slices: int
    lines: int
    points: int


@dataclass(frozen=True)
class SessionRun:
    """A run of a session as the session stage sees it: paired, ordered, packable or not."""

    run: str
    """The id of the run."""
    name: str
    """The BIDS name without the subject (``ses-1_task-rest_run-1``), for a local run's pages."""
    deriv: DerivRun
    raw: RawRun | None
    display: Display
    reason: str | None
    """One of :data:`PACK_REASONS` when the run cannot be packed, else None."""
    missing: tuple[FuncRole, ...] = ()
    """Roles of :data:`~surfcensus.sources.base.PACK_REQUIRED` that the run lacks."""


@dataclass(frozen=True)
class SessionPlan:
    """One session: its runs in acquisition order, its anchor, frame check and grids."""

    participant: str
    """``sub-01``."""
    session: str | None
    """The ``ses-`` label without ``ses-``; None in a dataset without sessions."""
    fs_subject: str | None
    """The name of the FreeSurfer subject in the subjects directory, or None: then the
    session is packed without contours."""
    runs: tuple[SessionRun, ...]
    """Every run of the session, those that cannot be packed included: the packable ones in
    acquisition order, which is the order of the session's stream (decision D38; the anchor
    is the first acquired by its rule, so it comes first without being moved there); runs
    that cannot be packed follow in name order."""
    anchor: Anchor | None
    """None when no run of the session can be packed."""
    frame: FrameCheck | None
    grids: Mapping[str, Grid]
    """``Grid.key`` -> grid, for the packable runs."""
    grid_of: Mapping[str, str]
    """Run id -> the key of its grid, for the packable runs."""
    files: Mapping[FuncRole, FuncFileRef] = field(default_factory=dict)
    """The session's own files (``xfm_anat_fsnative``)."""
    details: Mapping[str, str] = field(default_factory=dict)
    """Run id -> one sentence that says more than the reason of a run that cannot be packed,
    for example which files are not fetched. Shown on the local machine only."""
    order: tuple[str, ...] = ()
    """Every run of the session (packable or not, selected or not) in its acquisition order by
    the anchor's rule (``session.session_order``). Empty when the plan has no anchor."""
    anchor_note: str | None = None
    """When the first run by the anchor's rule has no readable stored transform to the
    anatomy, so that a later run is the anchor: one sentence that says so, by run names.
    None otherwise."""


@dataclass(frozen=True)
class RunPackInput:
    """Everything ``func.pack.pack_run`` needs about one run, with local paths."""

    run: str
    name: str
    files: Mapping[FuncRole, Path]
    """Local paths of the roles the run has (a lease in step 2, the files in place in step 1)."""
    grid: Grid
    display_to_anat: Matrix4
    """``P_r`` of the display geometry: the identity in the anatomy-space layout."""
    anchor_run: str
    anchor_matrix: Matrix4
    """``S_a``: the anchor's :attr:`RigidPath.matrix`."""
    raw_affine: RawAffine
    motion: Motion | None
    corrections: Corrections
    rigid: RigidPath
    """The run's own stored transform."""
    tr_s: float | None
    session_inputs: tuple[tuple[str, Literal["raw", "deriv", "freesurfer"], Path], ...] = ()
    """``(role, base, path)`` of files outside the run that its pack depends on, listed in
    ``pack.json`` after the run's own files: the anchor's stored transform
    (``anchor_xfm_to_anat``, for a run that is not the anchor), the session's
    ``xfm_anat_fsnative`` and the FreeSurfer files of the contours (``t1``, ``lh_white``,
    ``rh_white``, ``lh_pial``, ``rh_pial``). So a changed anchor or surface changes the pack's
    inputs and its content digest."""


# --- step 2 of the EPI interface: the run directory (docs/epi_interface.md, "Step 2: the plan")
#
# Laid out by work package E2.0 before the catalog (E2.A), the functional inventory (E2.B), the
# pack phase (E2.C) and the app in a run (E2.D) are written. The units are the participant
# (the page: one continuous stream over all its sessions, decision D38), the session (the unit
# of the one registration, decision D36, and of the pack phase) and the run (a row of
# ``packs``).

EPI_SESSION_PHASE = "epi_session"
"""The phase of the ``work`` ledger whose unit is one functional session: its registration,
frame check, contours, the packs of its runs and its ``session.json``. Its ``config_hash`` is
``EpiConfig.session_hash()``; the runs have rows of their own in ``packs``."""

FUNC_INVENTORY_PHASE = "func_inventory"
"""The name of phase 2f in a :class:`PhaseReport` and in ``config.STAGE_SCHEMAS``. It has no
ledger rows: its skip test is ``func_sessions.inventory_hash`` (decision D21), as the surface
inventory's is ``subjects.inventory_hash``."""

PACK_PHASE = "pack"
"""The name of the pack phase in a :class:`PhaseReport` (``surfcensus pack``)."""


def session_unit(session_uid: str) -> str:
    """The unit of a session in the ``work`` ledger: ``session:<session_uid>``."""
    return f"session:{session_uid}"


RawFound = Literal["given", "dataset root", "sourcedata/raw"]
"""How the raw dataset of a functional source was found (docs/epi_interface.md, "Which folders
are found by rule"): ``--raw DIR``; the BIDS dataset whose ``derivatives/`` holds the
derivative; ``sourcedata/raw/`` inside the derivative."""

DerivFound = Literal["surface layout", "folder itself", "derivatives folder"]
"""How the derivative was found (``func.discover.find_folders``): the fMRIPrep output that holds
the FreeSurfer subjects the surface source found; the folder that was named; the first folder
of ``derivatives/`` that a connector recognises."""


def func_signature(
    refs: Iterable[FuncFileRef],
    freesurfer: Mapping[FileRole, FileRef] | None = None,
    extra: Iterable[str] = (),
) -> str:
    """What a functional skip test compares: 16 hex characters that change when a listed file
    does (decision D21), the counterpart of ``sources.base.input_signature``.

    The SHA-256 of one line per functional file in the order given
    (``f<TAB>base<TAB>role<TAB>location<TAB>size<TAB>etag_or_mtime``, with ``<TAB>variant`` for a
    file that has one), then one line per FreeSurfer file in ``FileRole`` order
    (``fs<TAB>role<TAB>location<TAB>size<TAB>etag_or_mtime[<TAB>variant]``), then one line per
    ``extra`` string. Built from the listing alone, so deciding to skip opens no source file.
    Private, like every input signature: it never leaves ``catalog.sqlite``. The caller puts
    the refs in a fixed order (``FuncRole`` order within a run, runs in session order); the
    function does not sort them, so that the anchor's transform can be told from the run's own.
    """
    lines = []
    for ref in refs:
        fields = ["f", ref.base, ref.role.value, ref.location, str(ref.size), ref.etag_or_mtime]
        if ref.variant is not None:
            fields.append(ref.variant)
        lines.append("\t".join(fields))
    fs = freesurfer or {}
    for role in FileRole:
        if role in fs:
            item = fs[role]
            fields = ["fs", role.value, item.location, str(item.size), item.etag_or_mtime]
            if item.variant is not None:
                fields.append(item.variant)
            lines.append("\t".join(fields))
    lines.extend(extra)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class FuncSourceRecord:
    """The functional part of a dataset of a run: the derivative, the raw dataset, the
    connector (a row of ``func_sources``). Private: roots and identity never leave the catalog
    but as local paths inside a lease."""

    connector: str
    match: ConnectorMatch
    deriv_root: str
    """Absolute path of the derivative as found now."""
    deriv_identity: str
    """What is hashed into every ``participant_uid``, ``session_uid`` and ``run_uid``: the
    resolved derivative root when the functional source was first stored, kept across
    ``--relocate`` as a local identity is (decision D21). The raw root does not enter it."""
    raw_root: str | None
    """Absolute path of the raw dataset; None when none was found (runs: ``no raw series``)."""
    deriv_found: DerivFound
    raw_found: RawFound | None


@dataclass(frozen=True)
class FuncLeaseFacts:
    """What the private side of a run knows of one session when the pack phase leases it
    (``Catalog.func_lease_facts``): enough to list it again and plan it, by
    ``func.inventory.session_work``. Private."""

    source: FuncSourceRecord
    participant_uid: str
    session_uid: str
    participant: str
    session: str | None
    run_ids: Mapping[str, str]
    """Run key (``func.bids.run_key``) -> ``run_uid``, for every run of the session."""
    fs_subjects_dir: str | None
    fs_subject: str | None
    """The FreeSurfer subject directory's name, or None (the session has no contours)."""
    subject_uid: str | None
    inventory_hash: str
    """``func_sessions.inventory_hash`` as stored; the lease compares the listing with it and
    raises ``SourceError("changed", ...)`` when they differ."""


@dataclass(frozen=True)
class ContourJob:
    """What a pack worker needs to cut the contours of a session: the FreeSurfer subject
    directory (a local path while the lease is open) and the point map from FreeSurfer's
    scanner RAS to the anatomy's world (``Connector.anat_to_fsnative``)."""

    fs_dir: Path
    fs_subject: str
    anat_to_fsnative: Matrix4


@dataclass(frozen=True)
class SessionWork:
    """What the lease of a session hands a pack worker (``RunContext.func_lease``): the plan
    made again from the leased files (the same listing as the inventory's, which the lease
    checks), the inputs of every packable run, and the contour job. Local paths are valid
    while the lease is open."""

    session_uid: str
    participant_uid: str
    plan: SessionPlan
    """Run ids are ``run_uid``s."""
    inputs: tuple[RunPackInput, ...]
    """For the runs of ``plan.runs`` without a reason whose inputs could be read."""
    input_failures: Mapping[str, str]
    """``run_uid`` -> the detail of a packable run whose inputs could not be read; it is
    recorded as ``pack failure``."""
    contours: ContourJob | None
    contour_note: str
    """What the command says of the contours: ``contours from sub-01`` or why there are none."""


@dataclass(frozen=True)
class SessionResult:
    """What a pack worker returns for one session (``func.batch.pack_session_unit``)."""

    session_uid: str
    info: SessionInfo
    """The ``session.json`` that was written."""
    packs: Mapping[str, PackInfo]
    """``run_uid`` -> the ``pack.json`` of every run that is packed now, kept or made."""
    kept: tuple[str, ...]
    """Runs whose pack was taken from disk because its ``packs`` row was current."""
    failures: Mapping[str, tuple[str, str | None]]
    """``run_uid`` -> reason and detail of a packable run whose pack failed (``pack failure``,
    ``source changed``)."""
    seconds: float


@dataclass(frozen=True)
class PackRow:
    """A row of the ``packs`` table: the pack of one packable run, done or failed. A run that
    cannot be packed has no row: its reason is ``func_runs.reason``."""

    run_uid: str
    session_uid: str
    config_hash: str
    """``EpiConfig.pack_hash()``."""
    input_signature: str
    """Private: ``PublicCatalog.pack_signature`` of the run when it was packed."""
    status: Status
    pack_key: str | None = None
    """``privacy.pack_key``; None for a failed pack."""
    reason: str | None = None
    detail: str | None = None
    """Private in a census run; scrubbed."""
    path: str | None = None
    """``epi/<participant_uid>/<session_uid>/<run_uid>``, relative to the run directory."""
    n_volumes: int | None = None
    bytes: int | None = None
    """The compressed size of the pack's files."""
    content_digest: str | None = None
    """``PackInfo.content_digest``. Private: it covers the SHA-256 of the source files."""
    anchor_run_uid: str | None = None
    grid_key: str | None = None
    packed_at: str | None = None
    """UTC, ISO 8601. None in a row to be recorded: the catalog writes the current time."""


@dataclass(frozen=True)
class FuncRunRow:
    """A functional run as phase 2f stores it (``Catalog.upsert_func_session``): what the
    listing, the sidecars, the reports, the transforms and the headers say, and the place
    the session's plan gives it. Private as a whole (``files``, ``entities``); the public view
    ``public_func_runs`` leaves those out."""

    run_uid: str
    session_uid: str
    run_key: str
    name: str
    """The BIDS name without the subject (``ses-1_task-rest_run-1``)."""
    entities: RunEntities
    session_order: int | None
    """``SessionRunInfo.session_order``: the place among all runs of the session by the
    anchor's rule; None when the session has no anchor."""
    acquisition_order: int
    """``SessionRunInfo.acquisition_order``: the place in the session's list (the packable
    runs in acquisition order, then the others)."""
    anchor: bool
    selected: bool
    """False: ``--epi-runs`` leaves it out (reason ``not selected``)."""
    packable: bool
    """No reason from the listing and the headers: the pack phase makes its pack."""
    reason: str | None
    detail: str | None
    missing_roles: tuple[FuncRole, ...]
    n_volumes: int | None
    tr_s: float | None
    echoes: int
    display: Display
    spaces: tuple[str, ...]
    grid: Grid | None
    """The display grid of a packable run; None otherwise."""
    raw_affine: AffineChoice | None
    reference_image: bool | None
    rigid: bool | None
    dof: int | None
    corrections: Corrections | None
    frame: FrameDisplacement | None
    files: tuple[FuncFileRef, ...]
    """Every listed file of the run (raw and derivative), in ``FuncRole`` order."""


@dataclass(frozen=True)
class FuncSessionRow:
    """A functional session as phase 2f stores it, with its runs, in one transaction."""

    session_uid: str
    participant_uid: str
    session: str | None
    stream_order: int
    """The place of the session in its participant's stream (``func.api.session_key``)."""
    subject_uid: str | None
    """The FreeSurfer subject of the session in the surface catalog, or None."""
    fs_subject: str | None
    anchor: Anchor | None
    anchor_note: str | None
    frame: FrameCheck | None
    n_grids: int
    problem: str | None
    """``not fetched: ...`` for a session with placeholders, ``unreadable: ...`` for one whose
    plan could not be made (then every run is ``pack failure``); None otherwise."""
    inventory_hash: str
    """:func:`func_signature` over the files of every run and of the session, the FreeSurfer
    files of the contours, and ``EpiConfig.inventory_hash()``, the FreeSurfer subject's name
    and the run ids as extra lines."""
    files: tuple[FuncFileRef, ...]
    """The session's own files (``xfm_anat_fsnative``)."""
    runs: tuple[FuncRunRow, ...]
