"""A synthetic local folder with all three kinds of data of the EPI interface, with every
transform and every value known in closed form (docs/epi_interface.md, Testing).

:func:`write_func_dataset` writes

* a raw BIDS dataset: ``sub-*/[ses-*/]func/*_bold.nii.gz`` with JSON sidecars, one of them
  inherited from the top level (``task-nback_bold.json``);
* an fMRIPrep derivative in the anatomy-space layout (``space-T1w``), in either generation of
  file names of the spec's connector table: ``scanner-T1w`` (fMRIPrep 20.2.x to 23.0.2,
  written as 21.0.2) or ``boldref-T1w`` (23.2 and later, written as 25.2.3); with brain
  masks, reference images, ITK text transforms with a centre, confounds tables, summary and
  validation reports in ``sub-*/figures/``, and ``from-T1w_to-fsnative`` in ``sub-*/anat/``;
* the FreeSurfer subjects of that derivative in ``sourcedata/freesurfer/`` (the subject of
  ``tests/synth.py``, ellipsoids, at 2 mm and 642 vertices per surface).

With ``raw_inside`` (the default) the derivative is ``<root>/ds/derivatives/fmriprep`` of the
raw dataset ``<root>/ds``, the layout of a BIDS dataset root; without it the two lie side by
side, ``<root>/raw`` and ``<root>/fmriprep``, the case of ``--raw DIR``.

The closed forms. Every point is in RAS mm. ``Q`` (:data:`FSNATIVE_TO_T1W`) takes FreeSurfer's
scanner RAS (tkr RAS plus ``c_ras``) to the anatomy's world (fMRIPrep's T1w space); it is
the point map of ``from-T1w_to-fsnative``. ``S_r`` (``SyntheticRun.anat_to_scanner``) takes
the anatomy's world to the scanner world of run ``r``; it is the point map of the run's
``from-scanner_to-T1w`` (or ``from-boldref_to-T1w``) file, and ``S_r = B_s . D_r`` with ``B_s``
the scanner frame of the session and ``D_r`` the run's known displacement in the anatomy, so
the frame displacement of run ``r`` against the anchor ``a`` is ``inv(D_a) . D_r``. ``R_r``
(``SyntheticRun.raw_affine``) is the true voxel-to-scanner affine of the raw series, oblique by
6 degrees about x. The signal is the field :func:`series_value`, ``f(p, t) = A_t + b_t . p``,
linear in the anatomical position ``p`` and changing with the volume ``t``, so that

* the raw series holds ``f(inv(S_r) . R_r . u, t)`` at raw voxel ``u``;
* the preprocessed series holds ``f(G . v, t)`` at display voxel ``v`` (``G``: the display
  grid), as if the pipeline had placed it perfectly;
* the raw stream of run ``r`` through the anchor's transform, sampled by a cubic B-spline
  with the edge values repeated, equals ``f(inv(S_r) . S_a . G . v, t)`` to within
  :data:`SPLINE_TOLERANCE` wherever the sample lies :data:`SPLINE_MARGIN` raw voxels inside the
  raw field of view, and for the anchor itself equals the preprocessed series there
  (:func:`expected_raw_stream`, :func:`spline_interior`). A spline reproduces a linear field
  exactly; the edge rule does not, and its error falls by a factor of about 0.27 per voxel
  from the edge (measured on this fixture: at most 0.59 at the outermost voxel centres, 0.17
  one voxel in, 0.014 three voxels in, for values of about 600).

The design (:data:`DESIGN`): ``sub-01`` with two sessions and ``sub-02`` without sessions.

* ``sub-01 ses-1``, anchor by ``AcquisitionTime`` (written without leading zeros), which is not
  the first run in name order: four runs, one displaced by 3 mm, one rotated, one by 12 mm
  (flagged); the H06 case of stage 1 (qform and sform differ, the reference image has the
  qform), its mirror (the reference has the sform), and a run without a reference image.
* ``sub-01 ses-2``: an anchor that is the outlier, 14 mm from two runs that lie 1 mm apart.
* ``sub-02``: the name-order rule (a run without ``AcquisitionTime``, whose raw name is
  ``run-02`` where the derivative has ``run-2``), a 9-dof registration that is not rigid, a
  run without a summary report, a run on another grid (packed on its own grid, a group of its
  own with its own contours: decision D38 builds no grid common to the runs), and one run that
  cannot be packed for each reason the listing and the headers can show: no raw series, no
  brain mask, no transform to the anatomy, volume counts that differ, multi-echo, standard
  space only, native space only, and no preprocessed series (fMRIPrep's level ``minimal``).

``annex`` turns files of ``sub-01 ses-2 task-rest run-2`` into git-annex placeholders
(``tests/synth_annex.py``): with ``"link"`` its raw series, its preprocessed series and its
confounds table, with ``"pointer"`` its summary report and its confounds table, the small
files a connector reads. Both folders are then repositories (``make_repository``).

What is not fMRIPrep's own: the markup around the report lines follows the ``<li>`` list of
fMRIPrep's ``FunctionalSummary`` as stage 1's reader expects it, written here and not copied
from a report; the validation report holds a sentence that no code reads; the head-motion
transforms of the second generation are identities. The series are float32, so that the
closed forms hold to float32 precision; real series are integers with a scale.

The module uses nibabel and numpy only (and ``synth``, ``synth_annex``), never the package.
"""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import nibabel as nib
import numpy as np
from synth import SyntheticSubject, write_synthetic_subject  # pyrefly: ignore[missing-import]
from synth_annex import Kind, Placeholder, make_repository, unfetch  # pyrefly: ignore[missing-import]

Generation = Literal["scanner-T1w", "boldref-T1w"]
GENERATIONS: tuple[Generation, Generation] = ("scanner-T1w", "boldref-T1w")
VERSIONS: Mapping[Generation, str] = {"scanner-T1w": "21.0.2", "boldref-T1w": "25.2.3"}

N_VOLUMES = 12
TR_S = 2.0
VOXEL_MM = 8.0
BRAIN_CENTRE = (0.0, -18.0, 16.0)
"""The middle of the synthetic brain in the anatomy's world; the display grid is centred on
it and the rigid transforms turn about it."""
GRID_SHAPE = (17, 18, 14)
"""The display grid: 4,284 voxels of 8 mm, axes along R, A, S."""
GRID_B_SHAPE = (17, 18, 15)
"""The grid of the run on another grid: one slice more, same affine."""
RAW_SHAPE = (20, 21, 17)
RAW_TILT_DEG = 6.0
FS_VOXEL_MM = 2.0
FS_SHAPE = (96, 96, 96)
FS_SUBDIVISIONS = 3
LPS = np.diag([-1.0, -1.0, 1.0, 1.0])
SPLINE_MARGIN = 4
"""Raw voxels from the edge of the raw field of view (in voxel coordinates, from 0 and from
``n - 1``) beyond which a cubic-spline sample of a series equals the closed form."""
SPLINE_TOLERANCE = 0.025
"""How close it is there, for values of about 600: a fortieth of a gray level of an 8-bit
window of 1,000."""
MNI = "MNI152NLin2009cAsym"

FIELD_A = (600.0, 7.0)
"""``A_t = 600 + 7 t``."""
FIELD_B = ((1.5, -0.9, 2.1), (0.05, 0.02, -0.03))
"""``b_t = (1.5, -0.9, 2.1) + t (0.05, 0.02, -0.03)`` per mm."""

REGISTRATION = "FreeSurfer <code>bbregister</code> (boundary-based registration, BBR) - {dof} dof"
REPORT_TEMPLATE = (
    '<section id="summary">\n'
    "\t\t<details open>\n"
    "\t\t<summary>Summary</summary>\n"
    '\t\t<ul class="elem-desc">\n'
    "\t\t\t<li>Original orientation: LAS</li>\n"
    "\t\t\t<li>Repetition time (TR): 2s</li>\n"
    "\t\t\t<li>Phase-encoding (PE) direction: Anterior-Posterior</li>\n"
    "\t\t\t<li>{echo}</li>\n"
    "\t\t\t<li>Slice timing correction: {stc}</li>\n"
    "\t\t\t<li>Susceptibility distortion correction: {sdc}</li>\n"
    "\t\t\t<li>Registration: {registration}</li>\n"
    "\t\t\t<li>Non-steady-state volumes: 1</li>\n"
    "\t\t</ul>\n"
    "\t\t</details>\n"
    "</section>\n"
)
"""The summary report of a run, with the three lines a connector reads."""
FLB = 'FLB ("fieldmap-less", SyN-based)'


# --- geometry -----------------------------------------------------------------------------


def rotation(angles_deg: Sequence[float]) -> np.ndarray:
    """3x3 rotation by angles about x, y and z, in that order, in degrees."""
    ax, ay, az = np.radians(np.asarray(angles_deg, float))
    rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx


def rigid(
    angles_deg: Sequence[float] = (0.0, 0.0, 0.0),
    shift: Sequence[float] = (0.0, 0.0, 0.0),
    centre: Sequence[float] = BRAIN_CENTRE,
    scale: float = 1.0,
) -> np.ndarray:
    """4x4 point map: rotate (and scale) about ``centre``, then shift."""
    m = np.eye(4)
    m[:3, :3] = rotation(angles_deg) * scale
    c = np.asarray(centre, float)
    m[:3, 3] = c - m[:3, :3] @ c + np.asarray(shift, float)
    return m


def apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """``matrix`` applied to points ``(n, 3)``."""
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def voxel_centres(shape: Sequence[int]) -> np.ndarray:
    """Every voxel index of a grid as ``(n, 3)`` float, in Fortran (NIfTI) order."""
    i, j, k = np.meshgrid(*(np.arange(n, dtype=float) for n in shape), indexing="ij")
    return np.column_stack([a.ravel(order="F") for a in (i, j, k)])


def _grid_affine() -> np.ndarray:
    affine = np.diag([VOXEL_MM, VOXEL_MM, VOXEL_MM, 1.0])
    affine[:3, 3] = np.asarray(BRAIN_CENTRE) - VOXEL_MM * (np.asarray(GRID_SHAPE) - 1) / 2
    return affine


GRID_AFFINE = _grid_affine()
"""``G``: RAS axes of 8 mm, centred on :data:`BRAIN_CENTRE`. Shared by both display grids,
so that the second differs in shape only."""

FSNATIVE_TO_T1W = rigid((0.6, -0.4, 0.8), (0.7, -0.5, 0.3), centre=(0.0, 0.0, 0.0))
"""``Q``: FreeSurfer's scanner RAS to the anatomy's world."""

SESSION_FRAMES: Mapping[tuple[str, str | None], np.ndarray] = {
    ("sub-01", "1"): rigid((5.0, -3.0, 8.0), (4.0, -6.0, 10.0)),
    ("sub-01", "2"): rigid((-4.0, 6.0, -3.0), (-8.0, 5.0, -2.0)),
    ("sub-02", None): rigid((2.0, 4.0, -6.0), (6.0, 3.0, -9.0)),
}
"""``B_s``: the anatomy's world to the scanner world of the session."""


def series_value(points: np.ndarray, t: int) -> np.ndarray:
    """``f(p, t)``: the signal at anatomical points ``(n, 3)`` in volume ``t``, float64."""
    a = FIELD_A[0] + FIELD_A[1] * t
    b = np.asarray(FIELD_B[0]) + t * np.asarray(FIELD_B[1])
    return a + points @ b


def float32_affine(matrix: np.ndarray) -> np.ndarray:
    """The affine as a NIfTI sform stores it (float32), so that it reads back exactly."""
    return np.asarray(matrix, np.float32).astype(np.float64)


def raw_grid(anat_to_scanner: np.ndarray) -> np.ndarray:
    """The true affine of a raw series: 8 mm voxels tilted by 6 degrees about x, its middle
    voxel at the scanner point of :data:`BRAIN_CENTRE`."""
    affine = np.eye(4)
    affine[:3, :3] = rotation((RAW_TILT_DEG, 0.0, 0.0)) * VOXEL_MM
    middle = (np.asarray(RAW_SHAPE, float) - 1) / 2
    centre = apply(anat_to_scanner, np.asarray([BRAIN_CENTRE]))[0]
    affine[:3, 3] = centre - affine[:3, :3] @ middle
    return float32_affine(affine)


def itk_text(matrix: np.ndarray, centre: Sequence[float], kind: str) -> str:
    """An ITK text transform whose RAS point map is ``matrix``, written with ``centre`` (RAS)
    as its FixedParameters: ``y = M (x - c) + t + c`` in LPS."""
    lps = LPS @ matrix @ LPS
    c = (LPS @ np.array([*centre, 1.0]))[:3]
    m = lps[:3, :3]
    t = lps[:3, 3] - c + m @ c
    numbers = " ".join(repr(float(x)) for x in [*m.ravel(), *t])
    fixed = " ".join(repr(float(x)) for x in c)
    return (
        "#Insight Transform File V1.0\n#Transform 0\n"
        f"Transform: {kind}_double_3_3\nParameters: {numbers}\nFixedParameters: {fixed}\n"
    )


def power_fd(params: np.ndarray) -> np.ndarray:
    """Power's framewise displacement from six parameters per volume (translations in mm,
    rotations in radians, on a sphere of 50 mm); NaN for volume 0."""
    d = np.abs(np.diff(params, axis=0))
    return np.concatenate([[np.nan], d[:, :3].sum(axis=1) + 50.0 * d[:, 3:].sum(axis=1)])


# --- the design ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunDesign:
    participant: str
    session: str | None
    task: str
    run: str
    """As the derivative writes it."""
    case: str
    reason: str | None = None
    """The expected reason the run is not packed; None for a packable run."""
    raw_run: str | None = None
    """The run label of the raw name when it differs (``02`` for ``run-2``)."""
    time: str | None = None
    displacement: tuple[tuple[float, ...], tuple[float, ...], float] = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        1.0,
    )
    """``D_r`` as (angles in degrees, shift in mm, scale), about :data:`BRAIN_CENTRE`."""
    header: Literal["agree", "qform true", "sform true"] = "agree"
    reference: bool = True
    report: bool = True
    stc: Mapping[Generation, str] = field(
        default_factory=lambda: {"scanner-T1w": "Not applied", "boldref-T1w": "Not applied"}
    )
    sdc: str = "None"
    dof: int = 6
    native: bool = False
    """Also a preprocessed series in the run's own frame (layout 2)."""


def _stc(old: str, new: str) -> Mapping[Generation, str]:
    return {"scanner-T1w": old, "boldref-T1w": new}


DESIGN: tuple[RunDesign, ...] = (
    # sub-01 ses-1: the time rule; the anchor is task-rest run-2, not the first in name order
    RunDesign(
        "sub-01",
        "1",
        "motor",
        "1",
        "displaced 3 mm",
        time="9:20:1.25",
        displacement=((0.0, 0.0, 0.0), (3.0, 0.0, 0.0), 1.0),
        stc=_stc("Applied", "Applied"),
    ),
    RunDesign(
        "sub-01",
        "1",
        "rest",
        "1",
        "rotated, qform true (H06)",
        time="9:12:40.0",
        displacement=((2.0, 0.0, 1.5), (0.0, -1.0, 0.5), 1.0),
        header="qform true",
        native=True,
    ),
    RunDesign(
        "sub-01", "1", "rest", "2", "anchor, sform true", time="9:5:3.5", header="sform true"
    ),
    RunDesign(
        "sub-01",
        "1",
        "rest",
        "3",
        "displaced 12 mm, no reference",
        time="9:31:10",
        displacement=((0.0, 0.0, 0.0), (0.0, 12.0, 0.0), 1.0),
        reference=False,
        sdc=FLB,
    ),
    # sub-01 ses-2: the anchor is the outlier
    RunDesign(
        "sub-01",
        "2",
        "rest",
        "1",
        "anchor, outlier",
        time="14:2:5",
        displacement=((0.0, 0.0, 0.0), (0.0, 0.0, 14.0), 1.0),
    ),
    RunDesign("sub-01", "2", "rest", "2", "placeholders", time="14:15:30"),
    RunDesign(
        "sub-01",
        "2",
        "rest",
        "3",
        "1 mm from run 2",
        time="14:30:0",
        displacement=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0),
    ),
    # sub-02, no sessions: the name-order rule
    RunDesign(
        "sub-02",
        None,
        "nback",
        "1",
        "anchor by name",
        time="10:0:0",
        stc=_stc("Not applied", "n/a"),
    ),
    RunDesign(
        "sub-02",
        None,
        "nback",
        "2",
        "no time, raw run-02, no report",
        raw_run="02",
        displacement=((0.0, 0.0, 0.0), (0.0, 2.0, 0.0), 1.0),
        report=False,
    ),
    RunDesign(
        "sub-02",
        None,
        "nback",
        "3",
        "9 dof, not rigid",
        time="10:20:0",
        displacement=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.004),
        dof=9,
        stc=_stc("Not applied", "Skipped (too few volumes)"),
    ),
    RunDesign("sub-02", None, "rest", "1", "no raw series", reason="no raw series"),
    RunDesign("sub-02", None, "rest", "2", "no brain mask", reason="no brain mask"),
    RunDesign("sub-02", None, "rest", "3", "no transform", reason="no transform to the anatomy"),
    RunDesign("sub-02", None, "rest", "4", "volume counts differ", reason="volume counts differ"),
    RunDesign("sub-02", None, "rest", "5", "other grid"),
    RunDesign("sub-02", None, "me", "1", "multi-echo", reason="multi-echo"),
    RunDesign("sub-02", None, "std", "1", "standard space only", reason="standard space only"),
    RunDesign("sub-02", None, "native", "1", "native space only", reason="native space only"),
    RunDesign("sub-02", None, "min", "1", "level minimal", reason="no preprocessed series"),
)
"""Every run of the dataset."""

PLACEHOLDER_RUN = ("sub-01", "2", "rest", "2")
PLACEHOLDER_ROLES: Mapping[Kind, tuple[str, ...]] = {
    "link": ("raw_bold", "pre_bold", "confounds"),
    "pointer": ("summary_report", "confounds"),
}


# --- what is written ----------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticRun:
    design: RunDesign
    stem: str
    """The derivative's stem: ``sub-01_ses-1_task-rest_run-1``."""
    raw_stem: str | None
    """The raw series' stem, None for the run without a raw series."""
    key: str
    """The run within its participant, the run index as a number: ``ses-1_task-rest_run-1``."""
    name: str
    """The BIDS name without the subject: ``ses-1_task-rest_run-1``."""
    n_raw_volumes: int | None
    n_pre_volumes: int | None
    displacement: np.ndarray = field(compare=False, repr=False)
    """``D_r``."""
    anat_to_scanner: np.ndarray = field(compare=False, repr=False)
    """``S_r = B_s . D_r``: the point map of the run's stored transform."""
    raw_affine: np.ndarray = field(compare=False, repr=False)
    """``R_r``, the affine the raw values were sampled with."""
    header_affine: np.ndarray = field(compare=False, repr=False)
    """The sform of the raw header (nibabel's ``img.affine``)."""
    header_qform: np.ndarray = field(compare=False, repr=False)
    affine_choice: str
    """What the arbiter rule gives: ``header``, ``qform`` or ``header, unchecked``."""
    motion: np.ndarray = field(compare=False, repr=False)
    """Six motion parameters per volume, as the confounds table has them."""
    files: Mapping[str, Path] = field(compare=False)
    """``FuncRole`` value -> the file written (also those turned into placeholders), and for a
    second file of a role: ``raw_bold_echo1`` and ``raw_bold_echo2`` (``raw_bold`` is the
    first), ``reference_hmc`` (the second generation's other reference image),
    ``pre_bold_native`` and ``pre_mask_native`` (layout 2 beside layout 1)."""

    @property
    def packable(self) -> bool:
        return self.design.reason is None


@dataclass(frozen=True)
class SyntheticSession:
    participant: str
    session: str | None
    anchor: str
    """The derivative stem of the anchor run."""
    rule: str
    """``acquisition_time`` or ``name_order``."""
    order: tuple[str, ...]
    """The stems of the packable runs in the session's acquisition order (decided over every
    run of the session; the anchor, the first run by the rule with a stored transform, need
    not be packable and then is not among them)."""
    flagged: tuple[str, ...]
    """Packable runs whose frame displacement exceeds 10 mm somewhere in the brain mask."""
    anchor_is_outlier: bool
    frame: np.ndarray = field(compare=False, repr=False)
    """``B_s``."""


@dataclass(frozen=True)
class SyntheticFuncDataset:
    root: Path
    raw_root: Path
    deriv_root: Path
    fs_subjects_dir: Path
    generation: Generation
    version: str
    runs: tuple[SyntheticRun, ...]
    sessions: tuple[SyntheticSession, ...]
    subjects: Mapping[str, SyntheticSubject]
    """FreeSurfer subject name -> what ``synth.write_synthetic_subject`` wrote."""
    placeholders: tuple[Placeholder, ...]
    fsnative_files: Mapping[str, Path]
    """Participant -> its ``from-T1w_to-fsnative`` file."""

    def run(self, participant: str, session: str | None, task: str, run: str) -> SyntheticRun:
        for item in self.runs:
            d = item.design
            if (d.participant, d.session, d.task, d.run) == (participant, session, task, run):
                return item
        raise KeyError((participant, session, task, run))

    def session(self, participant: str, session: str | None) -> SyntheticSession:
        for item in self.sessions:
            if (item.participant, item.session) == (participant, session):
                return item
        raise KeyError((participant, session))

    def runs_of(self, participant: str, session: str | None) -> tuple[SyntheticRun, ...]:
        return tuple(
            r
            for r in self.runs
            if (r.design.participant, r.design.session) == (participant, session)
        )


def subject_world(subject: SyntheticSubject, tkr: np.ndarray) -> np.ndarray:
    """Points in the tkr RAS of a FreeSurfer subject, in the anatomy's world:
    ``Q . (tkr + c_ras)``."""
    return apply(FSNATIVE_TO_T1W, np.asarray(tkr, float) + np.asarray(subject.c_ras))


def brain_mask(subject: SyntheticSubject, shape: Sequence[int], affine: np.ndarray) -> np.ndarray:
    """The voxels of a grid whose centre lies inside either pial ellipsoid of the subject."""
    world = apply(affine, voxel_centres(shape))
    tkr = apply(np.linalg.inv(FSNATIVE_TO_T1W), world) - np.asarray(subject.c_ras)
    inside = np.zeros(len(tkr), bool)
    for centre in (subject.lh_centre, subject.rh_centre):
        q = (tkr - np.asarray(centre)) / np.asarray(subject.pial_semi_axes)
        inside |= (q**2).sum(axis=1) <= 1.0
    return inside.reshape(tuple(shape), order="F")


def expected_raw_stream(
    run: SyntheticRun, anchor: SyntheticRun, voxels: np.ndarray, t: int
) -> np.ndarray:
    """The raw stream of ``run`` through ``anchor``'s transform at display voxels ``(n, 3)``:
    ``f(inv(S_r) . S_a . G . v, t)``. A cubic-spline sample equals it where
    :func:`spline_interior` of :func:`raw_sample_voxels` holds."""
    points = apply(
        np.linalg.inv(run.anat_to_scanner) @ anchor.anat_to_scanner @ GRID_AFFINE, voxels
    )
    return series_value(points, t)


def spline_interior(raw_voxels: np.ndarray) -> np.ndarray:
    """Which raw voxel coordinates ``(n, 3)`` lie :data:`SPLINE_MARGIN` voxels inside the raw
    field of view along every axis."""
    top = np.asarray(RAW_SHAPE, float) - 1 - SPLINE_MARGIN
    return ((raw_voxels >= SPLINE_MARGIN) & (raw_voxels <= top)).all(axis=1)


def raw_sample_voxels(run: SyntheticRun, anchor: SyntheticRun, voxels: np.ndarray) -> np.ndarray:
    """Where the raw stream samples the raw series of ``run`` for display voxels ``(n, 3)``:
    ``inv(R_r) . S_a . G . v``, in raw voxel coordinates."""
    return apply(np.linalg.inv(run.raw_affine) @ anchor.anat_to_scanner @ GRID_AFFINE, voxels)


def expected_pre(voxels: np.ndarray, t: int) -> np.ndarray:
    """The preprocessed series at display voxels: ``f(G . v, t)``."""
    return series_value(apply(GRID_AFFINE, voxels), t)


# --- writing ------------------------------------------------------------------------------


def _write_nifti(
    path: Path,
    data: np.ndarray,
    sform: np.ndarray,
    qform: np.ndarray | None = None,
    sform_code: int = 1,
) -> None:
    img = nib.Nifti1Image(data, sform)
    img.set_sform(sform, code=sform_code)
    img.set_qform(sform if qform is None else qform, code=1)
    if data.ndim == 4:
        img.header.set_zooms((*img.header.get_zooms()[:3], TR_S))
    img.header.set_xyzt_units("mm", "sec")
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, path)


def _series(points: np.ndarray, shape: Sequence[int], n: int) -> np.ndarray:
    volumes = [series_value(points, t).reshape(tuple(shape), order="F") for t in range(n)]
    return np.stack(volumes, axis=-1).astype(np.float32)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _motion(index: int, n: int) -> np.ndarray:
    """Made-up motion parameters of the run with this index: known numbers, nothing moves."""
    t = np.arange(n, dtype=float)
    phase = 0.7 * index
    trans = 0.05 * np.column_stack([np.sin(t + phase), np.cos(t + phase), np.sin(2 * t)])
    rot = 0.001 * np.column_stack([np.cos(t + phase), np.sin(2 * t + phase), np.cos(3 * t)])
    params = np.column_stack([trans, rot])
    return params - params[0]


def _confounds(params: np.ndarray) -> str:
    fd = power_fd(params)
    names = ["framewise_displacement", "std_dvars", "dvars"]
    names += ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    names += ["non_steady_state_outlier00"]
    rows = ["\t".join(names)]
    for t in range(len(params)):
        first = t == 0
        cells = [
            "n/a" if first else repr(float(fd[t])),
            "n/a" if first else repr(1.0 + 0.125 * t),
            "n/a" if first else repr(20.0 + 2.5 * t),
            *(repr(float(x)) for x in params[t]),
            "1" if first else "0",
        ]
        rows.append("\t".join(cells))
    return "\n".join(rows) + "\n"


_HMC = (
    "#Transform {n}\nTransform: AffineTransform_double_3_3\n"
    "Parameters: 1 0 0 0 1 0 0 0 1 0 0 0\nFixedParameters: 0 0 0\n"
)


def write_func_dataset(
    root: Path,
    *,
    generation: Generation = "scanner-T1w",
    raw_inside: bool = True,
    annex: Kind | None = None,
    n_volumes: int = N_VOLUMES,
) -> SyntheticFuncDataset:
    """Write the dataset of :data:`DESIGN` under ``root`` (which may exist and be empty)."""
    root = Path(root)
    raw_root = root / "ds" if raw_inside else root / "raw"
    deriv_root = raw_root / "derivatives" / "fmriprep" if raw_inside else root / "fmriprep"
    fs_dir = deriv_root / "sourcedata" / "freesurfer"
    version = VERSIONS[generation]
    old = generation == "scanner-T1w"

    _write_json(
        raw_root / "dataset_description.json",
        {"BIDSVersion": "1.8.0", "Name": "synthetic EPI interface dataset"},
    )
    _write_json(raw_root / "task-nback_bold.json", {"RepetitionTime": TR_S, "TaskName": "nback"})
    _write_json(
        deriv_root / "dataset_description.json",
        {
            "BIDSVersion": "1.4.0",
            "DatasetType": "derivative",
            "GeneratedBy": [{"Name": "fMRIPrep", "Version": version}],
            "Name": "fMRIPrep - fMRI PREProcessing workflow",
        },
    )

    first = write_synthetic_subject(
        fs_dir,
        name="sub-01",
        voxel_size=FS_VOXEL_MM,
        shape=FS_SHAPE,
        subdivisions=FS_SUBDIVISIONS,
    )
    shutil.copytree(first.path, fs_dir / "sub-02")
    subjects = {"sub-01": first, "sub-02": _moved(first, fs_dir / "sub-02")}

    fsnative_files = {}
    for participant in ("sub-01", "sub-02"):
        anat = deriv_root / participant / "anat"
        anat.mkdir(parents=True, exist_ok=True)
        name = f"{participant}_from-T1w_to-fsnative_mode-image_xfm.txt"
        (anat / name).write_text(itk_text(FSNATIVE_TO_T1W, (1.0, -2.0, 3.0), "AffineTransform"))
        inverse = f"{participant}_from-fsnative_to-T1w_mode-image_xfm.txt"
        text = itk_text(np.linalg.inv(FSNATIVE_TO_T1W), (1.0, -2.0, 3.0), "AffineTransform")
        (anat / inverse).write_text(text)
        fsnative_files[participant] = anat / name

    grid_points = apply(GRID_AFFINE, voxel_centres(GRID_SHAPE))
    grid_b_points = apply(GRID_AFFINE, voxel_centres(GRID_B_SHAPE))
    raw_voxels = voxel_centres(RAW_SHAPE)
    mask_a = brain_mask(first, GRID_SHAPE, GRID_AFFINE).astype(np.uint8)
    mask_b = brain_mask(first, GRID_B_SHAPE, GRID_AFFINE).astype(np.uint8)
    pre_a = _series(grid_points, GRID_SHAPE, n_volumes)
    pre_b = _series(grid_b_points, GRID_B_SHAPE, n_volumes)

    runs = []
    for index, d in enumerate(DESIGN):
        runs.append(
            _write_run(
                d,
                index,
                raw_root,
                deriv_root,
                old,
                n_volumes,
                raw_voxels,
                mask_a,
                mask_b,
                pre_a,
                pre_b,
            )
        )

    sessions = tuple(_session(runs, key) for key in SESSION_FRAMES)
    placeholders: list[Placeholder] = []
    if annex is not None:
        make_repository(raw_root)
        make_repository(deriv_root)
        target = next(r for r in runs if _ident(r.design) == PLACEHOLDER_RUN)
        for role in PLACEHOLDER_ROLES[annex]:
            path = target.files[role]
            repository = raw_root if role.startswith("raw_") else deriv_root
            placeholders.extend(unfetch(repository, [path], annex))
    return SyntheticFuncDataset(
        root=root,
        raw_root=raw_root,
        deriv_root=deriv_root,
        fs_subjects_dir=fs_dir,
        generation=generation,
        version=version,
        runs=tuple(runs),
        sessions=sessions,
        subjects=subjects,
        placeholders=tuple(placeholders),
        fsnative_files=fsnative_files,
    )


def _moved(subject: SyntheticSubject, path: Path) -> SyntheticSubject:
    values = {name: getattr(subject, name) for name in subject.__dataclass_fields__}
    return SyntheticSubject(**(values | {"path": path}))


def _ident(d: RunDesign) -> tuple[str, str | None, str, str]:
    return (d.participant, d.session, d.task, d.run)


def _stem(
    participant: str, session: str | None, task: str, run: str, echo: str | None = None
) -> str:
    parts = [participant] + ([f"ses-{session}"] if session else []) + [f"task-{task}", f"run-{run}"]
    return "_".join(parts + ([f"echo-{echo}"] if echo else []))


def _write_run(
    d: RunDesign,
    index: int,
    raw_root: Path,
    deriv_root: Path,
    old: bool,
    n: int,
    raw_voxels: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    pre_a: np.ndarray,
    pre_b: np.ndarray,
) -> SyntheticRun:
    session_dir = [f"ses-{d.session}"] if d.session else []
    raw_func = raw_root.joinpath(d.participant, *session_dir, "func")
    func = deriv_root.joinpath(d.participant, *session_dir, "func")
    figures = deriv_root / d.participant / "figures"
    func.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    stem = _stem(d.participant, d.session, d.task, d.run)
    key = "_".join([*stem.split("_")[1:-1], f"run-{int(d.run)}"])
    name = "_".join(stem.split("_")[1:])
    files: dict[str, Path] = {}

    angles, shift, scale = d.displacement
    displacement = rigid(angles, shift, scale=scale)
    frame = SESSION_FRAMES[(d.participant, d.session)]
    s_r = frame @ displacement
    true = raw_grid(s_r)
    raw_points = apply(np.linalg.inv(s_r) @ true, raw_voxels)
    raw = _series(raw_points, RAW_SHAPE, n)
    wrong = float32_affine(true @ rigid((0.0, 0.0, 5.0), (0.0, 20.0, 0.0), centre=(0, 0, 0)))
    sform, qform, sform_code = true, true, 1
    if d.header == "qform true":  # H06: fMRIPrep's reference image has the qform
        sform, qform, sform_code = wrong, true, 2
    elif d.header == "sform true":  # the mirror: the reference image has the sform
        sform, qform, sform_code = true, wrong, 2
    choice = (
        "header, unchecked"
        if not d.reference
        else "qform"
        if d.header == "qform true"
        else "header"
    )

    # the raw series
    n_raw: int | None = n
    raw_stem: str | None = None
    if d.reason == "no raw series":
        n_raw = None
    elif d.reason == "multi-echo":
        raw_stem = _stem(d.participant, d.session, d.task, d.run)
        for echo in ("1", "2"):
            echo_stem = _stem(d.participant, d.session, d.task, d.run, echo)
            path = raw_func / f"{echo_stem}_bold.nii.gz"
            _write_nifti(path, raw, sform, qform, sform_code)
            sidecar = {"EchoTime": 0.012 * int(echo), "RepetitionTime": TR_S}
            if d.time:
                sidecar["AcquisitionTime"] = d.time
            _write_json(raw_func / f"{echo_stem}_bold.json", sidecar)
            files[f"raw_bold_echo{echo}"] = path
        files["raw_bold"] = files["raw_bold_echo1"]
    else:
        raw_stem = _stem(d.participant, d.session, d.task, d.raw_run or d.run)
        files["raw_bold"] = raw_func / f"{raw_stem}_bold.nii.gz"
        _write_nifti(files["raw_bold"], raw, sform, qform, sform_code)
        if d.time is not None:
            files["raw_sidecar"] = raw_func / f"{raw_stem}_bold.json"
            _write_json(
                files["raw_sidecar"],
                {"AcquisitionTime": d.time, "RepetitionTime": TR_S, "TaskName": d.task},
            )

    # the reference images, the transform and the head-motion transforms
    reference = raw[..., 0]
    if d.reference:
        if old:
            files["reference"] = func / f"{stem}_boldref.nii.gz"
            _write_nifti(files["reference"], reference, true)
        else:
            files["reference"] = func / f"{stem}_desc-coreg_boldref.nii.gz"
            _write_nifti(files["reference"], reference, true)
            files["reference_hmc"] = func / f"{stem}_desc-hmc_boldref.nii.gz"
            _write_nifti(files["reference_hmc"], reference, true)
    if d.reason != "no transform to the anatomy":
        if old:
            files["xfm_to_anat"] = func / f"{stem}_from-scanner_to-T1w_mode-image_xfm.txt"
            inverse = func / f"{stem}_from-T1w_to-scanner_mode-image_xfm.txt"
        else:
            files["xfm_to_anat"] = (
                func / f"{stem}_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt"
            )
            inverse = func / f"{stem}_from-T1w_to-boldref_mode-image_desc-coreg_xfm.txt"
        kind = "MatrixOffsetTransformBase" if old else "AffineTransform"
        files["xfm_to_anat"].write_text(itk_text(s_r, BRAIN_CENTRE, kind))
        inverse.write_text(itk_text(np.linalg.inv(s_r), BRAIN_CENTRE, kind))
        if not old:
            hmc = func / f"{stem}_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt"
            hmc.write_text(
                "#Insight Transform File V1.0\n" + "".join(_HMC.format(n=t) for t in range(n))
            )

    # the preprocessed series
    n_pre: int | None = n
    params = _motion(index, n)
    if d.reason == "no preprocessed series":
        n_pre, params = None, params[:0]
    elif d.reason == "standard space only":
        files["pre_bold"] = func / f"{stem}_space-{MNI}_desc-preproc_bold.nii.gz"
        _write_nifti(files["pre_bold"], pre_a, GRID_AFFINE)
        files["pre_mask"] = func / f"{stem}_space-{MNI}_desc-brain_mask.nii.gz"
        _write_nifti(files["pre_mask"], mask_a, GRID_AFFINE)
    elif d.reason == "native space only":
        files["pre_bold"] = func / f"{stem}_desc-preproc_bold.nii.gz"
        _write_nifti(files["pre_bold"], raw, true)
        files["pre_mask"] = func / f"{stem}_desc-brain_mask.nii.gz"
        _write_nifti(files["pre_mask"], np.ones(RAW_SHAPE, np.uint8), true)
    else:
        series, mask = (pre_b, mask_b) if d.case == "other grid" else (pre_a, mask_a)
        if d.reason == "volume counts differ":
            n_pre, series, params = n - 2, series[..., : n - 2], params[: n - 2]
        files["pre_bold"] = func / f"{stem}_space-T1w_desc-preproc_bold.nii.gz"
        _write_nifti(files["pre_bold"], series, GRID_AFFINE)
        files["pre_sidecar"] = func / f"{stem}_space-T1w_desc-preproc_bold.json"
        _write_json(files["pre_sidecar"], {"RepetitionTime": TR_S, "SkullStripped": False})
        if d.reason != "no brain mask":
            files["pre_mask"] = func / f"{stem}_space-T1w_desc-brain_mask.nii.gz"
            _write_nifti(files["pre_mask"], mask, GRID_AFFINE)
        if d.native:
            files["pre_bold_native"] = func / f"{stem}_desc-preproc_bold.nii.gz"
            _write_nifti(files["pre_bold_native"], raw, true)
            files["pre_mask_native"] = func / f"{stem}_desc-brain_mask.nii.gz"
            _write_nifti(files["pre_mask_native"], np.ones(RAW_SHAPE, np.uint8), true)
    if n_pre is not None:
        files["confounds"] = func / f"{stem}_desc-confounds_timeseries.tsv"
        files["confounds"].write_text(_confounds(params))

    # the reports
    if d.report:
        files["summary_report"] = figures / f"{stem}_desc-summary_bold.html"
        files["summary_report"].write_text(
            REPORT_TEMPLATE.format(
                echo=(
                    "Multi-echo EPI sequence: 2 echoes."
                    if d.reason == "multi-echo"
                    else "Single-echo EPI sequence."
                ),
                stc=d.stc["scanner-T1w" if old else "boldref-T1w"],
                sdc=d.sdc,
                registration=REGISTRATION.format(dof=d.dof),
            )
        )
        files["validation_report"] = figures / f"{stem}_desc-validation_bold.html"
        files["validation_report"].write_text(
            '<p class="elem-desc">A synthetic validation report; no code reads this.</p>\n'
        )
    if d.sdc != "None":
        files["sdc_figure"] = figures / f"{stem}_desc-sdc_bold.svg"
        files["sdc_figure"].write_text('<svg xmlns="http://www.w3.org/2000/svg"/>\n')

    return SyntheticRun(
        design=d,
        stem=stem,
        raw_stem=raw_stem,
        key=key,
        name=name,
        n_raw_volumes=n_raw,
        n_pre_volumes=n_pre,
        displacement=displacement,
        anat_to_scanner=s_r,
        raw_affine=true,
        header_affine=sform,
        header_qform=qform,
        affine_choice=choice,
        motion=params,
        files=files,
    )


def _time_seconds(text: str) -> float:
    h, m, s = text.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _session(runs: Sequence[SyntheticRun], key: tuple[str, str | None]) -> SyntheticSession:
    """The expected anchor, order and frame check of a session, from the design alone.

    The rule is decided over EVERY run of the session (decision D36 as built: the anchor does
    not depend on which runs are packed): the time rule when every run has a raw sidecar with
    a time, else name order. The anchor is the first run in that order that has a stored
    transform, packable or not. ``order`` is the packable runs in that order."""
    every = [r for r in runs if (r.design.participant, r.design.session) == key]
    by_name_all = sorted(every, key=lambda r: (r.design.task, int(r.design.run)))
    if all(r.design.time is not None and "raw_sidecar" in r.files for r in every):
        rule = "acquisition_time"
        session_order = sorted(by_name_all, key=lambda r: _time_seconds(r.design.time or ""))
    else:
        rule = "name_order"
        session_order = by_name_all
    anchor = next(r for r in session_order if "xfm_to_anat" in r.files)
    order = [r for r in session_order if r.packable]
    relative = {r.stem: np.linalg.inv(anchor.displacement) @ r.displacement for r in order}
    others = [r for r in order if r is not anchor]
    relative[anchor.stem] = np.eye(4)
    # every displacement of the design is a pure shift, a rotation of a few degrees or a scale
    # of 0.4 %, so "beyond 10 mm somewhere in the mask" follows from the shift alone here; the
    # test computes the maximum over the mask and compares
    flagged = tuple(r.stem for r in others if np.linalg.norm(relative[r.stem][:3, 3]) > 10.0)
    outlier = len(others) >= 2 and len(flagged) == len(others)
    if outlier:
        pairs = [(a, b) for i, a in enumerate(others) for b in others[i + 1 :]]
        outlier = all(
            np.linalg.norm((np.linalg.inv(a.displacement) @ b.displacement)[:3, 3]) <= 10.0
            for a, b in pairs
        )
    return SyntheticSession(
        participant=key[0],
        session=key[1],
        anchor=anchor.stem,
        rule=rule,
        order=tuple(r.stem for r in order),
        flagged=flagged,
        anchor_is_outlier=outlier,
        frame=SESSION_FRAMES[key],
    )


def tree_bytes(root: Path) -> int:
    """Bytes of the regular files under ``root``, links not followed."""
    return sum(
        p.lstat().st_size for p in Path(root).rglob("*") if p.is_file() and not p.is_symlink()
    )


# --- step 2: one folder that ``surfcensus review`` prepares for both interfaces --------------


@dataclass(frozen=True)
class FuncCounts:
    """What the functional inventory of a run should find in a dataset of :data:`DESIGN`,
    counted from the design (work package E2.0, for E2.B, E2.C and E2.Z)."""

    participants: tuple[str, ...]
    sessions: tuple[tuple[str, str | None], ...]
    """``(participant, session)`` in stream order: participants by label, then no session
    first, then the labels with numbers compared as numbers."""
    runs: int
    packable: int
    """Runs without a reason in the design: what the pack phase packs."""
    reasons: Mapping[str, int]
    """Reason -> how many runs the design gives it."""
    fs_subjects: tuple[str, ...]
    """The FreeSurfer subjects the surface side reviews (inventory, metrics, render)."""


def _session_sort(key: tuple[str, str | None]) -> tuple[str, int, tuple[int, str]]:
    participant, session = key
    if session is None:
        return (participant, 0, (0, ""))
    return (participant, 1, (int(session), "") if session.isdigit() else (1 << 30, session))


def func_counts(dataset: SyntheticFuncDataset) -> FuncCounts:
    """The counts of :class:`FuncCounts` for a dataset that :func:`write_review_folder` (or
    :func:`write_func_dataset`) wrote."""
    reasons: dict[str, int] = {}
    for run in dataset.runs:
        if run.design.reason is not None:
            reasons[run.design.reason] = reasons.get(run.design.reason, 0) + 1
    keys = sorted(
        {(r.design.participant, r.design.session) for r in dataset.runs}, key=_session_sort
    )
    return FuncCounts(
        participants=tuple(sorted({p for p, _ in keys})),
        sessions=tuple(keys),
        runs=len(dataset.runs),
        packable=sum(1 for r in dataset.runs if r.packable),
        reasons=reasons,
        fs_subjects=tuple(sorted(dataset.subjects)),
    )


def write_review_folder(
    root: Path,
    *,
    generation: Generation = "scanner-T1w",
    without_fs: Sequence[str] = (),
    extra_fs: Sequence[str] = (),
    annex: Kind | None = None,
) -> SyntheticFuncDataset:
    """A BIDS dataset root at ``<root>/ds`` that holds both kinds of data, for ``review`` on one
    folder with both interfaces (step 2's exit check, E2.Z): the raw functional series, the
    fMRIPrep derivative at ``derivatives/fmriprep`` and, in its ``sourcedata/freesurfer``, the
    FreeSurfer subjects ``sub-01`` and ``sub-02`` that the surface side inventories, scores
    (ranked as decision D16 says: no fsaverage, two subjects) and renders.

    ``without_fs`` removes FreeSurfer subjects after writing, for a participant whose sessions
    have no contours and no surface rating (``("sub-02",)``); the surface side then has one
    subject. ``extra_fs`` adds FreeSurfer subjects that have no functional data, each a copy
    of ``sub-01`` (``("sub-03",)``, for a subject page without sessions; step 3, E3.0); their
    names must be new and must not be a participant of the functional data. ``annex`` is that
    of :func:`write_func_dataset`.

    Measured when this was written (one shared 4-core Linux machine, Python 3.13): writing
    0.6 s; ``review(<root>/ds, serve=False)`` into a new run directory 9.6 s for the two
    FreeSurfer subjects (the default render at 320 px, without video, one process), of which
    rendering is most; ``func.api.epi_subject`` packs ``sub-01`` (two sessions) in 0.9 s and
    ``sub-02`` (one session) in 0.5 s. The two FreeSurfer subjects are copies of one
    ellipsoid subject (``sub-02`` is ``sub-01`` copied), so their renders differ only in
    their ``subject_uid``.
    """
    dataset = write_func_dataset(root, generation=generation, annex=annex)
    gone = set(without_fs)
    unknown = gone - set(dataset.subjects)
    if unknown:
        raise ValueError(f"no FreeSurfer subject {', '.join(sorted(unknown))}")
    extra = set(extra_fs)
    taken = extra & (set(dataset.subjects) | {r.design.participant for r in dataset.runs})
    if taken:
        raise ValueError(f"not a new FreeSurfer subject: {', '.join(sorted(taken))}")
    added: dict[str, SyntheticSubject] = {}
    for name in sorted(extra):
        shutil.copytree(dataset.subjects["sub-01"].path, dataset.fs_subjects_dir / name)
        added[name] = _moved(dataset.subjects["sub-01"], dataset.fs_subjects_dir / name)
    for name in sorted(gone):
        shutil.rmtree(dataset.fs_subjects_dir / name)
    if not gone and not added:
        return dataset
    kept = {name: s for name, s in dataset.subjects.items() if name not in gone} | added
    kept = dict(sorted(kept.items()))
    return SyntheticFuncDataset(
        root=dataset.root,
        raw_root=dataset.raw_root,
        deriv_root=dataset.deriv_root,
        fs_subjects_dir=dataset.fs_subjects_dir,
        generation=dataset.generation,
        version=dataset.version,
        runs=dataset.runs,
        sessions=dataset.sessions,
        subjects=kept,
        placeholders=dataset.placeholders,
        fsnative_files=dataset.fsnative_files,
    )


def write_session_dates(
    raw_root: Path,
    participant: str,
    *,
    sessions: Mapping[str, str] | None = None,
    scans: Mapping[str, Sequence[str]] | None = None,
) -> list[Path]:
    """Write the tables the acquisition dates of a participant's sessions are read from
    (decision D42) into the raw dataset: ``sessions`` as ``sub-X/sub-X_sessions.tsv`` with one
    row ``ses-<label>`` and its ``acq_time`` value as given (a date such as
    ``1900-01-02T10:00:00``, with ``Z`` or fractional seconds, ``n/a`` or any text; a label
    left out has no row), and ``scans`` as ``sub-X/ses-<label>/sub-X_ses-<label>_scans.tsv``
    with one row per value (the file names are made up). Returns the files written."""
    root = Path(raw_root) / participant
    written = []
    if sessions is not None:
        path = root / f"{participant}_sessions.tsv"
        rows = ["session_id\tacq_time"] + [f"ses-{lab}\t{value}" for lab, value in sessions.items()]
        path.write_text("\n".join(rows) + "\n")
        written.append(path)
    for label, values in (scans or {}).items():
        path = root / f"ses-{label}" / f"{participant}_ses-{label}_scans.tsv"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = ["filename\tacq_time"] + [
            f"func/{participant}_ses-{label}_task-x_run-{n}_bold.nii.gz\t{value}"
            for n, value in enumerate(values, 1)
        ]
        path.write_text("\n".join(rows) + "\n")
        written.append(path)
    return written
