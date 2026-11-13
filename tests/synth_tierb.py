"""Tier A and Tier B files for synthetic subjects, with everything known in closed form.

Tier B (:func:`add_tier_b`): ``surf/?h.sphere.reg``, ``surf/?h.thickness`` and
``label/?h.aparc.annot``. A native vertex with direction ``d`` on the subject's own
icosphere (its white position is ``centre + d * semi_axes``) is registered to the template
direction ``p = warp_directions(d, warp)``: a rotation about ``WARP_AXIS`` by the angle
``warp * sin(pi * t)``, where ``t = d . WARP_AXIS`` is the latitude about that axis. Each
latitude circle turns rigidly, so the map is smooth, keeps areas and orientation, is the
identity for ``warp = 0``, and has the inverse :func:`unwarp_directions`. Thickness and
parcel of the vertex are functions of ``p``: :func:`expected_thickness` and
``synth_template.parcel_of``. So the thickness map on the template mesh is, up to
interpolation, ``expected_thickness`` of the template's own directions.

Tier A (:func:`add_tier_a`): the stats files, the build stamp and the recon-all logs in the
layouts of the real files (``ds000102`` sub-01, FreeSurfer 6.0.1, and ``ds003020``
sub-UTS01, FreeSurfer 7.3.2, both under fMRIPrep; the plain layout is assumed, as in
``tests/data/recon_logs``). Headers and command lines carry the strings of
:data:`CANARIES`, which must never reach a table, a log line or an export.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import nibabel.freesurfer as fs
import numpy as np
from synth import (  # pyrefly: ignore[missing-import]
    SyntheticSubject,
    icosphere,
    write_synthetic_subject,
)
from synth_template import (  # pyrefly: ignore[missing-import]
    DESIKAN_NAMES,
    MEDIAL_WALL_M,
    SPHERE_RADIUS,
    SyntheticTemplate,
    desikan_colours,
    direction,
    medial_coordinate,
    mesh_vertex_areas,
    parcel_of,
)

from surfcensus.sources.base import HEMIS

__all__ = [
    "CANARIES",
    "NOISE_SD_MM",
    "THICKNESS_MAX_MM",
    "WARP_AXIS",
    "SyntheticCohort",
    "ThickPatch",
    "add_tier_a",
    "add_tier_b",
    "cohort_patches",
    "expected_thickness",
    "in_patch",
    "subject_directions",
    "synthetic_cohort",
    "unwarp_directions",
    "warp_directions",
]

WARP_AXIS = np.array([0.3, 0.5, 0.8]) / math.sqrt(0.3**2 + 0.5**2 + 0.8**2)
"""Axis of the registration warp: on no symmetry axis of the icosphere."""
THICKNESS_MAX_MM = 5.0
NOISE_SD_MM = 0.1
"""Standard deviation of the per-vertex noise that ``noise_seed`` adds."""
CANARIES: tuple[str, ...] = ("ds999999", "canary-host", "canary-user")
"""An accession, a host name and a user name that the Tier A files carry where the real
files carry real ones."""

_SUBJECTS_DIR = "/data/ds999999-fmriprep/sourcedata/freesurfer"
_WORK_DIR = "/data/work/ds999999"
_CREATE_STAMP = "created by surfcensus tests/synth_tierb.py"
_UNAME = "Linux canary-host 3.10.0-1160.45.1.el7.x86_64 #1 SMP x86_64 x86_64 x86_64 GNU/Linux"
_STATS_PARCELS = tuple(name for name in DESIKAN_NAMES if name not in ("unknown", "corpuscallosum"))
_DEFAULT_TRUNCATIONS: dict[str, tuple[int, int]] = {"lh": (12, 140), "rh": (30, 201)}


@dataclass(frozen=True)
class ThickPatch:
    """A disc of captured-looking cortex: every vertex whose template-space direction lies
    within ``radius_deg`` of ``centre`` gets the thickness ``value_mm``, clamped at 5.0."""

    centre: tuple[float, float, float]
    """A template-space direction; need not be normalized."""
    radius_deg: float
    value_mm: float


@dataclass(frozen=True)
class SyntheticCohort:
    """A directory of synthetic subjects as the session fixtures hand it out."""

    root: Path
    """A plain subjects directory: ``root/<name>`` per subject, and ``root/fsaverage`` as a
    link to the template."""
    subjects: tuple[SyntheticSubject, ...]
    """In name order."""
    template: SyntheticTemplate
    bad: Mapping[str, ThickPatch]
    """Subject name -> the patch it carries on both hemispheres."""
    no_sphere_reg: tuple[str, ...]
    """Names of subjects without ``surf/?h.sphere.reg``."""
    no_stats: tuple[str, ...]
    """Names of subjects without a ``stats/`` directory."""


def _rotate(directions: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rodrigues' rotation of each direction about ``WARP_AXIS`` by its own angle."""
    d = np.asarray(directions, dtype=float)
    cos, sin = np.cos(angle)[..., None], np.sin(angle)[..., None]
    along = (d @ WARP_AXIS)[..., None]
    return d * cos + np.cross(WARP_AXIS, d) * sin + WARP_AXIS * along * (1.0 - cos)


def warp_directions(directions: np.ndarray, warp: float) -> np.ndarray:
    """Subject directions -> template-space directions. Unit vectors in, unit vectors out."""
    d = np.asarray(directions, dtype=float)
    return _rotate(d, warp * np.sin(np.pi * (d @ WARP_AXIS)))


def unwarp_directions(directions: np.ndarray, warp: float) -> np.ndarray:
    """The inverse of :func:`warp_directions`: the rotation leaves the latitude about the
    axis unchanged, so it is undone by the opposite angle at the same latitude."""
    p = np.asarray(directions, dtype=float)
    return _rotate(p, -warp * np.sin(np.pi * (p @ WARP_AXIS)))


def in_patch(directions: np.ndarray, patch: ThickPatch) -> np.ndarray:
    """Which directions lie in the disc of ``patch`` (boundary included)."""
    centre = np.asarray(patch.centre, dtype=float)
    cosine = np.asarray(directions, dtype=float) @ (centre / np.linalg.norm(centre))
    return cosine >= math.cos(math.radians(patch.radius_deg))


def expected_thickness(directions: np.ndarray, patch: ThickPatch | None = None) -> np.ndarray:
    """Thickness in mm at template-space directions, without noise: ``2.5 + 0.35 z +
    0.25 x y`` (2.0 to 3.0 mm, the same on both hemispheres), and inside ``patch`` its value
    clamped at 5.0. The medial strip is not zeroed here; the thickness file has 0 there."""
    u = np.asarray(directions, dtype=float)
    thickness = 2.5 + 0.35 * u[..., 2] + 0.25 * u[..., 0] * u[..., 1]
    if patch is not None:
        thickness = np.where(in_patch(u, patch), min(patch.value_mm, THICKNESS_MAX_MM), thickness)
    return thickness


def _subdivisions(n_vertices: int) -> int:
    order = round(math.log((n_vertices - 2) / 10, 4))
    if 10 * 4**order + 2 != n_vertices:
        raise ValueError(f"{n_vertices} vertices are not an icosphere")
    return order


def subject_directions(subject: SyntheticSubject) -> tuple[np.ndarray, np.ndarray]:
    """``(directions, faces)`` of the icosphere a synthetic subject's surfaces are made of."""
    return icosphere(_subdivisions(subject.n_vertices))


def add_tier_b(
    subject: SyntheticSubject,
    template: SyntheticTemplate,
    *,
    warp: float = 0.15,
    patch: ThickPatch | None = None,
    hemis: Sequence[str] = HEMIS,
    noise_seed: int | None = None,
    flip_winding: bool = False,
    offset_mm: float = 0.0,
) -> None:
    """Write ``surf/?h.sphere.reg``, ``surf/?h.thickness`` and ``label/?h.aparc.annot`` for
    the hemispheres in ``hemis``.

    * ``sphere.reg``: ``warp_directions(d, warp)`` at radius 100 with the subject's faces;
      ``flip_winding`` reverses every face.
    * ``thickness`` (float32): ``expected_thickness(p, patch) + offset_mm``, plus Gaussian
      noise of ``NOISE_SD_MM`` drawn from ``noise_seed`` and the hemisphere's position in
      ``HEMIS`` when a seed is given, clipped to 0.1 to 5.0 mm; then the patch value as it
      is, without offset or noise; then 0 on the medial strip.
    * ``aparc.annot``: ``parcel_of(p, hemi)`` with the names and colours of the template,
      -1 on the medial strip.

    ``template`` says which template the files are registered to. Its rules are closed-form
    functions of the direction, so nothing is read from its directory.
    """
    if not isinstance(template, SyntheticTemplate):
        raise TypeError("add_tier_b registers to a template of write_synthetic_template")
    directions, faces = subject_directions(subject)
    registered = warp_directions(directions, warp)
    (subject.path / "label").mkdir(exist_ok=True)
    for hemi in hemis:
        fs.write_geometry(
            str(subject.path / "surf" / f"{hemi}.sphere.reg"),
            registered * SPHERE_RADIUS,
            faces[:, ::-1] if flip_winding else faces,
            create_stamp=_CREATE_STAMP,
        )
        thickness = expected_thickness(registered) + offset_mm
        if noise_seed is not None:
            rng = np.random.default_rng([noise_seed, HEMIS.index(hemi)])
            thickness = thickness + rng.normal(0.0, NOISE_SD_MM, size=len(thickness))
        thickness = np.clip(thickness, 0.1, THICKNESS_MAX_MM)
        if patch is not None:
            thickness = np.where(
                in_patch(registered, patch), min(patch.value_mm, THICKNESS_MAX_MM), thickness
            )
        thickness = np.where(medial_coordinate(registered, hemi) > MEDIAL_WALL_M, 0.0, thickness)
        fs.write_morph_data(
            str(subject.path / "surf" / f"{hemi}.thickness"),
            thickness.astype(np.float32),
            fnum=len(faces),
        )
        fs.write_annot(
            str(subject.path / "label" / f"{hemi}.aparc.annot"),
            parcel_of(registered, hemi),
            desikan_colours(),
            list(DESIKAN_NAMES),
            fill_ctab=True,
        )


def _build_stamp(fs_version: str) -> str:
    if int(fs_version.split(".")[0]) >= 7:
        return f"freesurfer-linux-ubuntu22_x86_64-{fs_version}-20220804-6354275"
    return f"freesurfer-Linux-centos6_x86_64-stable-pub-v{fs_version}-f53a55a"


def _parcel_rows(
    subject: SyntheticSubject, hemi: str
) -> list[tuple[str, int, int, int, float, float]]:
    """``(name, NumVert, SurfArea, GrayVol, ThickAvg, ThickStd)`` per parcel.

    From the subject's own thickness, annotation and white surface when Tier B is there:
    one row per parcel that has vertices, with the plain mean and standard deviation of its
    thickness. Otherwise every Desikan parcel with numbers made by rule and 2.5 mm.
    """
    thickness_file = subject.path / "surf" / f"{hemi}.thickness"
    annot_file = subject.path / "label" / f"{hemi}.aparc.annot"
    if not (thickness_file.is_file() and annot_file.is_file()):
        return [
            (name, 400 + 20 * k, 260 + 13 * k, round((260 + 13 * k) * 2.5), 2.5, 0.5)
            for k, name in enumerate(_STATS_PARCELS)
        ]
    thickness = fs.read_morph_data(str(thickness_file))
    labels, _, names = fs.read_annot(str(annot_file))
    white, faces = fs.read_geometry(str(subject.path / "surf" / f"{hemi}.white"))
    area = mesh_vertex_areas(white, faces)
    rows = []
    for index, raw in enumerate(names):
        name = raw.decode()
        members = labels == index
        if name not in _STATS_PARCELS or not members.any():
            continue
        surf_area = round(float(area[members].sum()))
        mean = round(float(thickness[members].mean()), 3)
        rows.append(
            (
                name,
                int(members.sum()),
                surf_area,
                round(surf_area * mean),
                mean,
                round(float(thickness[members].std()), 3),
            )
        )
    return rows


def _stats_header(program: str, cmdline: str, fs_version: str) -> list[str]:
    """The header lines that name the program and the machine, canaries included."""
    if int(fs_version.split(".")[0]) >= 7:
        version = f"# cvs_version {fs_version}"
    else:
        version = f"# cvs_version $Id: {program}.c,v 1.79 2016/03/14 15:15:34 greve Exp $"
    return [
        "# generating_program " + program,
        version,
        "# cmdline " + cmdline,
        "# sysname  Linux",
        "# hostname canary-host",
        "# machine  x86_64",
        "# user     canary-user",
    ]


def _aparc_stats(subject: SyntheticSubject, hemi: str, fs_version: str) -> str:
    name = subject.path.name
    fs7 = int(fs_version.split(".")[0]) >= 7
    rows = _parcel_rows(subject, hemi)
    n_vert = sum(row[1] for row in rows)
    area = sum(row[2] for row in rows)
    mean = sum(row[2] * row[4] for row in rows) / area if area else 0.0
    cmdline = (
        f"mris_anatomical_stats -th3 -mgz {'-noglobal ' if fs7 else ''}-cortex "
        f"../label/{hemi}.cortex.label -f ../stats/{hemi}.aparc.stats -b -a "
        f"../label/{hemi}.aparc.annot -c ../label/aparc.annot.ctab {name} {hemi} white "
    )
    lines = [
        "# Table of FreeSurfer cortical parcellation anatomical statistics ",
        "# ",
        "# CreationTime 2026/11/13-10:00:00-GMT",
        *_stats_header("mris_anatomical_stats", cmdline, fs_version),
        "# ",
        "# SUBJECTS_DIR " + _SUBJECTS_DIR,
        "# anatomy_type surface",
        f"# subjectname {name}",
        f"# hemi {hemi}",
        f"# AnnotationFile ../label/{hemi}.aparc.annot",
        "# AnnotationFileTimeStamp 2026/11/13 09:00:00",
        f"# Measure Cortex, NumVert, Number of Vertices, {n_vert}, unitless",
        f"# Measure Cortex, WhiteSurfArea, White Surface Total Area, {area:g}, mm^2",
        f"# Measure Cortex, MeanThickness, Mean Thickness, {mean:.5f}, mm",
    ]
    if not fs7:
        # FreeSurfer 6 writes the brain volumes here too, one of them without the comma
        # after its name (seen in ds000102 sub-01); FreeSurfer 7.3.2 under fMRIPrep passes
        # -noglobal and writes none (ds003020 sub-UTS01)
        lines += [
            "# Measure BrainSeg, BrainSegVol, Brain Segmentation Volume, 1158358.000000, mm^3",
            "# Measure Cortex, CortexVol Total cortical gray matter volume, 484608.441133, mm^3",
            "# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated Total Intracranial "
            "Volume, 1519867.314105, mm^3",
        ]
    columns = (
        ("StructName", "Structure Name", "NA"),
        ("NumVert", "Number of Vertices", "unitless"),
        ("SurfArea", "Surface Area", "mm^2"),
        ("GrayVol", "Gray Matter Volume", "mm^3"),
        ("ThickAvg", "Average Thickness", "mm"),
        ("ThickStd", "Thickness StdDev", "mm"),
        ("MeanCurv", "Integrated Rectified Mean Curvature", "mm^-1"),
        ("GausCurv", "Integrated Rectified Gaussian Curvature", "mm^-2"),
        ("FoldInd", "Folding Index", "unitless"),
        ("CurvInd", "Intrinsic Curvature Index", "unitless"),
    )
    lines.append(f"# NTableCols {len(columns)}")
    for number, (header, field, units) in enumerate(columns, start=1):
        lines += [
            f"# TableCol {number:2d} ColHeader {header}",
            f"# TableCol {number:2d} FieldName {field}",
            f"# TableCol {number:2d} Units     {units}",
        ]
    lines.append("# ColHeaders " + " ".join(column[0] for column in columns))
    lines += [
        f"{parcel:<36}{num_vert:9d}{surf_area:7d}{gray_vol:7d}  {thick:.3f} {std:.3f}"
        "     0.120     0.025       10     1.0"
        for parcel, num_vert, surf_area, gray_vol, thick, std in rows
    ]
    return "\n".join(lines) + "\n"


def _aseg_stats(subject: SyntheticSubject, holes: tuple[int, int], fs_version: str) -> str:
    name = subject.path.name
    cmdline = (
        "mri_segstats --seg mri/aseg.mgz --sum stats/aseg.stats --pv mri/norm.mgz --empty "
        f"--brainmask mri/brainmask.mgz --brain-vol-from-seg --subject {name} "
    )
    lines = [
        "# Title Segmentation Statistics ",
        "# ",
        *_stats_header("mri_segstats", cmdline, fs_version),
        "# anatomy_type volume",
        "# ",
        "# SUBJECTS_DIR " + _SUBJECTS_DIR,
        f"# subjectname {name}",
        "# Measure BrainSeg, BrainSegVol, Brain Segmentation Volume, 1158358.000000, mm^3",
        "# Measure Cortex, CortexVol, Total cortical gray matter volume, 484608.441133, mm^3",
        "# Measure lhSurfaceHoles, lhSurfaceHoles, Number of defect holes in lh surfaces "
        f"prior to fixing, {holes[0]}, unitless",
        "# Measure rhSurfaceHoles, rhSurfaceHoles, Number of defect holes in rh surfaces "
        f"prior to fixing, {holes[1]}, unitless",
        "# Measure SurfaceHoles, SurfaceHoles, Total number of defect holes in surfaces "
        f"prior to fixing, {holes[0] + holes[1]}, unitless",
        "# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated Total Intracranial Volume, "
        "1519867.314105, mm^3",
        "# SegVolFile mri/aseg.mgz ",
        "# ColorTable /opt/freesurfer/ASegStatsLUT.txt ",
        "# VoxelVolume_mm3 1 ",
        "# NRows 2 ",
        "# NTableCols 10 ",
        "# ColHeaders  Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin "
        "normMax normRange  ",
        "  1   4      8830     9198.1  Left-Lateral-Ventricle            20.8310    13.9629"
        "     0.0000    83.0000    83.0000 ",
        "  2  43      8015     8402.7  Right-Lateral-Ventricle           21.5022    14.1005"
        "     0.0000    85.0000    85.0000 ",
    ]
    return "\n".join(lines) + "\n"


def _invocation(
    style: str, name: str, arguments: str, stamp: str, body: Sequence[str], when: str
) -> list[str]:
    """One recon-all invocation as the script logs it, ending with its closing line."""
    script = "/opt/freesurfer/bin/recon-all"
    if style == "fs7-fmriprep":
        # FreeSurfer 7 logs the command line after the path of the script, on one line
        header = [
            when,
            f"cd {_SUBJECTS_DIR}/{name}",
            f"setenv SUBJECTS_DIR {_SUBJECTS_DIR}",
            f"{script} {arguments}",
            "",
        ]
    else:
        header = [when, f"{_SUBJECTS_DIR}/{name}", script, arguments]
    return [
        *header,
        f"subjid {name}",
        f"setenv SUBJECTS_DIR {_SUBJECTS_DIR}",
        "FREESURFER_HOME /opt/freesurfer",
        "Actual FREESURFER_HOME /opt/freesurfer",
        f"build-stamp.txt: {stamp}",
        _UNAME,
        *body,
        "#@#%# recon-all-run-time-hours 0.500",
        f"recon-all -s {name} finished without error at {when}",
        "done",
    ]


def _log(style: str, invocations: Sequence[list[str]]) -> str:
    separator = (
        "#New# invocation of recon-all "
        if style == "fs7-fmriprep"
        else ("New invocation of recon-all ")
    )
    lines: list[str] = []
    for position, invocation in enumerate(invocations):
        if position:
            lines += ["", "", separator, "", "", ""]
        lines += invocation
    return "\n".join(lines) + "\n"


def _truncation_lines(counts: tuple[int, int] | None, n_vertices: int) -> list[str]:
    if counts is None:
        return []
    return [
        f"{n_vertices} of {n_vertices} vertices processed",
        f"thickness calculation complete, {counts[0]}:{counts[1]} truncations.",
        "0 vertices at 0 distance",
    ]


def add_tier_a(
    subject: SyntheticSubject,
    *,
    log_style: Literal["fs6-fmriprep", "fs7-fmriprep", "plain"] = "fs6-fmriprep",
    flags: Sequence[str] = (),
    truncations: Mapping[str, tuple[int, int]] | None = None,
    holes: tuple[int, int] = (3, 5),
    fs_version: str = "6.0.1",
) -> None:
    """Write ``stats/?h.aparc.stats``, ``stats/aseg.stats``, ``scripts/build-stamp.txt`` and
    the recon-all logs.

    * Stats: the real layout, with the canaries in the header. ``fs_version`` below 7 keeps
      the brain-volume measures of FreeSurfer 6 with its malformed ``CortexVol`` line; from 7
      on they are absent, as in the 7.3.2 files. The parcel rows follow the subject's own
      thickness and annotation when :func:`add_tier_b` ran before (rows for the parcels that
      have vertices, 11 with the synthetic annotation); otherwise all 34 Desikan parcels get
      numbers made by rule. ``aseg.stats`` carries ``holes`` as ``lhSurfaceHoles`` and
      ``rhSurfaceHoles``.
    * Logs. ``fs6-fmriprep``: ``recon-all.log`` with four invocations and two per-hemisphere
      logs with two each, arguments alone on the line above ``subjid``. ``fs7-fmriprep``:
      five and two, arguments after the script path, ``-noT2pial -noFLAIRpial`` in the first
      invocation. ``plain``: one invocation in ``recon-all.log`` and no per-hemisphere log.
      The truncation line of a hemisphere is in its own log, or for ``plain`` in
      ``recon-all.log``, left first; the holes line is in ``recon-all.log``. Every
      invocation ends with ``finished without error``.
    * ``flags`` are added to the first invocation of ``recon-all.log``, for example
      ``("-T2pial",)`` or ``("-hires",)``.
    * ``truncations``: the two counts per hemisphere; None gives (12, 140) and (30, 201),
      and a hemisphere left out of a mapping gets no truncation line.

    The log style and ``fs_version`` are independent: the caller keeps them consistent.
    """
    name = subject.path.name
    stats, scripts = subject.path / "stats", subject.path / "scripts"
    stats.mkdir(exist_ok=True)
    scripts.mkdir(exist_ok=True)
    for hemi in HEMIS:
        (stats / f"{hemi}.aparc.stats").write_text(_aparc_stats(subject, hemi, fs_version))
    (stats / "aseg.stats").write_text(_aseg_stats(subject, holes, fs_version))
    stamp = _build_stamp(fs_version)
    (scripts / "build-stamp.txt").write_text(stamp + "\n")

    counts = _DEFAULT_TRUNCATIONS if truncations is None else truncations
    sd = f"-subjid {name} -sd {_SUBJECTS_DIR}"
    extra = "".join(f" {flag}" for flag in flags)
    holes_lines = [
        f"orig.nofix lheno = {2 - 2 * holes[0]:4d}, rheno = {2 - 2 * holes[1]}",
        f"orig.nofix lhholes = {holes[0]:4d}, rhholes = {holes[1]}",
        "Loading mri/aseg.mgz",
    ]

    def invocation(arguments: str, body: Sequence[str], minute: int) -> list[str]:
        when = f"Fri Nov 13 10:{minute:02d}:00 UTC 2026"
        return _invocation(log_style, name, arguments, stamp, body, when)

    if log_style == "plain":
        body = [
            line
            for hemi in HEMIS
            for line in (
                f"#@# Make Pial Surf {hemi} Fri Nov 13 10:30:00 UTC 2026",
                *_truncation_lines(counts.get(hemi), subject.n_vertices),
            )
        ]
        arguments = f"-s {name} -i {_WORK_DIR}/raw/{name}_T1w.nii.gz -all{extra}"
        log = _log(log_style, [invocation(arguments, [*body, *holes_lines], 0)])
        (scripts / "recon-all.log").write_text(log)
        for hemi in HEMIS:  # of an earlier call in another style
            (scripts / f"recon-all-{hemi}.log").unlink(missing_ok=True)
        return

    fs7 = log_style == "fs7-fmriprep"
    first = (
        f"-autorecon1 -i {_WORK_DIR}_{name}/fmriprep_wf/anat_validate/{name}_T1w_ras_valid.nii.gz "
        f"-noskullstrip{' -noT2pial -noFLAIRpial' if fs7 else ''} -openmp 7 {sd}{extra}"
    )
    main = [invocation(first, [], 0)]
    if fs7:
        main.append(invocation(f"-openmp 7 {sd} -gcareg", [], 5))
        main.append(invocation(f"-autorecon2-volonly -openmp 7 {sd} -nogcareg", [], 10))
    else:
        main.append(invocation(f"-autorecon2-volonly -openmp 7 {sd}", [], 10))
    main.append(invocation(f"-parallel {sd} -cortribbon", [], 40))
    main.append(
        invocation(
            f"-autorecon3 -openmp 7 {sd} -nosphere -nosurfreg -nojacobian_white -noavgcurv "
            "-nocortparc -nopial -noparcstats -nocortparc2 -noparcstats2 -nocortparc3 "
            "-noparcstats3 -nopctsurfcon -nocortribbon -nobalabels",
            holes_lines,
            50,
        )
    )
    (scripts / "recon-all.log").write_text(_log(log_style, main))
    for hemi in HEMIS:
        only = f" -{hemi}-only" if fs7 else ""
        per_hemi = [
            invocation(
                f"-autorecon-hemi {hemi} -noparcstats -noparcstats2 -noparcstats3 "
                f"-nohyporelabel -nobalabels{only} -openmp 7 {sd}",
                _truncation_lines(counts.get(hemi), subject.n_vertices),
                20,
            ),
            invocation(
                f"-autorecon-hemi {hemi} -nohyporelabel{only} -openmp 7 {sd} -notessellate "
                "-nosmooth1 -noinflate1 -noqsphere -nofix -nowhite -nosmooth2 -noinflate2 "
                "-nocurvHK -nocurvstats -nosphere -nosurfreg -nojacobian_white -noavgcurv "
                "-nocortparc -nopial -nocortparc2 -nocortparc3 -nopctsurfcon",
                [],
                45,
            ),
        ]
        (scripts / f"recon-all-{hemi}.log").write_text(_log(log_style, per_hemi))


def synthetic_cohort(
    root: Path,
    n: int,
    template: SyntheticTemplate,
    *,
    seed: int,
    bad: Mapping[int, ThickPatch] | None = None,
) -> list[SyntheticSubject]:
    """Write ``n`` small subjects ``root/sub-000``, ``root/sub-001``, ... with Tier A, B and C.

    Each has 642 vertices per surface and a 16^3 volume of 10 mm voxels, so it also renders
    in a fraction of a second. Subjects differ, by ``seed`` and their position, in the
    registration warp (0.05 to 0.25 rad), a thickness offset (standard deviation 0.15 mm),
    the per-vertex noise, the holes and the truncation counts. ``bad`` gives the subject at
    a position a :class:`ThickPatch`, on both hemispheres. Tier A is FreeSurfer 6.0.1 in
    the fMRIPrep layout, written after Tier B, so the stats follow the thickness.

    32 subjects cross the 30-subject rule of decision D16.
    """
    rng = np.random.default_rng(seed)
    subjects = []
    for position in range(n):
        subject = write_synthetic_subject(
            Path(root),
            name=f"sub-{position:03d}",
            voxel_size=10.0,
            shape=(16, 16, 16),
            subdivisions=3,
        )
        add_tier_b(
            subject,
            template,
            warp=float(rng.uniform(0.05, 0.25)),
            patch=(bad or {}).get(position),
            noise_seed=seed * 1000 + position,
            offset_mm=float(rng.normal(0.0, 0.15)),
        )
        add_tier_a(
            subject,
            holes=(int(rng.integers(0, 40)), int(rng.integers(0, 40))),
            truncations={
                "lh": (int(rng.integers(0, 30)), int(rng.integers(50, 200))),
                "rh": (int(rng.integers(0, 30)), int(rng.integers(50, 200))),
            },
        )
        subjects.append(subject)
    return subjects


def cohort_patches(*, value_mm: float = 5.0, radius_deg: float = 16.0) -> dict[str, ThickPatch]:
    """Three patches in the search region of the left hemisphere, 20 degrees from the pole:
    ``medial`` (pericalcarine), ``inferior`` (lingual) and ``superior`` (cuneus)."""
    return {
        name: ThickPatch(
            centre=direction("lh", from_pole_deg=20.0, azimuth_deg=azimuth),
            radius_deg=radius_deg,
            value_mm=value_mm,
        )
        for name, azimuth in (("medial", 0.0), ("inferior", -90.0), ("superior", 90.0))
    }
