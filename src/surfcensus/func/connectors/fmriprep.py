"""The fMRIPrep connector (docs/epi_interface.md, The fMRIPrep connector). Work package E1.A.

Two generations of file names: ``scanner-T1w`` (fMRIPrep 20.2.x to 23.0.2, the layout of the
prototype) and ``boldref-T1w`` (23.2 and later, where the transform starts from the frame of
fMRIPrep's reference image). The names below are those of the spec's table, read in the
sources of fMRIPrep on PyPI; ``<stem>`` is the run's BIDS stem. Runs of the second generation
are listed and packed only after the check that the spec asks for on a real derivative.

The summary report (``figures/<stem>_desc-summary_bold.html``) is read for the lines
``Slice timing correction:``, ``Susceptibility distortion correction:`` and ``Registration:``
(:data:`REPORT_LABELS`). ``Skipped`` reads as not applied and ``n/a`` as unknown, each with the
report's words kept. Head-motion correction is always applied by fMRIPrep and its report does
not say so per run (:data:`MOTION_SOURCE`).

What was read in fMRIPrep's sources (sdists on PyPI, 2026-11-14), ``interfaces/reports.py``,
class ``FunctionalSummary``: the template ``FUNCTIONAL_TEMPLATE`` has the three lines as
``<li>Label: {value}</li>`` in 20.2.3 (lines 29 to 46), 21.0.2, 23.0.2, 23.2.0, 24.1.1 and
25.2.3. The slice-timing value is ``Applied``, ``Not applied`` or ``Skipped (too few
volumes)`` in every one of them (20.2.3 lines 183 to 185; 25.2.3 lines 237 to 244), and
``n/a`` from 23.2.0 on, when fMRIPrep did not record it (23.2.0 lines 234 to 241). The
registration value is ``FreeSurfer <code>bbregister</code> (boundary-based registration, BBR)
- N dof``, or, when boundary-based registration was rejected, ``FreeSurfer
<code>mri_coreg</code> - N dof``; with FSL ``FSL <code>flirt</code> with boundary-based
registration (BBR) metric - N dof`` or ``FSL <code>flirt</code> rigid registration - 6 dof``
(20.2.3 lines 187 to 196); from 25.2.3 also ``Precomputed affine transformation`` (25.2.3 lines
246 to 247). The distortion-correction value is the method's name, ``None`` without one
(21.0.2 ``workflows/bold/base.py`` line 976; 23.2.0 ``workflows/bold/fit.py`` line 313; 25.2.3
``fit.py`` line 321; in 20.2.3 it comes from sdcflows, whose ``None`` is to verify).

The confounds table: fMRIPrep writes ``std_dvars`` under that header (``workflows/bold/
confounds.py``, ``AddTSVHeader(columns=["std_dvars"])``, 20.2.3 line 257, 25.2.3 line 373)
and renames the other columns from CamelCase (``FramewiseDisplacement``) to snake case in
``interfaces/confounds.py``, ``_gather_confounds`` (``camel_to_snake``, 20.2.3 line 199,
25.2.3 line 431). Tables with the CamelCase names (``FramewiseDisplacement``, ``stdDVARS``,
``NonSteadyStateOutlier00``), from versions before that renaming, are read as well (to verify:
no such table and no such version was read).

Nothing here raises on any text or file name but :class:`~surfcensus.func.connectors.base.
ConnectorError`, whose detail names a file by its name or its location in the derivative,
never by its path; a report that does not read gives ``unknown`` statements.
"""

from __future__ import annotations

import html
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from surfcensus.func import bids
from surfcensus.func.connectors.base import ConnectorError
from surfcensus.func.itk import ItkError, read_itk_file
from surfcensus.records import (
    MULTI_ECHO,
    NATIVE_SPACE_ONLY,
    NO_PREPROCESSED_SERIES,
    STANDARD_SPACE_ONLY,
    ConnectorMatch,
    Corrections,
    DerivRun,
    Display,
    FuncFileRef,
    Generation,
    Matrix4,
    Motion,
    RigidPath,
    Statement,
)
from surfcensus.sources.base import FuncRole

NAME = "fmriprep"

SCANNER_T1W: Generation = "scanner-T1w"
BOLDREF_T1W: Generation = "boldref-T1w"
FIRST_BOLDREF_VERSION = (23, 2)
"""The first version whose transforms start from ``boldref``."""

ANAT_SPACE = "T1w"
"""The anatomy's world; ``anat`` is the same space under fMRIPrep 23.2 and later."""
ANAT_SPACES: tuple[str, ...] = ("T1w", "anat")

FUNC_SUFFIXES: Mapping[Generation, Mapping[FuncRole, tuple[str, ...]]] = {
    SCANNER_T1W: {
        FuncRole.PRE_BOLD: ("_space-T1w_desc-preproc_bold.nii.gz",),
        FuncRole.PRE_SIDECAR: ("_space-T1w_desc-preproc_bold.json",),
        FuncRole.PRE_MASK: ("_space-T1w_desc-brain_mask.nii.gz",),
        FuncRole.REFERENCE: ("_boldref.nii.gz",),
        FuncRole.XFM_TO_ANAT: ("_from-scanner_to-T1w_mode-image_xfm.txt",),
        FuncRole.CONFOUNDS: ("_desc-confounds_timeseries.tsv",),
    },
    BOLDREF_T1W: {
        FuncRole.PRE_BOLD: (
            "_space-T1w_desc-preproc_bold.nii.gz",
            "_space-anat_desc-preproc_bold.nii.gz",
        ),
        FuncRole.PRE_SIDECAR: (
            "_space-T1w_desc-preproc_bold.json",
            "_space-anat_desc-preproc_bold.json",
        ),
        FuncRole.PRE_MASK: (
            "_space-T1w_desc-brain_mask.nii.gz",
            "_space-anat_desc-brain_mask.nii.gz",
        ),
        FuncRole.REFERENCE: ("_desc-coreg_boldref.nii.gz", "_desc-hmc_boldref.nii.gz"),
        FuncRole.XFM_TO_ANAT: ("_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt",),
        FuncRole.CONFOUNDS: ("_desc-confounds_timeseries.tsv",),
    },
}
"""Per generation and role of the anatomy-space layout, the candidate suffixes of a file in
``sub-X/[ses-Y/]func/``, first match first. Which reference image arbitrates in the second
generation is to verify (docs/epi_interface.md, What could not be verified). The series, its
sidecar and its mask are taken in one space: the first of :data:`ANAT_SPACES` that has the
series."""

NATIVE_SUFFIXES: Mapping[FuncRole, str] = {
    FuncRole.PRE_BOLD: "_desc-preproc_bold.nii.gz",
    FuncRole.PRE_SIDECAR: "_desc-preproc_bold.json",
    FuncRole.PRE_MASK: "_desc-brain_mask.nii.gz",
}
"""The series in the run's own frame (layout 2; listed, not packed in step 1)."""

FIGURE_SUFFIXES: Mapping[FuncRole, str] = {
    FuncRole.SUMMARY_REPORT: "_desc-summary_bold.html",
    FuncRole.VALIDATION_REPORT: "_desc-validation_bold.html",
    FuncRole.SDC_FIGURE: "_desc-sdc_bold.svg",
}
"""Files in ``sub-X/figures/``."""

FSNATIVE_SUFFIX = "_from-T1w_to-fsnative_mode-image_xfm.txt"
"""``sub-X[_ses-Y]<suffix>`` in ``sub-X/[ses-Y/]anat/``: points of fsnative to the anatomy."""

REPORT_LABELS: tuple[str, str, str] = (
    "Slice timing correction",
    "Susceptibility distortion correction",
    "Registration",
)

MOTION_SOURCE = (
    "fMRIPrep applies head-motion correction to every preprocessed series; its summary report"
    " does not state it per run"
)

CONFOUND_COLUMNS: tuple[str, str] = ("framewise_displacement", "std_dvars")
NON_STEADY_PREFIX = "non_steady_state_outlier"

OLD_CONFOUND_COLUMNS: tuple[str, str] = ("FramewiseDisplacement", "stdDVARS")
"""The CamelCase names of the two columns, before fMRIPrep renamed them (to verify)."""
OLD_NON_STEADY_PREFIX = "NonSteadyStateOutlier"

RIGID_TOLERANCE = 1e-3
"""``RigidPath.rigid``: every singular value of the 3x3 part within this of 1."""
REPORT_MAX_BYTES = 2 * 1024 * 1024
"""A larger summary report is not read; one is a few kB."""
WORDS_MAX = 160
"""The report's words are kept up to this many characters, as stage 1 did."""

_RUN_STEM_KEYS = ("sub", "ses", "task", "acq", "ce", "rec", "dir", "run")
_DESCRIPTION_MAX_BYTES = 1024 * 1024
_VERSION = re.compile(r"[vV]?([0-9]{1,4})\.([0-9]{1,4})")
_SPACE = re.compile(r"_space-([A-Za-z0-9]+)_")
_DOF = re.compile(r"([0-9]{1,2})\s*dof\b")
_TAGS = re.compile(r"<[^>]*>")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_BOLDREF_MARKS = (
    FUNC_SUFFIXES[BOLDREF_T1W][FuncRole.XFM_TO_ANAT][0],
    "_desc-coreg_boldref.nii.gz",
    "_desc-hmc_boldref.nii.gz",
)
"""Names that only the second generation writes; they decide a version that does not parse."""
_FALLBACK_REGISTRATIONS = ("FreeSurfer mri_coreg", "FSL flirt rigid registration")
"""The words of the registration line when boundary-based registration was rejected
(20.2.3 ``interfaces/reports.py`` lines 187 to 196, the second entry of each list)."""

# --- versions -------------------------------------------------------------------------------


def _version_tuple(version: object) -> tuple[int, int] | None:
    if not isinstance(version, str):
        return None
    m = _VERSION.match(version.strip())
    return (int(m.group(1)), int(m.group(2))) if m else None


def generation_of(version: str | None) -> Generation:
    """The generation of a stated fMRIPrep version: ``boldref-T1w`` from 23.2 on, else
    ``scanner-T1w`` (also for a version that does not parse, which is then decided by the
    listing in ``detect``). Only the major and minor number are read (``23.2.0rc1`` is
    23.2)."""
    parsed = _version_tuple(version)
    if parsed is not None and parsed >= FIRST_BOLDREF_VERSION:
        return BOLDREF_T1W
    return SCANNER_T1W


# --- the summary report ---------------------------------------------------------------------


def _report_line(text: str, label: str) -> str | None:
    """The words after ``<label>:`` in an ``<li>`` of the report: tags removed, entities
    decoded, white space collapsed, at most :data:`WORDS_MAX` characters; None without the
    line, ``""`` for an empty one. The first such line counts."""
    found = re.search(r"<li>\s*" + re.escape(label) + r"\s*:(.*?)</li>", text, re.S | re.I)
    if found is None:
        return None
    words = html.unescape(_TAGS.sub("", found.group(1)))
    words = " ".join(_CONTROL.sub(" ", words).split())
    return words[:WORDS_MAX]


def _missing(label: str) -> str:
    return f"the summary report has no line '{label}'"


def _unknown(note: str) -> Statement:
    return Statement("unknown", None, None, note)


def _motion_statement() -> Statement:
    return Statement("applied", None, None, MOTION_SOURCE)


def read_summary_report(text: str, sdc_figure_listed: bool | None) -> Corrections:
    """The corrections that a summary report states, each with the report's words and the
    role ``summary_report`` as its source; ``unknown`` for a line that is not there.

    Slice timing: ``Applied`` is applied; ``Not applied`` and ``Skipped (too few volumes)``
    are not applied; ``n/a`` and any other words are unknown. Distortion correction: ``None``
    is not applied, other words are applied and are the method. Registration: any words are
    applied, with the degrees of freedom they state (``- 6 dof``) and a note when they are a
    fallback of fMRIPrep. ``sdc_figure_listed`` is kept and, when it contradicts the report's
    distortion line, said in that statement's note. Only the first 1 MiB of ``text`` is read;
    nothing raises."""
    if not isinstance(text, str):
        text = ""
    text = text[: 1024 * 1024]
    source = FuncRole.SUMMARY_REPORT
    stc_label, sdc_label, reg_label = REPORT_LABELS

    words = _report_line(text, stc_label)
    lowered = (words or "").lower()
    if words is None:
        slice_timing = _unknown(_missing(stc_label))
    elif lowered.startswith("applied"):
        slice_timing = Statement("applied", words, source)
    elif lowered.startswith("not applied"):
        slice_timing = Statement("not applied", words, source)
    elif lowered.startswith("skipped"):
        slice_timing = Statement("not applied", words, source, "fMRIPrep skipped it")
    elif lowered == "n/a":
        note = "the report says n/a: fMRIPrep did not record whether it was applied"
        slice_timing = Statement("unknown", words, source, note)
    else:
        slice_timing = Statement(
            "unknown", words or None, source, "the report's words are not known"
        )

    words = _report_line(text, sdc_label)
    sdc_method = None
    if words is None:
        sdc = _unknown(_missing(sdc_label))
    elif not words:
        sdc = Statement("unknown", None, source, "the report's line is empty")
    elif words.lower() == "none":
        note = None
        if sdc_figure_listed:
            note = "the listing has a distortion-correction figure of the run all the same"
        sdc = Statement("not applied", words, source, note)
    else:
        note = None
        if sdc_figure_listed is False:
            note = "the listing has no distortion-correction figure of the run"
        sdc = Statement("applied", words, source, note)
        sdc_method = words

    words = _report_line(text, reg_label)
    dof = None
    if words is None:
        registration = _unknown(_missing(reg_label))
    elif not words:
        registration = Statement("unknown", None, source, "the report's line is empty")
    else:
        found = _DOF.findall(words)
        dof = int(found[-1]) if found else None
        note = None
        if words.startswith(_FALLBACK_REGISTRATIONS):
            note = "boundary-based registration was rejected and fMRIPrep fell back to this one"
        registration = Statement("applied", words, source, note)

    return Corrections(
        motion=_motion_statement(),
        sdc=sdc,
        sdc_method=sdc_method,
        slice_timing=slice_timing,
        registration=registration,
        dof=dof,
        sdc_figure_listed=sdc_figure_listed,
    )


def _without_report(note: str, sdc_figure_listed: bool | None) -> Corrections:
    """The corrections of a run whose report is absent or does not read. A distortion figure
    in the listing says that distortion correction ran, with an unknown method (stage 1's
    rule)."""
    if sdc_figure_listed:
        sdc = Statement(
            "applied",
            None,
            FuncRole.SDC_FIGURE,
            f"{note}; the listing has the distortion-correction figure of the run",
        )
    else:
        sdc = _unknown(note)
    return Corrections(
        motion=_motion_statement(),
        sdc=sdc,
        sdc_method=None,
        slice_timing=_unknown(note),
        registration=_unknown(note),
        dof=None,
        sdc_figure_listed=sdc_figure_listed,
    )


def _read_small(path: Path, limit: int) -> str | None:
    """A small text file, or None when it cannot be read, is larger than ``limit`` or is not
    UTF-8."""
    try:
        with open(path, "rb") as handle:
            data = handle.read(limit + 1)
    except (OSError, ValueError):
        return None
    if len(data) > limit:
        return None
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None


# --- the confounds table --------------------------------------------------------------------


def _cell(text: str) -> float | None:
    text = text.strip()
    if not text or text.lower() == "n/a" or len(text) > 64:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def read_confounds(path: Path) -> Motion:
    """Framewise displacement, standardised DVARS and the non-steady-state volumes of a
    confounds table (tab-separated, ``n/a`` for no number). Raises ``ConnectorError`` with
    ``invalid`` for a table without the two columns.

    The columns are :data:`CONFOUND_COLUMNS`, or the older :data:`OLD_CONFOUND_COLUMNS` when
    the table has those instead. A cell that is ``n/a``, empty, missing at the end of a short
    row, or not a finite number is None; values are kept as the table writes them (rounding is
    for whoever shows them). The non-steady-state volumes are the rows with a 1 in any column
    named ``non_steady_state_outlier*`` (or ``NonSteadyStateOutlier*``). ``unreadable`` for a
    file that cannot be opened, ``invalid`` for one that is not UTF-8 text."""
    name = Path(path).name
    try:
        data = Path(path).read_bytes()
    except OSError as error:
        raise ConnectorError(
            "unreadable", f"{name}: {error.strerror or 'cannot be read'}"
        ) from None
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ConnectorError("invalid", f"{name}: not UTF-8 text") from None
    lines = [line for line in text.splitlines() if line.strip()]
    header = [cell.strip() for cell in lines[0].split("\t")] if lines else []
    for fd_name, dvars_name in (CONFOUND_COLUMNS, OLD_CONFOUND_COLUMNS):
        if fd_name in header and dvars_name in header:
            break
    else:
        lacking = [c for c in CONFOUND_COLUMNS if c not in header]
        raise ConnectorError("invalid", f"{name}: no column {' or '.join(lacking)}")
    fd_at, dvars_at = header.index(fd_name), header.index(dvars_name)
    flags = [
        i
        for i, column in enumerate(header)
        if column.startswith((NON_STEADY_PREFIX, OLD_NON_STEADY_PREFIX))
    ]
    fd: list[float | None] = []
    dvars: list[float | None] = []
    non_steady: list[int] = []
    for t, line in enumerate(lines[1:]):
        cells = line.split("\t")

        def at(i: int, cells: Sequence[str] = cells) -> float | None:
            return _cell(cells[i]) if i < len(cells) else None

        fd.append(at(fd_at))
        dvars.append(at(dvars_at))
        if any(at(i) == 1.0 for i in flags):
            non_steady.append(t)
    return Motion(
        fd=tuple(fd), dvars=tuple(dvars), non_steady=tuple(non_steady), source=FuncRole.CONFOUNDS
    )


# --- geometry of a stored transform ---------------------------------------------------------


def _singular_values(matrix: Matrix4) -> tuple[float, float, float]:
    """The singular values of the 3x3 part, from the eigenvalues of ``A^T A`` by the closed
    form for a symmetric 3x3 matrix, in Python floats."""
    a = [list(row[:3]) for row in matrix[:3]]
    b = [[sum(a[k][i] * a[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    p1 = b[0][1] ** 2 + b[0][2] ** 2 + b[1][2] ** 2
    q = (b[0][0] + b[1][1] + b[2][2]) / 3.0
    if p1 == 0.0:
        eig = [b[0][0], b[1][1], b[2][2]]
    else:
        p2 = (b[0][0] - q) ** 2 + (b[1][1] - q) ** 2 + (b[2][2] - q) ** 2 + 2.0 * p1
        p = math.sqrt(p2 / 6.0)
        c = [[(b[i][j] - (q if i == j else 0.0)) / p for j in range(3)] for i in range(3)]
        det = (
            c[0][0] * (c[1][1] * c[2][2] - c[1][2] * c[2][1])
            - c[0][1] * (c[1][0] * c[2][2] - c[1][2] * c[2][0])
            + c[0][2] * (c[1][0] * c[2][1] - c[1][1] * c[2][0])
        )
        r = min(1.0, max(-1.0, det / 2.0))
        phi = math.acos(r) / 3.0
        e1 = q + 2.0 * p * math.cos(phi)
        e3 = q + 2.0 * p * math.cos(phi + 2.0 * math.pi / 3.0)
        eig = [e1, 3.0 * q - e1 - e3, e3]
    s = sorted(math.sqrt(max(e, 0.0)) for e in eig)
    return (s[0], s[1], s[2])


def _determinant(matrix: Matrix4) -> float:
    a = matrix
    return (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )


def shear_scale(matrix: Matrix4) -> float:
    """The largest deviation of a singular value of the 3x3 part from 1; 0 for a rigid map."""
    return max(abs(s - 1.0) for s in _singular_values(matrix))


def _read_transform(path: Path) -> Matrix4:
    try:
        return read_itk_file(path)
    except ItkError as error:
        raise ConnectorError("invalid", str(error)) from None
    except OSError as error:
        raise ConnectorError("unreadable", str(error.strerror or error)) from None


# --- the listing ----------------------------------------------------------------------------


def _stem_of(name: str) -> str | None:
    """The run's stem of a file name in ``func/``: the leading parts that are run entities
    (``sub`` first), when a part follows them."""
    parts = name.split("_")
    n = 0
    for part in parts:
        key, sep, label = part.partition("-")
        if not sep or key not in _RUN_STEM_KEYS or "." in part or not label:
            break
        n += 1
    if n == 0 or n == len(parts) or not parts[0].startswith("sub-"):
        return None
    return "_".join(parts[:n])


def _rest_entities(rest: str) -> tuple[dict[str, str], str]:
    """The ``key-label`` parts and the suffix (with extension) of what follows a stem."""
    parts = rest.split("_")
    found: dict[str, str] = {}
    for part in parts[:-1]:
        key, sep, label = part.partition("-")
        if sep:
            found.setdefault(key, label)
    return found, parts[-1]


def _variant(suffix: str) -> str:
    return suffix.lstrip("_").split(".", 1)[0]


def _space_of(suffix: str) -> str | None:
    m = _SPACE.search(suffix)
    return m.group(1) if m else None


def _list(folder: Path, root: Path) -> tuple[dict[str, os.stat_result], list[str]]:
    try:
        return bids.list_dir(folder, root)
    except OSError as error:
        raise ConnectorError("unreadable", str(error.strerror or error)) from None


def _func_dirs(root: Path, participant: str) -> list[tuple[str | None, Path]]:
    try:
        return bids.func_dirs(root, participant)
    except OSError as error:
        raise ConnectorError("unreadable", str(error.strerror or error)) from None


def _ref(
    role: FuncRole, path: Path, root: Path, st: os.stat_result, variant: str | None
) -> FuncFileRef:
    return bids.file_ref(role, "deriv", path, root, st, variant)


class FmriprepConnector:
    """The connector of fMRIPrep derivatives. See ``func.connectors.base.Connector``."""

    name = NAME

    def detect(self, folder: Path) -> ConnectorMatch | None:
        """An fMRIPrep derivative: ``dataset_description.json`` names fMRIPrep in
        ``GeneratedBy`` (the first entry with that name) or, in older versions, in
        ``PipelineDescription`` (to verify on a real file of 20.2.x), and the folder holds
        ``sub-*/[ses-*/]func/``. The version is the one stated, as written; a version that
        does not parse leaves the generation to the listing (names that only the second
        generation writes). A description that cannot be read, also a git-annex placeholder,
        is no derivative here. ``ConnectorError("unreadable")`` for a folder that cannot be
        listed."""
        root = Path(folder)
        text = _read_small(root / "dataset_description.json", _DESCRIPTION_MAX_BYTES)
        if text is None:
            return None
        try:
            description = json.loads(text)
        except (ValueError, RecursionError):
            return None
        if not isinstance(description, dict):
            return None
        stated = _stated_version(description)
        if stated is False:
            return None
        version = stated if isinstance(stated, str) else None
        _, dirs = _list(root, root)
        places: list[Path] = []
        for participant in dirs:
            try:
                places.extend(func for _, func in bids.func_dirs(root, participant))
            except OSError:
                continue
        if not places:
            return None
        if _version_tuple(version) is not None:
            generation = generation_of(version)
        else:
            generation = SCANNER_T1W
            for func in places:
                try:
                    files, _ = bids.list_dir(func, root)
                except OSError:
                    continue
                if any(name.endswith(_BOLDREF_MARKS) for name in files):
                    generation = BOLDREF_T1W
                    break
        return ConnectorMatch(connector=NAME, pipeline_version=version, generation=generation)

    def list_runs(self, folder: Path, participant: str, match: ConnectorMatch) -> list[DerivRun]:
        """Every functional run of ``participant`` in ``sub-X/[ses-Y/]func/``, from the
        listing: the files of a run are those whose names start with its stem (the leading
        run entities, ``sub`` to ``run``) and follow it with a part of their own. A group of
        files is a run when one of them has a name this connector knows (a role's suffix or a
        preprocessed series in any space). ``files`` holds the anatomy-space series, its
        sidecar and mask in one space (the first of :data:`ANAT_SPACES` that has the series),
        the reference image, the transform, the confounds table, and the three figures of
        ``sub-X/figures/``; ``variant`` is set for a role with several candidate names.
        ``spaces`` lists the space of every ``desc-preproc_bold`` series (``""`` for the run's
        own frame), ``echoes`` the number of echo labels among the run's files (1 without
        any). Sorted as :func:`surfcensus.func.bids.list_raw_runs`."""
        root = Path(folder)
        generation = match.generation if match.generation in FUNC_SUFFIXES else SCANNER_T1W
        suffixes = FUNC_SUFFIXES[generation]
        places = _func_dirs(root, participant)
        if not places:
            return []
        figures, _ = _list(root / participant / "figures", root)
        known = tuple(s for role in suffixes.values() for s in role) + tuple(
            NATIVE_SUFFIXES.values()
        )
        runs: list[DerivRun] = []
        for session, func in places:
            files, _ = _list(func, root)
            groups: dict[str, list[str]] = {}
            for name in sorted(files):
                stem = _stem_of(name)
                if stem is None:
                    continue
                groups.setdefault(stem, []).append(name)
            for stem, names in groups.items():
                try:
                    entities = bids.parse_entities(stem)
                except ValueError:
                    continue
                if (
                    f"sub-{entities.sub}" != participant
                    or entities.ses != session
                    or entities.task is None
                ):
                    continue
                rests = [name[len(stem) + 1 :] for name in names]
                spaces: set[str] = set()
                echoes: set[str] = set()
                is_run = False
                for name, rest in zip(names, rests, strict=True):
                    found, suffix = _rest_entities(rest)
                    if "echo" in found:
                        echoes.add(found["echo"])
                    if found.get("desc") == "preproc" and suffix in ("bold.nii.gz", "bold.nii"):
                        spaces.add(found.get("space", ""))
                        is_run = True
                    if name.endswith(known):
                        is_run = True
                if not is_run:
                    continue
                refs: dict[FuncRole, FuncFileRef] = {}
                present = set(names)
                chosen = next(
                    (
                        _space_of(s)
                        for s in suffixes.get(FuncRole.PRE_BOLD, ())
                        if stem + s in present
                    ),
                    None,
                )
                for role, candidates in suffixes.items():
                    spaced = role in (FuncRole.PRE_BOLD, FuncRole.PRE_SIDECAR, FuncRole.PRE_MASK)
                    for suffix in candidates:
                        if spaced and (chosen is None or _space_of(suffix) != chosen):
                            continue
                        if stem + suffix in present:
                            variant = _variant(suffix) if len(candidates) > 1 else None
                            refs[role] = _ref(
                                role, func / (stem + suffix), root, files[stem + suffix], variant
                            )
                            break
                for role, suffix in FIGURE_SUFFIXES.items():
                    if stem + suffix in figures:
                        path = root / participant / "figures" / (stem + suffix)
                        refs[role] = _ref(role, path, root, figures[stem + suffix], None)
                runs.append(
                    DerivRun(
                        entities=entities,
                        key=bids.run_key(entities),
                        order=bids.name_order(entities),
                        stem=stem,
                        spaces=tuple(sorted(spaces)),
                        files=refs,
                        echoes=max(1, len(echoes)),
                    )
                )
        runs.sort(key=lambda r: (r.entities.ses is not None, r.entities.ses or "", r.order, r.stem))
        return runs

    def session_files(
        self, folder: Path, participant: str, session: str | None, match: ConnectorMatch
    ) -> Mapping[FuncRole, FuncFileRef]:
        """``xfm_anat_fsnative``: ``sub-X/ses-Y/anat/sub-X_ses-Y<FSNATIVE_SUFFIX>`` for a
        session, else ``sub-X/anat/sub-X<FSNATIVE_SUFFIX>``; empty when neither is listed."""
        root = Path(folder)
        if not bids.valid_participant(participant):
            return {}
        places = []
        if session is not None:
            places.append(
                (root / participant / f"ses-{session}" / "anat", f"{participant}_ses-{session}")
            )
        places.append((root / participant / "anat", participant))
        for anat, prefix in places:
            files, _ = _list(anat, root)
            name = prefix + FSNATIVE_SUFFIX
            if name in files:
                role = FuncRole.XFM_ANAT_FSNATIVE
                return {role: _ref(role, anat / name, root, files[name], None)}
        return {}

    def display(self, run: DerivRun) -> Display:
        """``anat`` with the series' space when the run has the anatomy-space series; else
        why not: ``multi-echo`` for more than one echo (first), ``no preprocessed series``
        without any series or with an anatomy-space series under another name (a ``res-``
        entity, say), ``native space only`` with a series in the run's own frame,
        ``standard space only`` otherwise."""
        if run.echoes > 1:
            return Display(None, None, MULTI_ECHO)
        pre = run.files.get(FuncRole.PRE_BOLD)
        if pre is not None:
            space = _space_of("_" + pre.location.rsplit("/", 1)[-1])
            return Display("anat", space or ANAT_SPACE, None)
        if not run.spaces or any(s in ANAT_SPACES for s in run.spaces):
            return Display(None, None, NO_PREPROCESSED_SERIES)
        if "" in run.spaces:
            return Display(None, None, NATIVE_SPACE_ONLY)
        return Display(None, None, STANDARD_SPACE_ONLY)

    def mask(self, run: DerivRun) -> FuncRole:
        """``pre_mask``: the brain mask that :meth:`list_runs` took in the series' space."""
        return FuncRole.PRE_MASK

    def raw_to_anat(self, run: DerivRun, files: Mapping[FuncRole, Path]) -> RigidPath:
        """The point map of ``from-scanner_to-T1w`` (or ``from-boldref_to-T1w``), which maps
        a point of the anatomy to the run's scanner (or boldref) world. ``dof`` from the
        summary report's registration line when ``files`` has the report. ``rigid`` when the
        shear and scale are within :data:`RIGID_TOLERANCE` and the map keeps handedness.
        ``ConnectorError``: ``invalid`` without the transform in ``files`` or for a file that
        is not one affine transform, ``unreadable`` for one that cannot be read."""
        path = files.get(FuncRole.XFM_TO_ANAT)
        if path is None:
            raise ConnectorError("invalid", f"{run.stem}: no transform to the anatomy was given")
        matrix = _read_transform(path)
        scale = shear_scale(matrix)
        dof = None
        report = files.get(FuncRole.SUMMARY_REPORT)
        if report is not None:
            text = _read_small(report, REPORT_MAX_BYTES)
            if text is not None:
                dof = read_summary_report(text, None).dof
        return RigidPath(
            matrix=matrix,
            dof=dof,
            shear_scale=scale,
            rigid=scale <= RIGID_TOLERANCE and _determinant(matrix) > 0.0,
            source=FuncRole.XFM_TO_ANAT,
            reference=FuncRole.REFERENCE if FuncRole.REFERENCE in run.files else None,
        )

    def anat_to_fsnative(self, files: Mapping[FuncRole, Path]) -> Matrix4:
        """The point map of ``from-T1w_to-fsnative``: FreeSurfer's scanner RAS to the anatomy.
        ``ConnectorError`` as for :meth:`raw_to_anat`."""
        path = files.get(FuncRole.XFM_ANAT_FSNATIVE)
        if path is None:
            raise ConnectorError("invalid", "no transform from fsnative to the anatomy was given")
        return _read_transform(path)

    def motion(self, run: DerivRun, files: Mapping[FuncRole, Path]) -> Motion | None:
        """:func:`read_confounds` of the run's table; None when ``files`` has none."""
        path = files.get(FuncRole.CONFOUNDS)
        return None if path is None else read_confounds(path)

    def corrections(self, run: DerivRun, files: Mapping[FuncRole, Path]) -> Corrections:
        """:func:`read_summary_report` of the report in ``files``; without one, or for one
        that cannot be read, every statement is ``unknown`` with a note, but for a
        distortion figure in the listing, which says that distortion correction ran."""
        listed = FuncRole.SDC_FIGURE in run.files
        path = files.get(FuncRole.SUMMARY_REPORT)
        if path is None:
            return _without_report("no summary report of the run", listed)
        text = _read_small(path, REPORT_MAX_BYTES)
        if text is None:
            return _without_report("the summary report of the run cannot be read", listed)
        return read_summary_report(text, listed)

    def repetition_time(self, run: DerivRun, files: Mapping[FuncRole, Path]) -> float | None:
        """``RepetitionTime`` of the preprocessed series' sidecar, else the fourth zoom of its
        header (in seconds, from the header's time unit); None when neither is a positive
        finite number."""
        sidecar = files.get(FuncRole.PRE_SIDECAR)
        if sidecar is not None:
            text = _read_small(sidecar, _DESCRIPTION_MAX_BYTES)
            value = None
            if text is not None:
                try:
                    data = json.loads(text)
                except (ValueError, RecursionError):
                    data = None
                if isinstance(data, dict):
                    value = data.get("RepetitionTime")
            if (
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value > 0
            ):
                return float(value)
        series = files.get(FuncRole.PRE_BOLD)
        if series is None:
            return None
        return _header_tr(series)


def _header_tr(path: Path) -> float | None:
    try:
        import nibabel as nib

        header = nib.load(path).header
        zooms = header.get_zooms()
        if len(zooms) < 4:
            return None
        value = float(zooms[3])
        unit = header.get_xyzt_units()[1] if hasattr(header, "get_xyzt_units") else "sec"
    except Exception:
        return None
    value *= {"msec": 1e-3, "usec": 1e-6}.get(str(unit), 1.0)
    return value if math.isfinite(value) and value > 0 else None


def _stated_version(description: Mapping[str, object]) -> str | bool | None:
    """The version fMRIPrep states in a dataset description; None when it names fMRIPrep
    without a usable version; False when it does not name fMRIPrep."""
    entries: list[object] = []
    generated = description.get("GeneratedBy")
    if isinstance(generated, list):
        entries.extend(generated)
    entries.append(description.get("PipelineDescription"))
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("Name")
        if isinstance(name, str) and name.strip().casefold() == NAME:
            version = entry.get("Version")
            if isinstance(version, str) and 0 < len(version.strip()) <= 64:
                text = version.strip()
                return text if text.isprintable() else None
            return None
    return False


def make() -> FmriprepConnector:
    """The connector, for ``func.connectors.connector("fmriprep")``."""
    return FmriprepConnector()
