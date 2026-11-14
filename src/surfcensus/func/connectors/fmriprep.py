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
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from surfcensus.records import (
    ConnectorMatch,
    Corrections,
    DerivRun,
    Display,
    FuncFileRef,
    Generation,
    Matrix4,
    Motion,
    RigidPath,
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
generation is to verify (docs/epi_interface.md, What could not be verified)."""

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


def generation_of(version: str | None) -> Generation:
    """The generation of a stated fMRIPrep version: ``boldref-T1w`` from 23.2 on, else
    ``scanner-T1w`` (also for a version that does not parse, which is then decided by the
    listing in ``detect``)."""
    raise NotImplementedError("work package E1.A: func/connectors/fmriprep.py")


def read_summary_report(text: str, sdc_figure_listed: bool | None) -> Corrections:
    """The corrections that a summary report states, each with the report's words and the
    role ``summary_report`` as its source; ``unknown`` for a line that is not there."""
    raise NotImplementedError("work package E1.A: func/connectors/fmriprep.py")


def read_confounds(path: Path) -> Motion:
    """Framewise displacement, standardised DVARS and the non-steady-state volumes of a
    confounds table (tab-separated, ``n/a`` for no number). Raises ``ConnectorError`` with
    ``invalid`` for a table without the two columns."""
    raise NotImplementedError("work package E1.A: func/connectors/fmriprep.py")


class FmriprepConnector:
    """The connector of fMRIPrep derivatives. See ``func.connectors.base.Connector``."""

    name = NAME

    def detect(self, folder: Path) -> ConnectorMatch | None:
        raise NotImplementedError("work package E1.A: func/connectors/fmriprep.py")

    def list_runs(self, folder: Path, participant: str, match: ConnectorMatch) -> list[DerivRun]:
        raise NotImplementedError("work package E1.A: func/connectors/fmriprep.py")

    def session_files(
        self, folder: Path, participant: str, session: str | None, match: ConnectorMatch
    ) -> Mapping[FuncRole, FuncFileRef]:
        raise NotImplementedError("work package E1.A: func/connectors/fmriprep.py")

    def display(self, run: DerivRun) -> Display:
        raise NotImplementedError("work package E1.A: func/connectors/fmriprep.py")

    def mask(self, run: DerivRun) -> FuncRole:
        raise NotImplementedError("work package E1.A: func/connectors/fmriprep.py")

    def raw_to_anat(self, run: DerivRun, files: Mapping[FuncRole, Path]) -> RigidPath:
        raise NotImplementedError("work package E1.A: func/connectors/fmriprep.py")

    def anat_to_fsnative(self, files: Mapping[FuncRole, Path]) -> Matrix4:
        raise NotImplementedError("work package E1.A: func/connectors/fmriprep.py")

    def motion(self, run: DerivRun, files: Mapping[FuncRole, Path]) -> Motion | None:
        raise NotImplementedError("work package E1.A: func/connectors/fmriprep.py")

    def corrections(self, run: DerivRun, files: Mapping[FuncRole, Path]) -> Corrections:
        raise NotImplementedError("work package E1.A: func/connectors/fmriprep.py")

    def repetition_time(self, run: DerivRun, files: Mapping[FuncRole, Path]) -> float | None:
        raise NotImplementedError("work package E1.A: func/connectors/fmriprep.py")


def make() -> FmriprepConnector:
    """The connector, for ``func.connectors.connector("fmriprep")``."""
    return FmriprepConnector()
