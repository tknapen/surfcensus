"""The connector protocol: everything that is specific to one preprocessing pipeline
(docs/epi_interface.md, The connector; open question O29).

A connector says whether a folder is a derivative of its pipeline, lists the functional runs
of a participant with the files of every role, chooses the display grid, and reads the stored
transforms, the motion estimates and the statements about the corrections. It reads text and
headers only and never resamples; the rest of the package (session, geometry, pack, page)
sees only its answers, the records of ``surfcensus.records``.

Only the fMRIPrep connector is built (``func.connectors.fmriprep``). This module imports the
standard library, numpy for the type checker and ``records``/``sources.base`` only.

``files`` in the methods below is a mapping of roles to local paths: the files in place in
step 1, a lease in step 2 (``lease(run_uid, roles)``, as for renders). A connector never
opens a file that is not in it.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

from surfcensus.records import (
    ConnectorMatch,
    Corrections,
    DerivRun,
    Display,
    FuncFileRef,
    Matrix4,
    Motion,
    RigidPath,
)
from surfcensus.sources.base import FuncRole

CONNECTOR_REASONS: tuple[str, ...] = ("not a derivative", "unreadable", "invalid")


class ConnectorError(Exception):
    """A connector cannot read what it was given.

    ``reason`` is one of :data:`CONNECTOR_REASONS`: ``not a derivative`` (the folder is not
    one of this pipeline), ``unreadable`` (a file cannot be opened), ``invalid`` (a file is
    not what its name says: a transform that is not one affine, a table without the columns,
    a header that does not parse). ``detail`` names the file by its name, never by its path.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail

    def __reduce__(self) -> tuple[type[ConnectorError], tuple[str, str]]:
        # pickle would call __init__ with the one argument of Exception.args
        return type(self), (self.reason, self.detail)


@runtime_checkable
class Connector(Protocol):
    """One preprocessing pipeline's way of writing functional derivatives."""

    name: str
    """The connector's name, as ``func_sources.connector`` stores it: ``fmriprep``."""

    def detect(self, folder: Path) -> ConnectorMatch | None:
        """Whether ``folder`` is a derivative of this pipeline, with the version it states
        and the generation of its layout; None when it is not. Reads
        ``dataset_description.json`` and lists the folder; opens nothing else."""
        ...

    def list_runs(self, folder: Path, participant: str, match: ConnectorMatch) -> list[DerivRun]:
        """The functional runs of ``participant`` (``sub-01``) in the derivative, from the
        listing alone, in ``func.bids.name_order``. Every run of the derivative is listed,
        also one without a preprocessed series."""
        ...

    def session_files(
        self, folder: Path, participant: str, session: str | None, match: ConnectorMatch
    ) -> Mapping[FuncRole, FuncFileRef]:
        """The files of a session, from the listing: ``xfm_anat_fsnative`` from the session
        folder's ``anat/`` before the participant's (docs/epi_interface.md, The unit)."""
        ...

    def display(self, run: DerivRun) -> Display:
        """Which preprocessed series of the run is its display grid, or why there is none."""
        ...

    def mask(self, run: DerivRun) -> FuncRole:
        """The role of the brain mask on the display grid."""
        ...

    def raw_to_anat(self, run: DerivRun, files: Mapping[FuncRole, Path]) -> RigidPath:
        """The run's stored transform as a point map from the anatomy's world to the scanner
        world of its reference image (RAS mm), with its degrees of freedom when a file states
        them, and the role of the reference image that arbitrates the raw affine."""
        ...

    def anat_to_fsnative(self, files: Mapping[FuncRole, Path]) -> Matrix4:
        """The point map that takes FreeSurfer's scanner RAS to the anatomy's world, from the
        session's ``xfm_anat_fsnative`` (fMRIPrep: the file ``from-T1w_to-fsnative``)."""
        ...

    def motion(self, run: DerivRun, files: Mapping[FuncRole, Path]) -> Motion | None:
        """Framewise displacement, DVARS and the non-steady-state volumes, or None when the
        pipeline did not estimate them."""
        ...

    def corrections(self, run: DerivRun, files: Mapping[FuncRole, Path]) -> Corrections:
        """What the pipeline did to the preprocessed series, each statement with its source.
        A run without a report gets ``unknown`` statements, never an error."""
        ...

    def repetition_time(self, run: DerivRun, files: Mapping[FuncRole, Path]) -> float | None:
        """TR in seconds: from the preprocessed series' sidecar, else its header; None when
        neither gives a positive number."""
        ...
