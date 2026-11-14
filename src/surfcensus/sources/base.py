"""What every source shares: file roles and tiers, the records a source yields, the protocol.

A source lists datasets and subjects and hands out local paths; everything downstream of it
(inventory, metrics, rendering, rating, export) is the same for local and remote subjects
(docs/architecture.md). This module imports the standard library only.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from surfcensus.config import SourceConfig

Hemi = Literal["lh", "rh"]
HEMIS: tuple[Hemi, Hemi] = ("lh", "rh")
"""The two hemispheres, always in this order."""


class FileRole(StrEnum):
    """A file inside a FreeSurfer subject directory, named by what it is used for.

    The values are the role names of the ``files`` table in docs/architecture.md.
    """

    T1 = "t1"
    BRAINMASK = "brainmask"
    LH_WHITE = "lh_white"
    RH_WHITE = "rh_white"
    LH_PIAL = "lh_pial"
    RH_PIAL = "rh_pial"
    LH_THICKNESS = "lh_thickness"
    RH_THICKNESS = "rh_thickness"
    LH_SPHERE_REG = "lh_sphere_reg"
    RH_SPHERE_REG = "rh_sphere_reg"
    LH_ANNOT = "lh_annot"
    RH_ANNOT = "rh_annot"
    LH_STATS = "lh_stats"
    RH_STATS = "rh_stats"
    ASEG_STATS = "aseg_stats"
    TALAIRACH = "talairach"
    RECON_LOG = "recon_log"
    BUILD_STAMP = "build_stamp"
    # the per-hemisphere logs that recon-all writes under fMRIPrep; the thickness truncation
    # counts are in these and not in recon-all.log (seen on ds000102 sub-01, FreeSurfer 6.0.1)
    RECON_LOG_LH = "recon_log_lh"
    RECON_LOG_RH = "recon_log_rh"


RENDER_ROLES: tuple[FileRole, ...] = (
    FileRole.T1,
    FileRole.LH_WHITE,
    FileRole.RH_WHITE,
    FileRole.LH_PIAL,
    FileRole.RH_PIAL,
)
"""Roles a render cannot do without. ``brainmask`` is used when present."""

TIER_A: tuple[FileRole, ...] = (
    FileRole.LH_STATS,
    FileRole.RH_STATS,
    FileRole.ASEG_STATS,
    FileRole.BUILD_STAMP,
    FileRole.RECON_LOG,
    FileRole.RECON_LOG_LH,
    FileRole.RECON_LOG_RH,
)
"""Small text files: regional metrics and provenance (docs/data_sources.md, fetch tiers)."""

TIER_B: tuple[FileRole, ...] = (
    FileRole.LH_THICKNESS,
    FileRole.RH_THICKNESS,
    FileRole.LH_SPHERE_REG,
    FileRole.RH_SPHERE_REG,
    FileRole.LH_ANNOT,
    FileRole.RH_ANNOT,
)
"""What the fsaverage maps are computed from."""

TIER_C: tuple[FileRole, ...] = (*RENDER_ROLES, FileRole.BRAINMASK)
"""What a render reads."""

TIER_A_REQUIRED: tuple[FileRole, ...] = (FileRole.LH_STATS, FileRole.RH_STATS)
TIER_B_REQUIRED: tuple[FileRole, ...] = (
    FileRole.LH_THICKNESS,
    FileRole.RH_THICKNESS,
    FileRole.LH_SPHERE_REG,
    FileRole.RH_SPHERE_REG,
)
"""The annotation is used when present."""


class FuncRole(StrEnum):
    """A file of a functional run, or of the session it belongs to (docs/epi_interface.md,
    "File roles of a functional run").

    The FreeSurfer files that a pack reads stay roles of the FreeSurfer subject
    (:class:`FileRole`). ``raw_*`` roles lie in the raw BIDS dataset, the others in the
    derivative; :data:`FUNC_RAW_ROLES` says which.
    """

    RAW_BOLD = "raw_bold"
    RAW_SIDECAR = "raw_sidecar"
    """The JSON sidecar of the raw series, beside it or inherited from a higher level."""
    PRE_BOLD = "pre_bold"
    """The preprocessed series of the display layout the connector chose."""
    PRE_SIDECAR = "pre_sidecar"
    PRE_MASK = "pre_mask"
    """The brain mask on the grid of ``pre_bold``."""
    XFM_TO_ANAT = "xfm_to_anat"
    """The stored transform between the run's scanner (or boldref) frame and the anatomy."""
    REFERENCE = "reference"
    """The BOLD-space reference image that arbitrates the raw affine."""
    CONFOUNDS = "confounds"
    SUMMARY_REPORT = "summary_report"
    VALIDATION_REPORT = "validation_report"
    SDC_FIGURE = "sdc_figure"
    """Listed, never read: whether it is there is the cross-check of the report."""
    XFM_ANAT_FSNATIVE = "xfm_anat_fsnative"
    """Per session: the transform between the anatomy and FreeSurfer's scanner RAS."""


FUNC_RAW_ROLES: tuple[FuncRole, ...] = (FuncRole.RAW_BOLD, FuncRole.RAW_SIDECAR)
"""The roles whose files lie in the raw BIDS dataset; every other role lies in the derivative."""

FUNC_SESSION_ROLES: tuple[FuncRole, ...] = (FuncRole.XFM_ANAT_FSNATIVE,)
"""Roles of a session, not of a run."""

PACK_REQUIRED: tuple[FuncRole, ...] = (
    FuncRole.RAW_BOLD,
    FuncRole.PRE_BOLD,
    FuncRole.PRE_MASK,
    FuncRole.XFM_TO_ANAT,
    FuncRole.CONFOUNDS,
)
"""Roles of a run that a pack cannot do without. ``reference`` is used when present (without
it the raw affine is unchecked); the sidecars and the reports are read when present."""

FUNC_READ_ROLES: tuple[FuncRole, ...] = tuple(
    role for role in FuncRole if role not in (FuncRole.SDC_FIGURE, FuncRole.XFM_ANAT_FSNATIVE)
)
"""Roles of a run whose files a pack reads when they are there (every role but the listed-only
figure), in definition order. The session's ``xfm_anat_fsnative`` is read by the contours."""

SourceKind = Literal["local", "openneuro-derivatives", "openneuro-embedded"]
Layout = Literal["subjects_dir", "fmriprep", "fmriprep_legacy", "bids_derivatives"]


@dataclass(frozen=True)
class FileRef:
    """One file of a subject as the source's listing describes it. Nothing was opened."""

    role: FileRole
    location: str
    """POSIX path relative to the subject directory, for example ``surf/lh.pial.T1``."""
    size: int
    etag_or_mtime: str
    """Remote: the ETag without quotes. Local: ``st_mtime_ns`` in decimal."""
    variant: str | None = None
    """The file name read when the role has several candidates or is a link inside its own
    directory (``lh.pial.T1``); always one of the role's candidate names."""
    last_modified: str | None = None
    """Remote: ``LastModified`` of the listing, ISO 8601."""


UnfetchedKind = Literal["link", "pointer"]

UNFETCHED = "not fetched"
"""The reason of a subject's ``problem`` when files it lists have no content: the
placeholders of git-annex (a link into ``.git/annex/objects`` whose target is not there,
or a pointer file). Unlike ``unreadable`` it is no failure of the inventory: the subject is
listed with the files it has, and lacks the others."""


@dataclass(frozen=True)
class Unfetched:
    """A file of a subject whose name is there and whose content is not (git-annex)."""

    role: FileRole
    location: str
    """The candidate name of the role, relative to the subject directory (``surf/lh.pial``)."""
    fetch: str
    """What a fetch is asked for, relative to the subject directory: the file that the link
    leads to when that is another candidate name in the subject directory
    (``surf/lh.pial.T1`` for FreeSurfer 7's ``lh.pial -> lh.pial.T1``), else ``location``."""
    size: int | None
    """Bytes, from the ``-s`` field of the git-annex key; None when the key has none."""
    kind: UnfetchedKind
    """``link``: a link into the object store; ``pointer``: a pointer file."""


@dataclass(frozen=True)
class DatasetRecord:
    source: SourceKind
    identity: str
    """What is hashed into the pseudonym and every ``subject_uid``: the accession of a remote
    dataset; for a local one the resolved root at first inventory, kept when the data moves."""
    accession: str | None
    """None for a local dataset."""
    root: str
    """Local: absolute path. Remote: ``s3://bucket/prefix/``."""
    fs_root: str
    """The subjects directory relative to ``root``: ``sourcedata/freesurfer``, ``freesurfer``
    or ``.``."""
    layout: Layout
    pipeline: str
    """``fmriprep`` or ``freesurfer``."""
    pipeline_version: str | None = None
    template_location: str | None = None
    """An fsaverage directory next to the subjects, relative to ``root``."""
    derivative_ref: str | None = None
    snapshot_tag: str | None = None


@dataclass(frozen=True)
class SubjectRecord:
    dataset: DatasetRecord
    subject_label: str
    """The directory name."""
    fs_dir: str
    """POSIX, relative to ``dataset.root``."""
    files: Mapping[FileRole, FileRef]
    """One file per role the subject has, keyed in :class:`FileRole` definition order."""
    problem: str | None = None
    """Why the directory could not be listed, for example ``unreadable: Permission denied``;
    ``files`` is then empty. In a record from the catalog it can also be a ``not fetched: ...``
    sentence (:data:`UNFETCHED`) that the inventory wrote: the files named there are listed
    by name and have no content, and ``files`` holds the others."""
    unfetched: Mapping[FileRole, Unfetched] = field(default_factory=dict, compare=False)
    """Roles whose first candidate with a name is a git-annex link to content that is not
    there; such a role is not in ``files``. A source finds these from the listing alone
    (``readlink``, no file opened); pointer files, which are regular files, are recognised
    by the inventory (``patterns.annex_pointer``). Not stored as such: the inventory writes
    them into ``problem``."""


class SourceError(Exception):
    """A source cannot do what was asked.

    ``reason`` is one of ``layout``, ``not found``, ``changed``, ``unreadable``,
    ``unreachable``, ``invalid``; ``detail`` says what and where.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail

    def __reduce__(self) -> tuple[type[SourceError], tuple[str, str]]:
        # pickle would call __init__ with the one argument of Exception.args; workers of a
        # process pool send their exceptions to the parent pickled
        return type(self), (self.reason, self.detail)


@runtime_checkable
class Source(Protocol):
    """Where subjects come from. Downstream code sees local paths only."""

    kind: SourceKind

    def datasets(self, only: Collection[str] | None = None) -> Iterator[DatasetRecord]:
        """The datasets of the source: exactly one for a local source; for a remote one, one
        per dataset prefix, restricted to the accessions in ``only`` when given."""
        ...

    def subjects(self, dataset: DatasetRecord) -> Iterator[SubjectRecord]:
        """The subjects of ``dataset`` in ascending ``subject_label`` order. Lists and stats
        only, opens nothing. One file per role: the first existing candidate."""
        ...

    def read_bytes(
        self,
        subject: SubjectRecord,
        role: FileRole,
        *,
        byte_range: tuple[int, int] | None = None,
    ) -> bytes:
        """The content of a Tier A file, or its inclusive ``(first, last)`` byte range.

        Raises ``SourceError("changed", ...)`` when size or ``etag_or_mtime`` differ from
        the subject's :class:`FileRef`."""
        ...

    def lease(
        self, subject: SubjectRecord, roles: Collection[FileRole]
    ) -> AbstractContextManager[Mapping[FileRole, Path]]:
        """Local paths of the requested roles that the subject has, valid while the context
        is open. Absent roles are absent from the mapping. Local: the files in place.
        Raises ``SourceError("changed", ...)`` like :meth:`read_bytes`."""
        ...

    def template_dir(self, dataset: DatasetRecord) -> Path | None:
        """A local directory holding the ``fsaverage`` next to the subjects, or None."""
        ...

    def config(self) -> SourceConfig:
        """The stored description from which ``run.from_config`` builds an equal source."""
        ...


def input_signature(
    files: Mapping[FileRole, FileRef],
    roles: Collection[FileRole] | None = None,
    extra: Sequence[str] = (),
) -> str:
    """What a stage's skip test compares: 16 hex characters that change when a listed file does.

    The SHA-256 of one line ``role<TAB>location<TAB>size<TAB>etag_or_mtime`` per requested
    role that the subject has (every role it has when ``roles`` is None), in
    :class:`FileRole` definition order, followed by one line per ``extra`` string. A role
    with a ``variant`` adds ``<TAB>variant`` to its line. It is built from the listing
    alone, so deciding to skip reads no source file.

    For a remote subject this is a function of the public bucket listing: it is stored in
    ``catalog.sqlite`` and written nowhere else.
    """

    def line(role: FileRole, ref: FileRef) -> str:
        fields = [role.value, ref.location, str(ref.size), ref.etag_or_mtime]
        if ref.variant is not None:
            # a link keeps its location when it is pointed at another variant, and the pial
            # variants of one subject have one size: with an equal modification time (a
            # copy that keeps it) nothing else in the line would differ
            fields.append(ref.variant)
        return "\t".join(fields)

    lines = [
        line(role, files[role])
        for role in FileRole
        if role in files and (roles is None or role in roles)
    ]
    lines.extend(extra)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]
