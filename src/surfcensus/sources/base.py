"""What every source shares: file roles and tiers, the records a source yields, the protocol.

A source lists datasets and subjects and hands out local paths; everything downstream of it
(inventory, metrics, rendering, rating, export) is the same for local and remote subjects
(docs/architecture.md). This module imports the standard library only.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
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
    ``files`` is then empty."""


class SourceError(Exception):
    """A source cannot do what was asked.

    ``reason`` is one of ``layout``, ``not found``, ``changed``, ``unreadable``,
    ``unreachable``; ``detail`` says what and where.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


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
    :class:`FileRole` definition order, followed by one line per ``extra`` string. It is
    built from the listing alone, so deciding to skip reads no source file.

    For a remote subject this is a function of the public bucket listing: it is stored in
    ``catalog.sqlite`` and written nowhere else.
    """
    lines = [
        f"{role.value}\t{files[role].location}\t{files[role].size}\t{files[role].etag_or_mtime}"
        for role in FileRole
        if role in files and (roles is None or role in roles)
    ]
    lines.extend(extra)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]
