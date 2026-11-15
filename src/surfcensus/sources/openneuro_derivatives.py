"""The OpenNeuroDerivatives bucket as a source (work package RE).

Laid down by work package R0 (docs/dev/remote_plan.md): the constructor, how a run binds the
source to its cache and pseudonym, and ``config()``, which is what ``run.from_config`` reads
back. The listing, ``read_bytes``, ``lease`` and ``template_dir`` are written by RE on top of
``remote/s3.py`` and ``remote/cache.py``.

Two forms (decision D47, item 1):

* One dataset, for a local run (``review openneuro:dsXXXXXX``): ``accession`` and the subject
  globs are given. ``config()`` is ``SourceConfig(kind, bucket, prefix="fmriprep/",
  dataset=<pseudonym>, subjects=<globs>)``: the accession is in ``catalog.sqlite`` only, so
  the source must have been bound to its pseudonym by the run.
* Every dataset of the prefix, for a census run: ``config()`` is ``SourceConfig(kind, bucket,
  prefix="fmriprep/")``.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING

from surfcensus.config import ConfigError, SourceConfig
from surfcensus.remote import FMRIPREP_PREFIX, dataset_prefix
from surfcensus.sources.base import (
    DatasetRecord,
    FileRole,
    SourceKind,
    SubjectRecord,
)

if TYPE_CHECKING:
    from surfcensus.remote.cache import Cache
    from surfcensus.remote.s3 import S3Client

_RE = "work package RE: sources/openneuro_derivatives.py"

DATASET_PREFIX = re.compile(r"(ds[0-9]{6})-fmriprep/")
"""A common prefix of ``fmriprep/`` that is a dataset, relative to ``fmriprep/``."""

FS_ROOTS: tuple[str, ...] = ("sourcedata/freesurfer", "freesurfer")
"""Where an fMRIPrep derivative holds its FreeSurfer subjects, first match first
(docs/data_sources.md). ``freesurfer`` is the layout of older fMRIPrep versions; whether the
bucket has such datasets has not been looked at."""

_ACCESSION = re.compile(r"ds[0-9]{6}")


def accession_of(common_prefix: str, prefix: str = FMRIPREP_PREFIX) -> str | None:
    """The accession of a common prefix ``fmriprep/dsXXXXXX-fmriprep/`` of a listing, or None
    for any other prefix."""
    if not common_prefix.startswith(prefix):
        return None
    found = DATASET_PREFIX.fullmatch(common_prefix[len(prefix) :])
    return found.group(1) if found else None


def dataset_root(bucket: str, accession: str, prefix: str = FMRIPREP_PREFIX) -> str:
    """``s3://<bucket>/fmriprep/dsXXXXXX-fmriprep/``: ``DatasetRecord.root`` of a dataset,
    which is stored in ``catalog.sqlite`` only."""
    return f"s3://{bucket}/{dataset_prefix(accession, prefix)}"


class OpenNeuroDerivatives:
    """Datasets of the OpenNeuroDerivatives bucket, read anonymously and read-only.

    ``prefix`` is the bucket's prefix of fMRIPrep derivatives and names no dataset.
    ``accession`` narrows the source to one dataset (a local run), ``subjects`` are globs on
    its subject labels (a union; empty means all) and ``pseudonym`` its stored name, given
    by ``run.from_config`` or by :meth:`bind`. ``cache`` is the run's cache, given by the
    run; listing needs none, a lease does.
    """

    kind: SourceKind = "openneuro-derivatives"

    def __init__(
        self,
        *,
        client: S3Client,
        cache: Cache | None = None,
        prefix: str = FMRIPREP_PREFIX,
        accession: str | None = None,
        subjects: Sequence[str] = (),
        pseudonym: str | None = None,
    ) -> None:
        if not prefix.endswith("/") or _ACCESSION.search(prefix):
            raise ValueError(f"the prefix of a remote source names no dataset: {prefix!r}")
        if accession is not None and not _ACCESSION.fullmatch(accession):
            raise ValueError(f"an accession is ds and six digits, not {accession!r}")
        given = (subjects,) if isinstance(subjects, str) else tuple(subjects)
        if accession is None and (given or pseudonym is not None):
            raise ValueError("subject globs and a pseudonym are for a source of one dataset")
        self.client = client
        self.cache = cache
        self.prefix = prefix
        self.accession = accession
        self.globs: tuple[str, ...] = tuple(dict.fromkeys(given))
        self.pseudonym = pseudonym

    @property
    def bucket(self) -> str:
        return self.client.bucket

    def bind(self, *, cache: Cache, pseudonym: str | None = None) -> None:
        """Called by the run that takes the source: its cache and, for one dataset, its
        pseudonym. A source that holds another pseudonym already raises
        ``ConfigError("invalid", ...)``: it belongs to another run."""
        if pseudonym is not None:
            if self.accession is None:
                raise ValueError("a source of every dataset of its prefix has no pseudonym")
            if self.pseudonym is not None and self.pseudonym != pseudonym:
                raise ConfigError("invalid", "this source was taken by a run of another dataset")
            self.pseudonym = pseudonym
        self.cache = cache

    def config(self) -> SourceConfig:
        """What a run stores of the source. One dataset: kind, bucket, prefix, the pseudonym
        and the globs, never the accession; raises ``ConfigError("invalid", ...)`` before the
        source was bound to its pseudonym. Every dataset: kind, bucket and prefix."""
        if self.accession is None:
            return SourceConfig(kind=self.kind, bucket=self.bucket, prefix=self.prefix)
        if self.pseudonym is None:
            raise ConfigError(
                "invalid",
                "a source of one dataset is stored by its pseudonym, which the run that takes "
                "it gives it",
            )
        return SourceConfig(
            kind=self.kind,
            bucket=self.bucket,
            prefix=self.prefix,
            dataset=self.pseudonym,
            subjects=self.globs,
        )

    # --- work package RE ---------------------------------------------------------------------

    def datasets(self, only: Collection[str] | None = None) -> Iterator[DatasetRecord]:
        """One dataset: its record (``source``, ``identity`` and ``accession`` the accession,
        ``root`` :func:`dataset_root`, ``fs_root`` the first of :data:`FS_ROOTS` with a
        subject prefix, layout and pipeline as the local rules give them, ``pipeline_version``
        from ``dataset_description.json``, ``template_location`` when ``fs_root/fsaverage/``
        lists the template files). Every dataset: one record per common prefix of
        ``fmriprep/`` that :func:`accession_of` reads, restricted to ``only``. The listing is
        kept for the life of the source, so the inventory does not list twice. A listing
        that cannot be had raises ``SourceError`` as ``S3Client.list`` does."""
        raise NotImplementedError(_RE)

    def subjects(self, dataset: DatasetRecord) -> Iterator[SubjectRecord]:
        """The subjects of ``dataset`` in label order: one listing per subject prefix, roles
        by ``patterns.roles_from_keys`` and candidates by ``patterns.is_subject_candidate_keys``
        (the rules of a local directory), ``FileRef`` with the ETag without quotes and
        ``last_modified``; templates skipped; the globs applied, a glob that matches nothing
        raising ``SourceError("not found", ...)`` with at most 10 names. Never opens a file."""
        raise NotImplementedError(_RE)

    def read_bytes(
        self,
        subject: SubjectRecord,
        role: FileRole,
        *,
        byte_range: tuple[int, int] | None = None,
    ) -> bytes:
        """A Tier A file, whole or its inclusive range (the end clamped to the size, an empty
        range asks for nothing), through the retained cache, with ``If-Match`` on every GET.
        Raises ``SourceError`` as ``LocalSource.read_bytes`` does; a changed ETag is
        ``changed``."""
        raise NotImplementedError(_RE)

    def lease(
        self, subject: SubjectRecord, roles: Collection[FileRole]
    ) -> AbstractContextManager[Mapping[FileRole, Path]]:
        """``Cache.lease`` of the requested roles that the subject has, keyed by role. Raises
        ``SourceError`` as the cache and the client do, and ``ConfigError`` before the source
        was bound to a cache."""
        raise NotImplementedError(_RE)

    def template_dir(self, dataset: DatasetRecord) -> Path | None:
        """The ``fsaverage`` next to the subjects downloaded into the retained cache with the
        FreeSurfer subpaths (the eight files of ``patterns.TEMPLATE_FILES``, the optional
        ones, ``scripts/build-stamp.txt``); None when one of the eight is not listed."""
        raise NotImplementedError(_RE)
