"""The OpenNeuroDerivatives bucket as a source (docs/dev/remote_plan.md).

The constructor, how a run binds the source to its cache and pseudonym, ``config()``, which
is what ``run.from_config`` reads back, and the listing, ``read_bytes``, ``lease`` and
``template_dir`` on top of ``remote/s3.py`` and ``remote/cache.py``.

Two forms (decision D47, item 1):

* One dataset, for a local run (``review openneuro:dsXXXXXX``): ``accession`` and the subject
  globs are given. ``config()`` is ``SourceConfig(kind, bucket, prefix="fmriprep/",
  dataset=<pseudonym>, subjects=<globs>)``: the accession is in ``catalog.sqlite`` only, so
  the source must have been bound to its pseudonym by the run.
* Every dataset of the prefix, for a census run: ``config()`` is ``SourceConfig(kind, bucket,
  prefix="fmriprep/")``.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import secrets
import shutil
import threading
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal, TypeVar

from surfcensus.config import ConfigError, SourceConfig
from surfcensus.log import get_logger
from surfcensus.remote import FMRIPREP_PREFIX, UNREACHABLE, dataset_prefix, raw_prefix
from surfcensus.sources.base import (
    HEMIS,
    TIER_A,
    DatasetRecord,
    FileRef,
    FileRole,
    Layout,
    SourceError,
    SourceKind,
    SubjectRecord,
)
from surfcensus.sources.patterns import (
    TEMPLATE_FILES,
    TEMPLATE_OPTIONAL_FILES,
    is_subject_candidate_keys,
    is_template_name,
    key_variant,
    parse_pipeline_version,
    roles_from_keys,
)

if TYPE_CHECKING:
    from surfcensus.remote.cache import Cache
    from surfcensus.remote.s3 import S3Client, S3Object

_T = TypeVar("_T")
_U = TypeVar("_U")

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
    _memo_state: _Memo | None = None
    """The listings of this source, made on first use (:meth:`_memo`)."""

    def __init__(
        self,
        *,
        client: S3Client,
        cache: Cache | None = None,
        prefix: str = FMRIPREP_PREFIX,
        accession: str | None = None,
        subjects: Sequence[str] = (),
        pseudonym: str | None = None,
        raw_client: S3Client | None = None,
    ) -> None:
        if not prefix.endswith("/") or _ACCESSION.search(prefix):
            raise ValueError(f"the prefix of a remote source names no dataset: {prefix!r}")
        if accession is not None and not _ACCESSION.fullmatch(accession):
            raise ValueError(f"an accession is ds and six digits, not {accession!r}")
        given = (subjects,) if isinstance(subjects, str) else tuple(subjects)
        if accession is None and (given or pseudonym is not None):
            raise ValueError("subject globs and a pseudonym are for a source of one dataset")
        self.client = client
        self.raw_client = raw_client
        """The client of the raw OpenNeuro bucket (``remote.RAW_BUCKET``), for the EPI
        interface of one dataset (decision D50); None: the source has no functional files.
        Making it makes no request; it is asked only when the run's EPI interface is on."""
        self.cache = cache
        self.prefix = prefix
        self.accession = accession
        self.globs: tuple[str, ...] = tuple(dict.fromkeys(given))
        self.pseudonym = pseudonym

    @property
    def bucket(self) -> str:
        return self.client.bucket

    def _clients(self) -> tuple[S3Client, ...]:
        return (self.client,) if self.raw_client is None else (self.client, self.raw_client)

    def stop(self) -> None:
        """Stop both clients (``S3Client.stop``): no request is started or retried."""
        for client in self._clients():
            client.stop()

    def close(self) -> None:
        """Close both clients."""
        for client in self._clients():
            client.close()

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

    # --- the listing and the files ----------------------------------------------------------

    def datasets(self, only: Collection[str] | None = None) -> Iterator[DatasetRecord]:
        """One dataset: its record (``source``, ``identity`` and ``accession`` the accession,
        ``root`` :func:`dataset_root`, ``fs_root`` the first of :data:`FS_ROOTS` with a
        subject prefix, layout and pipeline as the local rules give them, ``pipeline_version``
        from ``dataset_description.json``, ``template_location`` when ``fs_root/fsaverage/``
        lists the template files). Every dataset: one record per common prefix of
        ``fmriprep/`` that :func:`accession_of` reads, restricted to ``only``. The listing is
        kept for the life of the source, so the inventory does not list twice. A listing
        that cannot be had raises ``SourceError`` as ``S3Client.list`` does.

        Layout and pipeline follow ``sources.local.detect_layout`` for a directory with the
        same prefixes: ``sourcedata/freesurfer`` with a subject is ``fmriprep``; else
        ``freesurfer`` with a subject is ``fmriprep_legacy`` when a ``fmriprep/`` prefix lies
        beside it (the version then read from ``fmriprep/dataset_description.json``) and
        ``bids_derivatives`` (pipeline ``freesurfer``, no version) when none does.

        A dataset whose prefixes hold no FreeSurfer subject: the source of one dataset raises
        ``SourceError("layout", ...)``, as a local folder without subjects does; the source
        of every dataset passes it over, with a line at info level (there is nothing of it
        to inventory). A source of one dataset whose prefix lists nothing raises
        ``SourceError("not found", ...)``. No detail names the accession or a key."""
        wanted = None if only is None else set(only)
        if self.accession is not None:
            if wanted is not None and self.accession not in wanted:
                return
            record = self._described(self.accession)
            if record is None:
                raise SourceError(
                    "layout",
                    "the bucket holds this dataset's fMRIPrep output, and no FreeSurfer "
                    f"subject in it ({' or '.join(FS_ROOTS)})",
                )
            yield record
            return
        for accession in self._accessions():
            if wanted is not None and accession not in wanted:
                continue
            record = self._described(accession)
            if record is None:
                get_logger().info("a dataset of the bucket holds no FreeSurfer subject")
                continue
            yield record

    def subjects(self, dataset: DatasetRecord) -> Iterator[SubjectRecord]:
        """The subjects of ``dataset`` in label order: one listing per subject prefix, roles
        by ``patterns.roles_from_keys`` and candidates by ``patterns.is_subject_candidate_keys``
        (the rules of a local directory), ``FileRef`` with the ETag without quotes and
        ``last_modified``; templates skipped; the globs applied, a glob that matches nothing
        raising ``SourceError("not found", ...)`` with at most 10 names. Never opens a file.

        The globs are matched against the names below ``fs_root`` that are not templates,
        before any subject prefix is listed, so a review of two subjects of a large dataset
        lists two prefixes; a name that matches and is no subject candidate (no keys below
        ``mri/`` and ``surf/``) is then left out. The prefixes are listed at most
        ``max_concurrency`` at a time. A subject prefix whose listing is refused (403, or
        any reason but ``unreachable``) is yielded with ``problem`` set and no files, as a
        local directory that cannot be listed is. Everything is listed before the first
        record is returned."""
        base = self._base_key(dataset)
        names = self._children(base, dataset.fs_root)
        labels = sorted(name for name in names if not is_template_name(name))
        labels = self._select(labels)
        root_key = f"{base}{dataset.fs_root}/"

        def listed(label: str) -> tuple[bool, dict[FileRole, FileRef]] | SourceError:
            try:
                return self._subject_files(f"{root_key}{label}/", label)
            except SourceError as exc:
                if exc.reason == UNREACHABLE:
                    raise
                return exc

        records: list[SubjectRecord] = []
        for label, found in zip(labels, self._parallel(listed, labels), strict=True):
            fs_dir = (PurePosixPath(dataset.fs_root) / label).as_posix()
            if isinstance(found, SourceError):
                records.append(
                    SubjectRecord(dataset, label, fs_dir, {}, f"unreadable: {found.detail}")
                )
                continue
            candidate, refs = found
            if candidate:
                records.append(SubjectRecord(dataset, label, fs_dir, refs))
        return iter(records)

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
        ``changed``.

        The first whole read of a Tier A file of a subject fetches every Tier A file of the
        subject into the retained cache together (at most seven at once, under the client's
        limit), so that the inventory's reads of the build stamp, the logs and the stats cost
        one round; a file whose fetch failed raises that error when it is read, and is tried
        again by a later read. A range is cut from the retained copy when it is there and
        asked from the bucket otherwise. A source that no run has bound to a cache, and a
        role outside Tier A, read from the bucket directly and keep nothing. Errors:
        ``not found`` for a role the subject lacks, ``changed`` (404, 412, another ETag or
        size), ``unreadable`` (403, or a cached copy that cannot be read), ``unreachable``
        (retries ran out); each detail starts with the file's place in the subject."""
        if byte_range is not None and not 0 <= byte_range[0] <= byte_range[1]:
            raise ValueError(f"byte_range is an inclusive (first, last), not {byte_range}")
        ref = subject.files.get(role)
        if ref is None:
            raise SourceError("not found", f"the subject has no {role.value} file")
        if byte_range is not None:
            first, last = byte_range[0], min(byte_range[1], ref.size - 1)
            if first > last:
                return b""
        else:
            first, last = 0, ref.size - 1
        obj = self._object(subject, ref)
        cache = self.cache
        if cache is not None and role in TIER_A:
            kept = cache.path_for(self.bucket, obj.key, obj.etag, retained=True)
            if byte_range is None and not _is_whole(kept, ref.size):
                kept = self._tier_a(subject, cache).get_or_raise(role)
            if _is_whole(kept, ref.size):
                return _read_part(kept, ref.location, first, last, whole=byte_range is None)
        if ref.size == 0:
            return b""
        data, _ = self.client.get(
            obj.key,
            byte_range=None if byte_range is None else (first, last),
            if_match=obj.etag,
            location=ref.location,
        )
        if byte_range is None and len(data) != ref.size:
            raise SourceError("changed", f"{ref.location}: its size is not the listed one")
        return data

    def lease(
        self, subject: SubjectRecord, roles: Collection[FileRole]
    ) -> AbstractContextManager[Mapping[FileRole, Path]]:
        """``Cache.lease`` of the requested roles that the subject has, keyed by role. Raises
        ``SourceError`` as the cache and the client do, and ``ConfigError`` before the source
        was bound to a cache.

        Tier A roles are leased with ``retain``, so that they stay in ``retained/`` after the
        lease (the stats files that the inventory fetches are read by the regional metrics
        without a request), the others without, so that they are removed on exit unless
        ``cache.retain_roles`` keeps them. A ``SourceError`` of a lease names the file's
        place in the subject (``surf/lh.white: ...``) where the cache names its basename.
        A ``CacheBudgetError`` passes as it is."""
        cache = self.cache
        if cache is None:
            raise ConfigError(
                "invalid",
                "a remote source hands out files once a run has given it its cache; open it "
                "through the run",
            )
        return self._lease(cache, subject, roles)

    @contextmanager
    def _lease(
        self, cache: Cache, subject: SubjectRecord, roles: Collection[FileRole]
    ) -> Iterator[Mapping[FileRole, Path]]:
        wanted = [role for role in FileRole if role in roles and role in subject.files]
        groups = (
            ([role for role in wanted if role in TIER_A], True),
            ([role for role in wanted if role not in TIER_A], False),
        )
        paths: dict[FileRole, Path] = {}
        with ExitStack() as stack:
            for group, retain in groups:
                if not group:
                    continue
                objects = {role: self._object(subject, subject.files[role]) for role in group}
                places = {role: subject.files[role].location for role in group}
                try:
                    got = stack.enter_context(cache.lease(self.client, objects, retain=retain))
                except SourceError as exc:
                    raise _placed(exc, places.values()) from exc
                paths.update(got)
            yield {role: paths[role] for role in wanted}

    def template_dir(self, dataset: DatasetRecord) -> Path | None:
        """The ``fsaverage`` next to the subjects downloaded into the retained cache with the
        FreeSurfer subpaths (the eight files of ``patterns.TEMPLATE_FILES``, the optional
        ones, ``scripts/build-stamp.txt``); None when one of the eight is not listed.

        Each file is fetched into ``retained/`` as any retained object is (keyed name, its
        basename). The directory returned is ``retained/<h[:2]>/<h>/fsaverage``, where ``h``
        is ``Cache.name`` of the bucket, the template's prefix and a digest of the ETags of
        its files: a tree of hard links to the retained files (a copy where the file system
        takes no hard link), built in this process's lease area and renamed into place in
        one step, so that no partial tree ever has that name. A tree that is there is
        returned as it is, without a request but the listing of the template's prefix, which
        the source keeps. Raises ``ConfigError`` before the source was bound to a cache and
        ``SourceError`` as the client and the cache do."""
        if dataset.template_location is None:
            return None
        cache = self.cache
        if cache is None:
            raise ConfigError(
                "invalid",
                "a remote source hands out its template once a run has given it its cache",
            )
        base = self._base_key(dataset)
        prefix = f"{base}{dataset.template_location}/"
        listed = self._template_listing(prefix)
        required = [name.format(h=hemi) for name in TEMPLATE_FILES for hemi in HEMIS]
        if any(name not in listed for name in required):
            return None
        optional = [name.format(h=hemi) for name in TEMPLATE_OPTIONAL_FILES for hemi in HEMIS]
        names = [*required, *(name for name in (*optional, _BUILD_STAMP) if name in listed)]
        wanted = {name: listed[name] for name in names}
        digest = hashlib.sha256(
            "\n".join(f"{name}\t{obj.size}\t{obj.etag}" for name, obj in wanted.items()).encode()
        ).hexdigest()
        target = cache.path_for(self.bucket, prefix.rstrip("/"), digest, retained=True)
        if _tree_whole(target, wanted):
            return target

        def kept(name: str) -> Path | SourceError:
            try:
                return cache.retained(self.client, wanted[name])
            except SourceError as exc:
                if exc.reason == UNREACHABLE:
                    raise
                return _placed(exc, (name,))

        found = self._parallel(kept, names, at_most=_TIER_A_AT_ONCE)
        files: dict[str, Path] = {}
        for name, path in zip(names, found, strict=True):
            if isinstance(path, SourceError):
                raise path
            files[name] = path
        _build_tree(cache, target, files)
        return target

    def template_size(self, dataset: DatasetRecord) -> int | None:
        """The bytes that :meth:`template_dir` would fetch for ``dataset``, from the listing of
        the template's prefix (kept, as :meth:`template_dir` keeps it); None when the dataset
        has no complete template. Fetches nothing, so that a command can say what it will
        fetch before it does."""
        if dataset.template_location is None:
            return None
        listed = self._template_listing(f"{self._base_key(dataset)}{dataset.template_location}/")
        required = [name.format(h=hemi) for name in TEMPLATE_FILES for hemi in HEMIS]
        if any(name not in listed for name in required):
            return None
        optional = [name.format(h=hemi) for name in TEMPLATE_OPTIONAL_FILES for hemi in HEMIS]
        names = [*required, *(name for name in (*optional, _BUILD_STAMP) if name in listed)]
        return sum(listed[name].size for name in names)

    # --- the functional files of the EPI interface (decision D50) ---------------------------

    def func_listing(self, dataset: DatasetRecord) -> FuncListing | None:
        """What the EPI interface reads of the selected subjects of ``dataset``, from listings
        alone (no file is fetched): per participant (the leading ``sub-<label>`` of a selected
        FreeSurfer subject that has files) the raw files of ``s3://openneuro.org/dsXXXXXX/``
        and the fMRIPrep files of the derivative that the planner and the connector read, and
        the FreeSurfer files of the contours (``func.inventory.FS_FILES``) of every selected
        subject of the participant, each under its place in the tree.

        Raw: ``dataset_description.json`` and ``*_bold.json`` at the root of the dataset; in
        ``sub-P/`` and ``sub-P/ses-*/`` the ``*_bold.json`` (inherited sidecars), the
        ``sessions.tsv`` and the ``scans.tsv``; in ``sub-P/[ses-*/]func/`` every
        ``*_bold.nii[.gz]`` and ``*_bold.json``. Derivative: ``dataset_description.json``; in
        ``sub-P/[ses-*/]func/`` the files of the roles the connector lists in either
        generation (``fmriprep.reads_func_name``); in ``sub-P/[ses-*/]anat/`` the
        ``from-T1w_to-fsnative`` transform; in ``sub-P/figures/`` the summary and validation
        reports and the distortion figure. Nothing else: no events, physio, sbref, other
        spaces or anatomy. A participant without such a derivative ``func/`` file is left
        out. None for a source without a raw client, a dataset of another layout than
        ``fmriprep``, and when no participant is left. Kept for the life of the source.
        Raises ``SourceError`` as the listings do."""
        if self.raw_client is None or self.accession is None or dataset.layout != "fmriprep":
            return None
        memo = self._memo()
        with memo.lock:
            if dataset.identity in memo.func_listings:
                return memo.func_listings[dataset.identity]
        from surfcensus.func.connectors.fmriprep import (
            fsnative_names,
            reads_figure_name,
            reads_func_name,
        )
        from surfcensus.func.inventory import FS_FILES

        raw_client = self.raw_client
        records = [r for r in self.subjects(dataset) if r.problem is None and r.files]
        by_participant: dict[str, list[SubjectRecord]] = {}
        for record in records:
            found = _PARTICIPANT.match(record.subject_label)
            if found is not None:
                by_participant.setdefault(found.group(0), []).append(record)
        base = self._base_key(dataset)
        raw_base = raw_prefix(dataset.identity)
        files: dict[str, tuple[FuncBucket, S3Object]] = {}
        participants: list[str] = []
        for participant, subjects in sorted(by_participant.items()):
            derivs: dict[str, S3Object] = {}
            for entry in self.client.list(f"{base}{participant}/"):
                if isinstance(entry, str):
                    continue
                parts = entry.key[len(base) :].split("/")
                if _deriv_wanted(
                    parts, participant, reads_func_name, reads_figure_name, fsnative_names
                ):
                    derivs["/".join(parts)] = entry
            if not any(place.split("/")[-2] == "func" for place in derivs):
                continue
            participants.append(participant)
            for place, obj in derivs.items():
                files[f"{FUNC_TREE_DERIV}/{place}"] = ("deriv", obj)
            for entry in raw_client.list(f"{raw_base}{participant}/"):
                if isinstance(entry, str):
                    continue
                parts = entry.key[len(raw_base) :].split("/")
                if _raw_wanted(parts, participant):
                    files[f"{FUNC_TREE_RAW}/{'/'.join(parts)}"] = ("raw", entry)
            for subject in subjects:
                for role in FS_FILES:
                    ref = subject.files.get(role)
                    if ref is not None:
                        place = f"{FUNC_TREE_FS}/{subject.subject_label}/{ref.location}"
                        files[place] = ("deriv", self._object(subject, ref))
        listing = None
        if participants:
            description = self._object_at(f"{base}{_DESCRIPTION}")
            if description is not None:
                files[f"{FUNC_TREE_DERIV}/{_DESCRIPTION}"] = ("deriv", description)
            for entry in raw_client.list(raw_base, delimiter="/"):
                if isinstance(entry, str):
                    continue
                name = entry.key[len(raw_base) :]
                if name == _DESCRIPTION or name.endswith(_BOLD_JSON):
                    files[f"{FUNC_TREE_RAW}/{name}"] = ("raw", entry)
            listing = FuncListing(
                files=dict(sorted(files.items())), participants=tuple(participants)
            )
        with memo.lock:
            memo.func_listings[dataset.identity] = listing
        return listing

    def func_size(self, dataset: DatasetRecord) -> tuple[int, int]:
        """The bytes of :meth:`func_listing` that :meth:`func_tree` would fetch now (a whole
        retained copy is not fetched again; before the source is bound to a cache every byte
        counts), and the number of participants it holds; ``(0, 0)`` without a listing."""
        listing = self.func_listing(dataset)
        if listing is None:
            return 0, 0
        cache = self.cache
        total = 0
        for kind, obj in listing.files.values():
            if cache is not None:
                kept = cache.path_for(
                    self._client_of(kind).bucket, obj.key, obj.etag, retained=True
                )
                if _is_whole(kept, obj.size):
                    continue
            total += obj.size
        return total, len(listing.participants)

    def func_tree(self, dataset: DatasetRecord) -> FuncTree | None:
        """The functional files of :meth:`func_listing` downloaded into the retained cache and
        laid out as a folder tree that the path-based planner of ``func/`` reads as it reads a
        local folder: ``retained/<h[:2]>/<h>/func/`` with ``raw/`` (a BIDS dataset) and
        ``deriv/`` (the derivative, with ``sourcedata/freesurfer/<subject>/`` inside it), hard
        links to the retained files (copies with their times where the file system takes no
        hard link). ``h`` is ``Cache.name`` of the bucket, the dataset's prefix and a digest
        over the place, bucket, size and ETag of every file listed, as the template's tree is
        named. The tree is built in this process's lease area and renamed into place in one
        step, with :data:`FUNC_TREE_OBJECTS` (the cache names of the objects it links) in it,
        so that no partial tree ever has that name. A tree that is whole is returned without a
        download. None without a listing.

        The times of every retained functional file are set to the object's ``LastModified``
        of the listing, so the listing facts that the planner hashes (size and modification
        time) are those of the bucket, and a cache that is fetched again changes no hash.

        A tree is built only when every listed file was fetched. A file whose object changed
        or went since it was listed (404, 412, another ETag, a body that is not the listed
        one) or that may not be read (403) raises one ``SourceError`` (the first fault's
        reason; the detail names up to five places in the tree and the count) once the other
        downloads ended, and no tree is built: the files that were fetched stay retained, so
        the next call fetches only the rest. A bucket that cannot be reached raises
        ``SourceError("unreachable", ...)``. A stop (Ctrl-C, SIGTERM) stops both clients and
        leaves nothing partial under ``retained/`` (``Cache.retained``). A tree that was built
        is kept for the life of the source. ``ConfigError`` before the source was bound to a
        cache."""
        memo = self._memo()
        with memo.lock:
            if dataset.identity in memo.func_trees:
                return memo.func_trees[dataset.identity]
        listing = self.func_listing(dataset)
        if listing is None:
            return None
        cache = self.cache
        if cache is None:
            raise ConfigError(
                "invalid",
                "a remote source hands out its functional files once a run has given it its cache",
            )
        digest = hashlib.sha256(
            "\n".join(
                f"{place}\t{self._client_of(kind).bucket}\t{obj.size}\t{obj.etag}"
                for place, (kind, obj) in listing.files.items()
            ).encode()
        ).hexdigest()
        target = cache.path_for(
            self.bucket, f"{self._base_key(dataset)}{FUNC_TREE}", digest, retained=True
        )
        if not _func_tree_whole(target, listing):
            places = list(listing.files)

            def kept(place: str) -> Path | SourceError:
                kind, obj = listing.files[place]
                try:
                    path = cache.retained(self._client_of(kind), obj)
                except SourceError as exc:
                    if exc.reason == UNREACHABLE:
                        raise
                    return _placed(exc, (place,))
                _bucket_time(path, obj)
                return path

            found = self._parallel(kept, places)
            faults = [
                (place, path)
                for place, path in zip(places, found, strict=True)
                if isinstance(path, SourceError)
            ]
            if faults:
                raise _fetch_fault(faults)
            links = {
                place: path
                for place, path in zip(places, found, strict=True)
                if isinstance(path, Path)
            }
            names = sorted(
                cache.name(self._client_of(kind).bucket, obj.key, obj.etag)
                for kind, obj in listing.files.values()
            )
            record = json.dumps({"objects": names}, indent=1) + "\n"
            _clear_stale_tree(cache, target)
            _build_tree(
                cache,
                target,
                links,
                extra={FUNC_TREE_OBJECTS: record.encode()},
                dirs=(FUNC_TREE_RAW, FUNC_TREE_DERIV),
                what="functional",
            )
        else:
            for place, (_, obj) in listing.files.items():
                _bucket_time(target / place, obj)
        tree = FuncTree(
            root=target,
            raw=target / FUNC_TREE_RAW,
            deriv=target / FUNC_TREE_DERIV,
            fs_subjects=target / FUNC_TREE_FS,
        )
        with memo.lock:
            memo.func_trees[dataset.identity] = tree
        return tree

    def func_object_names(self, dataset: DatasetRecord) -> list[str]:
        """The cache names of the objects of :meth:`func_listing` (``Cache.name``, keyed with
        the run secret), for ``Run.prune_cache`` to keep while a tree of the dataset could not
        be built: what was fetched is then not fetched again. Empty without a listing or
        before the source was bound to a cache; lists nothing that is not listed yet."""
        memo = self._memo()
        with memo.lock:
            listing = memo.func_listings.get(dataset.identity)
        cache = self.cache
        if listing is None or cache is None:
            return []
        return sorted(
            cache.name(self._client_of(kind).bucket, obj.key, obj.etag)
            for kind, obj in listing.files.values()
        )

    def func_stems(self, dataset: DatasetRecord) -> list[str] | None:
        """The derivative stems of the preprocessed series of the selected participants
        (``sub-01_ses-1_task-rest_run-1``), from :meth:`func_listing`, for the check of an
        ``--epi-runs`` glob before it is stored; None without a listing."""
        listing = self.func_listing(dataset)
        if listing is None:
            return None
        stems = []
        for place in listing.files:
            name = place.rsplit("/", 1)[-1]
            found = _PRE_BOLD.search(name)
            if place.split("/")[-2] == "func" and found is not None:
                stems.append(name[: found.start()])
        return sorted(dict.fromkeys(stems))

    def _client_of(self, kind: FuncBucket) -> S3Client:
        if kind == "deriv":
            return self.client
        if self.raw_client is None:
            raise ConfigError("invalid", "this source has no client of the raw bucket")
        return self.raw_client

    # --- helpers -------------------------------------------------------------------------------

    def _memo(self) -> _Memo:
        memo = self._memo_state
        if memo is None:
            memo = self._memo_state = _Memo()
        return memo

    def _accessions(self) -> list[str]:
        """The accessions of ``fmriprep/``, by one delimiter listing, kept."""
        memo = self._memo()
        with memo.lock:
            if memo.accessions is not None:
                return memo.accessions
        found = sorted(
            {
                accession
                for entry in self.client.list(self.prefix, delimiter="/")
                if isinstance(entry, str) and (accession := accession_of(entry, self.prefix))
            }
        )
        with memo.lock:
            memo.accessions = found
        return found

    def _described(self, accession: str) -> DatasetRecord | None:
        """The record of one dataset, from its listings; None when it holds no FreeSurfer
        subject. Kept for the life of the source."""
        memo = self._memo()
        with memo.lock:
            if accession in memo.records:
                return memo.records[accession]
        base = dataset_prefix(accession, self.prefix)
        files: dict[str, S3Object] = {}
        folders: set[str] = set()
        for entry in self.client.list(base, delimiter="/"):
            if isinstance(entry, str):
                folders.add(entry[len(base) :].rstrip("/"))
            else:
                files[entry.key[len(base) :]] = entry
        if not files and not folders and self.accession is not None:
            raise SourceError(
                "not found",
                "the bucket holds no fMRIPrep output of this dataset; check the accession",
            )
        record = None
        for fs_root in FS_ROOTS:
            if fs_root.split("/", 1)[0] not in folders:
                continue
            names = self._children(base, fs_root)
            if not self._holds_subject(f"{base}{fs_root}/", names):
                continue
            record = self._record(accession, base, fs_root, files, folders, names)
            break
        with memo.lock:
            memo.records[accession] = record
        return record

    def _record(
        self,
        accession: str,
        base: str,
        fs_root: str,
        files: Mapping[str, S3Object],
        folders: Collection[str],
        names: Collection[str],
    ) -> DatasetRecord:
        layout: Layout
        if fs_root == FS_ROOTS[0]:
            layout, description = "fmriprep", files.get(_DESCRIPTION)
        elif "fmriprep" in folders:
            layout = "fmriprep_legacy"
            description = self._object_at(f"{base}fmriprep/{_DESCRIPTION}")
        else:
            layout, description = "bids_derivatives", None
        version = None if description is None else self._pipeline_version(description)
        template = None
        if _TEMPLATE_NAME in names:
            location = f"{fs_root}/{_TEMPLATE_NAME}"
            listed = self._template_listing(f"{base}{location}/")
            required = [name.format(h=hemi) for name in TEMPLATE_FILES for hemi in HEMIS]
            if all(name in listed for name in required):
                template = location
        return DatasetRecord(
            source=self.kind,
            identity=accession,
            accession=accession,
            root=dataset_root(self.bucket, accession, self.prefix),
            fs_root=fs_root,
            layout=layout,
            pipeline="freesurfer" if layout == "bids_derivatives" else "fmriprep",
            pipeline_version=version,
            template_location=template,
        )

    def _object_at(self, key: str) -> S3Object | None:
        """The object of ``key``, by a listing of it as a prefix; None when it is not there."""
        for entry in self.client.list(key):
            if not isinstance(entry, str) and entry.key == key:
                return entry
        return None

    def _pipeline_version(self, description: S3Object) -> str | None:
        """The fMRIPrep version in ``dataset_description.json``; None when it is too large,
        cannot be had (but for an unreachable bucket, which is raised) or says nothing."""
        if description.size > _DESCRIPTION_MAX_BYTES:
            return None
        if description.size == 0:
            return parse_pipeline_version(b"")
        try:
            raw, _ = self.client.get(
                description.key, if_match=description.etag, location=_DESCRIPTION
            )
        except SourceError as exc:
            if exc.reason == UNREACHABLE:
                raise
            return None
        return parse_pipeline_version(raw)

    def _children(self, base: str, fs_root: str) -> list[str]:
        """The names of the prefixes directly below ``fs_root``, by a delimiter listing, kept."""
        prefix = f"{base}{fs_root}/"
        memo = self._memo()
        with memo.lock:
            if prefix in memo.children:
                return memo.children[prefix]
        names = sorted(
            entry[len(prefix) :].rstrip("/")
            for entry in self.client.list(prefix, delimiter="/")
            if isinstance(entry, str)
        )
        with memo.lock:
            memo.children[prefix] = names
        return names

    def _holds_subject(self, root_key: str, names: Sequence[str]) -> bool:
        """Whether a name below ``root_key`` is a subject candidate: the first that is not a
        template and has keys below ``mri/`` and ``surf/``, one delimiter listing each, in
        name order, until one is found."""
        for name in names:
            if is_template_name(name):
                continue
            prefix = f"{root_key}{name}/"
            tops = [
                (entry if isinstance(entry, str) else entry.key)[len(prefix) :]
                for entry in self.client.list(prefix, delimiter="/")
            ]
            if is_subject_candidate_keys(tops, name):
                return True
        return False

    def _subject_files(self, prefix: str, label: str) -> tuple[bool, dict[FileRole, FileRef]]:
        """Whether the subject's prefix is a subject candidate, and one ``FileRef`` per role,
        from one listing of it, kept (the refs, not the whole listing)."""
        memo = self._memo()
        with memo.lock:
            if prefix in memo.subjects:
                return memo.subjects[prefix]
        listed = {
            entry.key[len(prefix) :]: entry
            for entry in self.client.list(prefix)
            if not isinstance(entry, str)
        }
        found = (is_subject_candidate_keys(listed, label), _file_refs(listed))
        with memo.lock:
            memo.subjects[prefix] = found
        return found

    def _template_listing(self, prefix: str) -> dict[str, S3Object]:
        """Every object below the template's prefix, by its place relative to it, kept."""
        memo = self._memo()
        with memo.lock:
            if prefix in memo.templates:
                return memo.templates[prefix]
        found = {
            entry.key[len(prefix) :]: entry
            for entry in self.client.list(prefix)
            if not isinstance(entry, str)
        }
        with memo.lock:
            memo.templates[prefix] = found
        return found

    def _select(self, labels: list[str]) -> list[str]:
        if not self.globs:
            return labels
        for glob in self.globs:
            if not any(fnmatchcase(label, glob) for label in labels):
                shown = ", ".join(labels[:_MAX_NAMES]) or "none"
                more = f" and {len(labels) - _MAX_NAMES} more" if len(labels) > _MAX_NAMES else ""
                raise SourceError(
                    "not found",
                    f"no subject matches {glob!r}. Subjects in the dataset: {shown}{more}",
                )
        return [label for label in labels if any(fnmatchcase(label, g) for g in self.globs)]

    def _base_key(self, dataset: DatasetRecord) -> str:
        """The key of the dataset's prefix in the bucket, from its ``root``."""
        start = f"s3://{self.bucket}/"
        if dataset.source != self.kind or not dataset.root.startswith(start):
            raise SourceError("invalid", "the dataset is not one of this source's bucket")
        base = dataset.root[len(start) :]
        if not base.startswith(self.prefix) or not base.endswith("/"):
            raise SourceError("invalid", "the dataset is not one of this source's prefix")
        return base

    def _object(self, subject: SubjectRecord, ref: FileRef) -> S3Object:
        """The object that ``ref`` lists, from the record alone: no request is made."""
        from surfcensus.remote.s3 import S3Object

        key = f"{self._base_key(subject.dataset)}{subject.fs_dir}/{ref.location}"
        return S3Object(key, ref.size, ref.etag_or_mtime, ref.last_modified or "")

    def _tier_a(self, subject: SubjectRecord, cache: Cache) -> _Fetched:
        """Every Tier A file of ``subject`` in the retained cache, fetched together on the
        first call for this version of its files and kept."""
        roles = [role for role in TIER_A if role in subject.files]
        token = (
            subject.dataset.root,
            subject.fs_dir,
            tuple((role.value, subject.files[role].etag_or_mtime) for role in roles),
        )
        memo = self._memo()
        with memo.lock:
            fetched = memo.tier_a.get(token)
            if fetched is None:
                fetched = memo.tier_a[token] = _Fetched()
        with fetched.lock:
            if not fetched.done:
                objects = [self._object(subject, subject.files[role]) for role in roles]

                def kept(index: int) -> Path | SourceError:
                    try:
                        return cache.retained(self.client, objects[index])
                    except SourceError as exc:
                        return _placed(exc, (subject.files[roles[index]].location,))

                found = self._parallel(kept, range(len(roles)), at_most=_TIER_A_AT_ONCE)
                fetched.paths = dict(zip(roles, found, strict=True))
                fetched.done = True
        return fetched

    def _parallel(
        self,
        work: Callable[[_T], _U],
        items: Collection[_T],
        *,
        at_most: int | None = None,
    ) -> list[_U]:
        """``work`` of each item, in their order, at most the client's ``max_concurrency``
        (and ``at_most``) at once. An exception of ``work`` is raised once the items that
        were running ended; those not yet started are dropped.

        A stop of the command (``KeyboardInterrupt``, which SIGTERM is turned into) ends the
        work: the clients (that of the raw bucket too) are stopped, so that no request is
        started or retried and a body being read is given up, the items not yet started are
        dropped, and the interrupt is
        raised once the running ones ended, which a stalled read delays by its timeout at
        most (``remote.s3.READ_STALL_S``). The client can be used again afterwards."""
        items = list(items)
        workers = min(len(items), self.client.config.max_concurrency, at_most or len(items))
        if workers <= 1:
            return [work(item) for item in items]
        pool = ThreadPoolExecutor(max_workers=workers)
        found: list[_U] = []
        try:
            futures = [pool.submit(work, item) for item in items]
            for future in futures:
                # Waited for in short steps, as the cache waits for its downloads: a wait
                # without a timeout is ended only by a signal that reaches this thread, and
                # which thread of a process gets a signal is the system's choice.
                while not wait((future,), timeout=_POLL_S).done:
                    pass
                found.append(future.result())
        except Exception:
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        except BaseException:
            clients = self._clients()
            for client in clients:
                client.stop()
            try:
                pool.shutdown(wait=True, cancel_futures=True)
            finally:
                for client in clients:
                    client.stopped.clear()
            raise
        pool.shutdown(wait=True)
        return found


_POLL_S = 0.1
"""How long ``_parallel`` waits at a time for a result (``remote.cache`` waits likewise)."""
_DESCRIPTION = "dataset_description.json"
_DESCRIPTION_MAX_BYTES = 1_000_000
"""A ``dataset_description.json`` larger than this is not read: none is that large."""
_TEMPLATE_NAME = "fsaverage"
_BUILD_STAMP = "scripts/build-stamp.txt"
_MAX_NAMES = 10
"""Names listed when a glob matches nothing, as for a local source."""
_TIER_A_AT_ONCE = len(TIER_A)
"""At most this many Tier A (or template) files are fetched at once, under the client's
limit: the seven Tier A files of a subject in one round."""

FUNC_TREE = "func"
"""The name of the functional tree of a dataset in the retained cache (decision D50)."""
FUNC_TREE_RAW = "raw"
"""The raw BIDS dataset in the tree: the files of ``s3://openneuro.org/dsXXXXXX/``."""
FUNC_TREE_DERIV = "deriv"
"""The fMRIPrep derivative in the tree: the files of ``fmriprep/dsXXXXXX-fmriprep/``."""
FUNC_TREE_FS = f"{FUNC_TREE_DERIV}/sourcedata/freesurfer"
"""Where the tree holds the FreeSurfer subjects of the contours, whatever ``fs_root`` the
dataset has: the place ``func.discover`` finds them in a local fMRIPrep output."""
FUNC_TREE_OBJECTS = ".objects.json"
"""The file of a tree that names the cache names of the objects it links (keyed with the run
secret), inside the tree, so that ``Run.prune_cache`` keeps those objects; a tree without it is
not whole."""
FuncBucket = Literal["raw", "deriv"]
_PARTICIPANT = re.compile(r"sub-[A-Za-z0-9]+")
"""The participant of a FreeSurfer subject directory, as the inventory reads it."""
_BOLD_JSON = "_bold.json"
_RAW_SERIES = ("_bold.nii.gz", "_bold.nii", _BOLD_JSON)
_PRE_BOLD = re.compile(r"_space-[A-Za-z0-9]+_desc-preproc_bold\.nii(\.gz)?$")


@dataclass(frozen=True)
class FuncListing:
    """What :meth:`OpenNeuroDerivatives.func_listing` found: place in the tree ->
    (bucket kind, object), and the participants it covers."""

    files: Mapping[str, tuple[FuncBucket, S3Object]]
    participants: tuple[str, ...]

    @property
    def size(self) -> int:
        return sum(obj.size for _, obj in self.files.values())


@dataclass(frozen=True)
class FuncTree:
    """The functional tree of a dataset in the retained cache: its folders."""

    root: Path
    raw: Path
    deriv: Path
    fs_subjects: Path


def _deriv_wanted(
    parts: Sequence[str],
    participant: str,
    reads_func: Callable[[str], bool],
    reads_figure: Callable[[str], bool],
    fsnative: Callable[[str, str | None], tuple[str, ...]],
) -> bool:
    """Whether a key of the derivative below ``sub-P/`` (split at ``/``, ``sub-P`` first) is
    one the EPI interface reads."""
    if not parts or parts[0] != participant:
        return False
    if len(parts) == 3 and parts[1] == "figures":
        return reads_figure(parts[2])
    session = None
    if len(parts) == 4:
        if not parts[1].startswith("ses-"):
            return False
        session = parts[1][len("ses-") :]
    elif len(parts) != 3:
        return False
    if parts[-2] == "func":
        return reads_func(parts[-1])
    if parts[-2] == "anat":
        return parts[-1] in fsnative(participant, session)
    return False


def _raw_wanted(parts: Sequence[str], participant: str) -> bool:
    """Whether a key of a raw dataset below ``sub-P/`` (split at ``/``) is one the EPI
    interface reads: a series or sidecar in ``func/``, an inherited ``*_bold.json`` of the
    subject or a session, the ``sessions.tsv`` and the ``scans.tsv``."""
    if not parts or parts[0] != participant:
        return False
    name = parts[-1]
    if len(parts) == 2:
        return name.endswith(_BOLD_JSON) or name in (
            f"{participant}_sessions.tsv",
            f"{participant}_scans.tsv",
        )
    in_session = parts[1].startswith("ses-")
    if len(parts) == 3:
        if parts[1] == "func":
            return name.endswith(_RAW_SERIES)
        return in_session and (name.endswith(_BOLD_JSON) or name.endswith("_scans.tsv"))
    if len(parts) == 4:
        return in_session and parts[2] == "func" and name.endswith(_RAW_SERIES)
    return False


def _func_tree_whole(target: Path, listing: FuncListing) -> bool:
    """Whether ``target`` holds a whole tree: its :data:`FUNC_TREE_OBJECTS` reads and names no
    place left out (a tree of the first build of D50 with a non-empty ``left_out`` is not
    whole), and every listed place is there at its listed size."""
    try:
        record = json.loads((target / FUNC_TREE_OBJECTS).read_text(encoding="utf-8"))
        if record.get("left_out"):
            return False
        if not isinstance(record["objects"], list):
            return False
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False
    return all(_is_whole(target / place, obj.size) for place, (_, obj) in listing.files.items())


MAX_FAULT_PLACES = 5
"""Places that the fault of a functional fetch names; the rest are counted."""


def _fetch_fault(faults: Sequence[tuple[str, SourceError]]) -> SourceError:
    """One ``SourceError`` for the files of a tree that could not be fetched: the reason of the
    first, and a detail that names up to :data:`MAX_FAULT_PLACES` places and the count."""
    places = [place for place, _ in faults]
    shown = ", ".join(places[:MAX_FAULT_PLACES])
    more = f" and {len(places) - MAX_FAULT_PLACES} more" if len(places) > MAX_FAULT_PLACES else ""
    noun = "file was" if len(places) == 1 else "files were"
    return SourceError(
        faults[0][1].reason,
        f"{len(places)} {noun} refused or changed since they were listed: {shown}{more}",
    )


def _listed_ns(obj: S3Object) -> int | None:
    """``LastModified`` of a listing in nanoseconds since the epoch; None when it is missing
    or does not parse."""
    from datetime import datetime

    text = obj.last_modified
    if not text:
        return None
    try:
        stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        return None
    seconds = int(stamp.timestamp())
    return seconds * 1_000_000_000 + stamp.microsecond * 1_000


def _bucket_time(path: Path, obj: S3Object) -> None:
    """Set the access and modification time of a retained functional file to the object's
    ``LastModified`` of the listing, unless it has it already (decision D50, review: a cache
    that is fetched again keeps every inventory hash and pack signature). Nothing in the cache
    or the leases reads the time of a retained file."""
    ns = _listed_ns(obj)
    if ns is None:
        return
    try:
        if os.stat(path).st_mtime_ns != ns:
            os.utime(path, ns=(ns, ns))
    except OSError:
        pass


def _clear_stale_tree(cache: Cache, target: Path) -> None:
    """Move a tree at ``target`` that is not whole (one of the first build of D50 that named
    places left out, or one that lost a file) out of the way before a whole one is renamed
    there; it is removed from this process's lease area."""
    if not target.exists():
        return
    aside = cache.hold_area() / f".stale.{secrets.token_hex(8)}"
    try:
        os.rename(target, aside)
    except OSError:
        return
    shutil.rmtree(aside, ignore_errors=True)


def tree_objects(tree_root: Path) -> list[str] | None:
    """The cache names that a functional tree links, from its :data:`FUNC_TREE_OBJECTS`;
    None when it cannot be read (``Run.prune_cache``)."""
    try:
        record = json.loads((Path(tree_root) / FUNC_TREE_OBJECTS).read_text(encoding="utf-8"))
        return [str(name) for name in record["objects"]]
    except (OSError, ValueError, KeyError, TypeError):
        return None


@dataclass
class _Fetched:
    """The Tier A files of one version of one subject in the retained cache: a path, or the
    error of its fetch, per role."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    done: bool = False
    paths: dict[FileRole, Path | SourceError] = field(default_factory=dict)

    def get_or_raise(self, role: FileRole) -> Path:
        """The path of ``role``; raises the error of its fetch once, so that a later read
        fetches it again."""
        with self.lock:
            found = self.paths.get(role)
            if isinstance(found, SourceError):
                del self.paths[role]
                self.done = False
                raise found
        if found is None:  # not part of the round: tried again by the next one
            raise SourceError("changed", "the file was not fetched with its subject")
        return found


@dataclass
class _Memo:
    """What a source keeps of its listings, for the life of the source."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    accessions: list[str] | None = None
    records: dict[str, DatasetRecord | None] = field(default_factory=dict)
    children: dict[str, list[str]] = field(default_factory=dict)
    subjects: dict[str, tuple[bool, dict[FileRole, FileRef]]] = field(default_factory=dict)
    templates: dict[str, dict[str, S3Object]] = field(default_factory=dict)
    tier_a: dict[tuple[object, ...], _Fetched] = field(default_factory=dict)
    func_listings: dict[str, FuncListing | None] = field(default_factory=dict)
    func_trees: dict[str, FuncTree | None] = field(default_factory=dict)


def _file_refs(listed: Mapping[str, S3Object]) -> dict[FileRole, FileRef]:
    """One ``FileRef`` per role of a subject's listing, in ``FileRole`` order."""
    found = roles_from_keys(listed)
    refs: dict[FileRole, FileRef] = {}
    for role in FileRole:
        location = found.get(role)
        if location is None:
            continue
        obj = listed[location]
        refs[role] = FileRef(
            role=role,
            location=location,
            size=obj.size,
            etag_or_mtime=obj.etag.strip('"'),
            variant=key_variant(role, location),
            last_modified=obj.last_modified or None,
        )
    return refs


def _placed(exc: SourceError, places: Collection[str]) -> SourceError:
    """``exc`` with its detail starting at the file's place in the subject (or template) where
    the cache named the file's basename; the reason and the class are kept."""
    detail = exc.detail
    for place in places:
        name = PurePosixPath(place).name
        if detail.startswith(f"{place}:"):
            return exc
        if detail.startswith(f"{name}:"):
            detail = place + detail[len(name) :]
            break
    if detail == exc.detail:
        return exc
    try:
        made = type(exc)(exc.reason, detail)
    except TypeError:
        made = SourceError(exc.reason, detail)
    return made


def _is_whole(path: Path, size: int) -> bool:
    try:
        info = path.stat()
    except OSError:
        return False
    return path.is_file() and info.st_size == size


def _read_part(path: Path, location: str, first: int, last: int, *, whole: bool) -> bytes:
    """Bytes ``first`` to ``last`` of a retained file (all of it with ``whole``)."""
    try:
        with path.open("rb") as stream:
            if whole:
                return stream.read()
            stream.seek(first)
            return stream.read(last - first + 1)
    except OSError as exc:
        raise SourceError(
            "unreadable",
            f"{location}: the copy in the cache cannot be read: "
            f"{exc.strerror or type(exc).__name__}",
        ) from exc


def _tree_whole(target: Path, wanted: Mapping[str, S3Object]) -> bool:
    """Whether ``target`` holds every file of ``wanted`` at its size."""
    return all(_is_whole(target / name, obj.size) for name, obj in wanted.items())


def _build_tree(
    cache: Cache,
    target: Path,
    files: Mapping[str, Path],
    *,
    extra: Mapping[str, bytes] | None = None,
    dirs: Sequence[str] = (),
    what: str = "template",
) -> None:
    """Lay ``files`` (place in the tree to retained file) out below ``target``: hard links
    (or copies that keep the file's times) in a directory of this process's lease area,
    beside the files of ``extra`` (place to bytes, written last) and the empty folders
    ``dirs``, which is renamed to ``target`` in one step. When another process put a whole
    tree there first, it is kept."""
    # the staging tree is written into this process's lease area, whose lock the cache takes
    # on first use; when every file was retained already no call has taken it yet, and an
    # area without its lock could be pruned by another process or cleared by the first lease
    staging = cache.hold_area() / f".{what}.{secrets.token_hex(8)}"
    tree = staging / target.name
    try:
        tree.mkdir(parents=True)
        for name in dirs:
            (tree / name).mkdir(parents=True, exist_ok=True)
        for name, source in files.items():
            dest = tree / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, dest)
            except OSError:
                # a copy keeps the times, so that the listing facts of the planner (size and
                # modification time) are those of the retained file, as for a hard link
                shutil.copy2(source, dest)
        for name, data in (extra or {}).items():
            dest = tree / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(tree, target)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            # another process (or thread) renamed its tree there first: the same files
            if not all(
                _is_whole(target / name, path.stat().st_size) for name, path in files.items()
            ):
                raise ConfigError(
                    "invalid",
                    f"the cache holds an incomplete {what} directory; remove the run's "
                    "cache/retained folder and run the command again",
                ) from exc
            for name, data in (extra or {}).items():
                if not (target / name).is_file():
                    (target / name).write_bytes(data)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
