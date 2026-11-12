"""The local source: FreeSurfer subjects in a directory, read in place.

A path is recognized as one of the layouts of docs/architecture.md: a ``SUBJECTS_DIR``, an
fMRIPrep output (new and old layout), a BIDS derivatives root, or one subject directory.
Subjects are listed with ``stat()`` only; their files are handed out where they are.

Read-only by construction: nothing here opens for writing, creates or removes.
"""

from __future__ import annotations

import dataclasses
import math
import os
import re
import sys
import types
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from stat import S_ISDIR

from surfcensus.config import SourceConfig
from surfcensus.inventory.provenance import parse_dataset_description
from surfcensus.log import get_logger
from surfcensus.sources.base import (
    HEMIS,
    DatasetRecord,
    FileRef,
    FileRole,
    Layout,
    SourceError,
    SourceKind,
    SubjectRecord,
)
from surfcensus.sources.patterns import (
    ROLE_CANDIDATES,
    TEMPLATE_FILES,
    is_file,
    is_subject_candidate,
    resolve_subject_files,
    variant_name,
)

TEMPLATE_NAME = "fsaverage"
"""The directory next to the subjects that can serve as the template."""

_DESCRIPTION = "dataset_description.json"
_MAX_NAMES = 10  # names listed when a glob matches nothing


def _strerror(exc: OSError) -> str:
    # the error without the path, which could name a dataset
    return exc.strerror or type(exc).__name__


def _names(directory: Path) -> list[str] | None:
    """Sorted names of the entries of ``directory``; None if there is no such directory.

    Raises OSError when it exists and cannot be listed.
    """
    try:
        return sorted(os.listdir(directory))
    except (FileNotFoundError, NotADirectoryError):
        return None


def _is_dir(path: Path) -> bool:
    """Whether ``path`` is a directory or a link to one. Unlike ``Path.is_dir()`` this
    raises OSError when the path cannot be examined, on every Python version."""
    try:
        return S_ISDIR(path.stat().st_mode)
    except (FileNotFoundError, NotADirectoryError):
        return False


def _has_candidate(directory: Path) -> bool:
    """Whether a child of ``directory`` is a subject candidate. A child that cannot be
    examined does not count. Raises OSError when ``directory`` cannot be listed."""
    for name in _names(directory) or ():
        try:
            if is_subject_candidate(directory / name):
                return True
        except OSError:
            continue
    return False


def _keeping_name(path: Path | str) -> Path:
    """``path`` made absolute with every link resolved but one in its last component, so a
    subject directory that is a link keeps the name it is listed under."""
    path = Path(path)
    if path.name in ("", ".."):
        return path.resolve()
    return path.parent.resolve() / path.name


def _escape(name: str) -> str:
    """The glob that matches ``name`` and nothing else."""
    return re.sub(r"([*?\[])", r"[\1]", name)


def template_files_missing(directory: Path | str) -> tuple[str, ...] | None:
    """The files of ``patterns.TEMPLATE_FILES`` that ``directory`` lacks, for both
    hemispheres; ``()`` for a directory that can serve as the template, None when there is
    no such directory (a dangling link included).

    Raises OSError when the directory cannot be examined.
    """
    directory = Path(directory)
    if not _is_dir(directory):
        return None
    wanted = [name.format(h=hemi) for name in TEMPLATE_FILES for hemi in HEMIS]
    return tuple(name for name in wanted if not is_file(directory / name))


def find_adjacent_template(subjects_dir: Path | str) -> Path | None:
    """``subjects_dir/fsaverage`` (a directory or a link to one) when it holds every file of
    ``patterns.TEMPLATE_FILES`` for both hemispheres, else None.

    None also for a dangling link, for a directory that lacks files
    (:func:`template_files_missing` names them) and for one that cannot be examined.
    """
    directory = Path(subjects_dir) / TEMPLATE_NAME
    try:
        missing = template_files_missing(directory)
    except OSError:
        return None
    return directory if missing == () else None


def _pipeline_version(directory: Path) -> str | None:
    """The fMRIPrep version in ``directory/dataset_description.json``; None when the file is
    missing, unreadable or says nothing usable."""
    try:
        raw = (directory / _DESCRIPTION).read_bytes()
    except OSError:
        return None
    return parse_dataset_description(raw.decode("utf-8", errors="replace"))


def _record(root: Path, fs_root: str, layout: Layout) -> DatasetRecord:
    subjects_dir = root / fs_root
    fmriprep = layout in ("fmriprep", "fmriprep_legacy")
    version = None
    if fmriprep:
        version = _pipeline_version(root if layout == "fmriprep" else root / "fmriprep")
    try:
        has_template = _is_dir(subjects_dir / TEMPLATE_NAME)
    except OSError:
        has_template = False
    return DatasetRecord(
        source="local",
        identity=str(root),
        accession=None,
        root=str(root),
        fs_root=fs_root,
        layout=layout,
        pipeline="fmriprep" if fmriprep else "freesurfer",
        pipeline_version=version,
        # recorded whether or not the directory is complete, so that the template lookup can
        # say what it lacks; template_dir() answers only for one that can be used
        template_location=(PurePosixPath(fs_root) / TEMPLATE_NAME).as_posix()
        if has_template
        else None,
    )


def detect_layout(path: Path | str) -> tuple[DatasetRecord, tuple[str, ...]]:
    """The dataset record of ``path`` and the subject globs its layout implies.

    The first rule that matches decides (a subject candidate is a directory with ``mri/``
    and ``surf/`` that is not a template):

    1. ``path`` is itself a subject candidate: ``subjects_dir`` layout rooted at its parent;
       the implied glob is its own name, escaped.
    2. ``path/sourcedata/freesurfer`` holds a candidate: ``fmriprep``.
    3. ``path/freesurfer`` holds one: ``fmriprep_legacy`` when ``path/fmriprep`` is a
       directory, else ``bids_derivatives``.
    4. ``path`` is named ``fmriprep`` and ``path/../freesurfer`` holds one:
       ``fmriprep_legacy`` rooted at the parent.
    5. A child of ``path`` is a candidate: ``subjects_dir``.

    The record's ``root`` is resolved and its ``identity`` is that root. Raises
    :class:`SourceError`: ``not found`` when ``path`` is not a directory, ``unreadable`` when
    it or one of the places above cannot be listed, ``layout`` when no rule matches.
    """
    given = _keeping_name(path)
    try:
        return _detect(given)
    except OSError as exc:
        where = exc.filename if isinstance(exc.filename, str) else str(given)
        raise SourceError("unreadable", f"{where}: {_strerror(exc)}") from exc


def _detect(given: Path) -> tuple[DatasetRecord, tuple[str, ...]]:
    log = get_logger()
    if not _is_dir(given):
        raise SourceError("not found", f"{given} is not a directory")
    if is_subject_candidate(given):
        log.info("%s is one FreeSurfer subject directory", given)
        return _record(given.parent, ".", "subjects_dir"), (_escape(given.name),)

    base = given.resolve()
    new, old = base / "sourcedata" / "freesurfer", base / "freesurfer"
    if _has_candidate(new):
        try:
            both = _has_candidate(old)
        except OSError:  # a folder that is not used need not be readable
            both = False
        if both:
            log.warning(
                "%s and %s both hold FreeSurfer subjects; using the first and ignoring %s",
                new,
                old,
                old,
            )
        log.info("%s is an fMRIPrep output with subjects in sourcedata/freesurfer", base)
        return _record(base, "sourcedata/freesurfer", "fmriprep"), ()
    if _has_candidate(old):
        legacy = _is_dir(base / "fmriprep")
        log.info(
            "%s holds subjects in freesurfer/ (%s)",
            base,
            "next to an fmriprep/ folder" if legacy else "BIDS derivatives",
        )
        return _record(base, "freesurfer", "fmriprep_legacy" if legacy else "bids_derivatives"), ()
    # the folder as the user named it, then what it links to
    for folder in dict.fromkeys((given, base)):
        if folder.name == "fmriprep" and _has_candidate(folder.parent / "freesurfer"):
            log.info("%s is an fMRIPrep folder next to freesurfer/", folder)
            return _record(folder.parent, "freesurfer", "fmriprep_legacy"), ()
    if _has_candidate(base):
        log.info("%s is a FreeSurfer subjects directory", base)
        return _record(base, ".", "subjects_dir"), ()

    unreadable = _unexaminable(base)
    raise SourceError(
        "layout",
        f"no FreeSurfer subject found at {base}. A subject is a directory with mri/ and "
        "surf/ that is not a template; looked at the path itself, the folders in "
        "sourcedata/freesurfer/, in freesurfer/, in ../freesurfer/ (when the path is named "
        "fmriprep) and the path's own folders"
        + (f". Could not be examined: {', '.join(unreadable)}" if unreadable else ""),
    )


def _unexaminable(directory: Path) -> list[str]:
    names = []
    for name in _names(directory) or ():
        try:
            is_subject_candidate(directory / name)
        except OSError:
            names.append(name)
    return names[:_MAX_NAMES]


class LocalSource:
    """A directory of FreeSurfer subjects in one of the layouts of docs/architecture.md.

    ``path`` is what the user passes: a ``SUBJECTS_DIR``, an fMRIPrep output, a BIDS
    derivatives root, or one subject directory. ``subjects`` are globs on directory names
    (a union; empty means all). ``identity`` is given when a stored source is rebuilt, so
    that subjects keep their ``subject_uid`` after the data moved; otherwise it is the
    resolved root. Raises ``SourceError`` when the path is not a directory (``not found``),
    cannot be listed (``unreadable``) or matches no layout (``layout``).

    When ``path`` is one subject directory the source is its parent directory restricted to
    that name; ``subjects``, if given as well, must match the name.
    """

    kind: SourceKind = "local"
    dataset: DatasetRecord

    def __init__(
        self,
        path: Path | str,
        *,
        subjects: Sequence[str] = (),
        identity: str | None = None,
        field_strength: float | None = None,
    ) -> None:
        if field_strength is not None and not 0 < field_strength < math.inf:
            raise ValueError(
                f"field_strength is in tesla, positive and finite, not {field_strength}"
            )
        given = (subjects,) if isinstance(subjects, str) else tuple(subjects)
        globs = tuple(dict.fromkeys(given))
        record, implied = detect_layout(path)
        if implied:
            name = _keeping_name(path).name
            if globs and not any(fnmatchcase(name, glob) for glob in globs):
                raise SourceError(
                    "not found",
                    f"the subject directory {name} matches none of the subject globs "
                    f"{', '.join(globs)}",
                )
            globs = implied
        if identity is not None:
            record = dataclasses.replace(record, identity=identity)
        self.dataset = record
        self._globs = globs
        self._field_strength = field_strength

    @property
    def subjects_dir(self) -> Path:
        """The directory whose children are the subjects: ``root/fs_root``. A run directory
        must not lie inside it."""
        return Path(self.dataset.root) / self.dataset.fs_root

    def datasets(self, only: Collection[str] | None = None) -> Iterator[DatasetRecord]:
        """Exactly one dataset. ``only`` restricts remote sources by accession; a local
        dataset has none, so it is not looked at."""
        yield self.dataset

    def subjects(self, dataset: DatasetRecord) -> Iterator[SubjectRecord]:
        """The subject candidates among the children of the subjects directory, by name.

        Templates are skipped, by their name and by the name of what they link to, and so
        is a child whose name is not valid UTF-8, with a warning: a label has to be text. A
        child that cannot be examined is yielded with ``problem`` set and no files. Of two names
        for one directory the first is kept. Everything is listed before the first record
        is returned, so a glob that matches nothing raises ``SourceError("not found", ...)``
        here and not half-way through. ``dataset`` is attached to the records as given.
        """
        subjects_dir = self.subjects_dir
        try:
            names = _names(subjects_dir)
        except OSError as exc:
            raise SourceError("unreadable", f"{subjects_dir}: {_strerror(exc)}") from exc
        if names is None:
            raise SourceError("not found", f"{subjects_dir} is no longer a directory")

        log = get_logger()
        problems: dict[str, str] = {}
        labels = []
        for name in names:
            problem = None
            try:
                if not is_subject_candidate(subjects_dir / name):
                    continue
            except OSError as exc:
                problem = f"unreadable: {_strerror(exc)}"
            try:
                name.encode("utf-8")
            except UnicodeEncodeError:
                # a label is hashed into the subject_uid and stored as text; bytes that are
                # no text can be neither
                log.warning("%r in %s is skipped: its name is not valid UTF-8", name, subjects_dir)
                continue
            if problem is not None:
                problems[name] = problem
            labels.append(name)
        labels = self._select(labels)

        seen: dict[Path, str] = {}
        records = []
        for label in labels:
            directory = subjects_dir / label
            fs_dir = (PurePosixPath(self.dataset.fs_root) / label).as_posix()
            files: dict[FileRole, FileRef] = {}
            problem = problems.get(label)
            if problem is None:
                first = seen.setdefault(directory.resolve(), label)
                if first != label:
                    log.warning(
                        "%s is the same directory as %s and is listed once, as %s",
                        label,
                        first,
                        first,
                    )
                    continue
                try:
                    files = _list_files(directory)
                except OSError as exc:
                    problem = f"unreadable: {_strerror(exc)}"
            records.append(SubjectRecord(dataset, label, fs_dir, files, problem))
        return iter(records)

    def _select(self, labels: list[str]) -> list[str]:
        if not self._globs:
            return labels
        for glob in self._globs:
            if not any(fnmatchcase(label, glob) for label in labels):
                shown = ", ".join(labels[:_MAX_NAMES]) or "none"
                more = f" and {len(labels) - _MAX_NAMES} more" if len(labels) > _MAX_NAMES else ""
                raise SourceError(
                    "not found",
                    f"no subject matches {glob!r}. Subjects in {self.subjects_dir}: {shown}{more}",
                )
        return [label for label in labels if any(fnmatchcase(label, g) for g in self._globs)]

    def _path(self, subject: SubjectRecord, ref: FileRef) -> Path:
        return Path(self.dataset.root, subject.fs_dir, ref.location)

    def read_bytes(
        self,
        subject: SubjectRecord,
        role: FileRole,
        *,
        byte_range: tuple[int, int] | None = None,
    ) -> bytes:
        """The content of the file of ``role``, or its inclusive ``(first, last)`` byte range,
        cut short at the end of the file.

        Raises ``SourceError``: ``not found`` when the subject has no file of that role,
        ``changed`` when the file is gone or differs in size or modification time from the
        listing, ``unreadable`` when it cannot be opened or read.
        """
        if byte_range is not None and not 0 <= byte_range[0] <= byte_range[1]:
            raise ValueError(f"byte_range is an inclusive (first, last), not {byte_range}")
        ref = subject.files.get(role)
        if ref is None:
            raise SourceError("not found", f"{subject.subject_label} has no {role.value} file")
        try:
            with self._path(subject, ref).open("rb") as stream:
                _check_unchanged(ref, os.fstat(stream.fileno()))
                if byte_range is None:
                    return stream.read()
                stream.seek(byte_range[0])
                return stream.read(byte_range[1] - byte_range[0] + 1)
        except OSError as exc:
            raise _file_error(ref, exc) from exc

    def lease(
        self, subject: SubjectRecord, roles: Collection[FileRole]
    ) -> AbstractContextManager[Mapping[FileRole, Path]]:
        """The files of the requested roles that the subject has, where they are. Nothing is
        copied and nothing is done on exit. Entering raises ``SourceError("changed", ...)``
        when a requested file is gone or differs from the listing."""
        return self._lease(subject, roles)

    @contextmanager
    def _lease(
        self, subject: SubjectRecord, roles: Collection[FileRole]
    ) -> Iterator[Mapping[FileRole, Path]]:
        paths: dict[FileRole, Path] = {}
        for role in FileRole:
            ref = subject.files.get(role)
            if ref is None or role not in roles:
                continue
            path = self._path(subject, ref)
            try:
                _check_unchanged(ref, path.stat())
            except OSError as exc:
                raise _file_error(ref, exc) from exc
            paths[role] = path
        yield paths

    def template_dir(self, dataset: DatasetRecord) -> Path | None:
        """The ``fsaverage`` next to the subjects when it has every template file, else
        None. Looked up on each call."""
        return find_adjacent_template(self.subjects_dir)

    def config(self) -> SourceConfig:
        """``SourceConfig(kind="local", root, identity, subjects, field_strength)``.

        ``root`` is the dataset's root and ``subjects`` the globs in force, so a source on one
        subject directory is stored as its parent with that subject's name as the glob.
        """
        return SourceConfig(
            kind="local",
            root=self.dataset.root,
            identity=self.dataset.identity,
            subjects=self._globs,
            field_strength=self._field_strength,
        )


def _list_files(directory: Path) -> dict[FileRole, FileRef]:
    """One :class:`FileRef` per role that ``directory`` has, from ``stat()`` alone. Size and
    modification time are those of the file a link leads to."""
    paths = resolve_subject_files(directory)
    files = {}
    for role in FileRole:
        if role not in paths:
            continue
        path = paths[role]
        stat = path.stat()
        files[role] = FileRef(
            role=role,
            location=path.relative_to(directory).as_posix(),
            size=stat.st_size,
            etag_or_mtime=str(stat.st_mtime_ns),
            variant=variant_name(role, path) if len(ROLE_CANDIDATES[role]) > 1 else None,
        )
    return files


def _check_unchanged(ref: FileRef, stat: os.stat_result) -> None:
    if stat.st_size != ref.size:
        what = f"its size went from {ref.size} to {stat.st_size} bytes"
    elif str(stat.st_mtime_ns) != ref.etag_or_mtime:
        what = "its modification time differs"
    else:
        return
    raise SourceError("changed", f"{ref.location} changed after it was listed: {what}")


def _file_error(ref: FileRef, exc: OSError) -> SourceError:
    if isinstance(exc, FileNotFoundError | NotADirectoryError):
        return SourceError("changed", f"{ref.location} was removed after it was listed")
    return SourceError("unreadable", f"{ref.location}: {_strerror(exc)}")


def local(
    path: Path | str, *, subjects: Sequence[str] = (), field_strength: float | None = None
) -> LocalSource:
    """``surfcensus.sources.local(path)``: a local source on ``path``."""
    return LocalSource(path, subjects=subjects, field_strength=field_strength)


class _CallableModule(types.ModuleType):
    """This module, callable as :func:`local`.

    ``surfcensus.sources.local`` names both this module and the function that makes a
    source (docs/architecture.md has both). Importing the module binds the name in the
    package to the module, so the module itself has to answer the call.
    """

    def __call__(
        self,
        path: Path | str,
        *,
        subjects: Sequence[str] = (),
        field_strength: float | None = None,
    ) -> LocalSource:
        return local(path, subjects=subjects, field_strength=field_strength)


sys.modules[__name__].__class__ = _CallableModule
