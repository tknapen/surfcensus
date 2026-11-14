"""The local source: FreeSurfer subjects in a directory, read in place.

A path is recognized as one of the layouts of docs/architecture.md: a ``SUBJECTS_DIR``, an
fMRIPrep output (new and old layout), a BIDS derivatives root, or one subject directory. The
root of a BIDS dataset is no layout of its own: its ``derivatives/`` folder is searched for
a folder in one of those layouts, and the source is the one that folder gives when it is
named itself (:func:`detect_layout`). Subjects are listed with ``stat()`` only; their files
are handed out where they are.

Read-only by construction: nothing here opens for writing, creates or removes.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import re
import shlex
import sys
import types
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from stat import S_ISDIR
from typing import Literal

from surfcensus.config import ConfigError, RunConfig, SourceConfig
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
    Unfetched,
)
from surfcensus.sources.patterns import (
    ROLE_CANDIDATES,
    TEMPLATE_FILES,
    annex_link,
    find_unfetched,
    is_file,
    is_subject_candidate,
    is_template_name,
    resolve_subject_files,
    size_words,
    unfetched_files,
    variant_name,
)

TEMPLATE_NAME = "fsaverage"
"""The directory next to the subjects that can serve as the template."""

DERIVATIVES = "derivatives"
"""The folder of a BIDS dataset that holds what pipelines made of it."""

LAYOUT_WORDS: dict[str, str] = {
    "subjects_dir": "a subjects directory",
    "fmriprep": "fMRIPrep layout",
    "fmriprep_legacy": "older fMRIPrep layout",
    "bids_derivatives": "BIDS derivatives layout",
}
"""Each layout in the words of the lines that ``review`` and ``inventory`` print."""

_DESCRIPTION = "dataset_description.json"
_MAX_NAMES = 10  # names listed when a glob matches nothing
_Notes = list[tuple[int, str]]  # what a detection has to say: logging level, message


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


def _is_text(name: str) -> bool:
    """Whether a name from the file system can be stored and hashed as text. One with bytes
    that are not UTF-8 (Latin-1, which Linux file systems take) cannot."""
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _escape(name: str) -> str:
    """The glob that matches ``name`` and nothing else."""
    return re.sub(r"([*?\[])", r"[\1]", name)


def _named(path: Path) -> bool:
    """Whether ``path`` is a file, or a git-annex link whose content is not fetched: in a
    DataLad clone ``dataset_description.json`` is often annexed, and a file that has a name
    marks a dataset as much as one that has its content. Raises OSError like ``is_file``."""
    return is_file(path) or annex_link(path) is not None


# --- what is not there in a DataLad dataset: subdatasets that are not installed, and the
# command that fetches files whose content is not there ----------------------------------------

_SUBMODULE_PATH = re.compile(r"^\s*path\s*=\s*(.+?)\s*$")


def _ancestors(path: Path) -> list[Path]:
    return [path, *path.parents]


def _is_repository(folder: Path) -> bool:
    return os.path.lexists(folder / ".git")


def _is_dataset(folder: Path) -> bool:
    try:
        return _is_repository(folder) and _is_dir(folder / ".datalad")
    except OSError:
        return False


def uninstalled_subdataset(folder: Path | str) -> Path | None:
    """The repository whose ``.gitmodules`` registers ``folder``, when ``folder`` is an
    empty directory: a subdataset (a git submodule) that is not installed, as a fresh
    ``datalad clone`` leaves every subdataset. None otherwise, also when it cannot be told.
    Reads the ``.gitmodules`` of the folder's ancestors, nothing else."""
    folder = Path(os.path.abspath(folder))
    try:
        if _names(folder) != []:
            return None
    except OSError:
        return None
    for holder in folder.parents:
        modules = holder / ".gitmodules"
        try:
            if not _is_repository(holder) or not is_file(modules):
                continue
            text = modules.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        relative = folder.relative_to(holder).as_posix()
        for line in text.splitlines():
            found = _SUBMODULE_PATH.match(line)
            if found and found.group(1).strip("\"'") == relative:
                return holder
        return None  # the innermost repository decides
    return None


def install_command(folder: Path | str, holder: Path | str) -> str:
    """The command that installs the subdataset ``folder`` of the repository ``holder``
    without fetching any file content: ``datalad -C <holder> get -n <folder>``, or for a
    repository that is no DataLad dataset ``git -C <holder> submodule update --init``."""
    holder = Path(holder)
    relative = shlex.quote(Path(os.path.abspath(folder)).relative_to(holder).as_posix())
    if _is_dataset(holder):
        return f"datalad -C {shlex.quote(str(holder))} get -n {relative}"
    return f"git -C {shlex.quote(str(holder))} submodule update --init {relative}"


def _not_installed(folder: Path, shown: str | None = None) -> str | None:
    """The sentence for a folder that is a subdataset that is not installed, or None;
    ``shown`` is how the message names the folder (the folder itself by default)."""
    holder = uninstalled_subdataset(folder)
    if holder is None:
        return None
    return (
        f"{shown or folder} is a subdataset that is not installed (an empty folder); install "
        f"it, without the file content, with: {install_command(folder, holder)}"
    )


def _fetch_root(path: Path, given: Path | None) -> tuple[Path, bool] | None:
    """Where a fetch of ``path`` is run, and whether that is a DataLad dataset: of the
    DataLad datasets that hold ``path``, the innermost that holds ``given`` (the folder a
    user named), else the outermost below ``given``, else the innermost; without DataLad
    the innermost git repository, for ``git annex get``. None outside any repository."""
    folders = _ancestors(path.parent)
    repositories = [folder for folder in folders if _is_repository(folder)]
    if not repositories:
        return None
    if not _is_dataset(repositories[0]):
        return repositories[0], False
    datasets = [folder for folder in repositories if _is_dataset(folder)]
    if given is not None:
        holding = [folder for folder in datasets if given.is_relative_to(folder)]
        if holding:
            return holding[0], True
        below = [folder for folder in datasets if folder.is_relative_to(given)]
        if below:
            return below[-1], True
    return datasets[0], True


def fetch_command(
    subjects_dir: Path | str,
    wanted: Mapping[str, Sequence[str]],
    given: Path | str | None = None,
) -> str | None:
    """One shell line that fetches the files ``wanted`` names (subject label -> names
    relative to the subject directory, as ``patterns.unfetched_files`` gives them), or None
    when they lie in no git repository.

    One subject: ``datalad -C <dataset> get <paths>``, the paths relative to the dataset;
    ``datalad get`` installs a subdataset on the way when one is needed (DataLad 1.8.0,
    tried 2026-11-14), so the dataset may be the one the user named. Several subjects: ``cd
    <dataset> && datalad get <subjects>/<prefix>*/<name> ...``, one shell pattern per name,
    since the line would otherwise grow with the subjects; the pattern can also match
    subjects whose files are there, which ``datalad get`` passes over. Without DataLad:
    ``git -C <repository> annex get <paths>`` in the repository that holds the files.
    """
    base = Path(subjects_dir).resolve()
    asked = None if given is None else Path(given).resolve()
    labels = sorted(wanted)
    if not labels:
        return None
    try:
        directories = {label: (base / label).resolve() for label in labels}
        first = directories[labels[0]] / wanted[labels[0]][0]
        found = _fetch_root(first, asked)
    except (OSError, RuntimeError, IndexError):
        return None
    if found is None:
        return None
    root, datalad = found
    linked = any(directory.parent != base for directory in directories.values())
    if len(labels) == 1 or linked or not base.is_relative_to(root):
        paths = []
        for label in labels:
            for name in wanted[label]:
                file = directories[label] / name
                if not file.is_relative_to(root):
                    continue
                paths.append(shlex.quote(file.relative_to(root).as_posix()))
        if not paths:
            return None
        tool = "datalad -C {root} get" if datalad else "git -C {root} annex get"
        return f"{tool.format(root=shlex.quote(str(root)))} {' '.join(paths)}"
    prefix = labels[0]
    for label in labels[1:]:  # the labels' common start, character by character
        while not label.startswith(prefix):
            prefix = prefix[:-1]
    names = list(dict.fromkeys(name for label in labels for name in wanted[label]))
    folder = base.relative_to(root).as_posix()
    head = "" if folder == "." else f"{shlex.quote(folder)}/"
    patterns = [
        f"{head}{shlex.quote(prefix) if prefix else ''}*/{shlex.quote(name)}" for name in names
    ]
    tool = "datalad get" if datalad else "git annex get"
    return f"cd {shlex.quote(str(root))} && {tool} {' '.join(patterns)}"


def fetch_line(
    config: RunConfig, problems: Mapping[str, object], given: Path | str | None = None
) -> str | None:
    """The line that says how to fetch the files without content of the subjects of a local
    run with one local source: ``problems`` maps subject labels to their ``problem``, and
    the subjects whose problem names such files (``patterns.unfetched_files``) count. None
    when none does, for any other run, and when the files lie in no git repository.
    ``given`` is the folder the user named, from which the dataset of the command follows
    (:func:`fetch_command`); without it, the BIDS dataset that holds the subjects
    (:func:`dataset_holding`), else the stored root."""
    wanted: dict[str, list[str]] = {}
    total, sized = 0, True
    for label, problem in problems.items():
        found = unfetched_files(problem)
        if found is None:
            continue
        wanted[label] = found[0]
        if found[1] is None:
            sized = False
        else:
            total += found[1]
    stored = [s for s in config.sources if s.kind == "local" and s.root is not None]
    if not wanted or config.mode != "local" or len(stored) != 1 or stored[0].root is None:
        return None
    try:
        record, _ = detect_layout(stored[0].root, nested=False, quiet=True)
    except SourceError:
        return None
    if given is None:
        # the BIDS dataset that holds the subjects when there is one, as review of it says
        given = dataset_holding(record) or Path(record.root)
    command = fetch_command(Path(record.root) / record.fs_root, wanted, given)
    if command is None:
        return None
    them = "them" if len(wanted) == 1 else f"the files of these {len(wanted)} subjects"
    amount = f" ({'at least ' if not sized else ''}{size_words(total)})" if total else ""
    return f"Fetch {them}{amount} with: {command}"


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


def source_places(subjects_dir: Path | str) -> list[tuple[str, Path]]:
    """The directories that a local source on ``subjects_dir`` reads, as ``(what, directory)``
    pairs: the subjects directory, and the directory behind every entry of it that is a link
    to a subject directory or to a template (``fsaverage*``).

    The subjects of a study are often links into another place, and that place is read as
    much as the directory that lists them. Only links are followed, and they are found in
    one listing, so a directory of plain subject directories costs no ``stat`` per subject.
    Every subject directory counts, also one that the subject globs of a run leave out. A
    subjects directory that cannot be listed (the data is not mounted) gives itself alone.
    Not followed: a file that is a link inside a subject directory; the renderer refuses an
    output beside such a file's target when it reads the file (``render.api``).
    """
    place = Path(subjects_dir)
    places = [("the subjects directory", place)]
    try:
        with os.scandir(place) as entries:
            links = sorted(entry.name for entry in entries if entry.is_symlink())
    except OSError:
        return places
    for name in links:
        entry = place / name
        try:
            target = entry.resolve()
            template = is_template_name(name) or is_template_name(target.name)
            if not (_is_dir(entry) if template else is_subject_candidate(entry)):
                continue
        except (OSError, RuntimeError):  # a link that leads nowhere, or in a circle
            continue
        what = "the template directory" if template else "the subject directory"
        places.append((f"{what} that {name} links to,", target))
    return places


def freesurfer_template_places(env: Mapping[str, str] = os.environ) -> list[tuple[str, Path]]:
    """``$FREESURFER_HOME/subjects/fsaverage`` as a place of :func:`place_holding`, when the
    variable is set: the template of a run may have been read from there."""
    home = env.get("FREESURFER_HOME")
    if not home:
        return []
    return [("the fsaverage directory", Path(home) / "subjects" / TEMPLATE_NAME)]


def place_holding(
    target: Path | str, places: Iterable[tuple[str, Path]]
) -> tuple[str, Path] | None:
    """The first of ``places`` that ``target`` is, or lies inside, or None.

    ``target`` is compared twice: as it is written (made absolute, ``..`` taken out, no link
    followed) and with every link resolved. So a path named through a link in the subjects
    directory is inside the source, and so is a path that names the link's target.
    """
    written = Path(os.path.abspath(target))
    resolved = Path(target).resolve()
    for what, place in places:
        named, real = Path(os.path.abspath(place)), place.resolve()
        if (
            written.is_relative_to(named)
            or written.is_relative_to(real)
            or resolved.is_relative_to(real)
        ):
            return what, place
    return None


def refuse_in_sources(
    target: Path | str,
    subjects_dirs: Iterable[Path | str],
    *,
    what: Literal["run", "output"],
    also: Iterable[tuple[str, Path]] = (),
) -> None:
    """Raise ``ConfigError("invalid", ...)`` when ``target`` is, or lies inside, a directory
    that a local source reads (:func:`source_places` of each of ``subjects_dirs``) or one
    of ``also``: sources are only read, so nothing the package writes goes there. ``what``
    says which directory ``target`` is, for the message."""
    places = [place for directory in subjects_dirs for place in source_places(directory)]
    found = place_holding(target, [*places, *also])
    if found is None:
        return
    raise ConfigError(
        "invalid",
        f"the {what} directory {target} lies inside {found[0]} {found[1]}, which is only "
        f"read; choose {'an' if what == 'output' else 'a'} {what} directory outside it",
    )


def stored_subjects_dirs(config: RunConfig) -> list[Path]:
    """The subjects directories of the local sources of a local run, from the roots in its
    ``config.yaml``; where a root holds no subjects now (the data is not mounted), the root
    itself. A census run keeps its roots in the private catalog and gives none here."""
    places = []
    for stored in config.sources:
        if stored.kind != "local" or stored.root is None:
            continue
        try:
            record, _ = detect_layout(stored.root, nested=False, quiet=True)
            places.append(Path(record.root) / record.fs_root)
        except SourceError:
            places.append(Path(stored.root))
    return places


def found_in(record: DatasetRecord, given: Path | str | None = None) -> str | None:
    """Where the subjects of ``record`` lie, for the line that says what was found: the
    subjects directory relative to ``given``, the folder a user named, when it lies below
    that folder (``derivatives/fmriprep/sourcedata/freesurfer`` for a dataset root), else
    relative to the record's root (``sourcedata/freesurfer``). None when the subjects
    directory is that folder itself, or holds it (one subject directory as path)."""
    subjects_dir = Path(record.root) / record.fs_root
    if given is not None:
        try:
            below = subjects_dir.resolve().relative_to(Path(given).resolve())
        except (ValueError, OSError):
            below = None
        if below is not None and below.parts:
            return below.as_posix()
    return None if record.fs_root == "." else record.fs_root


def where_subjects_lie(config: RunConfig, given: Path | str | None = None) -> str | None:
    """:func:`found_in` for the source of a local run with one local source, from the root
    in its ``config.yaml``; None for any other run and when the root holds no subjects now.
    A census run keeps its roots, and where the subjects lie under them, in the catalog."""
    stored = [s for s in config.sources if s.kind == "local" and s.root is not None]
    if config.mode != "local" or len(stored) != 1 or stored[0].root is None:
        return None
    try:
        record, _ = detect_layout(stored[0].root, nested=False, quiet=True)
    except SourceError:
        return None
    return found_in(record, given)


def dataset_holding(record: DatasetRecord, given: Path | str | None = None) -> Path | None:
    """The BIDS dataset in whose ``derivatives/`` folder the subjects of ``record`` lie, or
    None: the outermost directory with a ``dataset_description.json`` that has the
    subjects directory below its ``derivatives/``, or ``given``, the folder a user named,
    when the subjects were found below its ``derivatives/`` (rule 6 of
    :func:`detect_layout`), whatever else it holds. A run directory is not made inside it
    unless it is asked for (``review.default_run_dir``)."""
    subjects_dir = (Path(record.root) / record.fs_root).resolve()
    found = None
    for folder in (subjects_dir, *subjects_dir.parents):  # from the inside out
        if folder.name != DERIVATIVES:
            continue
        try:
            if _named(folder.parent / _DESCRIPTION):
                found = folder.parent
        except OSError:
            continue
    if given is not None:
        asked = Path(given).resolve()
        try:
            below = _is_dir(asked / DERIVATIVES) and subjects_dir.is_relative_to(
                (asked / DERIVATIVES).resolve()
            )
        except OSError:
            below = False
        if below and (found is None or asked in found.parents):
            found = asked
    return found


def refuse_run_in_sources(run_dir: Path | str, config: RunConfig) -> None:
    """Raise ``ConfigError("invalid", ...)`` when the run directory of a local run is, or
    lies inside, a directory that one of its sources reads: a run that was moved into its
    own data is not written to. For the commands that open a run without building its
    sources (the rater app, ``finalize``, ``calibration freeze``)."""
    refuse_in_sources(
        run_dir, stored_subjects_dirs(config), what="run", also=freesurfer_template_places()
    )


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


def detect_layout(
    path: Path | str, *, nested: bool = True, quiet: bool = False
) -> tuple[DatasetRecord, tuple[str, ...]]:
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
    6. ``path`` is the root of a BIDS dataset, which is to say it has a ``derivatives/``
       folder, or it is that folder (by its name, or the name of what it links to). The
       record is the one that rules 2 to 5 give for a folder below, as if that folder had
       been named: ``derivatives/`` itself first (so ``derivatives/freesurfer/``), then
       the folders of ``derivatives/`` that hold subjects themselves
       (``derivatives/freesurfer-7.3.2/``), then those with ``sourcedata/freesurfer/``
       (``derivatives/fmriprep/sourcedata/freesurfer/``), then those with ``freesurfer/``
       (``derivatives/fmriprep/freesurfer/``), each group in name order.

    Rule 6 is a way to find a folder, not a layout: root, identity, layout and pipeline are
    those of the folder found, so the dataset root and that folder name the same source
    with the same ``subject_uid``. Nothing deeper is looked at (three folders below
    ``derivatives/`` at most), nor a pipeline folder that is itself a subject or is named
    like a template. When subjects lie in more than one of these places, also beside a
    match of rules 2 to 5, the first is used and a warning names the others with the
    folder to give for each. A folder below ``derivatives/`` that cannot be examined is
    passed over with a warning, unless nothing else holds a subject.

    ``nested=False`` leaves rule 6 out. A source that is rebuilt from the root a run
    stores is read that way: the stored root is the folder that was found, and a run whose
    subjects are gone from it stops with ``layout`` and does not go on with other subjects
    that lie below. ``quiet=True`` logs nothing, for a caller that only asks where the
    subjects are.

    The record's ``root`` is resolved and its ``identity`` is that root. Raises
    :class:`SourceError`: ``not found`` when ``path`` is not a directory, ``unreadable`` when
    it or one of the places of rules 2 to 5 cannot be listed, ``layout`` when no rule
    matches, and ``invalid`` when the root is not valid UTF-8: a run stores it, and hashes
    it into the pseudonym and every ``subject_uid``, as text.
    """
    given = _keeping_name(path)
    notes: _Notes = []
    try:
        record, implied = _detect(given, notes, nested)
    except OSError as exc:
        where = exc.filename if isinstance(exc.filename, str) else str(given)
        raise SourceError("unreadable", f"{where}: {_strerror(exc)}") from exc
    if not _is_text(record.root):
        raise SourceError(
            "invalid",
            f"the path {record.root!r} is not valid UTF-8; the root of a source is stored "
            "and hashed as text",
        )
    if not quiet:
        log = get_logger()
        for level, message in notes:
            log.log(level, "%s", message)
    return record, implied


def _inside_uninstalled(path: Path) -> str:
    """``; <sentence>`` when the nearest folder of ``path`` that exists is a subdataset that
    is not installed (:func:`_not_installed`), else empty."""
    for folder in Path(os.path.abspath(path)).parents:
        try:
            if not _is_dir(folder):
                continue
        except OSError:
            return ""
        sentence = _not_installed(folder)
        return "" if sentence is None else f"; {sentence}"
    return ""


def _detect(given: Path, notes: _Notes, nested: bool) -> tuple[DatasetRecord, tuple[str, ...]]:
    if not _is_dir(given):
        raise SourceError("not found", f"{given} is not a directory{_inside_uninstalled(given)}")
    if is_subject_candidate(given):
        notes.append((logging.INFO, f"{given} is one FreeSurfer subject directory"))
        return _record(given.parent, ".", "subjects_dir"), (_escape(given.name),)

    base = given.resolve()
    record = _direct(base, (given, base), notes)
    derivatives = None
    if nested:
        try:
            derivatives = _derivatives_of(given, base)
        except OSError:
            if record is None:
                raise
    if derivatives is None:
        if record is None:
            raise SourceError("layout", _nothing_found(base, bids=nested))
        return record, ()

    searched = _Searched()
    found = _below(derivatives, searched, itself=derivatives != base)
    if record is not None:
        found.insert(0, (record, []))
    seen: dict[Path, tuple[DatasetRecord, _Notes]] = {}
    for candidate, said in found:  # one subjects directory can be reached from two folders
        seen.setdefault(_subjects_dir(candidate).resolve(), (candidate, said))
    for place, why in searched.unread.items():
        if seen:  # with nothing found, the error names them
            notes.append(
                (
                    logging.WARNING,
                    f"{_shown(place, base)} could not be examined ({why}) and was not "
                    "searched for FreeSurfer subjects",
                )
            )
    if not seen:
        raise SourceError("layout", _nothing_below(base, derivatives, searched))
    (used, said), *others = seen.values()
    if used is not record:
        notes.extend(said)
        notes.append(
            (
                logging.INFO,
                f"{base} is a BIDS dataset with FreeSurfer subjects in "
                f"{_shown(_subjects_dir(used), base)}",
            )
        )
    if others:
        unused = "; ".join(
            f"{_shown(_subjects_dir(other), base)} (to use it, give {other.root} as the path)"
            for other, _ in others[:_MAX_NAMES]
        )
        notes.append(
            (
                logging.WARNING,
                f"FreeSurfer subjects lie in more than one place under {base}: using "
                f"{_shown(_subjects_dir(used), base)}, not {unused}",
            )
        )
    return used, ()


def _direct(base: Path, names: Sequence[Path], notes: _Notes) -> DatasetRecord | None:
    """Rules 2 to 5 of :func:`detect_layout` for the folder ``base``, which is resolved;
    ``names`` are the folder as it was named and as it resolves. None when no rule matches.
    Raises OSError when one of the places cannot be listed."""
    new, old = base / "sourcedata" / "freesurfer", base / "freesurfer"
    if _has_candidate(new):
        try:
            both = _has_candidate(old)
        except OSError:  # a folder that is not used need not be readable
            both = False
        if both:
            notes.append(
                (
                    logging.WARNING,
                    f"{new} and {old} both hold FreeSurfer subjects; using the first and "
                    f"ignoring {old}",
                )
            )
        notes.append(
            (logging.INFO, f"{base} is an fMRIPrep output with subjects in sourcedata/freesurfer")
        )
        return _record(base, "sourcedata/freesurfer", "fmriprep")
    if _has_candidate(old):
        legacy = _is_dir(base / "fmriprep")
        beside = "next to an fmriprep/ folder" if legacy else "BIDS derivatives"
        notes.append((logging.INFO, f"{base} holds subjects in freesurfer/ ({beside})"))
        return _record(base, "freesurfer", "fmriprep_legacy" if legacy else "bids_derivatives")
    # the folder as the user named it, then what it links to
    for folder in dict.fromkeys(names):
        if folder.name == "fmriprep" and _has_candidate(folder.parent / "freesurfer"):
            notes.append((logging.INFO, f"{folder} is an fMRIPrep folder next to freesurfer/"))
            return _record(folder.parent, "freesurfer", "fmriprep_legacy")
    if _has_candidate(base):
        notes.append((logging.INFO, f"{base} is a FreeSurfer subjects directory"))
        return _record(base, ".", "subjects_dir")
    return None


# --- the root of a BIDS dataset: the search below derivatives/ (rule 6) ---------------------------


@dataclasses.dataclass
class _Searched:
    """What the search below ``derivatives/`` went through, for what is said about it."""

    folders: list[str] = dataclasses.field(default_factory=list)
    """The folders of ``derivatives/`` that were searched, by name."""
    unread: dict[Path, str] = dataclasses.field(default_factory=dict)
    """Places that could not be examined, each with the system's reason."""

    def failed(self, folder: Path, exc: OSError) -> None:
        """Note that ``folder`` could not be searched: under the place the error names,
        when it names one, since that may be a folder inside it."""
        place = Path(exc.filename) if isinstance(exc.filename, str) else folder
        self.unread.setdefault(place, _strerror(exc))


_ORDER = {".": 0, "sourcedata/freesurfer": 1, "freesurfer": 2}
"""The order of the folders of ``derivatives/``, by where each keeps its subjects."""


def _subjects_dir(record: DatasetRecord) -> Path:
    return Path(record.root) / record.fs_root


def _shown(place: Path, base: Path) -> str:
    """``place`` relative to ``base`` where it lies below it, for a message; else as it is."""
    try:
        below = place.relative_to(base)
    except ValueError:
        return str(place)
    return below.as_posix() if below.parts else "the folder itself"


def _derivatives_of(given: Path, base: Path) -> Path | None:
    """The ``derivatives/`` folder that rule 6 searches: that of ``base``, or ``base`` when
    it is that folder itself, by the name it was given under or the name of what that
    links to; None when there is none. Raises OSError when it cannot be examined."""
    folder = base / DERIVATIVES
    if _is_dir(folder):
        return folder.resolve()
    if DERIVATIVES in (given.name, base.name):
        return base
    return None


def _below(
    derivatives: Path, searched: _Searched, *, itself: bool
) -> list[tuple[DatasetRecord, _Notes]]:
    """The records that rules 2 to 5 give for ``derivatives`` (with ``itself``) and for each
    of its folders, in the order of rule 6, each with what its detection has to say. A
    folder that is a subject candidate, is named like a template or has a name that is not
    valid UTF-8 is not searched; one that cannot be examined goes to ``searched.unread``.
    """

    def probe(folder: Path, names: Sequence[Path]) -> list[tuple[DatasetRecord, _Notes]]:
        said: _Notes = []
        try:
            record = _direct(folder, names, said)
        except OSError as exc:
            searched.failed(folder, exc)
            return []
        return [] if record is None else [(record, said)]

    found = probe(derivatives, (derivatives,)) if itself else []
    try:
        names = _names(derivatives) or []
    except OSError as exc:
        searched.failed(derivatives, exc)
        names = []
    folders = []
    for name in names:
        folder = derivatives / name
        try:
            if (
                not _is_dir(folder)
                or not _is_text(name)
                or is_template_name(name)
                or is_subject_candidate(folder)
            ):
                continue
            searched.folders.append(name)
            folders.extend(probe(folder.resolve(), (folder, folder.resolve())))
        except OSError as exc:
            searched.failed(folder, exc)
    # names are sorted, and the sort keeps their order within one kind of folder
    folders.sort(key=lambda item: _ORDER[item[0].fs_root])
    return found + folders


def _looks_like_bids(base: Path) -> bool:
    """Whether ``base`` has the two marks of a BIDS dataset: ``dataset_description.json``
    and a ``sub-*`` folder."""
    try:
        if not _named(base / _DESCRIPTION):
            return False
        return any(n.startswith("sub-") and _is_dir(base / n) for n in _names(base) or ())
    except OSError:
        return False


def _without_subjects(derivatives: Path, folders: Sequence[str]) -> list[tuple[Path, bool]]:
    """The folders below ``derivatives`` that are there for FreeSurfer output and hold no
    subject, for the message of a search that found none: ``sourcedata/freesurfer`` and
    ``freesurfer`` in it and in each of ``folders``, and a folder whose own name starts
    with ``freesurfer``; each with whether it is empty. An empty one is what a DataLad
    subdataset that is not installed looks like (:func:`_uninstalled_below` tells)."""
    places = []
    for folder in (derivatives, *(derivatives / name for name in folders)):
        places += [folder / "sourcedata" / "freesurfer", folder / "freesurfer"]
        if folder != derivatives and folder.name.lower().startswith("freesurfer"):
            places.append(folder)
    found: dict[Path, bool] = {}
    for place in places:
        try:
            names = _names(place)
        except OSError:
            continue
        if names is not None:
            found.setdefault(place, not names)
    return list(found.items())


def _uninstalled_below(base: Path, derivatives: Path, folders: Sequence[str]) -> list[str]:
    """The sentences of :func:`_not_installed` for ``derivatives``, its ``folders`` and the
    folders for FreeSurfer output in them that are subdatasets that are not installed,
    each folder named relative to ``base``."""
    places = [derivatives, *(derivatives / name for name in folders)]
    places += [place for place, empty in _without_subjects(derivatives, folders) if empty]
    said = [_not_installed(place, _shown(place, base)) for place in dict.fromkeys(places)]
    return [sentence for sentence in said if sentence is not None]


def _nothing_found(base: Path, *, bids: bool) -> str:
    """The detail of ``layout`` for a path without a ``derivatives/`` folder."""
    if bids and _looks_like_bids(base):
        return (
            f"no FreeSurfer output found in {base}: it looks like a BIDS dataset without "
            "derivatives. Looked for subjects (directories with mri/ and surf/) in "
            f"{DERIVATIVES}/ (there is no such folder), sourcedata/freesurfer/, freesurfer/ "
            "and the dataset's own folders; if they are elsewhere, give that folder as the path"
        )
    unreadable = _unexaminable(base)
    return (
        f"no FreeSurfer subject found at {base}. A subject is a directory with mri/ and "
        "surf/ that is not a template; looked at the path itself, the folders in "
        "sourcedata/freesurfer/, in freesurfer/, in ../freesurfer/ (when the path is named "
        "fmriprep) and the path's own folders"
        + (f". Could not be examined: {', '.join(unreadable)}" if unreadable else "")
    )


def _nothing_below(base: Path, derivatives: Path, searched: _Searched) -> str:
    """The detail of ``layout`` for a dataset root, or a ``derivatives/`` folder, in which
    the search of rule 6 found no subject."""
    where = "the path" if derivatives == base else f"{DERIVATIVES}/"
    listed = ", ".join(searched.folders[:_MAX_NAMES]) or "none"
    more = len(searched.folders) - _MAX_NAMES
    uninstalled = _uninstalled_below(base, derivatives, searched.folders)
    if uninstalled:
        # nothing can be found before the subdataset is there: that is all there is to say
        return f"no FreeSurfer subject found in {base}: " + "; ".join(uninstalled[:_MAX_NAMES])
    detail = (
        f"no FreeSurfer subject found in {base}. A subject is a directory with mri/ and "
        f"surf/ that is not a template; looked in {where} and its folders ({listed}"
        f"{f' and {more} more' if more > 0 else ''}), in each for subject directories, "
        "sourcedata/freesurfer/ and freesurfer/"
    )
    if derivatives != base:
        detail += ", and at the path's own sourcedata/freesurfer/, freesurfer/ and folders"
    bare = [
        f"{_shown(place, base)}{' (empty)' if empty else ''}"
        for place, empty in _without_subjects(derivatives, searched.folders)
    ]
    if bare:
        detail += f". Without a subject: {', '.join(bare[:_MAX_NAMES])}"
    unread = [f"{_shown(place, base)} ({why})" for place, why in searched.unread.items()]
    unread += _unexaminable(base)
    if unread:
        detail += f". Could not be examined: {', '.join(unread[:_MAX_NAMES])}"
    return detail


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
    derivatives root, one subject directory, or the root of a BIDS dataset with one of
    these below its ``derivatives/`` folder. ``subjects`` are globs on directory names
    (a union; empty means all). ``identity`` is given when a stored source is rebuilt, so
    that subjects keep their ``subject_uid`` after the data moved; otherwise it is the
    resolved root. ``nested`` says whether the path is searched below ``derivatives/``
    (rule 6 of :func:`detect_layout`); by default it is unless an ``identity`` is given. A
    source that is rebuilt from the root a run stores is not searched (``run.from_config``
    says so): that root is the folder that was found. Raises ``SourceError`` when the path
    is not a directory (``not found``), cannot be listed (``unreadable``) or matches no
    layout (``layout``).

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
        nested: bool | None = None,
    ) -> None:
        if field_strength is not None and not 0 < field_strength < math.inf:
            raise ValueError(
                f"field_strength is in tesla, positive and finite, not {field_strength}"
            )
        given = (subjects,) if isinstance(subjects, str) else tuple(subjects)
        globs = tuple(dict.fromkeys(given))
        for glob in globs:
            if not _is_text(glob):  # a run stores the globs of its source as text
                raise SourceError("invalid", f"the subject glob {glob!r} is not valid UTF-8")
        searched = identity is None if nested is None else nested
        record, implied = detect_layout(path, nested=searched)
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
            if not _is_text(name):
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
            unfetched: dict[FileRole, Unfetched] = {}
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
                    files, unfetched = _list_files(directory)
                except OSError as exc:
                    problem = f"unreadable: {_strerror(exc)}"
            records.append(SubjectRecord(dataset, label, fs_dir, files, problem, unfetched))
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


def _list_files(directory: Path) -> tuple[dict[FileRole, FileRef], dict[FileRole, Unfetched]]:
    """One :class:`FileRef` per role that ``directory`` has, from ``stat()`` alone, and the
    roles whose candidate is a git-annex link without content (``patterns.find_unfetched``,
    from ``readlink``). Size and modification time are those of the file a link leads to."""
    paths = resolve_subject_files(directory)
    unfetched = find_unfetched(directory, paths)
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
    return files, unfetched


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
