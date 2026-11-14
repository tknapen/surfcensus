"""Where the functional part of a dataset of a run lies: the derivative, the raw BIDS dataset
and the FreeSurfer subjects directory, by the rules of docs/epi_interface.md, "Which folders
are found by rule" (step 2, work package E2.B).

One implementation of the rules, not two. Step 1 wrote them in ``func.api``; they live here
now (``find_folders``, ``FuncFolders``, ``refuse_out`` and the places that are read), and
``func.api`` imports them, so ``surfcensus epi-session`` and a run directory find the same
folders for the same path. What this module adds for a run:

* :func:`func_source_of` starts from the dataset record the surface source found (its root and
  ``fs_root``), never from a search of its own below ``derivatives/``: a stored root is not
  searched again (decision D35), and the derivative of a run is the one that holds the run's
  FreeSurfer subjects, else the folder itself, else the first folder of ``derivatives/`` that a
  connector recognises.
* :func:`read_places` and :func:`stored_read_places`: the derivative and the raw dataset as
  places that are only read, so that a run directory or an export there is refused like one
  in a FreeSurfer source (``sources.local.refuse_in_sources``; the rule of D35 extended), and
  :func:`places_of_subjects_dir`, the same places found from a FreeSurfer subjects directory
  alone, which ``sources.local.refuse_in_sources`` adds to the places of every subjects
  directory it is given.

Private side of a run, like ``inventory/subjects.py``: it handles roots. It lists folders and
reads ``dataset_description.json``; it opens no other file, fetches nothing and writes nothing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from surfcensus.func import bids
from surfcensus.func import connectors as connectors_
from surfcensus.func.connectors.base import Connector, ConnectorError
from surfcensus.records import ConnectorMatch, DerivFound, FuncSourceRecord, RawFound
from surfcensus.sources.base import SourceError

if TYPE_CHECKING:
    from surfcensus.config import RunConfig
    from surfcensus.sources.base import DatasetRecord


@dataclass(frozen=True)
class FuncFolders:
    """The three folders a session needs, as :func:`find_folders` found them."""

    deriv_root: Path
    match: ConnectorMatch
    raw_root: Path | None
    """None: the runs are listed with ``no raw series`` and the message names ``--raw``."""
    fs_subjects_dir: Path | None
    """None: the sessions are packed without contours."""


# --- the folders ----------------------------------------------------------------------------


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(path))


def _detect(folder: Path) -> tuple[Connector, ConnectorMatch] | None:
    try:
        return connectors_.detect(folder)
    except ConnectorError as err:
        raise SourceError("unreadable", f"{folder}: {err.detail}") from None


def _subjects_dir_of(folder: Path) -> Path | None:
    """The FreeSurfer subjects directory that the surface source finds for an fMRIPrep
    output ``folder`` (``sourcedata/freesurfer/``, or ``freesurfer/`` beside it), or None."""
    from surfcensus.sources.local import detect_layout

    try:
        record, _ = detect_layout(folder, nested=False, quiet=True)
    except SourceError:
        return None
    if record.layout not in ("fmriprep", "fmriprep_legacy"):
        return None
    return Path(record.root) / record.fs_root


def _deriv_candidates(record_root: Path, fs_root: str, layout: str) -> list[Path]:
    """Where the fMRIPrep output lies for a surface layout: the folder above
    ``sourcedata/freesurfer``, or the ``fmriprep/`` folder beside ``freesurfer/``."""
    subjects = record_root / fs_root
    if layout == "fmriprep":
        return [record_root]
    if layout == "fmriprep_legacy":
        return [record_root / "fmriprep"]
    out = []
    if subjects.name == "freesurfer" and subjects.parent.name == "sourcedata":
        out.append(subjects.parent.parent)
    if subjects.name == "freesurfer":
        out.append(subjects.parent / "fmriprep")
    return out


def _derivative(
    given: Path, record: DatasetRecord | None = None
) -> tuple[Path, ConnectorMatch, Path | None, DerivFound]:
    """The derivative of ``given``, its match, the FreeSurfer subjects directory, and how the
    derivative was found. With ``record`` (a run: the dataset the surface source found and
    stored), the surface layout is that record's; without it, the one
    ``sources.local.detect_layout`` finds for ``given``."""
    from surfcensus.sources.local import detect_layout

    looked: list[Path] = []
    # 1. what the surface source finds, then the fMRIPrep output that holds those subjects
    if record is None:
        try:
            record, _ = detect_layout(given, quiet=True)
        except SourceError as err:
            if err.reason in ("not found", "invalid"):
                raise
            record = None
    if record is not None:
        subjects = Path(record.root) / record.fs_root
        for candidate in _deriv_candidates(Path(record.root), record.fs_root, record.layout):
            looked.append(candidate)
            if candidate.is_dir():
                found = _detect(candidate)
                if found is not None:
                    return candidate, found[1], subjects, "surface layout"
    # 2. the folder itself is a derivative (one that holds no FreeSurfer subjects, say)
    looked.append(given)
    found = _detect(given)
    if found is not None:
        return given, found[1], _subjects_dir_of(given), "folder itself"
    # 3. the folders of derivatives/, in name order
    derivatives = given if given.name == "derivatives" else given / "derivatives"
    if derivatives.is_dir():
        try:
            names = sorted(e.name for e in os.scandir(derivatives) if e.is_dir())
        except OSError as err:
            raise SourceError("unreadable", f"{derivatives}: {err.strerror or err}") from None
        for name in names:
            candidate = derivatives / name
            looked.append(candidate)
            found = _detect(candidate)
            if found is not None:
                return candidate, found[1], _subjects_dir_of(candidate), "derivatives folder"
    shown = ", ".join(_shown(p, given) for p in dict.fromkeys(looked))
    raise SourceError(
        "layout",
        f"no derivative of a known pipeline ({', '.join(connectors_.CONNECTORS)}) with "
        f"functional runs found in {given}: looked at {shown}. A derivative has a "
        "dataset_description.json that names the pipeline and sub-*/[ses-*/]func/ folders",
    )


def _is_raw(folder: Path) -> bool:
    try:
        return bids.is_raw_dataset(folder)
    except OSError as err:
        raise SourceError("unreadable", f"{folder}: {err.strerror or err}") from None


def _raw_found(deriv: Path, raw: Path | None) -> tuple[Path | None, RawFound | None]:
    if raw is not None:
        raw = _absolute(raw)
        if not raw.is_dir():
            raise SourceError("not found", f"--raw {raw} is not a directory")
        if not _is_raw(raw):
            raise SourceError(
                "layout",
                f"--raw {raw} holds no raw BIDS functional series "
                "(sub-*/[ses-*/]func/*_bold.nii[.gz])",
            )
        return raw, "given"
    for chain in (_absolute(deriv), deriv.resolve()):
        for folder in chain.parents:
            if folder.name == "derivatives" and _is_raw(folder.parent):
                return folder.parent, "dataset root"
    inside = deriv / "sourcedata" / "raw"
    if inside.is_dir() and _is_raw(inside):
        return inside, "sourcedata/raw"
    return None, None


def _raw_dataset(deriv: Path, raw: Path | None) -> Path | None:
    return _raw_found(deriv, raw)[0]


def find_folders(path: Path, raw: Path | None = None) -> FuncFolders:
    """The derivative, the raw dataset and the FreeSurfer subjects directory of ``path`` by
    the rules of docs/epi_interface.md, "Which folders are found by rule", for a local folder.

    The derivative: the fMRIPrep output that holds the FreeSurfer subjects that the surface
    source finds for ``path`` (``sources.local.detect_layout``: the folder above
    ``sourcedata/freesurfer/``, or ``fmriprep/`` beside ``freesurfer/``); else ``path`` itself
    when a connector recognises it (a derivative without FreeSurfer subjects); else the first
    folder of ``path/derivatives/`` (or of ``path`` when it is ``derivatives/``), in name
    order, that a connector recognises. The raw dataset: ``raw``; else the BIDS dataset whose
    ``derivatives/`` holds the derivative; else ``sourcedata/raw/`` inside the derivative;
    else None. The FreeSurfer subjects directory: the one the surface source finds, or None.

    Raises ``SourceError``: ``not found`` for a path that is no directory, ``layout`` when no
    derivative of a known pipeline is found (the message says where it looked) or ``raw``
    holds no raw series, ``unreadable`` for a folder that cannot be listed.
    """
    given = _absolute(path)
    if not given.is_dir():
        raise SourceError("not found", f"{given} is not a directory")
    deriv, match, subjects, _ = _derivative(given)
    return FuncFolders(
        deriv_root=deriv,
        match=match,
        raw_root=_raw_dataset(deriv, raw),
        fs_subjects_dir=subjects,
    )


def func_source_of(record: DatasetRecord, raw: Path | None = None) -> FuncSourceRecord | None:
    """The functional source of a local dataset of a run, or None when its folder has no
    functional part (no derivative of a known connector with ``sub-*/[ses-*/]func/``): then
    no functional table gets a row and nothing is said but one line at info level.

    ``raw`` is ``EpiConfig.raw`` (``--raw DIR``); without it the raw dataset is found by rule
    and may be None (its runs are listed as ``no raw series`` and the message names
    ``--raw``). ``deriv_identity`` is the resolved derivative root; the caller keeps a stored
    identity (``Catalog.upsert_func_source``). Raises ``SourceError`` as ``find_folders``
    does: ``unreadable`` for a folder that cannot be listed, ``layout`` or ``not found`` for a
    ``raw`` that holds no raw functional series or is no directory.

    The rules are those of :func:`find_folders` with the record's surface layout in place of
    a new search: the fMRIPrep output that holds the record's subjects, else the record's
    root when a connector recognises it, else the first folder of its ``derivatives/`` (the
    root itself when it is ``derivatives/``) that one recognises.
    """
    given = _absolute(record.root)
    if not given.is_dir():
        raise SourceError("not found", f"{given} is not a directory")
    try:
        deriv, match, _, how = _derivative(given, record)
    except SourceError as err:
        if err.reason == "layout":
            return None
        raise
    raw_root, raw_how = _raw_found(deriv, raw)
    return FuncSourceRecord(
        connector=match.connector,
        match=match,
        deriv_root=str(deriv),
        deriv_identity=str(deriv.resolve()),
        raw_root=None if raw_root is None else str(raw_root),
        deriv_found=how,
        raw_found=raw_how,
    )


def folders_of(source: FuncSourceRecord, fs_subjects_dir: Path | str | None) -> FuncFolders:
    """The :class:`FuncFolders` of a stored functional source, with the FreeSurfer subjects
    directory of its dataset (the surface source's), for the planning of ``func.inventory``."""
    return FuncFolders(
        deriv_root=Path(source.deriv_root),
        match=source.match,
        raw_root=None if source.raw_root is None else Path(source.raw_root),
        fs_subjects_dir=None if fs_subjects_dir is None else Path(fs_subjects_dir),
    )


# --- places that are only read ----------------------------------------------------------------


def _places(roots: tuple[tuple[str, Path | None], ...]) -> list[tuple[str, Path]]:
    places: list[tuple[str, Path]] = []
    for what, root in roots:
        if root is None:
            continue
        places.append((what, root))
        try:
            with os.scandir(root) as entries:
                links = sorted(e.name for e in entries if e.is_symlink())
        except OSError:
            continue
        for name in links:
            try:
                target = (root / name).resolve()
            except (OSError, RuntimeError):
                continue
            if target.is_dir():
                places.append((f"the directory that {name} in {what} links to,", target))
    return places


def _read_places(folders: FuncFolders) -> list[tuple[str, Path]]:
    """The derivative and the raw dataset, and what a link at their top level leads to; the
    places of the FreeSurfer subjects directory come from ``sources.local.source_places``."""
    return _places((("the derivative", folders.deriv_root), ("the raw dataset", folders.raw_root)))


def read_places(source: FuncSourceRecord) -> list[tuple[str, Path]]:
    """The derivative and the raw dataset of ``source`` and what a link at their top level
    leads to, as ``(what, directory)`` pairs for ``sources.local.refuse_in_sources(also=...)``.
    The FreeSurfer subjects directory is a place of the surface source already."""
    raw = None if source.raw_root is None else Path(source.raw_root)
    return _places((("the derivative", Path(source.deriv_root)), ("the raw dataset", raw)))


def places_of_subjects_dir(subjects_dir: Path | str) -> list[tuple[str, Path]]:
    """The derivative and the raw dataset that a FreeSurfer subjects directory is read with,
    found by rule from it alone (the fMRIPrep output above ``sourcedata/freesurfer/`` or
    ``fmriprep/`` beside ``freesurfer/``, when a connector recognises it as a derivative with
    functional runs; the raw dataset of that derivative), as :func:`read_places` gives them;
    empty when there is none or a folder cannot be examined. A raw dataset named with
    ``--raw`` cannot be found from here: :func:`stored_read_places` adds it.

    A derivative with functional runs counts whatever the run's settings say (``--no-epi``):
    its files are a source of the tool, as every subject directory counts whatever the subject
    globs of a run select (``sources.local.source_places``)."""
    subjects = _absolute(subjects_dir)
    try:
        for candidate in _deriv_candidates(subjects, ".", "freesurfer"):
            if not candidate.is_dir():
                continue
            found = _detect(candidate)
            if found is None:
                continue
            return _read_places(
                FuncFolders(candidate, found[1], _raw_dataset(candidate, None), subjects)
            )
    except (SourceError, OSError):
        return []
    return []


def stored_read_places(config: RunConfig) -> list[tuple[str, Path]]:
    """:func:`read_places` of every local source of ``config``, found again from its stored
    root and ``EpiConfig.raw`` by listings alone (no catalog), for the commands that refuse a
    run directory from ``config.yaml`` (``sources.local.refuse_run_in_sources``: the rater
    app, ``finalize``, ``calibration freeze``, ``export``). A root that is not there gives no
    place. A raw dataset named with ``--raw`` is a place whether or not it can be used."""
    from surfcensus.config import epi_settings
    from surfcensus.sources.local import detect_layout

    places: list[tuple[str, Path]] = []
    if config.mode != "local":
        return places
    raw = epi_settings(config).raw
    for stored in config.sources:
        if stored.kind != "local" or stored.root is None:
            continue
        try:
            record, _ = detect_layout(stored.root, nested=False, quiet=True)
            source = func_source_of(record, None)
        except (SourceError, OSError):
            source = None
        if source is not None:
            places += read_places(source)
    if raw is not None:
        places += _places((("the raw dataset", _absolute(raw)),))
    return places


def refuse_out(out: Path, folders: FuncFolders) -> None:
    """Raise ``ConfigError("invalid", ...)`` when ``out`` is, or lies inside, a folder that is
    read (the raw dataset, the derivative, the FreeSurfer subjects directory) or what a link
    at the top level of one of them leads to."""
    from surfcensus.sources.local import refuse_in_sources

    subjects = [folders.fs_subjects_dir] if folders.fs_subjects_dir is not None else []
    refuse_in_sources(out, subjects, what="output", also=_read_places(folders))


def _shown(path: Path, base: Path) -> str:
    """``path`` relative to ``base`` when it lies inside it (``.`` for ``base`` itself), else
    as it is."""
    if path == base:
        return "."
    if path.is_relative_to(base):
        return path.relative_to(base).as_posix()
    return str(path)
