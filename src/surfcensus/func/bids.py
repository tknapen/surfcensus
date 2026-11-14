"""BIDS names of functional runs: entities, run keys, name order, sidecar times, and the
listing of the raw series of a participant. Work package E1.A.

A run is keyed by ``sub``, ``ses``, ``task``, ``acq``, ``ce``, ``rec``, ``dir``, ``run`` and
``echo`` (docs/epi_interface.md, The unit), the run index compared as a number, so that
``run-01`` and ``run-1`` are one run (stage 1 found that necessary in 3,102 of 9,462 runs).
A dataset without ``ses-`` folders has one session per participant. Everything here works from
names and the listing; only :func:`sidecar_time` opens a file.

What a name must be to name a run. Its first part is ``sub-<label>``, every label of the nine
entities is made of ASCII letters and digits, and the ``run`` and ``echo`` labels are made of
ASCII digits (at most 18). A name that is not so is no run: :func:`parse_entities` raises
``ValueError`` for it, and the listers pass it over. This is stricter than reading every name
that has a ``sub-`` part, and it keeps a run key unambiguous: a label holds no ``_`` or ``-``.

Errors. :func:`parse_entities` raises ``ValueError``; :func:`list_raw_runs` and
:func:`is_raw_dataset` raise ``OSError`` for a directory or a file that cannot be examined,
with the location relative to the folder given in place of a path, so that an unreadable run
is never taken for an absent one (decision D19). Nothing else here raises on any text or name.
"""

from __future__ import annotations

import datetime
import errno
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path

from surfcensus.records import DerivRun, FuncFileRef, RawRun, RunEntities
from surfcensus.sources.base import FuncRole

ENTITIES: tuple[str, ...] = ("sub", "ses", "task", "acq", "ce", "rec", "dir", "run", "echo")
"""The entities of :class:`~surfcensus.records.RunEntities`, in BIDS order."""
ORDER_ENTITIES: tuple[str, ...] = ("task", "acq", "ce", "rec", "dir")
"""Compared by name in the name-order rule, before the run index and the echo index."""
RAW_SUFFIXES: tuple[str, ...] = ("_bold.nii.gz", "_bold.nii")
SIDECAR_SUFFIX = "_bold.json"

_LABEL = re.compile(r"[A-Za-z0-9]+")
_INDEX = re.compile(r"[0-9]{1,18}")
_KEY_ENTITIES = ("ses", "task", "acq", "ce", "rec", "dir")
_MAGNITUDE = (None, "mag")
"""``part`` labels of a series that is a run: none, or the magnitude of complex data. The
phase, real and imaginary parts are not runs of their own (stage 1's listing did the same)."""
_RAW_KEYS = (*ENTITIES, "part", "chunk")
"""The entities a raw BOLD series' name may have (BIDS); a name with any other, such as a
derivative's ``space-`` or ``desc-``, is no raw series."""
_SIDECAR_MAX_BYTES = 4 * 1024 * 1024
"""A larger file named like a sidecar is not read: it is not one."""


def _parts(name: str) -> list[tuple[str, str]]:
    """The ``key-label`` parts of the last part of a name, before its first ``.``, in order."""
    base = str(name).replace("\\", "/").rsplit("/", 1)[-1].split(".", 1)[0]
    out = []
    for part in base.split("_"):
        key, sep, label = part.partition("-")
        if sep:
            out.append((key, label))
    return out


def _entity_dict(name: str) -> dict[str, str]:
    """Every ``key-label`` part of a name, the first of a key kept (no check of the labels)."""
    out: dict[str, str] = {}
    for key, label in _parts(name):
        out.setdefault(key, label)
    return out


def parse_entities(name: str) -> RunEntities:
    """The entities of a BIDS file name or stem (a path is allowed; its last part is read).
    Raises ``ValueError`` for a name without ``sub-``. Unknown entities are ignored; of an
    entity named twice the first is kept.

    Also ``ValueError`` for a name whose first part is not ``sub-<label>``, for a label of one
    of :data:`ENTITIES` that is not ASCII letters and digits, and for a ``run`` or ``echo``
    label that is not ASCII digits (module docstring). The message quotes no more than the
    entity."""
    if not isinstance(name, str):
        raise ValueError("a BIDS name is text")
    parts = _parts(name)
    if not parts or parts[0][0] != "sub":
        raise ValueError("not a BIDS name: it does not start with sub-<label>")
    found: dict[str, str] = {}
    for key, label in parts:
        if key in ENTITIES and key not in found:
            pattern = _INDEX if key in ("run", "echo") else _LABEL
            if not pattern.fullmatch(label):
                raise ValueError(f"not a BIDS name: the {key} label is not a valid label")
            found[key] = label
    return RunEntities(
        sub=found["sub"],
        ses=found.get("ses"),
        task=found.get("task"),
        acq=found.get("acq"),
        ce=found.get("ce"),
        rec=found.get("rec"),
        dir=found.get("dir"),
        run=found.get("run"),
        echo=found.get("echo"),
    )


def run_index(label: str | None) -> int | None:
    """A run or echo index as a number; None for a missing or unreadable one (anything but
    1 to 18 ASCII digits)."""
    if not isinstance(label, str) or not _INDEX.fullmatch(label):
        return None
    return int(label)


def run_key(entities: RunEntities) -> str:
    """A run within its participant, the same for the raw series and its derivative files:
    ``ses``, ``task``, ``acq``, ``ce``, ``rec``, ``dir`` as written and the run index as a
    number (``ses-1_task-rest_run-1`` for ``run-01``), no subject and no echo, so that the
    echoes of a multi-echo raw run share the key of the run the derivative has.

    A run label that is not a number (only in entities made by hand) is kept as written."""
    parts = [f"{k}-{getattr(entities, k)}" for k in _KEY_ENTITIES if getattr(entities, k)]
    if entities.run is not None:
        index = run_index(entities.run)
        parts.append(f"run-{index if index is not None else entities.run}")
    return "_".join(parts)


def name_order(entities: RunEntities) -> tuple[str | int, ...]:
    """Sort key of the name-order rule: task by name, then ``acq``, ``ce``, ``rec`` and
    ``dir`` by name (absent first), then the run index as a number (absent first), then the
    echo index.

    An absent label is ``""`` and an absent or unreadable index ``-1``, so that the tuples of
    any two runs compare."""
    labels = tuple(getattr(entities, k) or "" for k in ORDER_ENTITIES)
    run = run_index(entities.run)
    echo = run_index(entities.echo)
    return (*labels, -1 if run is None else run, -1 if echo is None else echo)


def run_name(entities: RunEntities) -> str:
    """The BIDS name of a run without the subject, as the pages show it:
    ``ses-1_task-rest_run-1`` (labels as written)."""
    return "_".join(f"{k}-{getattr(entities, k)}" for k in ENTITIES[1:] if getattr(entities, k))


_TIME = re.compile(r"([0-9]{1,2}):([0-9]{1,2}):([0-9]{1,2}(?:\.[0-9]{1,12})?)")
_EPOCH = datetime.datetime(1970, 1, 1)


def acquisition_seconds(value: object) -> tuple[str, float] | None:
    """``AcquisitionTime`` of a sidecar as ``(kind, seconds)``: kind ``time`` for a time of day
    ``H:M:S[.f]`` (sidecars write it without leading zeros, as in ``17:30:2.685000``), kind
    ``datetime`` for an ISO date and time; None when it cannot be read.

    A time of day is seconds since midnight (hours below 24, minutes below 60, seconds below
    61); it says nothing about the date, so a session that runs past midnight is ordered
    wrongly (not handled, as in stage 1). A date and time is seconds since 1970-01-01, an
    offset that it states taken into account; a date alone orders nothing and is None."""
    if not isinstance(value, str) or len(value) > 64:
        return None
    text = value.strip()
    m = _TIME.fullmatch(text)
    if m:
        h, minute, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
        if h < 24 and minute < 60 and s < 61:
            return "time", h * 3600 + minute * 60 + s
        return None
    if "T" not in text and " " not in text:
        return None
    if not text.isascii():
        return None
    try:
        stamp = datetime.datetime.fromisoformat(text)
    except ValueError:
        return None
    if stamp.tzinfo is not None:
        offset = stamp.utcoffset()
        stamp = stamp.replace(tzinfo=None) - (offset or datetime.timedelta(0))
    try:
        return "datetime", (stamp - _EPOCH).total_seconds()
    except OverflowError:  # pragma: no cover - timedelta covers every datetime
        return None


def sidecar_time(path: Path) -> str | None:
    """``AcquisitionTime`` of a JSON sidecar as written, None when the file cannot be read or
    the field is absent or not a string. A file larger than 4 MiB is not read."""
    try:
        with open(path, "rb") as handle:
            data = handle.read(_SIDECAR_MAX_BYTES + 1)
    except (OSError, ValueError):
        return None
    if len(data) > _SIDECAR_MAX_BYTES:
        return None
    try:
        value = json.loads(data.decode("utf-8-sig"))
    except (ValueError, RecursionError):
        return None
    if not isinstance(value, dict):
        return None
    found = value.get("AcquisitionTime")
    return found if isinstance(found, str) else None


# --- the listing ----------------------------------------------------------------------------


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix() or "."
    except ValueError:
        return path.name


def _unreadable(error: OSError, path: Path, root: Path) -> OSError:
    """The error of examining ``path``, naming it relative to ``root`` and not by its path."""
    return OSError(error.errno, f"{_relative(path, root)}: {error.strerror or 'cannot be read'}")


def _is_text(name: str) -> bool:
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:  # a name of bytes that are not UTF-8 (surrogate escapes)
        return False
    return True


def list_dir(folder: Path, root: Path) -> tuple[dict[str, os.stat_result], list[str]]:
    """The regular files (links followed) and the directories of ``folder`` by name, the
    directories sorted. Names that are not UTF-8 are passed over; a dangling link is no file.
    A folder that does not exist is empty. Raises ``OSError`` (relative to ``root``) for a
    folder or an entry that cannot be examined. Shared with the fMRIPrep connector."""
    files: dict[str, os.stat_result] = {}
    dirs: list[str] = []
    try:
        entries = list(os.scandir(folder))
    except (FileNotFoundError, NotADirectoryError):
        return files, dirs
    except OSError as error:
        raise _unreadable(error, folder, root) from None
    for entry in entries:
        if not _is_text(entry.name):
            continue
        try:
            st = entry.stat()
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError as error:
            if error.errno == errno.ELOOP:
                continue
            raise _unreadable(error, folder / entry.name, root) from None
        if stat.S_ISREG(st.st_mode):
            files[entry.name] = st
        elif stat.S_ISDIR(st.st_mode):
            dirs.append(entry.name)
    return files, sorted(dirs)


def func_dirs(root: Path, participant: str) -> list[tuple[str | None, Path]]:
    """The ``func/`` folders of a participant: ``(None, root/sub-X/func)`` and
    ``(label, root/sub-X/ses-<label>/func)`` for every session folder, sessions sorted. Only
    folders that exist. Raises ``OSError`` as :func:`list_dir`."""
    if not valid_participant(participant):
        return []
    subject = Path(root) / participant
    _, dirs = list_dir(subject, Path(root))
    out: list[tuple[str | None, Path]] = []
    if "func" in dirs:
        out.append((None, subject / "func"))
    for name in dirs:
        if name.startswith("ses-") and _LABEL.fullmatch(name[4:]):
            _, inner = list_dir(subject / name, Path(root))
            if "func" in inner:
                out.append((name[4:], subject / name / "func"))
    return out


def valid_participant(participant: object) -> bool:
    """Whether ``participant`` is ``sub-<label>`` with a label of ASCII letters and digits."""
    return (
        isinstance(participant, str)
        and participant.startswith("sub-")
        and bool(_LABEL.fullmatch(participant[4:]))
    )


def file_ref(
    role: FuncRole, base: str, path: Path, root: Path, st: os.stat_result, variant: str | None
) -> FuncFileRef:
    """A listed file as a :class:`FuncFileRef` (``base`` is ``raw`` or ``deriv``)."""
    return FuncFileRef(
        role=role,
        base="raw" if base == "raw" else "deriv",
        location=_relative(path, root),
        size=int(st.st_size),
        etag_or_mtime=str(st.st_mtime_ns),
        variant=variant,
    )


def is_raw_dataset(folder: Path) -> bool:
    """Whether ``folder`` holds ``sub-*/[ses-*/]func/*_bold.nii[.gz]``.

    Only names that are runs count (module docstring). False for a folder that does not
    exist; ``OSError`` for one that cannot be listed. A participant or session folder that
    cannot be listed is passed over here (the lister names it)."""
    root = Path(folder)
    _, dirs = list_dir(root, root)
    for participant in dirs:
        if not valid_participant(participant):
            continue
        try:
            places = func_dirs(root, participant)
        except OSError:
            continue
        for _, func in places:
            try:
                files, _ = list_dir(func, root)
            except OSError:
                continue
            if any(_raw_entities(name, participant) is not None for name in files):
                return True
    return False


def _raw_entities(name: str, participant: str) -> tuple[RunEntities, str] | None:
    """The entities and the stem (without ``_bold``) of a raw series' file name, or None for a
    name that is no raw run of ``participant``."""
    suffix = next((s for s in RAW_SUFFIXES if name.endswith(s)), None)
    if suffix is None:
        return None
    stem = name[: -len(suffix)]
    try:
        entities = parse_entities(stem)
    except ValueError:
        return None
    if f"sub-{entities.sub}" != participant or entities.task is None:
        return None
    parts = stem.split("_")
    if any(part.partition("-")[0] not in _RAW_KEYS or "-" not in part for part in parts):
        return None  # a derivative's name (space-, desc-) or no BIDS name
    if _entity_dict(stem).get("part") not in _MAGNITUDE:
        return None
    return entities, stem


def _applies(sidecar: Mapping[str, str], series: Mapping[str, str]) -> bool:
    """BIDS inheritance: every entity of the sidecar's name is that of the series, run and
    echo indices compared as numbers."""
    for key, label in sidecar.items():
        if key not in series:
            return False
        if key in ("run", "echo"):
            if run_index(label) is None or run_index(label) != run_index(series[key]):
                return False
        elif label != series[key]:
            return False
    return True


def _sidecar(
    root: Path,
    func: Path,
    stem: str,
    listings: Mapping[Path, Mapping[str, os.stat_result]],
) -> tuple[Path, os.stat_result] | None:
    """The sidecar of the raw series ``stem`` in ``func``: the one beside it, else the nearest
    inherited one (the levels from ``func`` up to ``root``; within a level the one with the
    most entities, then the first name)."""
    series = _entity_dict(stem)
    level = func
    while True:
        files = listings.get(level, {})
        candidates = []
        for name in files:
            if not name.endswith(SIDECAR_SUFFIX):
                continue
            entities = _entity_dict(name[: -len(SIDECAR_SUFFIX)])
            if _applies(entities, series):
                candidates.append((-len(entities), name))
        if candidates:
            name = min(candidates)[1]
            return level / name, files[name]
        if level == root or root not in level.parents:
            return None
        level = level.parent


def list_raw_runs(raw_root: Path, participant: str) -> list[RawRun]:
    """The raw series of ``participant`` (``sub-01``) in a raw BIDS dataset, from the listing,
    in :func:`name_order`, each with its sidecar beside it or inherited from a higher level
    (BIDS inheritance) when the listing has one. Locations are relative to ``raw_root``.

    Each echo of a multi-echo acquisition is a raw run of its own, with the ``key`` of the
    run (no echo in it) and its ``entities.echo``; two raw runs with one key and no echo are
    two names of one run (``run-1`` beside ``run-01``) and both are listed. Of a ``.nii.gz``
    and a ``.nii`` with one stem only the first is listed. A file in a ``ses-`` folder whose
    name names another session, the phase or real part of complex data (``part-``), and a
    name that is no run are passed over. Sorted by session (none first), name order, stem.
    The sidecar of a level is the one of that level with the most entities that apply; levels
    are not merged, so a time in a higher level than the nearest sidecar is not seen. Raises
    ``OSError`` as :func:`list_dir`."""
    root = Path(raw_root)
    if not valid_participant(participant):
        return []
    places = func_dirs(root, participant)
    if not places:
        return []
    listings: dict[Path, dict[str, os.stat_result]] = {}
    listings[root] = list_dir(root, root)[0]
    listings[root / participant] = list_dir(root / participant, root)[0]
    for session, func in places:
        if session is not None:
            listings[func.parent] = list_dir(func.parent, root)[0]
        listings[func] = list_dir(func, root)[0]
    runs = []
    for session, func in places:
        seen: set[str] = set()
        for suffix in RAW_SUFFIXES:
            for name in sorted(listings[func]):
                if not name.endswith(suffix):
                    continue
                parsed = _raw_entities(name, participant)
                if parsed is None:
                    continue
                entities, stem = parsed
                if entities.ses != session or stem in seen:
                    continue
                seen.add(stem)
                files = {
                    FuncRole.RAW_BOLD: file_ref(
                        FuncRole.RAW_BOLD, "raw", func / name, root, listings[func][name], None
                    )
                }
                sidecar = _sidecar(root, func, stem, listings)
                if sidecar is not None:
                    files[FuncRole.RAW_SIDECAR] = file_ref(
                        FuncRole.RAW_SIDECAR, "raw", sidecar[0], root, sidecar[1], None
                    )
                runs.append(
                    RawRun(
                        entities=entities,
                        key=run_key(entities),
                        order=name_order(entities),
                        stem=stem,
                        files=files,
                    )
                )
    runs.sort(key=lambda r: (r.entities.ses is not None, r.entities.ses or "", r.order, r.stem))
    return runs


def session_labels(runs: Sequence[RawRun | DerivRun]) -> list[str | None]:
    """The sessions the runs belong to, sorted, None (no ``ses-``) first."""
    labels = {run.entities.ses for run in runs}
    named = sorted(label for label in labels if label is not None)
    return ([None] if None in labels else []) + list(named)
