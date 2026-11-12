"""Provenance from text: build stamp, recon-all logs, ``dataset_description.json``.

Pure functions on text: no file access, no network. Stub of milestone 2: the names and
signatures are final, the bodies are written by work package 2.B. ``parse_dataset_description``
is complete already, because the local source, written beside that package, calls it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from surfcensus.records import Provenance
from surfcensus.sources.base import FileRef, FileRole

__all__ = [
    "EDIT_FLAGS",
    "BuildStamp",
    "Provenance",
    "ReconLog",
    "parse_build_stamp",
    "parse_dataset_description",
    "parse_recon_log",
    "subject_provenance",
]

_PACKAGE = "work package 2.B (inventory/provenance.py)"

EDIT_FLAGS: tuple[str, ...] = (
    "-autorecon2-cp",
    "-autorecon2-wm",
    "-autorecon2-pial",
    "-autorecon-pial",
)
"""recon-all flags of a rerun after manual edits."""


@dataclass(frozen=True)
class BuildStamp:
    raw: str
    """First line of ``scripts/build-stamp.txt``, stripped."""
    version: str | None
    """For example ``6.0.1``; None for a development build."""
    major: int | None


@dataclass(frozen=True)
class ReconLog:
    invocations: tuple[tuple[str, ...], ...]
    """Flag names of each recon-all invocation, in file order."""
    build_stamp: str | None
    """From the last ``build-stamp.txt: ...`` line."""
    truncations: tuple[tuple[int, int], ...]
    """Every ``thickness calculation complete, N:M truncations`` line, in file order."""
    holes: tuple[int, int] | None
    """The last ``orig.nofix lhholes = N, rhholes = M`` line."""
    finished: bool | None
    """None when the text is partial or says neither."""


def parse_build_stamp(text: str) -> BuildStamp:
    raise NotImplementedError(_PACKAGE)


def parse_recon_log(text: str, *, partial: bool = False) -> ReconLog:
    """``partial``: the text is the head of a log, so ``finished`` is None."""
    raise NotImplementedError(_PACKAGE)


def subject_provenance(
    texts: Mapping[FileRole, str],
    files: Mapping[FileRole, FileRef],
    subject_label: str,
    *,
    partial: bool = False,
) -> Provenance:
    """``texts``: the decoded content of the build stamp and the logs that could be read."""
    raise NotImplementedError(_PACKAGE)


def _version(entry: object) -> str | None:
    version = entry.get("Version") if isinstance(entry, dict) else None
    return version.strip() if isinstance(version, str) and version.strip() else None


def parse_dataset_description(text: str) -> str | None:
    """The fMRIPrep version in a ``dataset_description.json``, or None.

    ``GeneratedBy[*].Version`` of the entry whose ``Name`` is ``fMRIPrep`` (ignoring case),
    as in ``ds000102-fmriprep`` (fMRIPrep 21.0.2); else ``PipelineDescription.Version``, the
    form of older fMRIPrep versions (from memory, no such file was looked at). Text that
    is not JSON, or JSON of another shape, gives None: this never raises.
    """
    try:
        data = json.loads(text)
    except (ValueError, RecursionError):
        return None
    if not isinstance(data, dict):
        return None
    generated = data.get("GeneratedBy")
    for entry in generated if isinstance(generated, list) else []:
        name = entry.get("Name") if isinstance(entry, dict) else None
        if isinstance(name, str) and name.lower() == "fmriprep" and _version(entry):
            return _version(entry)
    return _version(data.get("PipelineDescription"))
