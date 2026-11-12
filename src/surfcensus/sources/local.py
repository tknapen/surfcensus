"""The local source: FreeSurfer subjects in a directory, read in place.

Stub of milestone 2: the names and signatures are final, the bodies are written by work
package 2.A. Read-only by construction: nothing here may open for writing, create or remove.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path

from surfcensus.config import SourceConfig
from surfcensus.sources.base import DatasetRecord, FileRole, SourceKind, SubjectRecord

_PACKAGE = "work package 2.A (sources/local.py)"


class LocalSource:
    """A directory of FreeSurfer subjects in one of the layouts of docs/architecture.md.

    ``path`` is what the user passes: a ``SUBJECTS_DIR``, an fMRIPrep output, a BIDS
    derivatives root, or one subject directory. ``subjects`` are globs on directory names
    (a union; empty means all). ``identity`` is given when a stored source is rebuilt, so
    that subjects keep their ``subject_uid`` after the data moved; otherwise it is the
    resolved root. Raises ``SourceError("layout", ...)`` when no layout matches.
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
        raise NotImplementedError(_PACKAGE)

    def datasets(self, only: Collection[str] | None = None) -> Iterator[DatasetRecord]:
        raise NotImplementedError(_PACKAGE)

    def subjects(self, dataset: DatasetRecord) -> Iterator[SubjectRecord]:
        raise NotImplementedError(_PACKAGE)

    def read_bytes(
        self,
        subject: SubjectRecord,
        role: FileRole,
        *,
        byte_range: tuple[int, int] | None = None,
    ) -> bytes:
        raise NotImplementedError(_PACKAGE)

    def lease(
        self, subject: SubjectRecord, roles: Collection[FileRole]
    ) -> AbstractContextManager[Mapping[FileRole, Path]]:
        raise NotImplementedError(_PACKAGE)

    def template_dir(self, dataset: DatasetRecord) -> Path | None:
        raise NotImplementedError(_PACKAGE)

    def config(self) -> SourceConfig:
        """``SourceConfig(kind="local", root, identity, subjects, field_strength)``."""
        raise NotImplementedError(_PACKAGE)


def local(
    path: Path | str, *, subjects: Sequence[str] = (), field_strength: float | None = None
) -> LocalSource:
    """``surfcensus.sources.local(path)``: a local source on ``path``."""
    return LocalSource(path, subjects=subjects, field_strength=field_strength)


def detect_layout(path: Path) -> tuple[DatasetRecord, tuple[str, ...]]:
    """The dataset record of ``path`` and the subject globs its layout implies (a single
    subject directory implies its own name). Raises ``SourceError("layout", ...)``."""
    raise NotImplementedError(_PACKAGE)


def find_adjacent_template(subjects_dir: Path) -> Path | None:
    """``subjects_dir/fsaverage`` when it holds every file of ``patterns.TEMPLATE_FILES`` for
    both hemispheres, else None."""
    raise NotImplementedError(_PACKAGE)


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
