"""Plain records that more than one module needs.

They live here so that no module has to import another one's module to name a type. This
module imports the standard library and ``sources/base.py`` only; the types of other
modules that its fields name are imported for the type checker alone.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from surfcensus.sources.base import FileRole, Hemi

if TYPE_CHECKING:
    import pandas as pd

    from surfcensus.config import RunConfig
    from surfcensus.render.model import RenderInfo
    from surfcensus.store import PublicCatalog

Status = Literal["done", "failed"]

METRIC_STAGES: tuple[str, ...] = ("regional", "fsaverage", "reference", "score")
"""The stages of phase 4 that ``surfcensus metrics`` runs when none is named, in order."""

ALL_METRIC_STAGES: tuple[str, ...] = (*METRIC_STAGES, "native", "merge")
"""Every stage that can be named: ``native`` gives cluster centroids their native position
from the vertex maps of the renders, ``merge`` joins the per-dataset parts of each table."""


@dataclass(frozen=True)
class Provenance:
    """What the inventory learns about a subject from its build stamp, logs and file names.

    Built by ``inventory.provenance.subject_provenance`` and stored by
    ``Catalog.upsert_subject``. A None says that the fact could not be established, which is
    different from False.
    """

    build_stamp: str | None
    fs_version: str | None
    """For example ``6.0.1``; None for a development build."""
    fs_major: int | None
    flags: tuple[str, ...]
    """Names of the recon-all flags of all invocations of all logs, sorted, without values."""
    n_invocations: int | None
    finished: bool | None
    t2pial: bool | None
    """None: no log was read and the pial variant does not say; likewise ``flairpial``."""
    flairpial: bool | None
    hires: bool | None
    expert: bool | None
    xopts: bool | None
    edited: bool | None
    pial_variant: str | None
    """``pial``, ``pial.T1``, ``pial.T2`` or ``pial.FLAIR``."""
    holes: tuple[int, int] | None
    """Surface holes before topology fixing, left and right."""
    truncations: Mapping[Hemi, tuple[int, int] | None]
    """The two truncation counts of the thickness measurement, in the order of the log."""
    participant: str | None
    """``sub-01`` of ``sub-01_ses-2``."""
    longitudinal: bool


@dataclass(frozen=True)
class PhaseReport:
    """What a phase command did, for the closing line and the exit code."""

    phase: str
    done: int
    skipped: int
    failed: int
    """Units in the failed state when the command ends, retried or not."""
    failures: tuple[tuple[str, str], ...]
    """``(unit, reason)``; a unit is a ``subject_uid``, ``dataset:<pseudonym>`` or ``run``."""
    notes: tuple[str, ...] = ()
    """Sentences for the user."""
    seconds: float = 0.0


@dataclass(frozen=True)
class WorkRow:
    """A row of the ``work`` ledger: one unit of one phase, done or failed."""

    phase: str
    unit: str
    """A ``subject_uid``, ``dataset:<pseudonym>`` or ``run``."""
    config_hash: str
    input_signature: str
    """Private: it never leaves the catalog (see ``sources.base.input_signature``)."""
    status: Status
    reason: str | None = None
    detail: str | None = None
    """Private; the scrubbed text of the failure."""
    output: str | None = None
    """Path relative to the run directory, when the unit has one output file."""
    surfcensus_version: str | None = None
    """None in a row to be recorded: the catalog writes the running version."""
    finished_at: str | None = None
    """UTC, ISO 8601. None in a row to be recorded: the catalog writes the current time."""


@dataclass(frozen=True)
class RenderRow:
    """A row of the ``renders`` table: the render of one subject, done or failed."""

    subject_uid: str
    config_hash: str
    """``RenderConfig.config_hash()``."""
    input_signature: str
    """Private, like ``WorkRow.input_signature``."""
    video: bool
    status: Status
    tile_hash: str | None = None
    """``RenderConfig.tile_hash()``; None for a failed render, likewise ``render_key``."""
    render_key: str | None = None
    reason: str | None = None
    """A ``RenderError`` reason, or ``source changed``."""
    detail: str | None = None
    media_path: str | None = None
    """``media/<subject_uid>``."""
    n_frames: int | None = None
    template_id: str | None = None
    rendered_at: str | None = None
    """UTC, ISO 8601. None in a row to be recorded: the catalog writes the current time."""


@dataclass(frozen=True)
class RenderResult:
    """What a render worker returns to the parent process."""

    info: RenderInfo
    extras: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    """Table name -> the rows of that table for this subject. Empty until the intensity
    metric is computed inside the render (milestone 5)."""


@dataclass(frozen=True)
class RunContext:
    """What a phase function may touch: the public side of a run.

    Phase code receives this and never a ``Run``, so it cannot reach the private catalog
    (accessions, roots, input signatures) even by attribute access.
    """

    run_dir: Path
    config: RunConfig
    public: PublicCatalog
    lease: Callable[[str, Collection[FileRole]], AbstractContextManager[Mapping[FileRole, Path]]]
    """``lease(subject_uid, roles)``: local paths of the roles the subject has, while open."""
    render_key: Callable[[str, Mapping[str, str]], str]
    """``render_key(tile_hash, digests)``: ``privacy.render_key`` with the run secret bound."""
    scrub: Callable[[str], str]
    """``privacy.scrub`` with this run's pseudonyms and roots."""
