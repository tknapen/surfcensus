"""A run: a run directory, its sources and the phases that fill it.

``Run`` is the Python API of docs/architecture.md; the CLI is a thin layer over it. It only
delegates: each method imports the phase it calls. ``from_config`` is complete; the body of
``Run`` is written in the last step of milestone 2 (work package 2.Z), and the metrics
methods in milestone 3.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from surfcensus.config import ConfigError, RunConfig, SourceConfig
from surfcensus.privacy import HashKey
from surfcensus.records import PhaseReport, RunContext
from surfcensus.sources.base import Source

if TYPE_CHECKING:
    import httpx
    import pandas as pd

    from surfcensus.store import PublicCatalog

METRIC_STAGES: tuple[str, ...] = ("regional", "fsaverage", "reference", "score")

_PACKAGE = "work package 2.Z (run.py)"
_METRICS = "metrics arrive in milestone 3"


class Run:
    """``Run(run_dir, source)`` creates a run; ``Run(run_dir)`` opens an existing one.

    Creating: ``config.yaml`` (a local run gets a new salt, each source contributes its
    ``config()``) and the catalog with its run secret; a census run requires ``key``.
    Opening rebuilds the sources from ``config.yaml``. A local source whose root matches no
    stored one raises ``ConfigError("other root", ...)`` unless ``relocate``, which replaces
    the stored root and keeps every ``subject_uid``.
    """

    run_dir: Path
    config: RunConfig

    def __init__(
        self,
        run_dir: Path | str,
        source: Source | Sequence[Source] | None = None,
        *,
        config: RunConfig | None = None,
        key: HashKey | None = None,
        relocate: bool = False,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        raise NotImplementedError(_PACKAGE)

    def inventory(
        self, *, datasets: Collection[str] | None = None, force: bool = False
    ) -> PhaseReport:
        """``datasets``: accessions or pseudonyms, here and in the other phases."""
        raise NotImplementedError(_PACKAGE)

    def metrics(
        self,
        *,
        datasets: Collection[str] | None = None,
        stages: Collection[str] = METRIC_STAGES,
        jobs: int = 1,
        force: bool = False,
        retry_failed: bool = False,
    ) -> PhaseReport:
        raise NotImplementedError(_METRICS)

    def render(
        self,
        *,
        jobs: int = 1,
        datasets: Collection[str] | None = None,
        subjects: Sequence[str] | None = None,
        all_subjects: bool | None = None,
        video: bool | None = None,
        force: bool = False,
        retry_failed: bool = False,
    ) -> PhaseReport:
        raise NotImplementedError(_PACKAGE)

    def scores(self) -> pd.DataFrame:
        """The primary rows of ``tables/scores.parquet``, sorted by rank."""
        raise NotImplementedError(_METRICS)

    def clusters(self) -> pd.DataFrame:
        """``clusters.parquet`` left-joined with ``clusters_native.parquet``."""
        raise NotImplementedError(_METRICS)

    def public(self) -> PublicCatalog:
        raise NotImplementedError(_PACKAGE)

    def context(self) -> RunContext:
        """What phase functions receive in place of the run."""
        raise NotImplementedError(_PACKAGE)

    def media_dir(self, subject_uid: str) -> Path:
        raise NotImplementedError(_PACKAGE)

    def close(self) -> None:
        raise NotImplementedError(_PACKAGE)

    def __enter__(self) -> Run:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def from_config(
    config: SourceConfig,
    run_config: RunConfig,
    run_dir: Path,
    *,
    root: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> Source:
    """The source that ``config`` describes.

    ``root`` overrides ``config.root``: a local source in a census run is stored without a
    path, and its root comes from the catalog. The remote kinds arrive in milestone 5; they
    take their client and cache from ``run_config``, ``run_dir`` and ``transport``.
    """
    if config.kind != "local":
        raise NotImplementedError(f"the {config.kind} source arrives in milestone 5")
    from surfcensus.sources.local import LocalSource

    path = root if root is not None else config.root
    if path is None:
        raise ConfigError("invalid", "a local source needs a root")
    return LocalSource(
        path,
        subjects=config.subjects,
        identity=config.identity,
        field_strength=config.field_strength,
    )
