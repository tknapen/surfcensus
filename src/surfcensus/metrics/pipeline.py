"""Phase 4 for a run: the metric stages, their skip tests and their tables. See
docs/phases.md.

A phase function receives a ``RunContext`` and never the run, so it names subjects by
``subject_uid`` and datasets by pseudonym.

Interface of milestone 3; the functions are written by package 3.Z.
"""

from __future__ import annotations

from collections.abc import Collection

from surfcensus.records import METRIC_STAGES, PhaseReport, RunContext
from surfcensus.templates import Template

_OWNER = "package 3.Z (pipeline)"


def run_metrics(
    ctx: RunContext,
    *,
    template: Template | None,
    datasets: Collection[str] | None = None,
    stages: Collection[str] = METRIC_STAGES,
    jobs: int = 1,
    force: bool = False,
    retry_failed: bool = False,
) -> PhaseReport:
    """Run the metric stages named in ``stages`` (names from ``records.ALL_METRIC_STAGES``)
    on what is pending. Run-level stages use every present subject and ignore ``datasets``."""
    raise NotImplementedError(f"metrics.pipeline.run_metrics: {_OWNER}")


def update_native(
    ctx: RunContext, *, template: Template | None, force: bool = False
) -> PhaseReport:
    """The stage ``native``: ``clusters_native.parquet`` from ``clusters.parquet`` and the
    vertex maps of the done renders. Reads no source file."""
    raise NotImplementedError(f"metrics.pipeline.update_native: {_OWNER}")


def merge_tables(ctx: RunContext) -> PhaseReport:
    """The stage ``merge``: every ``tables/<table>.parquet`` that has parts, from its parts."""
    raise NotImplementedError(f"metrics.pipeline.merge_tables: {_OWNER}")
