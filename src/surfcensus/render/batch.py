"""Phase 6 for a run: render the subjects of a run into ``media/<subject_uid>/``.

Stub of milestone 2: the names and signatures are final, the bodies are written by work
package 2.F. The ``template`` argument of ``run_render`` is added in milestone 3, when
``templates.py`` exists.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from pathlib import Path

from surfcensus.records import PhaseReport, RunContext

_PACKAGE = "work package 2.F (render/batch.py)"


def run_render(
    ctx: RunContext,
    *,
    jobs: int = 1,
    datasets: Collection[str] | None = None,
    subjects: Sequence[str] | None = None,
    all_subjects: bool | None = None,
    video: bool | None = None,
    force: bool = False,
    retry_failed: bool = False,
) -> PhaseReport:
    """Render the pending subjects and record every outcome in the ``renders`` table.

    ``datasets`` are pseudonyms and ``subjects`` are ``subject_uid``s, rendered in the order
    given. ``all_subjects`` and ``video`` default to what the run's configuration says.
    """
    raise NotImplementedError(_PACKAGE)


def render_check_sample(ctx: RunContext, *, fraction: float = 0.02, seed: int) -> Path:
    """Draw a seeded sample of the done renders for the by-eye check of phase 6 and write it
    to ``logs/render-check-<seed>.txt`` and ``.html``; returns the text file."""
    raise NotImplementedError(_PACKAGE)
