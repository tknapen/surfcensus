"""``surfcensus review PATH``: inventory, metrics, selection, render and the rater app on a
local directory of FreeSurfer subjects, in one call. See docs/architecture.md.

Interface of milestone 4; the function is written by package 4.Z.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from surfcensus.run import Run

_OWNER = "package 4.Z (integrator)"

DEFAULT_RUN_DIR = "surfcensus-run"
"""The name of the run directory that ``review`` makes beside the data when none is given."""


def review(
    path: Path | str,
    *,
    run_dir: Path | str | None = None,
    subjects: Sequence[str] | None = None,
    rater: str | None = None,
    blind: bool | None = None,
    require_marks: bool | None = None,
    field_strength: float | None = None,
    jobs: int | None = None,
    fsaverage: Path | str | None = None,
    video: bool | None = None,
    port: int = 8765,
    open_browser: bool = True,
    serve: bool = True,
    serve_after: int | None = None,
    relocate: bool = False,
    force: bool = False,
) -> Run:
    """Inventory, score, select and render the subjects under ``path`` and open the rater
    app on them; returns the run.

    Every option that maps to a stored value is tri-state: None keeps what the run has, and
    the defaults (not blind, marks not required, no video) apply only when the run is
    created. ``rater`` is not stored: it names the rater of this server process.

    The app opens when the first ``serve_after`` subjects of the queue are rendered, and
    the others are rendered while it runs, in the order of the queue
    (``store.queue_order``). None is 1, so the rating screen comes up at the first render
    (decision D33); 0 renders every subject first. ``serve=False`` renders every subject
    and stops. ``force`` is needed to change ``blind`` once a rating exists. Running it
    again does the same steps with everything unchanged skipped.
    """
    raise NotImplementedError(f"review.review: {_OWNER}")
