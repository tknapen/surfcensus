"""``surfcensus export``: one table per analysis unit, from the catalog, the metric tables
and ``app.sqlite``. See docs/phases.md, phase 8.

Datasets appear by pseudonym and subjects by ``subject_uid``; a local run also names its
subjects. Nothing here can reach what is private (decision D20): the catalog is opened as a
``PublicCatalog``, and ``app.sqlite`` is read with the standard library, by the names of
``surfcensus.app``, so the export works on a core install.

Interface of milestone 4. The names of the files and the columns of the rating tables are
final; the functions are written by package 4.E.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

_OWNER = "package 4.E (export)"

Format = Literal["csv", "parquet"]
Profile = Literal["analysis", "release"]
"""``analysis``: everything, pseudonymous, for the team's notebooks; the only profile of a
local run. ``release``: what a census publishes (coarser, selected subjects only)."""

EXPORT_DIRS: dict[Profile, str] = {"analysis": "export", "release": "release"}
"""Where a profile is written under the run directory when no other place is given."""

EXPORT_TABLES: tuple[str, ...] = (
    "subjects",
    "scores",
    "selection",
    "ratings",
    "marks",
    "final_marks",
    "clusters",
    "datasets",
    "behaviour",
)
"""Each is written as ``<name>.csv`` or ``<name>.parquet``."""
MANIFEST = "manifest.json"

RATINGS_COLUMNS: tuple[str, ...] = (
    "assignment_id",
    "subject_uid",
    "rater",
    "kind",
    "render_key",
    "render_current",
    "A",
    "B",
    "C",
    "unratable_reason",
    "other",
    "note",
    "revision",
    "protocol_version",
    "protocol_sha256",
    "app_version",
    "client_ts",
    "server_ts",
    "ms_on_subject",
)
"""``ratings``: one row per completed assignment, from its current submission.
``render_current`` says whether the rating was made on the tiles the subject has now."""

SUBJECT_RATING_COLUMNS: tuple[str, ...] = (
    "final_A",
    "final_B",
    "final_C",
    "final_method",
    "unratable_reason",
    "other",
    "n_ratings",
    "n_marks",
)
"""The last columns of ``subjects``: the final labels, whether any current submission
flagged another problem, and how many completed assignments and current marks the subject
has."""

BEHAVIOUR_COLUMNS: tuple[str, ...] = (
    "assignment_id",
    "subject_uid",
    "rater",
    "ms_on_subject",
    "n_frames_viewed",
    "n_plays",
    "n_pauses",
    "n_mode_switches",
    "n_back",
)
"""``behaviour``: one row per completed assignment, summed over its events."""


def ratings_frame(run_dir: Path | str) -> pd.DataFrame:
    """The ``ratings`` table (:data:`RATINGS_COLUMNS`), sorted by ``assignment_id``; also
    behind ``GET /api/export`` and the ratings of the review screen. Empty, with its
    columns, for a run without ``app.sqlite``."""
    raise NotImplementedError(f"export.tables.ratings_frame: {_OWNER}")


def subjects_frame(run_dir: Path | str) -> pd.DataFrame:
    """The ``subjects`` table, sorted by ``subject_uid``; also behind ``GET /api/review``.

    ``subject_uid``, ``dataset``, in a local run ``subject_label`` and ``participant``, the
    provenance columns of ``PublicCatalog.subjects()``, ``render_status``,
    ``render_reason``, every column of the primary row of ``scores.parquet``, then
    :data:`SUBJECT_RATING_COLUMNS`. The rating columns are empty for a run without
    ``app.sqlite``."""
    raise NotImplementedError(f"export.tables.subjects_frame: {_OWNER}")


def export_run(
    run_dir: Path | str,
    out_dir: Path | str | None = None,
    *,
    fmt: Format = "csv",
    profile: Profile = "analysis",
) -> list[Path]:
    """Write every table of :data:`EXPORT_TABLES` and ``manifest.json`` into ``out_dir``
    (default: ``export/`` or ``release/`` of the run directory), atomically and sorted;
    returns the paths written, the manifest last.

    Works on a run without ratings and on one without ``app.sqlite``. The manifest holds
    the run id, mode, profile, key fingerprint, package version, stage hashes, template id,
    reference ids, the versions and hashes of the protocol and the calibration set, and the
    row count and SHA-256 of every exported file. It has no reveal argument and cannot
    reach an accession.
    """
    raise NotImplementedError(f"export.tables.export_run: {_OWNER}")
