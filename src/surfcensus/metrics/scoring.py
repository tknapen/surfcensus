"""The composite score: subject-level metrics on one scale, and their maximum. See
docs/metrics.md. A ranking device for selection and review, not a classifier.

Every component is oriented so that larger is more suspicious. pandas and scipy only.

Interface of milestone 3; the functions are written by package 3.D.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import pandas as pd

from surfcensus.config import RegionConfig, ScoringConfig

_OWNER = "package 3.D (scoring)"

FALLBACK_COMPONENTS: tuple[str, ...] = ("ceiling_frac", "occ_ctrl_ratio")
"""What subjects are ranked on when no z-map can be computed (decision D16)."""


def subject_metrics(
    vertex_summary: pd.DataFrame,
    clusters: pd.DataFrame | None,
    regional: pd.DataFrame,
    subjects: pd.DataFrame,
    regions: RegionConfig,
    *,
    mass: Literal["excess", "sum_z"],
) -> pd.DataFrame:
    """One row per subject and ``reference_id`` with the metric columns of
    ``schemas.SCORES`` (``schemas.NO_REFERENCE`` when ``clusters`` is None).

    ``vertex_summary``, ``clusters`` and ``regional`` are the tables of those names;
    ``subjects`` is ``PublicCatalog.subjects()`` for the present subjects. Per-hemisphere
    metrics (largest-cluster mass and area, peak z, ceiling fractions) take the larger of
    the two hemispheres; the regional ratio pools both. Cluster metrics are 0 for a subject
    that has a z-map and no cluster, and null for one without a z-map. The ``mass_*``
    columns hold ``excess_mass`` or ``mass`` according to ``mass``.
    """
    raise NotImplementedError(f"metrics.scoring.subject_metrics: {_OWNER}")


def normal_scores(values: pd.Series) -> pd.Series:
    """Rank-based normal scores: ``norm.ppf((r - 0.5) / n)`` with ``r`` the minimum rank of
    the value among the ``n`` values that are not NaN (1 = smallest; tied values share the
    lowest rank of the tie). NaN stays NaN; one value gives 0."""
    raise NotImplementedError(f"metrics.scoring.normal_scores: {_OWNER}")


def robust_z(values: pd.Series) -> pd.Series:
    """``(x - median) / scale``, with the first usable scale of ``1.4826 * MAD``,
    ``IQR / 1.349``, the standard deviation (ddof 1) and 1. Usable: finite and above
    ``1e-6 * max(|x - median|)``. NaN stays NaN."""
    raise NotImplementedError(f"metrics.scoring.robust_z: {_OWNER}")


def quantile_scores(values: pd.Series, grid: Sequence[float]) -> pd.Series:
    """Normal scores against a reference population: ``grid`` holds its 0th to 100th
    percentiles of the component; the value's percentile is interpolated in it, clipped to
    [0.5, 99.5] and passed to ``norm.ppf``."""
    raise NotImplementedError(f"metrics.scoring.quantile_scores: {_OWNER}")


def composite(
    metrics: pd.DataFrame,
    strata: pd.Series,
    config: ScoringConfig,
    *,
    norms: Mapping[str, Sequence[float]] | None = None,
) -> pd.DataFrame:
    """``metrics`` with ``std_<component>`` for each component of ``config``, ``composite``
    (the maximum over the components that are not NaN), ``composite_component`` (the first
    in config order that attains it), ``n_components`` and ``score_kind`` =
    ``composite_z``.

    Each component is standardised over ``config.population``, within one
    ``reference_id``: the whole run, or the subject's stratum (``strata``: ``subject_uid``
    -> stratum name or None). With ``norms`` (component -> percentile grid) the scores are
    :func:`quantile_scores`. A subject with fewer components than configured gets the flag
    ``partial_score``.
    """
    raise NotImplementedError(f"metrics.scoring.composite: {_OWNER}")


def fallback_rank(metrics: pd.DataFrame) -> pd.DataFrame:
    """The ranking of decision D16: per component of :data:`FALLBACK_COMPONENTS` the
    percentile ``(r - 1) / (n - 1)`` with ``r`` the minimum rank among the ``n`` subjects
    with a value (0 when ``n`` is 1); ``composite`` is the larger of the two,
    ``score_kind`` is ``fallback_rank`` and ``reference_id`` is ``schemas.NO_REFERENCE``."""
    raise NotImplementedError(f"metrics.scoring.fallback_rank: {_OWNER}")


def add_rank(scores: pd.DataFrame) -> pd.DataFrame:
    """``scores`` with ``rank`` within each ``reference_id``: 1 is the highest composite;
    ties go by the mean of the ``std_*`` scores that are not NaN, descending, then by
    ``subject_uid``; subjects without a composite come after all others."""
    raise NotImplementedError(f"metrics.scoring.add_rank: {_OWNER}")
