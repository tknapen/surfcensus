"""Regional thickness from ``stats/?h.aparc.stats`` (Tier A): cheap, and available when
surfaces are missing. See docs/metrics.md.

Interface of milestone 3; the functions are written by package 3.A.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from surfcensus.config import RegionConfig
from surfcensus.fsio.stats import StatsFile
from surfcensus.sources.base import Hemi

_OWNER = "package 3.A (stats and regional metrics)"


def regional_rows(subject_uid: str, hemi: Hemi, stats: StatsFile) -> pd.DataFrame:
    """The rows of ``regional.parquet`` (``schemas.REGIONAL``) for one hemisphere of one
    subject: every parcel of the stats file."""
    raise NotImplementedError(f"metrics.regional.regional_rows: {_OWNER}")


def region_mean_thickness(rows: pd.DataFrame, parcels: Sequence[str]) -> float:
    """Mean of ``thick_avg_mm`` over ``parcels``, weighted by ``surf_area_mm2``, with both
    hemispheres pooled. Parcels that the rows lack are left out; NaN when none is there."""
    raise NotImplementedError(f"metrics.regional.region_mean_thickness: {_OWNER}")


def occ_ctrl_ratio(rows: pd.DataFrame, regions: RegionConfig) -> tuple[float, float, float]:
    """``(occipital mm, control mm, occipital / control)`` of one subject, from
    ``regions.occipital`` and ``regions.control``."""
    raise NotImplementedError(f"metrics.regional.occ_ctrl_ratio: {_OWNER}")
