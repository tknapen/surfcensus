"""Regional thickness from ``stats/?h.aparc.stats`` (Tier A): cheap, and available when
surfaces are missing. See docs/metrics.md.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from surfcensus.config import RegionConfig
from surfcensus.fsio.stats import StatsFile, aparc_frame
from surfcensus.log import get_logger
from surfcensus.sources.base import Hemi

__all__ = ["occ_ctrl_ratio", "region_mean_thickness", "regional_rows"]

_INT_COLUMNS = {"num_vert": "NumVert", "surf_area_mm2": "SurfArea", "gray_vol_mm3": "GrayVol"}
_FLOAT_COLUMNS = {"thick_avg_mm": "ThickAvg", "thick_std_mm": "ThickStd"}
_INT32_MAX = 2**31 - 1


def regional_rows(subject_uid: str, hemi: Hemi, stats: StatsFile) -> pd.DataFrame:
    """The rows of ``regional.parquet`` (``schemas.REGIONAL``) for one hemisphere of one
    subject: every parcel of the stats file, in parcel order.

    The table has no missing values, so a parcel whose mean or standard deviation of
    thickness is not a finite float32 gets no row, and a warning says how many were left
    out. FreeSurfer writes no row for a parcel without vertices, so the frame can also be
    short because the file is.

    Raises ValueError for a stats file that :func:`surfcensus.fsio.stats.aparc_frame`
    refuses and for a count that does not fit int32.
    """
    frame = aparc_frame(stats)
    with np.errstate(over="ignore"):
        floats = {
            column: frame[source].to_numpy().astype(np.float32)
            for column, source in _FLOAT_COLUMNS.items()
        }
    keep = np.logical_and.reduce([np.isfinite(values) for values in floats.values()])
    if not keep.all():
        get_logger().warning(
            "%s %s: %d of %d parcels have no finite thickness in the stats file and are left out",
            subject_uid,
            hemi,
            int((~keep).sum()),
            len(frame),
        )
    integers: dict[str, np.ndarray] = {}
    for column, source in _INT_COLUMNS.items():
        values = frame[source].to_numpy()[keep]
        if values.size and np.abs(values).max() > _INT32_MAX:
            raise ValueError(f"column {source}: a value does not fit 32 bits")
        integers[column] = values.astype(np.int32)
    n = int(keep.sum())
    rows = pd.DataFrame(
        {
            "subject_uid": pd.Series([subject_uid] * n, dtype=object),
            "hemi": pd.Series([hemi] * n, dtype=object),
            "parcel": pd.Series(frame["StructName"].to_numpy()[keep], dtype=object),
            **integers,
            **{column: values[keep] for column, values in floats.items()},
        }
    )
    return rows.sort_values("parcel", kind="stable", ignore_index=True)


def _column(rows: pd.DataFrame, name: str) -> np.ndarray:
    # na_value: after a join the column may be a nullable one
    return rows[name].to_numpy(dtype=np.float64, na_value=np.nan)


def region_mean_thickness(rows: pd.DataFrame, parcels: Sequence[str]) -> float:
    """Mean of ``thick_avg_mm`` over ``parcels``, weighted by ``surf_area_mm2``, with both
    hemispheres pooled. Parcels that the rows lack are left out; NaN when none is there.

    ``rows`` are rows of ``regional.parquet`` of one subject, of both hemispheres or of the
    one that has a stats file. A row without a positive area or without a thickness has no
    weight. A name given twice in ``parcels`` counts once.

    Raises ValueError when the rows lack one of the three columns, when they are of more
    than one subject, and when a parcel of a hemisphere has two rows: pooled, either would
    give a number that belongs to no subject.
    """
    missing = [c for c in ("parcel", "surf_area_mm2", "thick_avg_mm") if c not in rows.columns]
    if missing:
        raise ValueError(f"not rows of the regional table: no column {', '.join(missing)}")
    if "subject_uid" in rows.columns and rows["subject_uid"].nunique(dropna=False) > 1:
        raise ValueError("rows of more than one subject")
    if "hemi" in rows.columns and rows.duplicated(["hemi", "parcel"]).any():
        raise ValueError("more than one row for a parcel of one hemisphere")

    chosen = rows[rows["parcel"].isin(set(parcels))]
    area, thickness = _column(chosen, "surf_area_mm2"), _column(chosen, "thick_avg_mm")
    use = np.isfinite(thickness) & (area > 0)
    if not use.any():
        return float("nan")
    return float(np.sum(area[use] * thickness[use]) / np.sum(area[use]))


def occ_ctrl_ratio(rows: pd.DataFrame, regions: RegionConfig) -> tuple[float, float, float]:
    """``(occipital mm, control mm, occipital / control)`` of one subject, from
    ``regions.occipital`` and ``regions.control``.

    Each is :func:`region_mean_thickness` over the parcels of that list. The ratio is NaN
    when either mean is NaN or the control mean is not positive.
    """
    occipital = region_mean_thickness(rows, regions.occipital)
    control = region_mean_thickness(rows, regions.control)
    if np.isfinite(occipital) and control > 0:
        return occipital, control, occipital / control
    return occipital, control, float("nan")
