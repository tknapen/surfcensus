"""Thickness on the template mesh: resampling, the per-vertex reference, z-maps, strata.
See docs/metrics.md.

Everything downstream of the stored map (global median, reference, z, clusters) is computed
from the float16 file under ``tables/fsavg/``, never from the values in memory, so results
can be reproduced from the run directory alone. :func:`resample_hemi` therefore returns the
map as the file will hold it: every value is one that float16 represents exactly, and
:func:`read_maps` after :func:`write_maps` gives the same numbers back.

Medians are taken in float64 from sorted columns, and the smoothing is a sparse product
written out by scipy without BLAS, so a reference does not depend on the linear-algebra
library or the thread count.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jaxtyping import Float
from scipy import sparse

from surfcensus._io import read_npz, write_npz
from surfcensus.config import MetricsConfig
from surfcensus.fsio.annot import Annotation
from surfcensus.fsio.registration import SphereMap, build_sphere_map, resample
from surfcensus.fsio.surfaces import Surface
from surfcensus.metrics import MetricError
from surfcensus.reference.load import ReferenceMaps
from surfcensus.reference.strata import (
    ALL_STRATUM,
    dataset_stratum,
    field_version_stratum,
    version_stratum,
)
from surfcensus.sources.base import HEMIS, Hemi
from surfcensus.templates import TemplateHemi

MAD_TO_SD = 1.4826
"""A MAD times this estimates the standard deviation of a normal distribution."""
GLOBAL_MEDIAN_MIN_SHARE = 40
"""The global median needs a value at one vertex in this many of the mesh, counted over
cortex outside the search region: 1,024 of the 40,962 vertices of fsaverage6."""
MAP_DTYPE = "<f2"
"""Maps are stored as float16: thickness below 8 mm changes by at most 0.002 mm."""

_BLOCK_VERTICES = 4096
"""Vertices whose columns of a stack are sorted at a time, which bounds the working copies
(two of 160 MB for 5,000 subjects)."""
_DEGENERATE = "degenerate sphere"
_STRATUM_PREFIXES = ("dataset:", "field:", "fs:")


@dataclass(frozen=True)
class HemiMap:
    thickness: Float[np.ndarray, " v"]
    """float32, mm, on the template mesh, with the values of the float16 file; NaN where a
    value is missing and outside the template cortex."""
    n_fallback: int
    """Target vertices that took the value of the nearest source vertex."""
    n_missing_cortex: int
    """Template cortex vertices without a value."""
    global_median: float
    """:func:`global_median` of ``thickness``."""
    flipped: bool
    """The winding of the subject's sphere was reversed (``SphereMap.flipped``)."""


def resample_hemi(
    thickness: Float[np.ndarray, " n"],
    sphere_reg: Surface,
    template_hemi: TemplateHemi,
    *,
    annot: Annotation | None,
    config: MetricsConfig,
) -> tuple[HemiMap, SphereMap]:
    """One hemisphere's thickness on the template mesh, and the sphere map it went through.

    A source vertex counts when its thickness is finite and above 0 and, with an
    annotation, when that calls it cortex. Raises ``MetricError``: ``vertex count
    mismatch`` unless thickness, sphere and annotation agree in length; ``degenerate
    sphere`` for an unusable sphere and when more than ``config.max_fallback_frac`` of the
    targets fell back to the nearest vertex; ``low coverage`` when fewer than
    ``config.min_coverage`` of the template cortex vertices get a value.

    The values are rounded to float16, as :func:`write_maps` stores them, before the global
    median is taken. A value that float16 cannot hold (above 65,504) counts as missing.
    """
    counts = {"thickness": len(thickness), "sphere.reg": len(sphere_reg.vertices)}
    if annot is not None:
        counts["annotation"] = len(annot.labels)
    if len(set(counts.values())) > 1:
        raise MetricError(
            "vertex count mismatch",
            ", ".join(f"{name} {count}" for name, count in counts.items()) + " vertices",
        )
    try:
        smap = build_sphere_map(sphere_reg, template_hemi.sphere, k=config.knn)
    except ValueError as exc:
        text = str(exc)
        if not text.startswith(_DEGENERATE):
            raise
        raise MetricError(_DEGENERATE, text.removeprefix(_DEGENERATE).lstrip(": ")) from exc
    n_fallback = int(smap.fallback.sum())
    if n_fallback > config.max_fallback_frac * smap.n_target:
        raise MetricError(
            _DEGENERATE,
            f"{n_fallback} of {smap.n_target} template vertices lie in no face of the sphere "
            f"(at most {config.max_fallback_frac:.2%} may)",
        )

    values = np.asarray(thickness)
    valid = np.isfinite(values) & (values > 0)
    if annot is not None:
        valid &= annot.cortex()
    stored = _as_stored(
        resample(values, smap, valid=valid, min_valid_weight=config.min_valid_weight)
    ).astype(np.float32)
    cortex = template_hemi.cortex
    stored[~cortex] = np.nan
    n_cortex = int(cortex.sum())
    n_value = int(np.isfinite(stored).sum())
    if n_cortex == 0 or n_value < config.min_coverage * n_cortex:
        share = f" ({n_value / n_cortex:.1%}; " if n_cortex else " ("
        raise MetricError(
            "low coverage",
            f"{n_value} of {n_cortex} template cortex vertices have a value{share}"
            f"{config.min_coverage:.0%} are needed)",
        )
    hemi_map = HemiMap(
        thickness=stored,
        n_fallback=n_fallback,
        n_missing_cortex=n_cortex - n_value,
        global_median=global_median(stored, template_hemi),
        flipped=smap.flipped,
    )
    return hemi_map, smap


def global_median(thickness_fs: Float[np.ndarray, " v"], template_hemi: TemplateHemi) -> float:
    """Median over the finite values in cortex and outside the search region. NaN when
    fewer than 2.5% of the mesh's vertices qualify.

    The count needed is the number of mesh vertices divided by 40 and rounded down: 1,024
    on fsaverage6.
    """
    values = np.asarray(thickness_fs)
    _same_mesh(len(values), template_hemi, "a thickness map")
    use = template_hemi.cortex & ~template_hemi.regions["search"] & np.isfinite(values)
    if int(use.sum()) < max(1, len(values) // GLOBAL_MEDIAN_MIN_SHARE):
        return float("nan")
    return float(np.median(values[use].astype(np.float64)))


def write_maps(path: Path, maps: Mapping[Hemi, Float[np.ndarray, " v"]]) -> None:
    """Write ``tables/fsavg/<subject_uid>.npz``: ``lh_thickness`` and ``rh_thickness`` as
    ``<f2`` with NaN for missing, byte-identical for identical input.

    A hemisphere that ``maps`` lacks is written as all NaN, at the length of the other: a
    subject with one usable hemisphere has a file like every other subject. ValueError for
    no hemisphere at all, for a key that is no hemisphere, and for two maps of different
    lengths. A value that float16 cannot hold is written as NaN, and every NaN is written
    with the same bits.
    """
    unknown = sorted(set(maps) - set(HEMIS))
    if unknown:
        raise ValueError(f"maps are keyed by lh and rh, got {', '.join(map(str, unknown))}")
    given = {hemi: np.asarray(maps[hemi]) for hemi in HEMIS if hemi in maps}
    lengths = {len(array) for array in given.values()}
    if not given:
        raise ValueError("no map to write: neither hemisphere was given")
    if len(lengths) > 1 or any(array.ndim != 1 for array in given.values()):
        raise ValueError(
            "the two maps of a subject are on one mesh, got "
            + " and ".join(f"{hemi} {given[hemi].shape}" for hemi in given)
        )
    (length,) = lengths
    arrays = {
        f"{hemi}_thickness": _as_stored(given[hemi])
        if hemi in given
        else np.full(length, np.nan, dtype=MAP_DTYPE)
        for hemi in HEMIS
    }
    write_npz(path, arrays)


def read_maps(path: Path, n_vertices: int | None = None) -> dict[Hemi, Float[np.ndarray, " v"]]:
    """The two maps of a file written by :func:`write_maps`, as float32.

    ``FileNotFoundError`` for a file that is not there; ``ValueError``, naming the file and
    not its path, for one that is not an archive of the two float16 maps. With
    ``n_vertices``, the vertex count of the mesh the maps should be on, also for maps of
    another length: a file that was copied from a run on another mesh, or written over.
    """
    path = Path(path)
    try:
        arrays = read_npz(path)
    except FileNotFoundError:
        raise
    except (ValueError, OSError) as exc:
        raise ValueError(f"{path.name} cannot be read as an archive of arrays") from exc
    keys = [f"{hemi}_thickness" for hemi in HEMIS]
    if list(arrays) != keys:
        raise ValueError(f"{path.name} holds {sorted(arrays)}, not the two thickness maps")
    maps: dict[Hemi, np.ndarray] = {}
    for hemi, key in zip(HEMIS, keys, strict=True):
        array = arrays[key]
        if array.ndim != 1 or array.dtype != np.dtype(MAP_DTYPE):
            raise ValueError(f"{path.name}: {key} is not one axis of float16")
        maps[hemi] = array.astype(np.float32)
    if len(maps["lh"]) != len(maps["rh"]):
        raise ValueError(
            f"{path.name}: the maps have {len(maps['lh'])} and {len(maps['rh'])} vertices"
        )
    if n_vertices is not None and len(maps["lh"]) != n_vertices:
        raise ValueError(
            f"{path.name}: maps of {len(maps['lh'])} vertices, on a mesh of {n_vertices}"
        )
    return maps


def build_reference(
    stack: Float[np.ndarray, "s v"],
    template_hemi: TemplateHemi,
    adjacency: sparse.csr_matrix,
    *,
    min_vertex_n: int,
    smooth_iterations: int,
    mad_floor_mm: float,
) -> ReferenceMaps:
    """Per-vertex median and MAD of ``stack``: one row per subject, thickness minus that
    subject's global median.

    A vertex with fewer than ``min_vertex_n`` finite values has no reference. The MAD map
    is replaced ``smooth_iterations`` times by the mean over each vertex and its
    neighbours (NaN ignored, cortex only) and then floored at ``mad_floor_mm``.

    Values outside the template cortex are not read: ``n`` is 0 there. Smoothing fills no
    vertex: one without a MAD stays without, and is left out of its neighbours' means. The
    median of an even number of values is the mean of the two in the middle, as in numpy.

    ``stack`` covers the whole mesh, because the smoothing needs every vertex. It is read
    a block of columns at a time and never copied whole, so it may be a memory-mapped
    array.
    """
    n_subjects, n_vertices = stack.shape
    _same_mesh(n_vertices, template_hemi, "a stack of maps")
    if adjacency.shape != (n_vertices, n_vertices):
        raise ValueError(
            f"an adjacency of shape {adjacency.shape} for a mesh of {n_vertices} vertices"
        )
    median = np.full(n_vertices, np.nan)
    mad = np.full(n_vertices, np.nan)
    n = np.zeros(n_vertices, dtype=np.int32)
    columns = np.flatnonzero(template_hemi.cortex)
    for start in range(0, len(columns) if n_subjects else 0, _BLOCK_VERTICES):
        column = columns[start : start + _BLOCK_VERTICES]
        block = np.array(stack[:, column], dtype=np.float64)
        block[~np.isfinite(block)] = np.nan
        count = np.isfinite(block).sum(axis=0)
        centre = _column_median(block, count)
        np.subtract(block, centre, out=block)
        np.abs(block, out=block)
        spread = _column_median(block, count)
        enough = count >= min_vertex_n
        median[column] = np.where(enough, centre, np.nan)
        mad[column] = np.where(enough, spread, np.nan)
        n[column] = count

    if smooth_iterations:
        hood = (adjacency + sparse.identity(n_vertices, format="csr")).tocsr()
        hood.data[:] = 1.0
        for _ in range(smooth_iterations):
            has = np.isfinite(mad)
            total = hood @ np.where(has, mad, 0.0)
            number = hood @ has.astype(np.float64)
            mad = np.divide(total, number, out=np.full(n_vertices, np.nan), where=has)
    mad = np.maximum(mad, mad_floor_mm)  # NaN stays NaN
    return ReferenceMaps(median=median.astype(np.float32), mad=mad.astype(np.float32), n=n)


def zmap(
    thickness_fs: Float[np.ndarray, " v"], global_median: float, ref: ReferenceMaps
) -> Float[np.ndarray, " v"]:
    """``(thickness_fs - global_median - median) / (1.4826 * mad)``; NaN wherever a term is.

    float32, computed in float64. A vertex whose MAD is 0 (possible only with a floor of 0)
    has no scale and is NaN, not infinite.
    """
    values = np.asarray(thickness_fs, dtype=np.float64)
    if len(values) != len(ref.median):
        raise ValueError(f"a map of {len(values)} vertices and a reference of {len(ref.median)}")
    scale = MAD_TO_SD * ref.mad.astype(np.float64)
    residual = values - float(global_median) - ref.median.astype(np.float64)
    z = np.divide(residual, scale, out=np.full(len(values), np.nan), where=scale > 0)
    return z.astype(np.float32)


def assign_strata(subjects: pd.DataFrame, *, min_n: int) -> pd.Series:
    """``subject_uid`` -> the stratum whose reference the subject is scored against, or None.

    ``subjects`` has one row per subject that has a map, with the columns ``subject_uid``,
    ``dataset`` (the pseudonym), ``field_strength`` and ``fs_major``. A stratum exists
    when it has at least ``min_n`` members (:func:`stratum_members`). A subject gets the
    first existing one of its dataset, its field strength and FreeSurfer version, and
    ``all`` (names from ``reference.strata``); a subject with unknown field strength or
    version skips the middle one.

    The result is indexed by ``subject_uid`` in the order of ``subjects``.
    """
    levels = _levels(subjects)
    size: dict[str, int] = {}
    for names in levels:
        for name in names:
            if name is not None:
                size[name] = size.get(name, 0) + 1
    assigned = [
        next((name for name in names if name is not None and size[name] >= min_n), None)
        for names in levels
    ]
    index = pd.Index(subjects["subject_uid"].astype(object), name="subject_uid")
    return pd.Series(assigned, index=index, dtype=object, name="stratum")


def stratum_members(subjects: pd.DataFrame, stratum: str) -> list[str]:
    """The sorted ``subject_uid``s of every subject that matches the definition of
    ``stratum``, whether or not it is assigned to it.

    Besides the three kinds :func:`assign_strata` gives out, ``fs:<major>`` is understood
    (a stratum of a shipped reference). ValueError for a name of no known form.
    """
    if stratum != ALL_STRATUM and not stratum.startswith(_STRATUM_PREFIXES):
        raise ValueError(f"not the name of a stratum: {stratum!r}")
    uids = subjects["subject_uid"].tolist()
    members = []
    for uid, names, fs_major in zip(
        uids, _levels(subjects), subjects["fs_major"].tolist(), strict=True
    ):
        if stratum in names or stratum == version_stratum(_whole(fs_major)):
            members.append(str(uid))
    return sorted(members)


def _levels(subjects: pd.DataFrame) -> list[tuple[str | None, str | None, str]]:
    """Per row, the strata the subject belongs to, in the order they are tried: dataset,
    field strength and version, all."""
    missing = [
        column
        for column in ("subject_uid", "dataset", "field_strength", "fs_major")
        if column not in subjects.columns
    ]
    if missing:
        raise ValueError(f"the subjects lack the columns {', '.join(missing)}")
    if subjects["subject_uid"].duplicated().any():
        raise ValueError("a subject_uid is listed twice")
    return [
        (
            None if _absent(dataset) else dataset_stratum(str(dataset)),
            field_version_stratum(_number(field_strength), _whole(fs_major)),
            ALL_STRATUM,
        )
        for dataset, field_strength, fs_major in zip(
            subjects["dataset"].tolist(),
            subjects["field_strength"].tolist(),
            subjects["fs_major"].tolist(),
            strict=True,
        )
    ]


def _absent(value: object) -> bool:
    """None, NaN and ``pandas.NA``: the forms a missing value takes in a frame."""
    if value is None or value is pd.NA:
        return True
    return isinstance(value, float | np.floating) and math.isnan(value)


def _number(value: Any) -> float | None:
    return None if _absent(value) else float(value)


def _whole(value: Any) -> int | None:
    return None if _absent(value) else int(value)


def _same_mesh(n_vertices: int, template_hemi: TemplateHemi, what: str) -> None:
    if n_vertices != len(template_hemi.cortex):
        raise ValueError(
            f"{what} of {n_vertices} vertices on a template of {len(template_hemi.cortex)}"
        )


def _as_stored(values: np.ndarray) -> np.ndarray:
    """``values`` as the map file holds them: float16, with one NaN for everything that is
    missing or does not fit."""
    with np.errstate(over="ignore"):
        stored = np.asarray(values).astype(MAP_DTYPE)
    # a NaN made by arithmetic can carry a sign bit, which would show in the file's bytes
    stored[~np.isfinite(stored)] = np.nan
    return stored


def _column_median(block: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Median of the non-NaN values of each column of ``block``, which has ``count`` of
    them; NaN for a column without any."""
    ordered = np.sort(block, axis=0)  # NaN sorts last
    last = np.maximum(count - 1, 0)
    low = np.take_along_axis(ordered, (last // 2)[None, :], axis=0)[0]
    high = np.take_along_axis(ordered, ((last + 1) // 2)[None, :], axis=0)[0]
    return np.where(count > 0, (low + high) / 2.0, np.nan)
