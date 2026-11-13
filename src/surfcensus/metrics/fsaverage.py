"""Thickness on the template mesh: resampling, the per-vertex reference, z-maps, strata.
See docs/metrics.md.

Everything downstream of the stored map (global median, reference, z, clusters) is computed
from the float16 file under ``tables/fsavg/``, never from the values in memory, so results
can be reproduced from the run directory alone.

Interface of milestone 3. ``HemiMap`` is final; the functions are written by package 3.H.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Float
from scipy import sparse

from surfcensus.config import MetricsConfig
from surfcensus.fsio.annot import Annotation
from surfcensus.fsio.registration import SphereMap
from surfcensus.fsio.surfaces import Surface
from surfcensus.reference.load import ReferenceMaps
from surfcensus.sources.base import Hemi
from surfcensus.templates import TemplateHemi

_OWNER = "package 3.H (maps and clusters)"


@dataclass(frozen=True)
class HemiMap:
    thickness: Float[np.ndarray, " v"]
    """float32, mm, on the template mesh; NaN where a value is missing and outside the
    template cortex."""
    n_fallback: int
    """Target vertices that took the value of the nearest source vertex."""
    n_missing_cortex: int
    """Template cortex vertices without a value."""
    global_median: float
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
    """
    raise NotImplementedError(f"metrics.fsaverage.resample_hemi: {_OWNER}")


def global_median(thickness_fs: Float[np.ndarray, " v"], template_hemi: TemplateHemi) -> float:
    """Median over the finite values in cortex and outside the search region. NaN when
    fewer than 2.5% of the mesh's vertices qualify."""
    raise NotImplementedError(f"metrics.fsaverage.global_median: {_OWNER}")


def write_maps(path: Path, maps: Mapping[Hemi, Float[np.ndarray, " v"]]) -> None:
    """Write ``tables/fsavg/<subject_uid>.npz``: ``lh_thickness`` and ``rh_thickness`` as
    ``<f2`` with NaN for missing, byte-identical for identical input."""
    raise NotImplementedError(f"metrics.fsaverage.write_maps: {_OWNER}")


def read_maps(path: Path) -> dict[Hemi, Float[np.ndarray, " v"]]:
    """The two maps of a file written by :func:`write_maps`, as float32."""
    raise NotImplementedError(f"metrics.fsaverage.read_maps: {_OWNER}")


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
    """
    raise NotImplementedError(f"metrics.fsaverage.build_reference: {_OWNER}")


def zmap(
    thickness_fs: Float[np.ndarray, " v"], global_median: float, ref: ReferenceMaps
) -> Float[np.ndarray, " v"]:
    """``(thickness_fs - global_median - median) / (1.4826 * mad)``; NaN wherever a term is."""
    raise NotImplementedError(f"metrics.fsaverage.zmap: {_OWNER}")


def assign_strata(subjects: pd.DataFrame, *, min_n: int) -> pd.Series:
    """``subject_uid`` -> the stratum whose reference the subject is scored against, or None.

    ``subjects`` has one row per subject that has a map, with the columns ``subject_uid``,
    ``dataset`` (the pseudonym), ``field_strength`` and ``fs_major``. A stratum exists
    when it has at least ``min_n`` members (:func:`stratum_members`). A subject gets the
    first existing one of its dataset, its field strength and FreeSurfer version, and
    ``all`` (names from ``reference.strata``); a subject with unknown field strength or
    version skips the middle one.
    """
    raise NotImplementedError(f"metrics.fsaverage.assign_strata: {_OWNER}")


def stratum_members(subjects: pd.DataFrame, stratum: str) -> list[str]:
    """The sorted ``subject_uid``s of every subject that matches the definition of
    ``stratum``, whether or not it is assigned to it."""
    raise NotImplementedError(f"metrics.fsaverage.stratum_members: {_OWNER}")
