"""The mapping between a subject's registered sphere (``surf/?h.sphere.reg``) and a template
sphere: barycentric resampling and nearest-vertex correspondences. See docs/metrics.md.

One :class:`SphereMap` per hemisphere per subject serves three uses:

* thickness to the template mesh: ``resample(thickness, smap, valid=...)``;
* a template vertex to the native surface: ``smap.nearest_source[vertex]``;
* a native vertex to the template mesh: ``smap.nearest_target[index]``.

Interface of milestone 3; the functions are written by package 3.C.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float, Int

from surfcensus.fsio.surfaces import Surface

_OWNER = "package 3.C (registration)"


@dataclass(frozen=True)
class SphereMap:
    n_source: int
    n_target: int
    corners: Int[np.ndarray, "t 3"]
    """Source vertices that are combined for each target vertex."""
    weights: Float[np.ndarray, "t 3"]
    """Barycentric weights: non-negative, each row sums to 1; a fallback row is (1, 0, 0)."""
    fallback: Bool[np.ndarray, " t"]
    """True where no containing face was found and the nearest source vertex is used."""
    nearest_source: Int[np.ndarray, " t"]
    """The source vertex nearest each target vertex."""
    nearest_target: Int[np.ndarray, " s"]
    """The target vertex nearest each source vertex."""
    flipped: bool
    """The source faces were wound inward and were reversed before the search."""


def unit_sphere(vertices: Float[np.ndarray, "n 3"]) -> Float[np.ndarray, "n 3"]:
    """``vertices`` scaled to unit length.

    Raises ``ValueError("degenerate sphere: ...")`` for a radius that is zero or not finite,
    and when radii lie more than 1% from their median.
    """
    raise NotImplementedError(f"fsio.registration.unit_sphere: {_OWNER}")


def build_sphere_map(source: Surface, target: Float[np.ndarray, "t 3"], *, k: int = 8) -> SphereMap:
    """Map every vertex of the ``target`` sphere into the faces of the ``source`` sphere.

    Both are put on the unit sphere. For each target point the faces around its ``k``
    nearest source vertices are searched, nearest vertex first and faces in ascending
    index, for the first one that contains it; a point that none contains falls back to its
    nearest source vertex. A face wound inward never matches; when more than half of the
    source faces are, every face is reversed once and ``flipped`` is set. No input raises
    but a degenerate sphere (:func:`unit_sphere`). The result does not depend on thread
    count or platform.
    """
    raise NotImplementedError(f"fsio.registration.build_sphere_map: {_OWNER}")


def resample(
    values: Float[np.ndarray, " s"],
    smap: SphereMap,
    *,
    valid: Bool[np.ndarray, " s"] | None = None,
    min_valid_weight: float = 0.5,
) -> Float[np.ndarray, " t"]:
    """``values`` on the source vertices, interpolated to the target vertices.

    With ``valid``, the weights of corners that are not valid are set to 0. A target whose
    remaining weight is below ``min_valid_weight`` is NaN; otherwise the remaining weights
    are scaled to sum to 1.
    """
    raise NotImplementedError(f"fsio.registration.resample: {_OWNER}")
