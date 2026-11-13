"""Outlier clusters of a z-map on the template mesh, and their centroids on the native
surface. See docs/metrics.md.

Interface of milestone 3. ``Cluster`` is final; the functions are written by package 3.H.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from jaxtyping import Bool, Float, Int
from scipy import sparse

from surfcensus.sources.base import Hemi
from surfcensus.templates import TemplateHemi

_OWNER = "package 3.H (maps and clusters)"


@dataclass(frozen=True)
class Cluster:
    vertices: Int[np.ndarray, " c"]
    """Template vertices, ascending."""
    area_mm2: float
    """Sum of ``TemplateHemi.area`` over the members."""
    mass: float
    """Sum of z over the members."""
    excess_mass: float
    """Sum of (z - threshold) over the members: it goes to 0 as a cluster shrinks to the
    threshold, where ``mass`` jumps."""
    peak_z: float
    peak_vertex: int
    """The member with the highest z, the lowest index on a tie."""
    centroid_vertex: int
    """The member nearest, on the unit sphere, to the normalized z-weighted mean of the
    members' sphere positions; the lowest index on a tie."""


def find_clusters(
    z: Float[np.ndarray, " v"],
    mask: Bool[np.ndarray, " v"],
    template_hemi: TemplateHemi,
    adjacency: sparse.csr_matrix,
    *,
    threshold: float,
    min_vertices: int = 1,
) -> list[Cluster]:
    """Connected components of the vertices in ``mask`` with finite ``z > threshold`` that
    have at least ``min_vertices`` members.

    Ordered by area descending, then ``excess_mass`` descending, then lowest first vertex.
    """
    raise NotImplementedError(f"metrics.clusters.find_clusters: {_OWNER}")


def native_centroids(
    rows: pd.DataFrame,
    vmap: Mapping[str, np.ndarray],
    adjacency: Mapping[Hemi, sparse.csr_matrix],
) -> pd.DataFrame:
    """Rows of ``clusters_native.parquet`` for the rows of ``clusters.parquet`` of one
    subject, from the vertex map of its render alone.

    The candidates of a centroid are the entries of the map whose template counterpart is
    the centroid, or, if there are none, a mesh neighbour of it. The native vertex is the
    candidate whose pial position is nearest the candidates' mean pial position (lowest
    index on a tie). A centroid without candidates gets no row. The column ``template_id``
    is left to the caller.
    """
    raise NotImplementedError(f"metrics.clusters.native_centroids: {_OWNER}")
