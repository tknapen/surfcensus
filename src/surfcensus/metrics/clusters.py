"""Outlier clusters of a z-map on the template mesh, and their centroids on the native
surface. See docs/metrics.md.

Noise makes clusters in every subject, so a cluster carries two sums: ``mass`` (the sum of
z, the wording of docs/metrics.md) jumps from 0 to at least the threshold when one vertex
crosses it, while ``excess_mass`` (the sum of z minus the threshold) starts at 0.

How often noise alone crosses z = 3 depends on the size of the reference. Simulated on
2026-11-13 with independent Gaussian noise at every vertex of a 10,242-vertex mesh, each
subject scored against a reference that includes it (``metrics.fsaverage.build_reference``
with two smoothing iterations): 0.0030 of the vertices at 15 subjects, 0.0025 at 20, 0.0021
at 30, 0.0017 at 60 and 0.0015 at 200, against 0.00135 for a normal distribution; without
the smoothing 0.010 at 15 and 0.0050 at 30. The largest cluster of such a map had one to
three vertices. Real maps are smooth, so their chance clusters are larger.

Sums over the members of a cluster are accumulated in vertex order in float64
(``numpy.bincount``), so they do not depend on how numpy vectorizes a reduction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from jaxtyping import Bool, Float, Int
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from surfcensus.schemas import KEYS
from surfcensus.sources.base import HEMIS, Hemi
from surfcensus.templates import TemplateHemi

NATIVE_COLUMNS: tuple[str, ...] = (
    *KEYS["clusters_native"],
    "native_vertex",
    "tkr_x",
    "tkr_y",
    "tkr_z",
)
"""Columns of the frame :func:`native_centroids` returns: those of ``clusters_native``
without ``template_id``."""

_NATIVE_DTYPES: dict[str, str] = {
    "rank": "int16",
    "native_vertex": "int32",
    "tkr_x": "float32",
    "tkr_y": "float32",
    "tkr_z": "float32",
}


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

    Positive side only. Two vertices are connected when the mesh has an edge between them
    and both are above the threshold inside ``mask``: a cluster is clipped at the border of
    the mask. The centroid's weights are the members' z, which are positive for a threshold
    of 0 or more; the rule is not meant for a threshold below 0.
    """
    n_vertices = len(z)
    if len(template_hemi.sphere) != n_vertices:
        raise ValueError(
            f"a z-map of {n_vertices} vertices on a template of {len(template_hemi.sphere)}"
        )
    if adjacency.shape != (n_vertices, n_vertices):
        raise ValueError(
            f"an adjacency of shape {adjacency.shape} for a mesh of {n_vertices} vertices"
        )
    values = np.asarray(z, dtype=np.float64)
    above = mask & np.isfinite(values) & (values > threshold)
    member = np.flatnonzero(above)
    if not member.size:
        return []
    graph = adjacency.tocsr()[member][:, member]
    n_found, label = connected_components(graph, directed=False)
    member_z = values[member]
    size = np.bincount(label, minlength=n_found)
    area = np.bincount(
        label, weights=template_hemi.area[member].astype(np.float64), minlength=n_found
    )
    mass = np.bincount(label, weights=member_z, minlength=n_found)
    excess = np.bincount(label, weights=member_z - threshold, minlength=n_found)
    sphere = template_hemi.sphere[member]
    # the z-weighted sum of the members' positions. Its direction is that of the normalized
    # mean, and the member nearest that direction is the one with the largest projection on
    # the sum, so nothing is divided by a length that could be 0
    pull = np.stack(
        [
            np.bincount(label, weights=member_z * sphere[:, axis], minlength=n_found)
            for axis in range(3)
        ],
        axis=1,
    )
    order = np.argsort(label, kind="stable")  # members of one component, ascending
    start = np.concatenate([[0], np.cumsum(size)])

    clusters = []
    for found in range(n_found):
        if size[found] < min_vertices:
            continue
        position = order[start[found] : start[found + 1]]
        vertices = member[position]
        cluster_z = member_z[position]
        along = (
            sphere[position, 0] * pull[found, 0]
            + sphere[position, 1] * pull[found, 1]
            + sphere[position, 2] * pull[found, 2]
        )
        peak = int(np.argmax(cluster_z))  # the first of equal values: the lowest index
        clusters.append(
            Cluster(
                vertices=vertices.astype(np.int32),
                area_mm2=float(area[found]),
                mass=float(mass[found]),
                excess_mass=float(excess[found]),
                peak_z=float(cluster_z[peak]),
                peak_vertex=int(vertices[peak]),
                centroid_vertex=int(vertices[int(np.argmax(along))]),
            )
        )
    clusters.sort(key=lambda c: (-c.area_mm2, -c.excess_mass, int(c.vertices[0])))
    return clusters


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

    This is a nearest-vertex mapping: the position is that of a native pial vertex, not an
    interpolated point. It needs no source file. The result has the columns
    :data:`NATIVE_COLUMNS` and the order of ``rows``. ValueError for a row whose hemisphere
    or centroid the adjacency does not have.

    ``vmap`` must come from a render made with the template the clusters are on: the
    counterparts in it are vertices of that template's mesh. The caller checks this (the
    render records its template id); nothing in a vertex map says which template it is of.
    A map without counterparts (every entry -1, a render made without a template) gives no
    rows.
    """
    needed = [*KEYS["clusters"], "centroid_vertex"]
    missing = [column for column in needed if column not in rows.columns]
    if missing:
        raise ValueError(f"the cluster rows lack the columns {', '.join(missing)}")
    meshes: dict[str, sparse.csr_matrix] = {
        str(hemi): matrix.tocsr() for hemi, matrix in adjacency.items()
    }
    lookup: dict[str, _Counterparts] = {
        hemi: _Counterparts(vmap, hemi) for hemi in HEMIS if f"{hemi}_fsavg6" in vmap
    }
    kept: list[int] = []
    native_vertex: list[int] = []
    position: list[np.ndarray] = []
    centroids = zip(rows["hemi"].tolist(), rows["centroid_vertex"].tolist(), strict=True)
    for row, (name, vertex) in enumerate(centroids):
        hemi, centroid = str(name), int(vertex)
        if hemi not in meshes or hemi not in lookup:
            raise ValueError(f"a cluster on {hemi!r}, for which there is no mesh or no vertex map")
        mesh, counterparts = meshes[hemi], lookup[hemi]
        if not 0 <= centroid < mesh.shape[0]:
            raise ValueError(
                f"centroid vertex {centroid} is not on the mesh of {mesh.shape[0]} vertices"
            )
        candidate = counterparts.of(np.array([centroid]))
        if not candidate.size:
            candidate = counterparts.of(
                mesh.indices[mesh.indptr[centroid] : mesh.indptr[centroid + 1]]
            )
        if not candidate.size:
            continue
        pial = counterparts.pial[candidate].astype(np.float64)
        delta = pial - pial.sum(axis=0) / len(pial)
        distance = delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1] + delta[:, 2] * delta[:, 2]
        nearest = candidate[int(np.argmin(distance))]  # candidates ascend: lowest index on a tie
        kept.append(row)
        native_vertex.append(int(counterparts.index[nearest]))
        position.append(counterparts.pial[nearest])

    out = pd.DataFrame(
        {
            column: rows[column]
            .iloc[kept]
            .astype(_NATIVE_DTYPES.get(column, "object"))
            .reset_index(drop=True)
            for column in KEYS["clusters_native"]
        }
    )
    out["native_vertex"] = np.asarray(native_vertex, dtype=np.int32)
    tkr = np.asarray(position, dtype=np.float32).reshape(-1, 3)
    for axis, column in enumerate(("tkr_x", "tkr_y", "tkr_z")):
        out[column] = tkr[:, axis]
    return out


class _Counterparts:
    """The entries of one hemisphere of a vertex map, to be found by template vertex."""

    def __init__(self, vmap: Mapping[str, np.ndarray], hemi: Hemi) -> None:
        self.index = np.asarray(vmap[f"{hemi}_index"])
        self.pial = np.asarray(vmap[f"{hemi}_pial"])
        template = np.asarray(vmap[f"{hemi}_fsavg6"])
        if not len(self.index) == len(self.pial) == len(template):
            raise ValueError(f"the {hemi} arrays of the vertex map differ in length")
        self._order = np.argsort(template, kind="stable")
        self._sorted = template[self._order]

    def of(self, template_vertices: np.ndarray) -> np.ndarray:
        """Positions in the map, ascending, of the entries whose counterpart is one of
        ``template_vertices``."""
        low = np.searchsorted(self._sorted, template_vertices, side="left")
        high = np.searchsorted(self._sorted, template_vertices, side="right")
        found = [self._order[a:b] for a, b in zip(low.tolist(), high.tolist(), strict=True)]
        return np.sort(np.concatenate(found)) if found else np.empty(0, dtype=np.int64)
