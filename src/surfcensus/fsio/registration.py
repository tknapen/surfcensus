"""The mapping between a subject's registered sphere (``surf/?h.sphere.reg``) and a template
sphere: barycentric resampling and nearest-vertex correspondences. See docs/metrics.md.

One :class:`SphereMap` per hemisphere per subject serves three uses:

* thickness to the template mesh: ``resample(thickness, smap, valid=...)``;
* a template vertex to the native surface: ``smap.nearest_source[vertex]``;
* a native vertex to the template mesh: ``smap.nearest_target[index]``.

A hemisphere is mapped on its own; nothing here knows which one it is. The right sphere is
not a mirror image of the left one: in ``ds000102`` sub-01 (FreeSurfer 6.0.1) and in the
``fsaverage`` next to it, no face of either hemisphere's ``sphere.reg`` is wound inward, and
the two ``sphere.reg`` files of that ``fsaverage`` are the same bytes (checked 2026-11-13).

Sums and products are written out with numpy's elementwise operations, without BLAS or
``einsum``, so that the weights do not depend on the linear-algebra library, the thread
count or the platform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float, Int
from scipy.spatial import cKDTree

from surfcensus.fsio.surfaces import Surface

RADIUS_TOLERANCE = 0.01
"""Radii may lie this fraction from their median."""
CONTAIN_TOLERANCE = 1e-9
"""A point counts as inside a face while no barycentric coordinate is below minus this
fraction of their sum, so that a point on an edge is in one of the two faces."""
COINCIDENT = 1e-12
"""Chord distance on the unit sphere, about ten thousand rounding steps, below which a
target point is a source vertex and takes that vertex alone. The spacing of a FreeSurfer
sphere is about 5e-3."""


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


def _dot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x[..., 0] * y[..., 0] + x[..., 1] * y[..., 1] + x[..., 2] * y[..., 2]


def _cross(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.empty(np.broadcast_shapes(x.shape, y.shape), dtype=np.float64)
    out[..., 0] = x[..., 1] * y[..., 2] - x[..., 2] * y[..., 1]
    out[..., 1] = x[..., 2] * y[..., 0] - x[..., 0] * y[..., 2]
    out[..., 2] = x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]
    return out


def unit_sphere(vertices: Float[np.ndarray, "n 3"]) -> Float[np.ndarray, "n 3"]:
    """``vertices`` scaled to unit length.

    Raises ``ValueError("degenerate sphere: ...")`` for a radius that is zero or not finite,
    and when radii lie more than 1% from their median.
    """
    v = np.asarray(vertices, dtype=np.float64)
    if len(v) == 0:
        raise ValueError("degenerate sphere: no vertices")
    radius = np.sqrt(_dot(v, v))
    usable = np.isfinite(radius) & (radius > 0)
    if not usable.all():
        raise ValueError(
            f"degenerate sphere: {int((~usable).sum())} of {len(v)} vertices have a radius "
            "that is zero or not finite"
        )
    median = float(np.median(radius))
    off = np.abs(radius - median) > RADIUS_TOLERANCE * median
    if off.any():
        raise ValueError(
            f"degenerate sphere: {int(off.sum())} of {len(v)} vertices lie more than "
            f"{RADIUS_TOLERANCE:.0%} from the median radius {median:.6g} "
            f"(radii {float(radius.min()):.6g} to {float(radius.max()):.6g})"
        )
    return v / radius[:, None]


def _neighbours(
    tree: cKDTree, tree_points: np.ndarray, points: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """The ``k`` points of the tree nearest each of ``points``: ``(index, d2)``, both
    ``(n, k)``, each row ordered by squared distance and then by index.

    The distances are computed again here: the order then does not rest on the arithmetic
    of the tree, and equal distances come in index order.
    """
    _, index = tree.query(points, k=k)
    index = np.asarray(index, dtype=np.int64).reshape(len(points), k)
    delta = tree_points[index] - points[:, None, :]
    d2 = _dot(delta, delta)
    order = np.argsort(index, axis=1, kind="stable")
    index = np.take_along_axis(index, order, axis=1)
    d2 = np.take_along_axis(d2, order, axis=1)
    order = np.argsort(d2, axis=1, kind="stable")
    return np.take_along_axis(index, order, axis=1), np.take_along_axis(d2, order, axis=1)


def _nearest(
    tree: cKDTree, tree_points: np.ndarray, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The point of the tree nearest each of ``points``, the lowest index among equally
    near ones: ``(index, d2)``."""
    n = len(tree_points)
    k = min(2, n)
    index, d2 = _neighbours(tree, tree_points, points, k)
    nearest, nearest_d2 = index[:, 0].copy(), d2[:, 0].copy()
    # while the farthest of the candidates is as near as the nearest, there may be more
    rows = np.flatnonzero(d2[:, -1] == d2[:, 0])
    while rows.size and k < n:
        k = min(4 * k, n)
        index, d2 = _neighbours(tree, tree_points, points[rows], k)
        nearest[rows] = index[:, 0]
        rows = rows[d2[:, -1] == d2[:, 0]]
    return nearest, nearest_d2


def _faces_by_vertex(faces: np.ndarray, n_vertices: int) -> tuple[np.ndarray, np.ndarray]:
    """``(start, face)``: the faces at vertex ``v`` are ``face[start[v]:start[v + 1]]``, in
    ascending order."""
    vertex = faces.ravel()
    order = np.argsort(vertex, kind="stable")  # stable: face indices stay ascending
    start = np.concatenate([[0], np.cumsum(np.bincount(vertex, minlength=n_vertices))])
    return start, order // 3


def build_sphere_map(source: Surface, target: Float[np.ndarray, "t 3"], *, k: int = 8) -> SphereMap:
    """Map every vertex of the ``target`` sphere into the faces of the ``source`` sphere.

    Both are put on the unit sphere. For each target point the faces around its ``k``
    nearest source vertices are searched, nearest vertex first and faces in ascending
    index, for the first one that contains it; a point that none contains falls back to its
    nearest source vertex. A face wound inward never matches; when more than half of the
    source faces are, every face is reversed once and ``flipped`` is set. No input raises
    but a degenerate sphere (:func:`unit_sphere`). The result does not depend on thread
    count or platform.

    A target point that is a source vertex (within :data:`COINCIDENT`) takes that vertex
    alone, with the row of a fallback, ``(v, v, v)`` and ``(1, 0, 0)``, and ``fallback``
    False: a sphere mapped onto itself reproduces any field exactly.

    A face that names a vertex the sphere does not have is ``ValueError("degenerate
    sphere: ...")`` as well.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, not {k}")
    src = unit_sphere(source.vertices)
    tgt = unit_sphere(target)
    n_source, n_target = len(src), len(tgt)
    faces = np.asarray(source.faces, dtype=np.int64)
    if faces.size and (faces.min() < 0 or faces.max() >= n_source):
        raise ValueError(
            f"degenerate sphere: faces name vertices outside 0 to {n_source - 1} "
            f"({int(faces.min())} to {int(faces.max())})"
        )

    def normals(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        a, b, c = src[f[:, 0]], src[f[:, 1]], src[f[:, 2]]
        bxc = _cross(b, c)
        return bxc, _cross(c, a), _cross(a, b), _dot(a, bxc)

    bxc, cxa, axb, det = normals(faces)
    flipped = bool(2 * int((det < 0).sum()) > len(faces))
    if flipped:
        # reversed, not negated: the map is then the one of the same sphere wound outward,
        # bit for bit
        faces = np.ascontiguousarray(faces[:, ::-1])
        bxc, cxa, axb, det = normals(faces)
    outward = det > 0

    tree = cKDTree(src)
    k = min(k, n_source)
    neighbour, neighbour_d2 = _neighbours(tree, src, tgt, k)
    nearest_source = neighbour[:, 0].copy()
    if k < n_source:
        # all k candidates equally near: a lower index may be among those left out
        tied = np.flatnonzero(neighbour_d2[:, -1] == neighbour_d2[:, 0])
        if tied.size:
            nearest_source[tied] = _nearest(tree, src, tgt[tied])[0]
    nearest_target = _nearest(cKDTree(tgt), tgt, src)[0]

    corners = np.repeat(nearest_source[:, None], 3, axis=1)
    weights = np.zeros((n_target, 3), dtype=np.float64)
    weights[:, 0] = 1.0
    resolved = neighbour_d2[:, 0] <= COINCIDENT**2
    fallback = ~resolved

    start, face_at = _faces_by_vertex(faces, n_source)
    unresolved = np.flatnonzero(~resolved)
    for j in range(k):
        if not unresolved.size:
            break
        point = unresolved
        vertex = neighbour[point, j]
        degree = start[vertex + 1] - start[vertex]
        slot = 0
        while point.size:
            more = degree > slot
            point, vertex, degree = point[more], vertex[more], degree[more]
            if not point.size:
                break
            face = face_at[start[vertex] + slot]
            p = tgt[point]
            c0, c1, c2 = _dot(p, bxc[face]), _dot(p, cxa[face]), _dot(p, axb[face])
            total = c0 + c1 + c2
            floor = -CONTAIN_TOLERANCE * total
            inside = outward[face] & (total > 0) & (c0 >= floor) & (c1 >= floor) & (c2 >= floor)
            if inside.any():
                found = point[inside]
                w = np.stack([c0[inside], c1[inside], c2[inside]], axis=1)
                np.maximum(w, 0.0, out=w)
                weights[found] = w / (w[:, 0] + w[:, 1] + w[:, 2])[:, None]
                corners[found] = faces[face[inside]]
                fallback[found] = False
                keep = ~inside
                point, vertex, degree = point[keep], vertex[keep], degree[keep]
            slot += 1
        unresolved = unresolved[fallback[unresolved]]

    return SphereMap(
        n_source=n_source,
        n_target=n_target,
        corners=corners.astype(np.int32),
        weights=weights,
        fallback=fallback,
        nearest_source=nearest_source.astype(np.int32),
        nearest_target=nearest_target.astype(np.int32),
        flipped=flipped,
    )


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
    are scaled to sum to 1. A target that loses no weight keeps its weights as they are, so
    flags that are all true change nothing. A corner without weight contributes nothing,
    whatever its value: a NaN or an infinity there does not reach the result.
    """
    if len(values) != smap.n_source:
        raise ValueError(f"{len(values)} values for a sphere of {smap.n_source} vertices")
    v = np.asarray(values, dtype=np.float64)[smap.corners]
    w = smap.weights
    if valid is None:
        return _weighted(v, w)
    if len(valid) != smap.n_source:
        raise ValueError(f"{len(valid)} flags for a sphere of {smap.n_source} vertices")
    kept = np.where(valid[smap.corners], w, 0.0)
    lost = (kept != w).any(axis=1)
    # a row that lost nothing has the weight 1, not its sum, which may be a rounding step off
    total = np.where(lost, kept[:, 0] + kept[:, 1] + kept[:, 2], 1.0)
    keep = (total >= min_valid_weight) & (total > 0)
    out = np.full(smap.n_target, np.nan)
    out[keep] = _weighted(
        v[keep], np.where(lost[keep, None], kept[keep] / total[keep, None], w[keep])
    )
    return out


def _weighted(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    term = np.zeros_like(v)
    np.multiply(v, w, out=term, where=w > 0)
    return term[:, 0] + term[:, 1] + term[:, 2]
