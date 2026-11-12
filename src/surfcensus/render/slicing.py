"""Mesh-plane intersection for axis-aligned planes (docs/rendering.md, "Contours").

Conventions:

* The plane is ``x[axis] == position`` in the frame of the vertices (tkr RAS for FreeSurfer
  surfaces). A vertex is on the positive side when ``x[axis] >= position``, so a vertex exactly
  on the plane counts as positive.
* Every face with vertices on both sides gives exactly one segment, between its two edge
  crossings. A crossing is interpolated from the positive towards the negative end of its edge,
  so the two faces that share an edge compute bit-identical points; its coordinate along
  ``axis`` is then set to ``position`` exactly.
* Orientation: a segment runs along ``e_axis x n``, with ``n`` the face normal given by the
  right-hand rule of the face winding. On a closed mesh wound with outward normals every
  contour runs counterclockwise in the coordinates ``((axis + 1) % 3, (axis + 2) % 3)``, that
  is, seen from the positive end of the axis, and every crossing point ends exactly as many
  segments as it starts (one each when the plane passes through no vertex).
* Segments come in ascending order of face index, from :func:`plane_segments` and from
  :class:`MeshSlicer` alike.
* A plane through a vertex can give zero-length segments (a face that touches the plane with
  its only positive vertex). An edge lying in the plane is produced once, by the face on its
  negative side.
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Float, Int


def plane_segments(
    vertices: Float[np.ndarray, "n 3"], faces: Int[np.ndarray, "m 3"], axis: int, position: float
) -> Float[np.ndarray, "s 2 3"]:
    """Segments (start, end) where a triangle mesh crosses the plane ``x[axis] == position``.

    One pass over all faces; use :class:`MeshSlicer` to slice the same mesh many times.
    """
    vertices, faces = _checked(vertices, faces)
    side = (vertices[:, _checked_axis(axis)] >= position)[faces]
    crossing = side.any(axis=1) & ~side.all(axis=1)
    return _segments(vertices, faces[crossing], axis, position)


class MeshSlicer:
    """Slices one mesh many times along the three axes.

    Per axis, faces are sorted by their minimum coordinate and carry the running maximum of
    their maximum coordinate, so a call bisects twice and touches only the faces in between.
    The mesh is copied; later changes to the input arrays do not affect the slicer.
    """

    def __init__(self, vertices: Float[np.ndarray, "n 3"], faces: Int[np.ndarray, "m 3"]) -> None:
        self._vertices, self._faces = _checked(vertices, faces, copy=True)
        self._index: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for axis in range(3):
            x = self._vertices[:, axis][self._faces]
            low, high = x.min(axis=1), x.max(axis=1)
            order = np.argsort(low, kind="stable")
            low, high = low[order], high[order]
            self._index.append((order, low, high, np.maximum.accumulate(high)))

    def segments(self, axis: int, position: float) -> Float[np.ndarray, "s 2 3"]:
        """Same segments, in the same order, as :func:`plane_segments`."""
        order, low, high, running_high = self._index[_checked_axis(axis)]
        # a face crosses when low < position <= high; before `start` every high is < position
        start = int(np.searchsorted(running_high, position, side="left"))
        stop = int(np.searchsorted(low, position, side="left"))
        window = slice(start, max(start, stop))
        hit = np.sort(order[window][high[window] >= position])
        return _segments(self._vertices, self._faces[hit], axis, position)


def _segments(
    vertices: Float[np.ndarray, "n 3"], faces: Int[np.ndarray, "s 3"], axis: int, position: float
) -> Float[np.ndarray, "s 2 3"]:
    """One segment per face; every face given must have vertices on both sides of the plane."""
    positive = vertices[:, axis][faces] >= position
    # the vertex that is alone on its side, and the other two in winding order after it
    lone_positive = positive.sum(axis=1) == 1
    lone = np.argmax(positive == lone_positive[:, None], axis=1)
    rows = np.arange(faces.shape[0])
    a, b, c = (faces[rows, (lone + k) % 3] for k in range(3))
    # Walking a -> b -> c -> a, the segment starts on the edge that leaves the positive side and
    # ends on the edge that returns to it. Each edge is given as (positive end, negative end).
    positive_end = np.stack([np.where(lone_positive, a, c), np.where(lone_positive, a, b)], axis=1)
    negative_end = np.stack([np.where(lone_positive, b, a), np.where(lone_positive, c, a)], axis=1)
    p, q = vertices[positive_end], vertices[negative_end]
    dp, dq = p[..., axis] - position, q[..., axis] - position
    out = p + (dp / (dp - dq))[..., None] * (q - p)
    out[..., axis] = position
    return out


def _checked(
    vertices: Float[np.ndarray, "n 3"], faces: Int[np.ndarray, "m 3"], copy: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """The mesh as float64 and intp arrays (new ones if ``copy``), refusing a malformed one."""
    v = np.array(vertices, dtype=np.float64) if copy else np.asarray(vertices, dtype=np.float64)
    f = np.array(faces, dtype=np.intp) if copy else np.asarray(faces, dtype=np.intp)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"expected vertices (n, 3) and faces (m, 3), got {v.shape} and {f.shape}")
    if not np.isfinite(v).all():
        raise ValueError("vertex coordinates must be finite")
    if f.size and (f.min() < 0 or f.max() >= v.shape[0]):
        raise ValueError(f"face indices must lie in 0 .. {v.shape[0] - 1}")
    return v, f


def _checked_axis(axis: int) -> int:
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1 or 2, got {axis!r}")
    return int(axis)
