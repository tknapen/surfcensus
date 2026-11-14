"""White and pial contours of both hemispheres for every slice of every plane of a display grid,
as geometry that the player page draws.

A module for pack_run.py; it has no command line.

The surfaces are taken into the VOXEL space of the display grid (inverse of its affine), so a
slice is the plane "voxel coordinate along one axis = whole number" and is cut by the package's
MeshSlicer whatever the orientation of the grid. A whole voxel coordinate is the centre of a
voxel, so the plane of slice k runs through the centres of its voxels, as in the surface renders
of the package (docs/rendering.md).

One file per plane (contour_bytes), little-endian, compressed with zlib by the caller:

  uint32  n_slices, n_surfaces, n_lines, n_points
  uint32  line_start[n_slices * n_surfaces + 1]   first line of every (slice, surface) cell;
                                                  cell = slice * n_surfaces + surface
  uint32  point_start[n_lines + 1]                first point of every line
  int16   xy[2 * n_points]                        per line: its first point, then differences
                                                  to the point before, in units of 1 / UNIT voxel

A point is (a, b): the voxel coordinates along the two in-plane axes in ascending order of axis
(sagittal: j, k; coronal: i, k; axial: i, j). A line whose last point equals its first is closed.
Surfaces are in the order of SURFACES. Points are rounded to 1 / UNIT voxel (0.06 mm for a 4 mm
voxel), and segments that have no length after rounding are dropped.
"""

import struct

import numpy as np

from surfcensus.render.slicing import MeshSlicer

UNIT = 64  # points are stored in 1/64 voxel
SURFACES = ("lh.white", "rh.white", "lh.pial", "rh.pial")  # white first: pial is drawn over it
PLANES = ("sagittal", "coronal", "axial")  # the slice axis of plane n is voxel axis n
_LIMIT = 32000  # of int16, with room for a difference between two clamped points


def to_voxel(vertices, grid_affine):
    """World (T1w RAS) coordinates as voxel coordinates of the display grid."""
    inv = np.linalg.inv(np.asarray(grid_affine, float))
    return np.asarray(vertices, float) @ inv[:3, :3].T + inv[:3, 3]


def chain(segments):
    """Integer segments (n, 2, 2) joined into polylines: a list of (m, 2) int arrays.

    A segment follows another when it starts where that one ends (MeshSlicer orients every
    segment the same way round its contour). Segments without length are dropped. A closed line
    repeats its first point at the end.
    """
    seg = [s for s in segments.tolist() if s[0] != s[1]]
    starts = {}
    for n, (a, _) in enumerate(seg):
        starts.setdefault(tuple(a), []).append(n)
    ends = {}
    for _, b in seg:
        ends[tuple(b)] = ends.get(tuple(b), 0) + 1
    used = [False] * len(seg)

    def walk(first):
        line = [seg[first][0], seg[first][1]]
        used[first] = True
        while True:
            here = starts.get(tuple(line[-1]), ())
            nxt = next((n for n in here if not used[n]), None)
            if nxt is None:
                return np.array(line, dtype=np.int64)
            used[nxt] = True
            line.append(seg[nxt][1])

    lines = []
    # open lines first, from the segments that nothing leads into, then what is left (loops)
    for n, (a, _) in enumerate(seg):
        if not used[n] and ends.get(tuple(a), 0) < len(starts[tuple(a)]):
            lines.append(walk(n))
    for n in range(len(seg)):
        if not used[n]:
            lines.append(walk(n))
    return lines


def plane_lines(slicers, axis, n_slices):
    """Per slice and surface the polylines of one plane, in fixed-point in-plane coordinates."""
    keep = [a for a in range(3) if a != axis]
    cells = []
    for k in range(n_slices):
        for name in SURFACES:
            seg = slicers[name].segments(axis, float(k))[:, :, keep]
            fixed = np.clip(np.rint(seg * UNIT), -_LIMIT // 2, _LIMIT // 2).astype(np.int64)
            cells.append(chain(fixed))
    return cells


def contour_bytes(cells, n_slices):
    """The file of one plane (module docstring) from the cells of plane_lines, uncompressed."""
    line_start, point_start, xy = [0], [0], []
    for lines in cells:
        for line in lines:
            delta = np.diff(line, axis=0, prepend=np.zeros((1, 2), np.int64))
            xy.append(delta)
            point_start.append(point_start[-1] + len(line))
        line_start.append(len(point_start) - 1)
    points = np.concatenate(xy) if xy else np.zeros((0, 2), np.int64)
    if len(points) and np.abs(points).max() > 32767:
        raise ValueError("a contour point does not fit the fixed-point format")
    head = struct.pack("<4I", n_slices, len(SURFACES), len(point_start) - 1, len(points))
    return b"".join(
        (
            head,
            np.asarray(line_start, "<u4").tobytes(),
            np.asarray(point_start, "<u4").tobytes(),
            points.astype("<i2").tobytes(),
        )
    )


def read_contours(data):
    """The inverse of contour_bytes: cells[slice * n_surfaces + surface] = list of (m, 2) arrays
    in voxel units. For tests; the page has its own reader (player/logic.js)."""
    n_slices, n_surfaces, n_lines, n_points = struct.unpack_from("<4I", data, 0)
    at = 16
    n_cells = n_slices * n_surfaces
    line_start = np.frombuffer(data, "<u4", n_cells + 1, at)
    at += 4 * (n_cells + 1)
    point_start = np.frombuffer(data, "<u4", n_lines + 1, at)
    at += 4 * (n_lines + 1)
    delta = np.frombuffer(data, "<i2", 2 * n_points, at).reshape(-1, 2).astype(np.int64)
    cells = []
    for c in range(n_cells):
        lines = []
        for n in range(line_start[c], line_start[c + 1]):
            part = delta[point_start[n] : point_start[n + 1]]
            lines.append(np.cumsum(part, axis=0) / UNIT)
        cells.append(lines)
    return cells


def session_contours(surfaces, grid_shape, grid_affine):
    """{plane name: uncompressed file bytes} and counts, from surfaces in T1w world space.

    surfaces: {(hemi, kind): (vertices, faces)} as pipeline/geom.py's surfaces_in_t1w returns.
    """
    slicers = {
        f"{hemi}.{kind}": MeshSlicer(to_voxel(v, grid_affine), f)
        for (hemi, kind), (v, f) in surfaces.items()
    }
    files, counts = {}, {}
    for axis, plane in enumerate(PLANES):
        cells = plane_lines(slicers, axis, int(grid_shape[axis]))
        files[plane] = contour_bytes(cells, int(grid_shape[axis]))
        counts[plane] = {
            "slices": int(grid_shape[axis]),
            "lines": sum(len(lines) for lines in cells),
            "points": sum(len(line) for lines in cells for line in lines),
        }
    return files, counts
