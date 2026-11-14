"""The white and pial contours of both hemispheres for every slice of every plane of a display
grid. Work package E1.B.

The surfaces go from FreeSurfer's tkr RAS through ``T1.mgz`` (vox2ras times the inverse of
vox2ras-tkr) to scanner RAS, through ``anat_to_fsnative`` to the anatomy's world, and through
``(P_r . G)^-1`` to display voxels, where the package's ``MeshSlicer`` cuts them with the
planes of whole voxel coordinates (the centres of the voxels). The file of a plane is that of
stage 1's ``pack_contours.py``, little-endian, compressed by the pack::

  uint32  n_slices, n_surfaces, n_lines, n_points
  uint32  line_start[n_slices * n_surfaces + 1]
  uint32  point_start[n_lines + 1]
  int16   xy[2 * n_points]     per line: its first point, then differences, in 1/64 voxel

A point is the voxel coordinates along the two in-plane axes in ascending order of axis;
surfaces are in the order of ``func.SURFACES``.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from pathlib import Path

import nibabel as nib
import numpy as np
from jaxtyping import Float, Int

from surfcensus.fsio.surfaces import read_surface
from surfcensus.func import CONTOUR_UNIT, PLANES, SURFACES
from surfcensus.func.geometry import compose, inverse
from surfcensus.records import IDENTITY4, Grid, Matrix4, PlaneContours, to_matrix4
from surfcensus.render.slicing import MeshSlicer
from surfcensus.sources.base import FileRole
from surfcensus.sources.patterns import resolve_subject_files

_LIMIT = 32000
"""Of int16, with room for a difference between two clamped points: a point is clamped to
half of it, 250 voxels from the grid's origin."""

_ROLES: Mapping[str, FileRole] = {
    "lh.white": FileRole.LH_WHITE,
    "rh.white": FileRole.RH_WHITE,
    "lh.pial": FileRole.LH_PIAL,
    "rh.pial": FileRole.RH_PIAL,
}


def tkr_to_display(
    fs_subject_dir: Path,
    anat_to_fsnative: Matrix4,
    grid: Grid,
    display_to_anat: Matrix4 = IDENTITY4,
) -> Matrix4:
    """The point map from FreeSurfer's tkr RAS to display voxels:
    ``inv(P_r . G) . Q . vox2ras . inv(vox2ras_tkr)``, with ``Q`` the anatomy's world from
    FreeSurfer's scanner RAS (``anat_to_fsnative``) and the two affines of ``mri/T1.mgz``, in
    Python floats. Reads the header of ``T1.mgz`` only."""
    files = resolve_subject_files(fs_subject_dir)
    if FileRole.T1 not in files:
        raise FileNotFoundError("mri/T1.mgz is missing from the FreeSurfer subject")
    t1 = nib.load(str(files[FileRole.T1]))
    vox2ras = to_matrix4(np.asarray(t1.affine, dtype=np.float64))
    vox2ras_tkr = to_matrix4(np.asarray(t1.header.get_vox2ras_tkr(), dtype=np.float64))
    display = compose(display_to_anat, grid.affine)
    return compose(inverse(display), anat_to_fsnative, vox2ras, inverse(vox2ras_tkr))


def _apply(matrix: Matrix4, points: np.ndarray) -> np.ndarray:
    """``matrix`` applied to points ``(n, 3)`` entry by entry, without numpy's linear
    algebra."""
    p = np.asarray(points, dtype=np.float64)
    columns = [
        p[:, 0] * matrix[r][0] + p[:, 1] * matrix[r][1] + p[:, 2] * matrix[r][2] + matrix[r][3]
        for r in range(3)
    ]
    return np.column_stack(columns)


def surfaces_in_display(
    fs_subject_dir: Path,
    anat_to_fsnative: Matrix4,
    grid: Grid,
    display_to_anat: Matrix4 = IDENTITY4,
) -> Mapping[str, tuple[Float[np.ndarray, "n 3"], Int[np.ndarray, "m 3"]]]:
    """``lh.white``, ``rh.white``, ``lh.pial``, ``rh.pial`` of a FreeSurfer subject as
    (vertices in display voxel coordinates, faces). The pial file is resolved as the surface
    source resolves it (``?h.pial``, else its variants).

    The vertices, in tkr RAS as stored, go through :func:`tkr_to_display`: through ``T1.mgz``
    to FreeSurfer's scanner RAS, through ``anat_to_fsnative`` to the anatomy's world, through
    ``inv(P_r . G)`` to display voxels; one matrix, applied entry by entry. Raises
    ``FileNotFoundError`` for a missing surface or ``T1.mgz``, naming the file and not the
    path."""
    files = resolve_subject_files(fs_subject_dir)
    missing = [name for name in SURFACES if _ROLES[name] not in files]
    if missing:
        raise FileNotFoundError(f"surf/{missing[0]} is missing from the FreeSurfer subject")
    matrix = tkr_to_display(fs_subject_dir, anat_to_fsnative, grid, display_to_anat)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in SURFACES:
        surface = read_surface(files[_ROLES[name]])
        out[name] = (_apply(matrix, surface.vertices), surface.faces)
    return out


def chain(segments: Int[np.ndarray, "n 2 2"]) -> Sequence[Int[np.ndarray, "m 2"]]:
    """Integer segments joined into polylines; a closed line repeats its first point.

    A segment follows another when it starts where that one ends (``MeshSlicer`` orients every
    segment the same way round its contour). Segments without length are dropped. Open lines
    come first, each from a segment that nothing leads into, in the order of the segments;
    then the loops, each from its first unused segment. Stage 1's ``pack_contours.chain``."""
    seg = [s for s in np.asarray(segments).tolist() if s[0] != s[1]]
    starts: dict[tuple[int, ...], list[int]] = {}
    for n, (a, _) in enumerate(seg):
        starts.setdefault(tuple(a), []).append(n)
    ends: dict[tuple[int, ...], int] = {}
    for _, b in seg:
        ends[tuple(b)] = ends.get(tuple(b), 0) + 1
    used = [False] * len(seg)

    def walk(first: int) -> np.ndarray:
        line = [seg[first][0], seg[first][1]]
        used[first] = True
        while True:
            here = starts.get(tuple(line[-1]), [])
            nxt = next((n for n in here if not used[n]), None)
            if nxt is None:
                return np.array(line, dtype=np.int64)
            used[nxt] = True
            line.append(seg[nxt][1])

    lines = []
    for n, (a, _) in enumerate(seg):
        if not used[n] and ends.get(tuple(a), 0) < len(starts[tuple(a)]):
            lines.append(walk(n))
    for n in range(len(seg)):
        if not used[n]:
            lines.append(walk(n))
    return lines


def _plane_cells(
    slicers: Mapping[str, MeshSlicer], axis: int, n_slices: int
) -> list[list[np.ndarray]]:
    """Per slice and surface the polylines of one plane, in fixed point, in-plane axes in
    ascending order."""
    keep = [a for a in range(3) if a != axis]
    cells = []
    for k in range(n_slices):
        for name in SURFACES:
            seg = slicers[name].segments(axis, float(k))[:, :, keep]
            fixed = np.clip(np.rint(seg * CONTOUR_UNIT), -_LIMIT // 2, _LIMIT // 2)
            cells.append(list(chain(fixed.astype(np.int64))))
    return cells


def _contour_bytes(cells: Sequence[Sequence[np.ndarray]], n_slices: int) -> bytes:
    line_start, point_start, xy = [0], [0], []
    for lines in cells:
        for line in lines:
            xy.append(np.diff(line, axis=0, prepend=np.zeros((1, 2), np.int64)))
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


def plane_contours(
    surfaces: Mapping[str, tuple[Float[np.ndarray, "n 3"], Int[np.ndarray, "m 3"]]],
    grid: Grid,
) -> tuple[PlaneContours, ...]:
    """The three planes of a grid, in the order of ``func.PLANES``.

    Plane ``n`` is cut at every whole coordinate ``0 .. shape[n] - 1`` of voxel axis ``n``
    (the centres of its voxels) by ``render.slicing.MeshSlicer``; the in-plane coordinates of
    each segment are rounded to ``1 / CONTOUR_UNIT`` voxel and clamped to 250 voxels, segments
    without length after rounding are dropped, and the rest joined by :func:`chain`. The file
    of a plane is that of the module docstring. ``surfaces`` must hold the four of
    ``func.SURFACES`` (vertices in display voxels, as :func:`surfaces_in_display` gives)."""
    missing = [name for name in SURFACES if name not in surfaces]
    if missing:
        raise ValueError(f"no surface {missing[0]}")
    slicers = {name: MeshSlicer(*surfaces[name]) for name in SURFACES}
    out = []
    for axis, plane in enumerate(PLANES):
        n_slices = int(grid.shape[axis])
        cells = _plane_cells(slicers, axis, n_slices)
        out.append(
            PlaneContours(
                plane=plane,
                data=_contour_bytes(cells, n_slices),
                slices=n_slices,
                lines=sum(len(lines) for lines in cells),
                points=sum(len(line) for lines in cells for line in lines),
            )
        )
    return tuple(out)


def read_contours(data: bytes) -> list[list[Float[np.ndarray, "m 2"]]]:
    """The inverse of the file layout: ``cells[slice * n_surfaces + surface]`` is a list of
    lines in voxel units. For tests; the page has its own reader."""
    n_slices, n_surfaces, n_lines, n_points = struct.unpack_from("<4I", data, 0)
    at = 16
    n_cells = n_slices * n_surfaces
    line_start = np.frombuffer(data, "<u4", n_cells + 1, at)
    at += 4 * (n_cells + 1)
    point_start = np.frombuffer(data, "<u4", n_lines + 1, at)
    at += 4 * (n_lines + 1)
    delta = np.frombuffer(data, "<i2", 2 * n_points, at).reshape(-1, 2).astype(np.int64)
    if at + 4 * n_points != len(data):
        raise ValueError("the contour file is not of its stated length")
    cells: list[list[np.ndarray]] = []
    for c in range(n_cells):
        lines = []
        for n in range(int(line_start[c]), int(line_start[c + 1])):
            part = delta[point_start[n] : point_start[n + 1]]
            lines.append(np.cumsum(part, axis=0) / CONTOUR_UNIT)
        cells.append(lines)
    return cells
