"""The display geometry: the grid, the voxel map of the raw stream, the resampling. Work
package E1.B.

Let ``G`` be the affine of the display grid (voxel to world), ``S_r`` the point map of run
``r`` from the anatomy's world to its scanner world (``RigidPath.matrix``), ``S_a`` that of
the session's anchor run, ``R_r`` the arbitrated raw affine of run ``r`` and ``P_r`` the map
from the display world to the anatomy's world (the identity in the anatomy-space layout). The
preprocessed stream is the stored series on ``G``. The raw stream of run ``r`` samples the raw
series at voxel ``R_r^-1 . S_a . P_r . G . v`` for every display voxel ``v``: a cubic B-spline
(``scipy.ndimage.affine_transform``, order 3, edge values repeated), 0 wherever the point lies
outside the outermost raw voxel centres along any axis. No other transform is ever applied to
the raw data (D36).

The portable voxel map is computed in Python floats in a fixed order (stage 1's
``session.voxel_map(portable=True)``), so that it does not depend on the linear-algebra
library numpy is linked with.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import nibabel as nib
import numpy as np
from jaxtyping import Float
from scipy import ndimage

from surfcensus.records import IDENTITY4, Grid, Matrix4, to_matrix4

GRID_TOLERANCE = 1e-4
"""Two runs lie on one grid when their shapes are equal and their affines agree to this."""
SPLINE_ORDER = 3


def _product(a: Matrix4, b: Matrix4) -> Matrix4:
    # the sum of each entry starts from 0 and adds k = 0, 1, 2, 3 in that order, as stage 1's
    # portable map does, so the digits do not depend on how a library would order the sum
    return to_matrix4(
        [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
    )


def compose(*matrices: Matrix4) -> Matrix4:
    """The product of the matrices, left to right, in Python floats in a fixed order.

    ``compose(a, b, c)`` is ``(a . b) . c``: applied to a point, ``c`` acts first. Every entry
    is a sum over ``k = 0 .. 3`` in that order. Without matrices it is the identity."""
    out = IDENTITY4
    for n, matrix in enumerate(matrices):
        out = matrix if n == 0 else _product(out, matrix)
    return to_matrix4(out)


def inverse(matrix: Matrix4) -> Matrix4:
    """The inverse of an affine by cofactors, in Python floats. Raises ``ValueError`` for one
    that cannot be inverted."""
    (a00, a01, a02), (a10, a11, a12), (a20, a21, a22) = (row[:3] for row in matrix[:3])
    c00, c01, c02 = a11 * a22 - a12 * a21, a12 * a20 - a10 * a22, a10 * a21 - a11 * a20
    det = a00 * c00 + a01 * c01 + a02 * c02
    if det == 0.0 or not math.isfinite(det):
        raise ValueError("the affine cannot be inverted")
    inv = [
        [c00 / det, (a02 * a21 - a01 * a22) / det, (a01 * a12 - a02 * a11) / det],
        [c01 / det, (a00 * a22 - a02 * a20) / det, (a02 * a10 - a00 * a12) / det],
        [c02 / det, (a01 * a20 - a00 * a21) / det, (a00 * a11 - a01 * a10) / det],
    ]
    t = [matrix[0][3], matrix[1][3], matrix[2][3]]
    rows = [[*inv[i], -sum(inv[i][k] * t[k] for k in range(3))] for i in range(3)]
    return to_matrix4([*rows, [0.0, 0.0, 0.0, 1.0]])


def grid_of(series: Path) -> Grid:
    """The grid of a NIfTI series: its first three dimensions and ``img.affine``. Reads the
    header only."""
    img = nib.load(str(series))
    shape = tuple(int(n) for n in img.shape[:3])
    if len(shape) != 3:
        raise ValueError(f"{Path(series).name}: not a 3D or 4D image")
    return Grid(shape=(shape[0], shape[1], shape[2]), affine=to_matrix4(img.affine))


def same_grid(a: Grid, b: Grid) -> bool:
    """Equal shapes, and affines within :data:`GRID_TOLERANCE`."""
    if tuple(a.shape) != tuple(b.shape):
        return False
    return all(
        abs(x - y) <= GRID_TOLERANCE
        for row_a, row_b in zip(a.affine, b.affine, strict=True)
        for x, y in zip(row_a, row_b, strict=True)
    )


def _column(grid: Grid, axis: int) -> tuple[float, float, float]:
    return (grid.affine[0][axis], grid.affine[1][axis], grid.affine[2][axis])


def voxel_mm(grid: Grid) -> tuple[float, float, float]:
    """The voxel size along each voxel axis, in mm: the length of each column of the 3x3
    part of the affine."""
    x, y, z = (math.sqrt(sum(c * c for c in _column(grid, axis))) for axis in range(3))
    return (x, y, z)


def oblique_deg(grid: Grid) -> float:
    """The largest angle between a voxel axis and the world axis nearest it, in degrees; 0
    for a grid whose voxel axes run along world axes (in any order and sense)."""
    worst = 0.0
    for axis, size in enumerate(voxel_mm(grid)):
        if size == 0.0:
            raise ValueError("a voxel axis of the grid has no length")
        cosine = min(1.0, max(abs(c) for c in _column(grid, axis)) / size)
        worst = max(worst, math.degrees(math.acos(cosine)))
    return worst


_LETTERS = (("L", "R"), ("P", "A"), ("I", "S"))
_VIEWS: Mapping[str, tuple[int, int, bool]] = {
    # plane: the voxel axis along the panel's horizontal and vertical, and whether the
    # horizontal runs against the voxel axis (stage 1's player/logic.js, SPEC): sagittal with
    # posterior at the left, coronal and axial radiological, superior or anterior up
    "sagittal": (1, 2, False),
    "coronal": (0, 2, True),
    "axial": (0, 1, True),
}


def _letter(direction: Sequence[float]) -> str:
    axis = max(range(3), key=lambda n: (abs(direction[n]), -n))  # the first of equal ones
    return _LETTERS[axis][direction[axis] > 0]


def edge_letters(grid: Grid) -> Mapping[str, tuple[str, str, str, str]]:
    """Plane -> the anatomical letters at the left, right, top and bottom edge of the plane as
    the page draws it (``R``, ``L``, ``A``, ``P``, ``S``, ``I`` of the world axis nearest each
    direction).

    The page draws plane ``n`` with the voxel axis given by stage 1's player along the
    horizontal (axis 1 for sagittal, 0 for coronal and axial), the coronal and axial ones
    running from right to left, and the vertical axis (2, 2, 1) running upward. On a grid in
    RAS order that gives ``P A S I``, ``R L S I`` and ``R L A P``."""
    out: dict[str, tuple[str, str, str, str]] = {}
    for plane, (h, v, flip) in _VIEWS.items():
        along = _column(grid, h)
        left = along if flip else tuple(-c for c in along)
        up = _column(grid, v)
        out[plane] = (
            _letter(left),
            _letter([-c for c in left]),
            _letter(up),
            _letter([-c for c in up]),
        )
    return out


def voxel_map(
    raw_affine: Matrix4,
    anchor_matrix: Matrix4,
    grid: Grid,
    display_to_anat: Matrix4 = IDENTITY4,
    portable: bool = True,
) -> Float[np.ndarray, "4 4"]:
    """``R_r^-1 . S_a . P_r . G``: display voxel to raw voxel. ``portable`` False is numpy's
    inverse and product, the expression of the prototype.

    Portable, the map is :func:`compose` of :func:`inverse` of ``raw_affine``, the anchor's
    matrix, ``display_to_anat`` and the grid's affine, in Python floats; numpy's form is
    ``np.linalg.inv(R) @ S_a @ P_r @ G``, whose last bits can depend on the LAPACK and BLAS that
    numpy is linked with. An identity ``display_to_anat`` (the anatomy-space layout) is left out
    of either product, so that both give the bits of stage 1's ``session.voxel_map``."""
    middle = () if display_to_anat == IDENTITY4 else (display_to_anat,)
    if portable:
        return np.array(
            compose(inverse(raw_affine), anchor_matrix, *middle, grid.affine), dtype=np.float64
        )
    product = np.linalg.inv(np.array(raw_affine, dtype=np.float64)) @ np.array(
        anchor_matrix, dtype=np.float64
    )
    for matrix in (*middle, grid.affine):
        product = product @ np.array(matrix, dtype=np.float64)
    return product


_INSIDE: dict[tuple[object, ...], np.ndarray] = {}
_INSIDE_LIMIT = 64


def _inside(
    source_shape: tuple[int, ...], vox: np.ndarray, shape: tuple[int, int, int]
) -> np.ndarray:
    """Where a display voxel's sample lies within the outermost voxel centres of the source:
    an order-0 lookup of a volume of ones with the constant 0 outside, stage 1's rule. Kept
    for the last maps, since every volume of a run has the same one."""
    key = (source_shape, shape, vox.tobytes())
    found = _INSIDE.get(key)
    if found is None:
        if len(_INSIDE) >= _INSIDE_LIMIT:
            _INSIDE.clear()
        found = ndimage.affine_transform(
            np.ones(source_shape, np.uint8),
            vox[:3, :3],
            offset=vox[:3, 3],
            output_shape=shape,
            order=0,
            mode="constant",
            cval=0,
        ).astype(bool)
        _INSIDE[key] = found
    return found


def resample_voxels(
    volume: Float[np.ndarray, "a b c"],
    vox: Float[np.ndarray, "4 4"],
    shape: tuple[int, int, int],
    order: int = SPLINE_ORDER,
) -> Float[np.ndarray, "i j k"]:
    """``volume`` at the voxel centres of the display grid through the voxel map, float32;
    0 outside the outermost voxel centres of ``volume``.

    The rule of stage 1 (``session.resample_voxels``): a B-spline of ``order`` with the edge
    values repeated (scipy's mode ``nearest``), then 0 wherever the sample lies outside
    ``[0, n - 1]`` in the voxel coordinates of ``volume`` along any axis, decided by an order-0
    lookup of a volume of ones with the constant 0 outside. So a display voxel whose sample lies
    within half a raw voxel beyond the outermost raw voxel centre is 0, not the edge value."""
    vox = np.asarray(vox, dtype=np.float64)
    grid_shape = (int(shape[0]), int(shape[1]), int(shape[2]))
    out = ndimage.affine_transform(
        volume,
        vox[:3, :3],
        offset=vox[:3, 3],
        output_shape=grid_shape,
        order=order,
        mode="nearest",
        output=np.float32,
    )
    out[~_inside(tuple(volume.shape), vox, grid_shape)] = 0
    return out


def raw_stream(
    raw_bold: Path,
    raw_affine: Matrix4,
    grid: Grid,
    anchor_matrix: Matrix4,
    display_to_anat: Matrix4 = IDENTITY4,
    volumes: Sequence[int] | None = None,
    portable: bool = True,
) -> Iterator[tuple[int, Float[np.ndarray, "i j k"]]]:
    """The raw stream of a run: ``(t, float32 volume on the display grid)`` per volume, the
    stored values scaled per volume (``* slope + inter``), one volume read at a time through
    the array proxy. ``volumes``: indices to yield, all when None.

    A volume is ``stored[..., t].astype(float32) * slope + inter``, the values of stage 1's
    ``session.raw_volumes``, read with nibabel's per-slice read of the stored values so that
    no more than one volume is in memory; the file is kept open between volumes, so that a
    compressed series is decompressed once when the volumes are read in order. Every volume
    goes through the same voxel map (:func:`voxel_map`): the series is neither
    motion-corrected nor slice-time-corrected nor unwarped (D36). An index outside the series
    raises ``IndexError``."""
    vox = voxel_map(raw_affine, anchor_matrix, grid, display_to_anat, portable)
    img = nib.load(str(raw_bold), keep_file_open=True)
    try:
        proxy = img.dataobj
        shape = tuple(int(n) for n in img.shape)
        n = shape[3] if len(shape) > 3 else 1
        slope = float(getattr(proxy, "slope", 1.0))
        inter = float(getattr(proxy, "inter", 0.0))
        for t in range(n) if volumes is None else volumes:
            t = int(t)
            if not 0 <= t < n:
                raise IndexError(f"{Path(raw_bold).name} has {n} volumes, not volume {t}")
            stored = _stored_volume(proxy, len(shape), t)
            yield t, resample_voxels(stored.astype(np.float32) * slope + inter, vox, grid.shape)
    finally:
        del img


def _stored_volume(proxy: object, ndim: int, t: int) -> np.ndarray:
    """Volume ``t`` of a series in its stored values, before the scaling of the header."""
    index = (slice(None), slice(None), slice(None)) + ((t,) if ndim > 3 else ())
    index += (0,) * max(0, ndim - 4)  # a 5D image: its first entry along the higher axes
    read = getattr(proxy, "_get_unscaled", None)
    if read is not None:  # nibabel's ArrayProxy: the stored values of one volume
        return np.asarray(read(index))
    return np.asarray(proxy)[index]  # pyrefly: ignore[bad-index]
