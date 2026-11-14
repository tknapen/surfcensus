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

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np
from jaxtyping import Float

from surfcensus.records import IDENTITY4, Grid, Matrix4

GRID_TOLERANCE = 1e-4
"""Two runs lie on one grid when their shapes are equal and their affines agree to this."""
SPLINE_ORDER = 3


def compose(*matrices: Matrix4) -> Matrix4:
    """The product of the matrices, left to right, in Python floats in a fixed order."""
    raise NotImplementedError("work package E1.B: func/geometry.py")


def inverse(matrix: Matrix4) -> Matrix4:
    """The inverse of an affine by cofactors, in Python floats. Raises ``ValueError`` for one
    that cannot be inverted."""
    raise NotImplementedError("work package E1.B: func/geometry.py")


def grid_of(series: Path) -> Grid:
    """The grid of a NIfTI series: its first three dimensions and ``img.affine``."""
    raise NotImplementedError("work package E1.B: func/geometry.py")


def same_grid(a: Grid, b: Grid) -> bool:
    """Equal shapes, and affines within :data:`GRID_TOLERANCE`."""
    raise NotImplementedError("work package E1.B: func/geometry.py")


def voxel_mm(grid: Grid) -> tuple[float, float, float]:
    """The voxel size along each voxel axis, in mm."""
    raise NotImplementedError("work package E1.B: func/geometry.py")


def oblique_deg(grid: Grid) -> float:
    """The largest angle between a voxel axis and the world axis nearest it, in degrees."""
    raise NotImplementedError("work package E1.B: func/geometry.py")


def edge_letters(grid: Grid) -> Mapping[str, tuple[str, str, str, str]]:
    """Plane -> the anatomical letters at the left, right, top and bottom edge of the plane as
    the page draws it (``R``, ``L``, ``A``, ``P``, ``S``, ``I`` of the world axis nearest each
    direction)."""
    raise NotImplementedError("work package E1.B: func/geometry.py")


def voxel_map(
    raw_affine: Matrix4,
    anchor_matrix: Matrix4,
    grid: Grid,
    display_to_anat: Matrix4 = IDENTITY4,
    portable: bool = True,
) -> Float[np.ndarray, "4 4"]:
    """``R_r^-1 . S_a . P_r . G``: display voxel to raw voxel. ``portable`` False is numpy's
    inverse and product, the expression of the prototype."""
    raise NotImplementedError("work package E1.B: func/geometry.py")


def resample_voxels(
    volume: Float[np.ndarray, "a b c"],
    vox: Float[np.ndarray, "4 4"],
    shape: tuple[int, int, int],
    order: int = SPLINE_ORDER,
) -> Float[np.ndarray, "i j k"]:
    """``volume`` at the voxel centres of the display grid through the voxel map, float32;
    0 outside the outermost voxel centres of ``volume``."""
    raise NotImplementedError("work package E1.B: func/geometry.py")


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
    the array proxy. ``volumes``: indices to yield, all when None."""
    raise NotImplementedError("work package E1.B: func/geometry.py")
