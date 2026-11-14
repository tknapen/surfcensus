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

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int

from surfcensus.records import IDENTITY4, Grid, Matrix4, PlaneContours


def surfaces_in_display(
    fs_subject_dir: Path,
    anat_to_fsnative: Matrix4,
    grid: Grid,
    display_to_anat: Matrix4 = IDENTITY4,
) -> Mapping[str, tuple[Float[np.ndarray, "n 3"], Int[np.ndarray, "m 3"]]]:
    """``lh.white``, ``rh.white``, ``lh.pial``, ``rh.pial`` of a FreeSurfer subject as
    (vertices in display voxel coordinates, faces). The pial file is resolved as the surface
    source resolves it (``?h.pial``, else its variants)."""
    raise NotImplementedError("work package E1.B: func/contours.py")


def plane_contours(
    surfaces: Mapping[str, tuple[Float[np.ndarray, "n 3"], Int[np.ndarray, "m 3"]]],
    grid: Grid,
) -> tuple[PlaneContours, ...]:
    """The three planes of a grid, in the order of ``func.PLANES``."""
    raise NotImplementedError("work package E1.B: func/contours.py")


def read_contours(data: bytes) -> list[list[Float[np.ndarray, "m 2"]]]:
    """The inverse of the file layout: ``cells[slice * n_surfaces + surface]`` is a list of
    lines in voxel units. For tests; the page has its own reader."""
    raise NotImplementedError("work package E1.B: func/contours.py")


def chain(segments: Int[np.ndarray, "n 2 2"]) -> Sequence[Int[np.ndarray, "m 2"]]:
    """Integer segments joined into polylines; a closed line repeats its first point."""
    raise NotImplementedError("work package E1.B: func/contours.py")
