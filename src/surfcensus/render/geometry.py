"""Occipital pole, view box, panels, and the coordinate chain pixel <-> voxel <-> tkr RAS.

Conventions (docs/rendering.md):

* tkr RAS: x to the subject's right, y anterior, z superior, in mm.
* An integer voxel index is the centre of that voxel.
* Panel pixel coordinates are continuous, origin at the top-left corner of the tile, px to the
  right and py down. Pixel (column c, row r) is sampled at its centre (c + 0.5, r + 0.5).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float

from surfcensus.render.model import (
    PANEL_ORDER,
    BoxInfo,
    Frame,
    ModeInfo,
    PanelEdges,
    PanelInfo,
    PanelName,
    PoleInfo,
    RenderConfig,
)

# (tkr axis normal to the slices, tkr axis along the horizontal, tkr axis along the vertical)
_PANEL_AXES: dict[PanelName, tuple[int, int, int]] = {
    "sagittal": (0, 1, 2),
    "coronal": (1, 0, 2),
    "axial": (2, 0, 1),
}
# anatomical letter at the low and high end of each tkr axis
_AXIS_LETTERS: tuple[tuple[str, str], ...] = (("L", "R"), ("P", "A"), ("I", "S"))


class NotAxisAlignedError(ValueError):
    """The volume's voxel axes are not parallel to the tkr axes."""


@dataclass(frozen=True)
class AxisMap:
    """How each tkr axis runs through the voxel grid of an axis-aligned volume.

    For tkr axis ``w``: ``world = offset[w] + step[w] * voxel[voxel_axis[w]]``.
    """

    voxel_axis: tuple[int, int, int]
    step: tuple[float, float, float]
    """Signed mm per voxel."""
    offset: tuple[float, float, float]
    """tkr coordinate of voxel index 0."""
    n: tuple[int, int, int]
    """Number of voxels along that voxel axis."""

    def world_to_voxel(self, w: int, world: float | np.ndarray) -> float | np.ndarray:
        return (world - self.offset[w]) / self.step[w]

    def voxel_to_world(self, w: int, voxel: float | np.ndarray) -> float | np.ndarray:
        return self.offset[w] + self.step[w] * voxel


def axis_map(
    vox2ras_tkr: Float[np.ndarray, "4 4"], shape: tuple[int, int, int], rtol: float = 1e-4
) -> AxisMap:
    """Read the axis correspondence off a voxel-to-tkr affine, or refuse an oblique one."""
    linear = np.asarray(vox2ras_tkr, dtype=float)[:3, :3]
    if not np.isfinite(vox2ras_tkr).all():
        raise NotAxisAlignedError("the affine has non-finite entries")
    voxel_axis: list[int] = []
    step: list[float] = []
    for w in range(3):
        row = linear[w]
        a = int(np.argmax(np.abs(row)))
        rest = np.delete(row, a)
        if row[a] == 0 or np.abs(rest).max() > rtol * abs(row[a]):
            raise NotAxisAlignedError(
                f"voxel axes are not parallel to the tkr axes (row {w} of the affine: {row})"
            )
        voxel_axis.append(a)
        step.append(float(row[a]))
    if sorted(voxel_axis) != [0, 1, 2]:
        raise NotAxisAlignedError(f"affine does not map voxel axes one-to-one: {voxel_axis}")
    offset = [float(vox2ras_tkr[w, 3]) for w in range(3)]
    n = [int(shape[a]) for a in voxel_axis]
    return AxisMap(
        voxel_axis=(voxel_axis[0], voxel_axis[1], voxel_axis[2]),
        step=(step[0], step[1], step[2]),
        offset=(offset[0], offset[1], offset[2]),
        n=(n[0], n[1], n[2]),
    )


def find_poles(
    lh_pial: Float[np.ndarray, "n 3"], rh_pial: Float[np.ndarray, "m 3"]
) -> dict[str, PoleInfo]:
    """Occipital pole per hemisphere: the pial vertex with the most posterior y (decision D12)."""
    poles: dict[str, PoleInfo] = {}
    for hemi, xyz in (("lh", lh_pial), ("rh", rh_pial)):
        if xyz.shape[0] == 0:
            raise ValueError(f"{hemi} pial surface has no vertices")
        v = int(np.argmin(xyz[:, 1]))
        poles[hemi] = PoleInfo(vertex=v, tkr=_vec3(xyz[v]))
    return poles


def compute_box(
    lh_pial: Float[np.ndarray, "n 3"],
    rh_pial: Float[np.ndarray, "m 3"],
    poles: dict[str, PoleInfo],
    config: RenderConfig,
) -> BoxInfo:
    """View box in tkr RAS, centred on the midline at the mean of the two poles."""
    lh_pole = np.array(poles["lh"].tkr)
    rh_pole = np.array(poles["rh"].tkr)
    cy = float((lh_pole[1] + rh_pole[1]) / 2)
    cz = float((lh_pole[2] + rh_pole[2]) / 2)
    offsets = config.box_offsets()

    def in_range(xyz: np.ndarray) -> np.ndarray:
        return (
            (xyz[:, 1] >= cy + offsets[0, 1])
            & (xyz[:, 1] <= cy + offsets[1, 1])
            & (xyz[:, 2] >= cz + offsets[0, 2])
            & (xyz[:, 2] <= cz + offsets[1, 2])
        )

    lh_in, rh_in = in_range(lh_pial), in_range(rh_pial)
    if lh_in.any() and rh_in.any():
        # medial-most vertices: largest x on the left, smallest x on the right
        cx = float((lh_pial[lh_in, 0].max() + rh_pial[rh_in, 0].min()) / 2)
    else:
        cx = float((lh_pole[0] + rh_pole[0]) / 2)
    centre = np.array([cx, cy, cz])
    return BoxInfo(
        centre=_vec3(centre), low=_vec3(centre + offsets[0]), high=_vec3(centre + offsets[1])
    )


def slice_indices(
    amap: AxisMap, w: int, low: float, high: float, step_mm: float
) -> tuple[list[int], list[float]]:
    """Voxel indices of the slices between two tkr coordinates, low to high, and their planes.

    Positions run from ``low`` in steps of ``step_mm`` and are snapped to the nearest voxel
    centre. Duplicates and positions outside the volume are dropped.
    """
    count = int(np.floor((high - low) / step_mm + 1e-9)) + 1
    positions = low + step_mm * np.arange(count)
    voxel = np.floor(np.asarray(amap.world_to_voxel(w, positions)) + 0.5).astype(int)
    indices: list[int] = []
    for v in voxel.tolist():
        if 0 <= v < amap.n[w] and (not indices or v != indices[-1]):
            indices.append(v)
    planes = [float(amap.voxel_to_world(w, float(v))) for v in indices]
    return indices, planes


def field_of_view(
    box: BoxInfo, config: RenderConfig
) -> tuple[Float[np.ndarray, " 3"], Float[np.ndarray, " 3"]]:
    """Low and high corner, in tkr RAS, of the cube that the panels show.

    Every panel is a square that spans the cube along its two in-plane axes. The side is the
    largest box extent plus the margin on both sides, centred on the midpoint of the box.
    """
    low, high = np.array(box.low), np.array(box.high)
    side = float((high - low).max() + 2 * config.margin_mm)
    if not side > 0:
        raise ValueError(f"the view box and its margin leave the panels no extent ({side:g} mm)")
    mid = (low + high) / 2
    return mid - side / 2, mid + side / 2


def clickable_region(
    box: BoxInfo, panels: Mapping[PanelName, PanelInfo], config: RenderConfig
) -> Float[np.ndarray, "3 2 3"]:
    """What the tiles show, as one box per panel in tkr RAS (docs/rendering.md).

    ``region[p]`` holds the low and the high corner of the box of panel ``p`` of
    ``PANEL_ORDER``: along the slice axis from half a slice step before the panel's first slice
    plane to half a step past its last, along the two in-plane axes the field of view. The
    region is the union of the three boxes. Every point that a click on a tile can name lies in
    it, on a slice plane. The half step makes each slice stand for one step of the sweep.
    """
    fov_low, fov_high = field_of_view(box, config)
    half_step = config.slice_step_mm / 2
    region = np.empty((len(PANEL_ORDER), 2, 3))
    for p, name in enumerate(PANEL_ORDER):
        panel = panels[name]
        region[p] = fov_low, fov_high
        region[p, 0, panel.world_axis] = panel.slice_mm[0] - half_step
        region[p, 1, panel.world_axis] = panel.slice_mm[-1] + half_step
    return region


def in_region(
    xyz: Float[np.ndarray, "n 3"], region: Float[np.ndarray, "boxes 2 3"], margin_mm: float = 0.0
) -> Bool[np.ndarray, " n"]:
    """Which points lie in a union of boxes, each grown by ``margin_mm`` on every side.

    The boundary counts as inside.
    """
    inside = np.zeros(len(xyz), dtype=bool)
    for low, high in region:
        inside |= ((xyz >= low - margin_mm) & (xyz <= high + margin_mm)).all(axis=1)
    return inside


def build_panels(box: BoxInfo, amap: AxisMap, config: RenderConfig) -> dict[PanelName, PanelInfo]:
    """The three panels: slice lists, tile size, edge letters and pixel <-> voxel affines."""
    low, high, centre = np.array(box.low), np.array(box.high), np.array(box.centre)
    fov_low, fov_high = field_of_view(box, config)
    size = config.panel_px

    panels: dict[PanelName, PanelInfo] = {}
    for name in PANEL_ORDER:
        w, h, v = _PANEL_AXES[name]
        slices, planes = slice_indices(amap, w, low[w], high[w], config.slice_step_mm)
        if not slices:
            raise ValueError(f"the view box lies outside the volume along the {name} axis")
        # tkr coordinate at the left / right edge and at the top / bottom edge of the tile
        if h == 1:
            high_at_left = config.sagittal_anterior_left
        else:
            high_at_left = config.radiological
        h0, h1 = (fov_high[h], fov_low[h]) if high_at_left else (fov_low[h], fov_high[h])
        v0, v1 = fov_high[v], fov_low[v]  # superior (or anterior, in the axial panel) at the top
        # voxel = scale * pixel + shift along the horizontal (a, c) and the vertical (b, d)
        a, c = (h1 - h0) / (size * amap.step[h]), (h0 - amap.offset[h]) / amap.step[h]
        b, d = (v1 - v0) / (size * amap.step[v]), (v0 - amap.offset[v]) / amap.step[v]
        pix2vox = np.array([[a, 0.0, c], [0.0, b, d], [0.0, 0.0, 1.0]])
        # The inverse written out, each entry one correctly rounded division. np.linalg.inv
        # would give the last bit of the shifts, and the sign of the zeros, of the LAPACK
        # that numpy is linked with, and both go into render.json.
        vox2pix = np.array([[1.0 / a, 0.0, -c / a], [0.0, 1.0 / b, -d / b], [0.0, 0.0, 1.0]])
        letters_h, letters_v = _AXIS_LETTERS[h], _AXIS_LETTERS[v]
        edges = PanelEdges(
            left=letters_h[1] if high_at_left else letters_h[0],
            right=letters_h[0] if high_at_left else letters_h[1],
            top=letters_v[1],
            bottom=letters_v[0],
        )
        centre_slice = int(np.argmin(np.abs(np.array(planes) - centre[w])))
        panels[name] = PanelInfo(
            name=name,
            world_axis=w,
            slice_voxel_axis=amap.voxel_axis[w],
            h_voxel_axis=amap.voxel_axis[h],
            v_voxel_axis=amap.voxel_axis[v],
            slices=tuple(slices),
            slice_mm=tuple(planes),
            centre_slice=centre_slice,
            width=size,
            height=size,
            pix2vox=_matrix3(pix2vox),
            vox2pix=_matrix3(vox2pix),
            edges=edges,
        )
    return panels


def _ping_pong(n: int) -> list[int]:
    """0 .. n-1 and back, without repeating either end: one loop of length max(1, 2n - 2)."""
    forward = list(range(n))
    return forward + forward[-2:0:-1]


def frame_tables(panels: dict[PanelName, PanelInfo]) -> dict[str, ModeInfo]:
    """Frame tables of the synchronized and the one-at-a-time mode (docs/rendering.md)."""
    counts = [len(panels[name].slices) for name in PANEL_ORDER]
    longest = max(counts)

    def locked(f: int, n: int) -> int:
        return 0 if longest == 1 else round(f * (n - 1) / (longest - 1))

    sync: list[Frame] = [
        (locked(f, counts[0]), locked(f, counts[1]), locked(f, counts[2]))
        for f in _ping_pong(longest)
    ]
    hold = [panels[name].centre_slice for name in PANEL_ORDER]
    seq: list[Frame] = []
    for p, n in enumerate(counts):
        for s in _ping_pong(n):
            frame = list(hold)
            frame[p] = s
            seq.append((frame[0], frame[1], frame[2]))
    return {"sync": ModeInfo(frames=tuple(sync)), "seq": ModeInfo(frames=tuple(seq))}


def pixel_to_voxel(
    panel: PanelInfo, slice_pos: int, px: float | np.ndarray, py: float | np.ndarray
) -> Float[np.ndarray, "... 3"]:
    """Continuous voxel coordinates (i, j, k) of points on one tile of a panel."""
    px_a, py_a = np.broadcast_arrays(np.asarray(px, dtype=float), np.asarray(py, dtype=float))
    m = panel.pix2vox_array()
    out = np.empty((*px_a.shape, 3), dtype=float)
    out[..., panel.slice_voxel_axis] = panel.slices[slice_pos]
    out[..., panel.h_voxel_axis] = m[0, 0] * px_a + m[0, 2]
    out[..., panel.v_voxel_axis] = m[1, 1] * py_a + m[1, 2]
    return out


def voxel_to_pixel(
    panel: PanelInfo, ijk: Float[np.ndarray, "... 3"]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Panel pixel coordinates of voxel coordinates, and their coordinate along the slice axis.

    Returns ``(px, py, slice_coordinate)``. ``slice_coordinate`` is a voxel coordinate along
    ``panel.slice_voxel_axis``; use :func:`nearest_slice` to find the tile that shows it.
    """
    ijk = np.asarray(ijk, dtype=float)
    m = panel.vox2pix_array()
    px = m[0, 0] * ijk[..., panel.h_voxel_axis] + m[0, 2]
    py = m[1, 1] * ijk[..., panel.v_voxel_axis] + m[1, 2]
    # np.asarray: a single point (shape (3,)) gives 0-d arrays, not numpy scalars
    return np.asarray(px), np.asarray(py), np.asarray(ijk[..., panel.slice_voxel_axis])


def nearest_slice(panel: PanelInfo, slice_coordinate: float) -> int:
    """Position in ``panel.slices`` of the tile nearest a voxel coordinate along the slice axis."""
    return int(np.argmin(np.abs(np.array(panel.slices, dtype=float) - slice_coordinate)))


def apply_affine(
    affine: Float[np.ndarray, "4 4"], xyz: Float[np.ndarray, "... 3"]
) -> Float[np.ndarray, "... 3"]:
    """Apply a 4 x 4 affine to points, for example voxel -> tkr RAS with vox2ras_tkr."""
    affine = np.asarray(affine, dtype=float)
    return np.asarray(xyz, dtype=float) @ affine[:3, :3].T + affine[:3, 3]


def tkr_to_pixel(
    panel: PanelInfo, vox2ras_tkr: Float[np.ndarray, "4 4"], xyz: Float[np.ndarray, "... 3"]
) -> tuple[np.ndarray, np.ndarray]:
    """Panel pixel coordinates of tkr RAS points (their position along the slice axis ignored)."""
    ijk = apply_affine(np.linalg.inv(np.asarray(vox2ras_tkr, dtype=float)), xyz)
    px, py, _ = voxel_to_pixel(panel, ijk)
    return px, py


def crosshair_pixels(
    panels: dict[PanelName, PanelInfo], frame: Frame
) -> dict[PanelName, tuple[float, float]]:
    """Where the other two planes cross each panel in a frame: (px of the vertical line, py of
    the horizontal line)."""
    # voxel coordinate of the current slice along each voxel axis
    current = np.zeros(3)
    for name, pos in zip(PANEL_ORDER, frame, strict=True):
        panel = panels[name]
        current[panel.slice_voxel_axis] = panel.slices[pos]
    out: dict[PanelName, tuple[float, float]] = {}
    for name in PANEL_ORDER:
        px, py, _ = voxel_to_pixel(panels[name], current)
        out[name] = (float(px), float(py))
    return out


def _vec3(a: np.ndarray) -> tuple[float, float, float]:
    return (float(a[0]), float(a[1]), float(a[2]))


def _matrix3(
    m: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    # + 0.0 turns a negative zero into zero, so that none is written to render.json
    rows = [(float(r[0]) + 0.0, float(r[1]) + 0.0, float(r[2]) + 0.0) for r in m]
    return (rows[0], rows[1], rows[2])
