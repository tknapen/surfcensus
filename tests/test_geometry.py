"""Independent check of render/geometry.py and render/model.py against docs/rendering.md.

Expected values come from the specification: worked out by hand, or by a brute-force
computation written here, never from the implementation. Conventions: tkr RAS in mm (x right,
y anterior, z superior), integer voxel index = voxel centre, pixel origin at the top-left
corner of the tile with px right and py down.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from functools import cached_property

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from surfcensus.render import geometry as g
from surfcensus.render.model import (
    PANEL_ORDER,
    BoxInfo,
    InputFile,
    PanelInfo,
    RenderConfig,
    RenderInfo,
)

# (tkr axis normal to the slices, along the horizontal, along the vertical): "Display orientation"
AXES = {"sagittal": (0, 1, 2), "coronal": (1, 0, 2), "axial": (2, 0, 1)}
LETTERS = (("L", "R"), ("P", "A"), ("I", "S"))  # low and high end of tkr x, y, z


def affine(rows: list[list[float]]) -> np.ndarray:
    return np.array([*rows, [0, 0, 0, 1]], dtype=float)


LIA_1MM = affine([[-1, 0, 0, 128], [0, 0, 1, -128], [0, -1, 0, 128]])  # FreeSurfer conformed
LIA_HIRES = affine([[-0.7, 0, 0, 112], [0, 0, 0.7, -112], [0, -0.7, 0, 112]])  # 320^3
LIA_2MM = affine([[-2, 0, 0, 128], [0, 0, 2, -128], [0, -2, 0, 128]])  # 128^3
RAS_ANISO = affine([[1, 0, 0, -50], [0, 2, 0, -120], [0, 0, 3, -210]])  # (100, 120, 140)
PIR_ANISO = affine([[0, 0, 0.8, -60], [-1.5, 0, 0, 90], [0, -2.5, 0, 100]])  # (120, 80, 150)

CENTRE = (0.3, -95.2, -10.6)  # deliberately off the voxel centres, and never half-way
PIR_CENTRE = (0.3, -40.1, -20.3)


class Case:
    """A volume, a box built directly from the docs (centre plus offsets), and a config."""

    def __init__(
        self,
        vox2ras_tkr: np.ndarray,
        shape: tuple[int, int, int],
        centre: tuple[float, float, float] = CENTRE,
        config: RenderConfig | None = None,
    ) -> None:
        self.affine, self.shape, self.centre = vox2ras_tkr, shape, centre
        self.config = config or RenderConfig()
        c = self.config
        low = [centre[0] + c.box_x_mm[0], centre[1] + c.box_y_mm[0], centre[2] + c.box_z_mm[0]]
        high = [centre[0] + c.box_x_mm[1], centre[1] + c.box_y_mm[1], centre[2] + c.box_z_mm[1]]
        self.box = BoxInfo.model_validate({"centre": centre, "low": low, "high": high})

    @cached_property
    def amap(self) -> g.AxisMap:
        return g.axis_map(self.affine, self.shape)

    @cached_property
    def panels(self) -> dict[str, PanelInfo]:
        return dict(g.build_panels(self.box, self.amap, self.config))

    def to_tkr(self, ijk: np.ndarray) -> np.ndarray:
        return ijk @ self.affine[:3, :3].T + self.affine[:3, 3]

    def fov_edges(self, name: str) -> tuple[float, float, float, float]:
        """tkr coordinate at the left, right, top and bottom edge of a panel, from the docs:
        square, side = largest box extent + 2 margins, centred on the box midpoint."""
        low, high = np.array(self.box.low), np.array(self.box.high)
        half = ((high - low).max() + 2 * self.config.margin_mm) / 2
        mid = (low + high) / 2
        _, h, v = AXES[name]
        flag = (
            self.config.sagittal_anterior_left if name == "sagittal" else self.config.radiological
        )
        left, right = (mid[h] + half, mid[h] - half) if flag else (mid[h] - half, mid[h] + half)
        return float(left), float(right), float(mid[v] + half), float(mid[v] - half)

    def expected_pixel(self, name: str, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        left, right, top, bottom = self.fov_edges(name)
        _, h, v = AXES[name]
        size = self.config.panel_px
        px = (xyz[..., h] - left) / (right - left) * size
        py = (xyz[..., v] - top) / (bottom - top) * size
        return px, py

    def voxel_centres(self, w: int) -> np.ndarray:
        """tkr coordinate along axis w of every voxel centre along the matching voxel axis."""
        a = int(np.argmax(np.abs(self.affine[w, :3])))
        return self.affine[w, 3] + self.affine[w, a] * np.arange(self.shape[a])


NON_DEFAULT = RenderConfig(
    box_x_mm=(-20, 25),
    box_y_mm=(-5, 30),
    box_z_mm=(-10, 40),
    margin_mm=4,
    panel_px=200,
    slice_step_mm=2,
)
CASES = {
    "conformed": Case(LIA_1MM, (256, 256, 256)),
    "hires": Case(LIA_HIRES, (320, 320, 320)),
    "coarse": Case(LIA_2MM, (128, 128, 128)),
    "ras": Case(RAS_ANISO, (100, 120, 140), (1.1, -60.2, 5.3)),
    "pir": Case(PIR_ANISO, (120, 80, 150), PIR_CENTRE),
    "neurological": Case(LIA_1MM, (256, 256, 256), config=RenderConfig(radiological=False)),
    "anterior_left": Case(
        LIA_1MM, (256, 256, 256), config=RenderConfig(sagittal_anterior_left=True)
    ),
    "both_flipped": Case(
        PIR_ANISO,
        (120, 80, 150),
        PIR_CENTRE,
        RenderConfig(radiological=False, sagittal_anterior_left=True),
    ),
    "non_default": Case(LIA_1MM, (256, 256, 256), config=NON_DEFAULT),
}
all_cases = pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())


def test_the_cases_cover_both_values_of_both_orientation_switches() -> None:
    """The defaults (decision D13) are radiological and posterior at the left of the sagittal
    panel; the tests over all cases see each switch both ways, alone and together."""
    default = RenderConfig()
    assert (default.radiological, default.sagittal_anterior_left) == (True, False)
    seen = {(c.config.radiological, c.config.sagittal_anterior_left) for c in CASES.values()}
    assert seen == {(True, False), (False, False), (True, True), (False, True)}


def brute_slices(
    centres: np.ndarray, voxel_mm: float, low: float, high: float, step: float
) -> tuple[list[int], np.ndarray, bool]:
    """The 'Slices' paragraph, literally: positions from low to high in steps, each snapped to
    the nearest voxel centre, positions outside the volume and duplicates dropped.

    Returns the voxel indices, the positions, and whether any position is so close to half-way
    between two centres, or to the volume edge, that the docs do not decide the answer.
    """
    count = int(np.floor((high - low) / step + 1e-6)) + 1
    positions = low + step * np.arange(count)
    dist = np.abs(positions[:, None] - centres[None, :])
    two = np.sort(dist, axis=1)[:, :2]
    undecided = bool(
        (np.abs(two[:, 1] - two[:, 0]) < 1e-6).any()
        or (np.abs(two[:, 0] - voxel_mm / 2) < 1e-6).any()
    )
    indices: list[int] = []
    for i in dist.argmin(axis=1)[two[:, 0] <= voxel_mm / 2].tolist():
        if i not in indices:
            indices.append(i)
    return indices, positions, undecided


# ---------------------------------------------------------------- axis_map


def test_axis_map_conformed_lia() -> None:
    amap = g.axis_map(LIA_1MM, (256, 256, 256))
    assert amap.voxel_axis == (0, 2, 1)
    assert amap.step == (-1.0, 1.0, -1.0)
    assert amap.offset == (128.0, -128.0, 128.0)
    assert amap.n == (256, 256, 256)


def test_axis_map_hires_lia() -> None:
    amap = g.axis_map(LIA_HIRES, (320, 320, 320))
    assert amap.voxel_axis == (0, 2, 1)
    assert amap.step == (-0.7, 0.7, -0.7)
    assert amap.offset == (112.0, -112.0, 112.0)
    assert amap.n == (320, 320, 320)


def test_axis_map_signed_permutations_with_anisotropic_voxels() -> None:
    ras = g.axis_map(RAS_ANISO, (100, 120, 140))
    assert (ras.voxel_axis, ras.step) == ((0, 1, 2), (1.0, 2.0, 3.0))
    assert (ras.offset, ras.n) == ((-50.0, -120.0, -210.0), (100, 120, 140))
    # voxel axis 0 runs posterior, 1 inferior, 2 right
    pir = g.axis_map(PIR_ANISO, (120, 80, 150))
    assert (pir.voxel_axis, pir.step) == ((2, 0, 1), (0.8, -1.5, -2.5))
    assert (pir.offset, pir.n) == ((-60.0, 90.0, 100.0), (150, 120, 80))


@all_cases
def test_axis_map_agrees_with_the_affine(case: Case) -> None:
    ijk = np.random.default_rng(0).uniform(-5, 300, size=(50, 3))
    xyz = case.to_tkr(ijk)
    for w in range(3):
        a = case.amap.voxel_axis[w]
        np.testing.assert_allclose(case.amap.voxel_to_world(w, ijk[:, a]), xyz[:, w], atol=1e-9)
        np.testing.assert_allclose(case.amap.world_to_voxel(w, xyz[:, w]), ijk[:, a], atol=1e-9)


def rotated(vox2ras_tkr: np.ndarray, angle: float, axis: int) -> np.ndarray:
    i, j = (a for a in range(3) if a != axis)
    rot = np.eye(4)
    rot[i, i] = rot[j, j] = np.cos(angle)
    rot[i, j], rot[j, i] = -np.sin(angle), np.sin(angle)
    return rot @ vox2ras_tkr


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("base", [LIA_1MM, LIA_HIRES], ids=["conformed", "hires"])
def test_axis_map_rotation_tolerance(base: np.ndarray, axis: int) -> None:
    """Tolerance 1e-4 relative: 1e-3 rad is refused, 1e-5 rad is accepted."""
    with pytest.raises(g.NotAxisAlignedError):
        g.axis_map(rotated(base, 1e-3, axis), (256, 256, 256))
    with pytest.raises(g.NotAxisAlignedError):
        g.axis_map(rotated(base, -1e-3, axis), (256, 256, 256))
    assert g.axis_map(rotated(base, 1e-5, axis), (256, 256, 256)).voxel_axis == (0, 2, 1)


def test_not_axis_aligned_error_is_a_value_error() -> None:
    assert issubclass(g.NotAxisAlignedError, ValueError)


@pytest.mark.parametrize(
    "linear",
    [
        np.zeros((3, 3)),
        [[1, 0, 0], [0, 0, 0], [0, 0, 1]],  # a tkr axis no voxel axis reaches
        [[1, 0, 0], [1, 0, 0], [0, 0, 1]],  # two tkr axes on one voxel axis
        [[1, 0, 0], [0.5, 1, 0], [0, 0, 1]],  # shear
        [[1, 1, 0], [-1, 1, 0], [0, 0, 1]],  # 45 degrees
    ],
    ids=["zero", "zero_row", "rank_two", "shear", "oblique"],
)
def test_axis_map_refuses_degenerate_and_oblique_affines(linear: list[list[float]]) -> None:
    bad = np.eye(4)
    bad[:3, :3] = linear
    with pytest.raises(g.NotAxisAlignedError):
        g.axis_map(bad, (10, 10, 10))


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_axis_map_refuses_non_finite_affines(value: float) -> None:
    bad = LIA_1MM.copy()
    bad[0, 0] = value
    with pytest.raises(g.NotAxisAlignedError):
        g.axis_map(bad, (256, 256, 256))


# ---------------------------------------------------------------- poles and box

LH = np.array(
    [
        [-25.0, -80.0, 10.0],
        [-14.0, -101.0, -3.0],  # most posterior: the lh pole
        [-3.0, -90.0, 5.0],  # medial-most inside the y and z ranges
        [-1.0, 20.0, 0.0],  # more medial, but far anterior of the box
        [-0.5, -90.0, 40.0],  # more medial, but above the box
    ]
)
RH = np.array(
    [
        [4.0, -85.0, 0.0],  # medial-most inside the y and z ranges
        [0.5, -80.0, -60.0],  # more medial, but below the box
        [16.0, -97.0, -1.0],  # most posterior: the rh pole
        [22.0, -70.0, 12.0],
    ]
)


def test_find_poles_is_the_most_posterior_vertex_per_hemisphere() -> None:
    poles = g.find_poles(LH, RH)
    assert set(poles) == {"lh", "rh"}
    assert (poles["lh"].vertex, poles["lh"].tkr) == (1, (-14.0, -101.0, -3.0))
    assert (poles["rh"].vertex, poles["rh"].tkr) == (2, (16.0, -97.0, -1.0))


def test_find_poles_accepts_float32_and_refuses_an_empty_surface() -> None:
    poles = g.find_poles(LH.astype(np.float32), RH.astype(np.float32))
    assert (poles["lh"].vertex, poles["rh"].vertex) == (1, 2)
    assert poles["rh"].tkr == (16.0, -97.0, -1.0)
    with pytest.raises(ValueError, match="rh"):
        g.find_poles(LH, np.empty((0, 3)))


def test_compute_box_default() -> None:
    """Centre y, z: mean of the poles (-99, -2), so the ranges are y in [-107, -54] and
    z in [-37, 23]. Medial-most inside them: x = -3 (lh) and 4 (rh), midpoint 0.5.
    Including the three out-of-range vertices would give (-0.5 + 0.5) / 2 = 0."""
    box = g.compute_box(LH, RH, g.find_poles(LH, RH), RenderConfig())
    assert box.centre == pytest.approx((0.5, -99.0, -2.0))
    assert box.low == pytest.approx((-29.5, -107.0, -37.0))
    assert box.high == pytest.approx((30.5, -54.0, 23.0))


def test_compute_box_non_default_config_changes_ranges_and_corners() -> None:
    """With y in [-104, -91] and z in [-12, 38] only the poles themselves are in range, so the
    centre x is (-14 + 16) / 2 = 1."""
    config = RenderConfig(box_x_mm=(-20, 25), box_y_mm=(-5, 8), box_z_mm=(-10, 40))
    box = g.compute_box(LH, RH, g.find_poles(LH, RH), config)
    assert box.centre == pytest.approx((1.0, -99.0, -2.0))
    assert box.low == pytest.approx((-19.0, -104.0, -12.0))
    assert box.high == pytest.approx((26.0, -91.0, 38.0))


def test_compute_box_falls_back_to_the_poles_when_a_hemisphere_is_out_of_range() -> None:
    """Centre z is 0, so the lh vertices (z >= 40) are all above the range [-35, 25]. The
    centre x is then the mean pole x, (-10 + 12) / 2 = 1, not the midpoint (-4 + 2) / 2 = -1
    of the medial-most vertices."""
    lh = np.array([[-10.0, -100.0, 40.0], [-4.0, -95.0, 45.0]])
    rh = np.array([[12.0, -98.0, -40.0], [2.0, -90.0, -10.0]])
    box = g.compute_box(lh, rh, g.find_poles(lh, rh), RenderConfig())
    assert box.centre == pytest.approx((1.0, -99.0, 0.0))
    assert box.low == pytest.approx((-29.0, -107.0, -35.0))
    assert box.high == pytest.approx((31.0, -54.0, 25.0))


def test_box_offsets_rows_are_low_and_high() -> None:
    np.testing.assert_array_equal(RenderConfig().box_offsets(), [[-30, -8, -35], [30, 45, 25]])


# ---------------------------------------------------------------- slices


@pytest.mark.parametrize("centre", [CENTRE, (0.0, -95.0, -10.0), (0.5, -95.5, -10.5)])
def test_default_slice_counts(centre: tuple[float, float, float]) -> None:
    """61 sagittal, 54 coronal and 61 axial slices, also when every position is half-way
    between two voxel centres."""
    panels = Case(LIA_1MM, (256, 256, 256), centre).panels
    assert [len(panels[name].slices) for name in PANEL_ORDER] == [61, 54, 61]


def test_slices_snap_to_voxel_centres_conformed() -> None:
    """Box centre (0.3, -95.2, -10.6): positions -29.7 .. 30.3 snap to x = -30 .. 30, voxel
    i = 128 - x; y = -103 .. -50, voxel k = y + 128; z = -46 .. 14, voxel j = 128 - z."""
    panels = CASES["conformed"].panels
    assert panels["sagittal"].slices == tuple(range(158, 97, -1))
    assert panels["sagittal"].slice_mm == tuple(float(x) for x in range(-30, 31))
    assert panels["coronal"].slices == tuple(range(25, 79))
    assert panels["coronal"].slice_mm == tuple(float(y) for y in range(-103, -49))
    assert panels["axial"].slices == tuple(range(174, 113, -1))
    assert panels["axial"].slice_mm == tuple(float(z) for z in range(-46, 15))


def test_slices_deduplicate_when_voxels_are_larger_than_the_step() -> None:
    """2 mm voxels with centres on even mm, 1 mm step: x = -30 .. 30, y = -104 .. -50 and
    z = -46 .. 14 in steps of 2; i = (128 - x) / 2, k = (y + 128) / 2, j = (128 - z) / 2."""
    panels = CASES["coarse"].panels
    assert panels["sagittal"].slices == tuple(range(79, 48, -1))
    assert panels["sagittal"].slice_mm == tuple(float(x) for x in range(-30, 31, 2))
    assert panels["coronal"].slices == tuple(range(12, 40))
    assert panels["coronal"].slice_mm == tuple(float(y) for y in range(-104, -49, 2))
    assert panels["axial"].slices == tuple(range(87, 56, -1))
    assert panels["axial"].slice_mm == tuple(float(z) for z in range(-46, 15, 2))
    assert [panels[n].centre_slice for n in PANEL_ORDER] == [15, 4, 18]  # x 0, y -96, z -10


def test_hires_planes_are_within_half_a_voxel_of_the_requested_positions() -> None:
    case = CASES["hires"]
    for name, count in zip(PANEL_ORDER, (61, 54, 61), strict=True):
        w = AXES[name][0]
        requested = case.box.low[w] + np.arange(count)
        planes = np.array(case.panels[name].slice_mm)
        assert planes.shape == (count,)  # 1 mm steps over 0.7 mm voxels: no duplicates
        assert np.abs(planes - requested).max() <= 0.35 + 1e-9


@all_cases
def test_slices_match_brute_force_and_ascend(case: Case) -> None:
    for name, panel in case.panels.items():
        w = AXES[name][0]
        centres = case.voxel_centres(w)
        voxel_mm = float(abs(centres[1] - centres[0]))
        expected, _, undecided = brute_slices(
            centres, voxel_mm, case.box.low[w], case.box.high[w], case.config.slice_step_mm
        )
        assert not undecided
        assert list(panel.slices) == expected
        np.testing.assert_allclose(panel.slice_mm, centres[expected], atol=1e-9)
        assert np.all(np.diff(panel.slice_mm) > 0)
        assert len(set(panel.slices)) == len(panel.slices)


@settings(max_examples=200, deadline=None)
@given(
    centre=st.floats(-40, 40),
    shift=st.floats(-0.5, 0.5),
    voxel_mm=st.sampled_from([0.5, 0.7, 0.8, 1.0, 1.25, 2.0, 3.0]),
    step=st.sampled_from([0.5, 1.0, 1.5, 2.0]),
    sign=st.sampled_from([-1.0, 1.0]),
)
def test_slice_indices_property(
    centre: float, shift: float, voxel_mm: float, step: float, sign: float
) -> None:
    """Against the brute-force reading of the docs, for volumes that may end inside the box."""
    n = 200
    offset = shift - sign * voxel_mm * n / 2
    centres = offset + sign * voxel_mm * np.arange(n)
    low, high = centre - 30, centre + 30
    expected, _, undecided = brute_slices(centres, voxel_mm, low, high, step)
    assume(not undecided)
    amap = g.AxisMap(
        voxel_axis=(0, 1, 2),
        step=(sign * voxel_mm, 1.0, 1.0),
        offset=(offset, 0.0, 0.0),
        n=(n, 9, 9),
    )
    indices, planes = g.slice_indices(amap, 0, low, high, step)
    assert indices == expected
    np.testing.assert_allclose(planes, centres[expected], atol=1e-9)
    assert np.all(np.diff(planes) > 0)


def test_slices_outside_the_volume_are_dropped() -> None:
    """Conformed voxel centres run from y = -128 to 127. A box centred on y = -125.2 asks for
    -133.2 .. -80.2; the positions below -128.5 are outside, leaving y = -128 .. -80."""
    panels = Case(LIA_1MM, (256, 256, 256), (0.3, -125.2, -10.6)).panels
    assert panels["coronal"].slices == tuple(range(0, 49))
    assert panels["coronal"].slice_mm == tuple(float(y) for y in range(-128, -79))
    assert panels["coronal"].centre_slice == 3  # y = -125
    assert len(panels["sagittal"].slices) == len(panels["axial"].slices) == 61


def test_box_outside_the_volume_is_an_error() -> None:
    case = Case(LIA_1MM, (256, 256, 256), (0.3, -300.0, -10.6))
    with pytest.raises(ValueError, match="outside the volume"):
        g.build_panels(case.box, case.amap, case.config)


# ---------------------------------------------------------------- panels


def test_panel_axes_conformed_and_pir() -> None:
    def axes(case: Case) -> dict[str, tuple[int, int, int, int]]:
        return {
            name: (p.world_axis, p.slice_voxel_axis, p.h_voxel_axis, p.v_voxel_axis)
            for name, p in case.panels.items()
        }

    assert tuple(CASES["conformed"].panels) == ("sagittal", "coronal", "axial") == PANEL_ORDER
    # LIA: x along voxel axis 0, y along 2, z along 1
    assert axes(CASES["conformed"]) == {
        "sagittal": (0, 0, 2, 1),
        "coronal": (1, 2, 0, 1),
        "axial": (2, 1, 0, 2),
    }
    # PIR: x along voxel axis 2, y along 0, z along 1
    assert axes(CASES["pir"]) == {
        "sagittal": (0, 2, 0, 1),
        "coronal": (1, 0, 2, 1),
        "axial": (2, 1, 2, 0),
    }
    for case in CASES.values():
        assert all(p.name == name for name, p in case.panels.items())


@pytest.mark.parametrize(
    ("case", "edges"),
    [
        # default: side 60 + 20 = 80 mm around the box midpoint (0.3, -76.7, -15.6)
        (
            "conformed",
            {
                "sagittal": (-116.7, -36.7, 24.4, -55.6),
                "coronal": (40.3, -39.7, 24.4, -55.6),
                "axial": (40.3, -39.7, -36.7, -116.7),
            },
        ),
        # box extents 45, 35, 50 and margin 4: side 58 mm around the midpoint (2.8, -82.7, 4.4)
        (
            "non_default",
            {
                "sagittal": (-111.7, -53.7, 33.4, -24.6),
                "coronal": (31.8, -26.2, 33.4, -24.6),
                "axial": (31.8, -26.2, -53.7, -111.7),
            },
        ),
    ],
)
def test_field_of_view_by_hand(case: str, edges: dict[str, tuple[float, ...]]) -> None:
    """tkr coordinate at the left, right, top and bottom edge of each tile."""
    c = CASES[case]
    size = c.config.panel_px
    for name, (left, right, top, bottom) in edges.items():
        panel = c.panels[name]
        _, h, v = AXES[name]
        assert (panel.width, panel.height) == (size, size)
        xyz = c.to_tkr(g.pixel_to_voxel(panel, 0, np.array([0.0, size]), np.array([0.0, size])))
        np.testing.assert_allclose(xyz[:, h], [left, right], atol=1e-9)
        np.testing.assert_allclose(xyz[:, v], [top, bottom], atol=1e-9)
    if case == "conformed":  # 4 px per mm
        one_px = c.to_tkr(g.pixel_to_voxel(c.panels["axial"], 0, np.array([0.0, 1.0]), 0.0))
        assert abs(one_px[1, 0] - one_px[0, 0]) == pytest.approx(0.25)


@all_cases
def test_tile_corners_span_the_square_field_of_view(case: Case) -> None:
    size = float(case.config.panel_px)
    low, high = np.array(case.box.low), np.array(case.box.high)
    side = (high - low).max() + 2 * case.config.margin_mm
    for name, panel in case.panels.items():
        w, h, v = AXES[name]
        left, right, top, bottom = case.fov_edges(name)
        assert (panel.width, panel.height) == (case.config.panel_px, case.config.panel_px)
        for pos in (0, len(panel.slices) - 1):
            ijk = g.pixel_to_voxel(
                panel, pos, np.array([0.0, size, 0.0, size]), np.array([0.0, 0.0, size, size])
            )
            xyz = case.to_tkr(ijk)
            np.testing.assert_allclose(xyz[:, h], [left, right, left, right], atol=1e-9)
            np.testing.assert_allclose(xyz[:, v], [top, top, bottom, bottom], atol=1e-9)
            np.testing.assert_allclose(xyz[:, w], panel.slice_mm[pos], atol=1e-9)
            # square, and centred on the box midpoint (not the box centre) in both directions
            assert abs(xyz[1, h] - xyz[0, h]) == pytest.approx(side)
            assert abs(xyz[2, v] - xyz[0, v]) == pytest.approx(side)
            assert xyz[:, h].mean() == pytest.approx((low[h] + high[h]) / 2)
            assert xyz[:, v].mean() == pytest.approx((low[v] + high[v]) / 2)


@pytest.mark.parametrize("radiological", [True, False])
@pytest.mark.parametrize("anterior_left", [True, False])
@pytest.mark.parametrize("vox2ras_tkr", [LIA_1MM, PIR_ANISO], ids=["lia", "pir"])
def test_edge_letters_agree_with_the_direction_of_tkr(
    vox2ras_tkr: np.ndarray, anterior_left: bool, radiological: bool
) -> None:
    shape, centre = (
        ((256, 256, 256), CENTRE) if vox2ras_tkr is LIA_1MM else ((120, 80, 150), PIR_CENTRE)
    )
    config = RenderConfig(radiological=radiological, sagittal_anterior_left=anterior_left)
    case = Case(vox2ras_tkr, shape, centre, config)
    sag, cor, axi = (case.panels[name].edges for name in PANEL_ORDER)
    # the documented orientation
    assert (sag.left, sag.right) == (("A", "P") if anterior_left else ("P", "A"))
    assert (cor.left, cor.right) == (axi.left, axi.right)
    assert (cor.left, cor.right) == (("R", "L") if radiological else ("L", "R"))
    assert (sag.top, sag.bottom) == (cor.top, cor.bottom) == ("S", "I")
    assert (axi.top, axi.bottom) == ("A", "P")
    # and what the pixel affine actually does
    size = float(case.config.panel_px)
    for name, panel in case.panels.items():
        _, h, v = AXES[name]
        xyz = case.to_tkr(
            g.pixel_to_voxel(panel, 0, np.array([0.0, size, 0.0]), np.array([0.0, 0.0, size]))
        )
        increases_right = xyz[1, h] > xyz[0, h]
        increases_down = xyz[2, v] > xyz[0, v]
        assert panel.edges.right == LETTERS[h][1 if increases_right else 0]
        assert panel.edges.left == LETTERS[h][0 if increases_right else 1]
        assert panel.edges.bottom == LETTERS[v][1 if increases_down else 0]
        assert panel.edges.top == LETTERS[v][0 if increases_down else 1]
        assert not increases_down  # superior (axial: anterior) is always at the top
        if name != "sagittal":
            assert increases_right == (not radiological)  # radiological: x falls left to right
        else:
            assert increases_right == (not anterior_left)


def test_centre_slice_is_nearest_the_box_centre_not_the_midpoint() -> None:
    """Box centre (0.3, -95.2, -10.6) is nearest x = 0, y = -95, z = -11. The box midpoint in
    y (-76.7) and z (-15.6) would give positions 26 and 30."""
    panels = CASES["conformed"].panels
    assert [panels[name].centre_slice for name in PANEL_ORDER] == [30, 8, 35]
    assert panels["sagittal"].slice_mm[30] == 0.0
    assert panels["coronal"].slice_mm[8] == -95.0
    assert panels["axial"].slice_mm[35] == -11.0


@all_cases
def test_centre_slice_brute_force(case: Case) -> None:
    for name, panel in case.panels.items():
        distance = np.abs(np.array(panel.slice_mm) - case.centre[AXES[name][0]])
        assert distance[panel.centre_slice] == distance.min()


# ---------------------------------------------------------------- pixel <-> voxel <-> tkr


def test_apply_affine() -> None:
    ijk = np.array([[0.0, 0.0, 0.0], [128.0, 128.0, 128.0], [10.0, 20.0, 30.0]])
    np.testing.assert_allclose(
        g.apply_affine(LIA_1MM, ijk), [[128, -128, 128], [0, 0, 0], [118, -98, 108]]
    )
    np.testing.assert_allclose(g.apply_affine(PIR_ANISO, ijk[2]), [-36.0, 75.0, 50.0])


@all_cases
def test_pix2vox_and_vox2pix_are_inverse_homogeneous_affines(case: Case) -> None:
    pts = np.array([[0.0, 0.0, 1.0], [17.25, 211.5, 1.0], [-3.0, 400.0, 1.0]])
    for panel in case.panels.values():
        p2v, v2p = panel.pix2vox_array(), panel.vox2pix_array()
        np.testing.assert_allclose(p2v @ v2p, np.eye(3), atol=1e-9)
        np.testing.assert_allclose(v2p @ p2v, np.eye(3), atol=1e-9)
        np.testing.assert_array_equal(p2v[2], [0, 0, 1])
        np.testing.assert_array_equal(v2p[2], [0, 0, 1])
        # the stored matrix, used as a client would, gives what pixel_to_voxel gives
        ijk = g.pixel_to_voxel(panel, 0, pts[:, 0], pts[:, 1])
        hv1 = pts @ p2v.T
        np.testing.assert_allclose(hv1[:, 0], ijk[:, panel.h_voxel_axis], atol=1e-9)
        np.testing.assert_allclose(hv1[:, 1], ijk[:, panel.v_voxel_axis], atol=1e-9)
        np.testing.assert_allclose(hv1[:, 2], 1.0)
        assert {panel.slice_voxel_axis, panel.h_voxel_axis, panel.v_voxel_axis} == {0, 1, 2}


@all_cases
def test_vox2pix_is_the_inverse_in_closed_form_without_negative_zeros(case: Case) -> None:
    """Each entry is one IEEE division of entries of ``pix2vox``, so ``render.json`` does not
    depend on the LAPACK numpy is linked with. ``np.linalg.inv`` gave ``-0.0`` in the axial
    panel with OpenBLAS and other last bits of the shifts with reference LAPACK."""
    for panel in case.panels.values():
        (a, zero_h, c), (zero_v, b, d), _ = panel.pix2vox
        assert panel.vox2pix == ((1 / a, 0.0, -c / a), (0.0, 1 / b, -d / b), (0.0, 0.0, 1.0))
        assert (zero_h, zero_v) == (0.0, 0.0)
        for matrix in (panel.pix2vox, panel.vox2pix):
            assert not np.signbit(np.array(matrix)[np.array(matrix) == 0]).any()
        dumped = json.dumps(panel.model_dump(mode="json")["vox2pix"])
        assert "-0.0," not in dumped and "-0.0]" not in dumped


def test_a_tile_edge_on_voxel_coordinate_0_gives_no_negative_zero() -> None:
    """The field of view of the default box around x = 88 ends at tkr x = 128, which is voxel
    coordinate 0 of a conformed volume. The shift of the coronal and the axial ``pix2vox`` is
    then (128 - 128) / -1, a negative zero as it is computed, and ``-0.0`` in ``render.json``
    if it is stored so. No case of ``CASES`` has a shift of zero."""
    case = Case(LIA_1MM, (256, 256, 256), centre=(88.0, -95.2, -10.6))
    assert case.fov_edges("coronal")[0] == 128.0 and case.amap.offset[0] == 128.0
    for name in ("coronal", "axial"):
        assert case.panels[name].pix2vox[0] == (0.25, 0.0, 0.0)
        assert case.panels[name].vox2pix[0] == (4.0, 0.0, 0.0)
    for panel in case.panels.values():
        for matrix in (panel.pix2vox, panel.vox2pix):
            assert not np.signbit(np.array(matrix)[np.array(matrix) == 0]).any()
        assert "-0.0" not in json.dumps(panel.model_dump(mode="json"))


def test_a_box_without_extent_is_refused() -> None:
    # RenderConfig refuses this when it is made; model_copy does not validate
    flat = RenderConfig().model_copy(
        update={"box_x_mm": (0, 0), "box_y_mm": (0, 0), "box_z_mm": (0, 0), "margin_mm": 0}
    )
    case = Case(LIA_1MM, (256, 256, 256), config=flat)
    with pytest.raises(ValueError, match="no extent"):
        g.build_panels(case.box, case.amap, flat)


# ---------------------------------------------------------------- clickable region


def test_field_of_view_is_the_cube_around_the_box_midpoint() -> None:
    low, high = g.field_of_view(CASES["conformed"].box, CASES["conformed"].config)
    np.testing.assert_allclose(low, [-39.7, -116.7, -55.6], atol=1e-9)
    np.testing.assert_allclose(high, [40.3, -36.7, 24.4], atol=1e-9)
    low, high = g.field_of_view(CASES["non_default"].box, NON_DEFAULT)
    np.testing.assert_allclose(low, [-26.2, -111.7, -24.6], atol=1e-9)
    np.testing.assert_allclose(high, [31.8, -53.7, 33.4], atol=1e-9)


def test_clickable_region_by_hand() -> None:
    """Default config on the conformed volume, box centre (0.3, -95.2, -10.6). The slice planes
    are the voxel centres nearest the box ends: x -30 to 30, y -103 to -50, z -46 to 14. Each
    panel's box reaches half a slice step (0.5 mm) past them along its slice axis and spans the
    field of view (80 mm around the box midpoint) along the other two."""
    c = CASES["conformed"]
    region = g.clickable_region(c.box, c.panels, c.config)
    expected = [
        [[-30.5, -116.7, -55.6], [30.5, -36.7, 24.4]],  # sagittal
        [[-39.7, -103.5, -55.6], [40.3, -49.5, 24.4]],  # coronal
        [[-39.7, -116.7, -46.5], [40.3, -36.7, 14.5]],  # axial
    ]
    np.testing.assert_allclose(region, expected, atol=1e-9)


@all_cases
def test_clickable_region_holds_every_point_of_every_tile(case: Case) -> None:
    """Corners and centre of the first and last tile of each panel lie in that panel's box,
    whose faces are the tile edges and, across the slices, half a slice step beyond the planes."""
    region = g.clickable_region(case.box, case.panels, case.config)
    assert region.shape == (3, 2, 3)
    size = float(case.config.panel_px)
    half_step = case.config.slice_step_mm / 2
    for p, name in enumerate(PANEL_ORDER):
        panel = case.panels[name]
        w, h, v = AXES[name]
        left, right, top, bottom = case.fov_edges(name)
        low, high = region[p]
        np.testing.assert_allclose([low[h], high[h]], sorted([left, right]), atol=1e-9)
        np.testing.assert_allclose([low[v], high[v]], [bottom, top], atol=1e-9)
        assert low[w] == pytest.approx(min(panel.slice_mm) - half_step)
        assert high[w] == pytest.approx(max(panel.slice_mm) + half_step)
        px, py = np.array([0.0, size, 0.0, size, size / 2]), np.array([0.0, 0.0, size, size, 7.5])
        for pos in (0, len(panel.slices) - 1):
            xyz = case.to_tkr(g.pixel_to_voxel(panel, pos, px, py))
            assert g.in_region(xyz, region[p : p + 1], 1e-9).all()
            assert g.in_region(xyz, region, 1e-9).all()
            # a point just outside the tile, or a slice step beyond the last plane, is not
            beyond = case.to_tkr(g.pixel_to_voxel(panel, pos, np.array([-0.5]), np.array([7.5])))
            assert not g.in_region(beyond, region[p : p + 1]).any()
        past = case.to_tkr(g.pixel_to_voxel(panel, 0, np.array([7.5]), np.array([7.5])))
        past[0, w] = max(panel.slice_mm) + case.config.slice_step_mm
        assert not g.in_region(past, region[p : p + 1]).any()


def test_in_region_is_a_union_of_boxes_with_the_boundary_and_a_margin() -> None:
    region = np.array([[[0.0, 0.0, 0.0], [10.0, 2.0, 2.0]], [[0.0, 0.0, 0.0], [2.0, 10.0, 2.0]]])
    points = np.array(
        [
            [5.0, 1.0, 1.0],  # in the first box only
            [1.0, 5.0, 1.0],  # in the second only
            [1.0, 1.0, 1.0],  # in both
            [5.0, 5.0, 1.0],  # in the bounding box of the union, in neither box
            [10.0, 2.0, 0.0],  # on a corner of the first
            [10.5, 1.0, 1.0],  # 0.5 mm beyond the first
            [10.5, 2.5, 2.5],  # 0.5 mm beyond it along every axis
            [1.0, 1.0, 3.5],  # 1.5 mm beyond both
        ]
    )
    assert g.in_region(points, region).tolist() == [True, True, True, False, True] + [False] * 3
    assert g.in_region(points, region, 0.5).tolist() == [True, True, True, False] + [True] * 3 + [
        False
    ]
    assert g.in_region(points, region, 3.0).all()
    assert (
        g.in_region(points, region[:1]).tolist() == [True, False, True, False, True] + [False] * 3
    )
    assert g.in_region(np.empty((0, 3)), region).shape == (0,)
    assert g.in_region(points, np.empty((0, 2, 3))).tolist() == [False] * 8


def test_voxel_centre_convention_by_hand() -> None:
    """Box centre (0.125, -95.375, 0.125) on the conformed volume puts pixel centres of every
    fourth column and row exactly on voxel centres (4 px per mm):
    coronal column c is x = 40 - c / 4, i = 88 + c / 4; row r is z = 35 - r / 4, j = 93 + r / 4;
    axial row r is y = -37 - r / 4, k = 91 - r / 4. The sagittal panel has posterior at the
    left, so its column c is y = -116.75 + c / 4, k = 11.25 + c / 4, and columns 3, 7, ... sit
    on voxel centres."""
    panels = Case(LIA_1MM, (256, 256, 256), (0.125, -95.375, 0.125)).panels
    c, r = np.meshgrid(np.arange(0.0, 320.0, 4.0), np.arange(0.0, 320.0, 4.0))
    cor = g.pixel_to_voxel(panels["coronal"], 5, c + 0.5, r + 0.5)
    np.testing.assert_array_equal(cor[..., 0], 88 + c / 4)
    np.testing.assert_array_equal(cor[..., 1], 93 + r / 4)
    np.testing.assert_array_equal(cor[..., 2], panels["coronal"].slices[5])
    sag = g.pixel_to_voxel(panels["sagittal"], 7, c + 3.5, r + 0.5)
    np.testing.assert_array_equal(sag[..., 0], panels["sagittal"].slices[7])
    np.testing.assert_array_equal(sag[..., 1], 93 + r / 4)
    np.testing.assert_array_equal(sag[..., 2], 12 + c / 4)
    axi = g.pixel_to_voxel(panels["axial"], 0, c + 0.5, r + 0.5)
    np.testing.assert_array_equal(axi[..., 0], 88 + c / 4)
    np.testing.assert_array_equal(axi[..., 1], panels["axial"].slices[0])
    np.testing.assert_array_equal(axi[..., 2], 91 - r / 4)
    # one pixel to the right is a quarter of a voxel, not a whole one
    assert g.pixel_to_voxel(panels["coronal"], 5, 1.5, 0.5)[0] == 88.25


@all_cases
def test_voxel_centres_land_where_the_field_of_view_says(case: Case) -> None:
    """Integer voxel coordinates -> tkr through the affine -> tile position from the documented
    field of view -> pixel_to_voxel gives the integers back; voxel_to_pixel agrees."""
    rng = np.random.default_rng(1)
    for name, panel in case.panels.items():
        ijk = rng.integers(0, case.shape, size=(40, 3)).astype(float)
        pos = int(rng.integers(len(panel.slices)))
        ijk[:, panel.slice_voxel_axis] = panel.slices[pos]
        px, py = case.expected_pixel(name, case.to_tkr(ijk))
        back = g.pixel_to_voxel(panel, pos, px, py)
        np.testing.assert_allclose(back, ijk, atol=1e-9)
        np.testing.assert_array_equal(np.round(back, 9), ijk)
        qx, qy, along = g.voxel_to_pixel(panel, ijk)
        np.testing.assert_allclose(qx, px, atol=1e-9)
        np.testing.assert_allclose(qy, py, atol=1e-9)
        np.testing.assert_array_equal(along, panel.slices[pos])


coords = st.lists(st.tuples(st.floats(0, 1), st.floats(0, 1)), min_size=1, max_size=6)


@settings(max_examples=40, deadline=None)
@given(name=st.sampled_from(sorted(CASES)), pts=coords)
def test_click_round_trip_pixels(name: str, pts: list[tuple[float, float]]) -> None:
    """Random clicks, on and around the tile, on every slice of every panel:
    pixel -> voxel -> pixel, and pixel -> voxel -> tkr -> pixel."""
    case = CASES[name]
    px, py = (np.array(pts) * 1.5 - 0.25).T * case.config.panel_px
    for pname, panel in case.panels.items():
        for pos in range(len(panel.slices)):
            ijk = g.pixel_to_voxel(panel, pos, px, py)
            np.testing.assert_array_equal(ijk[:, panel.slice_voxel_axis], panel.slices[pos])
            xyz = case.to_tkr(ijk)
            np.testing.assert_allclose(xyz[:, AXES[pname][0]], panel.slice_mm[pos], atol=1e-9)
            ex, ey = case.expected_pixel(pname, xyz)
            np.testing.assert_allclose((ex, ey), (px, py), atol=1e-7)
            qx, qy, along = g.voxel_to_pixel(panel, ijk)
            np.testing.assert_allclose((qx, qy), (px, py), atol=1e-7)
            tx, ty = g.tkr_to_pixel(panel, case.affine, xyz)
            np.testing.assert_allclose((tx, ty), (px, py), atol=1e-7)
            assert g.nearest_slice(panel, float(along[0])) == pos


@settings(max_examples=40, deadline=None)
@given(name=st.sampled_from(sorted(CASES)), pts=coords)
def test_click_round_trip_voxels(name: str, pts: list[tuple[float, float]]) -> None:
    """Random voxels on every slice of every panel: voxel -> pixel -> voxel, and a tkr point
    on the slice plane -> pixel -> voxel -> tkr."""
    case = CASES[name]
    uv = np.array(pts)
    for panel in case.panels.values():
        ijk = np.empty((len(pts), 3))
        ijk[:, panel.h_voxel_axis] = uv[:, 0] * (case.shape[panel.h_voxel_axis] - 1)
        ijk[:, panel.v_voxel_axis] = uv[:, 1] * (case.shape[panel.v_voxel_axis] - 1)
        for pos in range(len(panel.slices)):
            ijk[:, panel.slice_voxel_axis] = panel.slices[pos]
            px, py, _ = g.voxel_to_pixel(panel, ijk)
            np.testing.assert_allclose(g.pixel_to_voxel(panel, pos, px, py), ijk, atol=1e-8)
            xyz = case.to_tkr(ijk)
            tx, ty = g.tkr_to_pixel(panel, case.affine, xyz)
            back = g.apply_affine(case.affine, g.pixel_to_voxel(panel, pos, tx, ty))
            np.testing.assert_allclose(back, xyz, atol=1e-8)


@all_cases
def test_pixel_to_voxel_accepts_scalars(case: Case) -> None:
    panel = case.panels["coronal"]
    ijk = g.pixel_to_voxel(panel, 0, 12.5, 40.0)
    assert ijk.shape == (3,)
    np.testing.assert_allclose(
        ijk, g.pixel_to_voxel(panel, 0, np.array([12.5]), np.array([40.0]))[0]
    )


@all_cases
def test_a_single_point_maps_to_a_pixel(case: Case) -> None:
    panel = case.panels["coronal"]
    ijk = g.pixel_to_voxel(panel, 0, 12.5, 40.0)
    px, py, along = g.voxel_to_pixel(panel, ijk)
    assert (float(px), float(py)) == pytest.approx((12.5, 40.0))
    assert float(along) == panel.slices[0]
    tx, ty = g.tkr_to_pixel(panel, case.affine, case.to_tkr(ijk))
    assert (float(tx), float(ty)) == pytest.approx((12.5, 40.0))


@all_cases
def test_nearest_slice(case: Case) -> None:
    for panel in case.panels.values():
        slices = np.array(panel.slices, dtype=float)
        gap = np.abs(np.diff(slices)).min()
        for pos, s in enumerate(panel.slices):
            assert g.nearest_slice(panel, s) == pos
            assert g.nearest_slice(panel, s + 0.4 * gap) == pos
            assert g.nearest_slice(panel, s - 0.4 * gap) == pos
        # beyond either end: the end slice
        assert g.nearest_slice(panel, 2 * slices[0] - slices[1]) == 0
        assert g.nearest_slice(panel, 2 * slices[-1] - slices[-2]) == len(slices) - 1


# ---------------------------------------------------------------- frame tables


def ping_pong(n: int) -> list[int]:
    """Forward frames, then the same frames in reverse without the two end frames."""
    forward = list(range(n))
    return forward + forward[::-1][1:-1]


def check_sync(frames: tuple[tuple[int, int, int], ...], counts: list[int]) -> None:
    longest = max(counts)
    assert len(frames) == 2 * longest - 2
    assert frames[0] == (0, 0, 0)
    assert frames[longest - 1] == tuple(n - 1 for n in counts)
    for f in range(longest):
        for p, n in enumerate(counts):
            exact = f * (n - 1) / (longest - 1)
            # the docs' round(): either neighbour is accepted for an exact half
            assert abs(frames[f][p] - exact) <= 0.5
            if abs(exact - np.floor(exact) - 0.5) > 1e-9:
                assert frames[f][p] == int(np.floor(exact + 0.5))
    # reverse without the two end frames; the loop wraps without showing a frame twice
    assert list(frames[longest:]) == list(frames[longest - 2 : 0 : -1])
    assert all(frames[i] != frames[i - 1] for i in range(len(frames)))
    # phase-locked: every panel shows every one of its slices
    for p, n in enumerate(counts):
        assert {frame[p] for frame in frames} == set(range(n))


@all_cases
def test_sync_frame_table(case: Case) -> None:
    counts = [len(case.panels[name].slices) for name in PANEL_ORDER]
    check_sync(g.frame_tables(case.panels)["sync"].frames, counts)


@all_cases
def test_seq_frame_table(case: Case) -> None:
    """Each panel in turn ping-pongs over its own slices; the other two hold centre_slice."""
    counts = [len(case.panels[name].slices) for name in PANEL_ORDER]
    hold = [case.panels[name].centre_slice for name in PANEL_ORDER]
    expected = []
    for p, n in enumerate(counts):
        for s in ping_pong(n):
            frame = list(hold)
            frame[p] = s
            expected.append(tuple(frame))
    frames = g.frame_tables(case.panels)["seq"].frames
    assert list(frames) == expected
    assert len(frames) == sum(2 * n - 2 for n in counts)


def test_default_frame_tables() -> None:
    """120 synchronized and 346 one-at-a-time frames; holds at slices 30, 8 and 35."""
    tables = g.frame_tables(CASES["conformed"].panels)
    assert set(tables) == {"sync", "seq"}
    sync, seq = tables["sync"].frames, tables["seq"].frames
    assert (len(sync), len(seq)) == (120, 346)
    assert sync[0] == (0, 0, 0)
    assert sync[60] == (60, 53, 60)
    # 30 * 53 / 60 = 26.5: an exact half goes to the even slice. This is the poster frame.
    assert sync[30] == sync[90] == (30, 26, 30)
    assert sync[15] == (15, 13, 15)  # round(15 * 53 / 60) = round(13.25)
    assert sync[45] == (45, 40, 45)  # round(45 * 53 / 60) = round(39.75)
    assert sync[61] == sync[59]
    assert sync[119] == sync[1]
    assert seq[0] == (0, 8, 35)
    assert seq[60] == (60, 8, 35)
    assert seq[119] == (1, 8, 35)
    assert seq[120] == (30, 0, 35)
    assert seq[173] == (30, 53, 35)
    assert seq[226] == (30, 8, 0)
    assert seq[345] == (30, 8, 1)


def test_sync_frame_table_rounds_an_exact_half_to_the_even_slice() -> None:
    """docs/rendering.md, "Frame tables". With 5, 3 and 5 slices the coronal panel is at
    f / 2: half-way at f = 1, where the even slice is the lower one, and at f = 3, where it
    is the higher one."""
    case = Case(
        LIA_1MM,
        (256, 256, 256),
        config=RenderConfig(box_x_mm=(0, 4), box_y_mm=(0, 2), box_z_mm=(0, 4)),
    )
    assert [len(case.panels[name].slices) for name in PANEL_ORDER] == [5, 3, 5]
    sync = g.frame_tables(case.panels)["sync"].frames
    assert [frame[1] for frame in sync] == [0, 0, 1, 2, 2, 2, 1, 0]
    assert [frame[0] for frame in sync] == [frame[2] for frame in sync] == [0, 1, 2, 3, 4, 3, 2, 1]


def test_frame_tables_with_a_box_one_slice_thick() -> None:
    thin = Case(LIA_1MM, (256, 256, 256), config=RenderConfig(box_x_mm=(0, 0)))
    counts = [len(thin.panels[name].slices) for name in PANEL_ORDER]
    assert counts == [1, 54, 61]
    tables = g.frame_tables(thin.panels)
    check_sync(tables["sync"].frames, counts)
    assert {frame[0] for frame in tables["seq"].frames} == {0}
    # a one-slice panel has a one-frame ping-pong (the docs' 2n - 2 would be 0)
    assert len(tables["seq"].frames) == 1 + 106 + 120


def test_frame_tables_with_a_single_slice_everywhere() -> None:
    config = RenderConfig(box_x_mm=(0, 0), box_y_mm=(0, 0), box_z_mm=(0, 0))
    point = Case(LIA_1MM, (256, 256, 256), config=config)
    tables = g.frame_tables(point.panels)
    for mode in ("sync", "seq"):
        assert len(tables[mode].frames) >= 1
        assert set(tables[mode].frames) == {(0, 0, 0)}


def test_crosshairs_by_hand() -> None:
    """Frame (30, 8, 35) of the default case is the point x = 0, y = -95, z = -11. At 4 px per
    mm from the edges of test_field_of_view_by_hand: y is 21.7 mm from the posterior edge,
    z 35.4 mm from the superior edge, x 40.3 mm from the right-of-subject edge."""
    lines = g.crosshair_pixels(CASES["conformed"].panels, (30, 8, 35))
    assert lines["sagittal"] == pytest.approx((86.8, 141.6))
    assert lines["coronal"] == pytest.approx((161.2, 141.6))
    assert lines["axial"] == pytest.approx((161.2, 233.2))


@all_cases
def test_crosshairs_mark_the_other_two_planes(case: Case) -> None:
    tables = g.frame_tables(case.panels)
    frames = tables["sync"].frames[::7] + tables["seq"].frames[::11]
    for frame in frames:
        point = np.array(
            [case.panels[name].slice_mm[pos] for name, pos in zip(PANEL_ORDER, frame, strict=True)]
        )
        lines = g.crosshair_pixels(case.panels, frame)
        assert set(lines) == set(PANEL_ORDER)
        for name in PANEL_ORDER:
            px, py = case.expected_pixel(name, point)
            assert lines[name] == pytest.approx((float(px), float(py)), abs=1e-7)


# ---------------------------------------------------------------- model


def changed(value: object) -> object:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.25
    if isinstance(value, str):
        return "#123456"
    if isinstance(value, tuple):
        return (value[0], value[1] + 0.25)
    raise AssertionError(f"no way to change {value!r}")


def test_config_hash_is_twelve_hex_characters_and_equal_for_equal_configs() -> None:
    assert re.fullmatch(r"[0-9a-f]{12}", RenderConfig().config_hash())
    assert RenderConfig().config_hash() == RenderConfig().config_hash()
    # integers and lists validate to the same floats and tuples
    spelled = RenderConfig.model_validate(
        {"box_x_mm": [-30, 30], "margin_mm": 10, "radiological": True, "panel_px": 320}
    )
    assert spelled == RenderConfig()
    assert spelled.config_hash() == RenderConfig().config_hash()


def test_config_hash_is_sensitive_to_every_field() -> None:
    base = RenderConfig()
    hashes = {"": base.config_hash()}
    for field in RenderConfig.model_fields:
        other = RenderConfig.model_validate({field: changed(getattr(base, field))})
        assert other != base, field
        hashes[field] = other.config_hash()
    assert len(set(hashes.values())) == len(RenderConfig.model_fields) + 1, hashes


def test_config_hash_is_stable_across_processes() -> None:
    code = (
        "from surfcensus.render.model import RenderConfig as C;"
        "print(C().config_hash(), C(radiological=False, box_x_mm=(-20, 25)).config_hash())"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONHASHSEED": "12345"},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    here = [
        RenderConfig().config_hash(),
        RenderConfig(radiological=False, box_x_mm=(-20, 25)).config_hash(),
    ]
    assert out == here


def test_config_is_frozen_and_rejects_unknown_and_invalid_fields() -> None:
    config = RenderConfig()
    with pytest.raises(ValidationError):
        setattr(config, "panel_px", 640)  # noqa: B010
    assert config.panel_px == 320
    for bad in (
        {"panel_pixels": 320},
        {"panel_px": 8},
        {"margin_mm": -1},
        {"slice_step_mm": 0},
        {"box_x_mm": (-30, 0, 30)},
    ):
        with pytest.raises(ValidationError):
            RenderConfig.model_validate(bad)


def test_config_hash_of_the_defaults_is_unchanged() -> None:
    """The hash decides whether a render is up to date, so it may change only when the
    serialized value of a default changes (or the schema version). Validation rules such as
    the colour format are not part of it."""
    assert RenderConfig().config_hash() == "17f83a0c9b6c"


def test_config_defaults_are_the_documented_ones() -> None:
    c = RenderConfig()
    assert (c.box_x_mm, c.box_y_mm, c.box_z_mm) == ((-30, 30), (-8, 45), (-35, 25))
    assert (c.margin_mm, c.slice_step_mm, c.panel_px) == (10, 1, 320)
    assert (c.radiological, c.sagittal_anterior_left) == (True, False)
    assert (c.pial_colour, c.white_colour, c.line_width_px) == ("#FFD400", "#FF3B30", 1.25)
    assert (c.window_percentiles, c.brainmask_dilation) == ((1, 99.5), 3)
    assert (c.webp_quality, c.fps, c.crf) == (80, 12, 23)
    assert c.vertex_map_margin_mm == 5  # the mark snap radius of docs/rater_app.md


def make_info(case: Case, reverse: bool = False) -> RenderInfo:
    """A RenderInfo assembled from the geometry functions, as the renderer will."""
    poles = g.find_poles(LH, RH)
    box = g.compute_box(LH, RH, poles, case.config)
    panels = g.build_panels(box, case.amap, case.config)
    inputs = {
        "t1": InputFile(name="T1.mgz", size=5_600_000, sha256="ab" * 32),
        "lh_pial": InputFile(name="lh.pial.T1", size=5_300_000, sha256="cd" * 32),
        "rh_pial": InputFile(name="rh.pial.T1", size=5_300_001, sha256="ef" * 32),
    }
    versions = {"numpy": "2.5.4", "matplotlib": "3.10.0", "pillow": "11.0.0", "nibabel": "5.3.2"}
    if reverse:  # same content, other insertion order
        inputs = dict(reversed(inputs.items()))
        versions = dict(reversed(versions.items()))
        poles = dict(reversed(poles.items()))
        panels = dict(reversed(panels.items()))
    return RenderInfo.model_validate(
        {
            "surfcensus_version": "0.0.1",
            "config": case.config,
            "config_hash": case.config.config_hash(),
            "volume": {
                "shape": case.shape,
                "voxel_size_mm": [abs(s) for s in case.amap.step],
                "vox2ras_tkr": case.affine.tolist(),
                "vox2ras": (case.affine + np.array([[0, 0, 0, 1.5]] * 3 + [[0, 0, 0, 0]])).tolist(),
            },
            "inputs": inputs,
            "poles": poles,
            "box": box,
            "window": (3.0, 141.5),
            "panels": panels,
            "modes": g.frame_tables(panels),
            "video": {"fps": 12, "n_frames": 120, "width": 960, "height": 320},
            "vertex_map": {"n_lh": 1234, "n_rh": 1250, "n_lh_visible": 900, "n_rh_visible": 911},
            "versions": versions,
        }
    )


@pytest.mark.parametrize("name", ["conformed", "hires", "pir", "non_default"])
def test_render_info_serializes_deterministically_and_round_trips(name: str) -> None:
    info = make_info(CASES[name])
    text = info.to_json()
    assert (
        text == make_info(CASES[name]).to_json() == make_info(CASES[name], reverse=True).to_json()
    )
    assert text.endswith("\n")
    back = RenderInfo.model_validate_json(text)
    assert back == info
    assert back.to_json() == text
    # the affines survive exactly, so a client converts clicks as the server does
    for pname in PANEL_ORDER:
        np.testing.assert_array_equal(
            back.panels[pname].pix2vox_array(), info.panels[pname].pix2vox_array()
        )
    np.testing.assert_array_equal(np.array(back.volume.vox2ras_tkr), CASES[name].affine)


def test_render_info_contents() -> None:
    info = make_info(CASES["conformed"])
    data = json.loads(info.to_json())
    assert data["schema_version"] == 2
    assert data["vertex_map"] == {
        "file": "vertex_map.npz",
        "fsaverage": None,
        "n_lh": 1234,
        "n_rh": 1250,
        "n_lh_visible": 900,
        "n_rh_visible": 911,
    }
    assert data["config_hash"] == RenderConfig().config_hash()
    assert RenderConfig.model_validate(data["config"]) == RenderConfig()
    assert list(data["panels"]) == sorted(PANEL_ORDER)
    assert [len(data["panels"][name]["slices"]) for name in PANEL_ORDER] == [61, 54, 61]
    assert (len(data["modes"]["sync"]["frames"]), len(data["modes"]["seq"]["frames"])) == (120, 346)
    for mode in data["modes"].values():  # every frame names a tile that exists
        for frame in mode["frames"]:
            for name, pos in zip(PANEL_ORDER, frame, strict=True):
                assert 0 <= pos < len(data["panels"][name]["slices"])
    assert info.tiles.gray_path("sagittal", 7) == "tiles/sagittal_007.webp"
    assert info.tiles.contours_path("axial", 60) == "tiles/axial_060_c.webp"
    assert set(InputFile.model_fields) == {"name", "size", "sha256"}


def test_render_info_json_has_no_paths_and_no_timestamps() -> None:
    def walk(node: object, key: str = "") -> list[tuple[str, object]]:
        if isinstance(node, dict):
            return [item for k, v in node.items() for item in walk(v, k)]
        if isinstance(node, list):
            return [item for v in node for item in walk(v, key)]
        return [(key, node)]

    leaves = walk(json.loads(make_info(CASES["conformed"]).to_json()))
    keys = {key for key, _ in leaves}
    assert not [k for k in keys if re.search(r"time|date|stamp|_at$|path|dir|root", k)]
    strings = {(key, value) for key, value in leaves if isinstance(value, str)}
    # the only separators are in the two relative tile-name templates
    assert {key for key, value in strings if "/" in value} == {"gray", "contours"}
    for _, value in strings:
        assert "\\" not in value
        assert not value.startswith(("/", "~")) and not re.match(r"[A-Za-z]:", value)
        assert not re.search(r"\d{4}-\d{2}-\d{2}|\d{2}:\d{2}", value)
    assert os.getcwd() not in make_info(CASES["conformed"]).to_json()
    with pytest.raises(ValidationError):  # nowhere to put anything else
        RenderInfo.model_validate(
            {**make_info(CASES["conformed"]).model_dump(), "rendered_at": "x"}
        )
