"""Tests of render/frames.py on the synthetic subject, whose geometry is known analytically.

Expected values are computed here from the conventions of docs/rendering.md (tkr RAS in mm,
integer voxel index = voxel centre, pixel (column c, row r) sampled at (c + 0.5, r + 0.5) with
the origin at the top-left corner), not with the helpers that frames.py itself uses.

The golden image ``tests/data/frames_strip_golden.png`` is regenerated with::

    SURFCENSUS_REGENERATE_GOLDEN=1 uv run pytest tests/test_frames.py -k golden
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFont
from scipy import ndimage
from scipy.spatial import cKDTree
from synth import (  # pyrefly: ignore[missing-import]
    PIAL_SEMI_AXES,
    T1_GRAY,
    T1_HEAD,
    T1_WHITE,
    WHITE_SEMI_AXES,
    SyntheticSubject,
)

from surfcensus.fsio.surfaces import read_surface
from surfcensus.fsio.volumes import read_volume
from surfcensus.render import frames, geometry
from surfcensus.render.model import PANEL_ORDER, BoxInfo, Frame, PanelInfo, RenderConfig
from surfcensus.render.slicing import MeshSlicer

# (tkr axis normal to the slices, along the horizontal, along the vertical): "Display orientation"
AXES = {"sagittal": (0, 1, 2), "coronal": (1, 0, 2), "axial": (2, 0, 1)}
SURFACES = ("lh_white", "rh_white", "lh_pial", "rh_pial")
ORIENTATIONS = [
    RenderConfig(radiological=r, sagittal_anterior_left=a)
    for r in (True, False)
    for a in (True, False)
]
orientation_ids = [
    f"{'radiological' if c.radiological else 'neurological'}-"
    f"{'anterior' if c.sagittal_anterior_left else 'posterior'}_left"
    for c in ORIENTATIONS
]
GOLDEN = Path(__file__).parent / "data" / "frames_strip_golden.png"


@dataclass(frozen=True)
class Scene:
    """A synthetic subject with its panels, and the display geometry worked out from the docs."""

    subject: SyntheticSubject
    data: np.ndarray
    affine: np.ndarray
    """vox2ras_tkr of the volume."""
    config: RenderConfig
    box: BoxInfo
    panels: dict[str, PanelInfo]
    slicers: dict[str, MeshSlicer]

    def fov_edges(self, name: str) -> tuple[float, float, float, float]:
        """tkr coordinate at the left, right, top and bottom edge of a panel: a square of the
        largest box extent plus two margins, centred on the box midpoint."""
        low, high = np.array(self.box.low), np.array(self.box.high)
        half = ((high - low).max() + 2 * self.config.margin_mm) / 2
        mid = (low + high) / 2
        _, h, v = AXES[name]
        high_at_left = (
            self.config.sagittal_anterior_left if name == "sagittal" else self.config.radiological
        )
        left, right = (
            (mid[h] + half, mid[h] - half) if high_at_left else (mid[h] - half, mid[h] + half)
        )
        return float(left), float(right), float(mid[v] + half), float(mid[v] - half)

    def pixel_of(self, name: str, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Panel pixel coordinates of tkr points."""
        left, right, top, bottom = self.fov_edges(name)
        _, h, v = AXES[name]
        size = self.config.panel_px
        return (
            (xyz[..., h] - left) / (right - left) * size,
            (xyz[..., v] - top) / (bottom - top) * size,
        )

    def tkr_of_pixel(self, name: str, pos: int, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """tkr points of panel pixel coordinates on the tile of slice ``pos``."""
        left, right, top, bottom = self.fov_edges(name)
        w, h, v = AXES[name]
        size = self.config.panel_px
        xyz = np.empty((*np.shape(px), 3))
        xyz[..., w] = self.panels[name].slice_mm[pos]
        xyz[..., h] = left + px / size * (right - left)
        xyz[..., v] = top + py / size * (bottom - top)
        return xyz

    @property
    def voxel_px(self) -> float:
        left, right, _, _ = self.fov_edges("coronal")
        return self.subject.voxel_size * self.config.panel_px / abs(right - left)

    def ellipse(self, name: str, pos: int, surface: str) -> np.ndarray:
        """Densely sampled section of the two ellipsoids of ``surface`` ("pial" or "white") by
        the plane of a slice, in panel pixel coordinates, shape (n, 2); empty if it misses."""
        w, h, v = AXES[name]
        semi = PIAL_SEMI_AXES if surface == "pial" else WHITE_SEMI_AXES
        t = np.linspace(0, 2 * np.pi, 20000)
        points = []
        for centre in (self.subject.lh_centre, self.subject.rh_centre):
            u = (self.panels[name].slice_mm[pos] - centre[w]) / semi[w]
            if abs(u) < 1:
                s = np.sqrt(1 - u * u)
                xyz = np.zeros((t.size, 3))
                xyz[:, h] = centre[h] + semi[h] * s * np.cos(t)
                xyz[:, v] = centre[v] + semi[v] * s * np.sin(t)
                points.append(np.column_stack(self.pixel_of(name, xyz)))
        return np.concatenate(points) if points else np.empty((0, 2))

    def grazing(self, name: str, pos: int) -> bool:
        """The plane passes within the outer 3 % of an ellipsoid's semi-axis, where the section
        is a few voxels across and neither the mesh nor the voxel grid resolves it."""
        w = AXES[name][0]
        plane = self.panels[name].slice_mm[pos]
        return any(
            0.97 < abs((plane - centre[w]) / semi[w]) < 1.03
            for centre in (self.subject.lh_centre, self.subject.rh_centre)
            for semi in (PIAL_SEMI_AXES, WHITE_SEMI_AXES)
        )

    def layers(self, name: str, pos: int, *surfaces: str) -> list[tuple[np.ndarray, str]]:
        chosen = {k: s for k, s in self.slicers.items() if k.endswith(surfaces or SURFACES)}
        return frames.contour_layers(self.panels[name], pos, self.affine, chosen, self.config)


_loaded: dict[
    Path, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, MeshSlicer]]
] = {}


def make_scene(
    subject: SyntheticSubject, config: RenderConfig | None = None, box: BoxInfo | None = None
) -> Scene:
    if subject.path not in _loaded:
        volume = read_volume(subject.path / "mri" / "T1.mgz")
        surfaces = {
            f"{hemi}_{name}": read_surface(subject.path / "surf" / f"{hemi}.{name}")
            for hemi in ("lh", "rh")
            for name in ("white", "pial")
        }
        _loaded[subject.path] = (
            volume.data,
            volume.vox2ras_tkr,
            {k: s.vertices for k, s in surfaces.items()},
            {k: MeshSlicer(s.vertices, s.faces) for k, s in surfaces.items()},
        )
    data, affine, vertices, slicers = _loaded[subject.path]
    config = config or RenderConfig()
    if box is None:
        lh, rh = vertices["lh_pial"], vertices["rh_pial"]
        box = geometry.compute_box(lh, rh, geometry.find_poles(lh, rh), config)
    panels = geometry.build_panels(box, geometry.axis_map(affine, subject.shape), config)
    return Scene(subject, data, affine, config, box, dict(panels), slicers)


@pytest.fixture(scope="module")
def scene(synth_subject: SyntheticSubject) -> Scene:
    return make_scene(synth_subject)


@pytest.fixture(scope="module", params=["synth_subject", "synth_subject_hires"])
def any_scene(request: pytest.FixtureRequest) -> Scene:
    """The 1 mm and the 0.8 mm subject in turn."""
    return make_scene(request.getfixturevalue(request.param))


@pytest.fixture(scope="module")
def renderer(scene: Scene):
    size = scene.config.panel_px
    with frames.ContourRenderer(size, size, scene.config.line_width_px) as r:
        yield r


def spread(panel: PanelInfo, count: int) -> list[int]:
    """``count`` slice positions spread evenly over a panel, ends included."""
    return sorted({round(f * (len(panel.slices) - 1)) for f in np.linspace(0, 1, count)})


def line_pixels(rgba: np.ndarray) -> np.ndarray:
    """Centres (px, py) of the pixels a line covers by more than half, shape (n, 2)."""
    rows, cols = np.nonzero(rgba[..., 3] > 127)
    return np.column_stack([cols + 0.5, rows + 0.5])


def window_value(value: np.ndarray | float, window: tuple[float, float]) -> np.ndarray:
    """lo -> 0, hi -> 255, linear, clipped, rounded; refuses ties, which the docs leave open."""
    lo, hi = window
    scaled = np.clip((np.asarray(value, dtype=float) - lo) / (hi - lo), 0, 1) * 255
    assert (np.abs(scaled - np.floor(scaled) - 0.5) > 1e-6).all()
    return np.floor(scaled + 0.5).astype(np.uint8)


# ---------------------------------------------------------------- gray_tile


def expected_gray(
    s: Scene, data: np.ndarray, name: str, pos: int, c: int, r: int, window: tuple[float, float]
) -> int:
    """One pixel, straight from the docs: pixel centre -> tkr -> voxel -> nearest -> window."""
    xyz = s.tkr_of_pixel(name, pos, np.array(c + 0.5), np.array(r + 0.5))
    ijk = np.linalg.solve(s.affine, [*xyz, 1.0])[:3]
    w = int(np.argmax(np.abs(s.affine[AXES[name][0], :3])))
    assert abs(ijk[w] - round(ijk[w])) < 1e-3, "the slice plane passes through voxel centres"
    assert (np.abs(ijk - np.floor(ijk) - 0.5) > 1e-3).all(), "pixel centre on a voxel boundary"
    nearest = np.floor(ijk + 0.5).astype(int)
    if (nearest < 0).any() or (nearest >= data.shape).any():
        return 0
    return int(window_value(data[tuple(nearest)], window))


@pytest.mark.parametrize("window", [(0.0, 255.0), (20.0, 97.0)])
def test_gray_tile_matches_a_per_pixel_computation(
    any_scene: Scene, window: tuple[float, float]
) -> None:
    s = any_scene
    rng = np.random.default_rng(0)
    size = s.config.panel_px
    outside = 0
    for name, panel in s.panels.items():
        for pos in spread(panel, 3):
            tile = frames.gray_tile(s.data, panel, pos, window)
            assert tile.shape == (size, size) and tile.dtype == np.uint8
            pixels = [(0, 0), (size - 1, 0), (0, size - 1), (size - 1, size - 1)]
            pixels += rng.integers(0, size, size=(150, 2)).tolist()
            for c, r in pixels:
                expected = expected_gray(s, s.data, name, pos, c, r, window)
                assert tile[r, c] == expected, (name, pos, c, r)
            outside += int((tile == 0).sum())
    assert outside > 0


@pytest.mark.parametrize("config", ORIENTATIONS, ids=orientation_ids)
def test_gray_tile_matches_a_per_pixel_computation_in_every_orientation(
    synth_subject: SyntheticSubject, config: RenderConfig
) -> None:
    s = make_scene(synth_subject, config)
    rng = np.random.default_rng(1)
    for name, panel in s.panels.items():
        pos = len(panel.slices) // 3
        tile = frames.gray_tile(s.data, panel, pos, (0.0, 255.0))
        for c, r in rng.integers(0, config.panel_px, size=(100, 2)).tolist():
            assert tile[r, c] == expected_gray(s, s.data, name, pos, c, r, (0.0, 255.0))


def test_gray_tile_pixels_outside_the_volume_are_zero(any_scene: Scene) -> None:
    s = any_scene
    panel = s.panels["sagittal"]
    # in-volume background (0) maps to 100 with this window, so only the outside is 0
    tile = frames.gray_tile(s.data, panel, 5, (-100.0, 155.0))
    # the volume ends half a voxel behind its most posterior voxel centre
    k = int(np.argmax(np.abs(s.affine[1, :3])))
    centres = s.affine[1, 3] + s.affine[1, k] * np.arange(s.subject.shape[k])
    edge = centres.min() - s.subject.voxel_size / 2
    cols = np.arange(panel.width)
    y = s.tkr_of_pixel("sagittal", 5, cols + 0.5, np.full(panel.width, 0.5))[:, 1]
    assert (np.abs(y - edge) > 1e-6).all()
    assert 0 < (y < edge).sum() < panel.width
    assert (tile[:, y < edge] == 0).all()
    assert (tile[:, y >= edge] >= 100).all()

    beyond = panel.model_copy(update={"slices": (s.subject.shape[0],), "slice_mm": (0.0,)})
    assert not frames.gray_tile(s.data, beyond, 0, (0.0, 255.0)).any()


def test_gray_tile_window_and_dtypes(scene: Scene) -> None:
    panel = scene.panels["coronal"]
    pos = len(panel.slices) // 2
    raw = frames.gray_tile(scene.data, panel, pos, (0.0, 255.0))
    assert set(np.unique(raw)) == {T1_HEAD, T1_GRAY, T1_WHITE}

    tile = frames.gray_tile(scene.data, panel, pos, (float(T1_HEAD), float(T1_WHITE)))
    assert set(np.unique(tile)) == {0, 135, 255}  # (70 - 25) / (110 - 25) * 255 = 135
    assert np.array_equal(tile == 135, raw == T1_GRAY)
    clipped = frames.gray_tile(scene.data, panel, pos, (60.0, 82.0))
    assert set(np.unique(clipped)) == {0, 116, 255}  # (70 - 60) / (82 - 60) * 255 = 115.9
    assert np.array_equal(clipped == 0, raw == T1_HEAD)

    for dtype in (">i2", np.float32, np.float64, np.int64):
        again = frames.gray_tile(scene.data.astype(dtype), panel, pos, (20.0, 97.0))
        assert np.array_equal(again, window_value(raw, (20.0, 97.0)))
    with_nan = scene.data.astype(np.float32)
    with_nan[with_nan == T1_HEAD] = np.nan
    assert np.array_equal(
        frames.gray_tile(with_nan, panel, pos, (0.0, 255.0)), np.where(raw == T1_HEAD, 0, raw)
    )
    for bad in ((5.0, 5.0), (9.0, 2.0)):
        with pytest.raises(ValueError):
            frames.gray_tile(scene.data, panel, pos, bad)


def test_gray_tile_maps_a_ramp_linearly(scene: Scene) -> None:
    # intensity 1.3 * i along voxel axis 0, which runs along tkr x: one value per coronal column
    ramp = np.broadcast_to(
        (1.3 * np.arange(scene.subject.shape[0]))[:, None, None], scene.subject.shape
    )
    window = (70.3, 140.9)
    panel = scene.panels["coronal"]
    tile = frames.gray_tile(ramp, panel, 0, window)
    cols = np.arange(panel.width)
    x = scene.tkr_of_pixel("coronal", 0, cols + 0.5, np.full(panel.width, 0.5))[:, 0]
    i = np.floor((x - scene.affine[0, 3]) / scene.affine[0, 0] + 0.5)
    expected = window_value(1.3 * i, window)
    assert expected.min() == 0 and expected.max() == 255 and np.unique(expected).size > 50
    assert np.array_equal(tile, np.broadcast_to(expected, tile.shape))


@pytest.mark.parametrize("config", ORIENTATIONS, ids=orientation_ids)
def test_marker_voxel_appears_where_the_edge_letters_say(
    synth_subject: SyntheticSubject, config: RenderConfig
) -> None:
    """One bright voxel on the right, anterior, superior side of the field of view."""
    s = make_scene(synth_subject, config)
    marker = np.array([16.0, -41.0, 27.0])
    mid = (np.array(s.box.low) + np.array(s.box.high)) / 2
    assert ((marker - mid) > 10).all()
    ijk = np.linalg.solve(s.affine, [*marker, 1.0])[:3]
    voxel = np.round(ijk).astype(int)
    assert np.abs(ijk - voxel).max() < 1e-6
    data = s.data.copy()
    data[tuple(voxel)] = 255

    size = config.panel_px
    towards = ("R", "A", "S")  # the side of each tkr axis the marker is on
    for name, panel in s.panels.items():
        _, h, v = AXES[name]
        pos = panel.slices.index(int(voxel[panel.slice_voxel_axis]))
        tile = frames.gray_tile(data, panel, pos, (0.0, 255.0))
        rows, cols = np.nonzero(tile == 255)
        # the voxel covers half a voxel either side of its centre: exactly these pixel centres
        px, py = (float(a) for a in s.pixel_of(name, marker))
        half = s.voxel_px / 2
        centres = np.arange(size) + 0.5
        assert sorted(set(cols)) == np.flatnonzero(np.abs(centres - px) < half).tolist()
        assert sorted(set(rows)) == np.flatnonzero(np.abs(centres - py) < half).tolist()
        assert rows.size == len(set(rows)) * len(set(cols)) == round(s.voxel_px) ** 2

        edges = panel.edges
        assert (edges.left == towards[h]) == (cols.mean() < size / 2)
        assert (edges.right == towards[h]) == (cols.mean() > size / 2)
        assert (edges.top == towards[v]) == (rows.mean() < size / 2)
        assert (edges.bottom == towards[v]) == (rows.mean() > size / 2)
        # and the letters are the ones the docs give for this orientation
        at_left = config.sagittal_anterior_left if name == "sagittal" else config.radiological
        assert (edges.left == towards[h]) == at_left
        assert edges.top == towards[v]
    assert not (s.data == 255).any()  # the shared fixture is untouched


# ---------------------------------------------------------------- ContourRenderer


def test_renderer_pixel_convention() -> None:
    """Pixel (c, r) covers [c, c + 1) x [r, r + 1), y down: a 1 px line along the middle of
    row 3 fills row 3 and nothing else."""
    with frames.ContourRenderer(32, 24, 1.0) as r:
        rgba = r.render([(np.array([[[5.0, 3.5], [15.0, 3.5]]]), "#00FF00")])
        assert rgba.shape == (24, 32, 4) and rgba.dtype == np.uint8
        assert np.flatnonzero(rgba[..., 3].any(axis=1)).tolist() == [3]
        assert (rgba[3, 5:15, 3] == 255).all()
        assert not rgba[3, :4, 3].any() and not rgba[3, 16:, 3].any()  # round caps: 0.5 px
        assert (rgba[3, 5:15, :3] == (0, 255, 0)).all()

        rgba = r.render([(np.array([[[28.5, 2.0], [28.5, 20.0]]]), "#0000FF")])
        assert np.flatnonzero(rgba[..., 3].any(axis=0)).tolist() == [28]
        assert (rgba[2:20, 28, 3] == 255).all()

        # a line between two rows covers each by half
        rgba = r.render([(np.array([[[5.0, 10.0], [15.0, 10.0]]]), "#FFFFFF")])
        assert np.abs(rgba[9:11, 6:14, 3].astype(int) - 127.5).max() <= 1.5


def test_renderer_draws_layers_in_order_with_exact_colours() -> None:
    across = np.array([[[2.0, 16.0], [30.0, 16.0]]])
    down = np.array([[[16.0, 2.0], [16.0, 30.0]]])
    with frames.ContourRenderer(32, 32, 4.0) as r:
        first = r.render([(across, "#12AB34"), (down, "#FE10C8")])
        second = r.render([(down, "#FE10C8"), (across, "#12AB34")])
        single = r.render([(across, "#12AB34")])
    assert tuple(first[16, 16]) == (0xFE, 0x10, 0xC8, 255)
    assert tuple(second[16, 16]) == (0x12, 0xAB, 0x34, 255)
    assert tuple(first[16, 6]) == tuple(second[16, 6]) == (0x12, 0xAB, 0x34, 255)
    # a single colour is exact wherever the line has any coverage: nothing is desaturated
    covered = single[..., 3] > 0
    assert 0 < (single[..., 3] == 255).sum() < covered.sum()
    assert (single[covered][:, :3] == (0x12, 0xAB, 0x34)).all()
    assert not single[~covered].any()
    for bad in ("red", "#12AB3", "#12AB34FF"):
        with frames.ContourRenderer(8, 8, 1.0) as r, pytest.raises(ValueError):
            r.render([(across, bad)])


def test_renderer_joins_segments_into_even_lines() -> None:
    """Many short segments along a circle give the same tile as the circle in few segments
    would: no beads where the segments meet."""
    with frames.ContourRenderer(64, 64, 1.25) as r:
        tiles = []
        for n in (60, 2000):
            t = np.linspace(0, 2 * np.pi, n + 1)
            points = np.column_stack([32 + 24 * np.cos(t), 32 + 24 * np.sin(t)])
            segments = np.stack([points[:-1], points[1:]], axis=1)
            shuffled = segments[np.random.default_rng(0).permutation(n)]
            tiles.append(r.render([(shuffled, "#FFFFFF")])[..., 3].astype(float))
            in_order = r.render([(segments, "#FFFFFF")])[..., 3]
            assert np.abs(tiles[-1] - in_order).max() <= 2  # wherever the loop is started
    # total coverage is the area of the line: circumference times width
    for tile in tiles:
        assert tile.sum() / 255 == pytest.approx(2 * np.pi * 24 * 1.25, rel=0.02)
    assert np.abs(tiles[0] - tiles[1]).max() <= 16


def test_renderer_is_stateless_between_calls(
    scene: Scene, renderer: frames.ContourRenderer
) -> None:
    one = scene.layers("coronal", 30, "pial")
    other = scene.layers("axial", 20)
    assert len(one) == 2 and len(other) == 4
    first = renderer.render(one)
    assert first[..., 3].any()
    assert not np.array_equal(renderer.render(other), first)
    assert np.array_equal(renderer.render(one), first)
    assert not renderer.render([]).any()
    assert np.array_equal(renderer.render(one), first)
    size = scene.config.panel_px
    with frames.ContourRenderer(size, size, scene.config.line_width_px) as fresh:
        assert np.array_equal(fresh.render(one), first)


def test_renderer_empty_input_and_close() -> None:
    r = frames.ContourRenderer(40, 30, 1.25)
    for layers in ([], [(np.empty((0, 2, 2)), "#FFD400")]):
        rgba = r.render(layers)
        assert rgba.shape == (30, 40, 4) and rgba.dtype == np.uint8 and not rgba.any()
    # lines that lie outside the tile leave it empty too
    assert not r.render([(np.array([[[-50.0, -5.0], [90.0, -5.0]]]), "#FFD400")]).any()
    r.close()
    r.close()
    with pytest.raises(RuntimeError):
        r.render([])
    with pytest.raises(ValueError):
        frames.ContourRenderer(0, 30, 1.25)
    with pytest.raises(ValueError):
        frames.ContourRenderer(40, 30, 0.0)


# ---------------------------------------------------------------- contours on the subject


def test_contour_layers_order_colours_and_cropping(scene: Scene) -> None:
    config = scene.config
    panel = scene.panels["axial"]
    pos = len(panel.slices) // 2
    layers = scene.layers("axial", pos)
    assert [colour for _, colour in layers] == [config.white_colour] * 2 + [config.pial_colour] * 2
    pad = config.line_width_px
    for (segments, _), name in zip(layers, SURFACES, strict=True):
        assert segments.ndim == 3 and segments.shape[1:] == (2, 2)
        # the ellipses leave the field of view at the top: those segments are dropped
        whole = scene.slicers[name].segments(panel.world_axis, panel.slice_mm[pos])
        assert 0 < len(segments) < len(whole)
        px, py = segments[..., 0], segments[..., 1]
        inside = (
            (px.max(axis=1) >= -pad)
            & (px.min(axis=1) <= panel.width + pad)
            & (py.max(axis=1) >= -pad)
            & (py.min(axis=1) <= panel.height + pad)
        )
        assert inside.all()
        # nothing visible is lost: the kept segments are those of the whole contour in the tile
        wx, wy = scene.pixel_of("axial", whole)
        visible = ((wx > 0) & (wx < panel.width) & (wy > 0) & (wy < panel.height)).any(axis=1)
        assert len(segments) >= visible.sum()

    subset = {k: scene.slicers[k] for k in ("rh_pial", "lh_white")}
    two = frames.contour_layers(panel, pos, scene.affine, subset, config)
    assert [colour for _, colour in two] == [config.white_colour, config.pial_colour]
    assert np.array_equal(two[0][0], layers[0][0]) and np.array_equal(two[1][0], layers[3][0])
    with pytest.raises(ValueError):
        frames.contour_layers(
            panel, pos, scene.affine, {"lh_pail": scene.slicers["lh_pial"]}, config
        )
    custom = config.model_copy(update={"white_colour": "#010203", "pial_colour": "#040506"})
    recoloured = frames.contour_layers(panel, pos, scene.affine, scene.slicers, custom)
    assert [colour for _, colour in recoloured] == ["#010203"] * 2 + ["#040506"] * 2


@pytest.mark.parametrize("surface", ["pial", "white"])
def test_contours_follow_the_analytic_ellipse(any_scene: Scene, surface: str) -> None:
    s = any_scene
    size = s.config.panel_px
    checked = 0
    with frames.ContourRenderer(size, size, s.config.line_width_px) as r:
        for name, panel in s.panels.items():
            for pos in spread(panel, 16):
                if s.grazing(name, pos):
                    continue
                drawn = line_pixels(r.render(s.layers(name, pos, surface)))
                ellipse = s.ellipse(name, pos, surface)
                in_tile = ((ellipse > 1) & (ellipse < size - 1)).all(axis=1)
                if not in_tile.any():
                    assert drawn.size == 0, (name, pos)
                    continue
                # every line pixel is on the ellipse, and the whole ellipse is drawn
                assert cKDTree(ellipse).query(drawn)[0].max() <= 1.5, (name, pos)
                assert cKDTree(drawn).query(ellipse[in_tile])[0].max() <= 1.5, (name, pos)
                checked += 1
    assert checked >= 30


def step_band(gray: np.ndarray, inner: int, outer: int) -> np.ndarray:
    """Pixels on either side of the step between two intensities."""
    a, b = gray == inner, gray == outer
    return (a & ndimage.binary_dilation(b)) | (b & ndimage.binary_dilation(a))


@pytest.mark.parametrize(
    ("surface", "inner", "outer"), [("pial", T1_GRAY, T1_HEAD), ("white", T1_WHITE, T1_GRAY)]
)
def test_contours_sit_on_the_intensity_step(
    any_scene: Scene, surface: str, inner: int, outer: int
) -> None:
    """Image and contour registration: the pial contour runs along the 70 -> 25 step of the
    gray tile and the white contour along the 110 -> 70 step, to within one voxel."""
    s = any_scene
    size = s.config.panel_px
    reach = s.voxel_px + 1.0  # one voxel, plus a pixel for the width of the line
    shifts = [(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4)]
    checked = 0
    with frames.ContourRenderer(size, size, s.config.line_width_px) as r:
        for name, panel in s.panels.items():
            for pos in spread(panel, 9):
                if s.grazing(name, pos):
                    continue
                gray = frames.gray_tile(s.data, panel, pos, (0.0, 255.0))
                line = r.render(s.layers(name, pos, surface))[..., 3] > 127
                band = step_band(gray, inner, outer)
                assert line.any() == band.any(), (name, pos)
                # away from the tile border, where the line or the step may continue outside
                margin = int(np.ceil(reach)) + 1
                core = np.zeros_like(band)
                core[margin:-margin, margin:-margin] = True
                if not (line & core).any():
                    continue
                # each line pixel has both intensities within reach
                to_inner = ndimage.distance_transform_edt(gray != inner)
                to_outer = ndimage.distance_transform_edt(gray != outer)
                assert to_inner[line & core].max() <= reach, (name, pos)
                assert to_outer[line & core].max() <= reach, (name, pos)
                # each step pixel has the line within reach
                to_line = ndimage.distance_transform_edt(~line)
                assert to_line[band & core].max() <= reach, (name, pos)
                # and no shift of the contour by whole pixels fits the step better than
                # staying in place, give or take a pixel: a half-voxel offset would show here
                to_band = ndimage.distance_transform_edt(~band)
                rows, cols = np.nonzero(line & core)
                cost = [to_band[rows + dy, cols + dx].mean() for dx, dy in shifts]
                best = shifts[int(np.argmin(cost))]
                assert max(abs(best[0]), abs(best[1])) <= 1, (name, pos, best)
                checked += 1
    assert checked >= 15


def test_contour_tile_colours_and_transparency(
    scene: Scene, renderer: frames.ContourRenderer
) -> None:
    config = scene.config
    pial_rgb = tuple(int(config.pial_colour[i : i + 2], 16) for i in (1, 3, 5))
    white_rgb = tuple(int(config.white_colour[i : i + 2], 16) for i in (1, 3, 5))
    assert pial_rgb == (0xFF, 0xD4, 0x00) and white_rgb == (0xFF, 0x3B, 0x30)
    full = 0
    for name, panel in scene.panels.items():
        pos = len(panel.slices) // 4
        pial = renderer.render(scene.layers(name, pos, "pial"))
        white = renderer.render(scene.layers(name, pos, "white"))
        both = renderer.render(scene.layers(name, pos))
        for rgba, rgb in ((pial, pial_rgb), (white, white_rgb)):
            covered = rgba[..., 3] > 0
            assert covered.any() and not covered.all()
            assert (rgba[covered][:, :3] == rgb).all()
            assert not rgba[~covered].any()
            full += int((rgba[..., 3] == 255).sum())
        # pial is drawn over white: where either line is opaque, the tile shows its colour,
        # except that an opaque white pixel may have pial drawn over it
        assert (both[pial[..., 3] == 255] == (*pial_rgb, 255)).all()
        only_white = (white[..., 3] == 255) & (pial[..., 3] == 0)
        assert (both[only_white] == (*white_rgb, 255)).all()
        assert np.array_equal(both[..., 3] > 0, (pial[..., 3] > 0) | (white[..., 3] > 0))
        assert not both[both[..., 3] == 0].any()
    assert full > 0


def test_a_slice_between_the_hemispheres_gives_an_empty_tile(
    scene: Scene, renderer: frames.ContourRenderer
) -> None:
    panel = scene.panels["sagittal"]
    # medial-most pial x is -5.5 on the left and -2.5 on the right
    pos = int(np.argmin(np.abs(np.array(panel.slice_mm) - scene.subject.midline_x)))
    assert -5.5 < panel.slice_mm[pos] < -2.5
    layers = scene.layers("sagittal", pos)
    assert layers == []
    rgba = renderer.render(layers)
    assert rgba.shape == (panel.height, panel.width, 4) and not rgba.any()
    # a coronal slice behind both occipital poles
    assert scene.panels["coronal"].slice_mm[0] < -72.5
    assert scene.layers("coronal", 0) == []


def test_contours_do_not_depend_on_the_voxel_grid(
    synth_subject: SyntheticSubject, synth_subject_hires: SyntheticSubject
) -> None:
    """The same anatomy on a 1 mm and a 0.8 mm grid, on slice planes both grids have."""
    config = RenderConfig(slice_step_mm=4.0)
    box = BoxInfo(centre=(-4.0, -52.0, 4.0), low=(-36.0, -76.0, -28.0), high=(28.0, -28.0, 36.0))
    coarse, fine = (
        make_scene(synth_subject, config, box),
        make_scene(synth_subject_hires, config, box),
    )
    size = config.panel_px
    drawn = 0
    with frames.ContourRenderer(size, size, config.line_width_px) as r:
        for name in PANEL_ORDER:
            a_panel, b_panel = coarse.panels[name], fine.panels[name]
            np.testing.assert_allclose(a_panel.slice_mm, b_panel.slice_mm, atol=1e-4)
            assert len(a_panel.slices) >= 13
            for pos in range(len(a_panel.slices)):
                if coarse.grazing(name, pos):
                    continue  # a plane through a pole vertex on one grid can miss it on the other
                a = r.render(coarse.layers(name, pos))[..., 3]
                b = r.render(fine.layers(name, pos))[..., 3]
                near = np.ones((3, 3), dtype=bool)  # within one pixel
                assert not ((a > 127) & ~ndimage.binary_dilation(b > 127, near)).any()
                assert not ((b > 127) & ~ndimage.binary_dilation(a > 127, near)).any()
                assert np.abs(a.astype(int) - b).max() <= 8
                drawn += int((a > 127).sum())
    assert drawn > 10000


# ---------------------------------------------------------------- blend


def test_blend() -> None:
    rng = np.random.default_rng(0)
    gray = rng.integers(0, 256, size=(20, 30), dtype=np.uint8)
    contours = rng.integers(0, 256, size=(20, 30, 4), dtype=np.uint8)
    contours[:5, :, 3] = 0
    contours[5:10, :, 3] = 255
    out = frames.blend(gray, contours)
    assert out.shape == (20, 30, 3) and out.dtype == np.uint8
    assert (out[:5] == gray[:5, :, None]).all()
    assert np.array_equal(out[5:10], contours[5:10, :, :3])
    alpha = contours[..., 3:] / 255
    exact = contours[..., :3] * alpha + gray[..., None] * (1 - alpha)
    assert np.abs(out - exact).max() <= 0.5 + 1e-9
    assert tuple(
        frames.blend(np.full((1, 1), 100, np.uint8), np.array([[[200, 0, 50, 128]]], np.uint8))[
            0, 0
        ]
    ) == (150, 50, 75)


# ---------------------------------------------------------------- composite_strip


def uniform_tiles(size: int) -> dict[str, np.ndarray]:
    return {
        name: np.full((size, size, 3), level, dtype=np.uint8)
        for name, level in zip(PANEL_ORDER, (60, 100, 140), strict=True)
    }


def expected_crosshair(s: Scene, frame: Frame) -> dict[str, tuple[float, float]]:
    """Where the three current slice planes meet, in pixel coordinates of each panel. The
    vertical line is the column that holds px, the horizontal line the row that holds py."""
    point = np.array(
        [s.panels[name].slice_mm[pos] for name, pos in zip(PANEL_ORDER, frame, strict=True)]
    )
    out = {}
    for name in PANEL_ORDER:
        px, py = (float(a) for a in s.pixel_of(name, point))
        # on a pixel boundary the last bit of the arithmetic would decide
        assert min(px % 1, 1 - px % 1, py % 1, 1 - py % 1) > 0.02, (name, px, py)
        out[name] = (px, py)
    return out


# With the default 10 mm margin and 1 mm voxels every slice plane of the synthetic subject
# falls on a pixel boundary along y and z; these margins avoid that.
STRIP_CONFIGS = [
    *(c.model_copy(update={"margin_mm": 9.0}) for c in ORIENTATIONS),
    RenderConfig(panel_px=250, margin_mm=7.0),
]
strip_ids = [*orientation_ids, "250px"]
STRIP_FRAME = (7, 40, 22)


@pytest.mark.parametrize("config", STRIP_CONFIGS, ids=strip_ids)
def test_composite_strip_layout_and_crosshair(
    synth_subject: SyntheticSubject, config: RenderConfig
) -> None:
    s = make_scene(synth_subject, config)
    size = config.panel_px
    tiles = uniform_tiles(size)
    strip = frames.composite_strip(tiles, s.panels, STRIP_FRAME, config)
    assert strip.shape == (size, 3 * size, 3) and strip.dtype == np.uint8
    colour = np.array([0x33, 0xC7, 0xFF])
    keep = 16  # the letters stay within this distance of the edges
    crosshair = expected_crosshair(s, STRIP_FRAME)
    fractions = [a % 1 for pair in crosshair.values() for a in pair]
    assert max(fractions) > 0.5  # so rounding to nearest instead would show
    for i, name in enumerate(PANEL_ORDER):
        level = int(tiles[name][0, 0, 0])
        part = strip[:, i * size : (i + 1) * size]
        col, row = (int(np.floor(a)) for a in crosshair[name])
        assert keep < col < size - keep and keep < row < size - keep
        blended = np.floor(
            level * (1 - config.crosshair_alpha) + colour * config.crosshair_alpha + 0.5
        )
        inner = part[keep:-keep, keep:-keep]
        on = np.zeros(inner.shape[:2], dtype=bool)
        on[:, col - keep] = True
        on[row - keep, :] = True
        assert (inner[on] == blended).all(), name
        assert (inner[~on] == level).all(), name
        # the lines run from edge to edge
        assert (part[0, col] == blended).all() and (part[-1, col] == blended).all()
        assert (part[row, 0] == blended).all() and (part[row, -1] == blended).all()
    assert all(
        (tiles[name] == level).all()
        for name, level in zip(PANEL_ORDER, (60, 100, 140), strict=True)
    )


def test_composite_strip_crosshair_colour_and_alpha(synth_subject: SyntheticSubject) -> None:
    s = make_scene(synth_subject, STRIP_CONFIGS[0])
    size = s.config.panel_px
    col, row = (int(np.floor(a)) for a in expected_crosshair(s, STRIP_FRAME)["coronal"])
    for alpha, expected in ((1.0, (10, 200, 30)), (0.0, (100, 100, 100)), (0.5, (55, 150, 65))):
        config = s.config.model_copy(
            update={"crosshair_colour": "#0AC81E", "crosshair_alpha": alpha}
        )
        strip = frames.composite_strip(uniform_tiles(size), s.panels, STRIP_FRAME, config)
        assert tuple(strip[row, size + col]) == expected  # the crossing is blended once
        assert tuple(strip[row + 20, size + col]) == expected
        assert tuple(strip[row, size + col + 20]) == expected
        assert tuple(strip[row + 20, size + col + 20]) == (100, 100, 100)


def test_composite_strip_crosshair_on_a_pixel_boundary(scene: Scene) -> None:
    """With the default margin the planes of the 1 mm subject fall on pixel boundaries along
    y and z; pixel c covers [c, c + 1), so the line is the pixel that starts there."""
    size = scene.config.panel_px
    frame = (7, 40, 22)
    point = np.array([scene.panels[n].slice_mm[i] for n, i in zip(PANEL_ORDER, frame, strict=True)])
    px, py = (float(a) for a in scene.pixel_of("sagittal", point))
    assert abs(px - round(px)) < 1e-9 and abs(py - round(py)) < 1e-9
    strip = frames.composite_strip(uniform_tiles(size), scene.panels, frame, scene.config)
    changed = (strip[:, :size] != 60).any(axis=2)
    assert changed[40:-40, round(px)].all() and changed[round(py), 40:-40].all()
    assert changed[40:-40, 40:-40].sum() == 2 * (size - 80) - 1


def glyph(letter: str) -> np.ndarray:
    """The letter in Pillow's built-in bitmap font, cropped to its ink."""
    font = getattr(ImageFont, "load_default_imagefont", ImageFont.load_default)()
    mask = font.getmask(letter)
    ink = np.array(mask).reshape(mask.size[::-1]) > 0
    rows, cols = np.nonzero(ink)
    return ink[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


@pytest.mark.parametrize("config", ORIENTATIONS, ids=orientation_ids)
def test_composite_strip_letters_follow_the_orientation(
    synth_subject: SyntheticSubject, config: RenderConfig
) -> None:
    s = make_scene(synth_subject, config)
    size = config.panel_px
    strip = frames.composite_strip(uniform_tiles(size), s.panels, STRIP_FRAME, config)
    left_right = ("R", "L") if config.radiological else ("L", "R")
    expected = {
        "sagittal": (*(("A", "P") if config.sagittal_anterior_left else ("P", "A")), "S", "I"),
        "coronal": (*left_right, "S", "I"),
        "axial": (*left_right, "A", "P"),
    }
    half, near = size // 2, 20
    regions = (  # left, right, top, bottom: near the middle of each edge
        (slice(half - near, half + near), slice(0, near)),
        (slice(half - near, half + near), slice(size - near, size)),
        (slice(0, near), slice(half - near, half + near)),
        (slice(size - near, size), slice(half - near, half + near)),
    )
    for i, name in enumerate(PANEL_ORDER):
        part = strip[:, i * size : (i + 1) * size]
        white = (part == 255).all(axis=2)
        assert white.sum() == sum(glyph(letter).sum() for letter in expected[name])
        for letter, region in zip(expected[name], regions, strict=True):
            rows, cols = np.nonzero(white[region])
            assert rows.size, (name, letter)
            found = white[region][rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
            assert np.array_equal(found, glyph(letter)), (name, letter)
            # legible on any background: every pixel next to the letter is black
            around = ndimage.binary_dilation(white, np.ones((3, 3), dtype=bool)) & ~white
            assert not part[around].any()


def golden_strip(subject: SyntheticSubject) -> np.ndarray:
    config = RenderConfig(panel_px=96)
    s = make_scene(subject, config)
    frame = (10, 30, 31)
    tiles = {}
    with frames.ContourRenderer(96, 96, config.line_width_px) as r:
        for name, pos in zip(PANEL_ORDER, frame, strict=True):
            gray = frames.gray_tile(s.data, s.panels[name], pos, (0.0, 140.0))
            tiles[name] = frames.blend(gray, r.render(s.layers(name, pos)))
    expected_crosshair(s, frame)  # no line on a pixel boundary, where platforms could differ
    return frames.composite_strip(tiles, s.panels, frame, config)


def test_composite_strip_golden_image(synth_subject: SyntheticSubject) -> None:
    """The whole chain on the 1 mm synthetic subject at 96 px per panel, against a stored PNG.

    Regenerate with SURFCENSUS_REGENERATE_GOLDEN=1 and look at the image before committing it.
    """
    strip = golden_strip(synth_subject)
    assert strip.shape == (96, 288, 3)
    if os.environ.get("SURFCENSUS_REGENERATE_GOLDEN") == "1":
        GOLDEN.parent.mkdir(exist_ok=True)
        Image.fromarray(strip).save(GOLDEN, optimize=True)
        pytest.skip(f"regenerated {GOLDEN}")
    with Image.open(GOLDEN) as image:
        golden = np.array(image.convert("RGB"))
    assert golden.shape == strip.shape
    difference = np.abs(strip.astype(int) - golden)
    assert difference.mean() <= 2
    assert (difference.max(axis=2) > 40).mean() < 0.01
    assert np.array_equal(strip, golden_strip(synth_subject))  # and it is reproducible


# ---------------------------------------------------------------- timing


def test_time_per_tile(scene: Scene, renderer: frames.ContourRenderer) -> None:
    """Reports the time per 320 x 320 tile; the bounds are only a guard."""
    panel = scene.panels["coronal"]
    positions = spread(panel, 20)
    start = time.perf_counter()
    for pos in positions:
        frames.gray_tile(scene.data, panel, pos, (0.0, 140.0))
    gray = (time.perf_counter() - start) / len(positions)
    layers = [scene.layers("coronal", pos) for pos in positions]
    start = time.perf_counter()
    for one in layers:
        renderer.render(one)
    render = (time.perf_counter() - start) / len(positions)
    segments = np.mean([sum(len(seg) for seg, _ in one) for one in layers])
    print(
        f"\ngray_tile {gray * 1e3:.2f} ms, ContourRenderer.render {render * 1e3:.2f} ms per tile "
        f"({segments:.0f} segments per tile)"
    )
    assert gray < 0.02 and render < 0.15
