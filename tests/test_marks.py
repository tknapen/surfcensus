"""Tests of app/marks.py: from a click on a tile to voxels, tkr RAS, scanner RAS and vertices.

Expected values are worked out here, never with the code under test or with the vertex map:

* where a tile pixel lies in tkr RAS, from the rules of docs/rendering.md applied to the
  surfaces and to the header of ``T1.mgz`` as nibabel reads them (:func:`tile_layout`,
  :func:`click_point`);
* the nearest vertices, by computing every distance on the full surfaces
  (:func:`nearest_brute`), and the hemisphere rule of docs/dev/contract.md 7.3 written out
  again (:func:`expected_marks`);
* the template vertex of a native vertex, as the nearest template direction to its place
  on ``sphere.reg``, every distance computed.

The synthetic subjects are those of ``tests/synth.py`` (two ellipsoids whose pial surfaces
lie 3 mm apart at the midline). The last tests read ``ds000102`` sub-01 and the
``fsaverage`` next to it where the network tests have put them, and skip otherwise.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import nibabel.freesurfer as fs
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image
from scipy.spatial import cKDTree
from synth import SyntheticSubject, write_synthetic_subject  # pyrefly: ignore[missing-import]
from synth_template import write_synthetic_template  # pyrefly: ignore[missing-import]
from synth_tierb import add_tier_b  # pyrefly: ignore[missing-import]

from surfcensus.app.marks import (
    MARK_FIELDS,
    MarkError,
    VertexIndex,
    check_snap_radius,
    marked_vertices,
    resolve_mark,
    validate_mark,
)
from surfcensus.app.schemas import MarkIn, ResolvedMark
from surfcensus.render import geometry, render_subject
from surfcensus.render.model import PANEL_ORDER, RenderConfig, RenderInfo
from surfcensus.render.vertexmap import read_vertex_map
from surfcensus.sources.base import HEMIS
from surfcensus.sources.patterns import resolve_subject_files
from surfcensus.templates import build_template

# (tkr axis normal to the slices, along the horizontal, along the vertical), docs/rendering.md
AXES = {"sagittal": (0, 1, 2), "coronal": (1, 0, 2), "axial": (2, 0, 1)}
SURFACES = ("pial", "white")
CLASSES = ("A", "B", "C", "X")
FAR = (500.0, 0.0, 0.0)
ATOL = 1e-9

# --------------------------------------------------------------- hand-made vertex maps

Vertex = tuple[int, tuple[float, float, float], tuple[float, float, float], int]
"""(native index, pial position, white position, template vertex)."""


def vmap_of(lh: Sequence[Vertex] = (), rh: Sequence[Vertex] = ()) -> dict[str, np.ndarray]:
    """A vertex map with the arrays and dtypes of ``vertex_map.npz``."""
    vmap: dict[str, np.ndarray] = {}
    for hemi, rows in (("lh", lh), ("rh", rh)):
        vmap[f"{hemi}_index"] = np.array([r[0] for r in rows], dtype="<i4")
        vmap[f"{hemi}_pial"] = np.array([r[1] for r in rows], dtype="<f4").reshape(-1, 3)
        vmap[f"{hemi}_white"] = np.array([r[2] for r in rows], dtype="<f4").reshape(-1, 3)
        vmap[f"{hemi}_fsavg6"] = np.array([r[3] for r in rows], dtype="<i4")
        vmap[f"{hemi}_visible"] = np.ones(len(rows), dtype=bool)
    return vmap


def random_vmap(seed: int, n: int = 300) -> dict[str, np.ndarray]:
    """Vertices on a grid of half millimetres in a cube of 10 mm, so that many lie at
    exactly the same distance from a grid point, some at the same place, and none in order
    of their native index."""
    rng = np.random.default_rng(seed)

    def rows(count: int) -> list[Vertex]:
        native = rng.permutation(5 * count)[:count]
        pial = rng.integers(0, 21, size=(count, 3)) / 2
        white = rng.integers(0, 21, size=(count, 3)) / 2
        fsavg = rng.integers(-1, 50, size=count)
        return [
            (int(v), tuple(p), tuple(w), int(f))
            for v, p, w, f in zip(native, pial.tolist(), white.tolist(), fsavg, strict=True)
        ]

    return vmap_of(rows(n), rows(n + 17))


def brute_within(
    vmap: Mapping[str, np.ndarray],
    surface: str,
    point: Sequence[float],
    radius: float,
    hemis: Sequence[str] = HEMIS,
) -> list[tuple[str, int, float, int]]:
    """Every vertex within the radius, each distance computed: by distance, hemisphere,
    native index."""
    found = []
    for hemi in hemis:
        xyz = vmap[f"{hemi}_{surface}"].astype(float)
        dist = np.sqrt(((xyz - np.array(point, dtype=float)) ** 2).sum(axis=1))
        found += [
            (float(d), HEMIS.index(hemi), int(v), int(f))
            for d, v, f in zip(dist, vmap[f"{hemi}_index"], vmap[f"{hemi}_fsavg6"], strict=True)
            if d <= radius
        ]
    return [(HEMIS[h], v, d, f) for d, h, v, f in sorted(found)]


def test_nearest_names_hemisphere_vertex_distance_and_template_vertex() -> None:
    index = VertexIndex(
        vmap_of(
            lh=[(7, (0, 0, 0), (0, 0, 2), 70), (9, (3, 0, 0), (3, 0, 2), 90)],
            rh=[(4, (10, 0, 0), (9, 0, 0), -1)],
        )
    )
    assert index.n_vertices == {"lh": 2, "rh": 1} and index.has_counterparts
    assert index.nearest("pial", (2.0, 0.0, 0.0), 5.0) == ("lh", 9, 1.0, 90)
    assert index.nearest("pial", (1.0, 0.0, 0.0), 5.0) == ("lh", 7, 1.0, 70)
    assert index.nearest("white", (3.0, 0.0, 0.0), 5.0) == ("lh", 9, 2.0, 90)
    assert index.nearest("pial", (8.0, 0.0, 0.0), 5.0) == ("rh", 4, 2.0, -1)
    assert index.nearest("white", (8.0, 0.0, 0.0), 5.0) == ("rh", 4, 1.0, -1)
    # restricted to a hemisphere, the nearer vertex of the other one does not count
    assert index.nearest("pial", (8.0, 0.0, 0.0), 5.0, hemi="lh") == ("lh", 9, 5.0, 90)
    assert index.nearest("pial", (8.0, 0.0, 0.0), 4.0, hemi="lh") is None
    assert index.nearest("pial", (0.0, 40.0, 0.0), 5.0) is None


def test_the_radius_includes_its_boundary() -> None:
    index = VertexIndex(vmap_of(lh=[(3, (5, 0, 0), FAR, 1)], rh=[(8, (0, 0, -3), FAR, 2)]))
    origin = (0.0, 0.0, 0.0)
    assert index.nearest("pial", origin, 5.0, hemi="lh") == ("lh", 3, 5.0, 1)
    assert index.nearest("pial", origin, math.nextafter(5.0, 0.0), hemi="lh") is None
    assert index.nearest("pial", origin, 3.0) == ("rh", 8, 3.0, 2)
    assert index.nearest("pial", origin, math.nextafter(3.0, 0.0)) is None
    assert index.within("pial", origin, 5.0) == [("rh", 8, 3.0, 2), ("lh", 3, 5.0, 1)]
    assert index.within("pial", origin, math.nextafter(5.0, 0.0)) == [("rh", 8, 3.0, 2)]
    assert index.within("pial", origin, math.nextafter(3.0, 0.0)) == []
    # a radius of 0 finds the vertex at the point itself
    assert index.nearest("pial", (5.0, 0.0, 0.0), 0.0) == ("lh", 3, 0.0, 1)
    assert marked_vertices(index, [origin], 5.0)["lh"].tolist() == [3]
    assert marked_vertices(index, [origin], math.nextafter(5.0, 0.0))["lh"].tolist() == []


def test_equally_near_vertices_go_by_lh_first_and_then_by_native_index() -> None:
    # four vertices at distance 2 from the origin, listed in no order
    lh = [(12, (0, 2, 0), FAR, 1), (5, (2, 0, 0), FAR, 2), (30, (0, 0, -2), FAR, 3)]
    rh = [(1, (-2, 0, 0), FAR, 4), (2, (0, 0, 1), FAR, 5)]
    origin = (0.0, 0.0, 0.0)
    index = VertexIndex(vmap_of(lh, rh))
    assert index.nearest("pial", origin, 5.0) == ("rh", 2, 1.0, 5)
    assert index.nearest("pial", origin, 5.0, hemi="lh") == ("lh", 5, 2.0, 2)
    assert index.within("pial", origin, 2.0) == [
        ("rh", 2, 1.0, 5),
        ("lh", 5, 2.0, 2),
        ("lh", 12, 2.0, 1),
        ("lh", 30, 2.0, 3),
        ("rh", 1, 2.0, 4),
    ]
    # without the nearer right vertex: lh on the exact tie, though rh has the lower index
    tie = VertexIndex(vmap_of(lh, rh[:1]))
    assert tie.nearest("pial", origin, 5.0) == ("lh", 5, 2.0, 2)
    assert tie.nearest("pial", origin, 5.0, hemi="rh") == ("rh", 1, 2.0, 4)
    # two vertices at one place
    twins = VertexIndex(vmap_of(lh=[(8, (1, 1, 1), FAR, 0), (3, (1, 1, 1), FAR, 9)]))
    assert twins.nearest("pial", (1.0, 1.0, 2.0), 5.0) == ("lh", 3, 1.0, 9)


def test_a_hemisphere_without_vertices_is_searched_without_error() -> None:
    empty = VertexIndex(vmap_of())
    assert empty.n_vertices == {"lh": 0, "rh": 0} and not empty.has_counterparts
    assert empty.nearest("pial", (0.0, 0.0, 0.0), 5.0) is None
    assert empty.within("white", (0.0, 0.0, 0.0), 5.0) == []
    found = marked_vertices(empty, [(0.0, 0.0, 0.0)], 5.0)
    assert [found[h].tolist() for h in HEMIS] == [[], []]
    one = VertexIndex(vmap_of(rh=[(4, (1, 0, 0), (2, 0, 0), -1)]))
    assert not one.has_counterparts
    assert one.nearest("pial", (0.0, 0.0, 0.0), 5.0) == ("rh", 4, 1.0, -1)
    assert one.nearest("pial", (0.0, 0.0, 0.0), 5.0, hemi="lh") is None


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nearest_and_within_equal_a_search_over_every_vertex(seed: int) -> None:
    vmap = random_vmap(seed)
    index = VertexIndex(vmap)
    rng = np.random.default_rng(100 + seed)
    # on the grid of the vertices (exact ties) and off it
    points = np.concatenate(
        [rng.integers(-2, 23, size=(120, 3)) / 2, rng.uniform(-1.0, 11.0, size=(120, 3))]
    )
    n_ties = 0
    for point in points.tolist():
        p = (point[0], point[1], point[2])
        for surface in SURFACES:
            for radius in (0.5, 1.5, 3.0):
                for hemis in (HEMIS, ("lh",), ("rh",)):
                    hemi = None if len(hemis) == 2 else hemis[0]
                    expected = brute_within(vmap, surface, p, radius, hemis)
                    assert index.within(surface, p, radius, hemi=hemi) == expected  # type: ignore[arg-type]
                    nearest = index.nearest(surface, p, radius, hemi=hemi)  # type: ignore[arg-type]
                    assert nearest == (expected[0] if expected else None)
                    n_ties += len(expected) > 1 and expected[0][2] == expected[1][2]
    assert n_ties > 200  # the tie rules were asked for


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda m: m.pop("rh_fsavg6"), "no array rh_fsavg6"),
        (lambda m: m.pop("lh_pial"), "no array lh_pial"),
        (lambda m: m.update(lh_pial=m["lh_pial"][:, :2]), "lh_pial"),
        (lambda m: m.update(lh_white=m["lh_white"][:-1]), "lh_white"),
        (lambda m: m.update(rh_index=m["rh_index"].reshape(-1, 1)), "rh_index"),
        (lambda m: m.update(rh_index=m["rh_index"].astype(float)), "rh_index"),
        (lambda m: m.update(lh_fsavg6=m["lh_fsavg6"][:-1]), "lh_fsavg6"),
        (lambda m: m.update(lh_pial=m["lh_pial"].astype(int)), "lh_pial"),
        (lambda m: m["rh_white"].__setitem__((0, 1), np.nan), "rh_white"),
        (lambda m: m["lh_pial"].__setitem__((1, 2), np.inf), "lh_pial"),
    ],
)
def test_arrays_that_are_no_vertex_map_are_refused(
    change: Callable[[dict[str, np.ndarray]], Any], message: str
) -> None:
    vmap = random_vmap(5, n=6)
    VertexIndex(vmap)
    change(vmap)
    with pytest.raises(ValueError, match=message):
        VertexIndex(vmap)


def test_a_point_and_a_radius_must_be_finite() -> None:
    index = VertexIndex(random_vmap(6, n=6))
    for point in ((math.nan, 0.0, 0.0), (0.0, math.inf, 0.0)):
        with pytest.raises(ValueError, match="finite"):
            index.nearest("pial", point, 5.0)
        with pytest.raises(ValueError, match="finite"):
            index.within("pial", point, 5.0)
        with pytest.raises(ValueError, match="finite"):
            marked_vertices(index, [(0.0, 0.0, 0.0), point], 5.0)
    for radius in (-1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="distance"):
            index.nearest("pial", (0.0, 0.0, 0.0), radius)
        with pytest.raises(ValueError, match="distance"):
            index.within("pial", (0.0, 0.0, 0.0), radius)
        with pytest.raises(ValueError, match="distance"):
            marked_vertices(index, [(0.0, 0.0, 0.0)], radius)


@pytest.mark.parametrize("surface", SURFACES)
def test_marked_vertices_are_those_within_the_tolerance_of_any_point(surface: str) -> None:
    vmap = random_vmap(7)
    index = VertexIndex(vmap)
    rng = np.random.default_rng(8)
    places = np.concatenate([rng.integers(0, 21, size=(4, 3)) / 2, vmap[f"lh_{surface}"][:1]])
    points = [(float(x), float(y), float(z)) for x, y, z in places]  # one of them on a vertex
    for tolerance in (0.0, 1.0, 2.5):
        found = marked_vertices(index, points, tolerance, surface=surface)  # type: ignore[arg-type]
        assert sorted(found) == ["lh", "rh"]
        n_single = 0
        for hemi in HEMIS:
            xyz = vmap[f"{hemi}_{surface}"].astype(float)
            dist = np.sqrt(((xyz[:, None, :] - np.array(points)[None, :, :]) ** 2).sum(axis=2))
            expected = np.unique(vmap[f"{hemi}_index"][(dist <= tolerance).any(axis=1)])
            assert found[hemi].dtype == np.int32 and found[hemi].ndim == 1
            assert found[hemi].tolist() == expected.tolist()  # ascending, each once
            n_single += int((dist <= tolerance).any(axis=1).sum())
        assert n_single > {0.0: 0, 1.0: 5, 2.5: 40}[tolerance]
    # one point, the vertices of within(); no point, no vertex
    alone = marked_vertices(index, points[:1], 2.5, surface=surface)  # type: ignore[arg-type]
    for hemi in HEMIS:
        within = index.within(surface, points[0], 2.5, hemi=hemi)  # type: ignore[arg-type]
        assert alone[hemi].tolist() == sorted(v for _, v, _, _ in within)
    none = marked_vertices(index, [], 2.5, surface=surface)  # type: ignore[arg-type]
    assert [none[h].tolist() for h in HEMIS] == [[], []] and none["lh"].dtype == np.int32


def test_the_snap_radius_may_not_exceed_the_margin_of_the_vertex_map() -> None:
    check_snap_radius(5.0, 5.0)
    check_snap_radius(0.0, 0.0)
    check_snap_radius(2.0, 5.0)
    with pytest.raises(ValueError, match=r"snap radius of marks \(5\.5 mm\).*= 5 mm"):
        check_snap_radius(5.5, 5.0)
    with pytest.raises(ValueError, match="snap radius"):
        check_snap_radius(math.nextafter(5.0, 6.0), 5.0)
    for radius in (-1.0, math.nan, math.inf):
        with pytest.raises(ValueError):
            check_snap_radius(radius, 5.0)


# --------------------------------------------- the anatomy, read with nibabel alone


@dataclass(frozen=True)
class Anatomy:
    """What the files of a subject say: the two affines of ``T1.mgz`` and the surfaces."""

    vox2ras_tkr: np.ndarray
    vox2ras: np.ndarray
    c_ras: np.ndarray
    shape: tuple[int, ...]
    pial: dict[str, np.ndarray]
    white: dict[str, np.ndarray]
    pial_faces: dict[str, np.ndarray]


def read_anatomy(subject: Path, pial: str = "pial") -> Anatomy:
    header = nib.load(str(subject / "mri" / "T1.mgz")).header  # pyrefly: ignore
    read = {
        (hemi, name): fs.read_geometry(str(subject / "surf" / f"{hemi}.{name}"))
        for hemi in HEMIS
        for name in (pial, "white")
    }
    return Anatomy(
        vox2ras_tkr=np.asarray(header.get_vox2ras_tkr(), dtype=float),
        vox2ras=np.asarray(header.get_vox2ras(), dtype=float),
        c_ras=np.asarray(header["Pxyz_c"], dtype=float),
        shape=tuple(int(n) for n in header.get_data_shape()[:3]),
        pial={hemi: read[hemi, pial][0].astype(float) for hemi in HEMIS},
        white={hemi: read[hemi, "white"][0].astype(float) for hemi in HEMIS},
        pial_faces={hemi: read[hemi, pial][1] for hemi in HEMIS},
    )


@dataclass(frozen=True)
class Layout:
    """Where the tiles of a render lie in tkr RAS."""

    low: np.ndarray
    """The view box."""
    high: np.ndarray
    fov_low: np.ndarray
    """Low corner of the cube that the panels show; its side is ``side``."""
    side: float
    planes: dict[str, np.ndarray]
    """Panel -> tkr coordinate of each slice plane, low to high."""


def tile_layout(anatomy: Anatomy, config: RenderConfig) -> Layout:
    """docs/rendering.md, "Geometry": pole, box, slices and field of view, from the pial
    surfaces and the tkr affine."""
    pole = {hemi: anatomy.pial[hemi][np.argmin(anatomy.pial[hemi][:, 1])] for hemi in HEMIS}
    cy = (pole["lh"][1] + pole["rh"][1]) / 2
    cz = (pole["lh"][2] + pole["rh"][2]) / 2
    offsets = np.array([config.box_x_mm, config.box_y_mm, config.box_z_mm], dtype=float)

    def medial_x(hemi: str) -> np.ndarray:
        xyz = anatomy.pial[hemi]
        in_y = (xyz[:, 1] >= cy + offsets[1, 0]) & (xyz[:, 1] <= cy + offsets[1, 1])
        in_z = (xyz[:, 2] >= cz + offsets[2, 0]) & (xyz[:, 2] <= cz + offsets[2, 1])
        return xyz[in_y & in_z, 0]

    left, right = medial_x("lh"), medial_x("rh")
    if len(left) and len(right):
        cx = (left.max() + right.min()) / 2
    else:
        cx = (pole["lh"][0] + pole["rh"][0]) / 2
    centre = np.array([cx, cy, cz])
    low, high = centre + offsets[:, 0], centre + offsets[:, 1]
    side = float((high - low).max() + 2 * config.margin_mm)
    planes = {}
    for panel, (w, _, _) in AXES.items():
        column = int(np.argmax(np.abs(anatomy.vox2ras_tkr[w, :3])))
        step, origin = anatomy.vox2ras_tkr[w, column], anatomy.vox2ras_tkr[w, 3]
        voxels: list[int] = []
        k = 0
        while low[w] + k * config.slice_step_mm <= high[w] + 1e-9:
            # the nearest voxel centre; an exact half goes to the higher voxel index
            voxel = math.floor((low[w] + k * config.slice_step_mm - origin) / step + 0.5)
            if 0 <= voxel < anatomy.shape[column] and voxel not in voxels[-1:]:
                voxels.append(voxel)
            k += 1
        planes[panel] = origin + step * np.array(voxels, dtype=float)
    return Layout(low=low, high=high, fov_low=(low + high) / 2 - side / 2, side=side, planes=planes)


def high_at_left(config: RenderConfig, panel: str) -> bool:
    """Whether the high end of the panel's horizontal tkr axis is at the left edge: anterior
    in the sagittal panel with ``sagittal_anterior_left``, the subject's right in the other
    two in the radiological view."""
    return config.sagittal_anterior_left if panel == "sagittal" else config.radiological


def click_point(
    layout: Layout, config: RenderConfig, panel: str, slice_pos: int, px: float, py: float
) -> np.ndarray:
    """tkr RAS of tile pixel ``(px, py)``: origin at the top-left corner, px right, py
    down, the tile spanning the field of view; superior (anterior in the axial panel) up."""
    w, h, v = AXES[panel]
    along, down = px / config.panel_px * layout.side, py / config.panel_px * layout.side
    point = np.empty(3)
    point[w] = layout.planes[panel][slice_pos]
    if high_at_left(config, panel):
        point[h] = layout.fov_low[h] + layout.side - along
    else:
        point[h] = layout.fov_low[h] + along
    point[v] = layout.fov_low[v] + layout.side - down
    return point


def pixel_of(layout: Layout, config: RenderConfig, panel: str, point: np.ndarray) -> np.ndarray:
    """The inverse of :func:`click_point` within the tile's plane: ``(px, py)`` of points."""
    _, h, v = AXES[panel]
    along = (point[..., h] - layout.fov_low[h]) / layout.side * config.panel_px
    if high_at_left(config, panel):
        along = config.panel_px - along
    down = config.panel_px - (point[..., v] - layout.fov_low[v]) / layout.side * config.panel_px
    return np.stack([along, down], axis=-1)


def nearest_brute(
    points: np.ndarray, vertices: np.ndarray, reach: float
) -> tuple[np.ndarray, np.ndarray]:
    """For each point the nearest vertex within ``reach`` (-1 when there is none) and its
    distance, every distance computed; the lowest index among equally near vertices. A
    vertex outside the bounding box of the points grown by ``reach`` is farther than
    ``reach`` from each of them and is left out before the distances."""
    index = np.full(len(points), -1, dtype=np.int64)
    dist = np.full(len(points), np.inf)
    low, high = points.min(axis=0) - reach, points.max(axis=0) + reach
    keep = np.flatnonzero(((vertices >= low) & (vertices <= high)).all(axis=1))
    if not len(keep):
        return index, dist
    near = vertices[keep]
    for start in range(0, len(points), 32):
        d = points[start : start + 32, None, :] - near[None, :, :]
        d2 = d[..., 0] ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2
        best = d2.argmin(axis=1)
        found = np.sqrt(d2[np.arange(len(best)), best])
        ok = found <= reach
        index[start : start + 32][ok] = keep[best[ok]]
        dist[start : start + 32][ok] = found[ok]
    return index, dist


@dataclass(frozen=True)
class Expected:
    hemi: str | None
    pial_vertex: int | None
    pial_dist: float | None
    white_vertex: int | None
    white_dist: float | None
    source: str | None
    """The surface whose vertex gives the template vertex."""
    other_hemi_is_nearer: bool
    """The second surface has a nearer vertex in the other hemisphere, which is not taken."""

    @property
    def source_vertex(self) -> int | None:
        return {None: None, "pial": self.pial_vertex, "white": self.white_vertex}[self.source]


def expected_marks(
    anatomy: Anatomy, points: np.ndarray, classes: Sequence[str], reach: float
) -> list[Expected]:
    """Contract 7.3, "One hemisphere per mark", on the full surfaces."""
    near = {
        (surface, hemi): nearest_brute(points, getattr(anatomy, surface)[hemi], reach)
        for surface in SURFACES
        for hemi in HEMIS
    }

    def best(i: int, surface: str, hemis: Sequence[str]) -> tuple[str, int, float] | None:
        found = [
            (float(near[surface, hemi][1][i]), HEMIS.index(hemi))
            for hemi in hemis
            if near[surface, hemi][0][i] >= 0
        ]
        if not found:
            return None
        dist, h = min(found)  # the nearer hemisphere, lh on a tie
        return HEMIS[h], int(near[surface, HEMIS[h]][0][i]), dist

    out = []
    for i, mark_class in enumerate(classes):
        own, other = ("white", "pial") if mark_class == "C" else ("pial", "white")
        first = best(i, own, HEMIS)
        if first is not None:
            second, anywhere = best(i, other, (first[0],)), best(i, other, HEMIS)
            hemi, source = first[0], own
            elsewhere = anywhere is not None and anywhere[0] != hemi
        else:
            second = best(i, other, HEMIS)
            hemi, source = (None, None) if second is None else (second[0], other)
            elsewhere = False
        found = {own: first, other: second}
        pial, white = found["pial"], found["white"]
        out.append(
            Expected(
                hemi=hemi,
                pial_vertex=None if pial is None else pial[1],
                pial_dist=None if pial is None else pial[2],
                white_vertex=None if white is None else white[1],
                white_dist=None if white is None else white[2],
                source=source,
                other_hemi_is_nearer=elsewhere,
            )
        )
    return out


def nearest_direction(directions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """For each unit vector the nearest of the target unit vectors, every distance computed."""
    out = np.empty(len(directions), dtype=np.int64)
    for start in range(0, len(directions), 256):
        d = directions[start : start + 256, None, :] - targets[None, :, :]
        out[start : start + 256] = (d[..., 0] ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2).argmin(axis=1)
    return out


def unit(xyz: np.ndarray) -> np.ndarray:
    xyz = xyz.astype(float)
    return xyz / np.sqrt((xyz**2).sum(axis=1, keepdims=True))


class Spheres:
    """The template vertex of a native vertex, from the two ``sphere.reg`` files: the
    nearest of the first ``n_target`` template vertices on the unit sphere."""

    def __init__(self, subject: Path, fsaverage: Path, n_target: int) -> None:
        self._native = {
            hemi: unit(fs.read_geometry(str(subject / "surf" / f"{hemi}.sphere.reg"))[0])
            for hemi in HEMIS
        }
        self._target = {
            hemi: unit(
                fs.read_geometry(str(fsaverage / "surf" / f"{hemi}.sphere.reg"))[0][:n_target]
            )
            for hemi in HEMIS
        }
        self._known: dict[tuple[str, int], int] = {}

    def many(self, hemi: str, vertices: np.ndarray) -> np.ndarray:
        todo = np.array(
            sorted({int(v) for v in vertices if (hemi, int(v)) not in self._known}), dtype=int
        )
        found = nearest_direction(self._native[hemi][todo], self._target[hemi])
        self._known.update({(hemi, int(v)): int(t) for v, t in zip(todo, found, strict=True)})
        return np.array([self._known[hemi, int(v)] for v in vertices], dtype=int)

    def __call__(self, hemi: str, vertex: int) -> int:
        return int(self.many(hemi, np.array([vertex]))[0])


Click = tuple[str, int, float, float]
"""(panel, position in the panel's slice list, px, py)."""


@dataclass(frozen=True)
class Case:
    """One render and what is known of its subject without it."""

    config: RenderConfig
    info: RenderInfo
    vmap: dict[str, np.ndarray]
    index: VertexIndex
    anatomy: Anatomy
    layout: Layout
    spheres: Spheres | None
    """None for a render without a template."""

    @property
    def reach(self) -> float:
        return min(5.0, self.config.vertex_map_margin_mm)

    def point(self, click: Click) -> np.ndarray:
        return click_point(self.layout, self.config, *click)

    def grid(self, per_axis: int, *, inset: float = 0.0) -> list[Click]:
        """Clicks on every tile of every panel: a grid from edge to edge of the tile."""
        at = np.linspace(inset, self.config.panel_px - inset, per_axis).tolist()
        return [
            (panel, slice_pos, px, py)
            for panel in PANEL_ORDER
            for slice_pos in range(len(self.layout.planes[panel]))
            for px in at
            for py in at
        ]


def mark_of(click: Click, mark_class: str = "A", **more: Any) -> MarkIn:
    panel, slice_pos, px, py = click
    return MarkIn.model_validate(
        {"client_id": "m1", "class": mark_class, "panel": panel, "slice": slice_pos}
        | {"px": px, "py": py}
        | more
    )


def make_case(
    subject: Path,
    out: Path,
    config: RenderConfig,
    *,
    fsaverage: Path | None = None,
    mesh: str = "fsaverage6",
    n_target: int = 40_962,
    pial: str = "pial",
) -> Case:
    template = None if fsaverage is None else build_template(fsaverage, "explicit", mesh=mesh)  # type: ignore[arg-type]
    info = render_subject(
        resolve_subject_files(subject), out, config, template=template, video=False
    )
    vmap = read_vertex_map(out / "vertex_map.npz")
    anatomy = read_anatomy(subject, pial)
    return Case(
        config=config,
        info=info,
        vmap=vmap,
        index=VertexIndex(vmap),
        anatomy=anatomy,
        layout=tile_layout(anatomy, config),
        spheres=None if fsaverage is None else Spheres(subject, fsaverage, n_target),
    )


def check_clicks(
    case: Case, clicks: Sequence[Click], classes: Sequence[str]
) -> tuple[list[ResolvedMark], list[Expected]]:
    """Resolve every click through the render and its vertex map and compare each stored
    value with what the files of the subject give."""
    anatomy, layout = case.anatomy, case.layout
    points = np.array([case.point(click) for click in clicks])
    expected = expected_marks(anatomy, points, classes, case.reach)
    resolved = [
        resolve_mark(case.info, case.index, mark_of(click, c), max_dist_mm=case.reach)
        for click, c in zip(clicks, classes, strict=True)
    ]
    # coordinates: the pixel in tkr RAS, back through the affine to voxels, on to the scanner
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    vox = np.linalg.solve(anatomy.vox2ras_tkr, homogeneous.T).T
    ras = (anatomy.vox2ras @ vox.T).T[:, :3]
    np.testing.assert_allclose([r.tkr for r in resolved], points, rtol=0, atol=ATOL)
    np.testing.assert_allclose([r.vox for r in resolved], vox[:, :3], rtol=0, atol=ATOL)
    np.testing.assert_allclose([r.ras for r in resolved], ras, rtol=0, atol=ATOL)
    # a conformed volume: scanner RAS is tkr RAS moved by the c_ras of the header
    np.testing.assert_allclose([r.ras for r in resolved], points + anatomy.c_ras, rtol=0, atol=1e-4)
    outside = np.maximum(np.maximum(layout.low - points, points - layout.high), 0.0)
    np.testing.assert_allclose(
        [r.box_dist_mm for r in resolved], np.sqrt((outside**2).sum(axis=1)), rtol=0, atol=ATOL
    )

    def or_nan(values: Sequence[float | None]) -> np.ndarray:
        return np.array([math.nan if v is None else v for v in values], dtype=float)

    assert [r.hemi for r in resolved] == [e.hemi for e in expected]
    assert [r.pial_vertex for r in resolved] == [e.pial_vertex for e in expected]
    assert [r.white_vertex for r in resolved] == [e.white_vertex for e in expected]
    for name in ("pial_dist", "white_dist"):
        np.testing.assert_allclose(
            or_nan([getattr(r, f"{name}_mm") for r in resolved]),
            or_nan([getattr(e, name) for e in expected]),
            rtol=0,
            atol=ATOL,
        )
    if case.spheres is None:
        assert all(r.fsavg6_vertex is None and r.vertex_surface is None for r in resolved)
    else:
        fsavg: list[int | None] = [None] * len(expected)
        for hemi in HEMIS:
            rows = [i for i, e in enumerate(expected) if e.hemi == hemi]
            vertices = np.array([expected[i].source_vertex for i in rows], dtype=int)
            for i, target in zip(rows, case.spheres.many(hemi, vertices).tolist(), strict=True):
                fsavg[i] = target
        assert [r.fsavg6_vertex for r in resolved] == fsavg
        assert [r.vertex_surface for r in resolved] == [e.source for e in expected]
    return resolved, expected


# ------------------------------------------------------------------ synthetic renders

CONFIGS = {
    "default": RenderConfig(panel_px=64),
    "flipped": RenderConfig(panel_px=48, radiological=False, sagittal_anterior_left=True),
    "neurological": RenderConfig(panel_px=32, radiological=False),
    "anterior_left": RenderConfig(panel_px=32, sagittal_anterior_left=True),
    # another box, slices 2.5 mm apart, a vertex map that reaches 3 mm
    "coarse": RenderConfig(
        panel_px=40,
        box_x_mm=(-20.0, 26.0),
        box_y_mm=(-5.0, 30.0),
        margin_mm=4.0,
        slice_step_mm=2.5,
        vertex_map_margin_mm=3.0,
    ),
    # 0.8 mm voxels: slice planes that are not a slice step apart
    "hires": RenderConfig(panel_px=64),
}
WITH_TEMPLATE = ("default", "flipped", "coarse")
N_FSAVERAGE4 = 2_562


@pytest.fixture(scope="module")
def cases(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Case]:
    """Renders of two synthetic subjects (1 mm with 10,242 vertices per surface and Tier B
    files; 0.8 mm without), three of them with a template on fsaverage4. Read-only."""
    root = tmp_path_factory.mktemp("marks")
    template = write_synthetic_template(root / "template", order=5)
    subject: SyntheticSubject = write_synthetic_subject(root / "subjects")
    add_tier_b(subject, template, warp=0.2)
    hires = write_synthetic_subject(root / "hires", voxel_size=0.8, shape=(200, 200, 200))
    out = {}
    for name, config in CONFIGS.items():
        out[name] = make_case(
            hires.path if name == "hires" else subject.path,
            root / "media" / name,
            config,
            fsaverage=template.path if name in WITH_TEMPLATE else None,
            mesh="fsaverage4",
            n_target=N_FSAVERAGE4,
        )
    return out


def test_the_layout_worked_out_here_is_that_of_the_render(cases: dict[str, Case]) -> None:
    """The expected values of this file rest on :func:`tile_layout`."""
    for name, case in cases.items():
        info, layout = case.info, case.layout
        np.testing.assert_allclose(info.box.low, layout.low, rtol=0, atol=ATOL, err_msg=name)
        np.testing.assert_allclose(info.box.high, layout.high, rtol=0, atol=ATOL, err_msg=name)
        for panel in PANEL_ORDER:
            assert len(info.panels[panel].slices) == len(layout.planes[panel]) > 5
            np.testing.assert_allclose(
                info.panels[panel].slice_mm, layout.planes[panel], rtol=0, atol=ATOL
            )
        assert info.vertex_map is not None
        assert (info.vertex_map.fsaverage is not None) == (name in WITH_TEMPLATE)
        assert case.index.has_counterparts == (name in WITH_TEMPLATE)
        assert case.index.n_vertices == {"lh": info.vertex_map.n_lh, "rh": info.vertex_map.n_rh}
    assert [len(cases["hires"].layout.planes[p]) for p in PANEL_ORDER] == [61, 54, 61]
    assert [len(cases["coarse"].layout.planes[p]) for p in PANEL_ORDER] == [19, 15, 25]
    # the edges of the tiles, in the letters of render.json: the two flags turn the horizontal
    letters = {
        name: [cases[name].info.panels[p].edges.left for p in PANEL_ORDER]
        for name in ("default", "flipped", "neurological", "anterior_left")
    }
    assert letters == {
        "default": ["P", "R", "R"],
        "flipped": ["A", "L", "L"],
        "neurological": ["P", "L", "L"],
        "anterior_left": ["A", "R", "R"],
    }


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_every_tile_resolves_as_the_full_surfaces_say(cases: dict[str, Case], name: str) -> None:
    """Clicks from edge to edge of every tile of every panel, in every class: coordinates,
    vertices, distances, hemisphere and template vertex equal the independent ones. The
    vertex map therefore stands in for the surfaces up to its margin."""
    case = cases[name]
    clicks = case.grid(6)
    classes = [CLASSES[i % 4] for i in range(len(clicks))]
    resolved, expected = check_clicks(case, clicks, classes)
    assert len(clicks) == 36 * sum(len(p) for p in case.layout.planes.values())
    n = len(resolved)
    with_vertex = sum(r.hemi is not None for r in resolved)
    assert 0.2 * n < with_vertex < 0.9 * n  # brain and background
    assert {r.hemi for r in resolved} == {"lh", "rh", None}
    # each branch of the hemisphere rule was taken
    assert sum(e.source == "white" for e, c in zip(expected, classes, strict=True) if c != "C") > 5
    assert sum(e.source == "pial" for e, c in zip(expected, classes, strict=True) if c == "C") > 5
    assert sum(e.other_hemi_is_nearer for e in expected) > 5
    assert sum(r.pial_vertex is None and r.white_vertex is not None for r in resolved) > 5
    assert sum(r.box_dist_mm == 0 for r in resolved) > 100
    assert sum(r.box_dist_mm > 5 for r in resolved) > 100
    if case.spheres is not None:
        assert len({r.fsavg6_vertex for r in resolved if r.fsavg6_vertex is not None}) > 200


def test_a_mark_can_resolve_to_a_vertex_that_is_not_visible(cases: dict[str, Case]) -> None:
    """``visible`` says where a vertex lies, not which vertex a mark resolves to
    (docs/rendering.md): near a tile edge the nearest vertex lies outside what is shown."""
    case = cases["default"]
    clicks = case.grid(9)
    resolved, _ = check_clicks(case, clicks, ["A"] * len(clicks))
    hidden = 0
    for r in resolved:
        if r.pial_vertex is not None:
            row = np.searchsorted(case.vmap[f"{r.hemi}_index"], r.pial_vertex)
            assert case.vmap[f"{r.hemi}_index"][row] == r.pial_vertex
            hidden += not case.vmap[f"{r.hemi}_visible"][row]
    assert hidden > 20


def test_a_click_on_a_pial_vertex_gives_that_vertex(cases: dict[str, Case]) -> None:
    """The projection of a pial vertex onto the nearest slice plane, clicked in each panel
    that shows it: the vertex itself, its hemisphere, its distance from the plane, which is
    at most half a voxel, and the template vertex of its place on ``sphere.reg``."""
    for name in ("default", "flipped"):
        case = cases[name]
        layout, config, spheres = case.layout, case.config, case.spheres
        assert spheres is not None
        both = np.concatenate([case.anatomy.pial[h] for h in HEMIS])
        spacing = cKDTree(both).query(both, k=2)[0][:, 1]  # to the nearest other vertex
        offset = {"lh": 0, "rh": len(case.anatomy.pial["lh"])}
        tested = 0
        for hemi in HEMIS:
            for vertex in range(0, len(case.anatomy.pial[hemi]), 7):
                xyz = case.anatomy.pial[hemi][vertex]
                for panel, (w, _, _) in AXES.items():
                    planes = layout.planes[panel]
                    slice_pos = int(np.argmin(np.abs(planes - xyz[w])))
                    off_plane = abs(planes[slice_pos] - xyz[w])
                    px, py = pixel_of(layout, config, panel, xyz).tolist()
                    on_tile = 0 <= px <= config.panel_px and 0 <= py <= config.panel_px
                    # no other vertex can be as near to the clicked point as this one
                    if not on_tile or off_plane >= spacing[offset[hemi] + vertex] / 2:
                        continue
                    r = resolve_mark(case.info, case.index, mark_of((panel, slice_pos, px, py)))
                    assert (r.hemi, r.pial_vertex) == (hemi, vertex)
                    assert r.pial_dist_mm == pytest.approx(off_plane, abs=ATOL)
                    assert off_plane <= 0.5 + ATOL
                    assert r.fsavg6_vertex == spheres(hemi, vertex)
                    assert r.vertex_surface == "pial"
                    assert r.white_vertex is not None and r.white_dist_mm is not None
                    assert r.white_dist_mm >= 2.0  # the white ellipsoid lies 2.5 mm inside
                    tested += 1
        assert tested > 300, name


def test_between_the_hemispheres_both_vertices_are_of_one_hemisphere(
    cases: dict[str, Case],
) -> None:
    """The pial surfaces of the synthetic subject lie 3 mm apart at the midline and the
    white surfaces 8 mm. A mark in the gap takes its hemisphere from the surface its class
    is about and both vertices from that hemisphere, also where the other surface has a
    nearer vertex across the midline."""
    case = cases["default"]
    subject_gap = (-5.5, -2.5)  # tests/synth.py: midline at -4, pial semi-axis 28.5
    medial = {hemi: case.anatomy.pial[hemi][:, 0] for hemi in HEMIS}
    assert medial["lh"].max() == pytest.approx(subject_gap[0], abs=0.01)
    assert medial["rh"].min() == pytest.approx(subject_gap[1], abs=0.01)
    clicks: list[Click] = []
    for panel in ("coronal", "axial"):
        centre = pixel_of(
            case.layout, case.config, panel, np.array([-4.0, 0.0, 0.0])
        ).tolist()[0]
        per_mm = case.config.panel_px / case.layout.side
        for slice_pos in range(0, len(case.layout.planes[panel]), 3):
            for dx in np.linspace(-1.6, 1.6, 9):
                for py in np.linspace(8.0, 56.0, 7):
                    clicks.append((panel, slice_pos, centre + dx * per_mm, float(py)))
    points = np.array([case.point(c) for c in clicks])
    assert ((points[:, 0] > subject_gap[0]) & (points[:, 0] < subject_gap[1])).all()
    by_class = {}
    for mark_class in ("A", "C"):
        resolved, expected = check_clicks(case, clicks, [mark_class] * len(clicks))
        by_class[mark_class] = resolved
        both = [r for r in resolved if r.pial_vertex is not None and r.white_vertex is not None]
        assert len(both) > 100
        for r, point in zip(resolved, points, strict=True):
            # each vertex lies, on the surface of the mark's hemisphere, where its distance says
            for surface in SURFACES:
                vertex = getattr(r, f"{surface}_vertex")
                if vertex is not None:
                    xyz = getattr(case.anatomy, surface)[r.hemi][vertex]
                    assert math.dist(xyz, point) == pytest.approx(
                        getattr(r, f"{surface}_dist_mm"), abs=ATOL
                    )
        # the rule was needed: the vertex across the midline was nearer, and was not taken
        assert sum(e.other_hemi_is_nearer for e in expected) > 20
        own = "white" if mark_class == "C" else "pial"
        assert {r.vertex_surface for r in both} == {own}
    # class C follows the white surface: for some marks into the other hemisphere than class A
    differ = [
        (a, c)
        for a, c in zip(by_class["A"], by_class["C"], strict=True)
        if a.hemi is not None and c.hemi is not None and a.hemi != c.hemi
    ]
    assert len(differ) > 20
    assert all(a.fsavg6_vertex != c.fsavg6_vertex or a.hemi != c.hemi for a, c in differ)


# ----------------------------------------------- the hemisphere rule on hand-made maps

CLICK: Click = ("coronal", 20, 30.25, 22.5)


def hand_index(
    case: Case, lh: Sequence[Vertex] = (), rh: Sequence[Vertex] = ()
) -> tuple[RenderInfo, VertexIndex]:
    """A vertex index whose positions are offsets from the point of ``CLICK``, and the
    render with the vertex counts of that index."""
    origin = case.point(CLICK)

    def placed(rows: Sequence[Vertex]) -> list[Vertex]:
        return [
            (v, tuple(origin + np.array(p)), tuple(origin + np.array(w)), f)  # type: ignore[misc]
            for v, p, w, f in rows
        ]

    assert case.info.vertex_map is not None
    listed = case.info.vertex_map.model_copy(update={"n_lh": len(lh), "n_rh": len(rh)})
    info = case.info.model_copy(update={"vertex_map": listed})
    return info, VertexIndex(vmap_of(placed(lh), placed(rh)))


def resolved_fields(r: ResolvedMark) -> tuple[Any, ...]:
    def mm(value: float | None) -> float | None:
        return None if value is None else round(value, 3)

    return (
        r.hemi,
        r.pial_vertex,
        mm(r.pial_dist_mm),
        r.white_vertex,
        mm(r.white_dist_mm),
        r.fsavg6_vertex,
        r.vertex_surface,
    )


def test_the_class_of_a_mark_decides_which_surface_sets_the_hemisphere(
    cases: dict[str, Case],
) -> None:
    # pial: lh at 1 mm, rh at 2 mm; white: lh at 4 mm, rh at 1.5 mm
    info, index = hand_index(
        cases["default"],
        lh=[(11, (-1, 0, 0), (-4, 0, 0), 110)],
        rh=[(22, (2, 0, 0), (1.5, 0, 0), 220)],
    )
    for mark_class in ("A", "B", "X"):
        r = resolve_mark(info, index, mark_of(CLICK, mark_class))
        assert resolved_fields(r) == ("lh", 11, 1.0, 11, 4.0, 110, "pial"), mark_class
    r = resolve_mark(info, index, mark_of(CLICK, "C"))
    assert resolved_fields(r) == ("rh", 22, 2.0, 22, 1.5, 220, "white")


def test_the_other_surface_is_searched_in_the_hemisphere_of_the_first(
    cases: dict[str, Case],
) -> None:
    case = cases["default"]
    # the white vertex of the mark's hemisphere is out of range; the one across is not taken
    info, index = hand_index(
        case, lh=[(11, (-1, 0, 0), (-6, 0, 0), 110)], rh=[(22, (2, 0, 0), (1.5, 0, 0), 220)]
    )
    assert resolved_fields(resolve_mark(info, index, mark_of(CLICK, "A"))) == (
        "lh",
        11,
        1.0,
        None,
        None,
        110,
        "pial",
    )
    # two vertices of one hemisphere: the pial and the white vertex need not be the same one
    info, index = hand_index(
        case, lh=[(11, (-1, 0, 0), (-4, 0, 0), 110), (12, (0, 3, 0), (0, 2, 0), 120)]
    )
    assert resolved_fields(resolve_mark(info, index, mark_of(CLICK, "A"))) == (
        "lh",
        11,
        1.0,
        12,
        2.0,
        110,
        "pial",
    )
    assert resolved_fields(resolve_mark(info, index, mark_of(CLICK, "C"))) == (
        "lh",
        11,
        1.0,
        12,
        2.0,
        120,
        "white",
    )


def test_without_a_vertex_of_its_own_surface_a_mark_takes_the_other(
    cases: dict[str, Case],
) -> None:
    case = cases["default"]
    # no pial vertex within 5 mm, a white one in each hemisphere: the nearer one, rh
    info, index = hand_index(
        case, lh=[(11, (-7, 0, 0), (-4, 0, 0), 110)], rh=[(22, (6, 0, 0), (3, 0, 0), 220)]
    )
    assert resolved_fields(resolve_mark(info, index, mark_of(CLICK, "A"))) == (
        "rh",
        None,
        None,
        22,
        3.0,
        220,
        "white",
    )
    assert resolved_fields(resolve_mark(info, index, mark_of(CLICK, "C"))) == (
        "rh",
        None,
        None,
        22,
        3.0,
        220,
        "white",
    )
    # class C without a white vertex in range takes the pial one
    info, index = hand_index(
        case, lh=[(11, (-2, 0, 0), (-9, 0, 0), 110)], rh=[(22, (3, 0, 0), (8, 0, 0), 220)]
    )
    assert resolved_fields(resolve_mark(info, index, mark_of(CLICK, "C"))) == (
        "lh",
        11,
        2.0,
        None,
        None,
        110,
        "pial",
    )
    # with a smaller snap radius the lh pial vertex is the only one left; then none
    assert resolved_fields(resolve_mark(info, index, mark_of(CLICK, "A"), max_dist_mm=2.5))[
        :2
    ] == ("lh", 11)
    r = resolve_mark(info, index, mark_of(CLICK, "A"), max_dist_mm=1.5)
    assert resolved_fields(r) == (None,) * 7


def test_a_mark_without_a_vertex_in_range_is_kept_as_a_point(cases: dict[str, Case]) -> None:
    case = cases["default"]
    info, index = hand_index(case, lh=[(11, (-9, 0, 0), (-8, 0, 0), 110)])
    r = resolve_mark(info, index, mark_of(CLICK, "A"))
    assert resolved_fields(r) == (None,) * 7
    np.testing.assert_allclose(r.tkr, case.point(CLICK), rtol=0, atol=ATOL)
    assert all(math.isfinite(x) for x in (*r.vox, *r.tkr, *r.ras, r.box_dist_mm))
    # and the corner of a real tile, which shows background
    corner = resolve_mark(case.info, case.index, mark_of(("coronal", 3, 0.0, 0.0), "B"))
    assert resolved_fields(corner) == (None,) * 7 and corner.box_dist_mm > 0


def test_a_vertex_without_a_template_vertex_gives_a_mark_without_one(
    cases: dict[str, Case],
) -> None:
    case = cases["default"]
    # the nearest pial vertex has no counterpart; that of the white vertex is not taken in its place
    info, index = hand_index(
        case, lh=[(11, (-1, 0, 0), (-9, 0, 0), -1), (12, (0, 9, 0), (0, 2, 0), 120)]
    )
    assert resolved_fields(resolve_mark(info, index, mark_of(CLICK, "A"))) == (
        "lh",
        11,
        1.0,
        12,
        2.0,
        None,
        None,
    )
    assert resolved_fields(resolve_mark(info, index, mark_of(CLICK, "C"))) == (
        "lh",
        11,
        1.0,
        12,
        2.0,
        120,
        "white",
    )
    # a render without a template: every vertex of its map is without one, and that is no error
    plain = cases["neurological"]
    assert (plain.vmap["lh_fsavg6"] == -1).all() and (plain.vmap["rh_fsavg6"] == -1).all()
    on_brain = [
        r
        for click in plain.grid(5, inset=4.0)
        if (r := resolve_mark(plain.info, plain.index, mark_of(click))).pial_vertex is not None
    ]
    assert len(on_brain) > 500
    assert all(r.fsavg6_vertex is None and r.vertex_surface is None for r in on_brain)
    assert all(r.hemi is not None for r in on_brain)


def test_without_a_vertex_map_a_mark_has_coordinates_only(cases: dict[str, Case]) -> None:
    case = cases["flipped"]
    for click in case.grid(3)[::17]:
        with_map = resolve_mark(case.info, case.index, mark_of(click))
        without = resolve_mark(case.info, None, mark_of(click))
        assert (without.vox, without.tkr, without.ras) == (with_map.vox, with_map.tkr, with_map.ras)
        assert without.box_dist_mm == with_map.box_dist_mm
        assert resolved_fields(without) == (None,) * 7
    # a render that lists no vertex map resolves the same way
    bare = case.info.model_copy(update={"vertex_map": None, "template": None})
    r = resolve_mark(bare, None, mark_of(CLICK))
    np.testing.assert_allclose(r.tkr, case.point(CLICK), rtol=0, atol=ATOL)


def test_the_distance_to_the_view_box(cases: dict[str, Case]) -> None:
    """The default tiles show 10 mm more than the box in x and z and 13.5 mm more in y."""
    case = cases["default"]
    size = case.config.panel_px

    def distance(click: Click) -> float:
        return resolve_mark(case.info, None, mark_of(click)).box_dist_mm

    assert distance(("sagittal", 30, size / 2, size / 2)) == 0.0
    assert distance(("coronal", 0, size / 2, size / 2)) == 0.0
    # the corners of a tile: outside along both of its axes, inside along the slice axis
    for px in (0.0, size):
        for py in (0.0, size):
            assert distance(("sagittal", 30, px, py)) == pytest.approx(math.hypot(13.5, 10.0))
            assert distance(("coronal", 5, px, py)) == pytest.approx(math.hypot(10.0, 10.0))
            assert distance(("axial", 60, px, py)) == pytest.approx(math.hypot(10.0, 13.5))
    # the middle of an edge: outside along one axis
    assert distance(("sagittal", 0, 0.0, size / 2)) == pytest.approx(13.5)
    assert distance(("sagittal", 0, size / 2, size)) == pytest.approx(10.0)
    # on a face of the box and a pixel to either side of it (0.8 px per mm)
    per_mm = size / case.layout.side
    assert distance(("coronal", 5, 10.0 * per_mm, size / 2)) == pytest.approx(0.0, abs=1e-9)
    assert distance(("coronal", 5, 11.0 * per_mm, size / 2)) == 0.0
    assert distance(("coronal", 5, 9.0 * per_mm, size / 2)) == pytest.approx(1.0)
    # the 0.8 mm subject has slice planes off the 1 mm steps, so its end slices lie up to
    # half a voxel outside the box along the slice axis
    hires = cases["hires"]
    for panel, (w, _, _) in AXES.items():
        planes, half = hires.layout.planes[panel], hires.config.panel_px / 2
        beyond = max(hires.layout.low[w] - planes[0], planes[-1] - hires.layout.high[w], 0.0)
        ends = [
            resolve_mark(hires.info, None, mark_of((panel, s, half, half))).box_dist_mm
            for s in (0, len(planes) - 1)
        ]
        assert max(ends) == pytest.approx(beyond, abs=ATOL) and beyond <= 0.4


@settings(max_examples=300, deadline=None)
@given(
    name=st.sampled_from(sorted(CONFIGS)),
    panel=st.sampled_from(PANEL_ORDER),
    where=st.tuples(*[st.floats(0.0, 1.0)] * 3),
    mark_class=st.sampled_from(CLASSES),
)
def test_a_voxel_comes_back_through_its_pixel(
    cases: dict[str, Case],
    name: str,
    panel: str,
    where: tuple[float, float, float],
    mark_class: str,
) -> None:
    """Every panel and slice: ``voxel_to_pixel`` and then ``resolve_mark`` returns the
    voxel, and the two affines of the file give the two positions."""
    case = cases[name]
    info = case.info.panels[panel]  # type: ignore[index]
    size = case.config.panel_px
    slice_pos = min(int(where[0] * len(info.slices)), len(info.slices) - 1)
    # a voxel on the tile, from the two corners of the tile in voxel coordinates
    corners = geometry.pixel_to_voxel(info, slice_pos, np.array([0.0, size]), np.array([0.0, size]))
    voxel = corners[0] + (corners[1] - corners[0]) * np.array(where)
    voxel[info.slice_voxel_axis] = info.slices[slice_pos]
    px, py, along = geometry.voxel_to_pixel(info, voxel)
    assert float(along) == info.slices[slice_pos]
    px, py = float(np.clip(px, 0.0, size)), float(np.clip(py, 0.0, size))  # rounding at an edge
    r = resolve_mark(
        case.info, case.index, mark_of((panel, slice_pos, px, py), mark_class), max_dist_mm=3.0
    )
    np.testing.assert_allclose(r.vox, voxel, rtol=0, atol=1e-8)
    homogeneous = np.append(voxel, 1.0)
    np.testing.assert_allclose(r.tkr, (case.anatomy.vox2ras_tkr @ homogeneous)[:3], atol=1e-8)
    np.testing.assert_allclose(r.ras, (case.anatomy.vox2ras @ homogeneous)[:3], atol=1e-8)
    np.testing.assert_allclose(r.tkr, case.point((panel, slice_pos, px, py)), rtol=0, atol=1e-8)


def test_marked_vertices_through_the_vertex_map_are_those_of_the_surface(
    cases: dict[str, Case],
) -> None:
    """Up to the margin of the vertex map, the vertices near a mark on a tile are the same
    in the map as on the full surface."""
    for name, n_min in (("default", 150), ("coarse", 40)):
        case = cases[name]
        margin = case.config.vertex_map_margin_mm
        clicks = case.grid(4)[5::41]
        points = [case.point(click) for click in clicks]
        as_tuples = [(float(p[0]), float(p[1]), float(p[2])) for p in points]
        for surface in SURFACES:
            for tolerance in (2.0, margin):
                found = marked_vertices(case.index, as_tuples, tolerance, surface=surface)  # type: ignore[arg-type]
                total = 0
                for hemi in HEMIS:
                    xyz = getattr(case.anatomy, surface)[hemi]
                    near = np.zeros(len(xyz), dtype=bool)
                    for p in points:
                        near |= np.sqrt(((xyz - p) ** 2).sum(axis=1)) <= tolerance
                    assert found[hemi].tolist() == np.flatnonzero(near).tolist()
                    total += int(near.sum())
                assert total > n_min, (name, surface, tolerance)


# --------------------------------------------------------------------------- validation


def frame_showing(info: RenderInfo, mode: str, panel: str, slice_pos: int) -> int:
    frames = info.modes[mode].frames  # type: ignore[index]
    return next(f for f, shown in enumerate(frames) if shown[PANEL_ORDER.index(panel)] == slice_pos)  # type: ignore[arg-type]


def test_a_mark_on_a_tile_is_valid(cases: dict[str, Case]) -> None:
    info = cases["default"].info
    size = info.panels["sagittal"].width
    assert size == info.panels["sagittal"].height == 64
    for panel in PANEL_ORDER:
        last = len(info.panels[panel].slices) - 1
        for slice_pos in (0, last):
            for px, py in ((0.0, 0.0), (size, size), (0.0, size), (31.5, 0.25)):
                assert validate_mark(info, mark_of((panel, slice_pos, px, py))) is None
        for mode in ("sync", "seq"):
            frame = frame_showing(info, mode, panel, last)
            validate_mark(info, mark_of((panel, last, 1.0, 1.0), frame=frame, mode=mode))
            n_frames = len(info.modes[mode].frames)  # type: ignore[index]
            # the frame or the mode alone is the context of the click and is not checked
            validate_mark(info, mark_of((panel, 0, 1.0, 1.0), frame=n_frames + 5))
            validate_mark(info, mark_of((panel, 0, 1.0, 1.0), mode=mode))


@pytest.mark.parametrize(
    ("change", "field", "message"),
    [
        ({"slice": 61}, "slice", "the sagittal panel has 61 slices, got 61"),
        ({"slice": -1}, "slice", "the sagittal panel has 61 slices, got -1"),
        ({"panel": "coronal", "slice": 54}, "slice", "the coronal panel has 54 slices, got 54"),
        ({"px": -0.01}, "px", "px lies between 0 and 64, got -0.01"),
        ({"px": 64.5}, "px", "px lies between 0 and 64, got 64.5"),
        ({"py": -3.0}, "py", "py lies between 0 and 64, got -3"),
        ({"py": 1e9}, "py", "py lies between 0 and 64, got 1e+09"),
        ({"frame": 120, "mode": "sync"}, "frame", "the sync mode has 120 frames, got 120"),
        ({"frame": -1, "mode": "sync"}, "frame", "the sync mode has 120 frames, got -1"),
        ({"frame": 346, "mode": "seq"}, "frame", "the seq mode has 346 frames, got 346"),
        (
            {"frame": 11, "mode": "sync"},
            "frame",
            "frame 11 of the sync mode shows slice 11 of the sagittal panel, not slice 10",
        ),
        (
            {"frame": 3, "mode": "seq", "panel": "axial", "slice": 3},
            "frame",
            "frame 3 of the seq mode shows slice 35 of the axial panel, not slice 3",
        ),
        # the first field that is wrong, in the order of MARK_FIELDS
        ({"slice": 99, "px": -1.0, "py": -1.0, "frame": 500, "mode": "sync"}, "slice", "99"),
        ({"px": -1.0, "py": -1.0, "frame": 500, "mode": "sync"}, "px", "-1"),
        ({"py": 70.0, "frame": 500, "mode": "sync"}, "py", "70"),
    ],
)
def test_a_mark_off_the_render_is_refused(
    cases: dict[str, Case], change: dict[str, Any], field: str, message: str
) -> None:
    case = cases["default"]
    good = {"panel": "sagittal", "slice": 10, "px": 12.0, "py": 40.0}
    validate_mark(case.info, mark_of(tuple(good.values()), frame=10, mode="sync"))  # type: ignore[arg-type]
    values = good | change
    click = (values.pop("panel"), values.pop("slice"), values.pop("px"), values.pop("py"))
    mark = mark_of(click, **values)
    with pytest.raises(MarkError) as caught:
        validate_mark(case.info, mark)
    assert caught.value.field == field and field in MARK_FIELDS
    assert message in caught.value.message
    assert str(caught.value) == f"{field}: {caught.value.message}"
    # nothing is resolved from such a mark: slice -1 would read as the last tile
    for index in (case.index, None):
        with pytest.raises(MarkError) as again:
            resolve_mark(case.info, index, mark)
        assert (again.value.field, again.value.message) == (field, caught.value.message)


def test_a_mark_that_did_not_pass_the_model_is_refused_as_well(cases: dict[str, Case]) -> None:
    info = cases["default"].info
    good = mark_of(("sagittal", 10, 12.0, 40.0))
    with pytest.raises(ValueError, match="finite"):
        mark_of(("sagittal", 10, math.nan, 40.0))
    for field in ("px", "py"):
        for value in (math.nan, math.inf, -math.inf):
            with pytest.raises(MarkError) as caught:
                validate_mark(info, good.model_copy(update={field: value}))
            assert caught.value.field == field
    # a render that lacks a panel or a mode
    panels = {name: panel for name, panel in info.panels.items() if name != "sagittal"}
    with pytest.raises(MarkError) as caught:
        validate_mark(info.model_copy(update={"panels": panels}), good)
    assert (caught.value.field, caught.value.message) == (
        "panel",
        "the render has no sagittal panel",
    )
    with pytest.raises(MarkError) as caught:
        validate_mark(
            info.model_copy(update={"modes": {"seq": info.modes["seq"]}}),
            good.model_copy(update={"frame": 10, "mode": "sync"}),
        )
    assert (caught.value.field, caught.value.message) == ("frame", "the render has no sync mode")


def test_a_snap_radius_above_the_margin_of_the_render_is_refused(cases: dict[str, Case]) -> None:
    case, coarse = cases["default"], cases["coarse"]
    mark = mark_of(CLICK)
    assert case.config.vertex_map_margin_mm == 5.0 and coarse.config.vertex_map_margin_mm == 3.0
    resolve_mark(case.info, case.index, mark, max_dist_mm=5.0)
    resolve_mark(coarse.info, coarse.index, mark_of(("coronal", 3, 1.0, 1.0)), max_dist_mm=3.0)
    for index in (case.index, None):
        with pytest.raises(ValueError, match="snap radius") as caught:
            resolve_mark(case.info, index, mark, max_dist_mm=5.5)
        assert not isinstance(caught.value, MarkError)
    # the default of 5 mm on a render whose vertex map reaches 3 mm
    with pytest.raises(ValueError, match=r"snap radius of marks \(5 mm\).*= 3 mm"):
        resolve_mark(coarse.info, coarse.index, mark_of(("coronal", 3, 1.0, 1.0)))
    for radius in (-1.0, math.nan):
        with pytest.raises(ValueError):
            resolve_mark(case.info, case.index, mark, max_dist_mm=radius)
    # a smaller radius finds fewer vertices, never other ones
    clicks = case.grid(4)[::7]
    wide = [resolve_mark(case.info, case.index, mark_of(c)) for c in clicks]
    narrow = [resolve_mark(case.info, case.index, mark_of(c), max_dist_mm=1.0) for c in clicks]
    lost = 0
    for w, n in zip(wide, narrow, strict=True):
        if w.pial_dist_mm is not None and w.pial_dist_mm <= 1.0:
            assert (n.hemi, n.pial_vertex, n.pial_dist_mm) == (w.hemi, w.pial_vertex, w.pial_dist_mm)
        else:
            assert n.pial_vertex is None
            lost += w.pial_vertex is not None
    assert lost > 50 and sum(n.pial_vertex is not None for n in narrow) > 50


def test_the_vertex_index_of_another_render_is_refused(cases: dict[str, Case]) -> None:
    case, hires, plain = cases["default"], cases["hires"], cases["neurological"]
    mark = mark_of(CLICK)
    with pytest.raises(ValueError, match="not that of this render"):
        resolve_mark(case.info, hires.index, mark)
    # the same subject and vertices: counterparts where the render names no template
    assert plain.index.n_vertices == case.index.n_vertices
    with pytest.raises(ValueError, match="names no template"):
        resolve_mark(plain.info, case.index, mark)
    bare = case.info.model_copy(update={"vertex_map": None, "template": None})
    with pytest.raises(ValueError, match="lists no vertex map"):
        resolve_mark(bare, case.index, mark)


def test_a_resolved_mark_is_the_shape_of_the_api(cases: dict[str, Case]) -> None:
    case = cases["default"]
    posted = {
        "client_id": "m1",
        "class": "A",
        "panel": "sagittal",
        "slice": 31,
        "px": 21.5,
        "py": 30.25,
        "frame": frame_showing(case.info, "sync", "sagittal", 31),
        "mode": "sync",
        "ts": "2026-11-12T14:03:20.004Z",
    }
    r = resolve_mark(case.info, case.index, MarkIn.model_validate(posted))
    assert type(r) is ResolvedMark
    assert ResolvedMark.model_validate_json(r.model_dump_json()) == r
    assert r.hemi in HEMIS and r.vertex_surface == "pial" and r.fsavg6_vertex is not None
    for value in (r.pial_vertex, r.white_vertex, r.fsavg6_vertex):
        assert type(value) is int
    for value in (*r.vox, *r.tkr, *r.ras, r.pial_dist_mm, r.white_dist_mm, r.box_dist_mm):
        assert type(value) is float
    # the voxel of the slice is the one render.json lists
    panel = case.info.panels["sagittal"]
    assert r.vox[panel.slice_voxel_axis] == panel.slices[31]
    assert r.tkr[0] == panel.slice_mm[31]


# ------------------------------------------- the cached real subject (ds000102 sub-01)

REAL_SUBJECT = "ds000102-sub-01"
REAL_FSAVERAGE = "ds000102-fsaverage"
REAL_SUBJECT_FILES = (
    "mri/T1.mgz",
    "surf/lh.pial",
    "surf/rh.pial",
    "surf/lh.white",
    "surf/rh.white",
    "surf/lh.sphere.reg",
    "surf/rh.sphere.reg",
)
REAL_FSAVERAGE_FILES = tuple(
    f"{folder}/{hemi}.{name}"
    for hemi in HEMIS
    for folder, name in (
        ("surf", "sphere.reg"),
        ("surf", "white"),
        ("label", "cortex.label"),
        ("label", "aparc.annot"),
    )
)
N_FSAVERAGE6 = 40_962


def contour_pixels(media: Path, info: RenderInfo, surface: str) -> dict[str, np.ndarray]:
    """Panel -> rows ``(slice, column, row)`` of the tile pixels that a contour of the
    surface covers: opaque, in the surface's colour (pial #FFD400 drawn over white
    #FF3B30)."""
    out = {}
    for panel in PANEL_ORDER:
        found = []
        for slice_pos in range(len(info.panels[panel].slices)):
            tile = np.asarray(Image.open(media / info.tiles.contours_path(panel, slice_pos)))
            assert tile.shape == (info.panels[panel].height, info.panels[panel].width, 4)
            red, green, blue, alpha = (tile[..., c].astype(int) for c in range(4))
            if surface == "pial":
                drawn = (alpha >= 200) & (red >= 230) & (green >= 190) & (blue <= 60)
            else:
                drawn = (alpha >= 200) & (red >= 230) & (green <= 80) & (blue <= 70)
            rows, columns = np.nonzero(drawn)
            found.append(np.stack([np.full(len(rows), slice_pos), columns, rows], axis=1))
        out[panel] = np.concatenate(found)
    return out


def segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((p - a) * ab).sum(axis=-1) / (ab * ab).sum(axis=-1), 0.0, 1.0)
    return np.sqrt(((p - (a + t[..., None] * ab)) ** 2).sum(axis=-1))


def triangle_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Distance from points to triangles (arrays broadcast): to the plane where the
    projection falls inside the triangle, else to the nearest edge."""
    normal = np.cross(b - a, c - a)
    normal = normal / np.sqrt((normal**2).sum(axis=-1, keepdims=True))
    height = ((p - a) * normal).sum(axis=-1)
    foot = p - height[..., None] * normal
    inside = np.ones(height.shape, dtype=bool)
    for u, v in ((a, b), (b, c), (c, a)):
        inside &= (np.cross(v - u, foot - u) * normal).sum(axis=-1) >= 0
    edges = np.minimum(
        np.minimum(segment_distance(p, a, b), segment_distance(p, b, c)), segment_distance(p, c, a)
    )
    return np.where(inside, np.abs(height), edges)


def distance_to_pial(anatomy: Anatomy, points: np.ndarray, n_faces: int = 24) -> np.ndarray:
    """An upper bound of the distance from each point to the pial surfaces: the smallest
    distance to the ``n_faces`` faces of each hemisphere whose centroids are nearest."""
    best = np.full(len(points), np.inf)
    for hemi in HEMIS:
        corners = anatomy.pial[hemi][anatomy.pial_faces[hemi]]
        _, faces = cKDTree(corners.mean(axis=1)).query(points, k=n_faces)
        a, b, c = (corners[faces, k] for k in range(3))
        best = np.minimum(best, triangle_distance(points[:, None, :], a, b, c).min(axis=1))
    return best


def real_case(cache: Path, out: Path, config: RenderConfig) -> Case:
    return make_case(cache / REAL_SUBJECT, out, config, fsaverage=cache / REAL_FSAVERAGE)


def sampled_clicks(
    case: Case, media: Path, rng: np.random.Generator, n_pial: int, n_white: int, n_any: int
) -> tuple[list[Click], list[str], int]:
    """Clicks at the centres of drawn contour pixels of every panel, pial and white, and at
    random places of random tiles; the classes in turn. Also returns how many of the clicks
    are on pial contour pixels: these come first."""
    size = case.config.panel_px
    clicks: list[Click] = []
    for surface, n in (("pial", n_pial), ("white", n_white)):
        pixels = contour_pixels(media, case.info, surface)
        for panel in PANEL_ORDER:
            assert len(pixels[panel]) > 20 * n
            chosen = pixels[panel][rng.choice(len(pixels[panel]), size=n, replace=False)]
            # the pixel in column c and row r is sampled at its centre (c + 0.5, r + 0.5)
            clicks += [(panel, int(s), c + 0.5, r + 0.5) for s, c, r in chosen.tolist()]
    for panel in PANEL_ORDER:
        slices = rng.integers(0, len(case.layout.planes[panel]), size=n_any)
        at = rng.uniform(0.0, size, size=(n_any, 2))
        clicks += [(panel, int(s), float(x), float(y)) for s, (x, y) in zip(slices, at, strict=True)]
    return clicks, [CLASSES[i % 4] for i in range(len(clicks))], 3 * n_pial


def check_real_clicks(
    case: Case, media: Path, rng: np.random.Generator, n_pial: int, n_white: int, n_any: int
) -> dict[str, float]:
    """The checks of the real subject, for the test below and for a larger sample by hand."""
    clicks, classes, on_pial = sampled_clicks(case, media, rng, n_pial, n_white, n_any)
    resolved, expected = check_clicks(case, clicks, classes)
    # a click on a drawn pial contour is a point of the pial surface, to within the width
    # of the line and a pixel: this ties the pixel coordinates to the anatomy
    points = np.array([case.point(click) for click in clicks[:on_pial]])
    to_surface = distance_to_pial(case.anatomy, points)
    mm_per_px = case.layout.side / case.config.panel_px
    assert to_surface.max() < 2.0 * mm_per_px + 0.1
    assert np.median(to_surface) < 0.5 * mm_per_px
    assert all(r.pial_vertex is not None and r.hemi is not None for r in resolved[:on_pial])
    assert all(r.fsavg6_vertex is not None for r in resolved[:on_pial])
    # A, B and X take the template vertex from the pial vertex; C from the white one
    for r, mark_class in zip(resolved[:on_pial], classes, strict=False):
        assert r.vertex_surface == ("white" if mark_class == "C" else "pial") or (
            mark_class == "C" and r.white_vertex is None
        )
    hidden = 0
    for r in resolved[:on_pial]:
        row = np.searchsorted(case.vmap[f"{r.hemi}_index"], r.pial_vertex)
        hidden += not case.vmap[f"{r.hemi}_visible"][row]
    return {
        "clicks": len(clicks),
        "on_pial": on_pial,
        "with_vertex": sum(r.hemi is not None for r in resolved),
        "rh": sum(r.hemi == "rh" for r in resolved),
        "other_hemi_is_nearer": sum(e.other_hemi_is_nearer for e in expected),
        "not_visible": hidden,
        "max_pial_dist_mm": max(r.pial_dist_mm or 0.0 for r in resolved[:on_pial]),
        "max_to_surface_mm": float(to_surface.max()),
        "median_to_surface_mm": float(np.median(to_surface)),
        "template_vertices": len({(r.hemi, r.fsavg6_vertex) for r in resolved[:on_pial]}),
    }


@pytest.mark.parametrize(
    ("name", "config"),
    [
        ("default", RenderConfig()),
        ("flipped", RenderConfig(panel_px=160, radiological=False, sagittal_anterior_left=True)),
    ],
)
def test_cached_real_subject_marks_equal_the_full_surfaces(
    cached_subjects: Path, tmp_path: Path, name: str, config: RenderConfig
) -> None:
    """ds000102 sub-01 with the fsaverage next to it, rendered here: clicks on drawn pial
    and white contour pixels and at random places of every panel resolve to the voxel, the
    two positions, the vertices, their distances, the hemisphere and the fsaverage6 vertex
    that the T1 header, the four surfaces and the two ``sphere.reg`` files give."""
    present = all((cached_subjects / REAL_SUBJECT / f).is_file() for f in REAL_SUBJECT_FILES)
    present &= all((cached_subjects / REAL_FSAVERAGE / f).is_file() for f in REAL_FSAVERAGE_FILES)
    if not present:
        pytest.skip("the real subject and the real fsaverage are not in the pytest cache")
    case = real_case(cached_subjects, tmp_path / "media", config)
    info = case.info
    assert info.vertex_map is not None and info.vertex_map.fsaverage == "fsaverage6"
    assert case.index.n_vertices == {"lh": 37_356, "rh": 37_298}
    np.testing.assert_allclose(info.box.low, case.layout.low, rtol=0, atol=ATOL)
    np.testing.assert_allclose(info.box.high, case.layout.high, rtol=0, atol=ATOL)
    for panel in PANEL_ORDER:
        np.testing.assert_array_equal(info.panels[panel].slice_mm, case.layout.planes[panel])
    assert [len(case.layout.planes[p]) for p in PANEL_ORDER] == [61, 54, 61]
    letters = [info.panels[p].edges.left for p in PANEL_ORDER]
    assert letters == (["P", "R", "R"] if name == "default" else ["A", "L", "L"])

    seen = check_real_clicks(case, tmp_path / "media", np.random.default_rng(20261113), 150, 40, 60)
    assert seen["clicks"] == 3 * (150 + 40 + 60) and seen["on_pial"] == 450
    assert seen["max_pial_dist_mm"] < 1.5
    assert 100 < seen["rh"] < seen["with_vertex"] - 100
    assert seen["with_vertex"] < seen["clicks"]  # some random clicks are on background
    assert seen["other_hemi_is_nearer"] > 0
    assert seen["template_vertices"] > 300
