"""Sphere mapping: containing faces, barycentric weights, the nearest-vertex fallback, winding.

Synthetic spheres are icospheres, plain, rotated, or registered through the closed-form warp
of ``synth_tierb``. The last tests compare the map of a real subject onto the real fsaverage
sphere with an exhaustive search over all faces: one reads the pytest cache and skips when
the files are absent, its network twin fetches them.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from jaxtyping import TypeCheckError
from synth import (  # pyrefly: ignore[missing-import]
    SyntheticSubject,
    icosphere,
    write_synthetic_subject,
)
from synth_template import (  # pyrefly: ignore[missing-import]
    MEDIAL_WALL_M,
    SyntheticTemplate,
    medial_coordinate,
    write_synthetic_template,
)
from synth_tierb import (  # pyrefly: ignore[missing-import]
    add_tier_b,
    expected_thickness,
    warp_directions,
)

from surfcensus.fsio.registration import (
    COINCIDENT,
    CONTAIN_TOLERANCE,
    SphereMap,
    build_sphere_map,
    resample,
    unit_sphere,
)
from surfcensus.fsio.surfaces import Surface, read_surface
from surfcensus.sources.base import HEMIS

MAP_ARRAYS = ("corners", "weights", "fallback", "nearest_source", "nearest_target")


def surface(vertices: np.ndarray, faces: np.ndarray) -> Surface:
    return Surface(
        vertices=np.ascontiguousarray(vertices, dtype=np.float64),
        faces=np.ascontiguousarray(faces, dtype=np.int64),
    )


def rotation(a: float, b: float, c: float) -> np.ndarray:
    """A proper rotation from three angles (about z, then y, then x)."""
    (ca, sa), (cb, sb), (cc, sc) = ((np.cos(t), np.sin(t)) for t in (a, b, c))
    rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rx = np.array([[1, 0, 0], [0, cc, -sc], [0, sc, cc]])
    return rx @ ry @ rz


def random_directions(rng: np.random.Generator, n: int) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def warped_sphere(order: int, warp: float = 0.15, radius: float = 100.0) -> Surface:
    """An icosphere registered through the warp of ``synth_tierb``: vertex ``i`` sits at the
    template direction ``warp_directions(d_i)``."""
    directions, faces = icosphere(order)
    return surface(warp_directions(directions, warp) * radius, faces)


def assert_same_map(a: SphereMap, b: SphereMap, *, but: tuple[str, ...] = ()) -> None:
    for name in MAP_ARRAYS:
        if name not in but:
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
    assert (a.n_source, a.n_target) == (b.n_source, b.n_target)


def interpolated_points(smap: SphereMap, source_unit: np.ndarray) -> np.ndarray:
    """The point each row of the map stands for: the weighted corners, back on the sphere."""
    p = (source_unit[smap.corners] * smap.weights[:, :, None]).sum(axis=1)
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def solved_weights(source_unit: np.ndarray, corners: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Barycentric weights by a linear solve per row: ``p`` is a positive multiple of
    ``w0 a + w1 b + w2 c``. Independent of the triple products the package uses."""
    matrix = np.swapaxes(source_unit[corners], 1, 2)  # columns a, b, c
    w = np.linalg.solve(matrix, points[:, :, None])[:, :, 0]
    return w / w.sum(axis=1, keepdims=True)


def containing_faces(
    source_unit: np.ndarray, faces: np.ndarray, points: np.ndarray, *margins: float
) -> list[list[np.ndarray]]:
    """Exhaustive search. For each margin, for each point: the outward faces whose three edge
    planes the point is inside of, each by ``margin`` times the sum of the three triple
    products (with a negative margin a little outside still counts). No tree, no
    neighbourhood, BLAS products."""
    a, b, c = (source_unit[faces[:, i]] for i in range(3))
    n0, n1, n2 = np.cross(b, c), np.cross(c, a), np.cross(a, b)
    outward = np.einsum("ij,ij->i", a, n0) > 0
    found: list[list[np.ndarray]] = [[] for _ in margins]
    for begin in range(0, len(points), 64):
        p = points[begin : begin + 64]
        c0, c1, c2 = p @ n0.T, p @ n1.T, p @ n2.T
        total = c0 + c1 + c2
        lowest = np.minimum(np.minimum(c0, c1), c2)
        candidate = outward & (total > 0)
        for margin, per_point in zip(margins, found, strict=True):
            inside = candidate & (lowest >= margin * total)
            per_point.extend(np.flatnonzero(row) for row in inside)
    return found


def nearest_by_matrix(points: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(index, gap)``: the candidate with the largest cosine to each point, and how much
    nearer it is, as a cosine, than the runner-up."""
    index = np.empty(len(points), dtype=np.int64)
    gap = np.empty(len(points))
    for begin in range(0, len(points), 256):
        cosine = points[begin : begin + 256] @ candidates.T
        top = np.argpartition(cosine, -2, axis=1)[:, -2:]
        top_cos = np.take_along_axis(cosine, top, axis=1)
        index[begin : begin + 256] = cosine.argmax(axis=1)
        gap[begin : begin + 256] = np.abs(top_cos[:, 1] - top_cos[:, 0])
    return index, gap


# ---------------------------------------------------------------------------------------
# unit_sphere


def test_unit_sphere_scales_every_vertex_to_length_one() -> None:
    directions, _ = icosphere(2)
    radius = 100.0 * (1.0 + 0.009 * np.sin(np.arange(len(directions))))  # within 1%
    unit = unit_sphere(directions * radius[:, None])
    np.testing.assert_allclose(np.linalg.norm(unit, axis=1), 1.0, rtol=0, atol=4e-16)
    np.testing.assert_allclose(unit, directions, rtol=0, atol=1e-15)
    assert unit.dtype == np.float64


@pytest.mark.parametrize(
    ("change", "words"),
    [
        (lambda v: v.__setitem__(5, 0.0), "zero or not finite"),
        (lambda v: v.__setitem__((5, 1), np.nan), "zero or not finite"),
        (lambda v: v.__setitem__((5, 2), np.inf), "zero or not finite"),
        (lambda v: v.__setitem__(5, v[5] * 1.02), "from the median radius"),
        (lambda v: v.__setitem__(5, v[5] * 0.98), "from the median radius"),
    ],
)
def test_unit_sphere_refuses_a_degenerate_sphere(
    change: Callable[[np.ndarray], None], words: str
) -> None:
    vertices = icosphere(2)[0] * 100.0
    change(vertices)
    with pytest.raises(ValueError, match=r"^degenerate sphere: .*" + words):
        unit_sphere(vertices)


def test_unit_sphere_refuses_no_vertices() -> None:
    with pytest.raises(ValueError, match=r"^degenerate sphere: no vertices"):
        unit_sphere(np.empty((0, 3)))


def test_unit_sphere_accepts_radii_within_one_percent() -> None:
    vertices = icosphere(1)[0] * 100.0
    vertices[3] *= 1.009
    vertices[4] *= 0.991
    assert unit_sphere(vertices).shape == vertices.shape


@pytest.mark.parametrize("which", ["source", "target"])
def test_build_refuses_a_degenerate_sphere_on_either_side(which: str) -> None:
    directions, faces = icosphere(2)
    bad = directions * 100.0
    bad[7] = 0.0
    source = surface(bad if which == "source" else directions, faces)
    target = bad if which == "target" else directions
    with pytest.raises(ValueError, match=r"^degenerate sphere: "):
        build_sphere_map(source, target)


@pytest.mark.parametrize("index", [-1, 162, 10_000])
def test_build_refuses_a_face_that_names_no_vertex(index: int) -> None:
    directions, faces = icosphere(2)
    faces = faces.copy()
    faces[11, 1] = index
    with pytest.raises(ValueError, match=r"^degenerate sphere: faces name vertices"):
        build_sphere_map(surface(directions, faces), directions)


@pytest.mark.parametrize("k", [0, -3])
def test_build_refuses_k_below_one(k: int) -> None:
    directions, faces = icosphere(1)
    with pytest.raises(ValueError, match="k must be at least 1"):
        build_sphere_map(surface(directions, faces), directions, k=k)


# ---------------------------------------------------------------------------------------
# a sphere onto itself


@pytest.mark.parametrize("order", [0, 3, 5])
def test_a_sphere_onto_itself_is_the_identity(order: int) -> None:
    directions, faces = icosphere(order)
    n = len(directions)
    smap = build_sphere_map(surface(directions * 100.0, faces), directions * 100.0)

    assert (smap.n_source, smap.n_target) == (n, n)
    np.testing.assert_array_equal(smap.corners, np.repeat(np.arange(n)[:, None], 3, axis=1))
    np.testing.assert_array_equal(smap.weights, np.tile([1.0, 0.0, 0.0], (n, 1)))
    assert not smap.fallback.any()
    np.testing.assert_array_equal(smap.nearest_source, np.arange(n))
    np.testing.assert_array_equal(smap.nearest_target, np.arange(n))
    assert smap.flipped is False

    field = np.random.default_rng(order).normal(size=n)
    np.testing.assert_array_equal(resample(field, smap), field)
    np.testing.assert_array_equal(resample(field, smap, valid=np.ones(n, dtype=bool)), field)


def test_identity_survives_storage_as_float32_and_another_radius() -> None:
    """The template is read from a file (float32, radius 100) on one side and may be held as
    unit vectors on the other: normalising twice moves a coordinate by a rounding step."""
    directions, faces = icosphere(4)
    stored = (directions * 100.0).astype(np.float32).astype(np.float64)
    smap = build_sphere_map(surface(stored, faces), unit_sphere(stored))
    n = len(directions)
    np.testing.assert_array_equal(smap.corners[:, 0], np.arange(n))
    np.testing.assert_array_equal(smap.weights, np.tile([1.0, 0.0, 0.0], (n, 1)))
    field = np.random.default_rng(1).normal(size=n).astype(np.float32)
    out = resample(field, smap)
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, field.astype(np.float64))


def test_a_target_just_off_a_vertex_is_interpolated_not_snapped() -> None:
    """The coincidence rule covers rounding only: a point a millionth of the mesh spacing
    away from a vertex is found in a face, with nearly all weight on that vertex."""
    directions, faces = icosphere(3)
    source = surface(directions, faces)
    step = 1e3 * COINCIDENT
    target = directions[:50] + step * random_directions(np.random.default_rng(2), 50)
    smap = build_sphere_map(source, target)
    assert not smap.fallback.any()
    assert (smap.corners[:, 0] != smap.corners[:, 1]).all()  # rows are faces
    own = (smap.corners == np.arange(50)[:, None]) * smap.weights
    assert (own.sum(axis=1) > 1 - 1e-6).all()
    assert (smap.weights.max(axis=1) < 1.0).all()


# ---------------------------------------------------------------------------------------
# the known warp


WARP_TOLERANCE_MM = {4: 2.5e-3, 5: 6.5e-4}
"""Largest error of the barycentric value of the field ``expected_thickness`` (2.0 to 3.0
mm), by the order of the source icosphere; measured 2.0e-3 and 5.1e-4 mm."""


@pytest.mark.parametrize("order", [4, 5])
def test_smooth_field_through_the_known_warp(order: int) -> None:
    source = warped_sphere(order)
    registered = unit_sphere(source.vertices)
    values = expected_thickness(registered)
    target = icosphere(4)[0] @ rotation(0.3, -0.8, 1.9).T
    smap = build_sphere_map(source, target * 100.0)

    assert not smap.fallback.any()
    assert smap.flipped is False
    truth = expected_thickness(target)
    barycentric = np.abs(resample(values, smap) - truth)
    nearest = np.abs(values[smap.nearest_source] - truth)
    assert barycentric.max() < WARP_TOLERANCE_MM[order]
    # linear interpolation is second order in the spacing, the nearest vertex first order
    assert barycentric.max() < nearest.max() / 10
    assert barycentric.mean() < nearest.mean() / 10


def test_halving_the_spacing_quarters_the_error() -> None:
    target = icosphere(4)[0] @ rotation(0.3, -0.8, 1.9).T
    errors = []
    for order in (4, 5):
        source = warped_sphere(order)
        values = expected_thickness(unit_sphere(source.vertices))
        out = resample(values, build_sphere_map(source, target))
        errors.append(np.abs(out - expected_thickness(target)).mean())
    assert 3.0 < errors[0] / errors[1] < 5.0


def test_weights_are_barycentric_coordinates_of_the_target_point() -> None:
    source = warped_sphere(4)
    source_unit = unit_sphere(source.vertices)
    target = random_directions(np.random.default_rng(3), 5000)
    smap = build_sphere_map(source, target)

    assert not smap.fallback.any()
    assert (smap.weights >= 0).all()
    np.testing.assert_allclose(smap.weights.sum(axis=1), 1.0, rtol=0, atol=4e-16)
    np.testing.assert_allclose(interpolated_points(smap, source_unit), target, rtol=0, atol=1e-13)
    np.testing.assert_allclose(
        smap.weights, solved_weights(source_unit, smap.corners, target), rtol=0, atol=1e-11
    )
    # every row is a face of the mesh, in the face's own corner order
    face_rows = {tuple(face) for face in source.faces.tolist()}
    assert all(tuple(row) in face_rows for row in smap.corners.tolist())
    assert smap.corners.dtype == np.int32
    assert smap.weights.dtype == np.float64
    assert smap.fallback.dtype == np.bool_
    assert smap.nearest_source.dtype == np.int32
    assert smap.nearest_target.dtype == np.int32
    assert smap.nearest_target.shape == (source.vertices.shape[0],)


UNIT = st.floats(0.0, 1.0, allow_nan=False, width=64)
ANGLE = st.floats(-np.pi, np.pi, allow_nan=False, width=64)


@settings(max_examples=150, deadline=None)
@given(
    order=st.integers(0, 3),
    angles=st.tuples(ANGLE, ANGLE, ANGLE),
    uv=arrays(np.float64, (48, 2), elements=UNIT),
)
def test_every_point_of_a_rotated_icosphere_lies_in_a_face(
    order: int, angles: tuple[float, float, float], uv: np.ndarray
) -> None:
    directions, faces = icosphere(order)
    source_unit = unit_sphere(directions @ rotation(*angles).T)
    # poles, the seam and, without rotation, vertices of the mesh are all drawn
    z, longitude = 2.0 * uv[:, 0] - 1.0, 2.0 * np.pi * uv[:, 1]
    planar = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    points = unit_sphere(
        np.column_stack([planar * np.cos(longitude), planar * np.sin(longitude), z])
    )
    smap = build_sphere_map(surface(source_unit * 100.0, faces), points)

    assert not smap.fallback.any()
    assert smap.flipped is False
    assert (smap.weights >= 0).all()
    np.testing.assert_allclose(smap.weights.sum(axis=1), 1.0, rtol=0, atol=4e-16)
    # a point up to the tolerance outside its face is moved onto the edge: an error of at most
    # the tolerance times the length of an edge, which is 1.1 for the icosahedron
    np.testing.assert_allclose(
        interpolated_points(smap, source_unit), points, rtol=0, atol=2 * CONTAIN_TOLERANCE
    )
    (exhaustive,) = containing_faces(source_unit, faces, points, -10 * CONTAIN_TOLERANCE)
    face_of = {tuple(face): index for index, face in enumerate(faces.tolist())}
    for row, candidates, point in zip(smap.corners.tolist(), exhaustive, points, strict=True):
        if row[0] == row[1]:  # the point is a vertex
            assert np.linalg.norm(source_unit[row[0]] - point) <= COINCIDENT
        else:
            assert face_of[tuple(row)] in candidates


def test_a_point_on_an_edge_goes_to_the_face_with_the_lower_index() -> None:
    """Both faces at an edge contain its midpoint, to within the tolerance; faces are tried
    in ascending index."""
    directions, faces = icosphere(2)
    faces = faces[np.random.default_rng(22).permutation(len(faces))]
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique, inverse = np.unique(edges, axis=0, return_inverse=True)
    assert len(unique) == 480
    lower_face = np.full(480, len(faces))
    np.minimum.at(lower_face, inverse.ravel(), np.tile(np.arange(len(faces)), 3))
    midpoint = directions[unique[:, 0]] + directions[unique[:, 1]]
    midpoint /= np.linalg.norm(midpoint, axis=1, keepdims=True)

    smap = build_sphere_map(surface(directions, faces), midpoint)
    assert not smap.fallback.any()
    np.testing.assert_array_equal(smap.corners, faces[lower_face])
    ends = (smap.corners == unique[:, 0:1]) | (smap.corners == unique[:, 1:2])
    np.testing.assert_array_equal(ends.sum(axis=1), 2)
    np.testing.assert_allclose(smap.weights[ends], 0.5, rtol=0, atol=1e-12)
    np.testing.assert_allclose(smap.weights[~ends], 0.0, rtol=0, atol=1e-12)


def test_where_faces_overlap_the_first_at_the_nearest_vertex_wins() -> None:
    """A large face laid over the mesh, listed first: it takes the points whose nearest
    vertex is one of its corners, where it is the first face tried. Elsewhere the face of the
    mesh is found at the nearest vertex, before the search reaches a corner of the large one."""
    directions, faces = icosphere(2)
    large = icosphere(0)[1][0]  # a face of the icosahedron; its corners are vertices 0 to 11
    stacked = np.concatenate([large[None, :], faces])
    target = random_directions(np.random.default_rng(23), 20_000)
    plain = build_sphere_map(surface(directions, faces), target)
    overlaid = build_sphere_map(surface(directions, stacked), target)

    a, b, c = directions[large]
    in_large = (
        (target @ np.cross(a, b) > 0)
        & (target @ np.cross(b, c) > 0)
        & (target @ np.cross(c, a) > 0)
    )
    at_corner = np.isin(plain.nearest_source, large)
    taken = in_large & at_corner
    assert taken.sum() > 40
    assert (in_large & ~at_corner).sum() > 500
    assert (~in_large & at_corner).sum() > 100
    assert not overlaid.fallback.any()
    np.testing.assert_array_equal(overlaid.corners[taken], np.tile(large, (taken.sum(), 1)))
    np.testing.assert_array_equal(overlaid.corners[~taken], plain.corners[~taken])
    np.testing.assert_array_equal(overlaid.weights[~taken], plain.weights[~taken])
    np.testing.assert_allclose(
        interpolated_points(overlaid, directions), target, rtol=0, atol=1e-13
    )


# ---------------------------------------------------------------------------------------
# faces that must not match


def test_a_folded_face_never_matches_and_is_like_a_missing_one() -> None:
    directions, faces = icosphere(3)
    rng = np.random.default_rng(4)
    folded = np.sort(rng.choice(len(faces), size=40, replace=False))
    centroids = directions[faces[folded]].sum(axis=1)
    target = np.concatenate([unit_sphere(centroids), random_directions(rng, 3000)])

    bent = faces.copy()
    bent[folded] = bent[folded][:, ::-1]
    with_folds = build_sphere_map(surface(directions, bent), target)
    without = build_sphere_map(surface(directions, np.delete(faces, folded, axis=0)), target)

    assert with_folds.flipped is False
    assert with_folds.fallback[:40].all()  # nothing else covers the middle of a folded face
    assert_same_map(with_folds, without)
    folded_rows = {tuple(face) for face in bent[folded].tolist()}
    assert not any(tuple(row) in folded_rows for row in with_folds.corners.tolist())


@pytest.mark.parametrize("position", ["first", "last"])
def test_zero_area_faces_never_match_and_never_raise(position: str) -> None:
    directions, faces = icosphere(2)
    axis = {tuple(v): i for i, v in enumerate(np.round(directions, 12).tolist())}
    east, north, west = axis[(1.0, 0.0, 0.0)], axis[(0.0, 1.0, 0.0)], axis[(-1.0, 0.0, 0.0)]
    a, b, c = (int(i) for i in faces[17])
    flat = np.array(
        [
            [a, a, b],  # two corners are one vertex
            [a, b, b],
            [c, c, c],
            [east, north, west],  # three vertices on a great circle: the determinant is 0
            [c, b, a],  # face 17 a second time, folded
        ]
    )
    rng = np.random.default_rng(5)
    on_edge = directions[[a, east]] + directions[[b, north]]
    on_edge /= np.linalg.norm(on_edge, axis=1, keepdims=True)
    target = np.concatenate(
        [directions[[a, b, c, east, north, west]], on_edge, random_directions(rng, 2000)]
    )

    clean = build_sphere_map(surface(directions, faces), target)
    stacked = np.concatenate([flat, faces] if position == "first" else [faces, flat])
    dirty = build_sphere_map(surface(directions, stacked), target)

    assert not clean.fallback.any()
    assert_same_map(dirty, clean)


def test_a_sphere_of_zero_area_faces_only_falls_back_everywhere() -> None:
    directions, faces = icosphere(2)
    flat = faces.copy()
    flat[:, 2] = flat[:, 1]
    target = random_directions(np.random.default_rng(6), 500)
    smap = build_sphere_map(surface(directions, flat), target)
    assert smap.fallback.all()
    assert smap.flipped is False


# ---------------------------------------------------------------------------------------
# winding


def test_reversed_faces_give_the_same_map_and_set_flipped() -> None:
    source = warped_sphere(4)
    target = random_directions(np.random.default_rng(7), 4000)
    outward = build_sphere_map(source, target)
    assert outward.flipped is False
    assert not outward.fallback.any()

    reversed_map = build_sphere_map(surface(source.vertices, source.faces[:, ::-1]), target)
    assert reversed_map.flipped is True
    assert_same_map(reversed_map, outward)

    # another way of writing the opposite winding: the same faces, corners in another order
    swapped = build_sphere_map(surface(source.vertices, source.faces[:, [0, 2, 1]]), target)
    assert swapped.flipped is True
    np.testing.assert_array_equal(swapped.fallback, outward.fallback)
    np.testing.assert_array_equal(
        np.sort(swapped.corners, axis=1), np.sort(outward.corners, axis=1)
    )
    values = expected_thickness(unit_sphere(source.vertices))
    np.testing.assert_allclose(
        resample(values, swapped), resample(values, outward), rtol=0, atol=1e-14
    )


def test_a_mirrored_sphere_is_flipped_and_maps_the_mirrored_target() -> None:
    source = warped_sphere(4)
    target = random_directions(np.random.default_rng(8), 3000)
    mirror = np.array([-1.0, 1.0, 1.0])
    plain = build_sphere_map(source, target)
    mirrored = build_sphere_map(surface(source.vertices * mirror, source.faces), target * mirror)

    assert mirrored.flipped is True
    assert not mirrored.fallback.any()
    # the reversed faces list their corners the other way round, the weights follow
    np.testing.assert_array_equal(mirrored.corners, plain.corners[:, ::-1])
    np.testing.assert_allclose(mirrored.weights, plain.weights[:, ::-1], rtol=0, atol=1e-13)
    np.testing.assert_array_equal(mirrored.nearest_source, plain.nearest_source)
    np.testing.assert_array_equal(mirrored.nearest_target, plain.nearest_target)


def test_winding_is_decided_by_more_than_half_of_the_faces() -> None:
    directions, faces = icosphere(3)
    half = len(faces) // 2
    target = random_directions(np.random.default_rng(9), 3000)

    def with_reversed(count: int) -> Surface:
        changed = faces.copy()
        changed[:count] = changed[:count][:, ::-1]
        return surface(directions, changed)

    exactly_half = build_sphere_map(with_reversed(half), target)
    assert exactly_half.flipped is False
    assert_same_map(exactly_half, build_sphere_map(surface(directions, faces[half:]), target))

    one_more = build_sphere_map(with_reversed(half + 1), target)
    assert one_more.flipped is True
    # after the reversal the majority is as it was first, and the rest never matches
    assert_same_map(one_more, build_sphere_map(surface(directions, faces[: half + 1]), target))
    assert one_more.fallback.any()
    assert not one_more.fallback.all()


# ---------------------------------------------------------------------------------------
# fallback


def test_points_in_removed_faces_fall_back_to_the_nearest_vertex() -> None:
    source = warped_sphere(3)
    source_unit = unit_sphere(source.vertices)
    rng = np.random.default_rng(10)
    hubs = np.array([5, 77, 300])
    target = np.concatenate([source_unit[hubs], random_directions(rng, 6000)])
    full = build_sphere_map(source, target)
    assert not full.fallback.any()

    removed = np.isin(source.faces, hubs).any(axis=1)
    assert 15 <= removed.sum() <= 18
    holed = build_sphere_map(surface(source.vertices, source.faces[~removed]), target)

    was_in_hole = np.isin(full.corners, hubs).any(axis=1)
    was_in_hole[:3] = False  # the hubs themselves are source vertices, not points in a face
    assert was_in_hole.sum() > 30
    np.testing.assert_array_equal(holed.fallback, was_in_hole)
    np.testing.assert_array_equal(
        holed.corners[was_in_hole], np.repeat(holed.nearest_source[was_in_hole, None], 3, axis=1)
    )
    np.testing.assert_array_equal(
        holed.weights[was_in_hole], np.tile([1.0, 0.0, 0.0], (was_in_hole.sum(), 1))
    )
    for name in ("corners", "weights"):
        np.testing.assert_array_equal(
            getattr(holed, name)[~was_in_hole], getattr(full, name)[~was_in_hole]
        )
    np.testing.assert_array_equal(holed.nearest_source, full.nearest_source)
    np.testing.assert_array_equal(holed.nearest_target, full.nearest_target)

    values = rng.normal(size=len(source_unit))
    np.testing.assert_array_equal(
        resample(values, holed)[was_in_hole], values[holed.nearest_source[was_in_hole]]
    )


def test_a_sphere_without_faces_maps_by_nearest_vertex_alone() -> None:
    directions, _ = icosphere(2)
    target = np.concatenate([directions[:4], random_directions(np.random.default_rng(11), 300)])
    smap = build_sphere_map(surface(directions, np.empty((0, 3), dtype=np.int64)), target)
    assert smap.flipped is False
    np.testing.assert_array_equal(smap.fallback, np.arange(304) >= 4)  # the first four are vertices
    np.testing.assert_array_equal(smap.corners, np.repeat(smap.nearest_source[:, None], 3, axis=1))
    np.testing.assert_array_equal(smap.nearest_source[:4], np.arange(4))


def test_more_neighbours_only_ever_resolve_more_points() -> None:
    """A strong warp shears the faces, so the face of a point is often not at its nearest
    vertices. What few neighbours resolve, more neighbours resolve to the same face."""
    source = warped_sphere(4, warp=1.5)
    target = random_directions(np.random.default_rng(12), 20_000)
    maps = [build_sphere_map(source, target, k=k) for k in (1, 8, 32)]
    counts = [int(smap.fallback.sum()) for smap in maps]  # measured 3919, 377, 0
    assert counts[0] > 3000 > 1000 > counts[1] > 100
    assert counts[2] == 0
    for fewer, more in pairwise(maps):
        found = ~fewer.fallback
        assert not more.fallback[found].any()
        np.testing.assert_array_equal(fewer.corners[found], more.corners[found])
        np.testing.assert_array_equal(fewer.weights[found], more.weights[found])
        np.testing.assert_array_equal(fewer.nearest_source, more.nearest_source)
        np.testing.assert_array_equal(fewer.nearest_target, more.nearest_target)
    # the exhaustive search agrees that every point has a face
    source_unit = unit_sphere(source.vertices)
    lost = np.flatnonzero(maps[1].fallback)
    (inside,) = containing_faces(source_unit, source.faces, target[lost], CONTAIN_TOLERANCE)
    assert all(len(faces) == 1 for faces in inside)


def test_k_larger_than_the_sphere() -> None:
    directions, faces = icosphere(0)
    target = random_directions(np.random.default_rng(13), 400)
    smap = build_sphere_map(surface(directions, faces), target, k=50)
    assert not smap.fallback.any()
    np.testing.assert_allclose(interpolated_points(smap, directions), target, rtol=0, atol=1e-13)


# ---------------------------------------------------------------------------------------
# nearest vertices


def test_nearest_vertices_against_brute_force() -> None:
    source = warped_sphere(4)
    source_unit = unit_sphere(source.vertices)
    target = icosphere(3)[0] @ rotation(1.1, 0.4, -2.2).T
    smap = build_sphere_map(source, target * 50.0)

    distance = np.linalg.norm(target[:, None, :] - source_unit[None, :, :], axis=2)
    np.testing.assert_array_equal(smap.nearest_source, distance.argmin(axis=1))
    np.testing.assert_array_equal(smap.nearest_target, distance.argmin(axis=0))
    assert (smap.n_source, smap.n_target) == (2562, 642)


def test_equally_near_vertices_resolve_to_the_lowest_index() -> None:
    """Pairs of vertices mirrored in the equator are exactly equally far from a point on it."""
    rng = np.random.default_rng(14)
    north = random_directions(rng, 300)
    north[:, 2] = np.abs(north[:, 2]) + 0.05
    north /= np.linalg.norm(north, axis=1, keepdims=True)
    paired = np.concatenate([north, north * [1.0, 1.0, -1.0]])[rng.permutation(600)]
    angle = rng.uniform(0, 2 * np.pi, size=400)
    equator = np.column_stack([np.cos(angle), np.sin(angle), np.zeros(400)])
    no_faces = np.empty((0, 3), dtype=np.int64)

    def lowest_nearest(points: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        delta = unit_sphere(candidates)[None, :, :] - unit_sphere(points)[:, None, :]
        d2 = delta[..., 0] ** 2 + delta[..., 1] ** 2 + delta[..., 2] ** 2
        assert ((d2 == d2.min(axis=1, keepdims=True)).sum(axis=1) == 2).all()  # real ties
        return d2.argmin(axis=1)  # the first of equal minima

    to_equator = build_sphere_map(surface(paired, no_faces), equator)
    np.testing.assert_array_equal(to_equator.nearest_source, lowest_nearest(equator, paired))
    from_equator = build_sphere_map(surface(equator, no_faces), paired)
    np.testing.assert_array_equal(from_equator.nearest_target, lowest_nearest(equator, paired))


def test_lowest_index_among_more_equal_vertices_than_neighbours() -> None:
    """Vertices listed many times over (more often than ``k``): the first listing is the
    nearest, in both directions."""
    rng = np.random.default_rng(15)
    places = random_directions(rng, 12)
    listing = rng.permutation(np.repeat(np.arange(12), 40))  # 480 vertices at 12 places
    first = np.array([np.flatnonzero(listing == place)[0] for place in range(12)])
    repeated = places[listing]
    no_faces = np.empty((0, 3), dtype=np.int64)

    onto_places = build_sphere_map(surface(repeated, no_faces), places, k=8)
    np.testing.assert_array_equal(onto_places.nearest_source, first)
    np.testing.assert_array_equal(onto_places.corners, np.repeat(first[:, None], 3, axis=1))
    assert not onto_places.fallback.any()  # each place is a source vertex
    np.testing.assert_array_equal(onto_places.nearest_target, listing)

    onto_repeated = build_sphere_map(surface(places, no_faces), repeated, k=8)
    np.testing.assert_array_equal(onto_repeated.nearest_target, first)
    np.testing.assert_array_equal(onto_repeated.nearest_source, listing)


# ---------------------------------------------------------------------------------------
# resample


def hand_map() -> SphereMap:
    return SphereMap(
        n_source=4,
        n_target=6,
        corners=np.array(
            [[0, 1, 2], [0, 1, 2], [1, 2, 3], [3, 3, 3], [0, 1, 3], [0, 1, 2]], dtype=np.int32
        ),
        weights=np.array(
            [
                [0.5, 0.25, 0.25],
                [0.25, 0.5, 0.25],
                [0.25, 0.25, 0.5],
                [1.0, 0.0, 0.0],
                [0.25, 0.375, 0.375],
                [0.0, 0.5, 0.5],
            ]
        ),
        fallback=np.array([False, False, False, True, False, False]),
        nearest_source=np.array([0, 1, 3, 3, 1, 2], dtype=np.int32),
        nearest_target=np.array([0, 1, 2, 3], dtype=np.int32),
        flipped=False,
    )


def test_resample_without_flags_is_the_weighted_sum() -> None:
    out = resample(np.array([1.0, 2.0, 4.0, 8.0]), hand_map())
    np.testing.assert_array_equal(out, [2.0, 2.25, 5.5, 8.0, 4.0, 3.0])


def test_the_valid_rule_on_both_sides_of_the_threshold() -> None:
    values = np.array([1.0, 2.0, 4.0, 8.0])
    valid = np.array([False, True, True, True])

    # vertex 0 is not valid; the rows keep 0.5, 0.75, 1, 1, 0.75 and 1 of their weight
    def close(out: np.ndarray, expected: list[float]) -> None:
        np.testing.assert_array_equal(np.isnan(out), np.isnan(expected))
        np.testing.assert_allclose(out, expected, rtol=0, atol=1e-15, equal_nan=True)

    at = resample(values, hand_map(), valid=valid, min_valid_weight=0.5)
    close(at, [3.0, 8.0 / 3.0, 5.5, 8.0, 5.0, 3.0])
    above = resample(
        values, hand_map(), valid=valid, min_valid_weight=float(np.nextafter(0.5, 1.0))
    )
    close(above, [np.nan, 8.0 / 3.0, 5.5, 8.0, 5.0, 3.0])
    below = resample(
        values, hand_map(), valid=valid, min_valid_weight=float(np.nextafter(0.5, 0.0))
    )
    np.testing.assert_array_equal(below, at)
    strict = resample(values, hand_map(), valid=valid, min_valid_weight=1.0)
    close(strict, [np.nan, np.nan, 5.5, 8.0, np.nan, 3.0])
    lenient = resample(values, hand_map(), valid=valid, min_valid_weight=0.0)
    np.testing.assert_array_equal(lenient, at)
    # the default threshold is 0.5
    np.testing.assert_array_equal(resample(values, hand_map(), valid=valid), at)


def test_a_target_with_no_valid_corner_is_nan_whatever_the_threshold() -> None:
    values = np.array([1.0, 2.0, 4.0, 8.0])
    valid = np.array([True, True, True, False])
    for threshold in (0.0, 0.5):
        out = resample(values, hand_map(), valid=valid, min_valid_weight=threshold)
        # row 3 is a fallback row: its one vertex is not valid
        np.testing.assert_allclose(
            out, [2.0, 2.25, 3.0, np.nan, 1.6, 3.0], rtol=0, atol=1e-15, equal_nan=True
        )
    nothing = resample(values, hand_map(), valid=np.zeros(4, dtype=bool), min_valid_weight=0.0)
    assert np.isnan(nothing).all()


@pytest.mark.parametrize("junk", [np.nan, np.inf, -np.inf, 0.0, 1e30])
def test_values_at_corners_that_do_not_count_never_leak(junk: float) -> None:
    values = np.array([1.0, 2.0, 4.0, 8.0])
    valid = np.array([False, True, True, True])
    expected = resample(values, hand_map(), valid=valid)
    dirty = values.copy()
    dirty[0] = junk
    np.testing.assert_array_equal(resample(dirty, hand_map(), valid=valid), expected)
    # without flags, the last row gives vertex 0 the weight 0: its value is not read either
    plain = resample(dirty, hand_map())
    assert plain[5] == 3.0
    assert plain[3] == 8.0


def test_all_true_flags_change_nothing() -> None:
    source = warped_sphere(3)
    values = expected_thickness(unit_sphere(source.vertices))
    smap = build_sphere_map(source, random_directions(np.random.default_rng(16), 3000))
    flagged = resample(values, smap, valid=np.ones(len(values), dtype=bool), min_valid_weight=1.0)
    np.testing.assert_array_equal(flagged, resample(values, smap))


def test_flags_on_a_real_map_drop_weight_and_renormalise() -> None:
    source = warped_sphere(3)
    values = expected_thickness(unit_sphere(source.vertices))
    target = random_directions(np.random.default_rng(17), 3000)
    smap = build_sphere_map(source, target)
    valid = np.random.default_rng(18).random(len(values)) > 0.3

    out = resample(np.where(valid, values, -1000.0), smap, valid=valid, min_valid_weight=0.5)
    kept = np.where(valid[smap.corners], smap.weights, 0.0)
    total = kept.sum(axis=1)
    expected = np.where(
        total >= 0.5, (kept * values[smap.corners]).sum(axis=1) / np.maximum(total, 1e-300), np.nan
    )
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-13, equal_nan=True)
    assert 100 < np.isnan(out).sum() < 2000
    # the value at a vertex that does not count would pull the result far below the field
    assert np.nanmin(out) > 1.99


def test_resample_refuses_arrays_of_another_length() -> None:
    with pytest.raises(ValueError, match="5 values for a sphere of 4 vertices"):
        resample(np.zeros(5), hand_map())
    # under the import hook of the tests jaxtyping sees the two lengths first
    with pytest.raises((ValueError, TypeCheckError)):
        resample(np.zeros(4), hand_map(), valid=np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="3 flags for a sphere of 4 vertices"):
        inspect.unwrap(resample)(np.zeros(4), hand_map(), valid=np.ones(3, dtype=bool))


# ---------------------------------------------------------------------------------------
# determinism


def test_identical_output_across_runs_and_inputs_left_alone() -> None:
    source = warped_sphere(4)
    target = icosphere(4)[0] @ rotation(0.2, 0.7, -0.4).T
    vertices, faces = source.vertices.copy(), source.faces.copy()
    for array in (source.vertices, source.faces, target):
        array.setflags(write=False)

    first = build_sphere_map(source, target)
    second = build_sphere_map(surface(vertices.copy(), faces.copy()), target.copy())
    for name in MAP_ARRAYS:
        assert getattr(first, name).tobytes() == getattr(second, name).tobytes(), name
    np.testing.assert_array_equal(source.vertices, vertices)
    np.testing.assert_array_equal(source.faces, faces)

    values = expected_thickness(unit_sphere(vertices))
    values.setflags(write=False)
    assert resample(values, first).tobytes() == resample(values, second).tobytes()


def test_faces_as_int32_give_the_same_map() -> None:
    source = warped_sphere(3)
    target = random_directions(np.random.default_rng(19), 1000)
    narrow = Surface(vertices=source.vertices, faces=source.faces.astype(np.int32))
    assert_same_map(build_sphere_map(narrow, target), build_sphere_map(source, target))


# ---------------------------------------------------------------------------------------
# through the files of a synthetic subject


@pytest.fixture(scope="module")
def registered_subject(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[SyntheticTemplate, dict[str, SyntheticSubject]]:
    """A subject of 10,242 vertices registered with the default warp to a template of 2,562,
    and a second one whose ``sphere.reg`` files have every face reversed."""
    root = tmp_path_factory.mktemp("registration")
    template = write_synthetic_template(root, order=4, with_avg_area=False)
    subjects = {}
    for name, flip in (("sub-out", False), ("sub-in", True)):
        subject = write_synthetic_subject(
            root, name=name, voxel_size=10.0, shape=(16, 16, 16), subdivisions=5
        )
        add_tier_b(subject, template, flip_winding=flip)
        subjects[name] = subject
    return template, subjects


FILE_TOLERANCE_MM = 2.5e-4
"""Largest error of the resampled thickness of the synthetic subject (10,242 vertices, the
field stored as float32) on the template of 2,562 vertices, at targets whose three corners
all count; measured 1.5e-4 mm."""
FILE_EDGE_TOLERANCE_MM = 0.01
"""The same at the edge of the medial strip, where a target keeps one or two of its corners
and so takes the field from a little beside itself (10 targets per hemisphere); measured
5.6e-3 (lh) and 6.3e-3 mm (rh)."""


@pytest.mark.parametrize("hemi", HEMIS)
def test_synthetic_subject_files_map_as_constructed(
    registered_subject: tuple[SyntheticTemplate, dict[str, SyntheticSubject]], hemi: str
) -> None:
    template, subjects = registered_subject
    subject = subjects["sub-out"]
    sphere = read_surface(subject.path / "surf" / f"{hemi}.sphere.reg")
    target = read_surface(template.path / "surf" / f"{hemi}.sphere.reg").vertices
    thickness = fs.read_morph_data(str(subject.path / "surf" / f"{hemi}.thickness"))
    smap = build_sphere_map(sphere, target)

    assert (smap.n_source, smap.n_target) == (10_242, 2_562)
    assert smap.flipped is False
    assert not smap.fallback.any()

    directions = unit_sphere(target)
    registered = warp_directions(icosphere(5)[0], 0.15)  # where add_tier_b put each vertex
    nearest_template, template_gap = nearest_by_matrix(registered, directions)
    nearest_native, native_gap = nearest_by_matrix(directions, registered)
    assert template_gap.min() > 1e-9
    assert native_gap.min() > 1e-9
    np.testing.assert_array_equal(smap.nearest_target, nearest_template)
    np.testing.assert_array_equal(smap.nearest_source, nearest_native)

    valid = thickness > 0
    out = resample(thickness, smap, valid=valid)
    medial = medial_coordinate(directions, hemi)
    # the medial strip, where the file holds 0, is on opposite sides in the two hemispheres
    assert np.isfinite(out[medial < MEDIAL_WALL_M - 0.05]).all()
    assert np.isnan(out[medial > MEDIAL_WALL_M + 0.05]).all()
    assert 150 < np.isnan(out).sum() < 230  # measured 190 (lh) and 192 (rh)
    seen = np.isfinite(out)
    whole = valid[smap.corners].all(axis=1)
    error = np.abs(out - expected_thickness(directions))
    assert error[whole].max() < FILE_TOLERANCE_MM
    assert 3 <= (seen & ~whole).sum() <= 30  # measured 10 and 10
    assert error[seen & ~whole].max() < FILE_EDGE_TOLERANCE_MM
    # without the flags the zeros of the strip are averaged in, exactly at those targets
    leaked = resample(thickness, smap)
    np.testing.assert_array_equal(leaked[whole], out[whole])
    assert (leaked[seen & ~whole] < out[seen & ~whole] - 0.1).all()


@pytest.mark.parametrize("hemi", HEMIS)
def test_sphere_file_with_reversed_faces_gives_the_same_map(
    registered_subject: tuple[SyntheticTemplate, dict[str, SyntheticSubject]], hemi: str
) -> None:
    template, subjects = registered_subject
    target = read_surface(template.path / "surf" / f"{hemi}.sphere.reg").vertices
    maps = {
        name: build_sphere_map(read_surface(subject.path / "surf" / f"{hemi}.sphere.reg"), target)
        for name, subject in subjects.items()
    }
    assert maps["sub-out"].flipped is False
    assert maps["sub-in"].flipped is True
    assert_same_map(maps["sub-in"], maps["sub-out"])


# ---------------------------------------------------------------------------------------
# the real subject on the real fsaverage sphere

FSAVERAGE_CACHE = "ds000102-fsaverage"  # where conftest.real_template keeps the files
REAL_FALLBACKS = {"lh": 3, "rh": 2}  # of 40,962 fsaverage6 targets, measured 2026-11-13
N_FSAVERAGE6 = 40_962


def check_real_hemisphere(subject: Path, fsaverage: Path, hemi: str, n_sample: int) -> None:
    sphere = read_surface(subject / "surf" / f"{hemi}.sphere.reg")
    template = read_surface(fsaverage / "surf" / f"{hemi}.sphere.reg")
    assert template.vertices.shape == (163_842, 3)
    target = template.vertices[:N_FSAVERAGE6]
    started = time.perf_counter()
    smap = build_sphere_map(sphere, target)
    seconds = time.perf_counter() - started

    source_unit, target_unit = unit_sphere(sphere.vertices), unit_sphere(target)
    faces = sphere.faces
    assert smap.flipped is False  # the right sphere is wound outward like the left one
    assert int(smap.fallback.sum()) == REAL_FALLBACKS[hemi]
    assert (smap.weights >= 0).all()
    np.testing.assert_allclose(smap.weights.sum(axis=1), 1.0, rtol=0, atol=4e-16)
    resolved = ~smap.fallback
    np.testing.assert_allclose(
        interpolated_points(smap, source_unit)[resolved], target_unit[resolved], rtol=0, atol=1e-12
    )

    # a random sample of targets, and every fallback, against all faces of the sphere
    rng = np.random.default_rng(20261113 + HEMIS.index(hemi))
    sample = np.union1d(
        rng.choice(N_FSAVERAGE6, size=n_sample, replace=False), np.flatnonzero(smap.fallback)
    )
    points = target_unit[sample]
    loose, strict = containing_faces(
        source_unit, faces, points, -10 * CONTAIN_TOLERANCE, 10 * CONTAIN_TOLERANCE
    )
    face_of = {tuple(face): index for index, face in enumerate(faces.tolist())}
    nearest, gap = nearest_by_matrix(points, source_unit)
    several = uncovered = 0
    for row, (target_vertex, candidates, certain) in enumerate(
        zip(sample, loose, strict, strict=True)
    ):
        corners = tuple(smap.corners[target_vertex].tolist())
        several += len(certain) > 1
        uncovered += len(candidates) == 0
        if gap[row] > 1e-12:
            assert smap.nearest_source[target_vertex] == nearest[row]
        if smap.fallback[target_vertex]:
            # the search was right to give up: no face that contains the point touches one
            # of its eight nearest vertices
            assert corners == (nearest[row],) * 3
            eight = np.argsort(-(points[row] @ source_unit.T), kind="stable")[:8]
            assert not np.isin(faces[candidates], eight).any()
            continue
        assert face_of[corners] in candidates
    on_faces = sample[~smap.fallback[sample]]
    np.testing.assert_allclose(
        smap.weights[on_faces],
        solved_weights(source_unit, smap.corners[on_faces], target_unit[on_faces]),
        rtol=0,
        atol=1e-9,
    )

    # the other direction, on a sample of native vertices
    native = rng.choice(len(source_unit), size=n_sample, replace=False)
    nearest_template, template_gap = nearest_by_matrix(source_unit[native], target_unit)
    clear = template_gap > 1e-12
    np.testing.assert_array_equal(smap.nearest_target[native][clear], nearest_template[clear])
    assert clear.sum() > n_sample - 5

    # the interpolated thickness against the independent weights
    thickness = fs.read_morph_data(str(subject / "surf" / f"{hemi}.thickness")).astype(np.float64)
    assert len(thickness) == smap.n_source
    independent = (
        solved_weights(source_unit, smap.corners[on_faces], target_unit[on_faces])
        * thickness[smap.corners[on_faces]]
    ).sum(axis=1)
    np.testing.assert_allclose(resample(thickness, smap)[on_faces], independent, rtol=0, atol=1e-8)

    reversed_map = build_sphere_map(Surface(sphere.vertices, faces[:, ::-1].copy()), target)
    assert reversed_map.flipped is True
    assert_same_map(reversed_map, smap)
    print(
        f"{hemi}: {smap.n_source} native vertices, {int(smap.fallback.sum())} fallbacks, "
        f"of {len(sample)} sampled targets {several} in more than one face and {uncovered} in "
        f"none, map in {seconds:.2f} s"
    )


def check_real_template_onto_itself(fsaverage: Path) -> None:
    template = read_surface(fsaverage / "surf" / "lh.sphere.reg")
    n = len(template.vertices)
    smap = build_sphere_map(template, template.vertices)
    np.testing.assert_array_equal(smap.corners[:, 0], np.arange(n))
    np.testing.assert_array_equal(smap.weights, np.tile([1.0, 0.0, 0.0], (n, 1)))
    assert not smap.fallback.any()
    np.testing.assert_array_equal(smap.nearest_target, np.arange(n))
    field = np.random.default_rng(21).normal(size=n)
    np.testing.assert_array_equal(resample(field, smap), field)

    # fsaverage6 targets in the full sphere: each is its own vertex, by nesting
    nested = build_sphere_map(template, template.vertices[:N_FSAVERAGE6])
    np.testing.assert_array_equal(nested.nearest_source, np.arange(N_FSAVERAGE6))
    assert (nested.nearest_target[:N_FSAVERAGE6] == np.arange(N_FSAVERAGE6)).all()


def sphere_files_present(subject: Path, fsaverage: Path) -> bool:
    return all(
        (root / "surf" / f"{hemi}.{name}").is_file()
        for hemi in HEMIS
        for root, names in ((subject, ("sphere.reg", "thickness")), (fsaverage, ("sphere.reg",)))
        for name in names
    )


@pytest.mark.parametrize("hemi", HEMIS)
def test_cached_real_subject_agrees_with_exhaustive_search(
    cached_subjects: Path, hemi: str
) -> None:
    subject, fsaverage = cached_subjects / "ds000102-sub-01", cached_subjects / FSAVERAGE_CACHE
    if not sphere_files_present(subject, fsaverage):
        pytest.skip("the real subject and fsaverage sphere are not in the pytest cache")
    check_real_hemisphere(subject, fsaverage, hemi, n_sample=150)


def test_cached_real_template_onto_itself(cached_subjects: Path) -> None:
    fsaverage = cached_subjects / FSAVERAGE_CACHE
    if not (fsaverage / "surf" / "lh.sphere.reg").is_file():
        pytest.skip("the real fsaverage sphere is not in the pytest cache")
    check_real_template_onto_itself(fsaverage)


@pytest.mark.network
@pytest.mark.parametrize("hemi", HEMIS)
def test_real_subject_agrees_with_exhaustive_search(
    real_subject: Path, real_template: Path, hemi: str
) -> None:
    check_real_hemisphere(real_subject, real_template, hemi, n_sample=2000)


@pytest.mark.network
def test_real_template_onto_itself(real_template: Path) -> None:
    check_real_template_onto_itself(real_template)
