"""Mesh-plane intersection: property tests on triangle soups, closed meshes, vertex hits."""

from __future__ import annotations

import time
from functools import cache
from itertools import pairwise

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from surfcensus.render.slicing import MeshSlicer, plane_segments

AXES = st.sampled_from([0, 1, 2])


@cache
def icosphere(level: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit sphere from a subdivided icosahedron, faces wound with outward normals."""
    t = (1 + 5**0.5) / 2
    v = np.array(
        [
            [-1, t, 0],
            [1, t, 0],
            [-1, -t, 0],
            [1, -t, 0],
            [0, -1, t],
            [0, 1, t],
            [0, -1, -t],
            [0, 1, -t],
            [t, 0, -1],
            [t, 0, 1],
            [-t, 0, -1],
            [-t, 0, 1],
        ],
        dtype=float,
    )
    f = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ]
    )
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    for _ in range(level):
        edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        unique, inverse = np.unique(edges, axis=0, return_inverse=True)
        mid = v[unique].sum(axis=1)
        mid /= np.linalg.norm(mid, axis=1, keepdims=True)
        ab, bc, ca = (len(v) + inverse.reshape(3, -1)[k] for k in range(3))
        v = np.concatenate([v, mid])
        f = np.concatenate(
            [
                np.stack([f[:, 0], ab, ca], axis=1),
                np.stack([f[:, 1], bc, ab], axis=1),
                np.stack([f[:, 2], ca, bc], axis=1),
                np.stack([ab, bc, ca], axis=1),
            ]
        )
    tri = v[f]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    assert (np.einsum("ij,ij->i", normals, tri.mean(axis=1)) > 0).all()
    v.setflags(write=False)
    f.setflags(write=False)
    return v, f


def rotation(seed: int) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q * np.linalg.det(q)


def crossing_faces(vertices: np.ndarray, faces: np.ndarray, axis: int, position: float):
    """Indices of the faces with a sign change under the >= 0 rule, by a plain loop."""
    out = []
    for i, face in enumerate(faces):
        positive = [vertices[k, axis] - position >= 0 for k in face]
        if any(positive) and not all(positive):
            out.append(i)
    return out


def distance_to_edge(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from a point to the segment a-b: zero means collinear with parameter in [0, 1]."""
    ab = b - a
    length2 = float(ab @ ab)
    t = 0.0 if length2 == 0 else float(np.clip((point - a) @ ab / length2, 0.0, 1.0))
    return float(np.linalg.norm(a + t * ab - point))


def signed_area(segments: np.ndarray, axis: int) -> float:
    """Area enclosed by the segments, positive when they run counterclockwise in (u, v)."""
    u, v = (axis + 1) % 3, (axis + 2) % 3
    origin = segments[:, 0].mean(axis=0) if len(segments) else np.zeros(3)
    s = segments - origin
    return float(0.5 * np.sum(s[:, 0, u] * s[:, 1, v] - s[:, 1, u] * s[:, 0, v]))


def total_length(segments: np.ndarray) -> float:
    return float(np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1).sum())


def degrees(segments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per distinct endpoint: how many segments start there and how many end there."""
    n = len(segments)
    points = np.concatenate([segments[:, 0], segments[:, 1]])
    unique, inverse = np.unique(points, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    return (
        np.bincount(inverse[:n], minlength=len(unique)),
        np.bincount(inverse[n:], minlength=len(unique)),
    )


@st.composite
def soups(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, int, float]:
    """A random triangle soup and a plane; lattice soups make planes through vertices common."""
    n = draw(st.integers(3, 12))
    m = draw(st.integers(1, 20))
    if draw(st.booleans()):
        elements = st.integers(-3, 3).map(float)
        positions = st.integers(-8, 8).map(lambda k: k / 2)
    else:
        elements = st.floats(-10, 10, allow_nan=False)
        positions = st.floats(-11, 11, allow_nan=False)
    vertices = draw(arrays(np.float64, (n, 3), elements=elements))
    faces = draw(arrays(np.int64, (m, 3), elements=st.integers(0, n - 1)))
    axis = draw(AXES)
    position = draw(st.one_of(positions, st.sampled_from(vertices[:, axis].tolist())))
    return vertices, faces, axis, position


@settings(max_examples=300, deadline=None)
@given(soups())
def test_triangle_soup_properties(soup: tuple[np.ndarray, np.ndarray, int, float]) -> None:
    vertices, faces, axis, position = soup
    vertices_before, faces_before = vertices.copy(), faces.copy()

    segments = plane_segments(vertices, faces, axis, position)
    from_slicer = MeshSlicer(vertices, faces).segments(axis, position)

    assert np.array_equal(vertices, vertices_before) and np.array_equal(faces, faces_before)
    assert segments.dtype == np.float64 and segments.ndim == 3 and segments.shape[1:] == (2, 3)
    assert np.array_equal(segments, from_slicer)
    assert np.isfinite(segments).all()
    assert (segments[..., axis] == position).all()

    # one segment per face with a sign change, in ascending face order
    source = crossing_faces(vertices, faces, axis, position)
    assert len(segments) == len(source)
    for segment, face in zip(segments, faces[source], strict=True):
        tri = vertices[face]
        positive = tri[:, axis] >= position
        leaving = [k for k in range(3) if positive[k] and not positive[(k + 1) % 3]]
        entering = [k for k in range(3) if not positive[k] and positive[(k + 1) % 3]]
        assert len(leaving) == 1 and len(entering) == 1
        # starts on the edge where the winding leaves the positive side, ends where it returns
        for point, k in ((segment[0], leaving[0]), (segment[1], entering[0])):
            assert distance_to_edge(point, tri[k], tri[(k + 1) % 3]) < 1e-9


@settings(max_examples=150, deadline=None)
@given(
    seed=st.integers(0, 2**32 - 1),
    axis=AXES,
    fraction=st.floats(-0.9, 0.9),
    scales=st.tuples(*[st.floats(0.5, 3.0)] * 3),
    shift=st.tuples(*[st.floats(-50.0, 50.0)] * 3),
)
def test_closed_mesh_gives_closed_counterclockwise_loops(
    seed: int,
    axis: int,
    fraction: float,
    scales: tuple[float, float, float],
    shift: tuple[float, float, float],
) -> None:
    """A rotated, translated ellipsoid: every endpoint ends one segment and starts another."""
    sphere, faces = icosphere(4)
    rot = rotation(seed)
    vertices = (sphere * scales) @ rot.T + shift
    half_width = float(np.linalg.norm(rot[axis] * scales))  # of the ellipsoid along the axis
    position = shift[axis] + fraction * half_width
    assume(not np.any(vertices[:, axis] == position))

    segments = MeshSlicer(vertices, faces).segments(axis, position)
    assert np.array_equal(segments, plane_segments(vertices, faces, axis, position))
    assert len(segments) > 0 and (segments[..., axis] == position).all()
    starts, ends = degrees(segments)
    assert (starts == 1).all() and (ends == 1).all()

    # the section of an ellipsoid is an ellipse of known area; positive means counterclockwise
    central_area = np.pi * float(np.prod(scales)) / half_width
    area = signed_area(segments, axis)
    assert area > 0
    assert abs(area - central_area * (1 - fraction**2)) < 0.01 * central_area


@settings(max_examples=150, deadline=None)
@given(
    seed=st.integers(0, 2**32 - 1),
    axis=AXES,
    vertex=st.integers(0, 641),
    scales=st.tuples(*[st.floats(0.5, 3.0)] * 3),
    shift=st.tuples(*[st.floats(-50.0, 50.0)] * 3),
)
def test_plane_through_a_vertex_of_a_closed_mesh(
    seed: int,
    axis: int,
    vertex: int,
    scales: tuple[float, float, float],
    shift: tuple[float, float, float],
) -> None:
    sphere, faces = icosphere(3)
    rot = rotation(seed)
    vertices = (sphere * scales) @ rot.T + shift
    position = float(vertices[vertex, axis])

    segments = plane_segments(vertices, faces, axis, position)
    assert np.array_equal(segments, MeshSlicer(vertices, faces).segments(axis, position))
    assert np.isfinite(segments).all()
    assert (segments[..., axis] == position).all()
    assert len(segments) == len(crossing_faces(vertices, faces, axis, position))
    # nothing only when no vertex is on the negative side: the plane touches the lowest vertex
    assert (len(segments) == 0) == (position == vertices[:, axis].min())
    # contours still chain: each point starts as many segments as it ends
    starts, ends = degrees(segments)
    assert np.array_equal(starts, ends)

    half_width = float(np.linalg.norm(rot[axis] * scales))
    central_area = np.pi * float(np.prod(scales)) / half_width
    fraction = (position - shift[axis]) / half_width
    area = signed_area(segments, axis)
    assert abs(area - central_area * (1 - fraction**2)) < 0.03 * central_area


def test_unit_sphere_contour_length_converges() -> None:
    offsets = (-0.8, -0.3, 0.1, 0.55, 0.9)
    worst = []
    for level in (2, 3, 4, 5):
        vertices, faces = icosphere(level)
        errors = []
        for axis in range(3):
            for d in offsets:
                assert not np.any(vertices[:, axis] == d)
                expected = 2 * np.pi * np.sqrt(1 - d**2)
                length = total_length(plane_segments(vertices, faces, axis, d))
                assert length < expected  # the mesh is inscribed in the sphere
                errors.append(1 - length / expected)
        worst.append(max(errors))
    assert all(b < a / 2 for a, b in pairwise(worst))
    assert worst[-1] < 2e-3


def test_translated_and_rotated_sphere() -> None:
    sphere, faces = icosphere(4)
    centre = np.array([12.5, -80.25, 3.0])
    vertices = 40.0 * sphere @ rotation(7).T + centre
    slicer = MeshSlicer(vertices, faces)
    for axis in range(3):
        for d in (-30.3, -5.1, 0.2, 17.7, 33.0):
            segments = slicer.segments(axis, centre[axis] + d)
            radius = np.sqrt(40.0**2 - d**2)
            assert total_length(segments) == pytest.approx(2 * np.pi * radius, rel=5e-3)
            assert signed_area(segments, axis) == pytest.approx(np.pi * radius**2, rel=1e-2)
            in_plane = np.delete(segments.reshape(-1, 3) - centre, axis, axis=1)
            distance = np.linalg.norm(in_plane, axis=1)
            assert distance.max() <= radius + 1e-9 and distance.min() > 0.99 * radius


def test_reversing_the_winding_reverses_every_segment() -> None:
    vertices, faces = icosphere(3)
    for axis in range(3):
        forward = plane_segments(vertices, faces, axis, 0.123)
        backward = plane_segments(vertices, faces[:, ::-1], axis, 0.123)
        assert np.array_equal(backward, forward[:, ::-1])
        assert signed_area(backward, axis) < 0 < signed_area(forward, axis)


def test_single_triangle_direction() -> None:
    """Lone vertex above z = 0, normal (2, 2, 1): the segment runs along z x n = (-2, 2, 0)."""
    vertices = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0]])
    expected = np.array([[[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]])
    for face in ([0, 1, 2], [1, 2, 0], [2, 0, 1]):
        assert np.array_equal(plane_segments(vertices, np.array([face]), 2, 0.0), expected)
    flipped = plane_segments(vertices, np.array([[0, 2, 1]]), 2, 0.0)
    assert np.array_equal(flipped, expected[:, ::-1])


@pytest.mark.parametrize(
    ("heights", "expected"),
    [
        ((0.0, -1.0, -1.0), [[[0, 0, 0], [0, 0, 0]]]),  # touches with its only positive vertex
        ((0.0, 1.0, 1.0), []),  # touches from the positive side: all positive
        ((0.0, 0.0, -1.0), [[[1, 0, 0], [0, 0, 0]]]),  # edge in the plane, face below it
        ((0.0, 0.0, 1.0), []),  # edge in the plane, face above it
        ((0.0, 0.0, 0.0), []),  # face in the plane
        ((0.0, 1.0, -1.0), [[[0.5, 0.5, 0], [0, 0, 0]]]),  # through a vertex and the far edge
        ((0.0, -1.0, 1.0), [[[0, 0, 0], [0.5, 0.5, 0]]]),
    ],
)
def test_triangle_with_vertices_on_the_plane(
    heights: tuple[float, float, float], expected: list
) -> None:
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    vertices = np.column_stack([vertices, heights])
    faces = np.array([[0, 1, 2]])
    segments = plane_segments(vertices, faces, 2, 0.0)
    assert np.array_equal(segments, np.array(expected, dtype=float).reshape(-1, 2, 3))
    assert np.array_equal(segments, MeshSlicer(vertices, faces).segments(2, 0.0))


def test_octahedron_sliced_through_its_vertices() -> None:
    vertices = np.array(
        [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float
    )
    faces = np.array(
        [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [1, 0, 5], [2, 1, 5], [3, 2, 5], [0, 3, 5]]
    )
    # equator: only the four lower faces contribute, each its equatorial edge, counterclockwise
    equator = plane_segments(vertices, faces, 2, 0.0)
    assert np.array_equal(equator[:, 0], vertices[[0, 1, 2, 3]])
    assert np.array_equal(equator[:, 1], vertices[[1, 2, 3, 0]])
    assert signed_area(equator, 2) == 2.0
    # through the apex alone: its four faces give zero-length segments at the apex
    apex = plane_segments(vertices, faces, 0, 1.0)
    assert apex.shape == (4, 2, 3) and (apex == vertices[0]).all()
    # through the lowest vertex: everything is on the positive side
    assert plane_segments(vertices, faces, 0, -1.0).shape == (0, 2, 3)
    assert MeshSlicer(vertices, faces).segments(0, -1.0).shape == (0, 2, 3)


def test_empty_results() -> None:
    vertices, faces = icosphere(2)
    slicer = MeshSlicer(vertices, faces)
    for axis in range(3):
        for position in (-1.5, 1.0 + 1e-12, 1.5, np.inf, -np.inf):
            for segments in (
                plane_segments(vertices, faces, axis, position),
                slicer.segments(axis, position),
            ):
                assert segments.shape == (0, 2, 3) and segments.dtype == np.float64
    no_faces = np.empty((0, 3), dtype=int)
    assert plane_segments(vertices, no_faces, 0, 0.0).shape == (0, 2, 3)
    assert MeshSlicer(vertices, no_faces).segments(0, 0.0).shape == (0, 2, 3)


def test_inputs_are_not_modified_and_dtypes_are_accepted() -> None:
    vertices64, faces64 = icosphere(3)
    vertices = (vertices64 * 50).astype(np.float32)
    faces = faces64.astype(">i4")  # as stored in FreeSurfer surface files
    vertices_before, faces_before = vertices.copy(), faces.copy()

    slicer = MeshSlicer(vertices, faces)
    expected = plane_segments(vertices, faces, 1, 3.3)
    assert expected.dtype == np.float64 and len(expected) > 0
    assert np.array_equal(slicer.segments(1, 3.3), expected)
    assert np.array_equal(vertices, vertices_before) and np.array_equal(faces, faces_before)
    assert vertices.dtype == np.float32 and faces.dtype == np.dtype(">i4")

    # the slicer owns a copy of the mesh
    vertices += 100.0
    faces[:] = 0
    assert np.array_equal(slicer.segments(1, 3.3), expected)


def test_malformed_input_is_refused() -> None:
    vertices, faces = icosphere(1)
    with pytest.raises(ValueError, match="axis"):
        plane_segments(vertices, faces, 3, 0.0)
    with pytest.raises(ValueError, match="axis"):
        MeshSlicer(vertices, faces).segments(-1, 0.0)
    with pytest.raises(ValueError, match="face indices"):
        plane_segments(vertices, faces + 1, 0, 0.0)
    with pytest.raises(ValueError, match="face indices"):
        MeshSlicer(vertices, faces - 1)
    with pytest.raises(ValueError, match="finite"):
        MeshSlicer(np.where(vertices > 0.99, np.nan, vertices), faces)
    # a wrong shape is a ValueError, or a TypeError when the jaxtyping import hook is active
    with pytest.raises((ValueError, TypeError)):
        plane_segments(vertices[:, :2], faces, 0, 0.0)


def test_slicer_time_per_call() -> None:
    """Reports the time per call on a folded 327,680-face mesh; the bound is only a guard."""
    sphere, faces = icosphere(7)
    theta, phi = np.arccos(sphere[:, 2]), np.arctan2(sphere[:, 1], sphere[:, 0])
    radius = 80 * (
        1 + 0.25 * np.sin(25 * theta) * np.sin(27 * phi) + 0.05 * np.sin(31 * phi + 17 * theta)
    )
    vertices = sphere * radius[:, None]
    faces = faces[np.random.default_rng(0).permutation(len(faces))]

    start = time.perf_counter()
    slicer = MeshSlicer(vertices, faces)
    build = time.perf_counter() - start
    positions = np.linspace(-60, 60, 61) + 0.137
    counts, times = [], []
    for axis in range(3):
        for position in positions:
            start = time.perf_counter()
            segments = slicer.segments(axis, position)
            times.append(time.perf_counter() - start)
            counts.append(len(segments))
    assert np.array_equal(
        slicer.segments(2, positions[30]), plane_segments(vertices, faces, 2, positions[30])
    )
    per_call = float(np.mean(times))
    print(
        f"\nMeshSlicer: {len(faces)} faces, build {build * 1e3:.0f} ms, "
        f"{per_call * 1e3:.3f} ms per call (median {np.median(times) * 1e3:.3f}, "
        f"max {max(times) * 1e3:.3f}), {np.mean(counts):.0f} segments per slice "
        f"({min(counts)} to {max(counts)})"
    )
    assert min(counts) > 1000
    assert per_call < 0.05
