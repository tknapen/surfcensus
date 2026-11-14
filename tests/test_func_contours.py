"""The white and pial contours of the EPI interface (``func.contours``, work package E1.B),
against the closed form of ``tests/synth_func.py`` and against a cut of the meshes written in
the test with numpy only.

The surfaces go from FreeSurfer's tkr RAS through ``T1.mgz`` to scanner RAS, through the
fsnative transform to the anatomy's world and through the inverse of the grid's affine to
display voxels. In the fixture that is ``inv(G) . Q . (tkr + c_ras)`` with a ``c_ras`` that is
not zero, so a chain that skips ``T1.mgz`` or turns a transform round is off by millimetres.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from synth_func import (  # pyrefly: ignore[missing-import]
    FSNATIVE_TO_T1W,
    GRID_AFFINE,
    GRID_SHAPE,
    SyntheticFuncDataset,
    apply,
    subject_world,
    write_func_dataset,
)

from surfcensus.fsio.surfaces import read_surface
from surfcensus.func import CONTOUR_UNIT, PLANES, SURFACES, contours
from surfcensus.records import Grid, to_matrix4

GRID = Grid(GRID_SHAPE, to_matrix4(GRID_AFFINE))


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> SyntheticFuncDataset:
    return write_func_dataset(tmp_path_factory.mktemp("func"))


@pytest.fixture(scope="module")
def surfaces(dataset: SyntheticFuncDataset) -> dict:
    return dict(
        contours.surfaces_in_display(
            dataset.fs_subjects_dir / "sub-01", to_matrix4(FSNATIVE_TO_T1W), GRID
        )
    )


def closed_form(dataset: SyntheticFuncDataset, name: str) -> tuple[np.ndarray, np.ndarray]:
    subject = dataset.subjects["sub-01"]
    surface = read_surface(subject.path / "surf" / name)
    world = subject_world(subject, surface.vertices)
    return apply(np.linalg.inv(GRID_AFFINE), world), surface.faces


def test_the_surfaces_in_display_voxels_are_the_closed_form(
    dataset: SyntheticFuncDataset, surfaces: dict
) -> None:
    assert list(surfaces) == list(SURFACES)
    subject = dataset.subjects["sub-01"]
    assert np.linalg.norm(subject.c_ras) > 5.0  # so that skipping T1.mgz shows
    for name in SURFACES:
        want, faces = closed_form(dataset, name)
        got, got_faces = surfaces[name]
        np.testing.assert_allclose(got, want, atol=1e-9)
        np.testing.assert_array_equal(got_faces, faces)


def test_each_link_of_the_chain_matters(dataset: SyntheticFuncDataset, surfaces: dict) -> None:
    """With each transform of the chain left out or turned round, the vertices lie more than
    a tenth of a voxel (0.8 mm) from where the chain puts them: the test above would see
    every such error."""
    subject = dataset.subjects["sub-01"]
    tkr = read_surface(subject.path / "surf" / "lh.pial").vertices
    got = surfaces["lh.pial"][0]
    g_inv, q = np.linalg.inv(GRID_AFFINE), FSNATIVE_TO_T1W
    c_ras = np.asarray(subject.c_ras)
    wrong = {
        "no T1.mgz": apply(g_inv @ q, tkr),
        "T1.mgz the wrong way": apply(g_inv @ q, tkr - c_ras),
        "fsnative turned round": apply(g_inv @ np.linalg.inv(q), tkr + c_ras),
        "no fsnative transform": apply(g_inv, tkr + c_ras),
        "grid not inverted": apply(GRID_AFFINE @ q, tkr + c_ras),
    }
    for what, points in wrong.items():
        assert np.abs(points - got).max() > 0.1, what


def test_a_display_to_anatomy_map_is_inverted_with_the_grid(dataset: SyntheticFuncDataset) -> None:
    p = np.eye(4)
    p[:3, 3] = (4.0, -8.0, 2.0)
    moved = contours.surfaces_in_display(
        dataset.fs_subjects_dir / "sub-01", to_matrix4(FSNATIVE_TO_T1W), GRID, to_matrix4(p)
    )
    want, _ = closed_form(dataset, "rh.white")
    np.testing.assert_allclose(moved["rh.white"][0], want - np.array([0.5, -1.0, 0.25]), atol=1e-9)


def independent_crossings(
    vertices: np.ndarray, faces: np.ndarray, axis: int, k: float
) -> np.ndarray:
    """Every point where an edge of the mesh crosses the plane ``x[axis] == k`` (one end at or
    above it, the other below), in the two in-plane coordinates, written here."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    a, b = vertices[edges[:, 0]], vertices[edges[:, 1]]
    da, db = a[:, axis] - k, b[:, axis] - k
    cross = (da >= 0) != (db >= 0)
    t = da[cross] / (da[cross] - db[cross])
    points = a[cross] + t[:, None] * (b[cross] - a[cross])
    keep = [n for n in range(3) if n != axis]
    return points[:, keep]


def test_the_contours_lie_on_the_cut_of_the_closed_form_meshes(
    dataset: SyntheticFuncDataset, surfaces: dict
) -> None:
    """Every decoded point is within 1/64 voxel of a crossing of the closed-form mesh with its
    plane, and every crossing within 1/64 voxel of a decoded point, for every slice of every
    plane and every surface."""
    planes = contours.plane_contours(surfaces, GRID)
    assert [p.plane for p in planes] == list(PLANES)
    tolerance = 1.0 / CONTOUR_UNIT
    n_points = 0
    for axis, plane in enumerate(planes):
        assert plane.slices == GRID_SHAPE[axis]
        cells = contours.read_contours(plane.data)
        assert len(cells) == plane.slices * len(SURFACES)
        assert sum(len(c) for c in cells) == plane.lines
        assert sum(len(line) for c in cells for line in c) == plane.points
        for k in range(plane.slices):
            for s, name in enumerate(SURFACES):
                vertices, faces = closed_form(dataset, name)
                want = independent_crossings(vertices, faces, axis, float(k))
                lines = cells[k * len(SURFACES) + s]
                got = np.concatenate(lines) if lines else np.zeros((0, 2))
                assert (len(got) == 0) == (len(want) == 0), (plane.plane, k, name)
                if not len(want):
                    continue
                d = np.sqrt(((got[:, None, :] - want[None, :, :]) ** 2).sum(axis=2))
                assert d.min(axis=1).max() <= tolerance, (plane.plane, k, name)
                assert d.min(axis=0).max() <= tolerance, (plane.plane, k, name)
                for line in lines:  # every surface of the fixture is closed: loops only
                    np.testing.assert_array_equal(line[0], line[-1])
                n_points += len(got)
    assert n_points > 5000


def test_the_file_layout(surfaces: dict) -> None:
    for plane in contours.plane_contours(surfaces, GRID):
        n_slices, n_surfaces, n_lines, n_points = struct.unpack_from("<4I", plane.data)
        assert (n_slices, n_surfaces, n_lines, n_points) == (
            plane.slices,
            4,
            plane.lines,
            plane.points,
        )
        assert len(plane.data) == 16 + 4 * (n_slices * 4 + 1) + 4 * (n_lines + 1) + 4 * n_points
    with pytest.raises(ValueError):
        contours.read_contours(plane.data + b"\0\0")
    with pytest.raises(ValueError, match="no surface"):
        contours.plane_contours({"lh.white": surfaces["lh.white"]}, GRID)


def test_the_contours_are_the_same_bytes_twice(surfaces: dict) -> None:
    first = [p.data for p in contours.plane_contours(surfaces, GRID)]
    assert first == [p.data for p in contours.plane_contours(surfaces, GRID)]


def test_chain_joins_segments_into_lines() -> None:
    square = np.array(
        [[[0, 0], [1, 0]], [[1, 0], [1, 1]], [[1, 1], [0, 1]], [[0, 1], [0, 0]]], np.int64
    )
    [loop] = contours.chain(square[[2, 0, 3, 1]])
    assert len(loop) == 5 and (loop[0] == loop[-1]).all()
    open_line = np.array([[[5, 5], [6, 5]], [[3, 5], [4, 5]], [[4, 5], [5, 5]], [[7, 7], [7, 7]]])
    [line] = contours.chain(open_line)  # the zero-length segment is dropped
    np.testing.assert_array_equal(line, [[3, 5], [4, 5], [5, 5], [6, 5]])
    lines = contours.chain(np.concatenate([open_line[:3], square]))
    assert [len(x) for x in lines] == [4, 5]  # open lines first, then loops
    assert contours.chain(np.zeros((0, 2, 2), np.int64)) == []


def test_a_missing_surface_is_named(dataset: SyntheticFuncDataset, tmp_path: Path) -> None:
    import shutil

    copy = tmp_path / "sub-01"
    shutil.copytree(dataset.fs_subjects_dir / "sub-01", copy)
    (copy / "surf" / "rh.pial").unlink()
    with pytest.raises(FileNotFoundError, match=r"surf/rh\.pial") as error:
        contours.surfaces_in_display(copy, to_matrix4(FSNATIVE_TO_T1W), GRID)
    assert str(tmp_path) not in str(error.value)
