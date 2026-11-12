"""Tests of render/vertexmap.py: which vertices go into vertex_map.npz, and the file itself."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
from synth import SyntheticSubject  # pyrefly: ignore[missing-import]

from surfcensus.fsio.surfaces import Surface, read_surface
from surfcensus.render import vertexmap

# a region of one box, as low and high corner
BOX = np.array([[[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]])
FAR = (50.0, 50.0, 50.0)
# vertex: (pial position, white position); with a margin of 2 mm the box spans -2 .. 12
PIAL_WHITE = [
    (FAR, FAR),  # 0: neither inside
    ((5.0, 5.0, 5.0), FAR),  # 1: pial only
    (FAR, (1.0, 9.0, 3.0)),  # 2: white only
    ((11.5, 5.0, 5.0), FAR),  # 3: pial in the margin
    ((12.5, 5.0, 5.0), (5.0, -2.5, 5.0)),  # 4: both just outside the margin
    ((2.0, 2.0, 2.0), (3.0, 3.0, 3.0)),  # 5: both inside
    ((-2.0, 12.0, -2.0), FAR),  # 6: exactly on the grown boundary
    ((5.0, 5.0, 12.01), FAR),  # 7: inside in x and y only
]
FACES = np.array([[0, 1, 2]])


def surface(points: list[tuple[float, float, float]]) -> Surface:
    return Surface(vertices=np.array(points, dtype=float), faces=FACES)


def hand_surfaces() -> dict[str, Surface]:
    pial, white = surface([p for p, _ in PIAL_WHITE]), surface([w for _, w in PIAL_WHITE])
    # the right hemisphere has the two surfaces exchanged and a single vertex in range
    return {
        "lh_pial": pial,
        "lh_white": white,
        "rh_pial": surface([FAR, FAR, (4.0, 4.0, 4.0)]),
        "rh_white": surface([FAR, FAR, FAR]),
    }


def test_a_vertex_is_included_when_its_pial_or_white_position_is_in_the_grown_box() -> None:
    vmap = vertexmap.build_vertex_map(hand_surfaces(), BOX, 2.0)
    assert vmap["lh_index"].tolist() == [1, 2, 3, 5, 6]
    assert vmap["rh_index"].tolist() == [2]
    # both positions of every included vertex are stored, also the one outside the box
    np.testing.assert_array_equal(
        vmap["lh_pial"], np.float32([p for p, _ in PIAL_WHITE])[[1, 2, 3, 5, 6]]
    )
    np.testing.assert_array_equal(
        vmap["lh_white"], np.float32([w for _, w in PIAL_WHITE])[[1, 2, 3, 5, 6]]
    )
    np.testing.assert_array_equal(vmap["rh_pial"], np.float32([[4, 4, 4]]))
    np.testing.assert_array_equal(vmap["rh_white"], np.float32([FAR]))


def test_visible_marks_the_vertices_whose_pial_position_is_in_the_region_itself() -> None:
    vmap = vertexmap.build_vertex_map(hand_surfaces(), BOX, 2.0)
    # 1 and 5: pial in the box. 2: white only. 3 and 6: pial in the margin only.
    assert vmap["lh_index"].tolist() == [1, 2, 3, 5, 6]
    assert vmap["lh_visible"].tolist() == [True, False, False, True, False]
    assert vmap["rh_visible"].tolist() == [True]
    # the flag does not depend on the margin, the selection does
    none = vertexmap.build_vertex_map(hand_surfaces(), BOX, 0.0)
    assert (none["lh_index"].tolist(), none["lh_visible"].tolist()) == (
        [1, 2, 5],
        [True, False, True],
    )
    wide = vertexmap.build_vertex_map(hand_surfaces(), BOX, 2.5)
    assert wide["lh_index"].tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert wide["lh_visible"].tolist() == [True, False, False, False, True, False, False]
    surfaces = hand_surfaces()
    # a vertex on the face of the box is visible
    surfaces["rh_pial"] = surface([FAR, (10.0, 0.0, 5.0), (10.01, 0.0, 5.0)])
    vmap = vertexmap.build_vertex_map(surfaces, BOX, 2.0)
    assert (vmap["rh_index"].tolist(), vmap["rh_visible"].tolist()) == ([1, 2], [True, False])


def test_the_region_is_a_union_of_boxes_each_grown_by_the_margin() -> None:
    # two slabs that cross: 20 x 2 x 2 along x and 2 x 20 x 2 along y, as the panels' boxes do
    cross = np.array(
        [[[-10.0, -1.0, -1.0], [10.0, 1.0, 1.0]], [[-1.0, -10.0, -1.0], [1.0, 10.0, 1.0]]]
    )
    pial = [
        (9.0, 0.0, 0.0),  # 0: in the first slab
        (0.0, -9.0, 0.5),  # 1: in the second
        (5.0, 5.0, 0.0),  # 2: in the bounding box of the two, 4 mm from either
        (11.0, 1.5, -1.5),  # 3: within 1 mm of the first along every axis
        (0.0, 11.5, 0.0),  # 4: 1.5 mm beyond the second
        (0.5, 0.5, 0.5),  # 5: in both
    ]
    surfaces = {
        "lh_pial": surface(pial),
        "lh_white": surface([FAR] * len(pial)),
        "rh_pial": surface([FAR] * 3),
        "rh_white": surface([FAR] * 3),
    }
    for margin, index, visible in (
        (0.0, [0, 1, 5], [True, True, True]),
        (1.0, [0, 1, 3, 5], [True, True, False, True]),
        (2.0, [0, 1, 3, 4, 5], [True, True, False, False, True]),
        (4.0, [0, 1, 2, 3, 4, 5], [True, True, False, False, False, True]),
    ):
        vmap = vertexmap.build_vertex_map(surfaces, cross, margin)
        assert (vmap["lh_index"].tolist(), vmap["lh_visible"].tolist()) == (index, visible)
        assert vmap["rh_index"].shape == (0,)


@pytest.mark.parametrize(
    ("margin", "expected"), [(0.0, [1, 2, 5]), (1.5, [1, 2, 3, 5]), (2.5, [1, 2, 3, 4, 5, 6, 7])]
)
def test_the_margin_grows_the_box_on_every_side(margin: float, expected: list[int]) -> None:
    assert vertexmap.build_vertex_map(hand_surfaces(), BOX, margin)["lh_index"].tolist() == expected


def test_keys_dtypes_shapes_and_unset_counterparts() -> None:
    vmap = vertexmap.build_vertex_map(hand_surfaces(), BOX, 2.0)
    assert tuple(vmap) == vertexmap.KEYS
    assert vertexmap.KEYS == (
        "lh_index", "lh_pial", "lh_white", "lh_fsavg6", "lh_visible",
        "rh_index", "rh_pial", "rh_white", "rh_fsavg6", "rh_visible",
    )  # fmt: skip
    for hemi, k in (("lh", 5), ("rh", 1)):
        assert vmap[f"{hemi}_index"].dtype == np.int32 and vmap[f"{hemi}_index"].shape == (k,)
        assert vmap[f"{hemi}_fsavg6"].dtype == np.int32
        assert vmap[f"{hemi}_fsavg6"].tolist() == [-1] * k
        assert vmap[f"{hemi}_visible"].dtype == np.bool_ and vmap[f"{hemi}_visible"].shape == (k,)
        for name in ("pial", "white"):
            assert vmap[f"{hemi}_{name}"].dtype == np.float32
            assert vmap[f"{hemi}_{name}"].shape == (k, 3)


def test_an_empty_hemisphere_gives_empty_arrays_of_the_right_shape() -> None:
    far_box = np.array([[[-91.0, -1.0, -1.0], [-89.0, 1.0, 1.0]]])
    vmap = vertexmap.build_vertex_map(hand_surfaces(), far_box, 0.0)
    assert vmap["lh_index"].shape == (0,) and vmap["lh_pial"].shape == (0, 3)
    assert vmap["lh_white"].shape == (0, 3) and vmap["lh_fsavg6"].shape == (0,)
    assert vmap["lh_visible"].shape == (0,) and vmap["lh_visible"].dtype == np.bool_


def test_pial_and_white_must_have_the_same_vertices() -> None:
    surfaces = hand_surfaces()
    surfaces["lh_white"] = surface([FAR, FAR, FAR])
    with pytest.raises(ValueError, match="vertex count"):
        vertexmap.build_vertex_map(surfaces, BOX, 2.0)


def synth_surfaces(subject: SyntheticSubject) -> dict[str, Surface]:
    return {
        f"{hemi}_{name}": read_surface(subject.path / "surf" / f"{hemi}.{name}")
        for hemi in ("lh", "rh")
        for name in ("pial", "white")
    }


def test_synthetic_subject_against_a_per_vertex_loop(synth_subject: SyntheticSubject) -> None:
    surfaces = synth_surfaces(synth_subject)
    # two boxes that overlap, around the posterior ends of the ellipsoids
    boxes = [
        ((-34.0, -78.0, -30.0), (26.0, -15.0, 30.0)),
        ((-44.0, -68.0, -20.0), (36.0, -25.0, 20.0)),
    ]
    margin = 5.0

    def within(xyz: np.ndarray, grow: float) -> bool:
        return any(
            all(low[a] - grow <= xyz[a] <= high[a] + grow for a in range(3)) for low, high in boxes
        )

    vmap = vertexmap.build_vertex_map(surfaces, np.array(boxes), margin)
    for hemi in ("lh", "rh"):
        pial, white = surfaces[f"{hemi}_pial"].vertices, surfaces[f"{hemi}_white"].vertices
        expected = [
            v for v in range(len(pial)) if within(pial[v], margin) or within(white[v], margin)
        ]
        index = vmap[f"{hemi}_index"]
        assert 100 < len(expected) < len(pial)
        assert index.tolist() == expected
        assert np.all(np.diff(index) > 0)
        np.testing.assert_array_equal(vmap[f"{hemi}_pial"], pial[index].astype(np.float32))
        np.testing.assert_array_equal(vmap[f"{hemi}_white"], white[index].astype(np.float32))
        visible = [within(pial[v], 0.0) for v in expected]
        assert vmap[f"{hemi}_visible"].tolist() == visible
        assert 100 < sum(visible) < len(expected)
    # the white surface lies inside the pial one, so some vertices are in range by white alone
    lh_pial = surfaces["lh_pial"].vertices[vmap["lh_index"]]
    assert not all(within(xyz, margin) for xyz in lh_pial)


def test_round_trip_and_numpy_can_read_the_file(tmp_path: Path) -> None:
    vmap = vertexmap.build_vertex_map(hand_surfaces(), BOX, 2.0)
    path = tmp_path / "new" / "dir" / "vertex_map.npz"
    vertexmap.write_vertex_map(path, vmap)
    back = vertexmap.read_vertex_map(path)
    assert tuple(back) == vertexmap.KEYS
    with np.load(path) as archive:
        assert sorted(archive.files) == sorted(vertexmap.KEYS)
        for key in vertexmap.KEYS:
            assert back[key].dtype == vmap[key].dtype == archive[key].dtype
            np.testing.assert_array_equal(back[key], vmap[key])
            np.testing.assert_array_equal(archive[key], vmap[key])
    assert [p.name for p in path.parent.iterdir()] == ["vertex_map.npz"]  # no temporary file


def test_file_is_byte_identical_for_identical_input(tmp_path: Path) -> None:
    vmap = vertexmap.build_vertex_map(hand_surfaces(), BOX, 2.0)
    vertexmap.write_vertex_map(tmp_path / "a.npz", vmap)
    # the same content in another key order and in wider dtypes
    wide = {
        key: vmap[key].astype(np.float64 if vmap[key].dtype.kind == "f" else np.int64)  # also bool
        for key in reversed(vertexmap.KEYS)
    }
    vertexmap.write_vertex_map(tmp_path / "b.npz", wide)
    assert (tmp_path / "a.npz").read_bytes() == (tmp_path / "b.npz").read_bytes()
    with zipfile.ZipFile(tmp_path / "a.npz") as archive:
        entries = archive.infolist()
    # numpy.savez would stamp the entries with the time of writing
    assert [e.filename for e in entries] == [f"{key}.npy" for key in vertexmap.KEYS]
    assert {e.date_time for e in entries} == {(1980, 1, 1, 0, 0, 0)}
    assert {e.compress_type for e in entries} == {zipfile.ZIP_STORED}
    assert {(e.create_system, e.external_attr) for e in entries} == {(3, 0o644 << 16)}


def test_writing_refuses_a_map_with_other_keys_and_leaves_no_file(tmp_path: Path) -> None:
    vmap = vertexmap.build_vertex_map(hand_surfaces(), BOX, 2.0)
    del vmap["rh_fsavg6"]
    with pytest.raises(ValueError, match="keys"):
        vertexmap.write_vertex_map(tmp_path / "vertex_map.npz", vmap)
    assert list(tmp_path.iterdir()) == []
