"""Tests of render/vertexmap.py: which vertices go into vertex_map.npz, and the file itself."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
from synth import SyntheticSubject  # pyrefly: ignore[missing-import]

from surfcensus.fsio.surfaces import Surface, read_surface
from surfcensus.render import vertexmap
from surfcensus.render.model import BoxInfo

BOX = BoxInfo(centre=(5.0, 5.0, 5.0), low=(0.0, 0.0, 0.0), high=(10.0, 10.0, 10.0))
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


@pytest.mark.parametrize(
    ("margin", "expected"), [(0.0, [1, 2, 5]), (1.5, [1, 2, 3, 5]), (2.5, [1, 2, 3, 4, 5, 6, 7])]
)
def test_the_margin_grows_the_box_on_every_side(margin: float, expected: list[int]) -> None:
    assert vertexmap.build_vertex_map(hand_surfaces(), BOX, margin)["lh_index"].tolist() == expected


def test_keys_dtypes_shapes_and_unset_counterparts() -> None:
    vmap = vertexmap.build_vertex_map(hand_surfaces(), BOX, 2.0)
    assert tuple(vmap) == vertexmap.KEYS
    assert vertexmap.KEYS == (
        "lh_index", "lh_pial", "lh_white", "lh_fsavg6",
        "rh_index", "rh_pial", "rh_white", "rh_fsavg6",
    )  # fmt: skip
    for hemi, k in (("lh", 5), ("rh", 1)):
        assert vmap[f"{hemi}_index"].dtype == np.int32 and vmap[f"{hemi}_index"].shape == (k,)
        assert vmap[f"{hemi}_fsavg6"].dtype == np.int32
        assert vmap[f"{hemi}_fsavg6"].tolist() == [-1] * k
        for name in ("pial", "white"):
            assert vmap[f"{hemi}_{name}"].dtype == np.float32
            assert vmap[f"{hemi}_{name}"].shape == (k, 3)


def test_an_empty_hemisphere_gives_empty_arrays_of_the_right_shape() -> None:
    far_box = BoxInfo(centre=(-90.0, 0.0, 0.0), low=(-91.0, -1.0, -1.0), high=(-89.0, 1.0, 1.0))
    vmap = vertexmap.build_vertex_map(hand_surfaces(), far_box, 0.0)
    assert vmap["lh_index"].shape == (0,) and vmap["lh_pial"].shape == (0, 3)
    assert vmap["lh_white"].shape == (0, 3) and vmap["lh_fsavg6"].shape == (0,)


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
    box = BoxInfo(centre=(-4.0, -60.0, 5.0), low=(-34.0, -68.0, -30.0), high=(26.0, -15.0, 30.0))
    margin = 10.0
    vmap = vertexmap.build_vertex_map(surfaces, box, margin)
    for hemi in ("lh", "rh"):
        pial, white = surfaces[f"{hemi}_pial"].vertices, surfaces[f"{hemi}_white"].vertices
        expected = [
            v
            for v in range(len(pial))
            if any(
                all(box.low[a] - margin <= xyz[v, a] <= box.high[a] + margin for a in range(3))
                for xyz in (pial, white)
            )
        ]
        index = vmap[f"{hemi}_index"]
        assert 100 < len(expected) < len(pial)
        assert index.tolist() == expected
        assert np.all(np.diff(index) > 0)
        np.testing.assert_array_equal(vmap[f"{hemi}_pial"], pial[index].astype(np.float32))
        np.testing.assert_array_equal(vmap[f"{hemi}_white"], white[index].astype(np.float32))
    # the white surface lies inside the pial one, so some vertices are in range by white alone
    lh_pial = surfaces["lh_pial"].vertices[vmap["lh_index"]]
    low, high = np.array(box.low) - margin, np.array(box.high) + margin
    assert (~((lh_pial >= low) & (lh_pial <= high)).all(axis=1)).any()


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
        key: vmap[key].astype(np.float64 if vmap[key].dtype.kind == "f" else np.int64)
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
