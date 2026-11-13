"""The template: coarsening, what is derived from an fsaverage directory, its identity, the
cache, and the lookup of a run's template.

Synthetic directories come from ``synth_template``, whose layout is known in closed form.
Most tests use one of order 5 (10,242 vertices), which builds in milliseconds; the session
fixture of order 7 stands for the size of the real fsaverage. The last tests read the real
``fsaverage`` next to ``ds000102``: one from the pytest cache, skipped when the files are
absent, and its network twin, which fetches them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer as fs
import numpy as np
import pytest
from scipy import sparse
from synth import icosphere  # pyrefly: ignore[missing-import]
from synth_template import (  # pyrefly: ignore[missing-import]
    AVG_AREA_FACTOR_CAP,
    AVG_AREA_FACTOR_REST,
    DESIKAN_NAMES,
    POSTERIOR_CAP_DEG,
    TEMPLATE_BUILD_STAMP,
    SyntheticTemplate,
    desikan_colours,
    expected_regions,
    mesh_vertex_areas,
    parcel_of,
    template_mesh,
    template_white,
    write_label,
    write_synthetic_template,
)

from surfcensus import templates
from surfcensus._io import canonical_json, read_npz, sha256_file
from surfcensus.config import MeshName, MetricsConfig, RegionConfig
from surfcensus.sources.base import HEMIS
from surfcensus.templates import (
    MESH_ORDER,
    REGION_NAMES,
    TEMPLATE_JSON,
    TEMPLATE_NPZ,
    TEMPLATE_SCHEMA,
    Template,
    TemplateError,
    TemplateNotFound,
    build_template,
    coarsen_faces,
    find_template,
    icosahedron_vertices,
    read_template,
    template_candidates,
    write_template,
)

REQUIRED = (
    "surf/lh.sphere.reg",
    "surf/rh.sphere.reg",
    "surf/lh.white",
    "surf/rh.white",
    "label/lh.cortex.label",
    "label/rh.cortex.label",
    "label/lh.aparc.annot",
    "label/rh.aparc.annot",
)
AVG_AREA = ("surf/lh.white.avg.area.mgh", "surf/rh.white.avg.area.mgh")
NOTHING_FOUND = (
    ("freesurfer_home", "$FREESURFER_HOME/subjects/fsaverage", "not set"),
    ("adjacent", "next to the subjects", "absent"),
    ("bundled", "bundled copy", "not shipped"),
)
HEMI_ARRAYS = ("sphere", "faces", "white", "cortex", "parcel", "area")
HEMI_DTYPES = {
    "sphere": np.float64,
    "faces": np.int32,
    "white": np.float32,
    "cortex": np.bool_,
    "parcel": np.int16,
    "area": np.float32,
}


@pytest.fixture(scope="module")
def small(tmp_path_factory: pytest.TempPathFactory) -> SyntheticTemplate:
    """An ``fsaverage`` directory of order 5 with the average-area files. Read-only."""
    return write_synthetic_template(tmp_path_factory.mktemp("small"), order=5)


@pytest.fixture(scope="module")
def small_bare(tmp_path_factory: pytest.TempPathFactory) -> SyntheticTemplate:
    """The same without the average-area files. Read-only."""
    return write_synthetic_template(
        tmp_path_factory.mktemp("small_bare"), order=5, with_avg_area=False
    )


@pytest.fixture(scope="module")
def small4(small: SyntheticTemplate) -> Template:
    """``small`` on fsaverage4: one coarsening step."""
    return build_template(small.path, "explicit", mesh="fsaverage4")


@pytest.fixture(scope="module")
def from_order7(synth_template: SyntheticTemplate) -> Callable[[MeshName], Template]:
    """The order-7 directory on each mesh, built once per mesh."""
    built: dict[str, Template] = {}

    def build(mesh: MeshName) -> Template:
        if mesh not in built:
            built[mesh] = build_template(synth_template.path, "adjacent", mesh=mesh)
        return built[mesh]

    return build


def copy_of(fixture: SyntheticTemplate, target: Path) -> Path:
    """A copy of a fixture directory that a test may change."""
    shutil.copytree(fixture.path, target)
    return target


def rewrite_surface(
    path: Path, *, vertices: np.ndarray | None = None, faces: np.ndarray | None = None
) -> None:
    old_vertices, old_faces = fs.read_geometry(str(path))
    fs.write_geometry(
        str(path),
        old_vertices if vertices is None else vertices,
        old_faces if faces is None else faces,
    )


def same_template(first: Template, second: Template, *, origin: bool = True) -> None:
    """Every field equal, arrays in value and type."""
    for name in (
        "mesh",
        "order",
        "template_id",
        "files",
        "area_source",
        "build_stamp",
        "parcel_names",
        "regions",
    ):
        assert getattr(first, name) == getattr(second, name), name
    if origin:
        assert first.origin == second.origin
    assert tuple(first.hemis) == tuple(second.hemis) == HEMIS
    for hemi in HEMIS:
        one, two = first.hemis[hemi], second.hemis[hemi]
        assert one.pole_vertex == two.pole_vertex
        for name in HEMI_ARRAYS:
            x, y = getattr(one, name), getattr(two, name)
            assert x.dtype == y.dtype and np.array_equal(x, y), (hemi, name)
        assert tuple(one.regions) == tuple(two.regions) == REGION_NAMES
        for name in REGION_NAMES:
            x, y = one.regions[name], two.regions[name]
            assert x.dtype == y.dtype == np.bool_ and np.array_equal(x, y), (hemi, name)


def turned_to_lowest(faces: np.ndarray) -> np.ndarray:
    """Each face rotated to start with its lowest vertex; the winding stays."""
    faces = np.asarray(faces, dtype=np.int64)
    shift = np.argmin(faces, axis=1)
    return np.take_along_axis(faces, (shift[:, None] + np.arange(3)) % 3, axis=1)


def canonical(faces: np.ndarray) -> np.ndarray:
    """Faces as a sorted list of rows, each starting with its lowest vertex: equal for two
    meshes with the same faces and winding, whatever their order."""
    rows = turned_to_lowest(faces)
    return rows[np.lexsort(rows.T[::-1])]


def outward(sphere: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = (sphere[faces[:, corner]] for corner in range(3))
    return (a * np.cross(b, c)).sum(axis=1) > 0


# --- coarsening -------------------------------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_coarsening_an_icosphere_gives_the_faces_of_the_order_below(order: int) -> None:
    _, fine = icosphere(order)
    _, coarse = icosphere(order - 1)
    result = coarsen_faces(fine, icosahedron_vertices(order - 1))
    assert result.dtype == fine.dtype
    assert np.array_equal(result, coarse)
    as_int32 = coarsen_faces(fine.astype(np.int32), icosahedron_vertices(order - 1))
    assert as_int32.dtype == np.int32 and np.array_equal(as_int32, coarse)


def test_coarsening_follows_the_input_order_and_winding_of_the_middle_faces() -> None:
    _, fine = icosphere(3)
    _, coarse = icosphere(2)
    rng = np.random.default_rng(3)
    shuffled = fine[rng.permutation(len(fine))]
    turned = np.take_along_axis(
        shuffled, (rng.integers(0, 3, len(fine))[:, None] + np.arange(3)) % 3, axis=1
    )
    result = coarsen_faces(turned, icosahedron_vertices(2))
    assert not np.array_equal(result, coarse)
    assert np.array_equal(canonical(result), canonical(coarse))
    # the coarse faces come in the order of the faces between three new vertices
    n_coarse = icosahedron_vertices(2)
    plain_middle = fine[(fine >= n_coarse).all(axis=1)]
    position = {tuple(sorted(row)): at for at, row in enumerate(plain_middle.tolist())}
    middle = shuffled[(shuffled >= n_coarse).all(axis=1)]
    order = [position[tuple(sorted(row))] for row in middle.tolist()]
    assert order != sorted(order)
    assert np.array_equal(turned_to_lowest(result), turned_to_lowest(coarse[order]))
    # reversed faces give reversed faces
    reversed_faces = coarsen_faces(fine[:, ::-1], icosahedron_vertices(2))
    assert np.array_equal(canonical(reversed_faces[:, ::-1]), canonical(coarse))


def test_two_steps_of_coarsening_are_two_orders() -> None:
    _, fine = icosphere(4)
    once = coarsen_faces(fine, icosahedron_vertices(3))
    assert np.array_equal(coarsen_faces(once, icosahedron_vertices(2)), icosphere(2)[1])


def test_a_mesh_that_does_not_nest_is_refused() -> None:
    _, faces = icosphere(3)
    n_coarse = icosahedron_vertices(2)
    cases: dict[str, tuple[np.ndarray, int]] = {
        # the same sphere with its vertices numbered in another order
        "renumbered": (np.random.default_rng(0).permutation(faces.max() + 1)[faces], n_coarse),
        "a face is missing": (faces[:-1], n_coarse),
        "a face twice": (np.concatenate([faces, faces[-1:]]), n_coarse),
        "the wrong vertex count": (faces, icosahedron_vertices(1)),
        "nothing to drop": (faces, int(faces.max()) + 1),
        "a negative index": (np.where(faces == 5, -1, faces), n_coarse),
        "no faces": (faces[:0], n_coarse),
        "two vertices": (faces, 2),
    }
    for name, (broken, n) in cases.items():
        with pytest.raises(TemplateError) as info:
            coarsen_faces(broken, n)
        assert info.value.reason == "not icosahedral", name
    # every count is right, but one face joins new vertices that have no parent in common
    odd = faces.copy()
    middle = np.flatnonzero((faces >= n_coarse).all(axis=1))
    odd[middle[0], 0] = faces[middle[40], 0]
    with pytest.raises(TemplateError) as info:
        coarsen_faces(odd, n_coarse)
    assert (info.value.reason, info.value.detail) == (
        "not icosahedral",
        "1 faces have two corners that do not share exactly one parent",
    )


# --- what is derived --------------------------------------------------------------------------


@pytest.mark.parametrize("mesh", list(MESH_ORDER))
def test_each_mesh_name_gives_its_order_from_an_order_7_directory(
    from_order7: Callable[[MeshName], Template], mesh: MeshName
) -> None:
    template = from_order7(mesh)
    order = {"fsaverage4": 4, "fsaverage5": 5, "fsaverage6": 6, "fsaverage": 7}[mesh]
    n = 10 * 4**order + 2
    directions, faces = template_mesh(order)
    assert (template.mesh, template.order, template.origin) == (mesh, order, "adjacent")
    assert template.parcel_names == DESIKAN_NAMES
    assert template.build_stamp == TEMPLATE_BUILD_STAMP
    assert template.regions == RegionConfig()
    assert tuple(template.hemis) == HEMIS
    for hemi in HEMIS:
        half = template.hemis[hemi]
        for name, dtype in HEMI_DTYPES.items():
            array = getattr(half, name)
            assert array.dtype == dtype and not array.flags.writeable, name
            assert array.flags.c_contiguous
        assert half.sphere.shape == half.white.shape == (n, 3)
        assert half.cortex.shape == half.parcel.shape == half.area.shape == (n,)
        assert np.array_equal(half.faces, faces)
        assert outward(half.sphere, half.faces).all()
        assert np.abs(np.linalg.norm(half.sphere, axis=1) - 1).max() < 1e-12
        # the file stores float32 at radius 100
        assert np.abs(half.sphere - directions).max() < 1e-6
        assert np.array_equal(half.white, template_white(directions, hemi).astype(np.float32))
        assert all(not mask.flags.writeable for mask in half.regions.values())


@pytest.mark.parametrize("mesh", ["fsaverage6", "fsaverage4"])
def test_regions_cortex_and_parcels_are_the_fixtures(
    synth_template: SyntheticTemplate, from_order7: Callable[[MeshName], Template], mesh: MeshName
) -> None:
    template = from_order7(mesh)
    n = icosahedron_vertices(template.order)
    directions, _ = template_mesh(template.order)
    for hemi in HEMIS:
        half, masks = template.hemis[hemi], synth_template.masks[hemi]
        assert np.array_equal(half.cortex, masks["cortex"][:n])
        assert np.array_equal(half.parcel, parcel_of(directions, hemi).astype(np.int16))
        assert tuple(half.regions) == REGION_NAMES
        for name in REGION_NAMES:
            assert np.array_equal(half.regions[name], masks[name][:n]), (hemi, name)
            assert not (half.regions[name] & ~half.cortex).any()
            assert half.regions[name].any()
        parts = [half.regions[name] for name in ("medial", "inferior", "lateral")]
        assert np.array_equal(parts[0] | parts[1] | parts[2], half.regions["search"])
        assert np.array_equal(sum(part.astype(int) for part in parts), half.regions["search"])
        assert not (half.regions["search"] & half.regions["control"]).any()


def test_regions_on_the_mesh_of_the_directory_itself(small: SyntheticTemplate) -> None:
    template = build_template(small.path, "explicit", mesh="fsaverage5")
    for hemi in HEMIS:
        half = template.hemis[hemi]
        assert np.array_equal(half.cortex, small.masks[hemi]["cortex"])
        for name in REGION_NAMES:
            assert np.array_equal(half.regions[name], small.masks[hemi][name])


@pytest.mark.parametrize("cut_mm", [10.0, 30.0, 45.0, 60.0, 500.0])
def test_the_cut_moves_the_posterior_parcels_only(small: SyntheticTemplate, cut_mm: float) -> None:
    regions = RegionConfig(cut_mm=cut_mm)
    template = build_template(small.path, "explicit", mesh="fsaverage5", regions=regions)
    default = build_template(small.path, "explicit", mesh="fsaverage5")
    directions, _ = template_mesh(5)
    assert template.regions == regions
    for hemi in HEMIS:
        half, masks = template.hemis[hemi], small.masks[hemi]
        expected = expected_regions(directions, hemi, regions)
        for name in REGION_NAMES:
            assert np.array_equal(half.regions[name], expected[name]), (hemi, name)
        occipital = np.any([masks[name] for name in regions.occipital], axis=0)
        posterior = np.any([masks[name] for name in regions.posterior], axis=0)
        assert np.array_equal(half.regions["search"] & ~posterior, occipital)
        assert np.array_equal(half.regions["control"], default.hemis[hemi].regions["control"])
        kept = half.regions["search"] & posterior
        before = default.hemis[hemi].regions["search"] & posterior
        if cut_mm < 45.0:
            assert not (kept & ~before).any() and kept.sum() < before.sum()
        elif cut_mm > 45.0:
            assert not (before & ~kept).any() and kept.sum() > before.sum()
        if cut_mm == 500.0:
            assert np.array_equal(kept, posterior)
        # what the cut keeps lies within cut_mm of the pole, in y, on the white surface
        ahead = half.white[:, 1].astype(float) - float(half.white[half.pole_vertex, 1])
        assert (ahead[kept] <= cut_mm).all() and (ahead[posterior & ~kept] > cut_mm).all()


def test_a_missing_parcel_is_an_error_that_names_it(small: SyntheticTemplate) -> None:
    regions = RegionConfig(
        control=("caudalmiddlefrontal", "nosuchparcel"), inferior=("lingual", "another", "fusiform")
    )
    with pytest.raises(TemplateError) as info:
        build_template(small.path, "explicit", mesh="fsaverage4", regions=regions)
    assert info.value.reason == "missing parcel"
    assert info.value.detail == "label/lh.aparc.annot has no parcel another, nosuchparcel"
    with pytest.raises(TemplateError, match=r"missing parcel: .*ghost"):
        find_template(
            config=MetricsConfig(mesh="fsaverage4", regions=RegionConfig(occipital=("ghost",))),
            adjacent=small.path,
            env={},
        )


def test_the_pole_is_the_most_posterior_cortex_vertex(
    small: SyntheticTemplate, small4: Template, tmp_path: Path
) -> None:
    for hemi in HEMIS:
        half = small4.hemis[hemi]
        assert half.pole_vertex == small.pole_vertex == 21
        assert isinstance(half.pole_vertex, int)
        assert half.cortex[half.pole_vertex]
        assert half.white[half.pole_vertex, 1] == half.white[half.cortex, 1].min()

    # without vertex 21 in the cortex label the pole is the next most posterior cortex vertex
    directory = copy_of(small, tmp_path / "fsaverage")
    directions, _ = template_mesh(5)
    white = template_white(directions, "lh")
    listed = np.flatnonzero(small.masks["lh"]["cortex"])
    listed = listed[listed != 21]
    write_label(directory / "label" / "lh.cortex.label", listed, white[listed])
    moved = build_template(directory, "explicit", mesh="fsaverage4")
    n = icosahedron_vertices(4)
    candidates = [int(v) for v in listed if v < n]
    white_y = white.astype(np.float32)[:, 1]
    expected = min(candidates, key=lambda vertex: (white_y[vertex], vertex))
    assert moved.hemis["lh"].pole_vertex == expected != 21
    assert not moved.hemis["lh"].cortex[21]
    assert moved.hemis["rh"].pole_vertex == 21
    assert moved.template_id != small4.template_id

    # two cortex vertices equally far back: the lower index
    tied = white.copy()
    tied[[300, 250], 1] = -500.0
    rewrite_surface(directory / "surf" / "lh.white", vertices=tied)
    assert build_template(directory, "explicit", mesh="fsaverage4").hemis["lh"].pole_vertex == 250


def test_the_pole_is_taken_on_the_target_mesh(small: SyntheticTemplate, tmp_path: Path) -> None:
    """A vertex that the target mesh lacks cannot be its pole."""
    directory = copy_of(small, tmp_path / "fsaverage")
    directions, _ = template_mesh(5)
    white = template_white(directions, "lh")
    beyond = icosahedron_vertices(4) + 17
    assert small.masks["lh"]["cortex"][beyond]
    white[beyond, 1] = -500.0
    rewrite_surface(directory / "surf" / "lh.white", vertices=white)
    assert build_template(directory, "explicit", mesh="fsaverage5").hemis["lh"].pole_vertex == (
        beyond
    )
    assert build_template(directory, "explicit", mesh="fsaverage4").hemis["lh"].pole_vertex == 21


# --- area -------------------------------------------------------------------------------------


def file_areas(directory: Path, hemi: str) -> np.ndarray:
    image = nib.load(str(directory / "surf" / f"{hemi}.white.avg.area.mgh"))
    return np.asarray(image.dataobj, dtype=np.float64).ravel()


@pytest.mark.parametrize(
    ("source_order", "mesh"),
    [(5, "fsaverage5"), (5, "fsaverage4"), (7, "fsaverage6"), (7, "fsaverage5"), (7, "fsaverage4")],
)
def test_area_from_the_average_area_file(
    small: SyntheticTemplate,
    synth_template: SyntheticTemplate,
    from_order7: Callable[[MeshName], Template],
    source_order: int,
    mesh: MeshName,
) -> None:
    """The file's total is kept, and each vertex differs from its white-mesh area by the
    factor planted where it lies: zero, one, two and three coarsening steps."""
    source = small if source_order == 5 else synth_template
    template = (
        build_template(source.path, "explicit", mesh=mesh)
        if source_order == 5
        else from_order7(mesh)
    )
    assert template.area_source == "white.avg.area"
    assert set(template.files) == {*REQUIRED, *AVG_AREA}
    directions, faces = template_mesh(template.order)
    cap = directions[:, 1] < -np.cos(np.deg2rad(POSTERIOR_CAP_DEG))
    for hemi in HEMIS:
        half = template.hemis[hemi]
        values = file_areas(source.path, hemi)
        assert half.area.dtype == np.float32 and (half.area > 0).all()
        assert half.area.sum(dtype=np.float64) == pytest.approx(values.sum(), rel=1e-6)
        if template.order == source_order:
            assert np.array_equal(half.area, values.astype(np.float32))
        mesh_area = mesh_vertex_areas(template_white(directions, hemi), faces)
        ratio = half.area / mesh_area
        adjacency = template.adjacency(hemi)
        around = adjacency @ cap.astype(float)
        inside = cap & (around == adjacency.sum(axis=1).A1)
        outside = ~cap & (around == 0)
        assert inside.sum() > 200 and outside.sum() > 2000
        # the coarse faces cut the corners of the curved surface: a few parts in a thousand
        assert ratio[inside] == pytest.approx(AVG_AREA_FACTOR_CAP, rel=4e-3)
        assert ratio[outside] == pytest.approx(AVG_AREA_FACTOR_REST, rel=4e-3)
        assert ratio.min() > AVG_AREA_FACTOR_REST * 0.996
        assert ratio.max() < AVG_AREA_FACTOR_CAP * 1.004


def test_area_is_shared_between_the_two_vertices_a_dropped_vertex_lies_between(
    small: SyntheticTemplate, tmp_path: Path
) -> None:
    """One step by hand: a vertex keeps its own value and gets half of each dropped
    neighbour's. A nearest-vertex rule would hand each dropped value to one side whole."""
    directory = copy_of(small, tmp_path / "fsaverage")
    n = icosahedron_vertices(4)
    _, faces = template_mesh(5)
    rng = np.random.default_rng(11)
    for hemi in HEMIS:
        values = rng.uniform(0.5, 2.0, icosahedron_vertices(5)).astype(np.float32)
        nib.save(
            nib.MGHImage(values.reshape(-1, 1, 1), np.eye(4)),
            str(directory / "surf" / f"{hemi}.white.avg.area.mgh"),
        )
    template = build_template(directory, "explicit", mesh="fsaverage4")
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    for hemi in HEMIS:
        values = file_areas(directory, hemi)
        expected = values[:n].copy()
        pairs = {(int(a), int(b)) for a, b in edges if a >= n > b}
        pairs |= {(int(b), int(a)) for a, b in edges if b >= n > a}
        assert len(pairs) == 2 * (len(values) - n)
        for dropped, kept in pairs:
            expected[kept] += values[dropped] / 2
        assert np.allclose(template.hemis[hemi].area, expected, rtol=1e-6, atol=0)
        assert template.hemis[hemi].area.sum(dtype=np.float64) == pytest.approx(
            values.sum(), rel=1e-6
        )


@pytest.mark.parametrize("mesh", ["fsaverage5", "fsaverage4"])
def test_without_the_file_area_is_that_of_the_white_mesh(
    small: SyntheticTemplate, small_bare: SyntheticTemplate, mesh: MeshName
) -> None:
    bare = build_template(small_bare.path, "explicit", mesh=mesh)
    full = build_template(small.path, "explicit", mesh=mesh)
    assert (bare.area_source, full.area_source) == ("white mesh", "white.avg.area")
    assert set(bare.files) == set(REQUIRED)
    assert bare.template_id != full.template_id
    directions, faces = template_mesh(bare.order)
    for hemi in HEMIS:
        white = template_white(directions, hemi).astype(np.float32).astype(np.float64)
        expected = mesh_vertex_areas(white, faces)
        assert np.allclose(bare.hemis[hemi].area, expected, rtol=1e-6, atol=0)
        assert bare.hemis[hemi].area.dtype == np.float32
        # everything but the area is the same
        for name in ("sphere", "faces", "white", "cortex", "parcel"):
            assert np.array_equal(getattr(bare.hemis[hemi], name), getattr(full.hemis[hemi], name))
        assert not np.allclose(bare.hemis[hemi].area, full.hemis[hemi].area, rtol=0.1)


def test_one_average_area_file_alone_is_not_used(small: SyntheticTemplate, tmp_path: Path) -> None:
    directory = copy_of(small, tmp_path / "fsaverage")
    (directory / AVG_AREA[1]).unlink()
    template = build_template(directory, "explicit", mesh="fsaverage4")
    assert template.area_source == "white mesh"
    assert set(template.files) == set(REQUIRED)


def test_an_average_area_file_that_does_not_fit_is_refused(
    small: SyntheticTemplate, tmp_path: Path
) -> None:
    directory = copy_of(small, tmp_path / "fsaverage")
    path = directory / AVG_AREA[0]
    good = file_areas(directory, "lh").astype(np.float32)
    nib.save(nib.MGHImage(good[:-1].reshape(-1, 1, 1), np.eye(4)), str(path))
    with pytest.raises(TemplateError) as info:
        build_template(directory, "explicit", mesh="fsaverage4")
    assert info.value.reason == "vertex count mismatch"
    assert info.value.detail == (
        "surf/lh.white.avg.area.mgh has 10241 values and surf/lh.sphere.reg 10242 vertices"
    )
    for bad in (np.nan, -1.0):
        changed = good.copy()
        changed[7] = bad
        nib.save(nib.MGHImage(changed.reshape(-1, 1, 1), np.eye(4)), str(path))
        with pytest.raises(TemplateError) as info:
            build_template(directory, "explicit", mesh="fsaverage4")
        assert info.value.reason == "unreadable"
        assert info.value.detail.startswith("surf/lh.white.avg.area.mgh: ")


# --- adjacency --------------------------------------------------------------------------------


def test_adjacency_is_symmetric_binary_and_computed_once(small4: Template) -> None:
    for hemi in HEMIS:
        adjacency = small4.adjacency(hemi)
        faces = small4.hemis[hemi].faces
        n = len(small4.hemis[hemi].sphere)
        assert isinstance(adjacency, sparse.csr_matrix)
        assert adjacency.shape == (n, n) and adjacency.dtype == np.float64
        assert adjacency.has_sorted_indices
        dense = adjacency.toarray()
        expected = np.zeros((n, n))
        for first, second in ((0, 1), (1, 2), (2, 0)):
            expected[faces[:, first], faces[:, second]] = 1
            expected[faces[:, second], faces[:, first]] = 1
        assert np.array_equal(dense, expected)
        assert np.array_equal(dense, dense.T) and not dense.diagonal().any()
        degree = dense.sum(axis=1)
        assert sorted(np.unique(degree, return_counts=True)[1].tolist()) == [12, n - 12]
        assert set(np.unique(degree)) == {5.0, 6.0}
        assert small4.adjacency(hemi) is adjacency
    assert small4.adjacency("lh") is not small4.adjacency("rh")


def test_a_template_is_pickled_without_its_adjacency(small: SyntheticTemplate) -> None:
    template = build_template(small.path, "explicit", mesh="fsaverage4")
    plain = len(pickle.dumps(template))
    adjacency = template.adjacency("lh")
    assert len(pickle.dumps(template)) == plain
    copy = pickle.loads(pickle.dumps(template))
    same_template(copy, template)
    assert "_adjacency" not in vars(copy)
    assert (copy.adjacency("lh") != adjacency).nnz == 0


# --- identity ---------------------------------------------------------------------------------


def expected_id(directory: Path, names: tuple[str, ...], mesh: str, regions: RegionConfig) -> str:
    """The id by the rule of the docs, from the files on disk."""
    payload = {
        "schema": 1,
        "mesh": mesh,
        "order": {"fsaverage4": 4, "fsaverage5": 5, "fsaverage6": 6, "fsaverage": 7}[mesh],
        "regions": regions.model_dump(mode="json"),
        "files": {name: sha256_file(directory / name) for name in names},
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:12]


def test_the_id_is_the_hash_of_schema_mesh_regions_and_file_digests(
    small: SyntheticTemplate, small_bare: SyntheticTemplate, small4: Template
) -> None:
    assert TEMPLATE_SCHEMA == 1
    assert small4.template_id == expected_id(
        small.path, (*REQUIRED, *AVG_AREA), "fsaverage4", RegionConfig()
    )
    bare = build_template(small_bare.path, "explicit", mesh="fsaverage4")
    assert bare.template_id == expected_id(small_bare.path, REQUIRED, "fsaverage4", RegionConfig())
    assert small4.files == {
        name: ((small.path / name).stat().st_size, sha256_file(small.path / name))
        for name in (*REQUIRED, *AVG_AREA)
    }
    assert list(small4.files) == [*REQUIRED, *AVG_AREA]


def test_the_id_is_equal_for_two_copies_and_changes_with_content_and_settings(
    small: SyntheticTemplate, small4: Template, tmp_path: Path
) -> None:
    directory = copy_of(small, tmp_path / "elsewhere" / "fsaverage")
    copy = build_template(directory, "freesurfer_home", mesh="fsaverage4")
    assert copy.template_id == small4.template_id
    assert (copy.origin, small4.origin) == ("freesurfer_home", "explicit")
    same_template(copy, small4, origin=False)

    ids = {small4.template_id}
    ids.add(build_template(small.path, "explicit", mesh="fsaverage5").template_id)
    for regions in (
        RegionConfig(cut_mm=44.0),
        RegionConfig(control=("superiorfrontal",)),
        RegionConfig(medial=("cuneus", "pericalcarine")),
    ):
        changed = build_template(small.path, "explicit", mesh="fsaverage4", regions=regions)
        ids.add(changed.template_id)
    assert len(ids) == 5

    # the stamp is recorded and is not what the template is derived from
    (directory / "scripts" / "build-stamp.txt").write_text("another-stamp\nsecond line\n")
    stamped = build_template(directory, "explicit", mesh="fsaverage4")
    assert stamped.build_stamp == "another-stamp"
    assert stamped.template_id == small4.template_id
    (directory / "scripts" / "build-stamp.txt").unlink()
    assert build_template(directory, "explicit", mesh="fsaverage4").build_stamp is None

    # one vertex less in a cortex label: other bytes, another id
    directions, _ = template_mesh(5)
    listed = np.flatnonzero(small.masks["rh"]["cortex"])[:-1]
    write_label(
        directory / "label" / "rh.cortex.label", listed, template_white(directions, "rh")[listed]
    )
    assert build_template(directory, "explicit", mesh="fsaverage4").template_id not in ids


# --- directories that cannot be used ----------------------------------------------------------


def test_a_directory_that_lacks_files_is_not_found_and_says_which(
    small: SyntheticTemplate, tmp_path: Path
) -> None:
    directory = copy_of(small, tmp_path / "fsaverage")
    (directory / "label" / "rh.aparc.annot").unlink()
    (directory / "surf" / "lh.white").unlink()
    with pytest.raises(TemplateNotFound) as info:
        build_template(directory, "adjacent")
    assert info.value.looked_in == (
        ("adjacent", str(directory), "incomplete: surf/lh.white, label/rh.aparc.annot"),
    )
    for absent in (tmp_path / "nothing", directory / "surf" / "rh.white"):
        with pytest.raises(TemplateNotFound) as info:
            build_template(absent, "explicit")
        assert info.value.looked_in == (("explicit", str(absent), "absent"),)
    link = tmp_path / "dangling"
    link.symlink_to(tmp_path / "gone", target_is_directory=True)
    with pytest.raises(TemplateNotFound) as info:
        build_template(link, "adjacent")
    assert info.value.looked_in == (("adjacent", str(link), "absent"),)
    # a required file that is a dangling link is missing
    (directory / "surf" / "lh.white").symlink_to(tmp_path / "gone")
    with pytest.raises(TemplateNotFound, match="no fsaverage template found") as info:
        build_template(directory, "adjacent")
    assert info.value.looked_in[0][2] == "incomplete: surf/lh.white, label/rh.aparc.annot"


def test_a_directory_that_cannot_be_examined_is_not_taken_for_absent(
    small: SyntheticTemplate, monkeypatch: pytest.MonkeyPatch
) -> None:
    def denied(path: Path | str) -> bool:
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(templates, "is_file", denied)
    with pytest.raises(TemplateNotFound) as info:
        build_template(small.path, "adjacent")
    assert info.value.looked_in == (("adjacent", str(small.path), "unreadable: Permission denied"),)


def test_a_file_that_may_not_be_read_is_unreadable(
    small: SyntheticTemplate, monkeypatch: pytest.MonkeyPatch
) -> None:
    def denied(path: Path | str) -> str:
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(templates, "sha256_file", denied)
    with pytest.raises(TemplateError) as info:
        build_template(small.path, "adjacent")
    assert (info.value.reason, info.value.detail) == (
        "unreadable",
        "surf/lh.sphere.reg: Permission denied",
    )


@pytest.mark.parametrize(
    "name",
    [
        "surf/lh.sphere.reg",
        "surf/rh.white",
        "label/lh.cortex.label",
        "label/rh.aparc.annot",
        "surf/rh.white.avg.area.mgh",
    ],
)
@pytest.mark.parametrize("damage", ["cut", "empty", "text"])
def test_a_damaged_file_is_unreadable_and_named_without_its_path(
    small: SyntheticTemplate, tmp_path: Path, name: str, damage: str
) -> None:
    directory = copy_of(small, tmp_path / "ds999999-fmriprep" / "fsaverage")
    content = (directory / name).read_bytes()
    (directory / name).write_bytes(
        {"cut": content[: len(content) // 2], "empty": b"", "text": b"not such a file\n" * 40}[
            damage
        ]
    )
    with pytest.raises(TemplateError) as info:
        build_template(directory, "explicit", mesh="fsaverage4")
    assert info.value.reason == "unreadable"
    assert info.value.detail.startswith(f"{name}: ")
    assert len(info.value.detail) > len(name) + 2
    assert "\n" not in info.value.detail
    for private in (str(tmp_path), "ds999999", "/"):
        assert private not in info.value.detail.removeprefix(name)


def sphere_of(n_vertices: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    directions, faces = template_mesh(order)
    assert len(directions) == n_vertices
    return directions * 100.0, faces


def test_a_sphere_that_is_not_an_icosahedral_mesh_is_refused(
    small: SyntheticTemplate, tmp_path: Path
) -> None:
    directory = copy_of(small, tmp_path / "fsaverage")
    with pytest.raises(TemplateError) as info:
        build_template(directory, "explicit", mesh="fsaverage6")
    assert info.value.reason == "not icosahedral"
    assert info.value.detail == (
        "surf/lh.sphere.reg has 10242 vertices; fsaverage6 needs 10 * 4**k + 2 with k >= 6"
    )

    vertices, faces = sphere_of(10_242, 5)

    def refused(new_vertices: np.ndarray, new_faces: np.ndarray, mesh: MeshName) -> str:
        for name in ("lh.sphere.reg", "lh.white"):
            fs.write_geometry(str(directory / "surf" / name), new_vertices, new_faces)
        with pytest.raises(TemplateError) as info:
            build_template(directory, "explicit", mesh=mesh)
        assert info.value.reason == "not icosahedral"
        assert info.value.detail.startswith("surf/lh.sphere.reg")
        return info.value.detail

    # a vertex count that no icosahedral mesh has
    without_last = faces[(faces < 10_241).all(axis=1)]
    assert "has 10241 vertices" in refused(vertices[:-1], without_last, "fsaverage4")
    # the right vertex count with a face missing
    assert "20479 faces" in refused(vertices, faces[:-1], "fsaverage4")
    # the right counts, vertices numbered in another order: on a coarser mesh and on its own
    order = np.random.default_rng(1).permutation(10_242)
    renumbered = np.argsort(order)[faces]
    for mesh in ("fsaverage4", "fsaverage5"):
        assert "neighbours below" in refused(vertices[order], renumbered, mesh)
    # radii that differ by more than a hundredth
    stretched = vertices.copy()
    stretched[:500] *= 1.03
    assert "degenerate sphere" in refused(stretched, faces, "fsaverage4")
    # put back as it was, the directory builds
    for name in ("lh.sphere.reg", "lh.white"):
        fs.write_geometry(str(directory / "surf" / name), vertices, faces)
    rewrite_surface(
        directory / "surf" / "lh.white", vertices=template_white(vertices / 100.0, "lh")
    )
    assert build_template(directory, "explicit", mesh="fsaverage4").order == 4


def test_files_that_do_not_belong_to_the_sphere_are_refused(
    small: SyntheticTemplate, tmp_path: Path
) -> None:
    directions, faces = template_mesh(5)
    coarse_directions, coarse_faces = template_mesh(4)

    def reason_and_detail(change: Callable[[Path], None]) -> tuple[str, str]:
        directory = copy_of(small, tmp_path / f"case{len(list(tmp_path.iterdir()))}" / "fsaverage")
        change(directory)
        with pytest.raises(TemplateError) as info:
            build_template(directory, "explicit", mesh="fsaverage4")
        return info.value.reason, info.value.detail

    assert reason_and_detail(
        lambda d: fs.write_geometry(
            str(d / "surf" / "rh.white"), template_white(coarse_directions, "rh"), coarse_faces
        )
    ) == (
        "vertex count mismatch",
        "surf/rh.white has 2562 vertices and surf/rh.sphere.reg 10242",
    )
    assert reason_and_detail(
        lambda d: rewrite_surface(d / "surf" / "lh.white", faces=faces[:, [1, 2, 0]])
    ) == ("not icosahedral", "surf/lh.white does not have the faces of surf/lh.sphere.reg")
    assert reason_and_detail(
        lambda d: fs.write_annot(
            str(d / "label" / "lh.aparc.annot"),
            parcel_of(coarse_directions, "lh"),
            desikan_colours(),
            list(DESIKAN_NAMES),
            fill_ctab=True,
        )
    ) == (
        "vertex count mismatch",
        "label/lh.aparc.annot has 2562 labels and surf/lh.sphere.reg 10242 vertices",
    )
    assert reason_and_detail(
        lambda d: write_label(
            d / "label" / "rh.cortex.label", np.array([3, 10_242]), np.zeros((2, 3))
        )
    ) == (
        "vertex count mismatch",
        "label/rh.cortex.label lists vertex 10242 and surf/rh.sphere.reg has 10242",
    )
    # a label of the full mesh that names no vertex of the target mesh
    assert reason_and_detail(
        lambda d: write_label(
            d / "label" / "rh.cortex.label", np.array([9000, 5000]), np.zeros((2, 3))
        )
    ) == ("unreadable", "label/rh.cortex.label: lists no vertex of fsaverage4")
    not_finite = template_white(directions, "lh")
    not_finite[40, 2] = np.inf
    assert reason_and_detail(
        lambda d: rewrite_surface(d / "surf" / "lh.white", vertices=not_finite)
    ) == ("unreadable", "surf/lh.white: a coordinate is not finite")


def test_a_cortex_label_lists_vertices_of_the_full_mesh(
    small: SyntheticTemplate, tmp_path: Path
) -> None:
    """Indices beyond the target mesh are dropped, not an error, and order does not matter."""
    directory = copy_of(small, tmp_path / "fsaverage")
    directions, _ = template_mesh(5)
    listed = np.array([10_241, 7, 2_561, 2_562, 0, 99])
    write_label(
        directory / "label" / "lh.cortex.label", listed, template_white(directions, "lh")[listed]
    )
    template = build_template(directory, "explicit", mesh="fsaverage4")
    assert np.flatnonzero(template.hemis["lh"].cortex).tolist() == [0, 7, 99, 2_561]
    for name in REGION_NAMES:
        assert not (template.hemis["lh"].regions[name] & ~template.hemis["lh"].cortex).any()


def test_a_sphere_wound_inward_gives_outward_faces(
    small: SyntheticTemplate, small4: Template, tmp_path: Path
) -> None:
    directory = copy_of(small, tmp_path / "fsaverage")
    _, faces = template_mesh(5)
    for name in ("lh.sphere.reg", "lh.white"):
        rewrite_surface(directory / "surf" / name, faces=faces[:, ::-1])
    template = build_template(directory, "explicit", mesh="fsaverage4")
    half = template.hemis["lh"]
    assert outward(half.sphere, half.faces).all()
    assert np.array_equal(canonical(half.faces), canonical(small4.hemis["lh"].faces))
    assert (template.adjacency("lh") != small4.adjacency("lh")).nnz == 0
    assert np.array_equal(template.hemis["rh"].faces, small4.hemis["rh"].faces)


def test_parcels_of_both_hemispheres_index_one_list_of_names(
    small: SyntheticTemplate, small4: Template, tmp_path: Path
) -> None:
    """The right annotation with its names in another order, and one more."""
    directory = copy_of(small, tmp_path / "fsaverage")
    directions, _ = template_mesh(5)
    names = [*reversed(DESIKAN_NAMES), "extra"]
    old = parcel_of(directions, "rh")
    labels = np.where(old >= 0, len(DESIKAN_NAMES) - 1 - old, -1)
    colours = np.concatenate([desikan_colours()[::-1], [[1, 2, 3, 0]]]).astype(np.int32)
    fs.write_annot(
        str(directory / "label" / "rh.aparc.annot"), labels, colours, names, fill_ctab=True
    )
    template = build_template(directory, "explicit", mesh="fsaverage4")
    assert template.parcel_names == (*DESIKAN_NAMES, "extra")
    for hemi in HEMIS:
        assert np.array_equal(template.hemis[hemi].parcel, small4.hemis[hemi].parcel)
        assert template.hemis[hemi].parcel.dtype == np.int16
        for name in REGION_NAMES:
            assert np.array_equal(
                template.hemis[hemi].regions[name], small4.hemis[hemi].regions[name]
            )


# --- the cache --------------------------------------------------------------------------------


def test_the_cache_round_trips_exactly(small4: Template, tmp_path: Path) -> None:
    directory = tmp_path / "ds999999" / "templates" / small4.template_id
    write_template(directory, small4)
    assert sorted(path.name for path in directory.iterdir()) == [TEMPLATE_JSON, TEMPLATE_NPZ]
    back = read_template(directory)
    same_template(back, small4)
    for hemi in HEMIS:
        for name in HEMI_ARRAYS:
            assert not getattr(back.hemis[hemi], name).flags.writeable
    assert (back.adjacency("rh") != small4.adjacency("rh")).nnz == 0

    # written twice, and from what was read: the same bytes
    first = {path.name: path.read_bytes() for path in directory.iterdir()}
    write_template(directory, small4)
    write_template(tmp_path / "again", back)
    for name, content in first.items():
        assert (directory / name).read_bytes() == content
        assert (tmp_path / "again" / name).read_bytes() == content


def test_the_cache_files_are_as_documented_and_hold_no_path(
    small: SyntheticTemplate, small4: Template, tmp_path: Path
) -> None:
    directory = tmp_path / "cache"
    write_template(directory, small4)
    text = (directory / TEMPLATE_JSON).read_text()
    data = json.loads(text)
    assert set(data) == {
        "schema",
        "template_id",
        "mesh",
        "order",
        "origin",
        "area_source",
        "build_stamp",
        "parcel_names",
        "regions",
        "files",
        "pole_vertex",
    }
    assert data["schema"] == 1 and data["template_id"] == small4.template_id
    assert (data["mesh"], data["order"], data["origin"]) == ("fsaverage4", 4, "explicit")
    assert data["area_source"] == "white.avg.area"
    assert data["build_stamp"] == TEMPLATE_BUILD_STAMP
    assert tuple(data["parcel_names"]) == DESIKAN_NAMES
    assert data["regions"] == RegionConfig().model_dump(mode="json")
    assert data["pole_vertex"] == {"lh": 21, "rh": 21}
    assert data["files"] == {
        name: {"size": size, "sha256": digest} for name, (size, digest) in small4.files.items()
    }
    assert text.endswith("\n") and text == json.dumps(data, indent=1, sort_keys=True) + "\n"

    arrays = read_npz(directory / TEMPLATE_NPZ)
    expected = {}
    for hemi in HEMIS:
        expected |= {
            f"{hemi}_sphere": "<f8",
            f"{hemi}_faces": "<i4",
            f"{hemi}_white": "<f4",
            f"{hemi}_cortex": "|u1",
            f"{hemi}_parcel": "<i2",
            f"{hemi}_area": "<f4",
        }
        expected |= {f"{hemi}_region_{name}": "|u1" for name in REGION_NAMES}
    assert list(arrays) == list(expected)
    assert {key: array.dtype.str for key, array in arrays.items()} == expected

    for content in (text.encode(), (directory / TEMPLATE_NPZ).read_bytes()):
        for private in (str(small.path), str(tmp_path), small.path.parent.name):
            assert private.encode() not in content


def test_reading_a_cache_that_is_absent_or_not_a_template(small4: Template, tmp_path: Path) -> None:
    directory = tmp_path / "cache"
    with pytest.raises(FileNotFoundError):
        read_template(directory)
    write_template(directory, small4)
    good_json = (directory / TEMPLATE_JSON).read_bytes()
    good_npz = (directory / TEMPLATE_NPZ).read_bytes()
    data = json.loads(good_json)

    def refused(json_bytes: bytes = good_json, npz_bytes: bytes | None = good_npz) -> str:
        (directory / TEMPLATE_JSON).write_bytes(json_bytes)
        if npz_bytes is None:
            (directory / TEMPLATE_NPZ).unlink(missing_ok=True)
        else:
            (directory / TEMPLATE_NPZ).write_bytes(npz_bytes)
        with pytest.raises(ValueError) as info:
            read_template(directory)
        return str(info.value)

    def as_json(**changes: object) -> bytes:
        return json.dumps(data | changes).encode()

    assert "not JSON" in refused(b"{")
    assert "schema 2; this surfcensus reads 1" in refused(as_json(schema=2))
    assert "schema None" in refused(b"[]")
    assert "template.json: origin" in refused(as_json(origin="somewhere"))
    assert "template.json: unknown" in refused(as_json(unknown=1))
    # edited by hand: the content no longer hashes to the id
    for changes in (
        {"regions": data["regions"] | {"cut_mm": 10.0}},
        {"mesh": "fsaverage5", "order": 5},
        {"order": 5},
        {"template_id": "0123456789ab"},
        {"files": data["files"] | {"surf/lh.white": {"size": 1, "sha256": "0" * 64}}},
    ):
        assert "is not that of the template" in refused(as_json(**changes))
    assert "pole vertex 2562" in refused(as_json(pole_vertex={"lh": 2562, "rh": 21}))

    assert "template.npz cannot be read" in refused(npz_bytes=good_npz[: len(good_npz) // 2])
    assert "template.npz cannot be read" in refused(npz_bytes=b"not an archive")
    with pytest.raises(FileNotFoundError):
        refused(npz_bytes=None)

    arrays = read_npz_bytes(good_npz, tmp_path)
    without = {key: array for key, array in arrays.items() if key != "rh_area"}
    assert "does not hold the arrays" in refused(npz_bytes=npz_bytes(without, tmp_path))
    extra = arrays | {"rh_region_other": arrays["rh_cortex"]}
    assert "does not hold the arrays" in refused(npz_bytes=npz_bytes(extra, tmp_path))
    short = arrays | {"lh_cortex": arrays["lh_cortex"][:-1]}
    assert "lh_cortex is |u1 (2561,), not u1 (2562,)" in refused(
        npz_bytes=npz_bytes(short, tmp_path)
    )
    wide = arrays | {"lh_white": arrays["lh_white"].astype("<f8")}
    assert "lh_white is <f8" in refused(npz_bytes=npz_bytes(wide, tmp_path))

    (directory / TEMPLATE_JSON).write_bytes(good_json)
    (directory / TEMPLATE_NPZ).write_bytes(good_npz)
    same_template(read_template(directory), small4)


def read_npz_bytes(content: bytes, scratch: Path) -> dict[str, np.ndarray]:
    path = scratch / "read.npz"
    path.write_bytes(content)
    return read_npz(path)


def npz_bytes(arrays: Mapping[str, np.ndarray], scratch: Path) -> bytes:
    from surfcensus._io import write_npz

    path = scratch / "written.npz"
    write_npz(path, arrays)
    return path.read_bytes()


def test_writing_removes_the_json_first(
    small4: Template, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A write that stops half-way leaves no ``template.json``, so no template."""
    directory = tmp_path / "cache"
    write_template(directory, small4)

    def stops(path: Path | str, arrays: Mapping[str, np.ndarray]) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(templates, "write_npz", stops)
    with pytest.raises(OSError, match="No space left"):
        write_template(directory, small4)
    assert not (directory / TEMPLATE_JSON).exists()
    with pytest.raises(FileNotFoundError):
        read_template(directory)


# --- the lookup -------------------------------------------------------------------------------


def config4(fsaverage_dir: Path | None = None, **changes: object) -> MetricsConfig:
    return MetricsConfig(
        mesh="fsaverage4",
        fsaverage_dir=None if fsaverage_dir is None else str(fsaverage_dir),
        **changes,
    )


def test_candidates_are_the_places_of_decision_d14_in_order(tmp_path: Path) -> None:
    adjacent = tmp_path / "subjects" / "fsaverage"
    assert template_candidates(adjacent=adjacent, env={"FREESURFER_HOME": "/opt/freesurfer"}) == [
        ("freesurfer_home", Path("/opt/freesurfer/subjects/fsaverage")),
        ("adjacent", adjacent),
        ("bundled", None),
    ]
    for env in ({}, {"FREESURFER_HOME": ""}, {"SUBJECTS_DIR": "/data"}):
        assert template_candidates(adjacent=None, env=env) == [
            ("freesurfer_home", None),
            ("adjacent", None),
            ("bundled", None),
        ]


def test_the_environment_is_read_when_none_is_given(
    small: SyntheticTemplate, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "freesurfer"
    copy_of(small, home / "subjects" / "fsaverage")
    monkeypatch.setenv("FREESURFER_HOME", str(home))
    assert template_candidates(adjacent=None)[0] == (
        "freesurfer_home",
        home / "subjects" / "fsaverage",
    )
    assert find_template(config=config4(), adjacent=None).origin == "freesurfer_home"
    monkeypatch.delenv("FREESURFER_HOME")
    assert template_candidates(adjacent=None)[0] == ("freesurfer_home", None)


def test_the_first_usable_candidate_is_built(
    small: SyntheticTemplate, small_bare: SyntheticTemplate, small4: Template, tmp_path: Path
) -> None:
    home = tmp_path / "freesurfer"
    in_home = copy_of(small_bare, home / "subjects" / "fsaverage")
    env = {"FREESURFER_HOME": str(home)}
    bare_id = build_template(small_bare.path, "explicit", mesh="fsaverage4").template_id

    both = find_template(config=config4(), adjacent=small.path, env=env)
    assert (both.origin, both.template_id) == ("freesurfer_home", bare_id)
    only_home = find_template(config=config4(), adjacent=None, env=env)
    assert (only_home.origin, only_home.template_id) == ("freesurfer_home", bare_id)
    only_adjacent = find_template(config=config4(), adjacent=small.path, env={})
    assert (only_adjacent.origin, only_adjacent.template_id) == ("adjacent", small4.template_id)
    same_template(only_adjacent, small4, origin=False)

    # a FreeSurfer install whose fsaverage lacks a file is passed over
    (in_home / "label" / "lh.cortex.label").unlink()
    passed_over = find_template(config=config4(), adjacent=small.path, env=env)
    assert (passed_over.origin, passed_over.template_id) == ("adjacent", small4.template_id)

    # the mesh and the regions of the config are those of the template
    regions = RegionConfig(cut_mm=30.0)
    configured = find_template(
        config=MetricsConfig(mesh="fsaverage5", regions=regions), adjacent=small.path, env={}
    )
    assert (configured.mesh, configured.regions) == ("fsaverage5", regions)


def test_the_default_config_coarsens_the_real_size_to_fsaverage6(
    synth_template: SyntheticTemplate,
    from_order7: Callable[[MeshName], Template],
    tmp_path: Path,
) -> None:
    """The path a local run takes: an ``fsaverage`` link next to the subjects, default
    settings, a directory of the size of the real one."""
    link = tmp_path / "subjects" / "fsaverage"
    link.parent.mkdir()
    link.symlink_to(synth_template.path, target_is_directory=True)
    found = find_template(config=MetricsConfig(), adjacent=link, env={})
    assert (found.mesh, found.order, found.origin) == ("fsaverage6", 6, "adjacent")
    assert len(found.hemis["lh"].sphere) == 40_962 and len(found.hemis["rh"].faces) == 81_920
    same_template(found, from_order7("fsaverage6"))


def test_template_not_found_lists_every_place(small: SyntheticTemplate, tmp_path: Path) -> None:
    with pytest.raises(TemplateNotFound) as info:
        find_template(config=MetricsConfig(), adjacent=None, env={})
    assert info.value.looked_in == NOTHING_FOUND
    assert str(info.value) == "no fsaverage template found"

    home = tmp_path / "freesurfer"
    in_home = copy_of(small, home / "subjects" / "fsaverage")
    (in_home / "surf" / "rh.sphere.reg").unlink()
    (in_home / "label" / "lh.aparc.annot").unlink()
    adjacent = tmp_path / "subjects" / "fsaverage"
    with pytest.raises(TemplateNotFound) as info:
        find_template(config=MetricsConfig(), adjacent=adjacent, env={"FREESURFER_HOME": str(home)})
    assert info.value.looked_in == (
        (
            "freesurfer_home",
            str(in_home),
            "incomplete: surf/rh.sphere.reg, label/lh.aparc.annot",
        ),
        ("adjacent", str(adjacent), "absent"),
        ("bundled", "bundled copy", "not shipped"),
    )


def test_an_explicit_directory_that_is_unusable_does_not_fall_through(
    small: SyntheticTemplate, tmp_path: Path
) -> None:
    home = tmp_path / "freesurfer"
    copy_of(small, home / "subjects" / "fsaverage")
    env = {"FREESURFER_HOME": str(home)}
    assert find_template(config=config4(), adjacent=small.path, env=env).origin == "freesurfer_home"

    missing = tmp_path / "typo" / "fsaverage"
    with pytest.raises(TemplateNotFound) as info:
        find_template(config=config4(missing), adjacent=small.path, env=env)
    assert info.value.looked_in == (("explicit", str(missing), "absent"),)

    incomplete = copy_of(small, tmp_path / "incomplete" / "fsaverage")
    (incomplete / "surf" / "rh.white").unlink()
    with pytest.raises(TemplateNotFound) as info:
        find_template(config=config4(incomplete), adjacent=small.path, env=env)
    assert info.value.looked_in == (("explicit", str(incomplete), "incomplete: surf/rh.white"),)

    # one that can be used wins over the environment
    explicit = find_template(config=config4(small.path), adjacent=None, env=env)
    assert explicit.origin == "explicit"


class Pinned:
    """A run that was scored once: its template in the cache, and the id it is pinned to."""

    def __init__(self, source: Path, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.template = find_template(config=config4(), adjacent=source, env={})
        self.id = self.template.template_id
        write_template(cache_dir / self.id, self.template)


@pytest.fixture
def pinned(small: SyntheticTemplate, tmp_path: Path) -> Pinned:
    return Pinned(small.path, tmp_path / "run" / "templates")


def test_a_pinned_template_is_used_whatever_the_environment_looks_like(
    small_bare: SyntheticTemplate,
    pinned: Pinned,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    home = tmp_path / "freesurfer"
    copy_of(small_bare, home / "subjects" / "fsaverage")
    other = {"FREESURFER_HOME": str(home)}
    assert find_template(config=config4(), adjacent=None, env=other).template_id != pinned.id

    def not_read(path: Path | str) -> None:
        raise AssertionError("a source file was read although the template is cached")

    monkeypatch.setattr(templates, "read_surface", not_read)
    monkeypatch.setattr(templates, "sha256_file", not_read)
    with caplog.at_level(logging.WARNING, logger="surfcensus"):
        for adjacent, env in (
            (None, {}),
            (None, other),
            (small_bare.path, {}),
            (tmp_path / "gone", other),
        ):
            found = find_template(
                config=config4(),
                adjacent=adjacent,
                cache_dir=pinned.cache_dir,
                pinned=pinned.id,
                env=env,
            )
            same_template(found, pinned.template)
            assert found.origin == "adjacent"
    assert caplog.records == []


def test_a_pinned_template_survives_an_explicit_directory_that_has_gone(
    pinned: Pinned, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    gone = tmp_path / "unmounted" / "fsaverage"
    with caplog.at_level(logging.WARNING, logger="surfcensus"):
        found = find_template(
            config=config4(gone),
            adjacent=None,
            cache_dir=pinned.cache_dir,
            pinned=pinned.id,
            env={},
        )
    same_template(found, pinned.template)
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING and record.name == "surfcensus"
    message = record.getMessage()
    assert "cannot be used (absent)" in message and pinned.id in message
    assert str(tmp_path) not in message

    # without the pin the same directory is an error, not a warning
    with pytest.raises(TemplateNotFound):
        find_template(config=config4(gone), adjacent=None, cache_dir=pinned.cache_dir, env={})


def test_an_explicit_directory_equal_to_the_pin_changes_nothing(
    small: SyntheticTemplate, pinned: Pinned, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    same_content = copy_of(small, tmp_path / "copy" / "fsaverage")
    with caplog.at_level(logging.WARNING, logger="surfcensus"):
        found = find_template(
            config=config4(same_content),
            adjacent=None,
            cache_dir=pinned.cache_dir,
            pinned=pinned.id,
            env={},
        )
    same_template(found, pinned.template)
    assert found.origin == "adjacent"  # the cached template, as it was first found
    assert caplog.records == []


def test_an_explicit_directory_that_differs_from_the_pin_stops_unless_retemplate(
    small_bare: SyntheticTemplate, pinned: Pinned
) -> None:
    other_id = build_template(small_bare.path, "explicit", mesh="fsaverage4").template_id
    assert other_id != pinned.id
    arguments = {
        "config": config4(small_bare.path),
        "adjacent": None,
        "cache_dir": pinned.cache_dir,
        "pinned": pinned.id,
        "env": {},
    }
    with pytest.raises(TemplateError) as info:
        find_template(**arguments)
    assert info.value.reason == "changed"
    assert info.value.detail == f"{pinned.id} -> {other_id}; pass --retemplate"
    assert str(info.value) == f"changed: {pinned.id} -> {other_id}; pass --retemplate"

    new = find_template(**arguments, retemplate=True)
    assert (new.template_id, new.origin, new.area_source) == (other_id, "explicit", "white mesh")
    # the cache is the caller's to write: nothing was added to it
    assert [path.name for path in pinned.cache_dir.iterdir()] == [pinned.id]


def test_retemplate_ignores_the_pin_and_looks_again(
    small_bare: SyntheticTemplate, pinned: Pinned
) -> None:
    arguments = {"config": config4(), "cache_dir": pinned.cache_dir, "pinned": pinned.id, "env": {}}
    new = find_template(adjacent=small_bare.path, retemplate=True, **arguments)
    assert new.template_id != pinned.id and new.origin == "adjacent"
    with pytest.raises(TemplateNotFound) as info:
        find_template(adjacent=None, retemplate=True, **arguments)
    assert info.value.looked_in == NOTHING_FOUND


def test_a_pinned_run_whose_cache_and_candidates_are_gone_raises_changed(
    small: SyntheticTemplate, small_bare: SyntheticTemplate, pinned: Pinned, tmp_path: Path
) -> None:
    shutil.rmtree(pinned.cache_dir / pinned.id)
    gone = "the template this run was scored with is no longer available; "
    gone += "pass --fsaverage DIR or --retemplate"
    home = tmp_path / "freesurfer"
    copy_of(small_bare, home / "subjects" / "fsaverage")
    for adjacent, env, cache_dir in (
        (None, {}, pinned.cache_dir),
        (tmp_path / "unmounted" / "fsaverage", {}, pinned.cache_dir),
        (None, {}, None),
        # a template is found, but it is not the one the run was scored with
        (small_bare.path, {}, pinned.cache_dir),
        (None, {"FREESURFER_HOME": str(home)}, pinned.cache_dir),
    ):
        with pytest.raises(TemplateError) as info:
            find_template(
                config=config4(), adjacent=adjacent, cache_dir=cache_dir, pinned=pinned.id, env=env
            )
        assert (info.value.reason, info.value.detail) == ("changed", gone)

    # the explicit directory gone as well: one warning, then the same error
    with pytest.raises(TemplateError, match=r"changed: .*no longer available"):
        find_template(
            config=config4(tmp_path / "typo"),
            adjacent=None,
            cache_dir=pinned.cache_dir,
            pinned=pinned.id,
            env={},
        )
    # an explicit directory with another template names both ids
    with pytest.raises(TemplateError, match=f"changed: {pinned.id} -> [0-9a-f]{{12}}; pass"):
        find_template(
            config=config4(small_bare.path),
            adjacent=small.path,
            cache_dir=pinned.cache_dir,
            pinned=pinned.id,
            env={},
        )


@pytest.mark.parametrize("damage", ["deleted", "npz cut", "json of another id", "no cache_dir"])
def test_a_pinned_template_is_built_again_from_a_source_that_still_gives_its_id(
    small: SyntheticTemplate,
    small_bare: SyntheticTemplate,
    pinned: Pinned,
    tmp_path: Path,
    damage: str,
) -> None:
    cached = pinned.cache_dir / pinned.id
    cache_dir: Path | None = pinned.cache_dir
    if damage == "deleted":
        shutil.rmtree(cached)
    elif damage == "npz cut":
        content = (cached / TEMPLATE_NPZ).read_bytes()
        (cached / TEMPLATE_NPZ).write_bytes(content[: len(content) // 3])
    elif damage == "json of another id":
        other = build_template(small_bare.path, "explicit", mesh="fsaverage4")
        write_template(cached, other)
    else:
        cache_dir = None

    home = tmp_path / "freesurfer"
    copy_of(small_bare, home / "subjects" / "fsaverage")
    # the first candidate gives another id and is passed over for the one that gives the pin
    rebuilt = find_template(
        config=config4(),
        adjacent=small.path,
        cache_dir=cache_dir,
        pinned=pinned.id,
        env={"FREESURFER_HOME": str(home)},
    )
    same_template(rebuilt, pinned.template)
    explicit = find_template(
        config=config4(small.path), adjacent=None, cache_dir=cache_dir, pinned=pinned.id, env={}
    )
    same_template(explicit, pinned.template, origin=False)
    assert explicit.origin == "explicit"


def test_a_pin_on_another_mesh_or_other_regions_than_the_config_is_a_change(
    small: SyntheticTemplate, pinned: Pinned
) -> None:
    start = f"the configuration no longer matches the run's template {pinned.id} in "
    for config, what in (
        (MetricsConfig(mesh="fsaverage5"), "the mesh (fsaverage4, now fsaverage5)"),
        (config4(regions=RegionConfig(cut_mm=30.0)), "the regions"),
        (
            MetricsConfig(mesh="fsaverage5", regions=RegionConfig(control=("superiorfrontal",))),
            "the mesh (fsaverage4, now fsaverage5) and the regions",
        ),
    ):
        arguments = {
            "config": config,
            "adjacent": small.path,
            "cache_dir": pinned.cache_dir,
            "pinned": pinned.id,
            "env": {},
        }
        with pytest.raises(TemplateError) as info:
            find_template(**arguments)
        assert info.value.reason == "changed"
        assert info.value.detail == f"{start}{what}; pass --retemplate"
        new = find_template(**arguments, retemplate=True)
        assert (new.mesh, new.regions) == (config.mesh, config.regions)
        assert new.template_id != pinned.id


def test_a_pin_that_is_not_an_id_names_no_directory(
    pinned: Pinned, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pin comes from a catalog; it is never joined to a path unless it is 12 hex."""
    outside = tmp_path / "outside"
    write_template(outside, pinned.template)

    def not_read(directory: Path) -> Template:
        raise AssertionError(f"looked for a cached template in {directory}")

    monkeypatch.setattr(templates, "read_template", not_read)
    for pin in ("../../outside", str(outside), "", pinned.id.upper(), pinned.id + "0"):
        with pytest.raises(TemplateError, match=r"changed: .*no longer available"):
            find_template(
                config=config4(), adjacent=None, cache_dir=pinned.cache_dir, pinned=pin, env={}
            )


# --- the real fsaverage -----------------------------------------------------------------------

# where conftest.real_template and conftest.real_fsaverage5 keep the files
FSAVERAGE_CACHE = "ds000102-fsaverage"
FSAVERAGE5_CACHE = "ds000102-fsaverage5"
REAL_FILES = {  # bytes, from the bucket listing of 2026-11-13
    "surf/lh.sphere.reg": 5_898_546,
    "surf/rh.sphere.reg": 5_898_546,
    "surf/lh.white": 5_898_808,
    "surf/rh.white": 5_898_808,
    "label/lh.cortex.label": 6_117_569,
    "label/rh.cortex.label": 5_964_904,
    "label/lh.aparc.annot": 1_312_250,
    "label/rh.aparc.annot": 1_312_250,
    "surf/lh.white.avg.area.mgh": 655_672,
    "surf/rh.white.avg.area.mgh": 655_672,
    "scripts/build-stamp.txt": 49,
}
REAL_FSAVERAGE5_SPHERE = ("surf/lh.sphere.reg", 368_946)
REAL_ID = "b710aa6e0467"
"""fsaverage6 with the default regions from these files, average areas included."""
REAL_REGIONS = {  # vertices on fsaverage6, measured 2026-11-13
    "lh": {"search": 4856, "medial": 1851, "inferior": 1405, "lateral": 1600, "control": 5789},
    "rh": {"search": 4715, "medial": 1874, "inferior": 1351, "lateral": 1490, "control": 5804},
}
REAL_CORTEX = {"lh": 37_487, "rh": 37_482}
REAL_POLE = {"lh": 12_950, "rh": 5_558}
REAL_AVG_AREA_MM2 = {"lh": 82_219.95, "rh": 82_167.50}
REAL_MESH_AREA_MM2 = {"lh": 65_095.56, "rh": 64_691.81}
REAL_AREA_RATIO = {  # white-mesh area over average area on lh, by parcel
    "lateraloccipital": 0.588,
    "cuneus": 0.680,
    "lingual": 0.722,
    "insula": 1.009,
}


def check_real_template(fsaverage: Path, fsaverage5: Path, scratch: Path) -> None:
    template = build_template(fsaverage, "adjacent")
    assert (template.mesh, template.order, template.template_id) == ("fsaverage6", 6, REAL_ID)
    assert template.area_source == "white.avg.area"
    assert template.build_stamp == "freesurfer-x86_64-redhat-linux-gnu-dev4-20090216"
    assert template.files == {
        name: (size, sha256_file(fsaverage / name))
        for name, size in REAL_FILES.items()
        if name != "scripts/build-stamp.txt"
    }
    assert len(template.parcel_names) == 36 and template.parcel_names[0] == "unknown"
    for hemi in HEMIS:
        half = template.hemis[hemi]
        assert half.sphere.shape == (40_962, 3) and half.faces.shape == (81_920, 3)
        assert outward(half.sphere, half.faces).all()
        degree = template.adjacency(hemi).sum(axis=1).A1
        assert dict(zip(*np.unique(degree, return_counts=True), strict=True)) == {5: 12, 6: 40_950}
        assert int(half.cortex.sum()) == REAL_CORTEX[hemi]
        assert half.pole_vertex == REAL_POLE[hemi]
        assert {name: int(mask.sum()) for name, mask in half.regions.items()} == REAL_REGIONS[hemi]
        total = file_areas(fsaverage, hemi).sum()
        assert total == pytest.approx(REAL_AVG_AREA_MM2[hemi], abs=0.01)
        assert half.area.sum(dtype=np.float64) == pytest.approx(total, rel=1e-6)
        # a rule that gave each dropped vertex whole to the nearer side: 0.21 to 7.65 on lh
        assert 0.6 < half.area.min() and half.area.max() < 6.0

    # one more step gives the faces of the fsaverage5 that lies next to it
    vertices5, faces5 = fs.read_geometry(str(fsaverage5 / REAL_FSAVERAGE5_SPHERE[0]))
    lh = template.hemis["lh"]
    assert np.array_equal(coarsen_faces(lh.faces, 10_242), faces5)
    assert np.allclose(lh.sphere[:10_242] * 100.0, vertices5, atol=0.01)
    on5 = build_template(fsaverage, "adjacent", mesh="fsaverage5")
    assert np.array_equal(on5.hemis["lh"].faces, faces5)
    assert on5.hemis["lh"].area.sum(dtype=np.float64) == pytest.approx(
        REAL_AVG_AREA_MM2["lh"], abs=0.02
    )

    # with the average-area files withheld the area is that of the white mesh
    bare_dir = scratch / "fsaverage"
    for name in REAL_FILES:
        if "avg.area" not in name:
            (bare_dir / name).parent.mkdir(parents=True, exist_ok=True)
            (bare_dir / name).symlink_to(fsaverage / name)
    bare = build_template(bare_dir, "adjacent")
    assert bare.area_source == "white mesh" and bare.template_id != REAL_ID
    for hemi in HEMIS:
        assert bare.hemis[hemi].area.sum(dtype=np.float64) == pytest.approx(
            REAL_MESH_AREA_MM2[hemi], abs=1.0
        )
    for parcel, ratio in REAL_AREA_RATIO.items():
        members = (lh.parcel == template.parcel_names.index(parcel)) & lh.cortex
        measured = bare.hemis["lh"].area[members].sum() / lh.area[members].sum()
        assert measured == pytest.approx(ratio, abs=0.002), parcel

    write_template(scratch / "cache", template)
    same_template(read_template(scratch / "cache"), template)
    assert (scratch / "cache" / TEMPLATE_NPZ).stat().st_size < 6_500_000


def real_files_present(fsaverage: Path, fsaverage5: Path) -> bool:
    wanted = [(fsaverage / name, size) for name, size in REAL_FILES.items()]
    wanted.append((fsaverage5 / REAL_FSAVERAGE5_SPHERE[0], REAL_FSAVERAGE5_SPHERE[1]))
    return all(path.is_file() and path.stat().st_size == size for path, size in wanted)


def test_cached_real_template(cached_subjects: Path, tmp_path: Path) -> None:
    fsaverage, fsaverage5 = cached_subjects / FSAVERAGE_CACHE, cached_subjects / FSAVERAGE5_CACHE
    if not real_files_present(fsaverage, fsaverage5):
        pytest.skip("the real fsaverage template files are not in the pytest cache")
    check_real_template(fsaverage, fsaverage5, tmp_path)


@pytest.mark.network
def test_real_template(real_template: Path, real_fsaverage5: Path, tmp_path: Path) -> None:
    check_real_template(real_template, real_fsaverage5, tmp_path)


@pytest.mark.parametrize("damage", ["npz", "json", "other_id"])
def test_store_template_writes_a_damaged_cache_again(
    small4: Template, tmp_path: Path, damage: str
) -> None:
    """Worker processes read the template from this directory: one that does not read back
    as the template must not stay, whatever ``template.json`` says."""
    directory = templates.store_template(tmp_path, small4)
    files = {path.name: path.read_bytes() for path in sorted(directory.iterdir())}
    if damage == "npz":
        (directory / "template.npz").write_bytes(files["template.npz"][:1000])
    elif damage == "json":
        (directory / "template.json").write_text("{")
    else:
        text = files["template.json"].decode().replace(small4.template_id, "0" * 12)
        (directory / "template.json").write_text(text)
    assert templates.store_template(tmp_path, small4) == directory
    assert {path.name: path.read_bytes() for path in sorted(directory.iterdir())} == files
    same_template(templates.read_template(directory), small4)


def test_store_template_leaves_a_whole_cache_alone(small4: Template, tmp_path: Path) -> None:
    directory = templates.store_template(tmp_path, small4)
    before = {path.name: path.stat().st_mtime_ns for path in directory.iterdir()}
    assert templates.store_template(tmp_path, small4) == directory
    assert {path.name: path.stat().st_mtime_ns for path in directory.iterdir()} == before
