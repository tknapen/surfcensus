"""The synthetic template is what ``tests/synth_template.py`` says it is.

Everything is checked against nibabel and numpy, not against the package's own template
code, so these tests hold while that code is being written and say what it can rely on.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer as fs
import numpy as np
import pytest
from synth import WHITE_SEMI_AXES, icosphere  # pyrefly: ignore[missing-import]
from synth_template import (  # pyrefly: ignore[missing-import]
    AVG_AREA_FACTOR_CAP,
    AVG_AREA_FACTOR_REST,
    DESIKAN_NAMES,
    POSTERIOR_CAP_DEG,
    REGION_NAMES,
    SPHERE_RADIUS,
    TEMPLATE_BUILD_STAMP,
    WHITE_CENTRE,
    SyntheticTemplate,
    desikan_colours,
    direction,
    expected_regions,
    medial_coordinate,
    mesh_vertex_areas,
    parcel_of,
    template_mesh,
    template_white,
    write_synthetic_template,
)

from surfcensus.config import RegionConfig
from surfcensus.inventory.provenance import parse_build_stamp
from surfcensus.sources.base import HEMIS
from surfcensus.sources.local import find_adjacent_template, template_files_missing
from surfcensus.sources.patterns import TEMPLATE_FILES, TEMPLATE_OPTIONAL_FILES, is_template_name

DEFAULT_REGIONS = RegionConfig()
USED_PARCELS = {
    "pericalcarine",
    "cuneus",
    "lingual",
    "lateraloccipital",
    "precuneus",
    "fusiform",
    "superiorfrontal",
    "rostralmiddlefrontal",
    "caudalmiddlefrontal",
    "superiorparietal",
    "inferiortemporal",
}


def n_vertices(order: int) -> int:
    return 10 * 4**order + 2


def tree_digest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def parents_of_fine_vertices(faces: np.ndarray, n_coarse: int) -> np.ndarray:
    """(n - n_coarse, 2): the two neighbours below ``n_coarse`` of every vertex from
    ``n_coarse`` on, ascending. Fails if a vertex has another number of them."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.concatenate([edges, edges[:, ::-1]])
    down = np.unique(edges[(edges[:, 0] >= n_coarse) & (edges[:, 1] < n_coarse)], axis=0)
    n_fine = int(faces.max()) + 1 - n_coarse
    assert np.array_equal(np.bincount(down[:, 0] - n_coarse, minlength=n_fine), np.full(n_fine, 2))
    return down[:, 1].reshape(n_fine, 2)


def coarsen_by_the_documented_rule(faces: np.ndarray, n_coarse: int) -> np.ndarray:
    """The rule of docs (templates.coarsen_faces), written out a second time: a face of
    three fine vertices ``(m0, m1, m2)`` becomes ``(a, b, c)`` with ``a`` the parent shared
    by ``m2`` and ``m0``, ``b`` by ``m0`` and ``m1``, ``c`` by ``m1`` and ``m2``."""
    parents = parents_of_fine_vertices(faces, n_coarse)
    fine = faces[(faces >= n_coarse).all(axis=1)]
    assert len(fine) == 2 * n_coarse - 4

    def shared(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        p, q = parents[first - n_coarse], parents[second - n_coarse]
        in_q = (p[:, :1] == q).any(axis=1)
        assert np.array_equal(in_q, ~(p[:, 1:] == q).any(axis=1))  # exactly one is shared
        return np.where(in_q, p[:, 0], p[:, 1])

    m0, m1, m2 = fine.T
    return np.column_stack([shared(m2, m0), shared(m0, m1), shared(m1, m2)])


@pytest.fixture(scope="module")
def small_template(tmp_path_factory: pytest.TempPathFactory) -> SyntheticTemplate:
    """Order 5: 10,242 vertices, written in a fraction of a second."""
    return write_synthetic_template(tmp_path_factory.mktemp("template5"), order=5)


# ---------------------------------------------------------------------------------------
# the mesh


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_the_mesh_nests_and_coarsens_to_the_mesh_one_order_below(order: int) -> None:
    directions, faces = template_mesh(order)
    coarse_directions, coarse_faces = template_mesh(order - 1)
    assert directions.shape == (n_vertices(order), 3) and faces.shape == (20 * 4**order, 3)
    assert np.array_equal(directions[: len(coarse_directions)], coarse_directions)
    coarsened = coarsen_by_the_documented_rule(faces, len(coarse_directions))
    assert np.array_equal(coarsened, coarse_faces)


def test_the_mesh_of_the_default_order_can_be_coarsened_to_fsaverage6(
    synth_template: SyntheticTemplate,
) -> None:
    assert synth_template.order == 7 and synth_template.n_vertices == 163_842
    directions, faces = template_mesh(7)
    coarsened = coarsen_by_the_documented_rule(faces, n_vertices(6))
    assert np.array_equal(coarsened, template_mesh(6)[1])
    assert coarsened.shape == (81_920, 3)
    # wound outward, like the faces it came from
    a, b, c = (directions[coarsened[:, k]] for k in range(3))
    assert (np.einsum("ij,ij->i", a, np.cross(b, c)) > 0).all()
    degree = np.bincount(coarsened.ravel(), minlength=n_vertices(6))
    assert sorted(np.unique(degree)) == [5, 6] and (degree == 5).sum() == 12


def test_template_mesh_is_the_icosphere_of_the_subjects_and_is_read_only() -> None:
    directions, faces = template_mesh(3)
    reference, reference_faces = icosphere(3)
    assert np.array_equal(directions, reference) and np.array_equal(faces, reference_faces)
    assert faces.dtype == np.int32
    assert template_mesh(3)[0] is directions
    with pytest.raises(ValueError, match="read-only"):
        directions[0, 0] = 2.0


# ---------------------------------------------------------------------------------------
# the files


def test_the_directory_holds_every_template_file(small_template: SyntheticTemplate) -> None:
    path = small_template.path
    assert path.name == "fsaverage" and is_template_name(path.name)
    assert template_files_missing(path) == ()
    assert find_adjacent_template(path.parent) == path
    expected = {name.format(h=hemi) for name in TEMPLATE_FILES for hemi in HEMIS}
    expected |= {name.format(h=hemi) for name in TEMPLATE_OPTIONAL_FILES for hemi in HEMIS}
    expected.add("scripts/build-stamp.txt")
    assert set(tree_digest(path)) == expected
    stamp = parse_build_stamp((path / "scripts" / "build-stamp.txt").read_text())
    assert stamp.raw == TEMPLATE_BUILD_STAMP and stamp.version is None


def test_without_the_average_area_only_that_file_is_missing(tmp_path: Path) -> None:
    template = write_synthetic_template(tmp_path, name="fsaverage_bare", order=3)
    with_area = write_synthetic_template(tmp_path, name="fsaverage", order=3)
    plain = write_synthetic_template(tmp_path, name="other", order=3, with_avg_area=False)
    assert template.with_avg_area and not plain.with_avg_area
    assert not list(plain.path.glob("surf/*.avg.area.mgh"))
    assert template_files_missing(plain.path) == ()
    full, bare = tree_digest(with_area.path), tree_digest(plain.path)
    assert set(full) - set(bare) == {"surf/lh.white.avg.area.mgh", "surf/rh.white.avg.area.mgh"}
    assert all(full[name] == digest for name, digest in bare.items())


def test_two_writes_give_the_same_bytes(tmp_path: Path) -> None:
    first = write_synthetic_template(tmp_path / "a", order=4)
    second = write_synthetic_template(tmp_path / "b", order=4)
    assert tree_digest(first.path) == tree_digest(second.path)
    assert first.pole_vertex == second.pole_vertex


@pytest.mark.parametrize("hemi", HEMIS)
def test_sphere_and_white_surface(small_template: SyntheticTemplate, hemi: str) -> None:
    directions, faces = template_mesh(small_template.order)
    sphere, sphere_faces = fs.read_geometry(str(small_template.path / f"surf/{hemi}.sphere.reg"))
    white, white_faces = fs.read_geometry(str(small_template.path / f"surf/{hemi}.white"))
    assert len(sphere) == small_template.n_vertices == n_vertices(5)
    assert np.array_equal(sphere_faces, faces) and np.array_equal(white_faces, faces)
    np.testing.assert_allclose(np.linalg.norm(sphere, axis=1), SPHERE_RADIUS, atol=1e-4)
    np.testing.assert_allclose(sphere, directions * SPHERE_RADIUS, atol=1e-5)
    np.testing.assert_allclose(white, template_white(directions, hemi), atol=1e-5)
    np.testing.assert_allclose(
        white, np.array(WHITE_CENTRE[hemi]) + directions * np.array(WHITE_SEMI_AXES), atol=1e-5
    )


@pytest.mark.parametrize("hemi", HEMIS)
def test_the_pole_is_the_most_posterior_cortex_vertex(
    small_template: SyntheticTemplate, hemi: str
) -> None:
    directions, _ = template_mesh(small_template.order)
    white = fs.read_geometry(str(small_template.path / f"surf/{hemi}.white"))[0]
    cortex = small_template.masks[hemi]["cortex"]
    pole = small_template.pole_vertex
    assert np.array_equal(directions[pole], [0.0, -1.0, 0.0])
    assert pole < 42  # a vertex of every mesh from order 1 on
    assert cortex[pole]
    posterior = np.flatnonzero(white[:, 1] == white[cortex, 1].min())
    assert posterior.tolist() == [pole]
    assert white[pole, 1] == pytest.approx(WHITE_CENTRE[hemi][1] - WHITE_SEMI_AXES[1], abs=1e-5)


@pytest.mark.parametrize("hemi", HEMIS)
def test_annotation_and_cortex_label(small_template: SyntheticTemplate, hemi: str) -> None:
    directions, _ = template_mesh(small_template.order)
    masks = small_template.masks[hemi]
    labels, ctab, names = fs.read_annot(str(small_template.path / f"label/{hemi}.aparc.annot"))
    assert tuple(name.decode() for name in names) == DESIKAN_NAMES
    assert np.array_equal(ctab[:, :4], desikan_colours())
    assert np.array_equal(labels, parcel_of(directions, hemi))
    assert {DESIKAN_NAMES[index] for index in np.unique(labels[labels >= 0])} == USED_PARCELS
    assert np.array_equal(labels == -1, masks["medial_wall"])
    for name in USED_PARCELS:
        assert np.array_equal(labels == DESIKAN_NAMES.index(name), masks[name])
        assert masks[name].any()

    listed = fs.read_label(str(small_template.path / f"label/{hemi}.cortex.label"))
    assert (np.diff(listed) < 0).all()  # not sorted, like the real file
    cortex = np.zeros(len(labels), dtype=bool)
    cortex[listed] = True
    assert np.array_equal(cortex, masks["cortex"])
    # no vertex with a parcel lies outside the label and none inside it lacks one, as on
    # the real fsaverage (checked on its lh on 2026-11-13)
    assert np.array_equal(cortex, labels >= 0)
    assert 0.05 < masks["medial_wall"].mean() < 0.10


def test_the_colours_are_distinct_and_none_is_black() -> None:
    colours = desikan_colours()
    assert colours.shape == (36, 4) and colours[:, :3].min() >= 0 and colours[:, :3].max() <= 255
    assert len({tuple(row) for row in colours[:, :3]}) == 36
    assert (colours[:, :3].sum(axis=1) > 0).all()


@pytest.mark.parametrize("hemi", HEMIS)
def test_the_average_area_file_carries_the_planted_factors(
    small_template: SyntheticTemplate, hemi: str
) -> None:
    directions, faces = template_mesh(small_template.order)
    image = nib.load(str(small_template.path / f"surf/{hemi}.white.avg.area.mgh"))
    assert isinstance(image, nib.MGHImage)
    area = np.asarray(image.dataobj)
    # the shape and type of fsaverage/surf/lh.white.avg.area.mgh (read 2026-11-13)
    assert area.shape == (small_template.n_vertices, 1, 1) and area.dtype == np.dtype(">f4")
    mesh_area = mesh_vertex_areas(template_white(directions, hemi), faces)
    cap = small_template.masks[hemi]["posterior_cap"]
    np.testing.assert_allclose(area.ravel()[cap] / mesh_area[cap], AVG_AREA_FACTOR_CAP, rtol=1e-6)
    np.testing.assert_allclose(
        area.ravel()[~cap] / mesh_area[~cap], AVG_AREA_FACTOR_REST, rtol=1e-6
    )
    assert np.array_equal(cap, directions[:, 1] < -np.cos(np.deg2rad(POSTERIOR_CAP_DEG)))


def test_mesh_vertex_areas_sum_to_the_surface_area() -> None:
    directions, faces = template_mesh(5)
    area = mesh_vertex_areas(directions, faces)
    assert area.shape == (len(directions),) and (area > 0).all()
    assert area.sum() == pytest.approx(4 * np.pi, rel=2e-3)  # an inscribed mesh, a little less
    assert area.sum() < 4 * np.pi


# ---------------------------------------------------------------------------------------
# the rules


@pytest.mark.parametrize("hemi", HEMIS)
def test_direction_names_the_sectors_of_the_posterior_cap(hemi: str) -> None:
    def parcel(**where: float) -> str:
        return DESIKAN_NAMES[int(parcel_of(np.array(direction(hemi, **where)), hemi))]

    assert direction(hemi, from_pole_deg=0.0) == pytest.approx((0.0, -1.0, 0.0))
    assert parcel(from_pole_deg=0.0) == "pericalcarine"
    assert parcel(from_pole_deg=20.0, azimuth_deg=0.0) == "pericalcarine"
    assert parcel(from_pole_deg=20.0, azimuth_deg=90.0) == "cuneus"
    assert parcel(from_pole_deg=20.0, azimuth_deg=180.0) == "lateraloccipital"
    assert parcel(from_pole_deg=20.0, azimuth_deg=-90.0) == "lingual"
    assert parcel(from_pole_deg=60.0, azimuth_deg=90.0) == "precuneus"
    assert parcel(from_pole_deg=60.0, azimuth_deg=-90.0) == "fusiform"
    assert parcel(from_pole_deg=110.0, azimuth_deg=90.0) == "superiorparietal"
    assert parcel(from_pole_deg=110.0, azimuth_deg=-90.0) == "inferiortemporal"
    assert parcel(from_pole_deg=170.0, azimuth_deg=0.0) == "superiorfrontal"
    medial = np.array(direction(hemi, from_pole_deg=90.0, azimuth_deg=0.0))
    assert medial_coordinate(medial, hemi) == pytest.approx(1.0)
    assert medial[0] == pytest.approx(1.0 if hemi == "lh" else -1.0)
    assert int(parcel_of(medial, hemi)) == -1
    assert np.linalg.norm(direction(hemi, from_pole_deg=33.0, azimuth_deg=-71.0)) == pytest.approx(
        1
    )


def test_the_hemispheres_mirror_each_other() -> None:
    directions, _ = template_mesh(4)
    mirrored = directions * np.array([-1.0, 1.0, 1.0])
    assert np.array_equal(parcel_of(directions, "lh"), parcel_of(mirrored, "rh"))
    assert not np.array_equal(parcel_of(directions, "lh"), parcel_of(directions, "rh"))


@pytest.mark.parametrize("hemi", HEMIS)
@pytest.mark.parametrize("order", [4, 5])
def test_the_regions_follow_the_rule_of_the_docs(hemi: str, order: int) -> None:
    directions, _ = template_mesh(order)
    parcel = parcel_of(directions, hemi)
    cortex = parcel >= 0
    regions = expected_regions(directions, hemi)
    assert tuple(regions) == REGION_NAMES

    def named(*names: str) -> np.ndarray:
        return np.isin(parcel, [DESIKAN_NAMES.index(name) for name in names])

    for mask in regions.values():
        assert mask.any() and not (mask & ~cortex).any()
    parts = regions["medial"].astype(int) + regions["inferior"] + regions["lateral"]
    assert np.array_equal(parts, regions["search"].astype(int))  # a partition
    assert np.array_equal(regions["control"], named(*DEFAULT_REGIONS.control))
    assert not (regions["control"] & regions["search"]).any()
    # the occipital parcels are in whole, the posterior parcels up to the cut
    occipital = named(*DEFAULT_REGIONS.occipital)
    assert np.array_equal(regions["search"] & occipital, occipital)
    posterior = named(*DEFAULT_REGIONS.posterior)
    white_y = template_white(directions, hemi)[:, 1]
    behind = white_y <= white_y[int(np.argmin(white_y))] + DEFAULT_REGIONS.cut_mm
    assert np.array_equal(regions["search"] & posterior, posterior & behind)
    assert (posterior & behind).any() and (posterior & ~behind).any()  # the cut divides them
    assert np.array_equal(regions["lateral"], named("lateraloccipital"))
    assert not (regions["search"] & ~(occipital | posterior)).any()


@pytest.mark.parametrize("hemi", HEMIS)
def test_the_cut_moves_the_posterior_parcels_only(hemi: str) -> None:
    directions, _ = template_mesh(5)
    near = expected_regions(directions, hemi, RegionConfig(cut_mm=30.0))
    default = expected_regions(directions, hemi)
    far = expected_regions(directions, hemi, RegionConfig(cut_mm=60.0))
    parcel = parcel_of(directions, hemi)
    posterior = np.isin(parcel, [DESIKAN_NAMES.index(name) for name in DEFAULT_REGIONS.posterior])
    assert near["search"].sum() < default["search"].sum() < far["search"].sum()
    assert not ((near["search"] != far["search"]) & ~posterior).any()
    assert np.array_equal(near["lateral"], far["lateral"])
    assert np.array_equal(near["control"], far["control"])
    # another parcel list moves the parts
    swapped = expected_regions(
        directions, hemi, RegionConfig(medial=("lingual",), inferior=("cuneus",))
    )
    assert np.array_equal(swapped["search"], default["search"])
    assert np.array_equal(swapped["medial"], parcel == DESIKAN_NAMES.index("lingual"))


def test_the_masks_of_the_fixture_are_the_rules_and_are_read_only(
    small_template: SyntheticTemplate,
) -> None:
    directions, _ = template_mesh(small_template.order)
    for hemi in HEMIS:
        masks = small_template.masks[hemi]
        assert set(masks) == {
            "cortex",
            "medial_wall",
            "posterior_cap",
            *USED_PARCELS,
            *REGION_NAMES,
        }
        for name, mask in expected_regions(directions, hemi).items():
            assert np.array_equal(masks[name], mask)
        assert np.array_equal(masks["cortex"], ~masks["medial_wall"])
        with pytest.raises(ValueError, match="read-only"):
            masks["cortex"][0] = False
    # the masks of a coarser mesh are the first entries: the rules are functions of the
    # direction, and the meshes nest
    coarse_directions, _ = template_mesh(4)
    for name, mask in expected_regions(coarse_directions, "lh").items():
        assert np.array_equal(small_template.masks["lh"][name][: len(coarse_directions)], mask)


def test_the_template_of_the_session(synth_template: SyntheticTemplate) -> None:
    # what the default code path gets: order 7, every file, every region on fsaverage6 and
    # on the small mesh that tests ask for (fsaverage4)
    assert synth_template.path.name == "fsaverage" and synth_template.with_avg_area
    assert template_files_missing(synth_template.path) == ()
    assert synth_template.pole_vertex < 42
    sphere, faces = fs.read_geometry(str(synth_template.path / "surf/lh.sphere.reg"))
    assert sphere.shape == (163_842, 3) and faces.shape == (327_680, 3)
    for hemi in HEMIS:
        masks = synth_template.masks[hemi]
        assert masks["cortex"].shape == (163_842,)
        for count in (n_vertices(6), n_vertices(4)):
            for name in REGION_NAMES:
                assert masks[name][:count].sum() >= 50, (hemi, name, count)
        assert masks["cortex"][synth_template.pole_vertex]
        assert masks["search"][synth_template.pole_vertex]


@pytest.mark.parametrize("hemi", HEMIS)
def test_no_vertex_lies_on_the_default_cut(hemi: str) -> None:
    # the white surface is stored as float32 (rounding below 4e-6 mm at these coordinates),
    # and a vertex that close to the cut could fall on either side of it in the package and
    # in expected_regions. None of the meshes used comes nearer than a micrometre.
    directions, _ = template_mesh(7)
    white_y = template_white(directions, hemi)[:, 1]
    cut = WHITE_CENTRE[hemi][1] - WHITE_SEMI_AXES[1] + DEFAULT_REGIONS.cut_mm
    assert np.abs(white_y - cut).min() > 1e-3
