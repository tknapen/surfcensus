"""Outlier clusters on the template mesh and their centroids on the native surface.

z-maps are planted on templates of ``synth_template`` (an order-5 directory, used as
fsaverage4 and fsaverage5). Clusters are compared with a search written here, one vertex at
a time. The native surface of the last tests is the pial ellipsoid of the synthetic
subject, registered through the closed-form warp of ``synth_tierb``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.spatial import cKDTree
from synth import PIAL_SEMI_AXES, icosphere  # pyrefly: ignore[missing-import]
from synth_template import (  # pyrefly: ignore[missing-import]
    AVG_AREA_FACTOR_CAP,
    SyntheticTemplate,
    direction,
    mesh_vertex_areas,
    template_mesh,
    template_white,
    write_synthetic_template,
)
from synth_tierb import unwarp_directions, warp_directions  # pyrefly: ignore[missing-import]

from surfcensus import schemas
from surfcensus.metrics.clusters import NATIVE_COLUMNS, Cluster, find_clusters, native_centroids
from surfcensus.render import vertexmap
from surfcensus.sources.base import HEMIS
from surfcensus.store import read_table, write_table
from surfcensus.templates import Template, TemplateHemi, build_template

THRESHOLD = 3.0
WARP = 0.15
PIAL_CENTRE = np.array([-34.0, -10.0, 5.0])


@pytest.fixture(scope="module")
def small(tmp_path_factory: pytest.TempPathFactory) -> SyntheticTemplate:
    """An ``fsaverage`` directory of order 5. Read-only."""
    return write_synthetic_template(tmp_path_factory.mktemp("small"), order=5)


@pytest.fixture(scope="module")
def tpl4(small: SyntheticTemplate) -> Template:
    """2,562 vertices per hemisphere, about 4 degrees apart."""
    return build_template(small.path, "explicit", mesh="fsaverage4")


@pytest.fixture(scope="module")
def tpl5(small: SyntheticTemplate) -> Template:
    """10,242 vertices per hemisphere, about 2 degrees apart: the mesh of the directory."""
    return build_template(small.path, "explicit", mesh="fsaverage5")


def nearest_vertex(half: TemplateHemi, towards: tuple[float, float, float]) -> int:
    return int(np.argmax(half.sphere @ np.asarray(towards)))


def angle_deg(half: TemplateHemi, vertex: int) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(half.sphere @ half.sphere[vertex], -1.0, 1.0)))


def planted(half: TemplateHemi, members: np.ndarray | Sequence[int], z: float) -> np.ndarray:
    """A z-map that is ``z`` on ``members`` (a mask or indices) and 0 elsewhere."""
    values = np.zeros(len(half.sphere), dtype=np.float32)
    values[np.asarray(members)] = z
    return values


def everywhere(half: TemplateHemi) -> np.ndarray:
    return np.ones(len(half.sphere), dtype=bool)


def neighbours(adjacency: sparse.csr_matrix, vertex: int) -> np.ndarray:
    return adjacency.indices[adjacency.indptr[vertex] : adjacency.indptr[vertex + 1]]


def star(adjacency: sparse.csr_matrix, centre: int, size: int) -> list[int]:
    """A connected set of ``size`` vertices: ``centre`` and its first neighbours."""
    return [centre, *neighbours(adjacency, centre)[: size - 1].tolist()]


def apart(adjacency: sparse.csr_matrix, *groups: Sequence[int]) -> bool:
    """No vertex in two groups and no edge between two groups."""
    for position, first in enumerate(groups):
        for second in groups[position + 1 :]:
            if set(first) & set(second) or adjacency[list(first)][:, list(second)].nnz:
                return False
    return True


def components(
    z: np.ndarray, mask: np.ndarray, adjacency: sparse.csr_matrix, threshold: float
) -> list[list[int]]:
    """Connected sets of vertices above the threshold inside the mask, found by walking
    the mesh from each one; every set ascending, the sets by their first vertex."""
    above = mask & np.isfinite(z) & (z > threshold)
    seen = np.zeros(len(z), dtype=bool)
    found = []
    for seed in np.flatnonzero(above):
        if seen[seed]:
            continue
        seen[seed] = True
        todo, members = [int(seed)], []
        while todo:
            vertex = todo.pop()
            members.append(vertex)
            for other in neighbours(adjacency, vertex):
                if above[other] and not seen[other]:
                    seen[other] = True
                    todo.append(int(other))
        found.append(sorted(members))
    return found


def as_sets(clusters: list[Cluster]) -> list[list[int]]:
    return sorted(cluster.vertices.tolist() for cluster in clusters)


def expected_centroid(half: TemplateHemi, vertices: np.ndarray, z: np.ndarray) -> int:
    """The member nearest the normalized z-weighted mean of the members' positions."""
    positions = half.sphere[vertices]
    mean = (z[vertices].astype(np.float64)[:, None] * positions).sum(axis=0)
    mean /= np.linalg.norm(mean)
    return int(vertices[np.argmin(np.linalg.norm(positions - mean, axis=1))])


# --- find_clusters ---


def test_one_planted_disc_is_one_cluster(tpl4: Template) -> None:
    half, adjacency = tpl4.hemis["lh"], tpl4.adjacency("lh")
    centre = nearest_vertex(half, direction("lh", from_pole_deg=20.0, azimuth_deg=30.0))
    disc = angle_deg(half, centre) <= 12.0
    assert disc.sum() == 30 and half.regions["search"][disc].all()
    z = planted(half, disc, 5.0)
    (cluster,) = find_clusters(z, half.regions["search"], half, adjacency, threshold=THRESHOLD)

    assert cluster.vertices.dtype == np.int32
    assert np.array_equal(cluster.vertices, np.flatnonzero(disc))
    assert cluster.area_mm2 == pytest.approx(half.area[disc].sum(dtype=np.float64), rel=1e-12)
    assert cluster.mass == 5.0 * 30
    assert cluster.excess_mass == (5.0 - THRESHOLD) * 30
    assert cluster.peak_z == 5.0
    assert cluster.peak_vertex == int(np.flatnonzero(disc)[0])  # all equal: the lowest index
    assert cluster.centroid_vertex == centre
    assert type(cluster.area_mm2) is float and type(cluster.centroid_vertex) is int

    # the area is real cortex, not the template's own mesh: the disc's area on the white
    # ellipsoid, measured on the mesh of the directory, times the planted factor of the
    # average-area file. The coarse disc is that disc to within half a vertex spacing.
    assert tpl4.area_source == "white.avg.area"
    fine_directions, fine_faces = template_mesh(5)
    fine_area = mesh_vertex_areas(template_white(fine_directions, "lh"), fine_faces)
    in_disc = np.degrees(np.arccos(np.clip(fine_directions @ half.sphere[centre], -1, 1))) <= 12.0
    on_ellipsoid = float(fine_area[in_disc].sum())
    assert cluster.area_mm2 == pytest.approx(AVG_AREA_FACTOR_CAP * on_ellipsoid, rel=0.05)
    assert abs(cluster.area_mm2 - on_ellipsoid) > 0.3 * on_ellipsoid


def test_peak_and_centroid_follow_z(tpl4: Template) -> None:
    half, adjacency = tpl4.hemis["lh"], tpl4.adjacency("lh")
    centre = nearest_vertex(half, direction("lh", from_pole_deg=20.0, azimuth_deg=30.0))
    disc = np.flatnonzero(angle_deg(half, centre) <= 12.0)
    edge = int(disc[np.argmax(angle_deg(half, centre)[disc])])
    # high at one edge of the disc, falling off steeply across it
    z = planted(half, disc, 0.0)
    z[disc] = (3.1 + 30.0 * np.exp(-angle_deg(half, edge)[disc] / 5.0)).astype(np.float32)
    (cluster,) = find_clusters(z, everywhere(half), half, adjacency, threshold=THRESHOLD)
    assert np.array_equal(cluster.vertices, disc)
    assert cluster.peak_vertex == edge and cluster.peak_z == np.float32(33.1)
    assert cluster.mass == pytest.approx(float(z[disc].sum(dtype=np.float64)), rel=1e-12)
    assert cluster.excess_mass == pytest.approx(cluster.mass - THRESHOLD * len(disc), rel=1e-12)
    assert cluster.centroid_vertex == expected_centroid(half, disc, z)
    # drawn from the centre of the disc towards the heavy edge
    assert cluster.centroid_vertex != centre
    assert angle_deg(half, edge)[cluster.centroid_vertex] < angle_deg(half, edge)[centre]


def test_excess_mass_goes_to_zero_at_the_threshold_and_mass_does_not(tpl4: Template) -> None:
    half, adjacency = tpl4.hemis["lh"], tpl4.adjacency("lh")
    centre = nearest_vertex(half, direction("lh", from_pole_deg=20.0, azimuth_deg=30.0))
    disc = angle_deg(half, centre) <= 12.0
    n = int(disc.sum())
    excess = []
    for above in (1.0, 0.1, 0.001, 1e-6):
        z = planted(half, disc, THRESHOLD + above)
        (cluster,) = find_clusters(z, everywhere(half), half, adjacency, threshold=THRESHOLD)
        stored = float(np.float32(THRESHOLD + above))  # what float32 keeps of the z
        assert cluster.excess_mass == pytest.approx(n * (stored - THRESHOLD), rel=1e-9)
        assert cluster.mass == pytest.approx(n * stored, rel=1e-12)
        assert cluster.mass > n * THRESHOLD
        excess.append(cluster.excess_mass)
    assert excess == sorted(excess, reverse=True)
    assert excess[0] == n and excess[-1] < 1e-4

    assert (
        find_clusters(
            planted(half, disc, THRESHOLD), everywhere(half), half, adjacency, threshold=THRESHOLD
        )
        == []
    )


def test_the_threshold_is_exclusive(tpl4: Template) -> None:
    half, adjacency = tpl4.hemis["lh"], tpl4.adjacency("lh")
    a, b, c = star(adjacency, 100, 3)
    z = np.zeros(len(half.sphere), dtype=np.float32)
    z[a] = THRESHOLD
    z[b] = np.nextafter(np.float32(THRESHOLD), np.float32(np.inf))
    z[c] = np.nextafter(np.float32(THRESHOLD), np.float32(-np.inf))
    (cluster,) = find_clusters(z, everywhere(half), half, adjacency, threshold=THRESHOLD)
    assert cluster.vertices.tolist() == [b]
    assert 0 < cluster.excess_mass < 1e-6
    # positive side only
    assert find_clusters(-z, everywhere(half), half, adjacency, threshold=THRESHOLD) == []
    low = find_clusters(z, everywhere(half), half, adjacency, threshold=2.5)
    assert as_sets(low) == [sorted([a, b, c])]


def test_min_vertices_drops_small_clusters(tpl4: Template) -> None:
    half, adjacency = tpl4.hemis["lh"], tpl4.adjacency("lh")
    groups = [star(adjacency, 200, 1), star(adjacency, 900, 2), star(adjacency, 1700, 5)]
    assert apart(adjacency, *groups)
    z = planted(half, [vertex for group in groups for vertex in group], 4.0)

    def sizes(min_vertices: int) -> list[int]:
        found = find_clusters(
            z, everywhere(half), half, adjacency, threshold=THRESHOLD, min_vertices=min_vertices
        )
        return sorted(len(cluster.vertices) for cluster in found)

    assert sizes(1) == [1, 2, 5]
    assert sizes(2) == [2, 5]
    assert sizes(5) == [5]
    assert sizes(6) == []
    default = find_clusters(z, everywhere(half), half, adjacency, threshold=THRESHOLD)
    assert len(default) == 3


def test_two_discs_joined_by_one_vertex_are_one_cluster(tpl4: Template) -> None:
    half, adjacency = tpl4.hemis["lh"], tpl4.adjacency("lh")
    first_centre = nearest_vertex(half, direction("lh", from_pole_deg=20.0, azimuth_deg=30.0))
    first = [first_centre, *neighbours(adjacency, first_centre).tolist()]
    # a vertex next to the first disc, and a second disc next to that vertex and not to
    # the first
    bridge = next(
        int(v) for member in first for v in neighbours(adjacency, member) if v not in first
    )
    second_centre = next(
        int(v)
        for near in neighbours(adjacency, bridge)
        for v in neighbours(adjacency, near)
        if v != bridge
        and bridge not in neighbours(adjacency, v)
        and apart(adjacency, first, [int(v), *neighbours(adjacency, int(v)).tolist()])
    )
    second = [second_centre, *neighbours(adjacency, second_centre).tolist()]
    assert bridge not in first and bridge not in second and len(first) == len(second) == 7

    separate = planted(half, first + second, 4.0)
    found = find_clusters(separate, everywhere(half), half, adjacency, threshold=THRESHOLD)
    assert as_sets(found) == sorted([sorted(first), sorted(second)])

    joined = separate.copy()
    joined[bridge] = 3.5
    (one,) = find_clusters(joined, everywhere(half), half, adjacency, threshold=THRESHOLD)
    assert one.vertices.tolist() == sorted([*first, *second, bridge])
    assert one.mass == 4.0 * 14 + 3.5
    assert one.area_mm2 == pytest.approx(sum(c.area_mm2 for c in found) + half.area[bridge])

    # the bridge outside the mask: two clusters again, clipped at the mask
    mask = everywhere(half)
    mask[bridge] = False
    clipped = find_clusters(joined, mask, half, adjacency, threshold=THRESHOLD)
    assert as_sets(clipped) == as_sets(found)


def test_a_disc_across_medial_and_inferior_is_clipped_in_each(tpl5: Template) -> None:
    half, adjacency = tpl5.hemis["lh"], tpl5.adjacency("lh")
    centre = nearest_vertex(half, direction("lh", from_pole_deg=22.0, azimuth_deg=-45.0))
    disc = angle_deg(half, centre) <= 10.0
    regions = half.regions
    in_medial, in_inferior = disc & regions["medial"], disc & regions["inferior"]
    assert in_medial.sum() > 20 and in_inferior.sum() > 20
    assert not (disc & regions["lateral"]).any() and regions["search"][disc].all()
    rng = np.random.default_rng(2)
    z = np.zeros(len(half.sphere), dtype=np.float32)
    z[disc] = rng.uniform(3.5, 9.0, size=int(disc.sum())).astype(np.float32)

    def found(region: str) -> list[Cluster]:
        return find_clusters(z, regions[region], half, adjacency, threshold=THRESHOLD)

    (whole,) = found("search")
    (medial,) = found("medial")
    (inferior,) = found("inferior")
    assert found("lateral") == [] and found("control") == []
    assert np.array_equal(whole.vertices, np.flatnonzero(disc))
    assert np.array_equal(medial.vertices, np.flatnonzero(in_medial))
    assert np.array_equal(inferior.vertices, np.flatnonzero(in_inferior))
    assert whole.mass == pytest.approx(medial.mass + inferior.mass, rel=1e-12)
    assert whole.excess_mass == pytest.approx(medial.excess_mass + inferior.excess_mass)
    assert whole.area_mm2 == pytest.approx(medial.area_mm2 + inferior.area_mm2, rel=1e-12)
    assert whole.peak_z == max(medial.peak_z, inferior.peak_z)
    assert medial.centroid_vertex in medial.vertices
    assert inferior.centroid_vertex in inferior.vertices


def test_vertices_without_a_z_are_no_members(tpl5: Template) -> None:
    half, adjacency = tpl5.hemis["lh"], tpl5.adjacency("lh")
    centre = nearest_vertex(half, direction("lh", from_pole_deg=20.0, azimuth_deg=30.0))
    disc = angle_deg(half, centre) <= 12.0
    z = planted(half, disc, 5.0)
    # a band of missing values through the middle, wider than an edge is long
    side = half.sphere @ np.cross(half.sphere[centre], [0.0, 0.0, 1.0])
    band = disc & (np.abs(np.degrees(np.arcsin(side))) < 2.5)
    z[band] = np.nan
    z[np.flatnonzero(disc & ~band)[:2]] = (np.inf, -np.inf)
    found = find_clusters(z, everywhere(half), half, adjacency, threshold=THRESHOLD)
    assert len(found) == 2
    assert as_sets(found) == components(z, everywhere(half), adjacency, THRESHOLD)
    members = np.concatenate([cluster.vertices for cluster in found])
    assert np.isfinite(z[members]).all() and len(members) == int(disc.sum()) - int(band.sum()) - 2
    assert all(np.isfinite([c.mass, c.excess_mass, c.peak_z, c.area_mm2]).all() for c in found)

    nothing = np.full(len(half.sphere), np.nan, dtype=np.float32)
    assert find_clusters(nothing, everywhere(half), half, adjacency, threshold=THRESHOLD) == []
    assert find_clusters(z, ~everywhere(half), half, adjacency, threshold=THRESHOLD) == []


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("region", ["search", "control", "all"])
def test_clusters_of_a_random_map_are_those_of_a_walk_over_the_mesh(
    tpl4: Template, seed: int, region: str
) -> None:
    half, adjacency = tpl4.hemis["rh"], tpl4.adjacency("rh")
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=len(half.sphere))
    blobs = (adjacency @ noise + noise) / np.sqrt(adjacency.sum(axis=1).A1 + 1.0)
    z = (2.5 * blobs).astype(np.float32)
    z[rng.random(len(z)) < 0.05] = np.nan
    mask = everywhere(half) if region == "all" else half.regions[region]
    found = find_clusters(z, mask, half, adjacency, threshold=THRESHOLD)
    expected = components(z, mask, adjacency, THRESHOLD)
    assert len(expected) > 5 and max(map(len, expected)) > 3
    assert as_sets(found) == expected
    for cluster in found:
        vertices = cluster.vertices
        values = z[vertices].astype(np.float64)
        assert np.array_equal(vertices, np.sort(vertices)) and mask[vertices].all()
        assert cluster.mass == pytest.approx(values.sum(), rel=1e-12)
        assert cluster.excess_mass == pytest.approx((values - THRESHOLD).sum(), rel=1e-9)
        assert cluster.area_mm2 == pytest.approx(half.area[vertices].sum(dtype=np.float64))
        assert cluster.peak_z == values.max()
        assert cluster.peak_vertex == int(vertices[np.argmax(values)])
        assert cluster.centroid_vertex == expected_centroid(half, vertices, z)
    order = [(-c.area_mm2, -c.excess_mass, int(c.vertices[0])) for c in found]
    assert order == sorted(order)

    kept = find_clusters(z, mask, half, adjacency, threshold=THRESHOLD, min_vertices=3)
    assert as_sets(kept) == [members for members in expected if len(members) >= 3]


def test_clusters_are_ordered_by_area_then_excess_then_first_vertex(tpl4: Template) -> None:
    half, adjacency = tpl4.hemis["lh"], tpl4.adjacency("lh")
    flat = dataclasses.replace(half, area=np.ones(len(half.sphere), dtype=np.float32))
    mild, strong, tiny, twin = (
        star(adjacency, centre, size) for centre, size in ((2000, 5), (300, 5), (1200, 3), (800, 5))
    )
    assert apart(adjacency, mild, strong, tiny, twin)
    z = np.zeros(len(half.sphere), dtype=np.float32)
    z[mild], z[strong], z[tiny], z[twin] = 4.0, 6.0, 50.0, 4.0
    found = find_clusters(z, everywhere(half), flat, adjacency, threshold=THRESHOLD)
    # area first, although the smallest has by far the largest mass; then excess; then, for
    # two clusters equal in both, the one whose first vertex is lower
    assert [c.vertices.tolist() for c in found] == [
        sorted(strong),
        sorted(twin),
        sorted(mild),
        sorted(tiny),
    ]
    assert [c.area_mm2 for c in found] == [5.0, 5.0, 5.0, 3.0]
    assert [c.excess_mass for c in found] == [15.0, 5.0, 5.0, 141.0]
    assert min(twin) < min(mild)

    # with the template's own areas, six small vertices can weigh less than five large ones
    small_vertex, large_vertex = int(np.argmin(half.area)), int(np.argmax(half.area))
    many, few = star(adjacency, small_vertex, 6), star(adjacency, large_vertex, 5)
    assert apart(adjacency, many, few)
    assert half.area[few].sum() > half.area[many].sum()
    z = planted(half, many + few, 4.0)
    first, second = find_clusters(z, everywhere(half), half, adjacency, threshold=THRESHOLD)
    assert first.vertices.tolist() == sorted(few) and second.vertices.tolist() == sorted(many)
    assert first.excess_mass < second.excess_mass


def test_ties_of_peak_and_centroid_go_to_the_lowest_index(tpl4: Template) -> None:
    half, adjacency = tpl4.hemis["lh"], tpl4.adjacency("lh")
    # two neighbours that mirror each other in the plane x = 0: with equal z their mean lies
    # in the plane and both are equally near it, to the last bit
    graph = adjacency.tocoo()
    mirror = np.array([-1.0, 1.0, 1.0])
    pairs = [
        (int(a), int(b))
        for a, b in zip(graph.row, graph.col, strict=True)
        if a < b and np.array_equal(half.sphere[a] * mirror, half.sphere[b])
    ]
    assert len(pairs) >= 10
    for low, high in pairs[:10]:
        z = planted(half, [low, high], 4.0)
        (cluster,) = find_clusters(z, everywhere(half), half, adjacency, threshold=THRESHOLD)
        assert (cluster.peak_vertex, cluster.centroid_vertex) == (low, low)
        # the heavier one takes both, whichever its index
        z[high] = 4.5
        (cluster,) = find_clusters(z, everywhere(half), half, adjacency, threshold=THRESHOLD)
        assert (cluster.peak_vertex, cluster.centroid_vertex) == (high, high)

    # the peak among several equal maxima
    members = star(adjacency, 1500, 7)
    z = planted(half, members, 4.0)
    z[sorted(members)[2:5]] = 8.0
    (cluster,) = find_clusters(z, everywhere(half), half, adjacency, threshold=THRESHOLD)
    assert cluster.peak_vertex == sorted(members)[2] and cluster.peak_z == 8.0


def test_a_map_or_a_mesh_of_another_size_is_refused(tpl4: Template, tpl5: Template) -> None:
    half = tpl4.hemis["lh"]
    z = np.zeros(len(half.sphere), dtype=np.float32)
    with pytest.raises(ValueError, match="adjacency of shape"):
        find_clusters(z, everywhere(half), half, tpl5.adjacency("lh"), threshold=THRESHOLD)
    other = tpl5.hemis["lh"]
    with pytest.raises(ValueError, match="2562 vertices on a template of 10242"):
        find_clusters(z, everywhere(half), other, tpl5.adjacency("lh"), threshold=THRESHOLD)


def test_clusters_fit_the_cluster_table(tpl4: Template, tmp_path: Path) -> None:
    half, adjacency = tpl4.hemis["lh"], tpl4.adjacency("lh")
    groups = [star(adjacency, 200, 1), star(adjacency, 900, 2), star(adjacency, 1700, 5)]
    z = planted(half, [vertex for group in groups for vertex in group], 4.25)
    found = find_clusters(z, everywhere(half), half, adjacency, threshold=THRESHOLD)
    rows = pd.DataFrame(
        {
            "subject_uid": "0123456789abcdef",
            "reference_id": "set-0123456789ab",
            "hemi": "lh",
            "region": "search",
            "rank": range(1, len(found) + 1),
            "n_vertices": [len(c.vertices) for c in found],
            "area_mm2": [c.area_mm2 for c in found],
            "mass": [c.mass for c in found],
            "excess_mass": [c.excess_mass for c in found],
            "peak_z": [c.peak_z for c in found],
            "peak_vertex": [c.peak_vertex for c in found],
            "centroid_vertex": [c.centroid_vertex for c in found],
        }
    )
    write_table(
        tmp_path / "clusters.parquet", rows, schemas.CLUSTERS, sort_by=schemas.KEYS["clusters"]
    )
    back = read_table(tmp_path / "clusters.parquet", schemas.CLUSTERS)
    assert back["n_vertices"].tolist() == [5, 2, 1]
    assert back["excess_mass"].tolist() == [6.25, 2.5, 1.25]


# --- native_centroids ---


def cluster_rows(centroids: Sequence[tuple[str, str, int, int]]) -> pd.DataFrame:
    """Rows of the cluster table, as far as they matter here: (hemi, region, rank, centroid)."""
    return pd.DataFrame(
        {
            "subject_uid": pd.Series(["0123456789abcdef"] * len(centroids), dtype=object),
            "reference_id": pd.Series(["set-0123456789ab"] * len(centroids), dtype=object),
            "hemi": pd.Series([row[0] for row in centroids], dtype=object),
            "region": pd.Series([row[1] for row in centroids], dtype=object),
            "rank": pd.Series([row[2] for row in centroids], dtype="int16"),
            "n_vertices": pd.Series([1] * len(centroids), dtype="int32"),
            "centroid_vertex": pd.Series([row[3] for row in centroids], dtype="int32"),
        }
    )


def empty_vertex_map() -> dict[str, np.ndarray]:
    vmap: dict[str, np.ndarray] = {}
    for hemi in HEMIS:
        vmap[f"{hemi}_index"] = np.empty(0, dtype="<i4")
        vmap[f"{hemi}_pial"] = np.empty((0, 3), dtype="<f4")
        vmap[f"{hemi}_white"] = np.empty((0, 3), dtype="<f4")
        vmap[f"{hemi}_fsavg6"] = np.empty(0, dtype="<i4")
        vmap[f"{hemi}_visible"] = np.empty(0, dtype=bool)
    return vmap


def hand_map(
    index: Sequence[int], counterpart: Sequence[int], pial: Sequence[Sequence[float]]
) -> dict[str, np.ndarray]:
    """A vertex map whose left hemisphere holds the given entries."""
    vmap = empty_vertex_map()
    vmap["lh_index"] = np.asarray(index, dtype="<i4")
    vmap["lh_fsavg6"] = np.asarray(counterpart, dtype="<i4")
    vmap["lh_pial"] = np.asarray(pial, dtype="<f4").reshape(-1, 3)
    vmap["lh_white"] = vmap["lh_pial"] - np.float32(2.0)
    vmap["lh_visible"] = np.ones(len(vmap["lh_index"]), dtype=bool)
    return vmap


def path_mesh(n: int) -> dict[str, sparse.csr_matrix]:
    """A mesh of ``n`` vertices in a row, ``0 - 1 - ... - n-1``, for both hemispheres."""
    ends = np.arange(n - 1)
    line = sparse.coo_matrix((np.ones(n - 1), (ends, ends + 1)), shape=(n, n))
    adjacency = (line + line.T).tocsr()
    return {"lh": adjacency, "rh": adjacency}


def native_pial(directions: np.ndarray) -> np.ndarray:
    return PIAL_CENTRE + directions * np.array(PIAL_SEMI_AXES)


@pytest.fixture(scope="module")
def native(tpl5: Template) -> dict[str, np.ndarray]:
    """The vertex map of a subject of 40,962 vertices per surface (about 1 mm apart on the
    pial ellipsoid), registered to the template through the warp: the posterior part of the
    left hemisphere, as a render keeps the part its tiles show. The counterpart of a native
    vertex is the template vertex nearest its registered direction, found here with a tree."""
    directions, _ = icosphere(6)
    pial = native_pial(directions)
    counterpart = cKDTree(tpl5.hemis["lh"].sphere).query(warp_directions(directions, WARP))[1]
    shown = np.flatnonzero(pial[:, 1] < -20.0)
    vmap = hand_map(shown, counterpart[shown], pial[shown])
    assert set(vmap) == set(vertexmap.KEYS)
    return vmap


def test_a_centroid_lands_on_the_native_pial_surface_at_the_patch(
    tpl5: Template, native: dict[str, np.ndarray]
) -> None:
    half, adjacency = tpl5.hemis["lh"], tpl5.adjacency("lh")
    centre = nearest_vertex(half, direction("lh", from_pole_deg=20.0, azimuth_deg=30.0))
    z = planted(half, angle_deg(half, centre) <= 12.0, 5.0)
    (cluster,) = find_clusters(z, half.regions["search"], half, adjacency, threshold=THRESHOLD)
    assert cluster.centroid_vertex == centre

    rows = cluster_rows([("lh", "search", 1, cluster.centroid_vertex)])
    out = native_centroids(rows, native, {hemi: tpl5.adjacency(hemi) for hemi in HEMIS})
    assert tuple(out.columns) == NATIVE_COLUMNS and len(out) == 1
    assert {name: str(kind) for name, kind in out.dtypes.items()} == {
        "subject_uid": "object",
        "reference_id": "object",
        "hemi": "object",
        "region": "object",
        "rank": "int16",
        "native_vertex": "int32",
        "tkr_x": "float32",
        "tkr_y": "float32",
        "tkr_z": "float32",
    }
    row = out.iloc[0]
    assert (row.subject_uid, row.reference_id, row.hemi, row.region, row["rank"]) == (
        "0123456789abcdef",
        "set-0123456789ab",
        "lh",
        "search",
        1,
    )
    # where the centre of the patch lies on the subject's pial surface, in closed form
    at_patch = native_pial(unwarp_directions(half.sphere[centre], WARP))
    position = out[["tkr_x", "tkr_y", "tkr_z"]].to_numpy(dtype=np.float64)[0]
    assert np.linalg.norm(position - at_patch) < 1.0  # 0.11 mm measured
    # it is a vertex of the map: its counterpart is the centroid, its position the stored one
    entry = int(np.flatnonzero(native["lh_index"] == row.native_vertex)[0])
    assert native["lh_fsavg6"][entry] == centre
    assert np.array_equal(native["lh_pial"][entry], position.astype(np.float32))


def test_the_native_vertex_is_the_candidate_nearest_the_candidates_mean(
    tpl5: Template, native: dict[str, np.ndarray]
) -> None:
    adjacency = {hemi: tpl5.adjacency(hemi) for hemi in HEMIS}
    rng = np.random.default_rng(4)
    present = np.unique(native["lh_fsavg6"])
    centroids = rng.choice(present, size=40, replace=False)
    rows = cluster_rows([("lh", "search", rank, int(v)) for rank, v in enumerate(centroids, 1)])
    out = native_centroids(rows, native, adjacency)
    assert out["rank"].tolist() == list(range(1, 41))
    several = 0
    for centroid, vertex in zip(centroids, out["native_vertex"], strict=True):
        candidates = np.flatnonzero(native["lh_fsavg6"] == centroid)
        several += len(candidates) > 2
        pial = native["lh_pial"][candidates].astype(np.float64)
        distance = np.linalg.norm(pial - pial.mean(axis=0), axis=1)
        assert vertex == native["lh_index"][candidates[np.argmin(distance)]]
    assert several > 30


def test_a_centroid_without_a_counterpart_takes_a_neighbours(
    tpl5: Template, native: dict[str, np.ndarray]
) -> None:
    half, mesh = tpl5.hemis["lh"], tpl5.adjacency("lh")
    adjacency = {hemi: tpl5.adjacency(hemi) for hemi in HEMIS}
    centre = nearest_vertex(half, direction("lh", from_pole_deg=20.0, azimuth_deg=30.0))
    own = native["lh_fsavg6"] == centre
    assert own.sum() >= 3
    # as if no native vertex had this template vertex as its nearest
    without = {
        key: (value[~own] if key.startswith("lh_") else value) for key, value in native.items()
    }
    rows = cluster_rows([("lh", "search", 1, centre)])
    out = native_centroids(rows, without, adjacency)
    assert len(out) == 1

    around = np.flatnonzero(np.isin(without["lh_fsavg6"], neighbours(mesh, centre)))
    pial = without["lh_pial"][around].astype(np.float64)
    chosen = around[np.argmin(np.linalg.norm(pial - pial.mean(axis=0), axis=1))]
    assert out["native_vertex"][0] == without["lh_index"][chosen]
    assert without["lh_fsavg6"][chosen] in neighbours(mesh, centre)
    # still at the patch: within a template vertex spacing, which is about 2 mm here
    at_patch = native_pial(unwarp_directions(half.sphere[centre], WARP))
    position = out[["tkr_x", "tkr_y", "tkr_z"]].to_numpy(dtype=np.float64)[0]
    assert np.linalg.norm(position - at_patch) < 2.0  # 0.62 mm measured
    # with its own counterparts in the map the neighbours are not asked
    with_own = native_centroids(rows, native, adjacency)
    assert native["lh_fsavg6"][native["lh_index"] == with_own["native_vertex"][0]] == centre


def test_a_centroid_outside_the_map_gets_no_row(
    tpl5: Template, native: dict[str, np.ndarray]
) -> None:
    half = tpl5.hemis["lh"]
    adjacency = {hemi: tpl5.adjacency(hemi) for hemi in HEMIS}
    centre = nearest_vertex(half, direction("lh", from_pole_deg=20.0, azimuth_deg=30.0))
    frontal = int(np.flatnonzero(half.regions["control"])[0])
    assert half.sphere[frontal, 1] > 0.6
    rows = cluster_rows(
        [
            ("lh", "control", 1, frontal),  # the tiles do not show the frontal lobe
            ("lh", "search", 1, centre),
            ("rh", "search", 1, centre),  # the map has nothing of this hemisphere
            ("lh", "medial", 2, centre),
        ]
    )
    out = native_centroids(rows, native, adjacency)
    assert list(zip(out["hemi"], out["region"], out["rank"], strict=True)) == [
        ("lh", "search", 1),
        ("lh", "medial", 2),
    ]
    assert out["native_vertex"][0] == out["native_vertex"][1]

    # a render made without a template has no counterparts at all
    unmapped = dict(native)
    unmapped["lh_fsavg6"] = np.full_like(native["lh_fsavg6"], -1)
    nothing = native_centroids(rows, unmapped, adjacency)
    assert len(nothing) == 0 and tuple(nothing.columns) == NATIVE_COLUMNS
    assert (
        str(nothing["native_vertex"].dtype) == "int32" and str(nothing["tkr_x"].dtype) == "float32"
    )
    none = native_centroids(rows.iloc[:0], native, adjacency)
    assert len(none) == 0 and tuple(none.columns) == NATIVE_COLUMNS


def test_the_rule_on_a_map_written_by_hand() -> None:
    adjacency = path_mesh(6)
    vmap = hand_map(
        index=[3, 10, 11, 12, 20, 30, 31],
        counterpart=[-1, 0, 0, 0, 2, 4, 4],
        pial=[
            [9.0, 9.0, 9.0],
            [0.0, 0.0, 0.0],  # the three of template vertex 0: mean (2, 0, 0)
            [1.5, 0.0, 0.0],
            [4.5, 0.0, 0.0],
            [7.0, 1.0, 0.0],
            [0.0, 5.0, 0.0],  # the two of template vertex 4: equally far from their mean
            [0.0, 7.0, 0.0],
        ],
    )
    rows = cluster_rows(
        [
            ("lh", "search", 1, 0),  # its own three counterparts: the one nearest (2, 0, 0)
            ("lh", "search", 2, 4),  # a tie: the lower native index
            ("lh", "search", 3, 2),  # one counterpart
            ("lh", "search", 4, 1),  # none: those of its neighbours 0 and 2, mean (3.25, 0.25, 0)
            ("lh", "search", 5, 3),  # none: neighbours 2 and 4, mean (7/3, 13/3, 0)
            ("lh", "search", 6, 5),  # none, and its neighbour 4 has two
        ]
    )
    out = native_centroids(rows, vmap, adjacency)
    assert out["rank"].tolist() == [1, 2, 3, 4, 5, 6]
    assert out["native_vertex"].tolist() == [11, 30, 20, 12, 30, 30]
    assert out[["tkr_x", "tkr_y", "tkr_z"]].to_numpy().tolist() == [
        [1.5, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [7.0, 1.0, 0.0],
        [4.5, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 5.0, 0.0],
    ]
    # the pial position, not the white one
    assert not np.array_equal(vmap["lh_pial"], vmap["lh_white"])

    # a neighbour's counterparts are asked only when the centroid has none of its own
    lonely = hand_map(index=[5, 6], counterpart=[1, 3], pial=[[1.0, 0, 0], [3.0, 0, 0]])
    far = native_centroids(cluster_rows([("lh", "search", 1, 5)]), lonely, adjacency)
    assert len(far) == 0  # vertex 5 has the neighbour 4 only, and nothing maps to 4


def test_native_rows_fit_the_table(
    tpl5: Template, native: dict[str, np.ndarray], tmp_path: Path
) -> None:
    half = tpl5.hemis["lh"]
    adjacency = {hemi: tpl5.adjacency(hemi) for hemi in HEMIS}
    centre = nearest_vertex(half, direction("lh", from_pole_deg=20.0, azimuth_deg=30.0))
    frontal = int(np.flatnonzero(half.regions["control"])[0])
    # the rows as the pipeline has them: read from the cluster table
    clusters = cluster_rows(
        [("lh", "search", 1, centre), ("lh", "medial", 1, centre), ("lh", "control", 1, frontal)]
    ).assign(area_mm2=10.0, mass=8.0, excess_mass=2.0, peak_z=4.0, peak_vertex=centre)
    write_table(
        tmp_path / "clusters.parquet",
        clusters[schemas.CLUSTERS.names],
        schemas.CLUSTERS,
        sort_by=schemas.KEYS["clusters"],
    )
    rows = read_table(tmp_path / "clusters.parquet", schemas.CLUSTERS)
    assert rows["region"].tolist() == ["control", "medial", "search"]

    out = native_centroids(rows, native, adjacency)
    assert tuple(out.columns) == NATIVE_COLUMNS and out["region"].tolist() == ["medial", "search"]
    assert str(out["rank"].dtype) == "int16" and str(out["hemi"].dtype) == "object"
    path = tmp_path / "clusters_native.parquet"
    write_table(
        path,
        out.assign(template_id=tpl5.template_id),
        schemas.CLUSTERS_NATIVE,
        sort_by=schemas.KEYS["clusters_native"],
    )
    back = read_table(path, schemas.CLUSTERS_NATIVE)
    assert back["region"].tolist() == ["medial", "search"]
    assert back["native_vertex"].tolist() == out["native_vertex"].tolist()
    assert back["tkr_y"].tolist() == out["tkr_y"].tolist()


def test_rows_that_do_not_fit_the_mesh_are_refused(
    tpl5: Template, native: dict[str, np.ndarray]
) -> None:
    adjacency = {hemi: tpl5.adjacency(hemi) for hemi in HEMIS}
    rows = cluster_rows([("lh", "search", 1, 5)])
    with pytest.raises(ValueError, match="lack the columns rank, centroid_vertex"):
        native_centroids(rows.drop(columns=["rank", "centroid_vertex"]), native, adjacency)
    beyond = cluster_rows([("lh", "search", 1, 10_242)])
    with pytest.raises(ValueError, match="centroid vertex 10242 is not on the mesh of 10242"):
        native_centroids(beyond, native, adjacency)
    with pytest.raises(ValueError, match="no mesh or no vertex map"):
        native_centroids(rows, native, {"rh": adjacency["rh"]})
    ragged = dict(native)
    ragged["lh_fsavg6"] = native["lh_fsavg6"][:-1]
    with pytest.raises(ValueError, match="differ in length"):
        native_centroids(rows, ragged, adjacency)
