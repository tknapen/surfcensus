"""fsio: reading surfaces and volumes, normals, intensity window; and the synthetic subject."""

from __future__ import annotations

import warnings
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer as fs
import numpy as np
import pytest
from scipy import ndimage
from synth import (  # pyrefly: ignore[missing-import]
    T1_GRAY,
    T1_HEAD,
    T1_WHITE,
    SyntheticSubject,
    conformed_vox2ras_tkr,
    icosphere,
    snapshot_tree,
    write_synthetic_subject,
)

from surfcensus.fsio.surfaces import Surface, read_surface, vertex_normals
from surfcensus.fsio.volumes import Volume, intensity_window, read_volume

HEMIS = ("lh", "rh")


def _centre(subject: SyntheticSubject, hemi: str) -> np.ndarray:
    return np.array(subject.lh_centre if hemi == "lh" else subject.rh_centre)


def _volume(data: np.ndarray) -> Volume:
    return Volume(data=data, vox2ras_tkr=np.eye(4), vox2ras=np.eye(4), voxel_size=(1.0, 1.0, 1.0))


# --- synthetic mesh -----------------------------------------------------------------------


@pytest.mark.parametrize("subdivisions", [0, 1, 2, 3])
def test_icosphere_is_a_closed_outward_unit_sphere(subdivisions: int) -> None:
    v, f = icosphere(subdivisions)
    assert v.shape == (10 * 4**subdivisions + 2, 3) and v.dtype == np.float64
    assert f.shape == (20 * 4**subdivisions, 3) and np.issubdtype(f.dtype, np.integer)
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-12)
    assert set(np.unique(f)) == set(range(len(v)))
    # closed manifold with consistent winding: every directed edge occurs once, with its reverse
    directed = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    assert len(np.unique(directed, axis=0)) == len(directed)
    assert {tuple(e) for e in directed.tolist()} == {tuple(e[::-1]) for e in directed.tolist()}
    normals = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    assert (np.einsum("ij,ij->i", normals, v[f].mean(axis=1)) > 0).all()


def test_icosphere_has_the_axis_points_and_is_deterministic() -> None:
    v, f = icosphere(2)
    for axis_point in np.concatenate([np.eye(3), -np.eye(3)]):
        assert (v == axis_point).all(axis=1).sum() == 1
    v2, f2 = icosphere(2)
    assert np.array_equal(v, v2) and np.array_equal(f, f2)


def test_synthetic_subject_record(synth_subject: SyntheticSubject, tmp_path: Path) -> None:
    s = synth_subject
    assert s.path.name == "sub-synth" and s.path.is_dir()
    assert (s.voxel_size, s.shape, s.c_ras, s.midline_x) == (1.0, (160,) * 3, (3.0, -7.0, 11.0), -4)
    assert s.lh_centre == (-34.0, -10.0, 5.0) and s.rh_centre == (26.0, -12.0, 5.0)
    assert s.white_semi_axes == (26.0, 58.0, 42.0) and s.pial_semi_axes == (28.5, 60.5, 44.5)
    assert s.n_vertices == 10_242
    files = sorted(p.relative_to(s.path).as_posix() for p in s.path.rglob("*") if p.is_file())
    assert files == [
        "mri/T1.mgz",
        "mri/brainmask.mgz",
        "surf/lh.pial",
        "surf/lh.white",
        "surf/rh.pial",
        "surf/rh.white",
    ]
    bare = write_synthetic_subject(
        tmp_path, name="sub-bare", shape=(8, 8, 8), subdivisions=1, with_brainmask=False
    )
    assert bare.path == tmp_path / "sub-bare" and bare.n_vertices == 42
    assert not (bare.path / "mri" / "brainmask.mgz").exists()


# --- surfaces -----------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["synth_subject", "synth_subject_pial_t1"])
def test_read_surface_returns_the_ellipsoids(fixture: str, request: pytest.FixtureRequest) -> None:
    s: SyntheticSubject = request.getfixturevalue(fixture)
    pial_name = "pial.T1" if fixture == "synth_subject_pial_t1" else "pial"
    for hemi in HEMIS:
        white = read_surface(s.path / "surf" / f"{hemi}.white")
        pial = read_surface(str(s.path / "surf" / f"{hemi}.{pial_name}"))  # str paths work too
        for surface, semi in ((white, s.white_semi_axes), (pial, s.pial_semi_axes)):
            assert surface.vertices.shape == (s.n_vertices, 3)
            assert surface.vertices.dtype == np.float64
            assert surface.faces.shape == (2 * s.n_vertices - 4, 3)
            assert np.issubdtype(surface.faces.dtype, np.integer)
            assert surface.faces.min() == 0 and surface.faces.max() == s.n_vertices - 1
            # coordinates are stored as float32, hence the tolerance
            rel = (surface.vertices - _centre(s, hemi)) / np.array(semi)
            np.testing.assert_allclose((rel**2).sum(axis=1), 1.0, atol=1e-6)
        assert np.array_equal(white.faces, pial.faces)


def test_read_surface_matches_what_was_written(tmp_path: Path) -> None:
    s = write_synthetic_subject(tmp_path, shape=(8, 8, 8), subdivisions=2)
    unit, faces = icosphere(2)
    lh = read_surface(s.path / "surf" / "lh.pial")
    expected = (np.array(s.lh_centre) + unit * np.array(s.pial_semi_axes)).astype(np.float32)
    assert np.array_equal(lh.vertices, expected.astype(np.float64))
    assert np.array_equal(lh.faces, faces)


def test_surface_landmarks(synth_subject: SyntheticSubject) -> None:
    s = synth_subject
    lh = read_surface(s.path / "surf" / "lh.pial").vertices
    rh = read_surface(s.path / "surf" / "rh.pial").vertices
    # occipital poles: one most-posterior vertex each, rh 2 mm further back
    for xyz, centre in ((lh, s.lh_centre), (rh, s.rh_centre)):
        pole = xyz[np.argmin(xyz[:, 1])]
        assert pole.tolist() == [centre[0], centre[1] - s.pial_semi_axes[1], centre[2]]
        assert (xyz[:, 1] == pole[1]).sum() == 1
    assert lh[:, 1].min() - rh[:, 1].min() == 2.0
    # medial faces: 3 mm apart, straddling the midline
    assert lh[:, 0].max() == s.midline_x - 1.5 and rh[:, 0].min() == s.midline_x + 1.5


def test_read_surface_does_not_apply_c_ras(synth_subject: SyntheticSubject) -> None:
    path = synth_subject.path / "surf" / "lh.white"
    _, _, footer = fs.read_geometry(str(path), read_metadata=True)
    assert footer["cras"].tolist() == list(synth_subject.c_ras)  # the footer is there
    centroid = read_surface(path).vertices.mean(axis=0)
    np.testing.assert_allclose(centroid, synth_subject.lh_centre, atol=1e-5)


@pytest.mark.parametrize(
    ("fixture", "max_degrees"), [("synth_subject", 1.0), ("synth_subject_pial_t1", 2.0)]
)
def test_vertex_normals_point_outward_and_match_the_ellipsoid(
    fixture: str, max_degrees: float, request: pytest.FixtureRequest
) -> None:
    s: SyntheticSubject = request.getfixturevalue(fixture)
    pial_name = "pial.T1" if fixture == "synth_subject_pial_t1" else "pial"
    for hemi in HEMIS:
        for name, semi in (("white", s.white_semi_axes), (pial_name, s.pial_semi_axes)):
            surface = read_surface(s.path / "surf" / f"{hemi}.{name}")
            normals = vertex_normals(surface)
            assert normals.shape == surface.vertices.shape and normals.dtype == np.float64
            np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-12)
            rel = surface.vertices - _centre(s, hemi)
            assert (np.einsum("ij,ij->i", normals, rel) > 0).all()
            # gradient of (x/a)^2 + (y/b)^2 + (z/c)^2
            analytic = rel / np.array(semi) ** 2
            analytic /= np.linalg.norm(analytic, axis=1, keepdims=True)
            cosine = np.clip(np.einsum("ij,ij->i", normals, analytic), -1.0, 1.0)
            assert np.degrees(np.arccos(cosine)).max() < max_degrees


def test_vertex_normals_follow_the_winding_and_weight_by_area() -> None:
    unit, faces = icosphere(2)
    outward = vertex_normals(Surface(vertices=unit, faces=faces))
    inward = vertex_normals(Surface(vertices=unit, faces=faces[:, ::-1].copy()))
    np.testing.assert_allclose(inward, -outward, atol=1e-12)
    # vertex 0 is shared by a small face in the z = 0 plane and a face in the x = 0 plane that
    # is 4 times larger, so the normal is (4, 0, 1) / sqrt(17), not the unweighted (1, 0, 1)
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 2, 0], [0, 0, 2]])
    tilted = Surface(vertices=vertices, faces=np.array([[0, 1, 2], [0, 3, 4]]))
    np.testing.assert_allclose(vertex_normals(tilted)[0], np.array([4, 0, 1]) / np.sqrt(17))


def test_vertex_normals_are_zero_not_nan_where_there_is_no_area() -> None:
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [3, 0, 0], [9, 9, 9]])
    faces = np.array([[0, 1, 2], [1, 3, 4], [3, 3, 3]])  # a triangle, a collinear face, a point
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        normals = vertex_normals(Surface(vertices=vertices, faces=faces))
    assert np.isfinite(normals).all()
    assert normals[:3].tolist() == [[0, 0, 1]] * 3  # vertex 1 ignores its degenerate face
    assert normals[3:].tolist() == [[0, 0, 0]] * 3  # degenerate faces only, or unreferenced
    empty = Surface(vertices=vertices, faces=np.zeros((0, 3), dtype=int))
    assert not vertex_normals(empty).any()


def test_runtime_shape_checking_is_installed() -> None:
    # the import hook in conftest.py checks array shapes and dtypes of the whole package
    with pytest.raises(Exception, match=r"(?i)type-check"):
        Surface(vertices=np.zeros((4, 2)), faces=np.zeros((1, 3), dtype=int))
    with pytest.raises(Exception, match=r"(?i)type-check"):
        vertex_normals(np.zeros((4, 3)))  # type: ignore[arg-type]


# --- volumes ------------------------------------------------------------------------------


def test_read_volume_affines_1mm(synth_subject: SyntheticSubject) -> None:
    s = synth_subject
    t1 = read_volume(s.path / "mri" / "T1.mgz")
    assert t1.shape == t1.data.shape == (160, 160, 160)
    assert t1.data.dtype == np.uint8
    assert t1.voxel_size == (1.0, 1.0, 1.0) and all(type(z) is float for z in t1.voxel_size)
    expected = [[-1, 0, 0, 80], [0, 0, 1, -80], [0, -1, 0, 80], [0, 0, 0, 1]]
    assert t1.vox2ras_tkr.dtype == np.float64 and t1.vox2ras.dtype == np.float64
    assert t1.vox2ras_tkr.tolist() == expected
    assert np.array_equal(t1.vox2ras_tkr, conformed_vox2ras_tkr(1.0, s.shape))
    # scanner RAS is tkr RAS shifted by c_ras, so the two affines differ
    assert np.array_equal(t1.vox2ras[:, :3], t1.vox2ras_tkr[:, :3])
    assert (t1.vox2ras[:3, 3] - t1.vox2ras_tkr[:3, 3]).tolist() == list(s.c_ras)
    # the same affine straight from nibabel
    header = nib.MGHImage.from_filename(s.path / "mri" / "T1.mgz").header
    assert np.array_equal(header.get_vox2ras_tkr(), np.array(expected))
    assert nib.aff2axcodes(t1.vox2ras) == ("L", "I", "A")


def test_read_volume_affines_hires(synth_subject_hires: SyntheticSubject) -> None:
    s = synth_subject_hires
    t1 = read_volume(s.path / "mri" / "T1.mgz")
    assert t1.shape == (200, 200, 200)
    d = 0.8
    exact = np.array([[-d, 0, 0, 80], [0, 0, d, -80], [0, -d, 0, 80], [0, 0, 0, 1]])
    # the header stores the voxel size as float32; the offsets are exact
    assert np.array_equal(t1.vox2ras_tkr, exact.astype(np.float32).astype(np.float64))
    assert np.array_equal(t1.vox2ras_tkr, conformed_vox2ras_tkr(0.8, s.shape))
    np.testing.assert_allclose(t1.vox2ras_tkr, exact, rtol=0, atol=2e-8)
    assert t1.voxel_size == (float(np.float32(0.8)),) * 3
    np.testing.assert_allclose(t1.vox2ras[:3, 3] - t1.vox2ras_tkr[:3, 3], s.c_ras, atol=1e-5)


@pytest.mark.parametrize("fixture", ["synth_subject", "synth_subject_hires"])
def test_t1_intensities_follow_the_ellipsoids(fixture: str, request: pytest.FixtureRequest) -> None:
    s: SyntheticSubject = request.getfixturevalue(fixture)
    t1 = read_volume(s.path / "mri" / "T1.mgz")
    brainmask = read_volume(s.path / "mri" / "brainmask.mgz")
    assert np.array_equal(brainmask.vox2ras_tkr, t1.vox2ras_tkr)
    assert np.array_equal(brainmask.vox2ras, t1.vox2ras)
    assert set(np.unique(t1.data).tolist()) == {0, T1_HEAD, T1_GRAY, T1_WHITE}
    assert np.array_equal(brainmask.data, np.where(t1.data >= T1_GRAY, t1.data, 0))

    def value_at(tkr_xyz: tuple[float, float, float]) -> int:
        ijk = np.linalg.solve(t1.vox2ras_tkr, np.array([*tkr_xyz, 1.0]))[:3]
        i, j, k = np.floor(ijk + 0.5).astype(int)  # nearest voxel centre
        return int(t1.data[i, j, k])

    lx, ly, lz = s.lh_centre
    rx, ry, rz = s.rh_centre
    assert value_at(s.lh_centre) == T1_WHITE and value_at(s.rh_centre) == T1_WHITE
    assert value_at((lx + 27.25, ly, lz)) == T1_GRAY  # medial cortex, between white and pial
    assert value_at((rx, ry - 59.25, rz)) == T1_GRAY  # cortex at the rh occipital pole
    assert value_at((s.midline_x, -11.0, 5.0)) == T1_HEAD  # between the hemispheres
    assert value_at((rx, ry - 64.0, rz)) == T1_HEAD  # behind the rh pole
    assert value_at((lx, ly, lz + 70.0)) == 0  # above the head
    assert t1.data[0, 0, 0] == 0
    # voxel counts match the analytic volumes in mm^3 whatever the voxel size
    voxel_mm3 = float(np.prod(t1.voxel_size))
    white_mm3 = 2 * 4 / 3 * np.pi * np.prod(s.white_semi_axes)
    pial_mm3 = 2 * 4 / 3 * np.pi * np.prod(s.pial_semi_axes)
    assert (t1.data == T1_WHITE).sum() * voxel_mm3 == pytest.approx(white_mm3, rel=2e-3)
    assert (t1.data >= T1_GRAY).sum() * voxel_mm3 == pytest.approx(pial_mm3, rel=2e-3)


def test_read_volume_dtypes_and_dimensions(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=(4, 5, 6)).astype(np.float32)
    nib.save(nib.MGHImage(values, np.eye(4)), tmp_path / "float.mgh")
    volume = read_volume(tmp_path / "float.mgh")
    assert volume.data.dtype == np.float32 and volume.data.dtype.isnative
    assert np.array_equal(volume.data, values)

    frames = np.stack([values, values], axis=-1)
    nib.save(nib.MGHImage(frames, np.eye(4)), tmp_path / "two_frames.mgz")
    with pytest.raises(ValueError, match="3D"):
        read_volume(tmp_path / "two_frames.mgz")

    nib.save(nib.Nifti1Image(values, np.eye(4)), tmp_path / "volume.nii.gz")
    with pytest.raises(ValueError, match="MGH"):
        read_volume(tmp_path / "volume.nii.gz")


# --- intensity window ---------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["synth_subject", "synth_subject_hires"])
def test_intensity_window_on_the_synthetic_subject(
    fixture: str, request: pytest.FixtureRequest
) -> None:
    s: SyntheticSubject = request.getfixturevalue(fixture)
    t1 = read_volume(s.path / "mri" / "T1.mgz")
    brainmask = read_volume(s.path / "mri" / "brainmask.mgz")
    # dilating the mask by 3 voxels takes in a shell of head tissue around the cortex
    window = intensity_window(t1, brainmask)
    assert window == (float(T1_HEAD), float(T1_WHITE))
    assert all(type(w) is float for w in window)
    assert intensity_window(t1, brainmask, dilation=0) == (float(T1_GRAY), float(T1_WHITE))
    assert intensity_window(t1, None) == (float(T1_HEAD), float(T1_WHITE))
    # the definition, spelled out with scipy
    mask = ndimage.binary_dilation(brainmask.data > 0, iterations=3)
    lo, hi = np.percentile(t1.data[mask], [1.0, 99.5])
    assert window == (lo, hi)


def test_intensity_window_dilates_the_mask_by_whole_voxels() -> None:
    ramp = np.arange(1, 21, dtype=np.uint8).reshape(20, 1, 1)  # value = index + 1
    seed = np.zeros((20, 1, 1), dtype=np.uint8)
    seed[10] = 1
    t1, mask = _volume(ramp), _volume(seed)
    assert intensity_window(t1, mask, percentiles=(0.0, 100.0), dilation=3) == (8.0, 14.0)
    assert intensity_window(t1, mask, percentiles=(0.0, 100.0), dilation=1) == (10.0, 12.0)
    assert intensity_window(t1, mask, percentiles=(0.0, 100.0), dilation=0) == (11.0, 12.0)
    assert intensity_window(t1, mask, percentiles=(50.0, 50.0)) == (11.0, 12.0)
    # dilation is 6-connected: a diagonal neighbour is reached in two steps, not one
    cube = np.arange(27, dtype=np.float32).reshape(3, 3, 3) + 1
    corner = np.zeros((3, 3, 3), dtype=np.uint8)
    corner[0, 0, 0] = 1
    lo, hi = intensity_window(_volume(cube), _volume(corner), (0.0, 100.0), dilation=1)
    assert (lo, hi) == (1.0, 10.0)  # (0,0,0) and its three face neighbours: 1, 2, 4, 10


def test_intensity_window_without_brainmask_uses_non_zero_voxels() -> None:
    data = np.zeros((10, 10, 10), dtype=np.uint8)
    data.flat[:100] = np.arange(1, 101)
    lo, hi = intensity_window(_volume(data), None)
    assert (lo, hi) == pytest.approx((1.99, 99.505))
    assert intensity_window(_volume(data), None, percentiles=(0.0, 100.0)) == (1.0, 100.0)


def test_intensity_window_fallbacks_and_guarantee() -> None:
    data = np.zeros((6, 6, 6), dtype=np.uint8)
    data[2:4, 2:4, 2:4] = np.arange(10, 18).reshape(2, 2, 2)
    empty_mask = _volume(np.zeros((6, 6, 6), dtype=np.uint8))
    # a mask that selects nothing falls back to the non-zero voxels
    assert intensity_window(_volume(data), empty_mask, (0.0, 100.0)) == (10.0, 17.0)
    # nothing at all to look at
    zeros = _volume(np.zeros((6, 6, 6), dtype=np.uint8))
    assert intensity_window(zeros, None) == (0.0, 1.0)
    assert intensity_window(zeros, empty_mask) == (0.0, 1.0)
    # a constant image still gets a window of positive width
    flat = _volume(np.full((6, 6, 6), 110, dtype=np.uint8))
    lo, hi = intensity_window(flat, None)
    assert lo == 110.0 and hi > lo
    # non-finite voxels of a float volume are ignored
    floats = data.astype(np.float32)
    floats[0, 0, 0], floats[5, 5, 5] = np.nan, np.inf
    assert intensity_window(_volume(floats), None, (0.0, 100.0)) == (10.0, 17.0)


def test_intensity_window_refuses_a_mask_on_another_grid() -> None:
    t1 = _volume(np.ones((4, 4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="shape"):
        intensity_window(t1, _volume(np.ones((4, 4, 5), dtype=np.uint8)))


# --- read-only ----------------------------------------------------------------------------


def test_reading_leaves_the_subject_untouched(synth_subject: SyntheticSubject) -> None:
    before = snapshot_tree(synth_subject.path)
    assert len(before) == 8  # two directories and six files
    t1 = read_volume(synth_subject.path / "mri" / "T1.mgz")
    brainmask = read_volume(synth_subject.path / "mri" / "brainmask.mgz")
    intensity_window(t1, brainmask)
    for hemi in HEMIS:
        for name in ("white", "pial"):
            vertex_normals(read_surface(synth_subject.path / "surf" / f"{hemi}.{name}"))
    assert snapshot_tree(synth_subject.path) == before
