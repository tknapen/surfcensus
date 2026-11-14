"""The display geometry of the EPI interface (``func.geometry``, work package E1.B), against
the closed forms of ``tests/synth_func.py`` and against a recomputation in the test with
nibabel, numpy and scipy only.

The raw stream of run ``r`` samples the raw series at ``inv(R_r) . S_a . G . v`` (D36): one
rigid placement per session, that of the anchor, and nothing else. A cubic B-spline with the
edge values repeated, and 0 wherever the sample lies outside the outermost raw voxel centres.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from scipy import ndimage
from synth_func import (  # pyrefly: ignore[missing-import]
    GRID_AFFINE,
    GRID_B_SHAPE,
    GRID_SHAPE,
    RAW_SHAPE,
    RAW_TILT_DEG,
    SPLINE_TOLERANCE,
    SyntheticFuncDataset,
    SyntheticRun,
    expected_pre,
    expected_raw_stream,
    raw_sample_voxels,
    spline_interior,
    voxel_centres,
    write_func_dataset,
)

from surfcensus.func import geometry
from surfcensus.records import IDENTITY4, Grid, to_matrix4


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> SyntheticFuncDataset:
    return write_func_dataset(tmp_path_factory.mktemp("func"))


def anchor_of(dataset: SyntheticFuncDataset, run: SyntheticRun) -> SyntheticRun:
    session = dataset.session(run.design.participant, run.design.session)
    return next(r for r in dataset.runs if r.stem == session.anchor)


def m4(matrix: np.ndarray) -> tuple:
    return to_matrix4(matrix)


def rigid_matrix(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    m = np.eye(4)
    m[:3, :3] = q * rng.uniform(0.5, 3.0, size=3)
    m[:3, 3] = rng.normal(scale=50.0, size=3)
    return m


# --- matrices -----------------------------------------------------------------------------


def test_compose_and_inverse_agree_with_numpy() -> None:
    a, b, c = (rigid_matrix(n) for n in range(3))
    got = np.array(geometry.compose(m4(a), m4(b), m4(c)))
    np.testing.assert_allclose(got, a @ b @ c, rtol=1e-13, atol=1e-11)
    np.testing.assert_allclose(np.array(geometry.inverse(m4(a))), np.linalg.inv(a), atol=1e-13)
    assert geometry.compose() == IDENTITY4
    assert geometry.compose(m4(a)) == m4(a)
    # left to right: c acts first on a point
    p = np.array([1.0, 2.0, 3.0, 1.0])
    np.testing.assert_allclose(np.array(geometry.compose(m4(a), m4(b))) @ p, a @ (b @ p))


def test_an_affine_that_cannot_be_inverted_is_refused() -> None:
    flat = np.eye(4)
    flat[2, 2] = 0.0
    with pytest.raises(ValueError, match="inverted"):
        geometry.inverse(m4(flat))


def test_the_portable_product_is_python_floats_in_a_fixed_order() -> None:
    """The portable map is the Python-float product, whatever BLAS would give; and it agrees
    with numpy's to the last few bits."""
    raw, anchor, display = (rigid_matrix(n) for n in (4, 5, 6))
    grid = Grid((10, 11, 12), m4(display))
    portable = geometry.voxel_map(m4(raw), m4(anchor), grid)
    by_hand = np.array(
        geometry.compose(geometry.inverse(m4(raw)), m4(anchor), m4(display)), dtype=np.float64
    )
    assert portable.tobytes() == by_hand.tobytes()
    numpy_map = geometry.voxel_map(m4(raw), m4(anchor), grid, portable=False)
    np.testing.assert_array_equal(numpy_map, np.linalg.inv(raw) @ anchor @ display)
    assert np.abs(portable - numpy_map).max() < 1e-9


# --- the grid -----------------------------------------------------------------------------


def test_the_grid_of_a_series(dataset: SyntheticFuncDataset) -> None:
    run = dataset.run("sub-01", "1", "rest", "2")
    grid = geometry.grid_of(run.files["pre_bold"])
    assert grid.shape == GRID_SHAPE
    np.testing.assert_array_equal(np.array(grid.affine), GRID_AFFINE)
    assert grid.key == "e2babfc70bbc"  # the key pinned in tests/test_func_interfaces.py
    assert geometry.grid_of(run.files["pre_mask"]) == grid
    other = next(r for r in dataset.runs if r.design.case == "other grid")
    other_grid = geometry.grid_of(other.files["pre_bold"])
    assert other_grid.shape == GRID_B_SHAPE
    assert not geometry.same_grid(grid, other_grid)
    nudged = np.array(GRID_AFFINE)
    nudged[0, 3] += 0.5 * geometry.GRID_TOLERANCE
    assert geometry.same_grid(grid, Grid(GRID_SHAPE, m4(nudged)))
    nudged[0, 3] += geometry.GRID_TOLERANCE
    assert not geometry.same_grid(grid, Grid(GRID_SHAPE, m4(nudged)))


def test_voxel_size_and_obliqueness(dataset: SyntheticFuncDataset) -> None:
    grid = Grid(GRID_SHAPE, m4(GRID_AFFINE))
    assert geometry.voxel_mm(grid) == (8.0, 8.0, 8.0)
    assert geometry.oblique_deg(grid) == 0.0
    raw = geometry.grid_of(dataset.run("sub-01", "1", "rest", "2").files["raw_bold"])
    assert geometry.voxel_mm(raw) == pytest.approx((8.0, 8.0, 8.0), abs=1e-5)
    assert geometry.oblique_deg(raw) == pytest.approx(RAW_TILT_DEG, abs=1e-5)
    # an LIA grid (FreeSurfer's conformed order) is not oblique
    lia = np.array([[-2.0, 0, 0, 90], [0, 0, 2.0, -100], [0, -2.0, 0, 110], [0, 0, 0, 1]])
    assert geometry.oblique_deg(Grid((5, 5, 5), m4(lia))) == 0.0
    assert geometry.voxel_mm(Grid((5, 5, 5), m4(lia))) == (2.0, 2.0, 2.0)


def test_the_edge_letters() -> None:
    ras = Grid(GRID_SHAPE, m4(GRID_AFFINE))
    assert dict(geometry.edge_letters(ras)) == {
        "sagittal": ("P", "A", "S", "I"),
        "coronal": ("R", "L", "S", "I"),
        "axial": ("R", "L", "A", "P"),
    }
    las = np.diag([-3.0, 3.0, 3.0, 1.0])  # x runs to the left: the page shows it mirrored
    assert dict(geometry.edge_letters(Grid((4, 4, 4), m4(las)))) == {
        "sagittal": ("P", "A", "S", "I"),
        "coronal": ("L", "R", "S", "I"),
        "axial": ("L", "R", "A", "P"),
    }


# --- the voxel map ------------------------------------------------------------------------


def test_the_voxel_map_takes_a_display_voxel_to_the_raw_voxel(
    dataset: SyntheticFuncDataset,
) -> None:
    """``inv(R_r) . S_a . G``, in that direction, for every packable run of the design: it
    equals the closed form, and the reversed map (raw voxel to display voxel) does not."""
    voxels = voxel_centres(GRID_SHAPE)
    grid = Grid(GRID_SHAPE, m4(GRID_AFFINE))
    for run in dataset.runs:
        if not run.packable:
            continue
        anchor = anchor_of(dataset, run)
        want = raw_sample_voxels(run, anchor, voxels)
        for portable in (True, False):
            vox = geometry.voxel_map(
                m4(run.raw_affine), m4(anchor.anat_to_scanner), grid, portable=portable
            )
            got = voxels @ vox[:3, :3].T + vox[:3, 3]
            np.testing.assert_allclose(got, want, atol=1e-9)
            back = np.linalg.inv(vox)
            assert np.abs(voxels @ back[:3, :3].T + back[:3, 3] - want).max() > 1.0


def test_the_display_to_anatomy_map_sits_between_the_anchor_and_the_grid() -> None:
    raw, anchor, display, p = (rigid_matrix(n) for n in (7, 8, 9, 10))
    grid = Grid((6, 7, 8), m4(display))
    got = geometry.voxel_map(m4(raw), m4(anchor), grid, display_to_anat=m4(p))
    np.testing.assert_allclose(got, np.linalg.inv(raw) @ anchor @ p @ display, atol=1e-9)


# --- the resampling -----------------------------------------------------------------------


def independent_resample(volume: np.ndarray, vox: np.ndarray, shape: tuple) -> np.ndarray:
    """The edge rule written from its statement: a cubic B-spline with the edge values
    repeated, 0 where the sample lies outside ``[0, n - 1]`` along any axis."""
    voxels = voxel_centres(shape)
    u = voxels @ vox[:3, :3].T + vox[:3, 3]
    values = ndimage.map_coordinates(np.asarray(volume, np.float64), u.T, order=3, mode="nearest")
    inside = ((u >= 0) & (u <= np.asarray(volume.shape) - 1)).all(axis=1)
    return np.where(inside, values, 0.0).reshape(shape, order="F")


def test_the_edge_rule_is_zero_outside_the_outermost_voxel_centres() -> None:
    """A constant volume of 5, sampled along axis 0 at -0.3, 0, n - 1 and n - 0.7: the edge
    value repeated at the outermost centres, and 0 beyond them, also within half a voxel."""
    n = 6
    volume = np.full((n, 4, 4), 5.0, np.float32)
    vox = np.eye(4)
    vox[0, 0] = 1.0
    shape = (4, 4, 4)
    positions = np.array([-0.3, 0.0, n - 1.0, n - 0.7])
    # the display voxels 0..3 along axis 0 go to these raw positions: scale and shift
    vox[0, 0] = (positions[1] - positions[0]) / 1.0
    vox = np.eye(4)
    outs = []
    for x in positions:
        vox[0, 3] = x
        outs.append(geometry.resample_voxels(volume, vox, shape)[0, 1, 1])
    assert outs == [0.0, 5.0, 5.0, 0.0]
    # inside, the spline with the edge repeated reproduces a constant exactly, also at the edge
    vox[0, 3] = 0.0
    inside = geometry.resample_voxels(volume, vox, (n, 4, 4))
    assert (inside == 5.0).all()


def test_the_resampling_equals_an_independent_recomputation(
    dataset: SyntheticFuncDataset,
) -> None:
    grid = Grid(GRID_SHAPE, m4(GRID_AFFINE))
    crossed = 0
    for run in dataset.runs:
        if not run.packable:
            continue
        anchor = anchor_of(dataset, run)
        vox = geometry.voxel_map(m4(run.raw_affine), m4(anchor.anat_to_scanner), grid)
        volume = np.asarray(nib.load(run.files["raw_bold"]).dataobj[..., 3], np.float32)
        got = geometry.resample_voxels(volume, vox, GRID_SHAPE)
        want = independent_resample(volume, vox, GRID_SHAPE)
        assert got.dtype == np.float32 and got.shape == GRID_SHAPE
        np.testing.assert_array_equal(got == 0, want == 0)
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-3)
        crossed += int((got == 0).any())
    assert crossed >= 5  # the edge of the raw field of view is crossed in most runs


# --- the raw stream -----------------------------------------------------------------------


def test_the_raw_stream_is_the_closed_form_through_the_anchor(
    dataset: SyntheticFuncDataset,
) -> None:
    """Every packable run, every volume: through the ANCHOR's stored transform, the closed
    form inside the raw field of view, 0 outside it; the anchor's own stream is the
    preprocessed series there, the others are not (the head moved)."""
    grid = Grid(GRID_SHAPE, m4(GRID_AFFINE))
    voxels = voxel_centres(GRID_SHAPE)
    mask = np.asarray(nib.load(dataset.runs[0].files["pre_mask"]).dataobj).ravel(order="F") > 0
    for run in dataset.runs:
        if not run.packable:
            continue
        anchor = anchor_of(dataset, run)
        u = raw_sample_voxels(run, anchor, voxels)
        interior = spline_interior(u)
        outside = ~((u >= 0) & (u <= np.asarray(RAW_SHAPE) - 1)).all(axis=1)
        stream = geometry.raw_stream(
            run.files["raw_bold"], m4(run.raw_affine), grid, m4(anchor.anat_to_scanner)
        )
        seen = []
        for t, volume in stream:
            seen.append(t)
            got = volume.ravel(order="F").astype(np.float64)
            want = expected_raw_stream(run, anchor, voxels, t)
            assert np.abs(got - want)[interior].max() < SPLINE_TOLERANCE, (run.stem, t)
            assert (got[outside] == 0).all()
            if run is anchor:
                pre = expected_pre(voxels, t)
                assert np.abs(got - pre)[interior & mask].max() < SPLINE_TOLERANCE
        assert seen == list(range(run.n_raw_volumes or 0))


def test_every_run_is_drawn_through_the_anchors_transform(dataset: SyntheticFuncDataset) -> None:
    """A run that is not the anchor, drawn through its own transform, would lie where the
    pipeline put it; through the anchor's it lies off by the head's movement between runs."""
    grid = Grid(GRID_SHAPE, m4(GRID_AFFINE))
    voxels = voxel_centres(GRID_SHAPE)
    run = dataset.run("sub-01", "1", "motor", "1")  # displaced 3 mm
    anchor = anchor_of(dataset, run)
    assert anchor is not run
    u = raw_sample_voxels(run, anchor, voxels)
    interior = spline_interior(u)
    [(_, through_anchor)] = geometry.raw_stream(
        run.files["raw_bold"], m4(run.raw_affine), grid, m4(anchor.anat_to_scanner), volumes=[2]
    )
    [(_, through_own)] = geometry.raw_stream(
        run.files["raw_bold"], m4(run.raw_affine), grid, m4(run.anat_to_scanner), volumes=[2]
    )
    want = expected_raw_stream(run, anchor, voxels, 2)
    got = through_anchor.ravel(order="F")
    assert np.abs(got - want)[interior].max() < SPLINE_TOLERANCE
    own = through_own.ravel(order="F")
    assert np.abs(own - want)[interior].max() > 1.0
    assert (
        np.abs(own - expected_pre(voxels, 2))[
            interior & spline_interior(raw_sample_voxels(run, run, voxels))
        ].max()
        < SPLINE_TOLERANCE
    )


def test_volumes_are_read_one_at_a_time(
    dataset: SyntheticFuncDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither the whole stored series nor the whole scaled series is read."""
    from nibabel.arrayproxy import ArrayProxy

    run = dataset.run("sub-01", "1", "rest", "2")
    grid = Grid(GRID_SHAPE, m4(GRID_AFFINE))
    reads: list[tuple] = []
    original = ArrayProxy._get_unscaled

    def counted(self: ArrayProxy, slicer: tuple) -> np.ndarray:
        reads.append(tuple(slicer))
        return original(self, slicer)

    def refused(*_: object, **__: object) -> None:
        raise AssertionError("the whole series was read")

    monkeypatch.setattr(ArrayProxy, "_get_unscaled", counted)
    monkeypatch.setattr(ArrayProxy, "get_unscaled", refused)
    monkeypatch.setattr(ArrayProxy, "__array__", refused)
    out = list(
        geometry.raw_stream(
            run.files["raw_bold"],
            m4(run.raw_affine),
            grid,
            m4(run.anat_to_scanner),
            volumes=[7, 1, 7],
        )
    )
    assert [t for t, _ in out] == [7, 1, 7]
    assert [s[3] for s in reads] == [7, 1, 7]
    assert all(s[:3] == (slice(None),) * 3 for s in reads)
    with pytest.raises(IndexError, match="volume 12"):
        list(
            geometry.raw_stream(
                run.files["raw_bold"],
                m4(run.raw_affine),
                grid,
                m4(run.anat_to_scanner),
                volumes=[12],
            )
        )


def test_stored_values_are_scaled_per_volume(tmp_path: Path) -> None:
    """``stored.astype(float32) * slope + inter``, with the header's scale."""
    stored = np.arange(5 * 6 * 7 * 3, dtype=np.int16).reshape(5, 6, 7, 3, order="F")
    img = nib.Nifti1Image(stored, np.eye(4))
    img.header.set_slope_inter(2.0, 10.0)
    path = tmp_path / "scaled_bold.nii.gz"
    nib.save(img, path)
    grid = Grid((5, 6, 7), IDENTITY4)
    for t, volume in geometry.raw_stream(path, IDENTITY4, grid, IDENTITY4):
        want = stored[..., t].astype(np.float32) * 2.0 + 10.0
        np.testing.assert_allclose(volume, want, atol=1e-3)  # a spline through the identity
    # a 3D image is one volume
    nib.save(nib.Nifti1Image(stored[..., 0].astype(np.float32), np.eye(4)), tmp_path / "v.nii")
    [(t, volume)] = geometry.raw_stream(tmp_path / "v.nii", IDENTITY4, grid, IDENTITY4)
    assert t == 0
    np.testing.assert_allclose(volume, stored[..., 0].astype(np.float32), atol=1e-3)


def stream_digest(raw: Path, raw_affine: tuple, anchor: tuple, portable: bool) -> str:
    grid = Grid(GRID_SHAPE, to_matrix4(GRID_AFFINE))
    h = hashlib.sha256()
    for _, volume in geometry.raw_stream(raw, raw_affine, grid, anchor, portable=portable):
        h.update(volume.tobytes())
    return h.hexdigest()


_DIGEST_SCRIPT = """
import sys, json
sys.path.insert(0, {tests!r})
from test_func_geometry import stream_digest
from pathlib import Path
args = json.loads(sys.argv[1])
print(stream_digest(Path(args["raw"]), tuple(map(tuple, args["raw_affine"])),
      tuple(map(tuple, args["anchor"])), args["portable"]))
"""


def test_the_same_input_gives_the_same_bytes_in_two_processes(
    dataset: SyntheticFuncDataset,
) -> None:
    import json

    run = dataset.run("sub-01", "1", "rest", "1")
    anchor = anchor_of(dataset, run)
    raw_affine, matrix = m4(run.raw_affine), m4(anchor.anat_to_scanner)
    script = _DIGEST_SCRIPT.format(tests=str(Path(__file__).parent))
    for portable in (True, False):
        here = stream_digest(run.files["raw_bold"], raw_affine, matrix, portable)
        assert here == stream_digest(run.files["raw_bold"], raw_affine, matrix, portable)
        args = json.dumps(
            {
                "raw": str(run.files["raw_bold"]),
                "raw_affine": raw_affine,
                "anchor": matrix,
                "portable": portable,
            }
        )
        done = subprocess.run(
            [sys.executable, "-c", script, args], capture_output=True, text=True, check=True
        )
        assert done.stdout.strip() == here
