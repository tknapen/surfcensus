"""The pack of a session (``func.pack``, format ``epi-pack`` version 2): the window and the
rounding, the residuals against the reference volume, the chunks, the stills over the
steady-state volumes, the traces, ``pack.json`` as the private record written last, the
content digest, runs on two grids, and the refusals.

Every expected value is computed here from the NIfTI files with nibabel, numpy and scipy, or
from the closed forms of ``tests/synth_func.py``; only the decoder (``func.packread``) and, for
the raw stream, the geometry of work package E1.B are the package's. The tests that need E1.B's
functions skip while those raise ``NotImplementedError``.

The network test at the end is the exit check of E1.C: the worked example (ds000102 sub-01,
two runs of 146 volumes) packed by this code with numpy's voxel map decodes to the values of a
pack made by stage 1's prototype (``prototypes/func_viewer/stage1/pack_run.py``) of the same
files.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import zlib
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest
from scipy import ndimage
from synth_func import (  # pyrefly: ignore[missing-import]
    FSNATIVE_TO_T1W,
    GRID_AFFINE,
    SPLINE_TOLERANCE,
    SyntheticFuncDataset,
    SyntheticRun,
    apply,
    expected_pre,
    expected_raw_stream,
    raw_sample_voxels,
    spline_interior,
    voxel_centres,
    write_func_dataset,
)

from surfcensus.config import EpiConfig
from surfcensus.func import (
    PACK_JSON,
    PLANES,
    SESSION_JSON,
    STILLS,
    STREAMS,
    func_id,
    geometry,
)
from surfcensus.func import pack as packmod
from surfcensus.func.model import PackInfo, info_text
from surfcensus.func.pack import (
    MAX_CHUNK,
    PERCENTILE,
    WINDOW_STEP,
    PackError,
    chunk_volumes,
    content_digest,
    pack_run,
    pack_session,
    quantise,
    series_volumes,
    session_windows,
    volume_bytes,
    write_contours,
)
from surfcensus.func.packread import (
    decode_contours,
    decode_still,
    decode_volume,
    pack_on_disk,
    read_pack,
    read_session,
)
from surfcensus.records import (
    PACK_FAILURE,
    Anchor,
    Corrections,
    DerivRun,
    Display,
    FrameCheck,
    FrameDisplacement,
    Grid,
    Motion,
    PlaneContours,
    RawAffine,
    RigidPath,
    RunEntities,
    RunPackInput,
    SessionPlan,
    SessionRun,
    Statement,
    to_matrix4,
)
from surfcensus.sources.base import FuncRole

KEY = b"a fixed key of the tests"


def _needs(*functions: Any) -> None:
    """Skip while a function of another work package is a stub."""
    for function in functions:
        try:
            function()
        except NotImplementedError as err:
            pytest.skip(f"not written yet: {err}")
        except Exception:  # noqa: BLE001 - only the stub matters here
            pass


def _geometry_ready() -> None:
    _needs(
        lambda: geometry.voxel_mm(Grid((1, 1, 1), to_matrix4(np.eye(4)))),
        lambda: next(iter(geometry.raw_stream(Path("/nonexistent"), *([None] * 3)))),  # type: ignore[arg-type]
    )


# --- building the inputs from the fixture ---------------------------------------------------


def run_id(run: SyntheticRun) -> str:
    return func_id(KEY, "run", run.stem)


def session_id(participant: str, session: str | None) -> str:
    return func_id(KEY, "session", participant, session or "")


def motion_of(run: SyntheticRun) -> Motion:
    """What the confounds table of the fixture holds, read here with the standard library."""
    rows = run.files["confounds"].read_text().splitlines()
    names = rows[0].split("\t")

    def column(name: str) -> tuple[float | None, ...]:
        i = names.index(name)
        return tuple(
            None if r.split("\t")[i] == "n/a" else float(r.split("\t")[i]) for r in rows[1:]
        )

    outlier = column("non_steady_state_outlier00")
    return Motion(
        fd=column("framewise_displacement"),
        dvars=column("std_dvars"),
        non_steady=tuple(t for t, v in enumerate(outlier) if v),
        source=FuncRole.CONFOUNDS,
    )


def _statement() -> Statement:
    return Statement("not applied", "Not applied", FuncRole.SUMMARY_REPORT)


CORRECTIONS = Corrections(
    motion=Statement("applied", None, None),
    sdc=_statement(),
    sdc_method=None,
    slice_timing=_statement(),
    registration=Statement("applied", "bbregister", FuncRole.SUMMARY_REPORT),
    dof=6,
    sdc_figure_listed=False,
)


def run_input(
    run: SyntheticRun, anchor: SyntheticRun, *, motion: bool = True, **files: Path
) -> RunPackInput:
    """The input of one run, as E1.B and E1.A would make it, from the fixture's closed forms.

    The raw affine is the fixture's true ``R_r`` (what the arbiter rule gives on this
    fixture; the tests of E1.B check the rule), so nothing here depends on E1.A."""
    roles = {FuncRole(k): v for k, v in run.files.items() if k in FuncRole.__members__.values()}
    roles |= {FuncRole(k): v for k, v in files.items()}
    matrix = to_matrix4(run.anat_to_scanner)
    return RunPackInput(
        run=run_id(run),
        name=run.name,
        files=roles,
        grid=Grid(nib.load(str(roles[FuncRole.PRE_BOLD])).shape[:3], to_matrix4(GRID_AFFINE)),
        display_to_anat=to_matrix4(np.eye(4)),
        anchor_run=run_id(anchor),
        anchor_matrix=to_matrix4(anchor.anat_to_scanner),
        raw_affine=RawAffine(to_matrix4(run.raw_affine), "header", True),
        motion=motion_of(run) if motion else None,
        corrections=CORRECTIONS,
        rigid=RigidPath(matrix, 6, 0.0, True, FuncRole.XFM_TO_ANAT, FuncRole.REFERENCE),
        tr_s=2.0,
    )


def _zero_frame(flagged: bool = False) -> FrameDisplacement:
    return FrameDisplacement(0.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0), 0.0, 100, flagged)


def plan_of(
    ds: SyntheticFuncDataset, runs: list[SyntheticRun], anchor: SyntheticRun, fs: str | None
) -> SessionPlan:
    """A session plan of these runs (in this order) with the given anchor."""
    session_runs = []
    grids: dict[str, Grid] = {}
    grid_of: dict[str, str] = {}
    for r in runs:
        d = r.design
        deriv = DerivRun(RunEntities(sub=d.participant[4:]), r.key, (), r.stem, ("T1w",), {})
        session_runs.append(
            SessionRun(run_id(r), r.name, deriv, None, Display("anat", "T1w"), None)
        )
        pre = r.files["pre_bold"]
        grid = Grid(tuple(nib.load(str(pre)).shape[:3]), to_matrix4(GRID_AFFINE))
        grids[grid.key] = grid
        grid_of[run_id(r)] = grid.key
    frame = FrameCheck(
        anchor=run_id(anchor),
        limit_mm=10.0,
        runs={run_id(r): _zero_frame() for r in runs},
        flagged=(),
        shared_frame_plausible=True,
        anchor_is_outlier=False,
    )
    first = runs[0].design
    return SessionPlan(
        participant=first.participant,
        session=first.session,
        fs_subject=fs,
        runs=tuple(session_runs),
        anchor=Anchor(run_id(anchor), "acquisition_time", "the first by time"),
        frame=frame,
        grids=grids,
        grid_of=grid_of,
    )


def fake_contours(grid: Grid) -> tuple[PlaneContours, ...]:
    """Contour files of the right planes whose content is made up: the pack only stores them."""
    return tuple(
        PlaneContours(plane, f"contours of {plane} {grid.key}".encode() * 50, grid.shape[n], 3, 9)
        for n, plane in enumerate(PLANES)
    )


@pytest.fixture(scope="module")
def ds(tmp_path_factory: pytest.TempPathFactory) -> SyntheticFuncDataset:
    return write_func_dataset(tmp_path_factory.mktemp("func"))


def ses1(ds: SyntheticFuncDataset) -> tuple[list[SyntheticRun], SyntheticRun]:
    """The packable runs of sub-01 ses-1 in acquisition order, and the anchor."""
    s = ds.session("sub-01", "1")
    by_stem = {r.stem: r for r in ds.runs}
    return [by_stem[stem] for stem in s.order], by_stem[s.anchor]


@pytest.fixture(scope="module")
def packed(
    ds: SyntheticFuncDataset, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, list[SyntheticRun], SyntheticRun]:
    """sub-01 ses-1 packed with the real geometry and made-up contours."""
    _geometry_ready()
    runs, anchor = ses1(ds)
    plan = plan_of(ds, runs, anchor, "sub-01")
    contours = {key: fake_contours(grid) for key, grid in plan.grids.items()}
    out = tmp_path_factory.mktemp("out")
    sid = session_id("sub-01", "1")
    pack_session(out, sid, plan, [run_input(r, anchor) for r in runs], contours, EpiConfig())
    return out / sid, runs, anchor


def stored(path: Path) -> np.ndarray:
    """A series as nibabel scales it, float32, read whole (the test may)."""
    img = nib.load(str(path))
    return np.asarray(img.dataobj.get_unscaled()).astype(np.float32) * float(
        img.dataobj.slope
    ) + float(img.dataobj.inter)


def my_quantise(volume: np.ndarray, hi: float) -> np.ndarray:
    """The rule of the pack written out again: floor(v * 255 / hi + 0.5), clipped, float32."""
    v = np.where(np.isnan(volume), 0, volume).astype(np.float32)
    x = v * np.float32(255.0 / hi) + np.float32(0.5)
    return np.floor(np.clip(x, 0, 255)).astype(np.uint8)


def mask_of(run: SyntheticRun) -> np.ndarray:
    return np.asarray(nib.load(str(run.files["pre_mask"])).dataobj) > 0


def raw_expected(run: SyntheticRun, anchor: SyntheticRun, t: int) -> np.ndarray:
    """The raw stream of volume t computed here: a cubic spline at ``inv(R_r) S_a G v`` by
    ``map_coordinates``, 0 outside the outermost raw voxel centres."""
    shape = nib.load(str(run.files["pre_bold"])).shape[:3]
    voxels = voxel_centres(shape)
    where = raw_sample_voxels(run, anchor, voxels)
    raw = stored(run.files["raw_bold"])[..., t]
    values = ndimage.map_coordinates(raw, where.T, order=3, mode="nearest")
    top = np.asarray(raw.shape) - 1
    inside = ((where >= 0) & (where <= top)).all(axis=1)
    return np.where(inside, values, 0.0).reshape(shape, order="F").astype(np.float32)


# --- the values -----------------------------------------------------------------------------


def test_quantise_rounds_half_up_clips_and_maps_nan_to_zero() -> None:
    v = np.array([[[-3.0, 0.0, 1.49, 1.5, 2.5, 254.6, 300.0, np.nan, np.inf]]], np.float32)
    assert quantise(v, 0.0, 255.0).ravel().tolist() == [0, 0, 1, 2, 3, 255, 255, 0, 255]
    rng = np.random.default_rng(3)
    x = rng.normal(500, 300, (7, 8, 9)).astype(np.float32)
    assert np.array_equal(quantise(x, 0.0, 987.25), my_quantise(x, 987.25))
    assert quantise(x, 0.0, 987.25).dtype == np.uint8


def test_volume_bytes_are_in_fortran_order() -> None:
    v = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    data = volume_bytes(v)
    for i, j, k in np.ndindex(2, 3, 4):
        assert data[i + 2 * (j + 3 * k)] == v[i, j, k]


def test_chunk_volumes_fit_the_bytes_between_one_and_the_maximum() -> None:
    assert chunk_volumes(1000, 4 * 1024 * 1024) == MAX_CHUNK
    assert chunk_volumes(1000, 5000) == 5
    assert chunk_volumes(1000, 999) == 1
    assert chunk_volumes(109_368, 4 * 1024 * 1024) == 32  # the worked example, as stage 1


def test_series_volumes_reads_one_volume_at_a_time(
    ds: SyntheticFuncDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = ds.runs[0].files["pre_bold"]
    whole = stored(path)

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the series was read whole")

    monkeypatch.setattr(nib.arrayproxy.ArrayProxy, "get_unscaled", refuse)
    monkeypatch.setattr(nib.arrayproxy.ArrayProxy, "__array__", refuse)
    got = list(series_volumes(path))
    assert [t for t, _ in got] == list(range(whole.shape[3]))
    for t, vol in got:
        assert vol.dtype == np.float32 and np.array_equal(vol, whole[..., t])
    assert [t for t, _ in series_volumes(path, [3, 7])] == [3, 7]
    with pytest.raises(IndexError):
        list(series_volumes(path, [99]))


# --- what a pack holds ----------------------------------------------------------------------


def pooled_windows(runs: list[SyntheticRun]) -> tuple[float, float]:
    """The session's window and SD window, computed here over the runs as stored."""
    values, sds = [], []
    for r in runs:
        pre = stored(r.files["pre_bold"])
        mask = mask_of(r)
        values += [pre[..., t][mask] for t in range(0, pre.shape[3], WINDOW_STEP)]
        steady = pre[..., [t for t in range(pre.shape[3]) if t not in motion_of(r).non_steady]]
        sds.append(steady.astype(np.float64).std(axis=3, ddof=1)[mask])
    return (
        float(np.percentile(np.concatenate(values), PERCENTILE)),
        float(np.percentile(np.concatenate(sds), PERCENTILE)),
    )


def test_one_window_for_the_session_from_all_its_runs(packed) -> None:
    session_dir, runs, _ = packed
    hi, sd_hi = pooled_windows(runs)
    session = read_session(session_dir)
    assert session is not None and session.window == (0.0, hi)
    assert session.sd_window is not None and session.sd_window[1] == pytest.approx(sd_hi, 1e-9)
    for r in runs:
        info = read_pack(session_dir, run_id(r))
        assert info is not None
        assert info.window == session.window and info.sd_window == session.sd_window


def test_a_brighter_run_is_shown_brighter_and_changes_every_pack(
    ds: SyntheticFuncDataset, tmp_path: Path
) -> None:
    _geometry_ready()
    runs, anchor = ses1(ds)
    img = nib.load(str(runs[1].files["pre_bold"]))
    bright_path = tmp_path / "bright_bold.nii.gz"
    bright = np.asarray(img.dataobj, np.float32) * np.float32(1.5)
    nib.save(nib.Nifti1Image(bright, img.affine, img.header), bright_path)
    inputs = [run_input(r, anchor) for r in runs]
    inputs[1] = replace(inputs[1], files={**inputs[1].files, FuncRole.PRE_BOLD: bright_path})
    plan = plan_of(ds, runs, anchor, None)
    info = pack_session(tmp_path / "a", session_id("sub-01", "1"), plan, inputs, None, EpiConfig())
    session_dir = tmp_path / "a" / info.session
    assert info.window is not None
    a = read_pack(session_dir, run_id(runs[0]))
    b = read_pack(session_dir, run_id(runs[1]))
    assert a is not None and b is not None and a.window == b.window == info.window
    # the same voxel, one run 1.5 times the other: not normalised away by the display
    va = decode_volume(session_dir, a, "pre", 3).astype(float)
    vb = decode_volume(session_dir, b, "pre", 3).astype(float)
    inside = (va > 20) & (vb < 250)
    assert np.allclose(vb[inside], 1.5 * va[inside], atol=1.0)
    # without the bright run the window is another, and so is the pack of every other run
    fewer = plan_of(ds, [runs[0], runs[2], runs[3]], anchor, None)
    other = pack_session(
        tmp_path / "b", info.session, fewer, [inputs[0], *inputs[2:]], None, EpiConfig()
    )
    assert other.window != info.window
    c = read_pack(tmp_path / "b" / info.session, run_id(runs[0]))
    assert c is not None and c.content_digest != a.content_digest


def test_the_runs_lie_on_one_timeline_in_acquisition_order(packed) -> None:
    session_dir, runs, _ = packed
    session = read_session(session_dir)
    assert session is not None
    assert [x.run for x in session.runs] == [run_id(r) for r in runs]
    assert [x.first for x in session.runs] == [0, 12, 24, 36]
    assert [x.n_volumes for x in session.runs] == [12] * 4
    assert [x.tr_s for x in session.runs] == [2.0] * 4


def test_slice_offsets_place_slices_of_two_grids_in_one_frame() -> None:
    from surfcensus.func.pack import slice_offsets

    a = Grid((4, 5, 6), to_matrix4(GRID_AFFINE))
    b_affine = GRID_AFFINE.copy()
    b_affine[:3, 3] += (8.0, -16.0, 4.0)  # one voxel right, two back, half a voxel up
    b = Grid((4, 5, 7), to_matrix4(b_affine))
    axes, here = slice_offsets(a, a)
    assert axes == {"sagittal": "x", "coronal": "y", "axial": "z"}
    assert here["sagittal"] == (-12.0, -4.0, 4.0, 12.0)  # 8 mm voxels about the middle
    _, there = slice_offsets(b, a)
    assert there["sagittal"] == (-4.0, 4.0, 12.0, 20.0)
    assert there["coronal"][2] == here["coronal"][0]
    assert there["axial"][0] == here["axial"][0] + 4.0
    # an oblique grid: the offset of a slice is where its centre lies along the axis
    tilt = np.eye(4)
    c, s = np.cos(np.radians(20.0)), np.sin(np.radians(20.0))
    tilt[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
    oblique = Grid((4, 5, 6), to_matrix4(tilt @ GRID_AFFINE))
    axes, values = slice_offsets(oblique, a)
    assert axes == {"sagittal": "x", "coronal": "y", "axial": "z"}
    centre = apply(GRID_AFFINE, np.array([[1.5, 2.0, 2.5]]))[0]
    for k in range(5):
        point = apply(tilt @ GRID_AFFINE, np.array([[1.5, k, 2.5]]))[0]
        assert values["coronal"][k] == pytest.approx(point[1] - centre[1], abs=1e-6)


def test_the_preprocessed_stream_is_the_series_as_written_quantised(packed) -> None:
    session_dir, runs, _ = packed
    for r in runs:
        info = read_pack(session_dir, run_id(r))
        assert info is not None
        pre = stored(r.files["pre_bold"])
        shape = pre.shape[:3]
        for t in range(info.n_volumes):
            decoded = decode_volume(session_dir, info, "pre", t)
            assert np.array_equal(decoded, my_quantise(pre[..., t], info.window[1])), t
            # and the closed form, within the rounding of one gray level
            closed = expected_pre(voxel_centres(shape), t).reshape(shape, order="F")
            q = my_quantise(closed.astype(np.float32), info.window[1]).astype(int)
            assert np.abs(decoded.astype(int) - q).max() <= 1


def test_the_raw_stream_is_the_raw_series_through_the_anchor_and_nothing_else(packed) -> None:
    session_dir, runs, anchor = packed
    for r in runs:
        info = read_pack(session_dir, run_id(r))
        assert info is not None
        shape = nib.load(str(r.files["pre_bold"])).shape[:3]
        voxels = voxel_centres(shape)
        interior = spline_interior(raw_sample_voxels(r, anchor, voxels))
        assert interior.sum() > 50
        for t in (0, 5, info.n_volumes - 1):
            decoded = decode_volume(session_dir, info, "raw", t).astype(int)
            mine = my_quantise(raw_expected(r, anchor, t), info.window[1]).astype(int)
            # the portable voxel map differs from numpy's in the last bits: a sample on a
            # rounding boundary may land one gray level away
            assert np.abs(decoded - mine).max() <= 1
            assert (decoded == mine).mean() > 0.995
            closed = expected_raw_stream(r, anchor, voxels, t).astype(np.float32)
            step = 255.0 / info.window[1]
            allowed = SPLINE_TOLERANCE * step + 1
            got = decoded.ravel(order="F")[interior]
            want = my_quantise(closed[interior], info.window[1]).astype(int)
            assert np.abs(got - want).max() <= allowed


def test_chunks_hold_residuals_against_the_reference_volume(packed) -> None:
    session_dir, runs, _ = packed
    r = runs[1]
    info = read_pack(session_dir, run_id(r))
    assert info is not None
    pre = stored(r.files["pre_bold"])
    hi = info.window[1]
    sub = range(0, info.n_volumes, WINDOW_STEP)
    ref = np.floor(
        sum(my_quantise(pre[..., t], hi).astype(np.float64) for t in sub) / len(sub) + 0.5
    ).astype(np.uint8)
    assert np.array_equal(decode_still(session_dir, info, "pre", "ref"), ref)
    chunk = info.streams["pre"].chunks[0]
    data = zlib.decompress((session_dir / chunk.file).read_bytes())
    nvox = ref.size
    for n in range(chunk.count):
        values = my_quantise(pre[..., chunk.first + n], hi)
        residual = ((values.astype(int) - ref.astype(int)) % 256).astype(np.uint8)
        assert data[n * nvox : (n + 1) * nvox] == residual.tobytes(order="F")
    # the raw reference likewise, from the decoded raw volumes
    raw = [decode_volume(session_dir, info, "raw", t).astype(np.float64) for t in sub]
    raw_ref = np.floor(sum(raw) / len(sub) + 0.5).astype(np.uint8)
    assert np.array_equal(decode_still(session_dir, info, "raw", "ref"), raw_ref)


def test_chunk_boundaries(ds: SyntheticFuncDataset, tmp_path: Path) -> None:
    _geometry_ready()
    runs, anchor = ses1(ds)
    r = runs[0]
    inp = run_input(r, anchor)
    nvox = int(np.prod(inp.grid.shape))
    info = pack_run(tmp_path, inp, EpiConfig(chunk_bytes=5 * nvox))
    for stream in STREAMS:
        spans = [(c.first, c.count) for c in info.streams[stream].chunks]
        assert spans == [(0, 5), (5, 5), (10, 2)]
        assert [c.file.rsplit("/", 1)[1] for c in info.streams[stream].chunks] == [
            "c0000.bin",
            "c0001.bin",
            "c0002.bin",
        ]
    # the same values as a pack of one chunk, volume by volume, across the boundaries
    other = tmp_path / "one"
    whole = pack_run(other, inp, EpiConfig())
    assert whole.chunk_volumes == MAX_CHUNK and len(whole.streams["pre"].chunks) == 1
    plan = plan_of(ds, [r], anchor, None)
    for directory, item in ((tmp_path, info), (other, whole)):
        _write_session_json(directory, plan, item)
    for stream in STREAMS:
        for t in range(info.n_volumes):
            a = decode_volume(tmp_path, info, stream, t)
            b = decode_volume(other, whole, stream, t)
            assert np.array_equal(a, b), (stream, t)
    assert info.content_digest != whole.content_digest  # chunk_volumes is part of the pack


def _write_session_json(directory: Path, plan: SessionPlan, info: PackInfo) -> None:
    """A session.json for a run packed alone, so that the decoder knows its grid's shape."""
    from surfcensus.func.model import GridInfo, SessionInfo

    grid = plan.grids[info.grid_key]
    session = SessionInfo(
        surfcensus_version="test",
        session=directory.name,
        label="sub-01",
        config_hash="x",
        fd_threshold_mm=0.5,
        registration=None,
        grids=(
            GridInfo(
                grid_key=grid.key,
                shape=grid.shape,
                voxel_mm=(8.0, 8.0, 8.0),
                oblique_deg=0.0,
                edge_letters={},
                contours=None,
            ),
        ),
        runs=(),
        contours_from=None,
    )
    (directory / SESSION_JSON).write_text(info_text(session))


def test_the_stills_are_over_the_steady_state_volumes(packed) -> None:
    session_dir, runs, _ = packed
    session = read_session(session_dir)
    assert session is not None and session.sd_window is not None
    for r in runs:
        info = read_pack(session_dir, run_id(r))
        assert info is not None
        non_steady = motion_of(r).non_steady
        assert non_steady == (0,)
        assert info.stills_volumes == info.n_volumes - 1
        pre = stored(r.files["pre_bold"]).astype(np.float64)
        steady = pre[..., 1:]
        hi = info.window[1]
        mean = steady.mean(axis=3)
        assert np.array_equal(decode_still(session_dir, info, "pre", "mean"), my_quantise(mean, hi))
        sd = steady.std(axis=3, ddof=1)
        got = decode_still(session_dir, info, "pre", "sd").astype(int)
        assert np.abs(got - my_quantise(sd, session.sd_window[1]).astype(int)).max() <= 1
        # over every volume the mean is another: the signal changes with the volume
        assert not np.array_equal(my_quantise(pre.mean(axis=3), hi), my_quantise(mean, hi))
        traces = json.loads((session_dir / info.traces.file).read_text())
        assert traces["non_steady"] == [0] and traces["fd"][0] is None
        assert len(traces["fd"]) == len(traces["dvars"]) == info.n_volumes


def test_pack_json_is_the_private_record(packed, ds: SyntheticFuncDataset) -> None:
    session_dir, runs, anchor = packed
    for r in runs:
        info = read_pack(session_dir, run_id(r))
        assert info is not None
        text = (session_dir / run_id(r) / PACK_JSON).read_text()
        assert str(ds.root) not in text and "sub-0" not in text and ".nii" not in text
        assert "task" not in text and "/tmp" not in text
        roles = [i.role for i in info.inputs]
        assert roles == [role.value for role in FuncRole if role.value in roles]
        assert roles[:2] == ["raw_bold", "raw_sidecar"]
        for item in info.inputs:
            path = r.files[item.role]
            assert item.size == path.stat().st_size
            assert item.base == ("raw" if item.role.startswith("raw_") else "deriv")
        assert info.config_hash == EpiConfig().pack_hash()
        assert info.anchor_run == run_id(anchor) and info.session == session_dir.name
        assert info.content_digest == content_digest(session_dir, info)
        assert pack_on_disk(session_dir, run_id(r))
    session = read_session(session_dir)
    assert session is not None
    assert [x.run for x in session.runs] == [run_id(r) for r in runs]
    assert session.runs[0].anchor and session.registration is not None
    assert {g.grid_key for g in session.grids} == {read_pack(session_dir, run_id(runs[0])).grid_key}  # type: ignore[union-attr]


def test_two_packs_of_one_input_have_one_digest_and_the_same_bytes(
    ds: SyntheticFuncDataset, tmp_path: Path
) -> None:
    _geometry_ready()
    runs, anchor = ses1(ds)
    inp = run_input(runs[2], anchor)
    a = pack_run(tmp_path / "a", inp, EpiConfig())
    b = pack_run(tmp_path / "b", inp, EpiConfig())
    assert a.content_digest == b.content_digest
    # here, on one machine with one zlib, the compressed files are the same bytes too; that
    # is not promised across machines or zlib builds, which is why the digest is decoded
    files_a = _files(tmp_path / "a")
    assert files_a == _files(tmp_path / "b") and len(files_a) > 6


def test_a_pack_made_in_another_process_has_the_same_digest(
    ds: SyntheticFuncDataset, tmp_path: Path
) -> None:
    _geometry_ready()
    runs, anchor = ses1(ds)
    inp = run_input(runs[1], anchor)
    here = pack_run(tmp_path / "here", inp, EpiConfig())
    blob = tmp_path / "input.pickle"
    blob.write_bytes(pickle.dumps(inp))
    script = (
        "import pickle, sys\n"
        "from surfcensus.config import EpiConfig\n"
        "from surfcensus.func.pack import pack_run\n"
        "inp = pickle.loads(open(sys.argv[1], 'rb').read())\n"
        "print(pack_run(sys.argv[2], inp, EpiConfig()).content_digest)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script, str(blob), str(tmp_path / "there")],
        check=True,
        capture_output=True,
        text=True,
    )
    assert out.stdout.strip() == here.content_digest
    assert _files(tmp_path / "here") == _files(tmp_path / "there")


def _files(session_dir: Path) -> dict[str, bytes]:
    return {
        p.relative_to(session_dir).as_posix(): p.read_bytes()
        for p in sorted(session_dir.rglob("*"))
        if p.is_file()
    }


def _rewrite(session_dir: Path, info: PackInfo, stream: str, index: int, data: bytes) -> PackInfo:
    """Replace chunk ``index`` of a stream by ``data`` compressed at another level, with
    ``pack.json`` updated to list it."""
    import hashlib

    chunk = info.streams[stream].chunks[index]
    blob = zlib.compress(data, 1)
    (session_dir / chunk.file).write_bytes(blob)
    new = chunk.model_copy(update={"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()})
    chunks = list(info.streams[stream].chunks)
    chunks[index] = new
    streams = dict(info.streams)
    streams[stream] = info.streams[stream].model_copy(update={"chunks": tuple(chunks)})
    return info.model_copy(update={"streams": streams})


def test_the_digest_ignores_compressed_bytes_and_notices_one_changed_value(
    ds: SyntheticFuncDataset, tmp_path: Path
) -> None:
    _geometry_ready()
    runs, anchor = ses1(ds)
    info = pack_run(tmp_path, run_input(runs[0], anchor), EpiConfig())
    chunk = info.streams["raw"].chunks[0]
    data = zlib.decompress((tmp_path / chunk.file).read_bytes())
    same = _rewrite(tmp_path, info, "raw", 0, data)
    assert (tmp_path / chunk.file).read_bytes() != zlib.compress(data, 6)
    assert content_digest(tmp_path, same) == info.content_digest
    changed = bytearray(data)
    changed[1234] = (changed[1234] + 1) % 256
    other = _rewrite(tmp_path, info, "raw", 0, bytes(changed))
    assert content_digest(tmp_path, other) != info.content_digest
    # the JSON that defines a pack counts too: another window is another pack
    moved = same.model_copy(update={"window": (0.0, info.window[1] + 1)})
    assert content_digest(tmp_path, moved) != info.content_digest
    # and a file that is not the one pack.json lists is refused, not digested
    with pytest.raises(ValueError, match="SHA-256"):
        content_digest(tmp_path, info)


def test_the_digest_covers_the_contours_of_the_grid(packed) -> None:
    session_dir, runs, _ = packed
    info = read_pack(session_dir, run_id(runs[0]))
    assert info is not None
    path = session_dir / "contours" / info.grid_key / "axial.bin"
    original = path.read_bytes()
    try:
        path.write_bytes(zlib.compress(zlib.decompress(original), 9))
        assert content_digest(session_dir, info) == info.content_digest
        path.write_bytes(zlib.compress(zlib.decompress(original) + b"x"))
        assert content_digest(session_dir, info) != info.content_digest
    finally:
        path.write_bytes(original)


def test_pack_json_is_written_last_and_an_interrupted_pack_leaves_the_earlier_one(
    ds: SyntheticFuncDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _geometry_ready()
    runs, anchor = ses1(ds)
    inp = run_input(runs[0], anchor)
    written: list[str] = []
    real_bytes, real_text = Path.write_bytes, Path.write_text

    def record_bytes(self: Path, data: Any) -> int:
        written.append(self.name)
        return real_bytes(self, data)

    def record_text(self: Path, data: str, *args: Any, **kwargs: Any) -> int:
        written.append(self.name)
        return real_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_bytes", record_bytes)
    monkeypatch.setattr(Path, "write_text", record_text)
    first = pack_run(tmp_path, inp, EpiConfig())
    monkeypatch.undo()
    assert written[-1] == PACK_JSON and written.count(PACK_JSON) == 1
    assert len(written) == 2 * (3 + 1) + 1 + 1  # 2 streams x (stills + one chunk), traces, json
    run = run_id(runs[0])
    before = _files(tmp_path)

    def broken(*args: Any, **kwargs: Any) -> Iterator[tuple[int, np.ndarray]]:
        for t, vol in geometry.raw_stream(*args, **kwargs):
            if t == 7:
                raise RuntimeError("interrupted")
            yield t, vol

    monkeypatch.setattr(packmod.geometry, "raw_stream", broken)
    with pytest.raises(RuntimeError, match="interrupted"):
        pack_run(tmp_path, replace(inp, tr_s=3.0), EpiConfig())
    monkeypatch.undo()
    assert _files(tmp_path) == before  # the earlier pack, and no temporary directory
    assert [p.name for p in tmp_path.iterdir()] == [run]
    assert pack_on_disk(tmp_path, run) and read_pack(tmp_path, run) == first
    # a pack that lost a file, or its pack.json, is not whole
    (tmp_path / first.streams["pre"].chunks[0].file).unlink()
    assert not pack_on_disk(tmp_path, run)
    pack_run(tmp_path, inp, EpiConfig())
    assert pack_on_disk(tmp_path, run)
    (tmp_path / run / PACK_JSON).unlink()
    assert not pack_on_disk(tmp_path, run)


def test_a_new_pack_replaces_the_earlier_one_whole(
    ds: SyntheticFuncDataset, tmp_path: Path
) -> None:
    _geometry_ready()
    runs, anchor = ses1(ds)
    inp = run_input(runs[0], anchor)
    nvox = int(np.prod(inp.grid.shape))
    pack_run(tmp_path, inp, EpiConfig(chunk_bytes=2 * nvox))
    run = run_id(runs[0])
    assert len(list((tmp_path / run / "raw").iterdir())) == 3 + 6
    info = pack_run(tmp_path, inp, EpiConfig())
    assert sorted(p.name for p in (tmp_path / run / "raw").iterdir()) == [
        "c0000.bin",
        "mean.bin",
        "ref.bin",
        "sd.bin",
    ]
    assert read_pack(tmp_path, run) == info
    # what a killed call of this run left behind goes with the next pack of it
    (tmp_path / f".{run}.{'0' * 16}.tmp").mkdir()
    (tmp_path / f".{'f' * 16}.{'0' * 16}.tmp").mkdir()  # another run's: left alone
    pack_run(tmp_path, inp, EpiConfig())
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [run, f".{'f' * 16}.{'0' * 16}.tmp"]
    )


def test_runs_on_two_grids_are_two_groups_with_their_own_contours(
    ds: SyntheticFuncDataset, tmp_path: Path
) -> None:
    _geometry_ready()
    by_case = {r.design.case: r for r in ds.runs_of("sub-02", None)}
    anchor = by_case["anchor by name"]
    runs = [anchor, by_case["no time, raw run-02, no report"], by_case["other grid"]]
    plan = plan_of(ds, runs, anchor, "sub-02")
    assert len(plan.grids) == 2
    contours = {key: fake_contours(grid) for key, grid in plan.grids.items()}
    sid = session_id("sub-02", None)
    info = pack_session(
        tmp_path, sid, plan, [run_input(r, anchor) for r in runs], contours, EpiConfig()
    )
    session_dir = tmp_path / sid
    assert info.label == "sub-02" and info.contours_from == "sub-02"
    assert [x.reason for x in info.runs] == [None, None, None]
    keys = [plan.grid_of[run_id(r)] for r in runs]
    assert keys[0] == keys[1] != keys[2]
    assert [g.grid_key for g in info.grids] == [keys[0], keys[2]]
    shapes = {g.grid_key: g.shape for g in info.grids}
    for r, key in zip(runs, keys, strict=True):
        pack = read_pack(session_dir, run_id(r))
        assert pack is not None and pack.grid_key == key
        assert decode_volume(session_dir, pack, "pre", 0).shape == shapes[key]
    for grid in info.grids:
        assert grid.contours is not None
        for plane in PLANES:
            data = decode_contours(session_dir, grid.grid_key, plane)
            want = next(c for c in contours[grid.grid_key] if c.plane == plane).data
            assert data == want
            assert grid.contours[plane].file == f"contours/{grid.grid_key}/{plane}.bin"
    assert read_session(session_dir) == info


def test_a_run_that_cannot_be_packed_is_listed_and_the_others_are_packed(
    ds: SyntheticFuncDataset, tmp_path: Path
) -> None:
    _geometry_ready()
    by_case = {r.design.case: r for r in ds.runs_of("sub-02", None)}
    anchor = by_case["anchor by name"]
    differ = by_case["volume counts differ"]
    runs = [anchor, differ]
    plan = plan_of(ds, runs, anchor, None)
    sid = session_id("sub-02", None)
    stale = tmp_path / sid / run_id(differ)
    stale.mkdir(parents=True)
    (stale / PACK_JSON).write_text("{}")
    info = pack_session(
        tmp_path, sid, plan, [run_input(r, anchor) for r in runs], None, EpiConfig()
    )
    assert [x.reason for x in info.runs] == [None, "volume counts differ"]
    assert info.runs[1].grid_key is not None  # it was packable by the plan
    assert info.contours_from is None and all(g.contours is None for g in info.grids)
    assert not stale.exists() and pack_on_disk(tmp_path / sid, run_id(anchor))


@pytest.mark.parametrize("case", ["another grid", "empty", "no file"])
def test_a_brain_mask_off_the_grid_or_empty_is_no_brain_mask(
    case: str, ds: SyntheticFuncDataset, tmp_path: Path
) -> None:
    _geometry_ready()
    runs, anchor = ses1(ds)
    r = runs[0]
    inp = run_input(r, anchor)
    if case == "another grid":
        other = next(x for x in ds.runs if x.design.case == "other grid")
        inp = replace(inp, files={**inp.files, FuncRole.PRE_MASK: other.files["pre_mask"]})
    elif case == "empty":
        img = nib.load(str(r.files["pre_mask"]))
        empty = tmp_path / "empty_mask.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros(img.shape, np.uint8), img.affine), empty)
        inp = replace(inp, files={**inp.files, FuncRole.PRE_MASK: empty})
    else:
        inp = replace(inp, files={k: v for k, v in inp.files.items() if k != FuncRole.PRE_MASK})
    with pytest.raises(PackError) as err:
        pack_run(tmp_path / "out", inp, EpiConfig())
    assert err.value.reason == "no brain mask"
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


def test_volume_counts_of_the_confounds_table_must_agree(
    ds: SyntheticFuncDataset, tmp_path: Path
) -> None:
    runs, anchor = ses1(ds)
    inp = run_input(runs[0], anchor)
    assert inp.motion is not None
    short = replace(inp.motion, fd=inp.motion.fd[:-1], dvars=inp.motion.dvars[:-1])
    with pytest.raises(PackError, match="confounds table has 11 rows"):
        pack_run(tmp_path, replace(inp, motion=short), EpiConfig())


def test_a_run_without_motion_estimates_has_empty_traces_and_stills_over_every_volume(
    ds: SyntheticFuncDataset, tmp_path: Path
) -> None:
    _geometry_ready()
    runs, anchor = ses1(ds)
    info = pack_run(tmp_path, run_input(runs[0], anchor, motion=False), EpiConfig())
    assert info.stills_volumes == info.n_volumes
    traces = json.loads((tmp_path / info.traces.file).read_text())
    assert traces == {"dvars": [], "fd": [], "non_steady": []}


def test_an_unreadable_series_is_a_pack_failure_without_a_path(
    ds: SyntheticFuncDataset, tmp_path: Path
) -> None:
    runs, anchor = ses1(ds)
    broken = tmp_path / "broken" / "x_bold.nii.gz"
    broken.parent.mkdir()
    broken.write_bytes(b"not a nifti")
    inp = run_input(runs[0], anchor, pre_bold=broken)
    with pytest.raises(PackError) as err:
        pack_run(tmp_path / "out", inp, EpiConfig())
    assert err.value.reason == PACK_FAILURE and str(tmp_path) not in err.value.detail
    assert "pre_bold" in err.value.detail


def test_pack_error_pickles() -> None:
    err = pickle.loads(pickle.dumps(PackError("volume counts differ", "raw 12, pre 10")))
    assert (err.reason, err.detail) == ("volume counts differ", "raw 12, pre 10")


def test_write_contours_compresses_each_plane_and_needs_all_three(tmp_path: Path) -> None:
    grid = Grid((4, 5, 6), to_matrix4(np.eye(4)))
    files = write_contours(tmp_path, grid.key, fake_contours(grid))
    assert list(files) == list(PLANES)
    for n, plane in enumerate(PLANES):
        assert files[plane].slices == grid.shape[n]
        blob = (tmp_path / files[plane].file).read_bytes()
        assert len(blob) == files[plane].bytes
        assert zlib.decompress(blob) == fake_contours(grid)[n].data
    with pytest.raises(ValueError):
        write_contours(tmp_path, grid.key, fake_contours(grid)[:2])


def test_real_contours_of_the_fixture_survive_the_pack(
    ds: SyntheticFuncDataset, tmp_path: Path
) -> None:
    from surfcensus.func import contours as fc

    _needs(lambda: fc.read_contours(b""))
    runs, _ = ses1(ds)
    grid = Grid(nib.load(str(runs[0].files["pre_bold"])).shape[:3], to_matrix4(GRID_AFFINE))
    surfaces = fc.surfaces_in_display(
        ds.fs_subjects_dir / "sub-01", to_matrix4(FSNATIVE_TO_T1W), grid
    )
    planes = fc.plane_contours(surfaces, grid)
    write_contours(tmp_path, grid.key, planes)
    for plane in planes:
        data = decode_contours(tmp_path, grid.key, plane.plane)
        assert data == plane.data and len(fc.read_contours(data)) > 0


def test_the_stills_keep_their_names(packed) -> None:
    session_dir, runs, _ = packed
    info = read_pack(session_dir, run_id(runs[0]))
    assert info is not None
    for stream in STREAMS:
        for still in STILLS:
            entry = getattr(info.streams[stream], still)
            assert entry.file == f"{run_id(runs[0])}/{stream}/{still}.bin"


# --- the exit check on the worked example (network) -----------------------------------------

DERIV_URL = "https://openneuro-derivatives.s3.amazonaws.com/fmriprep/ds000102-fmriprep/"
RAW_URL = "https://s3.amazonaws.com/openneuro.org/ds000102/"
STEM = "sub-01_task-flanker_run-{n}"
WORKED_FILES: dict[str, tuple[str, int]] = {  # local name -> (url, size from the listing)
    "raw/run-{n}_bold.nii.gz": (RAW_URL + "sub-01/func/" + STEM + "_bold.nii.gz", 0),
    "deriv/run-{n}_space-T1w_desc-preproc_bold.nii.gz": (
        DERIV_URL + "sub-01/func/" + STEM + "_space-T1w_desc-preproc_bold.nii.gz",
        0,
    ),
    "deriv/run-{n}_space-T1w_desc-preproc_bold.json": (
        DERIV_URL + "sub-01/func/" + STEM + "_space-T1w_desc-preproc_bold.json",
        111,
    ),
    "deriv/run-{n}_space-T1w_desc-brain_mask.nii.gz": (
        DERIV_URL + "sub-01/func/" + STEM + "_space-T1w_desc-brain_mask.nii.gz",
        0,
    ),
    "deriv/run-{n}_boldref.nii.gz": (DERIV_URL + "sub-01/func/" + STEM + "_boldref.nii.gz", 0),
    "deriv/run-{n}_desc-confounds_timeseries.tsv": (
        DERIV_URL + "sub-01/func/" + STEM + "_desc-confounds_timeseries.tsv",
        0,
    ),
    "deriv/run-{n}_from-scanner_to-T1w_mode-image_xfm.txt": (
        DERIV_URL + "sub-01/func/" + STEM + "_from-scanner_to-T1w_mode-image_xfm.txt",
        0,
    ),
}
WORKED_SIZES = {  # from the bucket listings of 2026-11-14
    ("raw/run-{n}_bold.nii.gz", 1): 28_061_534,
    ("raw/run-{n}_bold.nii.gz", 2): 28_143_286,
    ("deriv/run-{n}_space-T1w_desc-preproc_bold.nii.gz", 1): 29_422_516,
    ("deriv/run-{n}_space-T1w_desc-preproc_bold.nii.gz", 2): 29_286_639,
    ("deriv/run-{n}_space-T1w_desc-preproc_bold.json", 1): 111,
    ("deriv/run-{n}_space-T1w_desc-preproc_bold.json", 2): 111,
    ("deriv/run-{n}_space-T1w_desc-brain_mask.nii.gz", 1): 2_223,
    ("deriv/run-{n}_space-T1w_desc-brain_mask.nii.gz", 2): 2_177,
    ("deriv/run-{n}_boldref.nii.gz", 1): 289_819,
    ("deriv/run-{n}_boldref.nii.gz", 2): 289_600,
    ("deriv/run-{n}_desc-confounds_timeseries.tsv", 1): 375_255,
    ("deriv/run-{n}_desc-confounds_timeseries.tsv", 2): 367_311,
    ("deriv/run-{n}_from-scanner_to-T1w_mode-image_xfm.txt", 1): 238,
    ("deriv/run-{n}_from-scanner_to-T1w_mode-image_xfm.txt", 2): 230,
}
FSNATIVE = (
    "deriv/anat_from-T1w_to-fsnative_mode-image_xfm.txt",
    DERIV_URL + "sub-01/anat/sub-01_from-T1w_to-fsnative_mode-image_xfm.txt",
    225,
)
PROTOTYPE = Path(__file__).resolve().parents[1] / "prototypes" / "func_viewer" / "stage1"


def _fetch_worked_example(root: Path) -> Path:
    """The functional files of ds000102 sub-01 that a pack reads (112 MB), in the layout of
    stage 1's DATA folder (``deriv/`` and ``raw/``, names cut to ``run-N``), in the pytest
    cache. Anonymous, read-only GETs, at most four at a time; a file already there with the
    size of the listing is not fetched again."""
    from concurrent.futures import ThreadPoolExecutor

    import httpx

    wanted = [(FSNATIVE[0], FSNATIVE[1], FSNATIVE[2])]
    for local, (url, _) in WORKED_FILES.items():
        for n in (1, 2):
            wanted.append((local.format(n=n), url.format(n=n), WORKED_SIZES[(local, n)]))

    def get(item: tuple[str, str, int]) -> None:
        local, url, size = item
        path = root / local
        if path.is_file() and path.stat().st_size == size:
            return
        with httpx.Client(timeout=300.0) as client:
            content = client.get(url).raise_for_status().content
        assert len(content) == size, f"{local}: {len(content)} bytes, expected {size}"
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        part.write_bytes(content)
        part.replace(path)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(get, wanted))
    return root


def _prototype_volumes(pack: Path, run: Mapping[str, Any], stream: str) -> Iterator[bytes]:
    files = run["streams"][stream]
    ref = np.frombuffer(zlib.decompress((pack / files["ref"]["file"]).read_bytes()), np.uint8)
    for chunk in files["chunks"]:
        data = np.frombuffer(zlib.decompress((pack / chunk["file"]).read_bytes()), np.uint8)
        for row in data.reshape(chunk["count"], ref.size):
            yield (row + ref).tobytes()


@pytest.mark.network
def test_the_worked_example_decodes_to_the_values_of_the_prototype_pack(
    request: pytest.FixtureRequest, real_subject: Path, tmp_path: Path
) -> None:
    from surfcensus.func import itk, session

    _geometry_ready()
    _needs(lambda: itk.read_itk(""), lambda: session.raw_affine_choice(Path("/x"), None))
    data = _fetch_worked_example(request.config.cache.mkdir("ds000102-sub-01-func"))

    # stage 1's pack of these files, with numpy's voxel map, made here by its own script
    proto = tmp_path / "prototype"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    subprocess.run(
        [
            sys.executable,
            "-I",
            str(PROTOTYPE / "pack_run.py"),
            str(data),
            str(real_subject),
            str(proto),
            "--id",
            "W01",
            "--no-contours",
        ],
        check=True,
        capture_output=True,
        env=env,
    )
    manifest = json.loads((proto / "manifest.json").read_text())
    assert manifest["registration"]["anchor"] == "R1"  # run-1, by name order
    assert manifest["registration"]["portable_voxel_map"] is False

    anchor_matrix = itk.read_itk_file(data / "deriv/run-1_from-scanner_to-T1w_mode-image_xfm.txt")
    sid = func_id(KEY, "session", "worked example")
    session_dir = tmp_path / "ours" / sid
    config = EpiConfig(portable_voxel_map=False)
    infos = []
    for n in (1, 2):
        files = {
            FuncRole.RAW_BOLD: data / f"raw/run-{n}_bold.nii.gz",
            FuncRole.PRE_BOLD: data / f"deriv/run-{n}_space-T1w_desc-preproc_bold.nii.gz",
            FuncRole.PRE_SIDECAR: data / f"deriv/run-{n}_space-T1w_desc-preproc_bold.json",
            FuncRole.PRE_MASK: data / f"deriv/run-{n}_space-T1w_desc-brain_mask.nii.gz",
            FuncRole.XFM_TO_ANAT: data / f"deriv/run-{n}_from-scanner_to-T1w_mode-image_xfm.txt",
            FuncRole.REFERENCE: data / f"deriv/run-{n}_boldref.nii.gz",
            FuncRole.CONFOUNDS: data / f"deriv/run-{n}_desc-confounds_timeseries.tsv",
        }
        proto_run = manifest["runs"][n - 1]
        raw_affine = session.raw_affine_choice(files[FuncRole.RAW_BOLD], files[FuncRole.REFERENCE])
        assert raw_affine.choice == proto_run["raw_header"]["affine"] == "header"
        own = itk.read_itk_file(files[FuncRole.XFM_TO_ANAT])
        grid = geometry.grid_of(files[FuncRole.PRE_BOLD])
        inp = RunPackInput(
            run=func_id(KEY, "run", str(n)),
            name=f"task-flanker_run-{n}",
            files=files,
            grid=grid,
            display_to_anat=to_matrix4(np.eye(4)),
            anchor_run=func_id(KEY, "run", "1"),
            anchor_matrix=anchor_matrix,
            raw_affine=raw_affine,
            motion=Motion(
                fd=tuple(proto_run["fd"]),
                dvars=tuple(proto_run["dvars"]),
                non_steady=tuple(proto_run["non_steady"]),
                source=FuncRole.CONFOUNDS,
            ),
            corrections=CORRECTIONS,
            rigid=RigidPath(own, 6, 0.0, True, FuncRole.XFM_TO_ANAT, FuncRole.REFERENCE),
            tr_s=proto_run["tr_s"],
        )
        # the prototype has a window per run; D38 has one per session, so each run is packed
        # here with the prototype's window of it passed in
        windows = (tuple(proto_run["window"]), tuple(proto_run["sd_window"]))
        info = pack_run(session_dir, inp, config, *windows)
        infos.append(info)
        # alone, a run is a session of one, whose window is the prototype's of that run
        assert session_windows([inp]) == windows
        assert info.n_volumes == proto_run["n_volumes"] == 146
        assert list(info.window) == proto_run["window"]
        assert list(info.sd_window) == proto_run["sd_window"]
        assert info.stills_volumes == proto_run["stills_volumes"]
        assert info.chunk_volumes == manifest["chunk_volumes"]
        for stream in STREAMS:
            ours = list(packmod.stream_bytes(session_dir, info, stream))
            theirs = list(_prototype_volumes(proto, proto_run, stream))
            assert len(ours) == len(theirs) == 146
            differ = [t for t in range(146) if ours[t] != theirs[t]]
            assert differ == [], (n, stream, differ[:5])
            for still in STILLS:
                mine = zlib.decompress(
                    (session_dir / getattr(info.streams[stream], still).file).read_bytes()
                )
                assert mine == zlib.decompress(
                    (proto / proto_run["streams"][stream][still]["file"]).read_bytes()
                ), (n, stream, still)
    # and a second pack of the same files has the same digest
    again = pack_run(tmp_path / "again" / sid, inp, config, *windows)
    assert again.content_digest == infos[1].content_digest
