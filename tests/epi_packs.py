"""Synthetic packs of one subject for the tests of the EPI page and its server (work package
E1.D), made at run time from ``tests/synth_func.py`` with the package's own session, geometry,
contour and pack code (``func.session.check_frame``, ``func.contours``, ``func.pack``).

:func:`make_stream` writes, under ``root``:

* the synthetic BIDS dataset of ``synth_func`` (``root/func``), with ``n_volumes`` volumes per
  run, and beside it ``root/extra``: the files this module adds, which are copies of fixture
  files changed in one way each (a source folder is never written into);
* ``root/packs``: the packs of ``sub-01`` as ``surfcensus epi-session --out`` would leave them,
  ``stream.json`` last.

The stream (decision D38) is two sessions of one subject:

* session A, ``sub-01 ses-1``: its four runs of the fixture in acquisition order (the anchor
  ``task-rest run-2`` first) and a fifth run on a grid of its own, ``task-motor run-1``
  resampled to 7 mm voxels on a shifted field of view (:data:`GRID_C_AFFINE`), so the session
  holds runs on two grids, each with its own real contours. ``task-rest run-1`` is made 1.6
  times brighter in both its raw and its preprocessed series (a gain change), so that the
  normalization toggle shows. ``task-rest run-3`` lies 12 mm from the anchor and is flagged by
  the real frame check; its raw affine is recorded as unchecked and its report says
  distortion correction was applied.
* session B, ``sub-01 ses-2``: its three runs, packed without contours (no FreeSurfer
  subject), and a fourth listed with the reason ``no raw series`` and not packed.

Chunks are made small (``chunk_bytes`` of :data:`CHUNK_VOLUMES` volumes of the 8 mm grid), so
every run has several, and a memory limit of 1 MiB evicts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from synth_func import (  # pyrefly: ignore[missing-import]
    FSNATIVE_TO_T1W,
    GRID_AFFINE,
    SyntheticFuncDataset,
    SyntheticRun,
    apply,
    brain_mask,
    float32_affine,
    series_value,
    voxel_centres,
    write_func_dataset,
)

from surfcensus import __version__
from surfcensus.config import EpiConfig
from surfcensus.func import STREAM_JSON, func_id
from surfcensus.func import contours as fc
from surfcensus.func.model import SessionInfo, StreamInfo, info_text
from surfcensus.func.pack import pack_session
from surfcensus.func.session import check_frame
from surfcensus.records import (
    Anchor,
    Corrections,
    DerivRun,
    Display,
    Grid,
    Motion,
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

KEY = b"the key of the EPI page tests"
CHUNK_VOLUMES = 4
"""Volumes per chunk on the 8 mm grid (4,284 voxels)."""
BRIGHTER = 1.6
"""The gain of the brighter run."""
GRID_C_SHAPE = (19, 20, 16)
GRID_C_CENTRE = (2.0, -21.0, 17.0)
GRID_C_VOXEL = 7.0


def _grid_c_affine() -> np.ndarray:
    affine = np.diag([GRID_C_VOXEL, GRID_C_VOXEL, GRID_C_VOXEL, 1.0])
    affine[:3, 3] = np.asarray(GRID_C_CENTRE) - GRID_C_VOXEL * (np.asarray(GRID_C_SHAPE) - 1) / 2
    return float32_affine(affine)


GRID_C_AFFINE = _grid_c_affine()
"""The grid of the fifth run of session A: 7 mm voxels, its middle 2 mm right, 3 mm back and
1 mm up of the fixture's grid (whose middle is :data:`synth_func.BRAIN_CENTRE`)."""


@dataclass(frozen=True)
class PackedRun:
    run: str
    """The run id."""
    session: str
    """The session id."""
    source: SyntheticRun
    grid: Grid
    brighter: bool = False


@dataclass(frozen=True)
class PackedStream:
    out: Path
    """The ``--out`` directory: ``stream.json`` and one directory per session."""
    dataset: SyntheticFuncDataset
    sessions: tuple[str, ...]
    infos: Mapping[str, SessionInfo]
    runs: tuple[PackedRun, ...]
    """The packed runs in stream order."""
    not_packed: tuple[str, ...]
    """Run ids listed with a reason."""


def run_id(stem: str, extra: str = "") -> str:
    return func_id(KEY, "run", stem, extra)


def session_id(participant: str, session: str | None) -> str:
    return func_id(KEY, "session", participant, session or "")


def _motion(run: SyntheticRun) -> Motion:
    rows = run.files["confounds"].read_text().splitlines()
    names = rows[0].split("\t")

    def column(name: str) -> tuple[float | None, ...]:
        i = names.index(name)
        values = [r.split("\t")[i] for r in rows[1:]]
        return tuple(None if v == "n/a" else float(v) for v in values)

    outlier = column("non_steady_state_outlier00")
    return Motion(
        fd=column("framewise_displacement"),
        dvars=column("std_dvars"),
        non_steady=tuple(t for t, v in enumerate(outlier) if v),
        source=FuncRole.CONFOUNDS,
    )


def _corrections(sdc: bool) -> Corrections:
    report = FuncRole.SUMMARY_REPORT
    return Corrections(
        motion=Statement("applied", None, None),
        sdc=Statement("applied", "FLB", report)
        if sdc
        else Statement("not applied", "None", report),
        sdc_method="FLB (fieldmap-less, SyN-based)" if sdc else None,
        slice_timing=Statement("not applied", "Not applied", report),
        registration=Statement("applied", "bbregister", report),
        dof=6,
        sdc_figure_listed=sdc,
    )


def _write_series(
    path: Path, shape: Sequence[int], affine: np.ndarray, n: int, gain: float
) -> None:
    points = apply(affine, voxel_centres(shape))
    volumes = [(gain * series_value(points, t)).reshape(tuple(shape), order="F") for t in range(n)]
    img = nib.Nifti1Image(np.stack(volumes, axis=-1).astype(np.float32), affine)
    img.set_sform(affine, code=1)
    img.set_qform(affine, code=1)
    img.header.set_zooms((*img.header.get_zooms()[:3], 2.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, path)


def _scaled_copy(source: Path, target: Path, gain: float) -> None:
    img = nib.load(str(source))
    data = np.asarray(img.dataobj, dtype=np.float32) * np.float32(gain)
    copy = nib.Nifti1Image(data, img.affine, img.header)
    target.parent.mkdir(parents=True, exist_ok=True)
    nib.save(copy, target)


def _contours(ds: SyntheticFuncDataset, grid: Grid) -> tuple:
    surfaces = fc.surfaces_in_display(
        ds.fs_subjects_dir / "sub-01", to_matrix4(FSNATIVE_TO_T1W), grid
    )
    return fc.plane_contours(surfaces, grid)


def _deriv(run: SyntheticRun) -> DerivRun:
    d = run.design
    return DerivRun(RunEntities(sub=d.participant[4:]), run.key, (), run.stem, ("T1w",), {})


def _plan(
    participant: str,
    session: str,
    items: Sequence[tuple[str, str, SyntheticRun, str | None]],
    grids: Mapping[str, Grid],
    grid_of: Mapping[str, str],
    anchor: str,
    frame_inputs: Mapping[str, tuple[SyntheticRun, Path, Grid]],
    fs: str | None,
) -> SessionPlan:
    runs = tuple(
        SessionRun(rid, name, _deriv(src), None, Display("anat", "T1w"), reason)
        for rid, name, src, reason in items
    )
    transforms = {rid: to_matrix4(src.anat_to_scanner) for rid, (src, _, _) in frame_inputs.items()}
    masks = {
        rid: (np.asarray(nib.load(str(mask)).dataobj) > 0, np.asarray(grid.affine, float))
        for rid, (_, mask, grid) in frame_inputs.items()
    }
    frame = check_frame(transforms, anchor, masks)
    return SessionPlan(
        participant=participant,
        session=session,
        fs_subject=fs,
        runs=runs,
        anchor=Anchor(anchor, "acquisition_time", "the first by time"),
        frame=frame,
        grids=grids,
        grid_of=grid_of,
    )


def _input(
    rid: str,
    src: SyntheticRun,
    anchor: SyntheticRun,
    anchor_id: str,
    grid: Grid,
    files: Mapping[FuncRole, Path],
    *,
    unchecked: bool = False,
    sdc: bool = False,
) -> RunPackInput:
    roles = {FuncRole(k): v for k, v in src.files.items() if k in FuncRole.__members__.values()}
    roles |= dict(files)
    return RunPackInput(
        run=rid,
        name=src.name,
        files=roles,
        grid=grid,
        display_to_anat=to_matrix4(np.eye(4)),
        anchor_run=anchor_id,
        anchor_matrix=to_matrix4(anchor.anat_to_scanner),
        raw_affine=RawAffine(
            to_matrix4(src.raw_affine),
            "header, unchecked" if unchecked else "header",
            not unchecked,
        ),
        motion=_motion(src),
        corrections=_corrections(sdc),
        rigid=RigidPath(to_matrix4(src.anat_to_scanner), 6, 0.0, True, FuncRole.XFM_TO_ANAT, None),
        tr_s=2.0,
    )


def make_stream(root: Path, *, n_volumes: int = 24) -> PackedStream:
    """Write the dataset, the extra files and the packs under ``root`` (see the module)."""
    root = Path(root)
    ds = write_func_dataset(root / "func", n_volumes=n_volumes)
    extra = root / "extra"
    out = root / "packs"
    config = EpiConfig(chunk_bytes=CHUNK_VOLUMES * int(np.prod((17, 18, 14))))
    grid_a = Grid((17, 18, 14), to_matrix4(GRID_AFFINE))
    grid_c = Grid(GRID_C_SHAPE, to_matrix4(GRID_C_AFFINE))
    subject = ds.subjects["sub-01"]

    # session A
    sa = ds.session("sub-01", "1")
    by_stem = {r.stem: r for r in ds.runs}
    a_runs = [by_stem[stem] for stem in sa.order]
    anchor_a = by_stem[sa.anchor]
    aid = session_id("sub-01", "1")
    anchor_a_id = run_id(anchor_a.stem)
    items: list[tuple[str, str, SyntheticRun, str | None]] = []
    inputs: list[RunPackInput] = []
    packed: list[PackedRun] = []
    frame_inputs: dict[str, tuple[SyntheticRun, Path, Grid]] = {}
    grid_of: dict[str, str] = {}
    for src in a_runs:
        rid = run_id(src.stem)
        files: dict[FuncRole, Path] = {}
        bright = src.design.case.startswith("rotated")
        if bright:
            for role in (FuncRole.RAW_BOLD, FuncRole.PRE_BOLD):
                target = extra / "bright" / Path(src.files[role.value]).name
                _scaled_copy(src.files[role.value], target, BRIGHTER)
                files[role] = target
        flagged = src.design.case.startswith("displaced 12")
        items.append((rid, src.name, src, None))
        inputs.append(
            _input(rid, src, anchor_a, anchor_a_id, grid_a, files, unchecked=flagged, sdc=flagged)
        )
        frame_inputs[rid] = (src, src.files["pre_mask"], grid_a)
        grid_of[rid] = grid_a.key
        packed.append(PackedRun(rid, aid, src, grid_a, bright))
    # the fifth run: task-motor run-1 again, on grid C
    motor = next(r for r in a_runs if r.design.task == "motor")
    cid = run_id(motor.stem, "grid C")
    c_dir = extra / "gridC"
    c_pre = c_dir / f"{motor.stem}_space-T1w_desc-preproc_bold.nii.gz"
    c_mask = c_dir / f"{motor.stem}_space-T1w_desc-brain_mask.nii.gz"
    _write_series(c_pre, GRID_C_SHAPE, GRID_C_AFFINE, n_volumes, 1.0)
    mask = brain_mask(subject, GRID_C_SHAPE, GRID_C_AFFINE).astype(np.uint8)
    mask_img = nib.Nifti1Image(mask, GRID_C_AFFINE)
    mask_img.set_sform(GRID_C_AFFINE, code=1)
    mask_img.set_qform(GRID_C_AFFINE, code=1)
    nib.save(mask_img, c_mask)
    items.append((cid, f"{motor.name}_grid-C", motor, None))
    inputs.append(
        _input(
            cid,
            motor,
            anchor_a,
            anchor_a_id,
            grid_c,
            {FuncRole.PRE_BOLD: c_pre, FuncRole.PRE_MASK: c_mask},
        )
    )
    frame_inputs[cid] = (motor, c_mask, grid_c)
    grid_of[cid] = grid_c.key
    packed.append(PackedRun(cid, aid, motor, grid_c))
    plan_a = _plan(
        "sub-01",
        "1",
        items,
        {grid_a.key: grid_a, grid_c.key: grid_c},
        grid_of,
        anchor_a_id,
        frame_inputs,
        "sub-01",
    )
    contours = {grid_a.key: _contours(ds, grid_a), grid_c.key: _contours(ds, grid_c)}
    info_a = pack_session(out, aid, plan_a, inputs, contours, config)

    # session B, without contours, with a run that is listed and not packed
    sb = ds.session("sub-01", "2")
    b_runs = [by_stem[stem] for stem in sb.order]
    anchor_b = by_stem[sb.anchor]
    bid = session_id("sub-01", "2")
    anchor_b_id = run_id(anchor_b.stem)
    items = []
    inputs = []
    frame_inputs = {}
    grid_of = {}
    for src in b_runs:
        rid = run_id(src.stem)
        items.append((rid, src.name, src, None))
        inputs.append(_input(rid, src, anchor_b, anchor_b_id, grid_a, {}))
        frame_inputs[rid] = (src, src.files["pre_mask"], grid_a)
        grid_of[rid] = grid_a.key
        packed.append(PackedRun(rid, bid, src, grid_a))
    missing = run_id("no raw series of ses-2")
    items.append((missing, "ses-2_task-rest_run-9", b_runs[0], "no raw series"))
    plan_b = _plan(
        "sub-01", "2", items, {grid_a.key: grid_a}, grid_of, anchor_b_id, frame_inputs, None
    )
    info_b = pack_session(out, bid, plan_b, inputs, None, config)

    stream = StreamInfo(surfcensus_version=__version__, label="sub-01", sessions=(aid, bid))
    (out / STREAM_JSON).write_text(info_text(stream))
    return PackedStream(
        out=out,
        dataset=ds,
        sessions=(aid, bid),
        infos={aid: info_a, bid: info_b},
        runs=tuple(packed),
        not_packed=(missing,),
    )


def stream_order(stream: PackedStream) -> list[tuple[str, str, int, int]]:
    """``(session, run, first in the subject's stream, volumes)`` of every packed run, from the
    ``session.json`` files: the order of the page's timeline."""
    out = []
    at = 0
    for sid in stream.sessions:
        for run in stream.infos[sid].runs:
            if run.n_volumes is None:
                continue
            out.append((sid, run.run, at, run.n_volumes))
            at += run.n_volumes
    return out
