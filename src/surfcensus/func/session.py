"""One registration per session: the anchor, the frame check, the raw affine, and the plan
of a session. Work package E1.B.

Decision of Tomas Knapen, 2026-11-13: "Raw bids means single per session registration to the
anatomy to ensure comparability across sessions, processing stages, and subjects." The raw
stream of every run of a session is drawn through ONE stored transform, that of the session's
anchor run (D36). These are statements about the raw data, recorded per run and shown:

* The anchor (:func:`pick_anchor`): the first run acquired, by ``AcquisitionTime`` of the raw
  sidecars when every candidate has a readable one, else the first in name order.
* The frame check (:func:`check_frame`): per run, the displacement over its brain mask between
  its own stored transform and the anchor's; flagged above ``frame_limit_mm`` (10 mm) somewhere
  in the mask; ``anchor_is_outlier`` when every other run of at least two is flagged and they
  agree with each other.
* The raw affine (:func:`raw_affine_choice`): the one that agrees, to 1e-4, with the run's
  BOLD-space reference image: the header's ``img.affine``, else the qform, else the reference
  image's own affine; without a usable reference image ``img.affine``, unchecked.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import nibabel as nib
import numpy as np
from jaxtyping import Bool, Float

from surfcensus.func import bids
from surfcensus.func.connectors.base import Connector
from surfcensus.func.geometry import compose, grid_of, inverse, same_grid
from surfcensus.records import (
    MULTI_ECHO,
    NATIVE_SPACE_ONLY,
    NO_BRAIN_MASK,
    NO_PREPROCESSED_SERIES,
    NO_RAW_SERIES,
    NO_TRANSFORM,
    VOLUME_COUNTS_DIFFER,
    AffineChoice,
    Anchor,
    AnchorCandidate,
    AnchorRule,
    DerivRun,
    FrameCheck,
    FrameDisplacement,
    FuncFileRef,
    Grid,
    Matrix4,
    RawAffine,
    RawRun,
    RigidPath,
    SessionPlan,
    SessionRun,
    to_matrix4,
)
from surfcensus.sources.base import PACK_REQUIRED, FuncRole

RULE_TIME: AnchorRule = "acquisition_time"
RULE_NAME: AnchorRule = "name_order"

AFFINE_HEADER: AffineChoice = "header"
AFFINE_QFORM: AffineChoice = "qform"
AFFINE_REFERENCE: AffineChoice = "reference"
AFFINE_UNCHECKED: AffineChoice = "header, unchecked"
AFFINE_TOLERANCE = 1e-4
"""Two affines agree when every entry is within this (absolute)."""

FRAME_LIMIT_MM = 10.0
"""The default of ``EpiConfig.frame_limit_mm``; where it comes from is in stage 1's
``session.frame_flag``."""


# --- the anchor ---------------------------------------------------------------------------


def _by_name(runs: Sequence[AnchorCandidate]) -> list[AnchorCandidate]:
    return sorted(runs, key=lambda r: (r.order, r.run))


def _seconds(runs: Sequence[AnchorCandidate]) -> tuple[dict[str, float] | None, str]:
    """The acquisition time of every run in seconds, or None and why the time rule does not
    apply."""
    times = [bids.acquisition_seconds(r.acquisition_time) for r in runs]
    n_read = sum(t is not None for t in times)
    if n_read < len(runs):
        n_given = sum(r.acquisition_time is not None for r in runs)
        return None, (
            f"AcquisitionTime given for {n_given} and readable for {n_read} of {len(runs)} runs"
        )
    if len({t[0] for t in times if t is not None}) > 1:
        return None, "AcquisitionTime is a time of day in some runs and a date and time in others"
    return {r.run: t[1] for r, t in zip(runs, times, strict=True) if t is not None}, ""


def pick_anchor(runs: Sequence[AnchorCandidate]) -> Anchor:
    """The anchor of ONE session among the runs that have a stored transform. Runs with equal
    times go by name order. Raises ``ValueError`` for no runs.

    The time rule applies when every run has an ``AcquisitionTime`` that
    ``func.bids.acquisition_seconds`` reads, all of one kind (times of day, or dates and
    times); the anchor is then the earliest. Otherwise the anchor is the first run in name
    order (``AnchorCandidate.order``, then the run id). ``why`` says what the rule saw; under
    the time rule it holds the anchor's time as written."""
    if not runs:
        raise ValueError("a session without runs has no anchor")
    if len({r.run for r in runs}) != len(runs):
        raise ValueError("a run is given twice")
    by_name = _by_name(runs)
    seconds, why = _seconds(runs)
    if seconds is None:
        return Anchor(run=by_name[0].run, rule=RULE_NAME, why=why)
    first = min(by_name, key=lambda r: seconds[r.run])  # min keeps the first of equal times
    return Anchor(
        run=first.run,
        rule=RULE_TIME,
        why=f"earliest AcquisitionTime of {len(runs)} runs ({first.acquisition_time})",
    )


def acquisition_order(runs: Sequence[AnchorCandidate], anchor: Anchor) -> tuple[str, ...]:
    """The run ids in acquisition order with the anchor first: by time under the time rule,
    else by name order; ties by name order.

    This is the order of the session's one continuous stream (D38), so it is the acquisition
    order exactly as the anchor rule defines it, and nothing is moved: the anchor is first
    because the rule that chose it is the rule that orders the runs (the earliest time, or the
    first name), not because it is put there. Under the time rule every run has a readable
    time (else the rule would not apply); a run without one goes after the timed ones, in
    name order. Raises ``ValueError`` when the anchor is not among the runs, or when the
    anchor given is not the first of the order its rule gives (an anchor of another rule or of
    other runs)."""
    if anchor.run not in {r.run for r in runs}:
        raise ValueError("the anchor is not among the runs")
    by_name = _by_name(runs)
    if anchor.rule == RULE_TIME:
        rank = {r.run: n for n, r in enumerate(by_name)}

        def when(r: AnchorCandidate) -> tuple[int, float, int]:
            read = bids.acquisition_seconds(r.acquisition_time)
            return (0, read[1], rank[r.run]) if read else (1, 0.0, rank[r.run])

        ordered = sorted(by_name, key=when)
    else:
        ordered = by_name
    order = tuple(r.run for r in ordered)
    if order[0] != anchor.run:
        raise ValueError("the anchor is not the first run in the order of its rule")
    return order


# --- the raw affine -----------------------------------------------------------------------


def _agree(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(np.abs(np.asarray(a, float) - np.asarray(b, float)) <= AFFINE_TOLERANCE))


def raw_affine_choice(raw_bold: Path, reference: Path | None) -> RawAffine:
    """Which affine the raw stream of a run uses; ``reference`` is the run's BOLD-space
    reference image, or None. A reference on another grid than the raw series leaves the
    choice unchecked.

    The stored transform maps to the scanner world of fMRIPrep's reference image of the run,
    which fMRIPrep made after validating the raw header and which keeps the sform or the qform
    of the raw header when the two differ. So the reference image decides: the header's
    ``img.affine`` (nibabel's sform-first affine) when the reference's affine agrees with it to
    :data:`AFFINE_TOLERANCE` in every entry; else the qform (when its code is not 0) when that
    agrees; else the reference image's own affine. Without a reference image, or with one whose
    first three dimensions differ from the raw series', ``img.affine``, recorded as
    ``header, unchecked``. Reads the headers only."""
    img = nib.load(str(raw_bold))
    header = np.asarray(img.affine, dtype=np.float64)
    if reference is None:
        return RawAffine(to_matrix4(header), AFFINE_UNCHECKED, reference_image=False)
    ref = nib.load(str(reference))
    if tuple(ref.shape[:3]) != tuple(img.shape[:3]):
        return RawAffine(to_matrix4(header), AFFINE_UNCHECKED, reference_image=True)
    ref_affine = np.asarray(ref.affine, dtype=np.float64)
    if _agree(ref_affine, header):
        return RawAffine(to_matrix4(header), AFFINE_HEADER, reference_image=True)
    get_qform = getattr(img.header, "get_qform", None)
    qform = get_qform(coded=True)[0] if get_qform is not None else None  # None for code 0
    if qform is not None and _agree(ref_affine, qform):
        return RawAffine(to_matrix4(qform), AFFINE_QFORM, reference_image=True)
    return RawAffine(to_matrix4(ref_affine), AFFINE_REFERENCE, reference_image=True)


# --- the frame check ----------------------------------------------------------------------


def _relative(run_matrix: Matrix4, anchor_matrix: Matrix4) -> Matrix4:
    """``inv(S_a) . S_r``, in Python floats."""
    return compose(inverse(anchor_matrix), run_matrix)


def frame_displacement(
    run_matrix: Matrix4,
    anchor_matrix: Matrix4,
    mask: Bool[np.ndarray, "i j k"],
    mask_affine: Float[np.ndarray, "4 4"],
    limit_mm: float = FRAME_LIMIT_MM,
) -> FrameDisplacement:
    """How far a run lies, drawn through the anchor's transform, from where its own puts it.

    Both matrices are point maps from the anatomy's world to a scanner world
    (``RigidPath.matrix``). The displacement ``q - p`` with ``q = inv(S_a) S_r p`` is taken at
    the centres of the mask's voxels; rotation about the mask's centroid, translation of the
    centroid, and the shear and scale of ``inv(S_a) S_r``. Raises ``ValueError`` for an empty
    mask.

    ``flagged``: the largest displacement exceeds ``limit_mm`` (a run at the limit exactly is
    not flagged). The matrix ``inv(S_a) S_r`` is formed in Python floats and the displacements
    are computed entry by entry, as ``(d - I) p + t`` per axis, so that a pure shift gives its
    length exactly and no number depends on numpy's linear algebra; the rotation and the shear
    and scale come from a singular value decomposition of the 3x3 part (the rotation nearest
    it, and the largest deviation of a singular value from 1)."""
    d = _relative(run_matrix, anchor_matrix)
    ijk = np.argwhere(np.asarray(mask)).astype(np.float64)
    if not len(ijk):
        raise ValueError("the brain mask is empty")
    a = np.asarray(mask_affine, dtype=np.float64)
    p = [
        ijk[:, 0] * a[r, 0] + ijk[:, 1] * a[r, 1] + ijk[:, 2] * a[r, 2] + a[r, 3] for r in range(3)
    ]
    e = [[d[r][c] - (1.0 if r == c else 0.0) for c in range(3)] for r in range(3)]
    diff = [p[0] * e[r][0] + p[1] * e[r][1] + p[2] * e[r][2] + d[r][3] for r in range(3)]
    dist = np.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
    three = np.array([row[:3] for row in d[:3]], dtype=np.float64)
    u, s, vt = np.linalg.svd(three)
    nearest = u @ vt
    w = (
        float(nearest[2, 1] - nearest[1, 2]),
        float(nearest[0, 2] - nearest[2, 0]),
        float(nearest[1, 0] - nearest[0, 1]),
    )
    trace = float(nearest[0, 0] + nearest[1, 1] + nearest[2, 2])
    rotation = math.degrees(math.atan2(math.sqrt(sum(x * x for x in w)), trace - 1.0))
    centroid = [float(np.mean(axis)) for axis in p]
    shift = tuple(sum(e[r][c] * centroid[c] for c in range(3)) + d[r][3] for r in range(3))
    max_mm = float(dist.max())
    return FrameDisplacement(
        mean_mm=float(dist.mean()),
        max_mm=max_mm,
        rotation_deg=rotation,
        translation_mm=math.sqrt(sum(x * x for x in shift)),
        translation_ras=(shift[0], shift[1], shift[2]),
        shear_scale=float(np.abs(s - 1.0).max()),
        n_voxels=len(ijk),
        flagged=max_mm > limit_mm,
    )


def check_frame(
    transforms: Mapping[str, Matrix4],
    anchor: str,
    masks: Mapping[str, tuple[Bool[np.ndarray, "i j k"], Float[np.ndarray, "4 4"]]],
    limit_mm: float = FRAME_LIMIT_MM,
) -> FrameCheck:
    """The frame check of one session: ``transforms`` and ``masks`` by run id, for every
    packable run; the anchor's own entry is all zeros.

    Each run's displacement is taken over its own brain mask. ``shared_frame_plausible``: no
    run is flagged. ``anchor_is_outlier``: at least two runs besides the anchor, every one of
    them flagged, and no pair of them flagged against each other (over the mask of the first
    of the pair in the order of ``transforms``): then the anchor, or its stored transform, is
    the odd one, and the whole session is drawn off the anatomy. Raises ``ValueError`` when the
    anchor or a run's mask is missing."""
    if anchor not in transforms:
        raise ValueError("the anchor is not among the runs")
    missing = [run for run in transforms if run not in masks]
    if missing:
        raise ValueError(f"no brain mask for {len(missing)} of the runs")
    runs: dict[str, FrameDisplacement] = {}
    for run, own in transforms.items():
        mask, affine = masks[run]
        found = frame_displacement(own, transforms[anchor], mask, affine, limit_mm)
        if run == anchor:
            found = FrameDisplacement(
                0.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0), 0.0, found.n_voxels, False
            )
        runs[run] = found
    others = [run for run in transforms if run != anchor]
    flagged = tuple(run for run in others if runs[run].flagged)
    outlier = False
    if len(others) >= 2 and len(flagged) == len(others):
        outlier = all(
            not frame_displacement(transforms[a], transforms[b], *masks[a], limit_mm).flagged
            for i, a in enumerate(others)
            for b in others[i + 1 :]
        )
    return FrameCheck(
        anchor=anchor,
        limit_mm=float(limit_mm),
        runs=runs,
        flagged=flagged,
        shared_frame_plausible=not flagged,
        anchor_is_outlier=outlier,
    )


# --- the plan of a session ----------------------------------------------------------------

_MISSING_REASONS: tuple[tuple[FuncRole, str], ...] = (
    (FuncRole.RAW_BOLD, NO_RAW_SERIES),
    (FuncRole.PRE_BOLD, NO_PREPROCESSED_SERIES),
    (FuncRole.XFM_TO_ANAT, NO_TRANSFORM),
    (FuncRole.PRE_MASK, NO_BRAIN_MASK),
    # a run without its confounds table is a run without the full preprocessed output
    # (fMRIPrep writes the table at level ``full`` only); no reason of its own exists
    (FuncRole.CONFOUNDS, NO_PREPROCESSED_SERIES),
)
"""The reason of a missing role of ``PACK_REQUIRED``, first missing first."""


def _n_volumes(path: Path) -> int:
    """The volumes of a NIfTI series from its header: the fourth dimension, 1 for a 3D one."""
    shape = nib.load(str(path)).shape
    return int(shape[3]) if len(shape) > 3 else 1


def _n_rows(path: Path) -> int:
    """The rows of a BIDS TSV table below its header line; lines that are empty are not
    rows."""
    with Path(path).open(encoding="utf-8") as stream:
        lines = [line for line in stream.read().splitlines() if line.strip()]
    return max(0, len(lines) - 1)


def _mask(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    if data.ndim > 3:
        data = data.reshape(data.shape[:3])
    return data > 0, np.asarray(img.affine, dtype=np.float64)


def _paths(
    files: Mapping[FuncRole, FuncFileRef], deriv_root: Path, raw_root: Path | None
) -> dict[FuncRole, Path]:
    """Local paths of the listed files: a raw file under ``raw_root``, a derivative file under
    ``deriv_root``; a raw file without a raw root is left out."""
    out: dict[FuncRole, Path] = {}
    for role, ref in files.items():
        if ref.base == "raw":
            if raw_root is not None:
                out[role] = Path(raw_root) / ref.location
        else:
            out[role] = Path(deriv_root) / ref.location
    return out


def _name(run: DerivRun) -> str:
    """The BIDS name without the subject: the derivative's stem after its ``sub-`` part."""
    first, _, rest = run.stem.partition("_")
    return rest if first.startswith("sub-") and rest else run.stem


def plan_session(
    participant: str,
    session: str | None,
    deriv_runs: Sequence[DerivRun],
    raw_runs: Sequence[RawRun],
    connector: Connector,
    deriv_root: Path,
    raw_root: Path | None,
    run_ids: Mapping[str, str],
    fs_subject: str | None,
    frame_limit_mm: float = FRAME_LIMIT_MM,
) -> SessionPlan:
    """The plan of one session from the listings: runs paired by ``key``, the reason of every
    run that cannot be packed, the anchor among the packable ones, the frame check, the grids.

    ``deriv_runs`` and ``raw_runs`` are those of the session (``entities.ses`` equal to
    ``session``); ``run_ids`` maps a run ``key`` to its id. Reads the raw sidecars, the stored
    transforms, the brain masks and the headers of the series (for volume counts and grids);
    no series is read beyond its header. A run is packable when it has every role of
    ``PACK_REQUIRED``, the connector gives it the ``anat`` layout, its raw and preprocessed
    series and its confounds table have one volume count, and it lies on the session's grid.

    Where the docstring above is read more closely:

    * The runs are those of the derivative; a raw series without a derivative run has nothing
      to be shown beside and is not in the plan. A raw run pairs with the derivative run of
      the same ``key``; several raw runs of one key (the echoes of a multi-echo acquisition),
      a raw run with an ``echo`` entity, or ``DerivRun.echoes`` above 1 make the run
      ``multi-echo``, and ``SessionRun.raw`` is the first of them in name order.
    * The reason of a run that cannot be packed is the first that applies of: what
      ``connector.display`` says (``native space only`` for the ``native`` layout, which
      step 1 does not pack); ``multi-echo``; the first missing role of ``PACK_REQUIRED`` (raw
      series, preprocessed series, transform, brain mask, confounds table, the last read as
      ``no preprocessed series``); a brain mask on another grid than the series
      (``no brain mask``); volume counts that differ.
    * Runs whose preprocessed series lie on different grids are all packed: each distinct
      grid (equal shape, affines within ``geometry.GRID_TOLERANCE``) is a group of its own,
      named in ``grids`` and ``grid_of`` by the key of its first run in acquisition order. No
      run gets ``runs on different grids``.
    * ``runs`` holds the packable runs first, in :func:`acquisition_order`, which is the order
      of the session's one continuous stream (D38), then the others in name order.
    * The session's own files come from ``connector.session_files``, with the match that
      ``connector.detect`` gives for ``deriv_root`` (empty when it gives none).
    * An unreadable file raises (``OSError``, nibabel's errors, ``ConnectorError``): the plan
      of a session is all or nothing.
    """
    raw_by_key: dict[str, list[RawRun]] = {}
    for raw in sorted(raw_runs, key=lambda r: (r.order, r.stem)):
        raw_by_key.setdefault(raw.key, []).append(raw)

    def run_id(run: DerivRun) -> str:
        found = run_ids.get(run.key)
        if found is None:
            raise ValueError(f"no id for the run {_name(run)}")
        return found

    listed: list[SessionRun] = []
    packable: dict[str, tuple[SessionRun, dict[FuncRole, Path], Grid]] = {}
    for run in sorted(deriv_runs, key=lambda r: (r.order, r.stem)):
        echoes = raw_by_key.get(run.key, [])
        raw = echoes[0] if echoes else None
        files = dict(run.files)
        if raw is not None:
            files.update({role: ref for role, ref in raw.files.items() if role not in files})
        paths = _paths(files, deriv_root, raw_root)
        missing = tuple(role for role in PACK_REQUIRED if role not in paths)
        display = connector.display(run)
        reason: str | None = None
        multi = len(echoes) > 1 or run.echoes > 1 or any(r.entities.echo for r in echoes)
        if display.layout is None:
            reason = display.reason or NO_PREPROCESSED_SERIES
        elif display.layout == "native":
            reason = NATIVE_SPACE_ONLY
        elif multi:
            reason = MULTI_ECHO
        else:
            reason = next((why for role, why in _MISSING_REASONS if role in missing), None)
        grid: Grid | None = None
        if reason is None:
            grid = grid_of(paths[FuncRole.PRE_BOLD])
            if not same_grid(grid, grid_of(paths[FuncRole.PRE_MASK])):
                reason = NO_BRAIN_MASK
        if reason is None:
            counts = {
                _n_volumes(paths[FuncRole.RAW_BOLD]),
                _n_volumes(paths[FuncRole.PRE_BOLD]),
                _n_rows(paths[FuncRole.CONFOUNDS]),
            }
            if len(counts) != 1:
                reason = VOLUME_COUNTS_DIFFER
        item = SessionRun(
            run=run_id(run),
            name=_name(run),
            deriv=run,
            raw=raw,
            display=display,
            reason=reason,
            missing=missing,
        )
        listed.append(item)
        if reason is None and grid is not None:
            packable[item.run] = (item, paths, grid)

    anchor = None
    frame = None
    grids: dict[str, Grid] = {}
    grid_of_run: dict[str, str] = {}
    order: tuple[str, ...] = ()
    if packable:
        candidates = [
            AnchorCandidate(
                run=run,
                order=item.deriv.order,
                acquisition_time=(
                    bids.sidecar_time(paths[FuncRole.RAW_SIDECAR])
                    if FuncRole.RAW_SIDECAR in paths
                    else None
                ),
            )
            for run, (item, paths, _) in packable.items()
        ]
        anchor = pick_anchor(candidates)
        order = acquisition_order(candidates, anchor)
        transforms: dict[str, Matrix4] = {}
        masks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for run in order:
            item, paths, grid = packable[run]
            rigid: RigidPath = connector.raw_to_anat(item.deriv, paths)
            transforms[run] = rigid.matrix
            masks[run] = _mask(paths[FuncRole.PRE_MASK])
            key = next((k for k, g in grids.items() if same_grid(g, grid)), None)
            if key is None:
                key = grid.key
                grids[key] = grid
            grid_of_run[run] = key
        frame = check_frame(transforms, anchor.run, masks, frame_limit_mm)

    position = {run: n for n, run in enumerate(order)}
    runs = tuple(
        sorted(
            listed,
            key=lambda r: (
                (0, position[r.run], (), "")
                if r.run in position
                else (1, 0, r.deriv.order, r.deriv.stem)
            ),
        )
    )
    match = connector.detect(Path(deriv_root))
    session_files = (
        dict(connector.session_files(Path(deriv_root), participant, session, match))
        if match is not None
        else {}
    )
    return SessionPlan(
        participant=participant,
        session=session,
        fs_subject=fs_subject,
        runs=runs,
        anchor=anchor,
        frame=frame,
        grids=grids,
        grid_of=grid_of_run,
        files=session_files,
    )
