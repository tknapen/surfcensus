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
import os
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
from jaxtyping import Bool, Float
from nibabel.filebasedimages import ImageFileError

from surfcensus.func import bids
from surfcensus.func.connectors.base import Connector, ConnectorError
from surfcensus.func.geometry import compose, grid_of, inverse, same_grid
from surfcensus.records import (
    MULTI_ECHO,
    NATIVE_SPACE_ONLY,
    NO_BRAIN_MASK,
    NO_CONFOUNDS,
    NO_PREPROCESSED_SERIES,
    NO_RAW_SERIES,
    NO_TRANSFORM,
    PACK_FAILURE,
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


def session_order(runs: Sequence[AnchorCandidate]) -> tuple[AnchorRule, str, tuple[str, ...]]:
    """The rule of a session, what it saw, and every run id in acquisition order by it.

    The time rule applies when every run has an ``AcquisitionTime`` that
    ``func.bids.acquisition_seconds`` reads, all of one kind (times of day, or dates and
    times); the runs are then in time order, equal times by name order. Otherwise they are in
    name order (``AnchorCandidate.order``, then the run id). Raises ``ValueError`` for no runs
    or a run given twice. :func:`plan_session` gives it every run of the session, so that
    neither the rule nor the order depends on which runs are packed."""
    if not runs:
        raise ValueError("a session without runs has no anchor")
    if len({r.run for r in runs}) != len(runs):
        raise ValueError("a run is given twice")
    by_name = _by_name(runs)
    seconds, why = _seconds(runs)
    if seconds is None:
        return RULE_NAME, why, tuple(r.run for r in by_name)
    ordered = sorted(by_name, key=lambda r: seconds[r.run])  # stable: ties keep name order
    return RULE_TIME, f"earliest AcquisitionTime of {len(runs)} runs", tuple(r.run for r in ordered)


def pick_anchor(runs: Sequence[AnchorCandidate]) -> Anchor:
    """The anchor of ONE session among the runs that have a stored transform. Runs with equal
    times go by name order. Raises ``ValueError`` for no runs.

    The time rule applies when every run has an ``AcquisitionTime`` that
    ``func.bids.acquisition_seconds`` reads, all of one kind (times of day, or dates and
    times); the anchor is then the earliest. Otherwise the anchor is the first run in name
    order (``AnchorCandidate.order``, then the run id). ``why`` says what the rule saw; under
    the time rule it holds the anchor's time as written."""
    rule, why, order = session_order(runs)
    first = next(r for r in runs if r.run == order[0])
    if rule == RULE_TIME:
        why = f"{why} ({first.acquisition_time})"
    return Anchor(run=first.run, rule=rule, why=why)


def acquisition_order(runs: Sequence[AnchorCandidate], anchor: Anchor) -> tuple[str, ...]:
    """The run ids in acquisition order: by time under the time rule, else by name order;
    ties by name order. The anchor comes first because its rule chose the first acquired.

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
    packable run and the anchor; the anchor's own entry is all zeros.

    Each run's displacement is taken over its own brain mask. The anchor need not have a mask
    (an anchor that is not packed, because it was not selected or lacks its mask, say): its
    entry is all zeros either way, with ``n_voxels`` 0 when it has none.
    ``shared_frame_plausible``: no run is flagged. ``anchor_is_outlier``: at least two runs
    besides the anchor, every one of them flagged, and no pair of them flagged against each
    other (over the mask of the first of the pair in the order of ``transforms``): then the
    anchor, or its stored transform, is the odd one, and the whole session is drawn off the
    anatomy. Raises ``ValueError`` when the anchor, or the mask of a run but the anchor, is
    missing."""
    if anchor not in transforms:
        raise ValueError("the anchor is not among the runs")
    missing = [run for run in transforms if run not in masks and run != anchor]
    if missing:
        raise ValueError(f"no brain mask for {len(missing)} of the runs")
    runs: dict[str, FrameDisplacement] = {}
    for run, own in transforms.items():
        if run == anchor:
            n_voxels = int(np.count_nonzero(masks[run][0])) if run in masks else 0
            runs[run] = FrameDisplacement(0.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0), 0.0, n_voxels, False)
            continue
        mask, affine = masks[run]
        runs[run] = frame_displacement(own, transforms[anchor], mask, affine, limit_mm)
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
    # fMRIPrep writes the table at level ``full`` only; the pack needs it for the volume count
    (FuncRole.CONFOUNDS, NO_CONFOUNDS),
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
    *,
    withheld: Mapping[str, tuple[str, str | None]] | None = None,
) -> SessionPlan:
    """The plan of one session from the listings: runs paired by ``key``, the reason of every
    run that cannot be packed, the anchor, the frame check, the grids.

    The anchor (decision D36: one rigid transform per session, that of its anchor run) is
    chosen among EVERY run of the session, before any reason a run is not packed: the runs
    are put in acquisition order by the rule of :func:`session_order` over all of them, and
    the anchor is the first in that order whose stored transform to the anatomy is listed and
    reads (``connector.raw_to_anat``). So which runs are packed, selected or fetched does not
    move the raw data: a run without its confounds table, its mask, an equal volume count or
    its raw series, a run that ``withheld`` holds back, and a native-space, standard-space or
    multi-echo run are anchors like any other when they come first. When the first run by the
    rule has no readable transform, the next one that has is the anchor and
    ``SessionPlan.anchor_note`` says so. ``withheld`` maps a run ``key`` to the reason and
    detail of a run that is listed and not packed whatever its files say (``not selected``
    for a run that ``--epi-runs`` leaves out, ``not fetched`` for a run with git-annex
    placeholders); of such a run only the raw sidecar and the stored transform are read.

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
      series, preprocessed series, transform, brain mask, confounds table, the last
      ``no confounds table``); a brain mask on another grid than the series
      (``no brain mask``); volume counts that differ; a stored transform that does not read
      (``pack failure``, with the error in ``details``). A run in ``withheld`` has the reason
      given there and no other.
    * Runs whose preprocessed series lie on different grids are all packed: each distinct
      grid (equal shape, affines within ``geometry.GRID_TOLERANCE``) is a group of its own,
      named in ``grids`` and ``grid_of`` by the key of its first run in acquisition order. No
      run gets ``runs on different grids``.
    * ``runs`` holds the packable runs first, in the session's acquisition order
      (:func:`session_order` over every run, ``SessionPlan.order``) restricted to them, which
      is the order of the session's one continuous stream (D38), then the others in name
      order. The anchor need not be packed, so it need not come first among the packed runs.
    * The frame check is over the packable runs, each over its own brain mask, against the
      anchor's transform, whether the anchor is packed or not.
    * The session's own files come from ``connector.session_files``, with the match that
      ``connector.detect`` gives for ``deriv_root`` (empty when it gives none).
    * A stored transform, the header of a series, a brain mask or a confounds table that
      cannot be read fails its own run (``pack failure``, the file named in ``details``,
      without a path), and the other runs are planned; the anchor needs its stored transform
      only, so a run whose series is damaged can be the anchor without being packed (review
      of step 2 of the EPI interface). Any other unreadable file raises (``OSError``,
      nibabel's errors, ``ConnectorError``).
    * A raw sidecar that is there and cannot be parsed gives its run no acquisition time, so
      the runs are put in name order; ``anchor_note`` names the sidecar.
    """
    raw_by_key: dict[str, list[RawRun]] = {}
    for raw in sorted(raw_runs, key=lambda r: (r.order, r.stem)):
        raw_by_key.setdefault(raw.key, []).append(raw)
    held = dict(withheld or {})

    def run_id(run: DerivRun) -> str:
        found = run_ids.get(run.key)
        if found is None:
            raise ValueError(f"no id for the run {_name(run)}")
        return found

    listed: list[SessionRun] = []
    details: dict[str, str] = {}
    all_paths: dict[str, dict[FuncRole, Path]] = {}
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
        if run.key in held:
            reason, detail = held[run.key]
            if detail:
                details[run_id(run)] = detail
        elif display.layout is None:
            reason = display.reason or NO_PREPROCESSED_SERIES
        elif display.layout == "native":
            reason = NATIVE_SPACE_ONLY
        elif multi:
            reason = MULTI_ECHO
        else:
            reason = next((why for role, why in _MISSING_REASONS if role in missing), None)
        grid: Grid | None = None
        reading: Path | None = None
        try:
            if reason is None:
                reading = paths[FuncRole.PRE_BOLD]
                grid = grid_of(reading)
                reading = paths[FuncRole.PRE_MASK]
                if not same_grid(grid, grid_of(reading)):
                    reason = NO_BRAIN_MASK
            if reason is None:
                counts = set()
                for role in (FuncRole.RAW_BOLD, FuncRole.PRE_BOLD):
                    reading = paths[role]
                    counts.add(_n_volumes(reading))
                reading = paths[FuncRole.CONFOUNDS]
                counts.add(_n_rows(reading))
                if len(counts) != 1:
                    reason = VOLUME_COUNTS_DIFFER
        except _HEADER_ERRORS as err:
            # a damaged header fails its own run; the others are planned (review of step 2)
            assert reading is not None
            reason, grid = PACK_FAILURE, None
            words = _error_words(err, deriv_root, raw_root)
            details[run_id(run)] = f"{reading.name} cannot be read: {words}"
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
        all_paths[item.run] = paths
        if reason is None and grid is not None:
            packable[item.run] = (item, paths, grid)

    # the stored transforms of every run that lists one, and the order of the session
    by_id = {item.run: item for item in listed}
    rigid_of: dict[str, RigidPath] = {}
    unreadable: dict[str, str] = {}
    for run_uid, item in by_id.items():
        paths = all_paths[run_uid]
        if FuncRole.XFM_TO_ANAT not in paths:
            continue
        try:
            rigid_of[run_uid] = connector.raw_to_anat(item.deriv, paths)
        except _TRANSFORM_ERRORS as err:
            unreadable[run_uid] = _error_words(err, deriv_root, raw_root)
    for run_uid, words in unreadable.items():
        if run_uid in packable:  # a packable run is one whose transform reads
            item = packable.pop(run_uid)[0]
            item = replace(item, reason=PACK_FAILURE)
            by_id[run_uid] = item
            details[run_uid] = f"its stored transform to the anatomy cannot be read: {words}"

    anchor = None
    anchor_note = None
    frame = None
    grids: dict[str, Grid] = {}
    grid_of_run: dict[str, str] = {}
    order: tuple[str, ...] = ()
    session_runs: tuple[str, ...] = ()
    candidates = [
        AnchorCandidate(
            run=run_uid,
            order=item.deriv.order,
            acquisition_time=(
                bids.sidecar_time(all_paths[run_uid][FuncRole.RAW_SIDECAR])
                if FuncRole.RAW_SIDECAR in all_paths[run_uid]
                else None
            ),
        )
        for run_uid, item in by_id.items()
    ]
    if candidates and rigid_of:
        rule, why, session_runs = session_order(candidates)
        first = next(r for r in session_runs if r in rigid_of)
        time_of = {c.run: c.acquisition_time for c in candidates}
        if rule == RULE_TIME:
            why = f"{why} ({time_of[first]})"
        anchor = Anchor(run=first, rule=rule, why=why)
        skipped = session_runs[: session_runs.index(first)]
        notes = []
        if rule != RULE_TIME:
            # a sidecar that is there and cannot be parsed is not "no time": say which one
            # put the order to the names (review of step 2)
            bad = sorted(
                {
                    f"{path.name} {problem}"
                    for path in (all_paths[r].get(FuncRole.RAW_SIDECAR) for r in by_id)
                    if path is not None and (problem := bids.sidecar_problem(path)) is not None
                }
            )
            if bad:
                notes.append(
                    f"the raw sidecar {', '.join(bad)}, so the runs are ordered by name and "
                    "not by acquisition time"
                )
        if skipped:
            names = ", ".join(by_id[r].name for r in skipped)
            words = "has" if len(skipped) == 1 else "have"
            notes.append(
                f"the first {_plural_runs(len(skipped))} by the rule ({names}) {words} no "
                f"readable stored transform to the anatomy, so the anchor is the next run, "
                f"{by_id[first].name}"
            )
        anchor_note = "; ".join(notes) or None
    if packable and anchor is not None:
        order = tuple(r for r in session_runs if r in packable)
        transforms: dict[str, Matrix4] = {anchor.run: rigid_of[anchor.run].matrix}
        masks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for run_uid in list(order):
            item, paths, grid = packable[run_uid]
            try:
                masks[run_uid] = _mask(paths[FuncRole.PRE_MASK])
            except _HEADER_ERRORS as err:
                packable.pop(run_uid)
                by_id[run_uid] = replace(item, reason=PACK_FAILURE)
                words = _error_words(err, deriv_root, raw_root)
                details[run_uid] = f"{paths[FuncRole.PRE_MASK].name} cannot be read: {words}"
                continue
            transforms[run_uid] = rigid_of[run_uid].matrix
            key = next((k for k, g in grids.items() if same_grid(g, grid)), None)
            if key is None:
                key = grid.key
                grids[key] = grid
            grid_of_run[run_uid] = key
        order = tuple(r for r in order if r in packable)
        frame = check_frame(transforms, anchor.run, masks, frame_limit_mm) if order else None

    position = {run_uid: n for n, run_uid in enumerate(order)}
    runs = tuple(
        sorted(
            by_id.values(),
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
        details=details,
        order=session_runs if anchor is not None else (),
        anchor_note=anchor_note,
    )


def _plural_runs(n: int) -> str:
    return "run" if n == 1 else f"{n} runs"


_TRANSFORM_ERRORS = (OSError, ValueError, EOFError, ConnectorError)
_HEADER_ERRORS = (OSError, ValueError, EOFError, zlib.error, ImageFileError)
"""What reading the header of a series, a mask or a confounds table raises for a file that
is damaged; the run then fails on its own (``pack failure`` with the file named)."""


def _error_words(err: BaseException, *roots: Path | None) -> str:
    """The text of an error without the roots of the paths it may hold."""
    text = str(getattr(err, "detail", None) or getattr(err, "strerror", None) or err)
    for root in roots:
        if root is not None:
            text = text.replace(f"{root}{os.sep}", "").replace(str(root), ".")
    return text
