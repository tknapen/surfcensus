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

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float

from surfcensus.func.connectors.base import Connector
from surfcensus.records import (
    AffineChoice,
    Anchor,
    AnchorCandidate,
    AnchorRule,
    DerivRun,
    FrameCheck,
    FrameDisplacement,
    Matrix4,
    RawAffine,
    RawRun,
    SessionPlan,
)

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


def pick_anchor(runs: Sequence[AnchorCandidate]) -> Anchor:
    """The anchor of ONE session among the runs that have a stored transform. Runs with equal
    times go by name order. Raises ``ValueError`` for no runs."""
    raise NotImplementedError("work package E1.B: func/session.py")


def acquisition_order(runs: Sequence[AnchorCandidate], anchor: Anchor) -> tuple[str, ...]:
    """The run ids in acquisition order with the anchor first: by time under the time rule,
    else by name order; ties by name order."""
    raise NotImplementedError("work package E1.B: func/session.py")


def raw_affine_choice(raw_bold: Path, reference: Path | None) -> RawAffine:
    """Which affine the raw stream of a run uses; ``reference`` is the run's BOLD-space
    reference image, or None. A reference on another grid than the raw series leaves the
    choice unchecked."""
    raise NotImplementedError("work package E1.B: func/session.py")


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
    mask."""
    raise NotImplementedError("work package E1.B: func/session.py")


def check_frame(
    transforms: Mapping[str, Matrix4],
    anchor: str,
    masks: Mapping[str, tuple[Bool[np.ndarray, "i j k"], Float[np.ndarray, "4 4"]]],
    limit_mm: float = FRAME_LIMIT_MM,
) -> FrameCheck:
    """The frame check of one session: ``transforms`` and ``masks`` by run id, for every
    packable run; the anchor's own entry is all zeros."""
    raise NotImplementedError("work package E1.B: func/session.py")


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
    """
    raise NotImplementedError("work package E1.B: func/session.py")
