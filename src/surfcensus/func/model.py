"""The contents of ``session.json``, ``pack.json`` and ``traces.json`` (docs/epi_interface.md,
"Where packs live").

Laid out before ``func.pack`` (which writes them, E1.C) and the page (which receives a shape
built from them, E1.D) are written, so that the two are written against one definition. The
models are frozen and refuse unknown keys, as ``render/model.py`` does; :func:`info_text` is
the one way they are written, so equal contents give equal bytes.

``session.json`` and ``pack.json`` stay in the run directory or the ``--out`` directory and are
never served (as ``render.json`` is not, decision D20): ``pack.json`` holds the sizes and
SHA-256 digests of the source files, which for a public dataset are those of public files.
What the page receives is a shape of ``app/schemas.py`` (E1.D) without them; a test compares
the fields of that shape with those of these models.

This module imports pydantic and the standard library only.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from surfcensus.func import PACK_FORMAT, PACK_VERSION
from surfcensus.records import AffineChoice, AnchorRule, Corrections, CorrectionState

READING = (
    "Raw: the raw BIDS series, placed once per session by the anchor run's registration,"
    " nothing else done to it. Preprocessed: the series as the pipeline wrote it. Their"
    " difference, and how each lies on the white and pial contours, is what this page shows:"
    " quality control of the raw data and of the preprocessing."
)
"""What the page says under its title (docs/epi_interface.md, The page in the package)."""

RULES: dict[str, str] = {
    "order": "a volume is nx*ny*nz bytes; voxel (i, j, k) is at i + nx*(j + ny*k)",
    "files": "every .bin file is a zlib stream",
    "chunks": (
        "a chunk holds chunk_volumes volumes (the last one fewer) as residuals:"
        " (value - ref) mod 256, ref being ref.bin of the run and stream"
    ),
    "ref": "rounded mean of the 8-bit volumes 0, 10, 20, ...",
    "window": (
        "per run, both streams: 0 to the 99.5th percentile of the preprocessed series inside"
        " the run's brain mask, over the volumes 0, 10, 20, ..."
    ),
    "quantise": "value = clip(floor((v - lo) * 255 / (hi - lo) + 0.5), 0, 255), float32",
    "sd_window": (
        "per run, both streams: 0 to the 99.5th percentile of the temporal SD of the"
        " preprocessed series inside the run's brain mask"
    ),
    "stills": (
        "temporal mean and SD (n - 1) over the volumes that the confounds table does not mark"
        " as non-steady-state, in float64, then 8 bit"
    ),
    "difference": "every difference view is raw minus pre of the two stored 8-bit values",
    "frame": (
        "per run, the displacement between its own stored transform and the anchor's over its"
        " brain mask (mm); flagged when the maximum exceeds limit_mm"
    ),
}
"""The rules of the formats of version 1 that version 2 keeps, as stage 1's manifest states
them (``pack_run.RULES``); written into ``session.json``."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PackFile(_Frozen):
    """A file of a pack: its path relative to the session directory, its compressed size and
    the SHA-256 of its compressed bytes, which the page checks."""

    file: str
    bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ChunkFile(PackFile):
    first: int = Field(ge=0)
    """The index of its first volume."""
    count: int = Field(ge=1)


class StreamFiles(_Frozen):
    ref: PackFile
    mean: PackFile
    sd: PackFile
    chunks: tuple[ChunkFile, ...]


class PackInput(_Frozen):
    """A file a pack was made from. No name and no path: a BIDS name names a task, a path
    names a place (decision D20)."""

    role: str
    """A ``FuncRole`` value, or a ``FileRole`` value for a FreeSurfer file of the contours."""
    base: Literal["raw", "deriv", "freesurfer"]
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PackSettings(_Frozen):
    """What decides the bytes of a pack beside its inputs."""

    portable_voxel_map: bool
    chunk_bytes: int
    window_step: int
    percentile: float
    zlib_level: int
    contour_unit: int


class StatementInfo(_Frozen):
    state: CorrectionState
    words: str | None
    source: str | None
    """A ``FuncRole`` value, or None."""
    note: str | None = None


class CorrectionsInfo(_Frozen):
    """``records.Corrections`` as the files hold it."""

    motion: StatementInfo
    sdc: StatementInfo
    sdc_method: str | None
    slice_timing: StatementInfo
    registration: StatementInfo
    dof: int | None
    sdc_figure_listed: bool | None


def corrections_info(corrections: Corrections) -> CorrectionsInfo:
    """The form of ``records.Corrections`` that ``pack.json`` and the shapes hold."""
    return CorrectionsInfo.model_validate(asdict(corrections))


class FrameInfo(_Frozen):
    """``records.FrameDisplacement`` of a run, rounded to 1e-4 as stage 1 wrote it."""

    mean_mm: float
    max_mm: float
    rotation_deg: float
    translation_mm: float
    translation_ras: tuple[float, float, float]
    shear_scale: float
    flagged: bool


class ContourFile(PackFile):
    slices: int = Field(ge=0)
    lines: int = Field(ge=0)
    points: int = Field(ge=0)


class GridInfo(_Frozen):
    """What the page needs of a grid. The affine stays out: its numbers are those of a public
    file, and the contours come in voxel coordinates."""

    grid_key: str = Field(pattern=r"^[0-9a-f]{12}$")
    shape: tuple[int, int, int]
    voxel_mm: tuple[float, float, float]
    oblique_deg: float
    edge_letters: dict[str, tuple[str, str, str, str]]
    """Plane -> the anatomical letters at its left, right, top and bottom edge, as the page
    draws the plane (``geometry.edge_letters``)."""
    contours: dict[str, ContourFile] | None
    """Plane -> its file; None when the session has no FreeSurfer subject."""


class RegistrationInfo(_Frozen):
    anchor: str
    """The run id of the anchor."""
    rule: AnchorRule
    limit_mm: float
    shared_frame_plausible: bool
    anchor_is_outlier: bool
    portable_voxel_map: bool
    reading: str = READING


class SessionRunInfo(_Frozen):
    """A run as ``session.json`` lists it, packed or not."""

    run: str
    name: str
    """The BIDS name without the subject; shown in a local run only."""
    acquisition_order: int = Field(ge=0)
    """0 for the anchor; runs that cannot be packed follow the others."""
    anchor: bool
    grid_key: str | None
    frame: FrameInfo | None
    reason: str | None
    """A ``records.PACK_REASONS`` value when the run cannot be packed, else None."""


class SessionInfo(_Frozen):
    """``session.json``."""

    format: Literal["epi-pack"] = PACK_FORMAT
    version: int = PACK_VERSION
    surfcensus_version: str
    session: str
    """The session id."""
    label: str
    """``sub-01 ses-1``, or ``sub-01`` in a dataset without sessions."""
    config_hash: str
    """``EpiConfig.session_hash()``."""
    fd_threshold_mm: float
    registration: RegistrationInfo | None
    """None when no run of the session can be packed."""
    grids: tuple[GridInfo, ...]
    runs: tuple[SessionRunInfo, ...]
    contours_from: str | None
    """Which FreeSurfer subject the contours were cut from, by its name in the subjects
    directory; None without contours."""
    rules: dict[str, str] = Field(default_factory=lambda: dict(RULES))


class TracesInfo(_Frozen):
    """``traces.json``: the motion estimates of a run as the page plots them."""

    fd: tuple[float | None, ...]
    dvars: tuple[float | None, ...]
    non_steady: tuple[int, ...]


class PackInfo(_Frozen):
    """``pack.json`` of one run. Never served."""

    format: Literal["epi-pack"] = PACK_FORMAT
    version: int = PACK_VERSION
    surfcensus_version: str
    run: str
    session: str
    config_hash: str
    """``EpiConfig.pack_hash()``."""
    settings: PackSettings
    inputs: tuple[PackInput, ...]
    """Every file the pack read, the anchor's transform and the reference image included, in
    a fixed order: raw roles, then derivative roles, in ``FuncRole`` order."""
    grid_key: str
    anchor_run: str
    n_volumes: int = Field(ge=1)
    tr_s: float | None
    window: tuple[float, float]
    sd_window: tuple[float, float]
    chunk_volumes: int = Field(ge=1)
    stills_volumes: int = Field(ge=1)
    """The steady-state volumes the stills are taken over."""
    raw_affine: AffineChoice
    reference_image: bool
    rigid: bool
    """False: the registration is used as stored and the page says "affine registration,
    not rigid"."""
    corrections: CorrectionsInfo
    streams: dict[str, StreamFiles]
    """``raw`` and ``pre``."""
    traces: PackFile
    content_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    """SHA-256 over the decoded 8-bit volumes, the references, the stills, the traces and
    the contour bytes in a fixed order (``func.pack.content_digest``): the identity of a pack,
    whatever zlib wrote it."""


Info = TypeVar("Info", SessionInfo, PackInfo, TracesInfo)


def info_text(info: BaseModel) -> str:
    """The text of a pack's JSON file: keys sorted, one space of indent, no NaN, newline."""
    return (
        json.dumps(info.model_dump(mode="json"), indent=1, sort_keys=True, allow_nan=False) + "\n"
    )


def read_info(path: Path | str, model: type[Info]) -> Info | None:
    """The model in ``path``, or None when the file is not there or does not read as one of
    this version (a pack being written, a damaged file, a pack of another version)."""
    try:
        return model.model_validate_json(Path(path).read_bytes())
    except (OSError, ValueError):
        return None
