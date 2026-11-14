"""The shapes of the EPI page's API: what the page receives (docs/epi_interface.md, The app;
work package E1.D).

Built from ``session.json``, ``pack.json`` and ``traces.json`` (``func.model``), which are
never served. What a shape leaves out of them is decided field by field and pinned by a test
(``tests/test_epi_app.py``): no input of a pack (the sizes and SHA-256 digests of the source
files), no affine, no path, no settings hash and no content digest. What it holds of a pack's
own files is their name relative to the session's media URL, their compressed size and the
SHA-256 of their compressed bytes, which the page checks before it inflates them.

The page plays one stream (decision D38): the sessions of :class:`EpiStreamOut` in order and,
within a session, the packed runs of :attr:`EpiSessionOut.runs` in order; ``first`` is the
index of a run's first volume within its session. A session names its media URL, so that the
same page is served from a run directory in step 2 with only the URLs changing.

This module needs pydantic, ``func.model`` and ``app.schemas`` (for ``HealthOut``).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from surfcensus.app.schemas import HealthOut
from surfcensus.func.model import CorrectionsInfo, FrameInfo
from surfcensus.records import AffineChoice, AnchorRule


class _Shape(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class FileOut(_Shape):
    """A file of a pack: its path relative to :attr:`EpiSessionOut.media_url`, its compressed
    size and the SHA-256 of its compressed bytes."""

    file: str
    bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ChunkOut(FileOut):
    first: int = Field(ge=0)
    """The index of its first volume within the run."""
    count: int = Field(ge=1)


class ContourOut(FileOut):
    slices: int = Field(ge=0)
    lines: int = Field(ge=0)
    points: int = Field(ge=0)


class StreamOut(_Shape):
    ref: FileOut
    mean: FileOut
    sd: FileOut
    chunks: tuple[ChunkOut, ...]


class TracesOut(_Shape):
    fd: tuple[float | None, ...]
    dvars: tuple[float | None, ...]
    non_steady: tuple[int, ...]


class GridOut(_Shape):
    """``func.model.GridInfo`` as the page receives it."""

    grid_key: str
    shape: tuple[int, int, int]
    voxel_mm: tuple[float, float, float]
    oblique_deg: float
    edge_letters: dict[str, tuple[str, str, str, str]]
    contours: dict[str, ContourOut] | None
    slice_normal: dict[str, tuple[float, float, float]]
    slice_mm: dict[str, tuple[float, ...]]
    token: str
    """Asked for with the contour files (``?v=``): changes whenever one of them does."""


class RegistrationOut(_Shape):
    anchor: str
    """The run id of the anchor."""
    rule: AnchorRule
    limit_mm: float
    shared_frame_plausible: bool
    anchor_is_outlier: bool
    portable_voxel_map: bool
    reading: str
    anchor_packed: bool = True
    """Whether the anchor is among the packed runs (``RegistrationInfo.anchor_packed``)."""
    note: str | None = None
    """Why a later run is the anchor, when the first by the rule has no readable transform."""


class EpiPackOut(_Shape):
    """What the page needs of ``pack.json`` and ``traces.json`` of a packed run."""

    raw_affine: AffineChoice
    reference_image: bool
    rigid: bool
    corrections: CorrectionsInfo
    chunk_volumes: int = Field(ge=1)
    stills_volumes: int = Field(ge=1)
    streams: dict[str, StreamOut]
    """``raw`` and ``pre``."""
    traces: TracesOut
    token: str
    """Asked for with the run's files (``?v=``): changes whenever one of them does."""


class EpiRunOut(_Shape):
    """A run of a session as ``session.json`` lists it, packed or not."""

    run_uid: str
    name: str
    """The BIDS name without the subject."""
    acquisition_order: int
    anchor: bool
    grid_key: str | None
    frame: FrameInfo | None
    reason: str | None
    """Why the run is not in the stream; None for a packed run."""
    n_volumes: int | None
    tr_s: float | None
    first: int | None
    window: tuple[float, float] | None
    sd_window: tuple[float, float] | None
    detail: str | None
    """One sentence beside the reason, when there is more to say."""
    pack: EpiPackOut | None
    """None for a run that is not packed, or whose pack is not whole on disk."""
    session_order: int | None = None
    """The place of the run among all runs of its session (``SessionRunInfo.session_order``)."""


class EpiSessionOut(_Shape):
    session_uid: str
    label: str
    media_url: str
    """The prefix of every file of :attr:`grids` and :attr:`runs`: ``/epi-media/<id>/``."""
    fd_threshold_mm: float
    registration: RegistrationOut | None
    grids: tuple[GridOut, ...]
    runs: tuple[EpiRunOut, ...]
    contours_from: str | None


class EpiStreamOut(_Shape):
    """``GET /api/epi/stream``: the subject's one stream."""

    label: str
    order: str
    """How the sessions were put in order (``StreamInfo.order``)."""
    reading: str
    """What the page says under its title (``func.model.READING``)."""
    memory_mib: int = Field(ge=1)
    """The limit of decoded chunks held by the page, for the whole stream."""
    rules: dict[str, str]
    """``func.model.RULES``, as ``session.json`` states them."""
    contour_unit: int = Field(ge=1)
    """Contour points are stored in ``1 / contour_unit`` voxel (``func.CONTOUR_UNIT``)."""
    sessions: tuple[EpiSessionOut, ...]


EPI_ROUTES: tuple[tuple[str, str, type[BaseModel] | None], ...] = (
    ("GET", "/", None),
    ("GET", "/epi", None),
    ("GET", "/api/epi/stream", EpiStreamOut),
    ("GET", "/epi-media/{session_uid}/{path}", None),
    ("GET", "/static/{path}", None),
    ("GET", "/healthz", HealthOut),
)
"""Every route of the pack-only server, and the shape it answers with (None: a page or a
file)."""
