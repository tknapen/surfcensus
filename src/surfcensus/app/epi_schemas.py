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

In a run directory (step 2) the main app serves the same page for each participant of the
run and an index of them (:class:`EpiIndexOut`), so ``app.schemas.ROUTES`` names these shapes.

Step 3 adds the subject pages of a run (docs/epi_interface.md, "Step 3: the plan (E3.0)"): a
list of every subject of the run with its surface state and its functional sessions
(:class:`SubjectsOut`), and a page per subject (:class:`SubjectPageOut`) with every session and
every run of it. Their routes are ``app.schemas.SUBJECT_ROUTES``.

This module needs pydantic, ``func.model`` and ``records``, and nothing of ``app/``: it is
imported by ``app.schemas``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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


# --- the index of a run directory (step 2) -------------------------------------------------


class EpiIndexSessionOut(_Shape):
    """A session of a participant in the index: what the catalog says of its runs, and
    whether the page shows it now (its ``session.json`` is in the run directory)."""

    session_uid: str
    label: str
    """The ``ses-`` label without ``ses-``; empty for a participant without sessions."""
    n_runs: int = Field(ge=0)
    """The runs listed (present), packable or not."""
    n_packable: int = Field(ge=0)
    """Those that the pack phase makes a pack of (packable and selected)."""
    n_packed: int = Field(ge=0)
    n_pending: int = Field(ge=0)
    """Packable runs with no pack of the present settings yet, done or failed."""
    n_failed: int = Field(ge=0)
    n_flagged: int = Field(ge=0)
    shown: bool
    """Whether the stream of the page holds this session now."""


class EpiParticipantOut(_Shape):
    """A participant of the run in ``GET /api/epi``, with the link to its stream."""

    participant_uid: str
    label: str | None
    """The BIDS participant label in a local run; None otherwise."""
    subject_uids: tuple[str, ...]
    """The FreeSurfer subjects of its sessions in the surface catalog, sorted."""
    n_sessions: int = Field(ge=0)
    n_runs: int = Field(ge=0)
    n_packable: int = Field(ge=0)
    n_packed: int = Field(ge=0)
    n_pending: int = Field(ge=0)
    n_failed: int = Field(ge=0)
    n_flagged: int = Field(ge=0)
    can_open: bool
    """Whether its stream shows a packed run now; the link is offered all the same."""
    url: str
    """``/epi/<participant_uid>``."""
    sessions: tuple[EpiIndexSessionOut, ...]
    """In stream order."""


class EpiIndexOut(_Shape):
    """``GET /api/epi``: the participants of the run with functional data, by label."""

    participants: tuple[EpiParticipantOut, ...]
    n_pending: int = Field(ge=0)
    """Runs of the whole run that wait for their pack: the page says that packs are still
    being made, and asks again."""


# --- the subject pages of a run directory (step 3) -----------------------------------------

SurfaceState = Literal["rated", "unratable", "rendered", "render failed", "not rendered", "none"]
"""The surface side of a row of :class:`SubjectsOut`, from ``export.tables.subjects_frame``,
the first that holds: ``unratable`` (``unratable_reason`` is set), ``rated`` (``n_ratings``
above 0), ``render failed`` (``render_status`` is ``failed``), ``rendered`` (``render_status``
is ``done``), ``not rendered`` (a subject otherwise); ``none`` for a participant row, which has
no FreeSurfer subject."""

RunState = Literal["packed", "pending", "failed", "cannot be packed", "not selected"]
"""The state of a run as the subject pages and the export say it, the values of
``schemas.PACK_STATES``, from ``func.batch.recorded_states``: what is recorded under the
present pack settings (``packed``: a ``packs`` row ``done``; ``failed``: a ``packs`` row
``failed``; ``pending``: packable and selected with neither), ``cannot be packed``
(``func_runs.packable`` false) and ``not selected`` (``--epi-runs`` leaves it out). Not what
``surfcensus status`` counts, which also asks the pack signature and the disk
(``func.batch.pack_status``): an outdated pack is ``packed`` here, since the page still plays
it, and pending there."""

StreamRule = Literal["acquisition time", "session label"]
"""How the sessions of a participant's stream are put in order (``store.SESSION_ORDERS``,
``func_participants.session_order``; decision D42)."""

RowKind = Literal["subject", "participant"]


class SubjectRowOut(_Shape):
    """One row of ``GET /api/subjects``: a present FreeSurfer subject of the run, with the
    functional sessions whose ``func_sessions.subject_uid`` is that subject, or a participant
    with functional sessions that have no FreeSurfer subject, with those sessions only. So
    every present session is on exactly one row."""

    key: str
    """What ``/subject/{key}`` takes: the ``subject_uid`` of a subject row, the
    ``participant_uid`` of a participant row."""
    kind: RowKind
    subject_uid: str | None
    """None for a participant row."""
    participant_uid: str | None
    """The participant of the row's sessions; None for a subject without sessions."""
    label: str | None
    """The subject label, or for a participant row the BIDS participant label; None outside
    a local run."""
    dataset: str
    """The pseudonym of the dataset."""
    surface: SurfaceState
    surface_reason: str | None
    """``unratable_reason``, else ``render_reason``; None when neither is set."""
    n_sessions: int = Field(ge=0)
    n_runs: int = Field(ge=0)
    n_packable: int = Field(ge=0)
    """Runs that are packable and selected."""
    n_packed: int = Field(ge=0)
    n_pending: int = Field(ge=0)
    n_failed: int = Field(ge=0)
    n_flagged: int = Field(ge=0)
    url: str
    """``/subject/<key>``."""
    epi_url: str | None
    """``/epi/<participant_uid>`` when the row has a session; None otherwise."""


class SubjectsOut(_Shape):
    """``GET /api/subjects``: the subject rows by label (then ``subject_uid``), then the
    participant rows by label (then ``participant_uid``)."""

    subjects: tuple[SubjectRowOut, ...]
    n_pending: int = Field(ge=0)
    """Runs of the whole run that wait for their pack: the page says so, and asks again."""


class FuncFrameOut(_Shape):
    """The frame check of a run (``func_runs.frame_*``): how far its own stored registration
    puts it from where the session's one registration does (decision D36)."""

    mean_mm: float
    max_mm: float
    rotation_deg: float
    translation_mm: float
    flagged: bool | None


class FuncRunOut(_Shape):
    """A run of a session on the subject page, from ``PublicCatalog.func_runs`` and
    ``func.batch.recorded_states``."""

    run_uid: str
    name: str | None
    """The BIDS name without the subject; None outside a local run."""
    acquisition_order: int | None
    session_order: int | None
    anchor: bool
    selected: bool
    n_volumes: int | None
    tr_s: float | None
    echoes: int = Field(ge=1)
    grid_key: str | None
    grid_shape: tuple[int, int, int] | None
    voxel_mm: tuple[float, float, float] | None
    oblique_deg: float | None
    raw_affine: AffineChoice | None
    reference_image: bool | None
    rigid: bool | None
    dof: int | None
    corrections: CorrectionsInfo | None
    """``func_runs.corrections`` parsed: what the pipeline did to the preprocessed series."""
    frame: FuncFrameOut | None
    state: RunState
    reason: str | None
    """Why it cannot be packed or is not selected (``func_runs.reason``), or why its pack
    failed (``packs.reason``); None otherwise."""
    detail: str | None
    epi_url: str | None
    """``/epi/<participant_uid>#run=<run_uid>`` for a ``packed`` run of a session that the
    stream shows now; None otherwise."""


class FuncSessionOut(_Shape):
    """A functional session on the subject page, with its runs."""

    session_uid: str
    participant_uid: str
    participant: str | None
    """The BIDS participant label in a local run; None otherwise."""
    label: str
    """The ``ses-`` label without ``ses-``; empty for a participant without sessions."""
    stream_order: int = Field(ge=0)
    """The place of the session in its participant's stream (``func_sessions.stream_order``)."""
    stream_rule: StreamRule
    """How the participant's sessions were put in that order (decision D42)."""
    subject_uid: str | None
    fs_subject: str | None
    anchor_run_uid: str | None
    anchor_name: str | None
    anchor_rule: AnchorRule | None
    anchor_note: str | None
    anchor_packed: bool
    """Whether the anchor is a ``packed`` run."""
    frame_limit_mm: float | None
    shared_frame_plausible: bool | None
    anchor_is_outlier: bool | None
    n_grids: int = Field(ge=0)
    problem: str | None
    shown: bool
    """Whether the stream of the participant's page holds this session now
    (``app.epi.Packs.shown_sessions``)."""
    runs: tuple[FuncRunOut, ...]
    """By ``acquisition_order``, then ``run_uid``."""


class SubjectPageOut(_Shape):
    """``GET /api/subjects/{key}``: one row of :class:`SubjectsOut` with its sessions."""

    subject: SubjectRowOut
    review_url: str | None
    """``/review#subject=<subject_uid>`` for a subject row; None for a participant row."""
    rate_url: str
    """``/``."""
    index_url: str
    """``/subjects``."""
    epi_index_url: str
    """``/epi``."""
    sessions: tuple[FuncSessionOut, ...]
    """By participant label (then ``participant_uid``), then ``stream_order``."""
