"""The shapes of the rater app's API: what the pages send and receive. See docs/rater_app.md.

The server and the frontend are both written against this module. It needs pydantic only
and imports ``protocol.py`` and ``render/model.py``, nothing from the rest of ``app/``.

Blindness is a property of these shapes: nothing a rating route returns (``SessionOut``,
``NextOut``, ``RatingOut``; a skip is answered with a ``NextOut``) has a field for a
dataset, a subject label, a score, a flag or another rater's rating, and the render a page
receives (:class:`RenderOut`) is ``render.json`` without the digests of its source files.
The unblinded shapes are ``ReviewOut``, ``SubjectOut`` and ``StatsOut``.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from surfcensus.protocol import CriterionId, MarkClass, Protocol
from surfcensus.render.model import (
    BoxInfo,
    FsaveragePole,
    ModeInfo,
    PanelInfo,
    PanelName,
    PoleInfo,
    RenderConfig,
    RenderInfo,
    TileFiles,
    Vec3,
    VertexMapInfo,
    VideoInfo,
    VolumeInfo,
    _Frozen,
)

# --- vocabulary --------------------------------------------------------------------------

AppMode = Literal["local", "team"]
RaterRole = Literal["rater", "adjudicator", "lead"]
RATER_ROLES: tuple[RaterRole, ...] = ("rater", "adjudicator", "lead")

AssignmentKind = Literal["calibration", "main", "repeat", "adjudication"]
ASSIGNMENT_KINDS: tuple[AssignmentKind, ...] = ("calibration", "main", "repeat", "adjudication")
"""What ``assignments.kind`` holds. A page is told ``calibration`` or ``main`` only
(:data:`ShownKind`): a repeat looks like a new subject."""
ShownKind = Literal["calibration", "main"]

PlaybackMode = Literal["sync", "seq"]
Surface = Literal["pial", "white"]
HemiName = Literal["lh", "rh"]

FinalMethod = Literal["consensus", "adjudicated", "single"]
FINAL_METHODS: tuple[FinalMethod, ...] = ("consensus", "adjudicated", "single")
UNRATABLE_CRITERION = "unratable"
"""``final_labels.criterion`` of the row that says a subject is unratable; its ``value`` is
the position of the reason in the protocol (``Protocol.unratable_index``)."""

EVENT_TYPES: tuple[str, ...] = (
    "load",
    "play",
    "pause",
    "frames",
    "mode",
    "speed",
    "zoom",
    "contours",
    "marks_hidden",
    "mark_add",
    "mark_delete",
    "mark_undo",
    "help",
    "back",
    "tour_open",
    "tour_close",
)
"""The interaction events a page logs. ``frames`` is run-length encoded:
``{"mode": "sync", "visits": [[frame, ms], ...]}``. ``tour_open`` says how the tour of the
rating screen was opened (``{"how": "first" | "key" | "help"}``) and ``tour_close`` how far
it was read (``{"step": the last step shown, from 1, "steps": how many there are,
"finished": whether it was closed from the last one, "ms": how long it was open}``). An
unknown type is stored as given."""
SKIP_EVENT = "skip"
"""The event the server stores when a rater skips a subject (``POST /api/skip``,
``app.db.record_skip``): ``ts`` is the client's time of the skip, like the ``ts`` of the
events a page sends, and the payload is ``{"server_ts": ..., "ms_on_subject": the time the
subject was on a visible page in the visit that ended with the skip, "n": the number of
the skip for its assignment, from 1}``."""
SERVER_EVENT_TYPES: tuple[str, ...] = (SKIP_EVENT,)
"""Events of an assignment that the server writes itself. A page cannot send one
(:class:`EventIn` refuses the type), so that the skips in the events are the skips the
server took."""
RUN_EVENT_TYPES: tuple[str, ...] = ("config_change",)
"""Events of the run, stored without an assignment by the server side."""

MAX_EVENTS_PER_CALL = 500
MAX_EVENT_PAYLOAD_BYTES = 16_000
"""Of the payload as compact JSON."""
MAX_MARKS_PER_RATING = 200
MAX_NOTE_CHARS = 10_000
MAX_REVIEW_ROWS = 1_000
"""Rows of the review table in one answer."""

MEDIA_FILE = re.compile(
    r"poster\.webp|anim\.mp4|tiles/(?:sagittal|coronal|axial)_(?:[0-9]{3}|[1-9][0-9]{3,})(?:_c)?\.webp"
)
"""The files of a render that ``GET /media/{subject_uid}/{path}`` serves (``fullmatch``):
the tiles, the poster and the video, under the names a render writes. ``render.json`` and
``vertex_map.npz`` are never served."""


def media_base(subject_uid: str, base_url: str | None = None) -> str:
    """Where the media of a subject are fetched from: ``/media/<subject_uid>/``, or under
    ``app.media_base_url`` when the run has one. Ends with a slash; a page appends a name
    from the render's ``tiles`` patterns and ``?v=<render_key>``."""
    root = "/media" if base_url is None else base_url.rstrip("/")
    return f"{root}/{subject_uid}/"


def tile_name(pattern: str, panel: str, index: int) -> str:
    """A tile's name from a pattern of ``TileFiles`` (``tiles/{panel}_{index:03d}.webp``).
    The patterns are Python format strings; a page replaces ``{panel}`` and ``{index:03d}``
    (the number padded with zeros to three digits) to the same effect."""
    return pattern.format(panel=panel, index=index)


class _Shape(_Frozen):
    """Frozen, and no key beyond the fields: a response with a stray key fails its test."""


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("a finite number is needed")
    return value


# --- the render a page receives -------------------------------------------------------------


class RenderOut(_Shape):
    """``render.json`` as a page receives it: every field of ``RenderInfo`` but ``inputs``
    and ``template``.

    Those two hold sizes and SHA-256 digests of source files, which stay in the run
    directory (decision D20), and a page has no use for them. Whether marks get an
    fsaverage vertex is in ``vertex_map.fsaverage``. Made by :func:`render_out`.
    """

    schema_version: int
    surfcensus_version: str
    config: RenderConfig
    config_hash: str
    volume: VolumeInfo
    poles: dict[HemiName, PoleInfo]
    box: BoxInfo
    window: tuple[float, float]
    panels: dict[PanelName, PanelInfo]
    modes: dict[PlaybackMode, ModeInfo]
    tiles: TileFiles
    video: VideoInfo | None = None
    poster: str | None = None
    vertex_map: VertexMapInfo | None = None
    fsaverage_poles: dict[HemiName, FsaveragePole] | None = None
    versions: dict[str, str]


RENDER_PRIVATE_FIELDS: frozenset[str] = frozenset({"inputs", "template"})
"""The fields of ``RenderInfo`` that no page receives."""


def render_out(info: RenderInfo) -> RenderOut:
    """The render of a subject for a page: ``info`` without :data:`RENDER_PRIVATE_FIELDS`."""
    return RenderOut.model_validate(info.model_dump(exclude=set(RENDER_PRIVATE_FIELDS)))


# --- marks ----------------------------------------------------------------------------------


class MarkIn(_Shape):
    """A click mark as a page posts it: a point on one tile. The server converts it to
    voxels, tkr RAS and vertices; a coordinate sent by a client is not trusted."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    client_id: str = Field(min_length=1, max_length=64)
    """The page's own name for the mark, unique within a rating; the answer repeats it."""
    mark_class: MarkClass = Field(alias="class")
    panel: PanelName
    slice: int
    """Position in the panel's slice list of the tile that was clicked."""
    px: float
    """Tile pixel coordinates, continuous, origin at the top-left corner of the tile:
    ``px = offset_x * tile_width / displayed_width``; likewise ``py``."""
    py: float
    frame: int | None = None
    """The frame of ``mode`` that was showing; with ``mode``, the playback context of the click."""
    mode: PlaybackMode | None = None
    ts: str | None = None
    """Client time of the click, ISO 8601."""

    _finite_pixels = field_validator("px", "py")(_finite)


class MarkVertices(_Shape):
    """Where a mark lies on the surfaces. Both vertices index one hemisphere's surfaces."""

    hemi: HemiName | None = None
    """The one hemisphere both vertices belong to; None when no vertex is in range."""
    pial_vertex: int | None = None
    pial_dist_mm: float | None = None
    white_vertex: int | None = None
    white_dist_mm: float | None = None
    fsavg6_vertex: int | None = None
    """The template vertex of the vertex named by ``vertex_surface``; None when the render
    has no counterparts (a run without a template) or no vertex is in range."""
    vertex_surface: Surface | None = None
    """Whose vertex ``fsavg6_vertex`` is the counterpart of: the surface the mark's class is
    about (white for C, pial otherwise) when it has a vertex in range, else the other."""
    box_dist_mm: float
    """Distance from the mark to the view box; 0 inside."""


class ResolvedMark(MarkVertices):
    """What the server computes from a :class:`MarkIn` (``app.marks.resolve_mark``)."""

    vox: Vec3
    """Continuous voxel coordinates."""
    tkr: Vec3
    ras: Vec3
    """Scanner RAS."""


class SavedMark(MarkVertices):
    """A stored mark in the answer to a rating: the page's id, the database's, the vertices."""

    client_id: str
    mark_id: int


class PreviousMark(MarkIn, MarkVertices):
    """A mark of the submission that a page is asked to revise: what was posted, with
    ``client_id`` set by the server (``m<mark_id>``), and what it resolved to."""

    mark_id: int


class ReviewMark(MarkVertices):
    """A mark on the review screen, with who made it. Unblinded."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    mark_id: int
    assignment_id: int
    rater: str
    mark_class: MarkClass = Field(alias="class")
    panel: PanelName
    slice: int
    px: float
    py: float
    vox: Vec3
    tkr: Vec3


# --- session --------------------------------------------------------------------------------


class RaterOut(_Shape):
    id: int
    name: str
    role: RaterRole


class SettingsOut(_Shape):
    mark_show_mm: float
    """A mark shows in a panel whose current slice passes within this distance of it."""
    blind: bool


class ExampleOut(_Shape):
    """A reference slice for the ``?`` screen, from ``calibration.yaml``."""

    criterion: CriterionId
    value: int
    panel: PanelName
    slice: int
    media_base: str
    render_key: str
    caption: str = ""
    tiles: TileFiles = TileFiles()


class SessionOut(_Shape):
    """``GET /api/session``."""

    mode: AppMode
    rater: RaterOut
    app_version: str
    protocol: Protocol
    protocol_sha256: str
    require_marks: bool
    """``protocol.marks_required``: the run's setting when it has one, else the protocol's."""
    settings: SettingsOut
    examples: tuple[ExampleOut, ...] = ()
    """Empty when the run has no calibration set, and for a rater who has not finished
    calibration."""


# --- the queue ------------------------------------------------------------------------------


class PreviousOut(_Shape):
    """The current submission of an assignment that is shown again for revision."""

    values: dict[CriterionId, int]
    unratable: str | None = None
    other: bool = False
    note: str = ""
    marks: tuple[PreviousMark, ...] = ()


class AssignmentOut(_Shape):
    """A subject to rate. Nothing about the subject but its media and its geometry."""

    assignment_id: int
    kind: ShownKind
    subject_uid: str
    media_base: str
    render_key: str
    render: RenderOut
    has_vertex_map: bool
    previous: PreviousOut | None = None
    """None for a new subject, and for a repeat."""


class PrefetchOut(_Shape):
    """A subject that comes next, so that a page can load its tiles ahead."""

    subject_uid: str
    media_base: str
    render_key: str
    n_slices: dict[PanelName, int]
    """How many tiles each panel has: with ``tiles``, the names to fetch."""
    tiles: TileFiles = TileFiles()


class ProgressOut(_Shape):
    done_today: int
    done_total: int
    remaining: int
    """Subjects the rater has still to rate, the skipped ones among them."""
    skipped: int
    """Subjects the rater skipped that wait to be offered again
    (``app.queue.subjects_skipped``); the one on screen is not counted."""
    rendered: int
    n_subjects: int


class NextOut(_Shape):
    """``GET /api/next``, ``GET /api/back``, the answer to ``POST /api/skip`` and the
    ``next`` of a stored rating."""

    status: Literal["assignment", "rendering", "done"]
    assignment: AssignmentOut | None = None
    prefetch: tuple[PrefetchOut, ...] = ()
    retry_after_s: float | None = None
    """With ``rendering``: when to ask again."""
    progress: ProgressOut

    @model_validator(mode="after")
    def _status_says_what_is_there(self) -> NextOut:
        if (self.status == "assignment") != (self.assignment is not None):
            raise ValueError("an assignment comes with the status 'assignment', and only with it")
        if (self.status == "rendering") != (self.retry_after_s is not None):
            raise ValueError("retry_after_s comes with the status 'rendering', and only with it")
        return self


# --- a rating -------------------------------------------------------------------------------


class RatingIn(_Shape):
    """``POST /api/rating``. Either ``unratable`` is a reason of the protocol (values may be
    empty, marks are ignored) or ``values`` has exactly the protocol's criteria. The rules
    that need the protocol and the render are the server's, with the field each names."""

    assignment_id: int
    protocol_version: str
    values: dict[str, int] = Field(default_factory=dict)
    unratable: str | None = None
    other: bool = False
    note: str = Field(default="", max_length=MAX_NOTE_CHARS)
    marks: tuple[MarkIn, ...] = Field(default=(), max_length=MAX_MARKS_PER_RATING)
    client_ts: str | None = None
    ms_on_subject: int = Field(ge=0)

    @model_validator(mode="after")
    def _client_ids_differ(self) -> RatingIn:
        ids = [mark.client_id for mark in self.marks]
        if len(set(ids)) != len(ids):
            raise ValueError("marks: two marks have the same client_id")
        return self


class SavedOut(_Shape):
    assignment_id: int
    submission_id: int
    revision: int
    marks: tuple[SavedMark, ...] = ()


class RatingOut(_Shape):
    saved: SavedOut
    next: NextOut


# --- a skip ---------------------------------------------------------------------------------


class SkipIn(_Shape):
    """``POST /api/skip``: the rater leaves the subject on screen without a rating. Nothing
    of what was entered is sent: it stays in the page as the draft of the assignment. The
    answer is a :class:`NextOut`, the subject that comes next.

    The assignment stays open and is offered again after every subject the rater has not
    been shown yet (``app.queue``). The server counts the skip on the assignment and stores
    an event of type :data:`SKIP_EVENT`. A request that is sent again, with the same
    ``client_ts``, is counted once."""

    assignment_id: int
    client_ts: str = Field(min_length=1, max_length=64)
    """Client time of the skip, ISO 8601. With the assignment it names the skip: a second
    request with the same two is the first one sent again."""
    ms_on_subject: int = Field(ge=0)
    """How long the subject was on a visible page in this visit. The ``ms_on_subject`` of
    a later submission is the time of the visit that ends with the submission."""


# --- events ---------------------------------------------------------------------------------


class EventIn(_Shape):
    ts: str
    type: str = Field(min_length=1, max_length=64)
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def _of_a_page(cls, value: str) -> str:
        if value in SERVER_EVENT_TYPES:
            raise ValueError(f"an event of type {value!r} is stored by the server, not sent")
        return value

    @field_validator("payload")
    @classmethod
    def _small(cls, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        except ValueError:
            # Python's parser takes NaN and Infinity, and a number such as 1e400 becomes
            # infinity; written out again, none of them is JSON
            raise ValueError("a payload holds a number that JSON has not (NaN, infinity)") from None
        size = len(text.encode())
        if size > MAX_EVENT_PAYLOAD_BYTES:
            raise ValueError(
                f"a payload of {size} bytes; at most {MAX_EVENT_PAYLOAD_BYTES} are stored"
            )
        return payload


class EventsIn(_Shape):
    """``POST /api/events``: a batch of interaction events of one assignment."""

    assignment_id: int
    events: tuple[EventIn, ...] = Field(max_length=MAX_EVENTS_PER_CALL)


class EventsOut(_Shape):
    stored: int


# --- unblinded: review, statistics ----------------------------------------------------------


class ReviewQuery(_Shape):
    """The query parameters of ``GET /api/review``."""

    sort: str | None = None
    """A column of the table. None: by ``primary_rank``, worst first, the subjects without
    a score last; rows that are equal in the column are in that order as well."""
    order: Literal["asc", "desc"] = "asc"
    filter: str = ""
    """Comma-separated ``column:value``; a row passes when it has that value, as text, in
    every column named. Empty: every row."""
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=200, ge=1, le=MAX_REVIEW_ROWS)


class ReviewOut(_Shape):
    """``GET /api/review``: one row per subject. ``columns`` names the keys of a row in
    display order: ``subject_uid``, in a local run ``subject_label``, ``dataset``,
    ``fs_version``, ``render_status``, ``render_reason``, the columns of the primary row of
    ``scores.parquet``, ``final_A``, ``final_B``, ``final_C``, ``final_method``,
    ``unratable_reason``, ``n_ratings``, ``n_marks``."""

    total: int
    """Rows that pass the filter, before ``offset`` and ``limit``."""
    columns: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]


class ReviewRating(_Shape):
    """A completed assignment on the review screen: the columns of the ``ratings`` export
    that the screen shows."""

    assignment_id: int
    rater: str
    kind: AssignmentKind
    render_key: str
    render_current: bool
    """Whether the rating was made on the tiles the subject has now."""
    A: int | None = None
    B: int | None = None
    C: int | None = None
    unratable_reason: str | None = None
    other: bool = False
    note: str = ""
    revision: int
    server_ts: str
    ms_on_subject: int | None = None


class ClusterOut(_Shape):
    """An outlier cluster whose centroid has a native position (``clusters_native``)."""

    reference_id: str
    hemi: HemiName
    region: str
    rank: int
    tkr: Vec3
    """Pial position of the native vertex of the centroid."""
    area_mm2: float
    peak_z: float


class SubjectOut(_Shape):
    """``GET /api/review/{subject_uid}``: one subject with everything known about it."""

    subject_uid: str
    media_base: str | None = None
    """None, like ``render_key`` and ``render``, for a subject without a ready render."""
    render_key: str | None = None
    render: RenderOut | None = None
    scores: dict[str, Any] | None = None
    """The primary row of ``scores.parquet``; None for a subject without one."""
    ratings: tuple[ReviewRating, ...] = ()
    marks: tuple[ReviewMark, ...] = ()
    """The marks of the current submissions."""
    clusters: tuple[ClusterOut, ...] = ()


class DayCount(_Shape):
    day: str
    """``YYYY-MM-DD``, UTC."""
    n: int


class RaterStats(_Shape):
    rater: str
    n_completed: int
    median_s: float | None = None
    by_day: tuple[DayCount, ...] = ()


class StatsOut(_Shape):
    """``GET /api/stats``. ``agreement`` and ``drift`` are filled in team mode (milestone 5)."""

    raters: tuple[RaterStats, ...]
    agreement: dict[str, Any] | None = None
    drift: dict[str, Any] | None = None


# --- errors, health -------------------------------------------------------------------------


class ErrorOut(_Shape):
    field: str
    """The part of the request that is wrong (``values``, ``marks.0.px``, ``note``), or
    ``request`` when it is none in particular."""
    message: str


class ErrorsOut(_Shape):
    """The body of every answer with a status of 400 or above."""

    errors: tuple[ErrorOut, ...]


class HealthOut(_Shape):
    ok: bool
    version: str


# --- routes ---------------------------------------------------------------------------------


class Route(NamedTuple):
    """One route of the app: what the server answers and what a page may call."""

    method: Literal["GET", "POST"]
    path: str
    """With ``{name}`` for a part of the path that varies; the ``{path}`` of a file may
    hold slashes."""
    response: type[BaseModel] | None = None
    """The shape of the JSON it answers with on success; None for a page, a file or CSV.
    Whatever the route, an answer with a status of 400 or above is an :class:`ErrorsOut`."""
    request: type[BaseModel] | None = None
    """The shape of the JSON body of a POST, or of the query parameters of a GET."""
    unblinded: bool = False
    """Shows what a rater must not see while rating: 403 while the rater has work left
    (``app.queue.has_work_left``), by the one rule of docs/rater_app.md."""


ROUTES: tuple[Route, ...] = (
    Route("GET", "/"),
    Route("GET", "/review", unblinded=True),
    Route("GET", "/static/{path}"),
    Route("GET", "/media/{subject_uid}/{path}"),
    Route("GET", "/healthz", HealthOut),
    Route("GET", "/api/session", SessionOut),
    Route("GET", "/api/next", NextOut),
    Route("GET", "/api/back", NextOut),
    Route("POST", "/api/rating", RatingOut, RatingIn),
    Route("POST", "/api/skip", NextOut, SkipIn),
    Route("POST", "/api/events", EventsOut, EventsIn),
    Route("GET", "/api/review", ReviewOut, ReviewQuery, unblinded=True),
    Route("GET", "/api/review/{subject_uid}", SubjectOut, unblinded=True),
    Route("GET", "/api/stats", StatsOut, unblinded=True),
    Route("GET", "/api/export", unblinded=True),
)
"""Every route of the app in milestone 4 (local mode). The server serves these and no
other, and the frontend calls none that is not here: ``/`` and ``/review`` are the pages,
``/static`` the frontend files, ``/media`` the files of a render that :data:`MEDIA_FILE`
names, ``/api/export`` the ratings as CSV.

``POST /api/skip`` answers 404 for an assignment that does not exist, 403 for another
rater's, and 409 for one that is not open for the rater: one that has a submission, or
whose render is no longer the subject's (``app.queue.SkipRefused``)."""
