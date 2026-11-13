"""The tables of ``app.sqlite``, the rater app's own database. See docs/rater_app.md.

One class per table. Timestamps are UTC ISO 8601 text (``app.db.utc_now``). Text columns
with a fixed vocabulary carry a CHECK constraint, so a row that some other program wrote
cannot hold a kind, a role or a class that the app does not know.

Nothing here is overwritten when a rater revises: a revision is a new row of
``submissions`` with the next ``revision``, ``current`` moves to it, and the marks of the
superseded submission get ``deleted_ts``.

Code outside ``app/`` reads these tables with the standard library, by the names here
(``surfcensus.app.APP_TABLES``).
"""

from __future__ import annotations

from sqlalchemy import (
    CheckConstraint,
    Column,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    text,
)
from sqlmodel import Field, SQLModel

from surfcensus.app.schemas import ASSIGNMENT_KINDS, FINAL_METHODS, RATER_ROLES, UNRATABLE_CRITERION
from surfcensus.protocol import CRITERION_IDS, MARK_CLASSES
from surfcensus.render.model import PANEL_ORDER


def _one_of(column: str, values: tuple[str, ...], *, nullable: bool = False) -> CheckConstraint:
    listed = ", ".join(f"'{value}'" for value in values)
    test = f'"{column}" IN ({listed})'
    return CheckConstraint(f'"{column}" IS NULL OR {test}' if nullable else test)


def _references(target: str, *, nullable: bool = False) -> Column:
    return Column(Integer, ForeignKey(target), nullable=nullable, index=True)


class Rater(SQLModel, table=True):
    __tablename__ = "raters"
    __table_args__ = (_one_of("role", RATER_ROLES),)

    rater_id: int | None = Field(default=None, primary_key=True)
    name: str = Field(sa_column=Column(String, nullable=False, unique=True))
    token_hash: str | None = None
    """Team mode: the hash of the rater's token. None for a local rater."""
    role: str = "rater"
    active: bool = True
    created_at: str


class Assignment(SQLModel, table=True):
    """A subject given to a rater. Open until ``completed_at`` is set."""

    __tablename__ = "assignments"
    __table_args__ = (UniqueConstraint("rater_id", "seq"), _one_of("kind", ASSIGNMENT_KINDS))

    assignment_id: int | None = Field(default=None, primary_key=True)
    rater_id: int = Field(sa_column=_references("raters.rater_id"))
    subject_uid: str = Field(index=True)
    kind: str = "main"
    seq: int
    """The rater's assignments in the order they were made, from 1."""
    render_key: str
    """``renders.render_key`` of the render that was shown: the tiles the rating is of."""
    assigned_at: str
    completed_at: str | None = None
    """When the first submission was stored; a revision leaves it."""
    consensus_served_at: str | None = None
    """Calibration: when the rater was shown the consensus, after which the rating is final."""


class Submission(SQLModel, table=True):
    """One press of ``Enter``: the values, flags and note of a rating, in one revision."""

    __tablename__ = "submissions"
    __table_args__ = (
        UniqueConstraint("assignment_id", "revision"),
        # at most one current submission per assignment: the view and the export count on it
        Index(
            "ux_submissions_current",
            "assignment_id",
            unique=True,
            sqlite_where=text('"current" = 1'),
        ),
    )

    submission_id: int | None = Field(default=None, primary_key=True)
    assignment_id: int = Field(sa_column=_references("assignments.assignment_id"))
    revision: int
    """From 1."""
    current: bool = True
    """True for the latest revision of the assignment, and for no other."""
    unratable_reason: str | None = None
    other: bool = False
    note: str = ""
    protocol_version: str
    protocol_sha256: str
    calibration_sha256: str | None = None
    """Of ``calibration.yaml``, for a submission of a calibration assignment."""
    app_version: str
    client_ts: str | None = None
    server_ts: str
    ms_on_subject: int | None = None


class Rating(SQLModel, table=True):
    """The value of one criterion in one submission. An unratable submission has none."""

    __tablename__ = "ratings"
    __table_args__ = (
        UniqueConstraint("submission_id", "criterion"),
        _one_of("criterion", CRITERION_IDS),
    )

    rating_id: int | None = Field(default=None, primary_key=True)
    submission_id: int = Field(sa_column=_references("submissions.submission_id"))
    assignment_id: int = Field(sa_column=_references("assignments.assignment_id"))
    criterion: str
    value: int


class Mark(SQLModel, table=True):
    """A click mark of one submission: the point on the tile as it was posted, and what the
    server resolved it to (``app.marks.resolve_mark``)."""

    __tablename__ = "marks"
    __table_args__ = (
        _one_of("class", MARK_CLASSES),
        _one_of("panel", PANEL_ORDER),
        _one_of("mode", ("sync", "seq"), nullable=True),
        _one_of("hemi", ("lh", "rh"), nullable=True),
        _one_of("vertex_surface", ("pial", "white"), nullable=True),
    )

    mark_id: int | None = Field(default=None, primary_key=True)
    submission_id: int = Field(sa_column=_references("submissions.submission_id"))
    assignment_id: int = Field(sa_column=_references("assignments.assignment_id"))
    mark_class: str = Field(sa_column=Column("class", String, nullable=False))
    """The column is ``class``, which is no name for a Python attribute."""
    panel: str
    slice: int
    """Position in the panel's slice list of the tile that was clicked."""
    frame: int | None = None
    mode: str | None = None
    px: float
    py: float
    vox_i: float
    """Continuous voxel coordinates; likewise ``vox_j`` and ``vox_k``."""
    vox_j: float
    vox_k: float
    tkr_x: float
    tkr_y: float
    tkr_z: float
    ras_x: float
    ras_y: float
    ras_z: float
    hemi: str | None = None
    pial_vertex: int | None = None
    pial_dist_mm: float | None = None
    white_vertex: int | None = None
    white_dist_mm: float | None = None
    fsavg6_vertex: int | None = None
    vertex_surface: str | None = None
    box_dist_mm: float
    created_ts: str
    """Client time of the click when the page sent one, else the time of the submission."""
    deleted_ts: str | None = None
    """Set when a later revision superseded the mark's submission."""


class Event(SQLModel, table=True):
    """An interaction event of an assignment, or (``assignment_id`` NULL) an event of the
    run, such as a changed setting."""

    __tablename__ = "events"

    event_id: int | None = Field(default=None, primary_key=True)
    assignment_id: int | None = Field(
        default=None, sa_column=_references("assignments.assignment_id", nullable=True)
    )
    ts: str
    type: str
    payload_json: str = "{}"


class FinalLabel(SQLModel, table=True):
    """The final value of a criterion for a subject, or (criterion ``unratable``) the reason
    it cannot be rated, as the position of the reason in the protocol."""

    __tablename__ = "final_labels"
    __table_args__ = (
        _one_of("criterion", (*CRITERION_IDS, UNRATABLE_CRITERION)),
        _one_of("method", FINAL_METHODS),
    )

    subject_uid: str = Field(primary_key=True)
    criterion: str = Field(primary_key=True)
    value: int | None = None
    method: str


class FinalMark(SQLModel, table=True):
    __tablename__ = "final_marks"
    __table_args__ = (
        _one_of("class", MARK_CLASSES),
        _one_of("hemi", ("lh", "rh"), nullable=True),
    )

    final_mark_id: int | None = Field(default=None, primary_key=True)
    subject_uid: str = Field(index=True)
    mark_class: str = Field(sa_column=Column("class", String, nullable=False))
    tkr_x: float
    tkr_y: float
    tkr_z: float
    hemi: str | None = None
    pial_vertex: int | None = None
    fsavg6_vertex: int | None = None
    source_mark_ids: str = "[]"
    """JSON array of the ``mark_id``s the final mark was taken from."""


class AppMeta(SQLModel, table=True):
    """``schema_version``, ``run_id`` and ``mode``: what ties the database to its run."""

    __tablename__ = "app_meta"

    key: str = Field(primary_key=True)
    value: str


TABLES: tuple[type[SQLModel], ...] = (
    Rater,
    Assignment,
    Submission,
    Rating,
    Mark,
    Event,
    FinalLabel,
    FinalMark,
    AppMeta,
)
"""In the order of ``surfcensus.app.APP_TABLES``."""
