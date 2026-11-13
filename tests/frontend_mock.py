"""A stand-in for the server of the rater app, for the tests of the frontend.

It serves the pages and the static files as the real server is to serve them
(``app.server.PAGES``, ``STATIC_DIR``, ``STATIC_URL``), the media of renders that exist on
disk, and the routes of ``app.schemas.ROUTES`` with bodies of the shapes of that module.
Every body it receives is validated against its shape, and every body it sends is made
through one, so a page that works against it speaks the API as the schemas define it. It
keeps no database: a queue of subjects in memory, the ratings and events it was sent, and a
log of every request, which the tests read.

It is not the server: the queue is a list, marks are not resolved to vertices (unless a
``resolve`` function is given), and nothing is checked that needs the catalog. The rules of
a submission that the page mirrors (``rating.js``) are applied, so a rating that the page
lets through and this server refuses shows a disagreement.

    uv run python tests/frontend_mock.py MEDIA_DIR [MEDIA_DIR ...] [--port N] [--blind]

serves renders that exist on disk (directories with ``render.json`` and ``tiles/``) for a
look at the pages in a browser.
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel, ValidationError

import surfcensus
from surfcensus.app import schemas
from surfcensus.app.server import PAGES, STATIC_DIR, STATIC_URL, TEMPLATES_DIR
from surfcensus.protocol import Protocol, parse_protocol, shipped_protocol_bytes
from surfcensus.render import geometry
from surfcensus.render.model import RenderInfo, read_render

MEDIA_CACHE = "private, max-age=31536000, immutable"
FIRST_ASSIGNMENT = 101
MARK_ID_BASE = 500


@dataclass
class MockSubject:
    subject_uid: str
    media_dir: Path
    render_key: str
    info: RenderInfo

    @property
    def render(self) -> schemas.RenderOut:
        return schemas.render_out(self.info)


def subject_from_media(media_dir: Path, position: int) -> MockSubject:
    """A subject of the queue from a media directory that holds a render."""
    info = read_render(media_dir)
    if info is None:
        raise ValueError(f"{media_dir} holds no render of this version")
    return MockSubject(
        subject_uid=f"{position + 1:02x}" + "9b2e41c07d2e4b"[: 14],
        media_dir=Path(media_dir),
        render_key=f"{position + 1:02x}" + "0a1b2c3d4e5f6a"[: 14],
        info=info,
    )


@dataclass
class Completed:
    assignment_id: int
    subject: MockSubject
    rating: schemas.RatingIn
    marks: list[schemas.SavedMark]
    revision: int


@dataclass
class MockState:
    """What the mock server knows, and what the tests set and read."""

    subjects: list[MockSubject]
    protocol: Protocol = field(default_factory=lambda: parse_protocol(shipped_protocol_bytes()))
    require_marks: bool = False
    blind: bool = False
    mark_show_mm: float = 3.0
    examples: tuple[schemas.ExampleOut, ...] = ()
    rendering_polls: int = 0
    """How many times ``GET /api/next`` answers ``rendering`` before it gives a subject."""
    fail_ratings: int = 0
    """How many submissions are answered with status 503 before one is stored."""
    refuse_rating: tuple[int, str, str] | None = None
    """(status, field, message): the next submission is refused with it, once."""
    review_closed: bool = False
    """The unblinded routes answer 403, as while the rater has work left."""
    cache_media: bool = True
    resolve: Callable[[MockSubject, schemas.MarkIn], dict[str, Any]] | None = None
    """Gives the vertex fields of a stored mark; without it a mark has none."""

    position: int = 0
    completed: list[Completed] = field(default_factory=list)
    requests: list[tuple[str, str]] = field(default_factory=list)
    """(method, path with query) of every request, in order."""
    ratings: list[schemas.RatingIn] = field(default_factory=list)
    events: list[schemas.EventsIn] = field(default_factory=list)
    hosts: set[str] = field(default_factory=set)
    """The Host header of every request."""

    def paths(self, prefix: str = "") -> list[str]:
        return [path for _, path in self.requests if path.startswith(prefix)]

    def event_types(self) -> list[str]:
        return [event.type for batch in self.events for event in batch.events]


def _json(model: BaseModel, status: int = 200) -> JSONResponse:
    return JSONResponse(model.model_dump(mode="json", by_alias=True), status_code=status)


def _errors(status: int, where: str, message: str) -> JSONResponse:
    body = schemas.ErrorsOut(errors=(schemas.ErrorOut(field=where, message=message),))
    return _json(body, status)


def _invalid(exc: ValidationError) -> JSONResponse:
    errors = tuple(
        schemas.ErrorOut(
            field=".".join(str(part) for part in error["loc"]) or "request", message=error["msg"]
        )
        for error in exc.errors()
    )
    return _json(schemas.ErrorsOut(errors=errors), 422)


class _Mock:
    def __init__(self, state: MockState) -> None:
        self.state = state
        self.pages = Environment(
            loader=FileSystemLoader(TEMPLATES_DIR), undefined=StrictUndefined, autoescape=True
        )

    # --- the queue ------------------------------------------------------------------------

    def assignment_id(self, position: int) -> int:
        return FIRST_ASSIGNMENT + position

    def progress(self) -> schemas.ProgressOut:
        state = self.state
        done = len(state.completed)
        return schemas.ProgressOut(
            done_today=done,
            done_total=done,
            remaining=len(state.subjects) - done,
            rendered=0 if state.rendering_polls else len(state.subjects),
            n_subjects=len(state.subjects),
        )

    def assignment(
        self, position: int, previous: schemas.PreviousOut | None = None
    ) -> schemas.AssignmentOut:
        subject = self.state.subjects[position]
        return schemas.AssignmentOut(
            assignment_id=self.assignment_id(position),
            kind="main",
            subject_uid=subject.subject_uid,
            media_base=schemas.media_base(subject.subject_uid),
            render_key=subject.render_key,
            render=subject.render,
            has_vertex_map=subject.info.vertex_map is not None,
            previous=previous,
        )

    def prefetch(self, position: int) -> tuple[schemas.PrefetchOut, ...]:
        return tuple(
            schemas.PrefetchOut(
                subject_uid=subject.subject_uid,
                media_base=schemas.media_base(subject.subject_uid),
                render_key=subject.render_key,
                n_slices={name: len(panel.slices) for name, panel in subject.info.panels.items()},
                tiles=subject.info.tiles,
            )
            for subject in self.state.subjects[position + 1 : position + 3]
        )

    def next(self) -> schemas.NextOut:
        state = self.state
        if state.rendering_polls > 0:
            state.rendering_polls -= 1
            return schemas.NextOut(status="rendering", retry_after_s=0.2, progress=self.progress())
        if state.position >= len(state.subjects):
            return schemas.NextOut(status="done", progress=self.progress())
        return schemas.NextOut(
            status="assignment",
            assignment=self.assignment(state.position),
            prefetch=self.prefetch(state.position),
            progress=self.progress(),
        )

    def previous(self, done: Completed) -> schemas.PreviousOut:
        marks = tuple(
            schemas.PreviousMark(
                **{
                    **posted.model_dump(by_alias=False),
                    **saved.model_dump(exclude={"client_id"}),
                    "client_id": f"m{saved.mark_id}",
                }
            )
            for posted, saved in zip(done.rating.marks, done.marks, strict=True)
        )
        values: dict[Any, int] = dict(done.rating.values)
        return schemas.PreviousOut(
            values=values,
            unratable=done.rating.unratable,
            other=done.rating.other,
            note=done.rating.note,
            marks=marks,
        )

    # --- a rating -------------------------------------------------------------------------

    def refuse(self, rating: schemas.RatingIn) -> tuple[str, str] | None:
        """The rules of a submission that the page mirrors: (field, message), or None."""
        protocol = self.state.protocol
        if rating.unratable is not None:
            if rating.unratable not in protocol.unratable_reasons:
                return "unratable", f"{rating.unratable!r} is not a reason of the protocol"
        else:
            if set(rating.values) != set(protocol.criterion_ids()):
                return "values", "a value is needed for every criterion of the protocol"
            for criterion, value in rating.values.items():
                if value not in protocol.values():
                    return f"values.{criterion}", f"{value} is not a value of the scale"
            if self.state.require_marks:
                classes = {mark.mark_class for mark in rating.marks}
                for criterion, value in sorted(rating.values.items()):
                    if value in protocol.positive_values() and criterion not in classes:
                        return "marks", (
                            f"criterion {criterion} is rated {protocol.label(value)} but has no "
                            f"mark of class {criterion}"
                        )
        if rating.other and not rating.note.strip():
            return "note", "an other problem needs a note"
        return None

    def store(self, rating: schemas.RatingIn) -> JSONResponse:
        state = self.state
        if state.fail_ratings > 0:
            state.fail_ratings -= 1
            return _errors(503, "request", "the server is busy")
        if state.refuse_rating is not None:
            status, where, message = state.refuse_rating
            state.refuse_rating = None
            return _errors(status, where, message)
        open_id = self.assignment_id(state.position)
        last = state.completed[-1] if state.completed else None
        is_open = state.position < len(state.subjects) and rating.assignment_id == open_id
        is_revision = last is not None and rating.assignment_id == last.assignment_id
        if not is_open and not is_revision:
            return _errors(404, "assignment_id", "no such assignment is open")
        if rating.protocol_version != state.protocol.version:
            return _errors(409, "protocol_version", "the protocol changed since the page loaded")
        refused = self.refuse(rating)
        if refused is not None:
            return _errors(422, *refused)
        subject = state.subjects[state.position] if is_open else state.completed[-1].subject
        saved = []
        for number, mark in enumerate(rating.marks):
            try:
                geometry.pixel_to_voxel(subject.info.panels[mark.panel], mark.slice, mark.px, mark.py)
            except IndexError:
                return _errors(422, f"marks.{number}.slice", "the panel has no such slice")
            vertices = state.resolve(subject, mark) if state.resolve else {"box_dist_mm": 0.0}
            saved.append(
                schemas.SavedMark(
                    client_id=mark.client_id,
                    mark_id=MARK_ID_BASE + 10 * len(state.ratings) + number,
                    **vertices,
                )
            )
        state.ratings.append(rating)
        if is_open:
            state.completed.append(Completed(rating.assignment_id, subject, rating, saved, 1))
            state.position += 1
        else:
            previous = state.completed[-1]
            state.completed[-1] = Completed(
                previous.assignment_id, subject, rating, saved, previous.revision + 1
            )
        done = state.completed[-1]
        return _json(
            schemas.RatingOut(
                saved=schemas.SavedOut(
                    assignment_id=done.assignment_id,
                    submission_id=900 + len(state.ratings),
                    revision=done.revision,
                    marks=tuple(saved),
                ),
                next=self.next(),
            )
        )

    # --- review ---------------------------------------------------------------------------

    def review_rows(self) -> list[dict[str, Any]]:
        rows = []
        done = {entry.subject.subject_uid: entry for entry in self.state.completed}
        for position, subject in enumerate(self.state.subjects):
            entry = done.get(subject.subject_uid)
            rating = entry.rating if entry else None
            values = rating.values if rating and rating.unratable is None else {}
            rows.append(
                {
                    "subject_uid": subject.subject_uid,
                    "subject_label": f"sub-{position + 1:03d}",
                    "dataset": "D-0f2d1eaa",
                    "fs_version": "6.0.1" if position % 2 == 0 else "7.3.2",
                    "render_status": "done",
                    "render_reason": None,
                    "primary_rank": position + 1,
                    "composite": round(2.0 - 1.3 * position, 3),
                    "composite_component": "mass_medial",
                    "flags": "",
                    "final_A": values.get("A"),
                    "final_B": values.get("B"),
                    "final_C": values.get("C"),
                    "final_method": "single" if rating else None,
                    "unratable_reason": rating.unratable if rating else None,
                    "n_ratings": 1 if rating else 0,
                    "n_marks": len(rating.marks) if rating else 0,
                }
            )
        return rows

    def review(self, query: schemas.ReviewQuery) -> JSONResponse:
        rows = self.review_rows()
        columns = tuple(rows[0]) if rows else ("subject_uid",)
        for part in filter(None, query.filter.split(",")):
            column, _, value = part.partition(":")
            if column not in columns:
                return _errors(422, "filter", f"there is no column {column!r}")
            rows = [row for row in rows if str(row[column]) == value]
        if query.sort is not None:
            if query.sort not in columns:
                return _errors(422, "sort", f"there is no column {query.sort!r}")
            sort = query.sort
            rows.sort(key=lambda row: (row[sort] is None, row[sort]), reverse=query.order == "desc")
        page = rows[query.offset : query.offset + query.limit]
        return _json(schemas.ReviewOut(total=len(rows), columns=columns, rows=tuple(page)))

    def subject(self, subject_uid: str) -> JSONResponse:
        found = [subject for subject in self.state.subjects if subject.subject_uid == subject_uid]
        if not found:
            return _errors(404, "subject_uid", "no such subject")
        subject = found[0]
        info = subject.info
        ratings: list[schemas.ReviewRating] = []
        marks: list[schemas.ReviewMark] = []
        for entry in self.state.completed:
            if entry.subject is not subject:
                continue
            values = entry.rating.values
            ratings.append(
                schemas.ReviewRating(
                    assignment_id=entry.assignment_id,
                    rater="local",
                    kind="main",
                    render_key=subject.render_key,
                    render_current=True,
                    A=values.get("A"),
                    B=values.get("B"),
                    C=values.get("C"),
                    unratable_reason=entry.rating.unratable,
                    other=entry.rating.other,
                    note=entry.rating.note,
                    revision=entry.revision,
                    server_ts="2026-11-13T10:00:00.000Z",
                    ms_on_subject=entry.rating.ms_on_subject,
                )
            )
            for posted, saved in zip(entry.rating.marks, entry.marks, strict=True):
                vox = geometry.pixel_to_voxel(
                    info.panels[posted.panel], posted.slice, posted.px, posted.py
                )
                tkr = geometry.apply_affine(np.array(info.volume.vox2ras_tkr), vox)
                marks.append(
                    schemas.ReviewMark(
                        **saved.model_dump(exclude={"client_id"}),
                        assignment_id=entry.assignment_id,
                        rater="local",
                        mark_class=posted.mark_class,
                        panel=posted.panel,
                        slice=posted.slice,
                        px=posted.px,
                        py=posted.py,
                        vox=(float(vox[0]), float(vox[1]), float(vox[2])),
                        tkr=(float(tkr[0]), float(tkr[1]), float(tkr[2])),
                    )
                )
        cluster = schemas.ClusterOut(
            reference_id="set-cecaad3b4478",
            hemi="lh",
            region="medial",
            rank=1,
            tkr=info.box.centre,
            area_mm2=412.6,
            peak_z=4.5,
        )
        row = next(row for row in self.review_rows() if row["subject_uid"] == subject_uid)
        return _json(
            schemas.SubjectOut(
                subject_uid=subject_uid,
                media_base=schemas.media_base(subject_uid),
                render_key=subject.render_key,
                render=subject.render,
                scores={key: row[key] for key in ("primary_rank", "composite", "flags")},
                ratings=tuple(ratings),
                marks=tuple(marks),
                clusters=(cluster,),
            )
        )


def create_mock_app(state: MockState) -> FastAPI:
    """The mock server on a state. Routes: exactly those of ``schemas.ROUTES``."""
    mock = _Mock(state)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def _log(request: Request, call_next: Any) -> Response:
        query = f"?{request.url.query}" if request.url.query else ""
        state.requests.append((request.method, request.url.path + query))
        state.hosts.add(request.headers.get("host", ""))
        return await call_next(request)

    def closed() -> JSONResponse | None:
        if state.review_closed:
            return _errors(403, "request", "you still have subjects to rate")
        return None

    def page(route: str) -> HTMLResponse:
        template = mock.pages.get_template(PAGES[route])
        text = template.render(static=STATIC_URL, version=surfcensus.__version__, mode="local")
        return HTMLResponse(text, headers={"Cache-Control": "no-cache"})

    @app.get("/")
    def rate_page() -> Response:
        return page("/")

    @app.get("/review")
    def review_page() -> Response:
        return closed() or page("/review")

    @app.get("/static/{path:path}")
    def static(path: str) -> Response:
        target = (STATIC_DIR / path).resolve()
        if not target.is_relative_to(STATIC_DIR.resolve()) or not target.is_file():
            return _errors(404, "path", "no such file")
        return FileResponse(target, headers={"Cache-Control": "no-cache"})

    @app.get("/media/{subject_uid}/{path:path}")
    def media(subject_uid: str, path: str) -> Response:
        found = [subject for subject in state.subjects if subject.subject_uid == subject_uid]
        if not found or schemas.MEDIA_FILE.fullmatch(path) is None:
            return _errors(404, "path", "no such file")
        target = found[0].media_dir / path
        if not target.is_file():
            return _errors(404, "path", "no such file")
        cache = MEDIA_CACHE if state.cache_media else "no-store"
        return FileResponse(target, headers={"Cache-Control": cache})

    @app.get("/healthz")
    def health() -> Response:
        return _json(schemas.HealthOut(ok=True, version=surfcensus.__version__))

    @app.get("/api/session")
    def session() -> Response:
        return _json(
            schemas.SessionOut(
                mode="local",
                rater=schemas.RaterOut(id=1, name="local", role="lead"),
                app_version=surfcensus.__version__,
                protocol=state.protocol,
                protocol_sha256="9c1f0d6a4b7e2f3a5c8d1e0f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b",
                require_marks=state.require_marks,
                settings=schemas.SettingsOut(mark_show_mm=state.mark_show_mm, blind=state.blind),
                examples=state.examples,
            )
        )

    @app.get("/api/next")
    def next_() -> Response:
        return _json(mock.next())

    @app.get("/api/back")
    def back() -> Response:
        if not state.completed:
            return _errors(404, "request", "there is no completed assignment")
        done = state.completed[-1]
        position = state.subjects.index(done.subject)
        return _json(
            schemas.NextOut(
                status="assignment",
                assignment=mock.assignment(position, mock.previous(done)),
                progress=mock.progress(),
            )
        )

    @app.post("/api/rating")
    async def rating(request: Request) -> Response:
        try:
            body = schemas.RatingIn.model_validate_json(await request.body())
        except ValidationError as exc:
            return _invalid(exc)
        return mock.store(body)

    @app.post("/api/events")
    async def events(request: Request) -> Response:
        try:
            body = schemas.EventsIn.model_validate_json(await request.body())
        except ValidationError as exc:
            return _invalid(exc)
        state.events.append(body)
        return _json(schemas.EventsOut(stored=len(body.events)))

    @app.get("/api/review")
    def review(request: Request) -> Response:
        try:
            query = schemas.ReviewQuery.model_validate(dict(request.query_params))
        except ValidationError as exc:
            return _invalid(exc)
        return closed() or mock.review(query)

    @app.get("/api/review/{subject_uid}")
    def subject(subject_uid: str) -> Response:
        return closed() or mock.subject(subject_uid)

    @app.get("/api/stats")
    def stats() -> Response:
        rater = schemas.RaterStats(rater="local", n_completed=len(state.completed))
        return closed() or _json(schemas.StatsOut(raters=(rater,)))

    @app.get("/api/export")
    def export() -> Response:
        lines = ["assignment_id,subject_uid,rater"]
        lines += [f"{done.assignment_id},{done.subject.subject_uid},local" for done in state.completed]
        return closed() or Response("\n".join(lines) + "\n", media_type="text/csv")

    return app


@dataclass
class MockHandle:
    url: str
    port: int
    state: MockState
    _server: uvicorn.Server
    _thread: threading.Thread

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)


def start_mock(state: MockState, *, port: int = 0) -> MockHandle:
    """The mock server in a thread, on 127.0.0.1; ``port=0`` takes a free port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    port = sock.getsockname()[1]
    config = uvicorn.Config(create_mock_app(state), log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("the mock server did not start")
        time.sleep(0.01)
    return MockHandle(f"http://127.0.0.1:{port}/", port, state, server, thread)


@contextmanager
def running_mock(state: MockState, *, port: int = 0) -> Iterator[MockHandle]:
    handle = start_mock(state, port=port)
    try:
        yield handle
    finally:
        handle.stop()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("media", type=Path, nargs="+")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--blind", action="store_true")
    parser.add_argument("--require-marks", action="store_true")
    args = parser.parse_args(argv)
    subjects = [subject_from_media(path, position) for position, path in enumerate(args.media)]
    state = MockState(subjects, blind=args.blind, require_marks=args.require_marks)
    with running_mock(state, port=args.port) as handle:
        print(handle.url, flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
