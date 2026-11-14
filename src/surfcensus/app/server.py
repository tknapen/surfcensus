"""The rater app as a web server: the routes of docs/rater_app.md, and starting and
stopping it.

The app opens the catalog with ``PublicCatalog.open(run_dir)``, read-only, so it cannot
reach an accession, a subject path or an input signature. Local mode has one rater per
server process and no login.

What a server on localhost has to hold against, and how:

- Another site that resolves its own name to 127.0.0.1 (DNS rebinding): every request whose
  ``Host`` header names a host that is not allowed is answered with 403.
- Another site, or another program on another port of this machine, that sends requests
  with the rater's browser: a request for ``/api`` and every request that is not a GET is
  refused with 403 when the browser says it comes from another origin (``Origin``,
  ``Sec-Fetch-Site``), a POST needs the content type ``application/json``, and no answer
  may be embedded or framed by another origin.
- A request for a file: ``/static`` serves what lies in the static directory of the package
  and ``/media`` the tiles, the poster and the video of a ready render, under the names a
  render writes; a name that leads anywhere else, directly or through a link, is a 404.
  ``render.json``, ``vertex_map.npz``, the catalog, the configuration and the database are
  never served. A path is a route or it is not: no answer redirects to another one.
- A body that is not what a page sends: a shape of ``app.schemas`` refuses it with 422, as
  does a number that no column of the database can hold; nothing of it is stored.
- Blindness: the rating routes answer with the shapes of ``app.schemas`` that hold nothing
  of a dataset, a label, a score or another rating, and an error of those routes says
  nothing that an exception said.

One request is one transaction of ``app.sqlite`` (``app.db``): everything it writes is
committed last, or not at all.
"""

from __future__ import annotations

import errno
import html
import os
import re
import signal
import socket
import sqlite3
import statistics
import sys
import threading
import time
import webbrowser
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Collection, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlsplit

import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel, ValidationError
from sqlalchemy import func
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError
from sqlmodel import Session, col, select
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

import surfcensus
from surfcensus.app import schemas
from surfcensus.app.db import get_or_create_rater, open_app_db, store_submission
from surfcensus.app.epi_schemas import EpiIndexOut, EpiIndexSessionOut, EpiParticipantOut
from surfcensus.app.events import record_setting, store_events
from surfcensus.app.finalize import record_single, sync_unratable
from surfcensus.app.marks import MarkError, VertexIndex, check_snap_radius, resolve_mark
from surfcensus.app.marks import validate_mark as check_mark
from surfcensus.app.models import Assignment, Mark, Rater, Rating, Submission
from surfcensus.app.queue import (
    Next,
    QueuePlan,
    SkipRefused,
    build_plan,
    has_work_left,
    is_open_for,
    next_assignment,
    previous_assignment,
    skip_assignment,
    subjects_left,
    subjects_skipped,
)
from surfcensus.app.schemas import AppMode
from surfcensus.calibration import CalibrationSet, load_calibration, validate_calibration
from surfcensus.config import ConfigError, epi_settings, load_config
from surfcensus.export.tables import frame_csv, marks_frame, ratings_frame, subjects_frame
from surfcensus.log import get_logger
from surfcensus.privacy import scrub
from surfcensus.protocol import install_protocol, load_protocol, marks_required
from surfcensus.records import SUBJECT_UID_PATTERN
from surfcensus.render.model import RenderInfo, read_render
from surfcensus.render.vertexmap import read_vertex_map
from surfcensus.schemas import SCHEMAS
from surfcensus.sources.local import refuse_run_in_sources
from surfcensus.stopping import StopFlag, StopRequest
from surfcensus.store import (
    MEDIA_DIR,
    PublicCatalog,
    clusters_with_native,
    epi_participant_dir,
    media_dir,
    queue_scores,
)

DEFAULT_PORT = 8765
PORT_TRIES = 20
"""How many ports after the one asked for ``start_server`` tries when it is taken."""
LOCAL_HOSTS: tuple[str, ...] = ("127.0.0.1", "localhost")
"""The hosts a local server binds to and answers for; a request whose ``Host`` header names
another host gets 403 (the port is ignored)."""
RETRY_AFTER_S = 2.0
"""What a page is told to wait when nothing is rendered yet."""

STATIC_DIR = Path(__file__).parent / "static"
"""The frontend files, served under :data:`STATIC_URL`: ``app.css`` and ``js/*.js``, plain ES
modules that make no request outside the server."""
STATIC_URL = "/static"
TEMPLATES_DIR = Path(__file__).parent / "templates"
"""The pages: ``base.html``, and :data:`PAGES` that extend it."""
PAGES: dict[str, str] = {"/": "rate.html", "/review": "review.html"}
"""Route -> template. A page is rendered with ``static`` (:data:`STATIC_URL`, the prefix of
every script and stylesheet it loads), ``version`` (the package version, which a page
appends to those URLs as ``?v=``) and ``mode``, and with nothing about a subject:
everything else arrives through the API. The EPI pages are not among them:
:meth:`_App.epi_page` renders the index (``epi_index.html``) with the address of its API, and
the page of a participant's stream (``epi.html``) with the address of the stream, its links
and how often it asks again."""

MEDIA_CACHE = "private, max-age=31536000, immutable"
"""For a file of a render that was asked for with ``?v=<render_key>`` of the render the
subject has now: the name with that key never holds other bytes."""
NO_CACHE = "no-cache"
NO_STORE = "no-store"
MAX_BODY_BYTES = 16_000_000
"""Of a POST. The largest body the shapes allow is a full batch of events, 8 MB."""
INDEX_CACHE = 16
"""How many subjects' vertex maps are kept as KD-trees."""
RENDER_CACHE = 64
"""How many subjects' ``render.json`` are kept parsed."""

_UID = re.compile(SUBJECT_UID_PATTERN)
_JSON = "application/json"
_SAME_ORIGIN = frozenset({"same-origin", "none"})
_BUSY = "the database of the app is in use or cannot be written; try again"
_FAILED = "the server failed on this request; its log says why"
_SETTINGS = ("blind", "require_marks", "mark_max_dist_mm")
# no template: the pages are the two of PAGES, and this is the refusal of one of them
_CLOSED_PAGE = (
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    "<title>surfcensus: {title}</title>\n"
    '<link rel="icon" href="data:,">\n<link rel="stylesheet" href="{static}/app.css">\n'
    '</head>\n<body>\n<main class="review">\n<p class="message" role="status">{message}</p>\n'
    '<p><a href="/">Back to the rating screen</a></p>\n</main>\n</body>\n</html>\n'
)
_INT64 = 2**63
_NO_ROUTE = "there is no such route"
_CLOSED = {
    "review": "you still have subjects to rate; the review opens when none is left",
    "epi": "you still have subjects to rate; the EPI movies open when none is left",
}
EPI_INDEX_URL = "/epi"
EPI_API = "/api/epi"
EPI_MEDIA_URL = "/epi-media"
EPI_POLL_S = 5
"""How often, in seconds, the EPI pages of a participant whose packs are still being made
ask the server again (the stream grows behind the app)."""
EPI_POLL_IDLE_S = 30
"""How often, in seconds, the EPI page of a participant with no pack to be made asks the
server again: packs can still be made later (``surfcensus pack`` in another process, a later
``review``), and a page left open notices them."""
EPI_MEMORY_MIB = 512
"""The decoded-chunk limit the EPI page of a run holds for a whole stream (that of
``surfcensus epi-session --serve``)."""
_EPI_PARTICIPANTS_S = 1.0
"""How long the participants of the catalog are taken as known between two requests (the
files of a pack are asked for many at a time)."""

_Model = TypeVar("_Model", bound=BaseModel)


# --- answers ------------------------------------------------------------------------------------


def _json(
    model: BaseModel, status: int = 200, headers: Mapping[str, str] | None = None
) -> Response:
    return JSONResponse(
        model.model_dump(mode="json", by_alias=True),
        status_code=status,
        headers={"Cache-Control": NO_STORE, **(headers or {})},
    )


def _errors(status: int, problems: Sequence[tuple[str, str]]) -> Response:
    body = schemas.ErrorsOut(
        errors=tuple(schemas.ErrorOut(field=where, message=message) for where, message in problems)
    )
    return _json(body, status)


class _Refused(Exception):
    """A request that is answered with a status of 400 or above and an ``ErrorsOut``."""

    def __init__(self, status: int, where: str, message: str) -> None:
        super().__init__(f"{status} {where}: {message}")
        self.status = status
        self.problems: list[tuple[str, str]] = [(where, message)]

    @classmethod
    def of(cls, status: int, problems: Sequence[tuple[str, str]]) -> _Refused:
        refused = cls(status, *problems[0])
        refused.problems = list(problems)
        return refused


def _invalid(exc: ValidationError) -> _Refused:
    return _Refused.of(
        422,
        [
            (".".join(str(part) for part in error["loc"]) or "request", str(error["msg"]))
            for error in exc.errors()
        ],
    )


def _beyond_a_column(value: object, where: str, found: list[tuple[str, str]]) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        if not -_INT64 <= value < _INT64:
            found.append((where or "request", "the number is too large to be stored"))
    elif isinstance(value, Mapping):
        for key, inner in value.items():
            _beyond_a_column(inner, f"{where}.{key}" if where else str(key), found)
    elif isinstance(value, list | tuple):
        for number, inner in enumerate(value):
            _beyond_a_column(inner, f"{where}.{number}" if where else str(number), found)


def _parse(shape: type[_Model], body: bytes, *, as_text: Collection[str] = ()) -> _Model:
    """The body of a POST in its shape: 422 with the fields the shape refuses, and for an
    integer that no column of the database can hold (JSON has no largest number, SQLite
    has). The fields named in ``as_text`` are stored as text and are not looked at."""
    try:
        parsed = shape.model_validate_json(body)
    except ValidationError as exc:
        raise _invalid(exc) from None
    found: list[tuple[str, str]] = []
    _beyond_a_column(parsed.model_dump(by_alias=True, exclude=set(as_text)), "", found)
    if found:
        raise _Refused.of(422, found)
    return parsed


async def _body(request: Request) -> bytes:
    """The body of a POST. 415 unless it is sent as JSON, which a page of another origin
    cannot do without asking first; 413 above :data:`MAX_BODY_BYTES`."""
    kind = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if kind != _JSON:
        raise _Refused(415, "request", f"the body of a POST is sent as {_JSON}")
    too_large = _Refused(413, "request", f"a body of at most {MAX_BODY_BYTES} bytes is taken")
    declared = request.headers.get("content-length", "")
    if declared.isascii() and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        raise too_large
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY_BYTES:
            raise too_large
        chunks.append(chunk)
    return b"".join(chunks)


def _plain(value: object) -> Any:
    """A cell of a frame as JSON holds it: None for a missing value, no numpy type."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if isinstance(value, float) and value != value:
        return None
    return value


def _cell_text(value: object) -> str:
    """A cell as the filter of the review table compares it."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _matches(value: object, wanted: str) -> bool:
    if _cell_text(value) == wanted:
        return True
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return float(wanted) == float(value)
    except ValueError:
        return False


# --- what guards every request ------------------------------------------------------------------


def _host_name(header: str) -> str | None:
    """The host of a ``Host`` header or of the authority of an origin, lower case and
    without its port; None for a value that is neither ``host`` nor ``host:port``."""
    value = header.strip().lower()
    if value.startswith("["):
        end = value.find("]")
        name, rest = (value[: end + 1], value[end + 1 :]) if end > 0 else ("", "")
    else:
        name, colon, port = value.partition(":")
        rest = colon + port
    if not name or (rest and not (rest[0] == ":" and rest[1:].isascii() and rest[1:].isdigit())):
        return None
    return name


def _security_headers(media_base_url: str | None) -> tuple[tuple[bytes, bytes], ...]:
    """Sent with every answer. A page loads scripts, styles and images from the server it
    came from and from nowhere else, but for the media of a run that has ``media_base_url``;
    no other origin may frame it, embed its answers or keep a handle on its window."""
    sources = "'self'"
    if media_base_url is not None:
        parts = urlsplit(media_base_url)
        if parts.scheme in ("http", "https") and parts.netloc:
            sources += f" {parts.scheme}://{parts.netloc}"
    policy = "; ".join(
        (
            "default-src 'self'",
            f"img-src {sources} data:",
            f"media-src {sources}",
            "object-src 'none'",
            "base-uri 'none'",
            "form-action 'self'",
            "frame-ancestors 'none'",
        )
    )
    headers = {
        "Content-Security-Policy": policy,
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "same-origin",
        "Cross-Origin-Resource-Policy": "same-origin",
        "Cross-Origin-Opener-Policy": "same-origin",
    }
    return tuple((name.lower().encode(), value.encode()) for name, value in headers.items())


class _Guard:
    """Refuses, before any route is looked at, a request for another host and a request of
    another origin that could change or read anything; adds the security headers to every
    answer."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        allowed_hosts: frozenset[str],
        headers: tuple[tuple[bytes, bytes], ...],
    ) -> None:
        self.app = app
        self.allowed_hosts = allowed_hosts
        self.headers = headers

    def refusal(self, scope: Scope) -> str | None:
        """Why the request is not answered, or None."""
        found: dict[str, list[str]] = {}
        for name, value in scope["headers"]:
            found.setdefault(name.decode("latin-1").lower(), []).append(value.decode("latin-1"))
        hosts = found.get("host", [])
        name = _host_name(hosts[0]) if len(hosts) == 1 else None
        if name is None or name not in self.allowed_hosts:
            return "this server does not answer for that host"
        path, method = str(scope.get("path", "")), str(scope.get("method", "GET")).upper()
        if method in ("GET", "HEAD") and not path.startswith("/api/"):
            # a page, a script, a tile: a link from elsewhere may lead to a page, and no
            # answer of these can be read or embedded by another origin (the headers)
            return None
        for origin in found.get("origin", []):
            scheme, _, authority = origin.strip().lower().partition("://")
            if scheme not in ("http", "https") or authority != hosts[0].strip().lower():
                return "requests from another origin are not taken"
        for site in found.get("sec-fetch-site", []):
            if site.strip().lower() not in _SAME_ORIGIN:
                return "requests from another origin are not taken"
        return None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_guarded(message: Message) -> None:
            if message["type"] == "http.response.start":
                message = {**message, "headers": [*message.get("headers", []), *self.headers]}
            await send(message)

        refusal = self.refusal(scope)
        if refusal is not None:
            await _errors(403, [("request", refusal)])(scope, receive, send_guarded)
            return
        await self.app(scope, receive, send_guarded)


# --- the state of one server --------------------------------------------------------------------


@dataclass(frozen=True)
class _Ready:
    """A render that can be shown: its key and its ``render.json``."""

    render_key: str
    info: RenderInfo


class _NotRendered(Exception):
    """A subject of the calibration set has no ready render to check its examples against."""


class _Lru:
    """A small cache of values that are costly to read and are not changed afterwards."""

    def __init__(self, size: int) -> None:
        self._size = size
        self._items: OrderedDict[object, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: object, make: Callable[[], Any]) -> Any:
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        value = make()
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self._size:
                self._items.popitem(last=False)
        return value


def _stamp(path: Path) -> tuple[int, int] | None:
    """What tells one state of a file from the next, or None for a file that is not there."""
    try:
        status = path.stat()
    except OSError:
        return None
    return status.st_mtime_ns, status.st_size


def _local_day(timestamp: str) -> str | None:
    """The day of a stored time on the clock of this machine."""
    try:
        return datetime.fromisoformat(timestamp).astimezone().date().isoformat()
    except ValueError:
        return None


class _App:
    """What the routes of one server share: the run, its protocol, its database, its rater."""

    def __init__(
        self,
        run_dir: Path | str,
        *,
        mode: AppMode,
        rater: str | None,
        allowed_hosts: Collection[str] | None,
    ) -> None:
        if mode == "team":
            raise ConfigError(
                "invalid",
                "team mode (tokens, several raters at once, adjudication) is milestone 5; "
                "this version serves a run in local mode",
            )
        if mode != "local":
            raise ConfigError("invalid", f"mode is local or team, not {mode!r}")
        self.run_dir = Path(run_dir).absolute()
        self.mode: AppMode = mode
        self.config = load_config(self.run_dir)
        # before anything is written: a run that was moved into its source stays as it is
        refuse_run_in_sources(self.run_dir, self.config)
        app = self.config.app
        try:
            check_snap_radius(app.mark_max_dist_mm, self.config.render.vertex_map_margin_mm)
        except ValueError as exc:
            raise ConfigError(
                "invalid",
                f"{exc}; lower app.mark_max_dist_mm in config.yaml to at most "
                f"{self.config.render.vertex_map_margin_mm:g}",
            ) from exc
        hosts = LOCAL_HOSTS if allowed_hosts is None else allowed_hosts
        self.allowed_hosts = frozenset(host.strip().lower() for host in hosts if host.strip())
        if not self.allowed_hosts:
            raise ConfigError("invalid", "the server needs at least one host to answer for")
        self.render_hash = self.config.render.config_hash()
        self.version = surfcensus.__version__
        self.pages = Environment(
            loader=FileSystemLoader(TEMPLATES_DIR), undefined=StrictUndefined, autoescape=True
        )
        self._renders = _Lru(RENDER_CACHE)
        self._indices = _Lru(INDEX_CACHE)
        self._scores: tuple[object, pd.DataFrame | None] | None = None
        self._synced: object = None
        self._lock = threading.Lock()
        self._closed = False
        self._packs: dict[str, Any] = {}
        self._epi_participants: tuple[float, pd.DataFrame] | None = None

        self.public = PublicCatalog.open(self.run_dir)
        # the EPI pages: a local run whose EPI interface is on (a census run has no EPI
        # phases, and its routes answer as routes that do not exist)
        self.epi_on = self.public.mode() == "local" and epi_settings(self.config).enabled
        self.engine: Engine | None = None
        try:
            install_protocol(self.run_dir)
            self.protocol, self.protocol_sha256 = load_protocol(self.run_dir)
            self.require_marks = marks_required(self.protocol, app)
            loaded = load_calibration(self.run_dir)
            self.calibration, self.calibration_sha256 = (None, None) if loaded is None else loaded
            if self.calibration is not None:
                self._check_calibration(self.calibration)
            self.engine = open_app_db(self.run_dir, run_id=self.public.run_id(), mode=mode)
            name = app.rater_name if rater is None else rater
            plan = self.plan()
            with Session(self.engine) as session:
                earlier = sorted(session.exec(select(Rater.name)).all())
                try:
                    made = get_or_create_rater(session, name, role="lead")
                except ValueError as exc:
                    raise ConfigError("invalid", str(exc)) from exc
                if rater is None and earlier and name not in earlier:
                    # opening the app is the first use of the default rater: say that it
                    # made one, since a rater more closes the unblinded screens to whoever
                    # has subjects left and is listed by `finalize` from then on
                    get_logger().warning(
                        "no rater was named, so the app rates as a new rater, %s, beside %s: "
                        "stop it and name the rater (--rater NAME) to go on as one of them",
                        name,
                        ", ".join(earlier),
                    )
                assert made.rater_id is not None
                self.rater_id = made.rater_id
                self.rater_name = made.name
                self._sync(session, plan)
                settings = {
                    "blind": app.blind,
                    "require_marks": self.require_marks,
                    "mark_max_dist_mm": app.mark_max_dist_mm,
                }
                for setting in _SETTINGS:
                    record_setting(session, setting, settings[setting])
                session.commit()
            self._synced = self._sync_state(plan)
        except (DBAPIError, sqlite3.Error) as exc:
            self.close()
            raise ConfigError("invalid", f"app.sqlite cannot be used: {scrub(str(exc))}") from exc
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if self.engine is not None:
            self.engine.dispose()
        self.public.close()

    # --- the run as it is now -------------------------------------------------------------------

    def scores(self) -> pd.DataFrame | None:
        """The primary scores of the run, or None for a run that has none this version
        reads; read again when the table changed."""
        stamp = _stamp(self.run_dir / "tables" / "scores.parquet")
        cached = self._scores
        if cached is not None and cached[0] == stamp:
            return cached[1]
        scores = queue_scores(self.run_dir)
        self._scores = (stamp, scores)
        return scores

    def plan(self) -> QueuePlan:
        return build_plan(
            self.public,
            self.scores(),
            self.config.app,
            self.mode,
            render_config_hash=self.render_hash,
            calibration=self.calibration,
        )

    def render_info(self, subject_uid: str) -> RenderInfo | None:
        """The ``render.json`` of the subject when it is one of the present configuration."""
        directory = media_dir(self.run_dir, subject_uid)
        stamp = _stamp(directory / "render.json")
        if stamp is None:
            return None
        info = self._renders.get((subject_uid, stamp), lambda: read_render(directory))
        if info is None or info.config_hash != self.render_hash:
            return None
        return info

    def ready(self, subject_uid: str) -> _Ready | None:
        """The render of a subject of the catalog when it can be shown: recorded as done
        under the present configuration, with a key, and readable. Selected or not."""
        row = self.public.render(subject_uid)
        if row is None or row.status != "done" or row.config_hash != self.render_hash:
            return None
        if not row.render_key:
            return None
        info = self.render_info(subject_uid)
        return None if info is None else _Ready(row.render_key, info)

    def _check_calibration(self, calibration: CalibrationSet) -> None:
        """``ConfigError`` for a calibration set that the protocol and the run cannot use
        (``calibration.validate_calibration``). The example slices of a subject are
        checked against its render when it has one that is ready: a server may start
        while renders are awaited (decision D33), and such a subject's examples are shown
        only once its render is there (:meth:`examples`)."""
        known = self.public.subject_uids(present=None)

        def render(subject_uid: str) -> RenderInfo:
            ready = self.ready(subject_uid)
            if ready is None:
                raise _NotRendered(subject_uid)
            return ready.info

        for subject in calibration.subjects:
            alone = CalibrationSet(version=calibration.version, subjects=(subject,))
            with suppress(_NotRendered):
                validate_calibration(alone, self.protocol, subjects=known, render=render)

    def vertex_index(
        self, subject_uid: str, render_key: str, info: RenderInfo
    ) -> VertexIndex | None:
        """The vertices of the subject's vertex map, or None when the render lists none or
        the file does not read as one: a mark is then stored as a point."""
        if info.vertex_map is None:
            return None
        path = media_dir(self.run_dir, subject_uid) / info.vertex_map.file
        stamp = _stamp(path)
        if stamp is None:
            get_logger().warning("%s has no vertex map on disk", subject_uid)
            return None
        try:
            return self._indices.get(
                (subject_uid, render_key, stamp), lambda: VertexIndex(read_vertex_map(path))
            )
        except (OSError, ValueError, KeyError) as exc:
            get_logger().warning(
                "the vertex map of %s cannot be read (%s); its marks are stored as points",
                subject_uid,
                scrub(str(exc)),
            )
            return None

    def media_base(self, subject_uid: str) -> str:
        return schemas.media_base(subject_uid, self.config.app.media_base_url)

    # --- one request, one transaction -----------------------------------------------------------

    @contextmanager
    def session(self) -> Iterator[tuple[Session, Rater]]:
        assert self.engine is not None
        with Session(self.engine) as session:
            rater = session.get(Rater, self.rater_id)
            if rater is None:
                raise _Refused(403, "request", "the rater of this server is not in the database")
            yield session, rater

    def _sync_state(self, plan: QueuePlan) -> object:
        return (tuple(sorted(plan.failed.items())), len(plan.render_keys), plan.pending)

    def _sync(self, session: Session, plan: QueuePlan) -> None:
        sync_unratable(session, plan, self.protocol)

    def advance(self, session: Session, rater: Rater, plan: QueuePlan) -> Next:
        """What the rater gets now. Before that, the subjects whose render has failed since
        the labels were last looked at get their unratable label, so that renders made
        while the server runs are labelled like those made before it started."""
        if self._sync_state(plan) != self._synced:
            self._sync(session, plan)
        return next_assignment(session, rater, plan)

    def committed(self, session: Session, plan: QueuePlan) -> None:
        session.commit()
        self._synced = self._sync_state(plan)

    def closed_to(self, session: Session, rater: Rater, plan: QueuePlan) -> bool:
        """Whether the unblinded routes are closed to the rater now."""
        n_raters = session.exec(select(func.count()).select_from(Rater)).one()
        applies = self.mode == "team" or self.config.app.blind or n_raters > 1
        return applies and has_work_left(session, rater, plan)

    def gate(self, what: str = "review") -> None:
        """403 for an unblinded route while it is closed to the rater (:meth:`closed_to`);
        ``what`` names the screens in the refusal."""
        plan = self.plan()
        with self.session() as (session, rater):
            closed = self.closed_to(session, rater, plan)
        if closed:
            raise _Refused(403, "request", _CLOSED[what])

    def epi_gate(self) -> None:
        """:meth:`gate` for the EPI routes, which a page asks many files of: the queue is
        looked at only where the rule can close the routes (a blind run, or several raters)."""
        with self.session() as (session, rater):
            n_raters = session.exec(select(func.count()).select_from(Rater)).one()
            if not (self.config.app.blind or n_raters > 1):
                return
            closed = has_work_left(session, rater, self.plan())
        if closed:
            raise _Refused(403, "request", _CLOSED["epi"])

    # --- shapes ---------------------------------------------------------------------------------

    def progress(
        self, session: Session, rater: Rater, plan: QueuePlan, shown: Assignment | None
    ) -> schemas.ProgressOut:
        completed = session.exec(
            select(Assignment.completed_at).where(
                Assignment.rater_id == self.rater_id, col(Assignment.completed_at).is_not(None)
            )
        ).all()
        today = datetime.now().astimezone().date().isoformat()
        ready = len(plan.render_keys)
        return schemas.ProgressOut(
            done_today=sum(_local_day(at or "") == today for at in completed),
            done_total=len(completed),
            remaining=subjects_left(session, rater, plan),
            skipped=subjects_skipped(session, rater, plan, shown),
            rendered=ready,
            # a subject whose render failed is never offered, so it is not waited for
            n_subjects=ready + plan.pending + plan.outdated,
        )

    def previous(self, session: Session, assignment: Assignment) -> schemas.PreviousOut | None:
        """The current submission of the assignment, as a page gets it back for revision."""
        submission = _current_submission(session, assignment)
        if submission is None:
            return None
        values = session.exec(
            select(Rating.criterion, Rating.value).where(
                Rating.submission_id == submission.submission_id
            )
        ).all()
        marks = session.exec(
            select(Mark)
            .where(Mark.submission_id == submission.submission_id, col(Mark.deleted_ts).is_(None))
            .order_by(col(Mark.mark_id))
        ).all()
        return schemas.PreviousOut.model_validate(
            {
                "values": dict(values),
                "unratable": submission.unratable_reason,
                "other": bool(submission.other),
                "note": submission.note or "",
                "marks": [
                    {
                        "client_id": f"m{mark.mark_id}",
                        "mark_id": mark.mark_id,
                        "class": mark.mark_class,
                        "panel": mark.panel,
                        "slice": mark.slice,
                        "px": mark.px,
                        "py": mark.py,
                        "frame": mark.frame,
                        "mode": mark.mode,
                        "ts": mark.created_ts,
                        **_vertices(mark),
                    }
                    for mark in marks
                ],
            }
        )

    def assignment_out(
        self, assignment: Assignment, info: RenderInfo, previous: schemas.PreviousOut | None
    ) -> schemas.AssignmentOut:
        assert assignment.assignment_id is not None
        uid = assignment.subject_uid
        listed = info.vertex_map
        return schemas.AssignmentOut(
            assignment_id=assignment.assignment_id,
            # a repeat looks like a new subject
            kind="calibration" if assignment.kind == "calibration" else "main",
            subject_uid=uid,
            media_base=self.media_base(uid),
            render_key=assignment.render_key,
            render=schemas.render_out(info),
            has_vertex_map=listed is not None
            and (media_dir(self.run_dir, uid) / listed.file).is_file(),
            previous=previous,
        )

    def shown_render(self, assignment: Assignment) -> RenderInfo:
        """The render an assignment shows. 503 when the catalog says it is there and the
        run directory does not have it."""
        info = self.render_info(assignment.subject_uid)
        if info is None:
            raise _Refused(
                503,
                "request",
                "the render of the next subject is recorded as done and cannot be read from "
                "the run directory; run `surfcensus render` on the run",
            )
        return info

    def prefetch(self, ahead: Sequence[tuple[str, str]]) -> tuple[schemas.PrefetchOut, ...]:
        out = []
        for subject_uid, render_key in ahead:
            info = self.render_info(subject_uid)
            if info is None:
                continue
            out.append(
                schemas.PrefetchOut(
                    subject_uid=subject_uid,
                    media_base=self.media_base(subject_uid),
                    render_key=render_key,
                    n_slices={name: len(panel.slices) for name, panel in info.panels.items()},
                    tiles=info.tiles,
                )
            )
        return tuple(out)

    def next_out(
        self, session: Session, rater: Rater, plan: QueuePlan, following: Next
    ) -> schemas.NextOut:
        assignment = following.assignment
        if following.status != "assignment" or assignment is None:
            return schemas.NextOut(
                status=following.status,
                retry_after_s=RETRY_AFTER_S if following.status == "rendering" else None,
                progress=self.progress(session, rater, plan, None),
            )
        info = self.shown_render(assignment)
        return schemas.NextOut(
            status="assignment",
            assignment=self.assignment_out(assignment, info, None),
            prefetch=self.prefetch(following.prefetch),
            progress=self.progress(session, rater, plan, assignment),
        )

    # --- the rating routes ----------------------------------------------------------------------

    def get_session(self) -> Response:
        plan = self.plan()
        with self.session() as (session, rater):
            examples = self.examples(session, plan)
            offered = self.epi_offered() and not self.closed_to(session, rater, plan)
            body = schemas.SessionOut(
                mode=self.mode,
                rater=schemas.RaterOut.model_validate(
                    {"id": self.rater_id, "name": rater.name, "role": rater.role}
                ),
                app_version=self.version,
                protocol=self.protocol,
                protocol_sha256=self.protocol_sha256,
                require_marks=self.require_marks,
                settings=schemas.SettingsOut(
                    mark_show_mm=self.config.app.mark_show_mm, blind=self.config.app.blind
                ),
                examples=examples,
                epi_url=EPI_INDEX_URL if offered else None,
            )
        return _json(body)

    def examples(self, session: Session, plan: QueuePlan) -> tuple[schemas.ExampleOut, ...]:
        """The reference slices of the calibration set, for a rater who has rated every
        subject of the set that the queue will give: nobody is shown the consensus of a
        subject before rating it."""
        if self.calibration is None:
            return ()
        done = {
            (uid, key)
            for uid, key in session.exec(
                select(Assignment.subject_uid, Assignment.render_key).where(
                    Assignment.rater_id == self.rater_id,
                    col(Assignment.completed_at).is_not(None),
                    col(Assignment.kind).in_(("main", "calibration")),
                )
            ).all()
        }
        selected = set(plan.order)
        for subject in self.calibration.subjects:
            uid = subject.subject_uid
            if uid not in selected or uid in plan.failed:
                continue
            if (uid, plan.render_keys.get(uid)) not in done:
                return ()
        out = []
        for subject in self.calibration.subjects:
            ready = self.ready(subject.subject_uid) if subject.examples else None
            if ready is None:
                continue
            for example in subject.examples:
                if example.slice >= len(ready.info.panels[example.panel].slices):
                    get_logger().warning(
                        "calibration.yaml: an example of %s names slice %d of the %s panel, "
                        "which its render does not have; it is not shown",
                        subject.subject_uid,
                        example.slice,
                        example.panel,
                    )
                    continue
                out.append(
                    schemas.ExampleOut(
                        criterion=example.criterion,
                        value=example.value,
                        panel=example.panel,
                        slice=example.slice,
                        media_base=self.media_base(subject.subject_uid),
                        render_key=ready.render_key,
                        caption=example.caption,
                        tiles=ready.info.tiles,
                    )
                )
        return tuple(out)

    def get_next(self) -> Response:
        plan = self.plan()
        with self.session() as (session, rater):
            body = self.next_out(session, rater, plan, self.advance(session, rater, plan))
            self.committed(session, plan)
        return _json(body)

    def get_back(self) -> Response:
        plan = self.plan()
        with self.session() as (session, rater):
            assignment = previous_assignment(session, rater)
            if assignment is None:
                raise _Refused(404, "request", "there is no earlier subject to go back to")
            if assignment.kind == "calibration" and assignment.consensus_served_at is not None:
                raise _Refused(
                    409,
                    "request",
                    "the consensus of that calibration subject was shown; its rating is final",
                )
            if plan.render_keys.get(assignment.subject_uid) != assignment.render_key:
                raise _Refused(
                    409,
                    "request",
                    "the render of the subject before has changed since; it cannot be shown again",
                )
            info = self.shown_render(assignment)
            body = schemas.NextOut(
                status="assignment",
                assignment=self.assignment_out(
                    assignment, info, self.previous(session, assignment)
                ),
                progress=self.progress(session, rater, plan, assignment),
            )
        return _json(body)

    def _owned(self, session: Session, assignment_id: int) -> Assignment:
        """The rater's assignment of that id: 404 when there is none, 403 for another
        rater's."""
        assignment = session.get(Assignment, assignment_id)
        if assignment is None:
            raise _Refused(404, "assignment_id", f"there is no assignment {assignment_id}")
        if assignment.rater_id != self.rater_id:
            raise _Refused(403, "assignment_id", "that assignment is another rater's")
        return assignment

    def post_rating(self, body: bytes) -> Response:
        rating = _parse(schemas.RatingIn, body)
        plan = self.plan()
        with self.session() as (session, rater):
            # rule 1: whose it is, and that the page is not behind the server
            assignment = self._owned(session, rating.assignment_id)
            again = self._sent_before(session, assignment, rating)
            if assignment.completed_at is not None and again is None:
                last = previous_assignment(session, rater)
                if last is None or last.assignment_id != assignment.assignment_id:
                    raise _Refused(
                        409,
                        "assignment_id",
                        "only the subject left last can be rated again",
                    )
                if assignment.kind == "calibration" and assignment.consensus_served_at is not None:
                    raise _Refused(
                        409,
                        "assignment_id",
                        "the consensus of that calibration subject was shown; its rating is final",
                    )
            if rating.protocol_version != self.protocol.version:
                raise _Refused(
                    409,
                    "protocol_version",
                    f"the protocol is {self.protocol.version}, the page has "
                    f"{rating.protocol_version}; reload the page",
                )
            if plan.render_keys.get(assignment.subject_uid) != assignment.render_key:
                raise _Refused(
                    409,
                    "assignment_id",
                    "the render of the subject has changed since the page loaded it",
                )
            if again is not None:
                submission, stored = again
            else:
                info = self.shown_render(assignment)
                self._check(rating, info)
                marks: list[schemas.ResolvedMark] = []
                if rating.unratable is None and rating.marks:
                    index = self.vertex_index(assignment.subject_uid, assignment.render_key, info)
                    marks = [self._resolve(info, index, mark) for mark in rating.marks]
                submission, stored = store_submission(
                    session,
                    assignment,
                    rating,
                    marks,
                    protocol_sha256=self.protocol_sha256,
                    app_version=self.version,
                    calibration_sha256=(
                        self.calibration_sha256 if assignment.kind == "calibration" else None
                    ),
                )
                record_single(session, assignment.subject_uid, self.protocol)
            assert submission.submission_id is not None
            saved = schemas.SavedOut(
                assignment_id=rating.assignment_id,
                submission_id=submission.submission_id,
                revision=submission.revision,
                marks=tuple(
                    schemas.SavedMark.model_validate(
                        {"client_id": posted.client_id, "mark_id": mark.mark_id, **_vertices(mark)}
                    )
                    # the marks of an unratable rating are not stored
                    for posted, mark in zip(
                        () if rating.unratable is not None else rating.marks, stored, strict=True
                    )
                ),
            )
            following = self.next_out(session, rater, plan, self.advance(session, rater, plan))
            self.committed(session, plan)
        return _json(schemas.RatingOut(saved=saved, next=following))

    def _sent_before(
        self, session: Session, assignment: Assignment, rating: schemas.RatingIn
    ) -> tuple[Submission, list[Mark]] | None:
        """The stored submission when ``rating`` is one press of ``Enter`` that arrives a
        second time: the page sends a rating again until it has an answer, and one press is
        one submission. A rating is known again by the client's time of the press and by
        its content: the values, the note, the flags, the time on the subject and, of each
        mark, its class, panel, slice, pixel, frame, mode and its time when one was sent.
        A body that differs in any of them is a revision."""
        if rating.client_ts is None or assignment.completed_at is None:
            return None
        current = _current_submission(session, assignment)
        if current is None or current.client_ts != rating.client_ts:
            return None
        marks = list(
            session.exec(
                select(Mark)
                .where(Mark.submission_id == current.submission_id, col(Mark.deleted_ts).is_(None))
                .order_by(col(Mark.mark_id))
            ).all()
        )
        posted = () if rating.unratable is not None else rating.marks
        values = dict(
            session.exec(
                select(Rating.criterion, Rating.value).where(
                    Rating.submission_id == current.submission_id
                )
            ).all()
        )
        same = (
            current.unratable_reason == rating.unratable
            and current.note == rating.note
            and bool(current.other) == rating.other
            and current.ms_on_subject == rating.ms_on_subject
            and values == ({} if rating.unratable is not None else rating.values)
            and len(marks) == len(posted)
            and all(
                (mark.mark_class, mark.panel, mark.slice, mark.px, mark.py, mark.frame, mark.mode)
                == (
                    sent.mark_class,
                    sent.panel,
                    sent.slice,
                    sent.px,
                    sent.py,
                    sent.frame,
                    sent.mode,
                )
                # a mark without a time of its own was stored with the server's
                and (sent.ts is None or mark.created_ts == sent.ts)
                for mark, sent in zip(marks, posted, strict=True)
            )
        )
        if not same:
            return None
        return current, (marks if rating.unratable is None else [])

    def _check(self, rating: schemas.RatingIn, info: RenderInfo) -> None:
        """Rules 2 to 5 of a submission, in that order; the first that fails is reported,
        with every entry that fails it."""
        protocol = self.protocol
        if rating.unratable is not None:
            if rating.unratable not in protocol.unratable_reasons:
                raise _Refused(
                    422,
                    "unratable",
                    f"{rating.unratable!r} is not an unratable reason of the protocol",
                )
        else:
            criteria = protocol.criterion_ids()
            if set(rating.values) != set(criteria):
                raise _Refused(
                    422,
                    "values",
                    f"a value is needed for exactly the criteria {', '.join(criteria)}, got "
                    f"{', '.join(sorted(rating.values)) or 'none'}",
                )
            scale = protocol.values()
            off = [
                (f"values.{criterion}", f"{rating.values[criterion]} is not a value of the scale")
                for criterion in criteria
                if rating.values[criterion] not in scale
            ]
            if off:
                raise _Refused.of(422, off)
            wrong = []
            for number, mark in enumerate(rating.marks):
                try:
                    check_mark(info, mark)
                except MarkError as exc:
                    wrong.append((f"marks.{number}.{exc.field}", exc.message))
            if wrong:
                raise _Refused.of(422, wrong)
            if self.require_marks:
                classes = {mark.mark_class for mark in rating.marks}
                positive = protocol.positive_values()
                unmarked = [
                    (
                        "marks",
                        f"criterion {criterion} is rated "
                        f"{protocol.label(rating.values[criterion])} but has no mark of class "
                        f"{criterion}",
                    )
                    for criterion in criteria
                    if rating.values[criterion] in positive and criterion not in classes
                ]
                if unmarked:
                    raise _Refused.of(422, unmarked)
        if rating.other and not rating.note.strip():
            raise _Refused(422, "note", "an other problem needs a note that says what it is")

    def _resolve(
        self, info: RenderInfo, index: VertexIndex | None, mark: schemas.MarkIn
    ) -> schemas.ResolvedMark:
        radius = self.config.app.mark_max_dist_mm
        if index is not None:
            try:
                return resolve_mark(info, index, mark, max_dist_mm=radius)
            except MarkError:
                raise
            except ValueError as exc:
                # the file is not the vertex map of this render.json: one of the two is
                # being written again
                get_logger().warning("a mark is stored as a point: %s", scrub(str(exc)))
        return resolve_mark(info, None, mark, max_dist_mm=radius)

    def post_skip(self, body: bytes) -> Response:
        skip = _parse(schemas.SkipIn, body)
        plan = self.plan()
        not_open = _Refused(
            409,
            "assignment_id",
            "the assignment is not open: it has a rating, or the render of its subject changed",
        )
        with self.session() as (session, rater):
            assignment = self._owned(session, skip.assignment_id)
            if not is_open_for(rater, assignment, plan):
                raise not_open
            try:
                following = skip_assignment(session, rater, assignment, skip, plan)
            except SkipRefused:
                raise not_open from None
            out = self.next_out(session, rater, plan, following)
            self.committed(session, plan)
        return _json(out)

    def post_events(self, body: bytes) -> Response:
        # a payload is stored as the JSON it came as, whatever numbers it holds
        batch = _parse(schemas.EventsIn, body, as_text=("events",))
        with self.session() as (session, _rater):
            assignment = self._owned(session, batch.assignment_id)
            stored = store_events(session, assignment, batch.events)
            session.commit()
        return _json(schemas.EventsOut(stored=stored))

    # --- pages and files ------------------------------------------------------------------------

    def page(self, route: str, to_a_reader: bool = False) -> Response:
        """A page. A review that is closed is refused like every unblinded route; a browser
        that asked for the page itself (``to_a_reader``) gets the reason as a page with a
        way back, not as JSON."""
        if route == "/review":
            closed = self._closed_page("review", self.gate, to_a_reader)
            if closed is not None:
                return closed
        template = self.pages.get_template(PAGES[route])
        text = template.render(static=STATIC_URL, version=self.version, mode=self.mode)
        return HTMLResponse(text, headers={"Cache-Control": NO_CACHE})

    def _closed_page(
        self, title: str, gate: Callable[[], None], to_a_reader: bool
    ) -> Response | None:
        """None when ``gate`` lets the request through; the refusal as a page for a browser
        that asked for a page; the refusal raised (an ``ErrorsOut``) otherwise."""
        try:
            gate()
        except _Refused as refused:
            if not to_a_reader:
                raise
            said = " ".join(message for _, message in refused.problems)
            text = _CLOSED_PAGE.format(static=STATIC_URL, title=title, message=html.escape(said))
            return HTMLResponse(
                text, status_code=refused.status, headers={"Cache-Control": NO_STORE}
            )
        return None

    def static(self, path: str) -> Response:
        missing = _Refused(404, "path", "there is no such file")
        try:
            root = STATIC_DIR.resolve(strict=True)
            target = (root / path).resolve(strict=True)
        except (OSError, ValueError, RuntimeError):
            raise missing from None
        if not target.is_relative_to(root) or not target.is_file():
            raise missing
        if any(part.startswith(".") for part in target.relative_to(root).parts):
            raise missing
        return FileResponse(target, headers={"Cache-Control": NO_CACHE})

    def media(self, subject_uid: str, path: str, version: str | None) -> Response:
        """A file of the subject's render: a tile, the poster or the video, under the name a
        render writes, in the media directory of a subject whose render is ready."""
        missing = _Refused(404, "path", "there is no such file")
        if self.config.app.media_base_url is not None:
            raise missing  # the media of this run are served elsewhere
        if _UID.fullmatch(subject_uid) is None or schemas.MEDIA_FILE.fullmatch(path) is None:
            raise missing
        row = self.public.render(subject_uid)
        if row is None or row.status != "done" or row.config_hash != self.render_hash:
            raise missing
        try:
            # the media directory of the run may itself be a link (to a larger disk); below
            # it, the name has to lead to a file of this subject's render, links followed
            root = (self.run_dir / MEDIA_DIR).resolve(strict=True) / subject_uid
            target = (root / path).resolve(strict=True)
        except (OSError, ValueError, RuntimeError):
            raise missing from None
        if not target.is_relative_to(root) or not target.is_file():
            raise missing
        if schemas.MEDIA_FILE.fullmatch(target.relative_to(root).as_posix()) is None:
            raise missing
        current = bool(row.render_key) and version == row.render_key
        return FileResponse(target, headers={"Cache-Control": MEDIA_CACHE if current else NO_CACHE})

    # --- the unblinded routes -------------------------------------------------------------------

    def review_columns(self, frame: pd.DataFrame) -> tuple[str, ...]:
        """The columns of the review table, in display order (``schemas.ReviewOut``)."""
        wanted = [
            "subject_uid",
            "subject_label",
            "dataset",
            "fs_version",
            "render_status",
            "render_reason",
            *SCHEMAS["scores"].names,
            "final_A",
            "final_B",
            "final_C",
            "final_method",
            "unratable_reason",
            "n_ratings",
            "n_marks",
        ]
        if self.public.mode() != "local":
            wanted.remove("subject_label")
        return tuple(dict.fromkeys(name for name in wanted if name in frame.columns))

    def get_review(self, params: Mapping[str, str]) -> Response:
        try:
            query = schemas.ReviewQuery.model_validate(dict(params))
        except ValidationError as exc:
            raise _invalid(exc) from None
        self.gate()
        frame = subjects_frame(self.run_dir)
        columns = self.review_columns(frame)
        rows = [
            {name: _plain(value) for name, value in zip(columns, values, strict=True)}
            for values in frame[list(columns)].itertuples(index=False, name=None)
        ]
        for part in filter(None, (piece.strip() for piece in query.filter.split(","))):
            column, colon, wanted = part.partition(":")
            if not colon or column not in columns:
                raise _Refused(
                    422, "filter", f"{part!r} is not column:value with a column of the table"
                )
            rows = [row for row in rows if _matches(row[column], wanted)]

        def missing_last(column: str) -> Callable[[dict[str, Any]], bool]:
            return lambda row: row.get(column) is None

        # the order of the queue: by primary_rank, the subjects without a score last
        rows.sort(key=lambda row: row["subject_uid"])
        if "primary_rank" in columns:
            rows.sort(key=lambda row: row["primary_rank"] or 0)
            rows.sort(key=missing_last("primary_rank"))
        if query.sort is not None:
            sort = query.sort
            if sort not in columns:
                raise _Refused(422, "sort", f"the table has no column {sort!r}")
            have = [row for row in rows if row[sort] is not None]
            lack = [row for row in rows if row[sort] is None]
            have.sort(key=lambda row: row[sort], reverse=query.order == "desc")
            rows = have + lack
        page = rows[query.offset : query.offset + query.limit]
        return _json(schemas.ReviewOut(total=len(rows), columns=columns, rows=tuple(page)))

    def get_subject(self, subject_uid: str) -> Response:
        self.gate()
        if _UID.fullmatch(subject_uid) is None or subject_uid not in set(
            self.public.subject_uids(present=None)
        ):
            raise _Refused(404, "subject_uid", "there is no such subject")
        ready = self.ready(subject_uid)
        scores = self.scores()
        own = None
        if scores is not None:
            mine = scores[scores["subject_uid"] == subject_uid]
            if len(mine):
                own = {name: _plain(value) for name, value in mine.iloc[0].items()}
        ratings = ratings_frame(self.run_dir)
        ratings = ratings[ratings["subject_uid"] == subject_uid]
        marks = marks_frame(self.run_dir)
        marks = marks[marks["subject_uid"] == subject_uid]
        body = schemas.SubjectOut(
            subject_uid=subject_uid,
            media_base=None if ready is None else self.media_base(subject_uid),
            render_key=None if ready is None else ready.render_key,
            render=None if ready is None else schemas.render_out(ready.info),
            scores=own,
            ratings=tuple(_review_rating(row) for row in _records(ratings)),
            marks=tuple(_review_mark(row) for row in _records(marks)),
            clusters=self.clusters(subject_uid),
        )
        return _json(body)

    def clusters(self, subject_uid: str) -> tuple[schemas.ClusterOut, ...]:
        """The outlier clusters of the subject whose centroid has a native position."""
        try:
            clusters = clusters_with_native(self.run_dir)
        except (ValueError, OSError):
            return ()
        clusters = clusters[(clusters["subject_uid"] == subject_uid) & clusters["tkr_x"].notna()]
        return tuple(
            schemas.ClusterOut(
                reference_id=row["reference_id"],
                hemi=row["hemi"],
                region=row["region"],
                rank=row["rank"],
                tkr=(row["tkr_x"], row["tkr_y"], row["tkr_z"]),
                area_mm2=row["area_mm2"],
                peak_z=row["peak_z"],
            )
            for row in _records(clusters)
        )

    def get_stats(self) -> Response:
        self.gate()
        with self.session() as (session, _rater):
            raters = session.exec(select(Rater).order_by(col(Rater.name))).all()
            rows = session.exec(
                select(Assignment.rater_id, Assignment.completed_at, Submission.ms_on_subject)
                .join(
                    Submission,
                    col(Submission.assignment_id) == col(Assignment.assignment_id),
                )
                .where(col(Assignment.completed_at).is_not(None), col(Submission.current))
            ).all()
            out = []
            for rater in raters:
                own = [(at, ms) for rater_id, at, ms in rows if rater_id == rater.rater_id]
                days: dict[str, int] = {}
                for at, _ms in own:
                    day = (at or "")[:10]
                    days[day] = days.get(day, 0) + 1
                timed = [ms for _at, ms in own if ms is not None]
                out.append(
                    schemas.RaterStats(
                        rater=rater.name,
                        n_completed=len(own),
                        median_s=statistics.median(timed) / 1000 if timed else None,
                        by_day=tuple(
                            schemas.DayCount(day=day, n=n) for day, n in sorted(days.items())
                        ),
                    )
                )
        return _json(schemas.StatsOut(raters=tuple(out)))

    def get_export(self) -> Response:
        self.gate()
        return Response(
            frame_csv(ratings_frame(self.run_dir)),
            media_type="text/csv",
            headers={
                "Cache-Control": NO_STORE,
                "Content-Disposition": 'attachment; filename="ratings.csv"',
            },
        )

    # --- the EPI pages (step 2 of docs/epi_interface.md) -----------------------------------------

    def epi_offered(self) -> bool:
        """Whether the run has EPI pages at all: a local run whose EPI interface is on and
        whose catalog has a functional source with a present participant."""
        return self.epi_on and self.public.has_func()

    def epi_available(self) -> None:
        """A route of the EPI pages on a run without them answers as a route that does not
        exist, as it did before step 2."""
        if not self.epi_offered():
            raise _Refused(404, "request", _NO_ROUTE)

    def epi_participants(self) -> pd.DataFrame:
        """The present participants of the catalog, read again after a second."""
        now = time.monotonic()
        cached = self._epi_participants
        if cached is not None and now - cached[0] < _EPI_PARTICIPANTS_S:
            return cached[1]
        frame = self.public.func_participants()
        frame = frame[[_true(v) for v in frame["present"]]] if len(frame) else frame
        self._epi_participants = (now, frame)
        return frame

    def epi_participant(self, participant_uid: str) -> dict[str, Any]:
        """The catalog row of a present participant; 404 for anything else."""
        missing = _Refused(404, "participant_uid", "there is no such participant")
        if _UID.fullmatch(participant_uid) is None:
            raise missing
        frame = self.epi_participants()
        rows = frame[frame["participant_uid"] == participant_uid]
        if not len(rows):
            raise missing
        return {name: _plain(value) for name, value in rows.iloc[0].items()}

    def packs_of(self, participant_uid: str) -> Any:
        """``app.epi.Packs`` on ``epi/<participant_uid>/`` of the run, with the media URLs of
        this app. Kept per participant, for the files it has parsed."""
        from surfcensus.app.epi import Packs  # app.epi imports this module

        with self._lock:
            packs = self._packs.get(participant_uid)
            if packs is None:
                packs = Packs(
                    epi_participant_dir(self.run_dir, participant_uid),
                    memory_mib=EPI_MEMORY_MIB,
                    media_url=f"{EPI_MEDIA_URL}/{participant_uid}",
                )
                self._packs[participant_uid] = packs
        return packs

    def epi_index(self) -> EpiIndexOut:
        """``EpiIndexOut`` from the catalog (what is listed, packed, waiting, failed,
        flagged) and the run directory (what the stream of each participant shows now)."""
        participants = self.epi_participants()
        sessions = self.public.func_sessions()
        runs = self.public.func_runs()
        packs = self.public.packs()
        pack_hash = epi_settings(self.config).pack_hash()
        state: dict[str, str] = {}
        for row in _records(packs):
            if row.get("config_hash") == pack_hash and row.get("status") in ("done", "failed"):
                state[str(row["run_uid"])] = str(row["status"])
        runs_of: dict[str, list[dict[str, Any]]] = {}
        for row in _records(runs):
            if _true(row.get("present")):
                runs_of.setdefault(str(row["session_uid"]), []).append(row)
        sessions_of: dict[str, list[dict[str, Any]]] = {}
        for row in _records(sessions):
            if _true(row.get("present")):
                sessions_of.setdefault(str(row["participant_uid"]), []).append(row)
        out = []
        local = self.public.mode() == "local"
        for row in _records(participants):
            uid = str(row["participant_uid"])
            packs_dir = self.packs_of(uid)
            shown = set(packs_dir.shown_sessions())
            own = sorted(
                sessions_of.get(uid, []),
                key=lambda s: (s.get("stream_order") or 0, str(s["session_uid"])),
            )
            listed = []
            for session_row in own:
                session_uid = str(session_row["session_uid"])
                counts = _epi_counts(runs_of.get(session_uid, []), state)
                listed.append(
                    EpiIndexSessionOut(
                        session_uid=session_uid,
                        label=str(session_row.get("session_label") or ""),
                        shown=session_uid in shown,
                        **counts,
                    )
                )
            totals = {
                name: sum(getattr(s, name) for s in listed)
                for name in ("n_runs", "n_packable", "n_packed", "n_pending", "n_failed")
            }
            label = row.get("participant") if local else None
            out.append(
                EpiParticipantOut(
                    participant_uid=uid,
                    label=None if label is None else str(label),
                    subject_uids=tuple(
                        sorted({str(s["subject_uid"]) for s in own if s.get("subject_uid")})
                    ),
                    n_sessions=len(listed),
                    n_flagged=sum(s.n_flagged for s in listed),
                    can_open=packs_dir.has_packed_run(),
                    url=f"{EPI_INDEX_URL}/{uid}",
                    sessions=tuple(listed),
                    **totals,
                )
            )
        out.sort(key=lambda p: (p.label or "", p.participant_uid))
        return EpiIndexOut(participants=tuple(out), n_pending=sum(p.n_pending for p in out))

    def get_epi_index(self) -> Response:
        self.epi_available()
        self.epi_gate()
        return _json(self.epi_index())

    def epi_waiting(self, participant_uid: str) -> bool:
        """Whether packs of the participant are still to be made: a packable run without a
        pack of the present settings, or a session that ``stream.json`` lists and the stream
        does not hold yet."""
        if self.packs_of(participant_uid).waiting() > 0:
            return True
        index = self.epi_index()
        return any(
            p.n_pending > 0 for p in index.participants if p.participant_uid == participant_uid
        )

    def epi_page(self, participant_uid: str | None, to_a_reader: bool) -> Response:
        """The index (``participant_uid`` None) or the EPI page of one participant's stream.
        Closed as ``/review`` is; a browser that asks for a closed page gets the reason as a
        page with a way back."""
        self.epi_available()
        closed = self._closed_page("EPI movies", self.epi_gate, to_a_reader)
        if closed is not None:
            return closed
        if participant_uid is None:
            text = self.pages.get_template("epi_index.html").render(
                static=STATIC_URL, version=self.version, mode=self.mode, api=EPI_API, rate_url="/"
            )
            return HTMLResponse(text, headers={"Cache-Control": NO_CACHE})
        self.epi_participant(participant_uid)
        waiting = self.epi_waiting(participant_uid)
        text = self.pages.get_template("epi.html").render(
            static=STATIC_URL,
            version=self.version,
            mode=self.mode,
            api=f"{EPI_API}/{participant_uid}",
            index_url=EPI_INDEX_URL,
            rate_url="/",
            # the page asks every EPI_POLL_S while packs of the participant wait and every
            # EPI_POLL_IDLE_S when none do; poll_s is the interval of now, and after each answer
            # the page asks the index which of the two applies
            waiting=waiting,
            poll_s=EPI_POLL_S if waiting else EPI_POLL_IDLE_S,
            poll_waiting_s=EPI_POLL_S,
            poll_idle_s=EPI_POLL_IDLE_S,
            index_api=EPI_API,
            participant_uid=participant_uid,
        )
        return HTMLResponse(text, headers={"Cache-Control": NO_CACHE})

    def get_epi_stream(self, participant_uid: str) -> Response:
        """``EpiStreamOut`` of one participant: the sessions of its ``stream.json`` whose
        ``session.json`` is there, or none while nothing is packed yet."""
        from surfcensus.app.epi import _Missing  # app.epi imports this module

        self.epi_available()
        self.epi_gate()
        row = self.epi_participant(participant_uid)
        packs = self.packs_of(participant_uid)
        try:
            body = packs.shape()
        except _Missing:
            label = row.get("participant") or participant_uid
            body = packs.empty_shape(str(label), str(row.get("session_order") or "session label"))
        return _json(body)

    def epi_media(
        self, participant_uid: str, session_uid: str, path: str, version: str | None
    ) -> Response:
        """A file of a pack of the participant, by the rules of ``app.epi.Packs.media``: a
        name of the patterns that its session lists, inside the session's directory."""
        from surfcensus.app.epi import _Missing  # app.epi imports this module

        self.epi_available()
        self.epi_gate()
        self.epi_participant(participant_uid)
        try:
            target, current = self.packs_of(participant_uid).media(session_uid, path, version)
        except _Missing:
            raise _Refused(404, "path", "there is no such file") from None
        return FileResponse(
            target,
            media_type="application/octet-stream",
            headers={"Cache-Control": MEDIA_CACHE if current else NO_CACHE},
        )

    # --- every route goes through here ----------------------------------------------------------

    def answer(
        self, work: Callable[..., Response], *args: Any, unblinded: bool = False
    ) -> Response:
        """The answer of a route, whatever happens in it: a refusal as its ``ErrorsOut``,
        and anything unexpected as a 500 that says nothing of what went wrong (the log
        does), so that no text of an exception reaches a rating page."""
        try:
            return work(*args)
        except _Refused as refused:
            return _errors(refused.status, refused.problems)
        except (DBAPIError, sqlite3.Error) as exc:
            get_logger().error("app.sqlite: %s", scrub(str(getattr(exc, "orig", exc))))
            return _errors(503, [("request", _BUSY)])
        except ConfigError as exc:
            get_logger().error("%s: %s", exc.reason, scrub(exc.detail))
            message = f"{exc.reason}: {scrub(exc.detail)}" if unblinded else _FAILED
            return _errors(500, [("request", message)])
        except Exception as exc:
            # the line without a path; the traceback, which has them, only when asked for
            get_logger().error("a request failed: %s: %s", type(exc).__name__, scrub(str(exc)))
            get_logger().debug("the traceback of that request", exc_info=True)
            return _errors(500, [("request", _FAILED)])


def _current_submission(session: Session, assignment: Assignment) -> Submission | None:
    return session.exec(
        select(Submission).where(
            Submission.assignment_id == assignment.assignment_id, col(Submission.current)
        )
    ).first()


def _vertices(mark: Mark) -> dict[str, Any]:
    """The fields of ``schemas.MarkVertices`` of a stored mark."""
    return {
        "hemi": mark.hemi,
        "pial_vertex": mark.pial_vertex,
        "pial_dist_mm": mark.pial_dist_mm,
        "white_vertex": mark.white_vertex,
        "white_dist_mm": mark.white_dist_mm,
        "fsavg6_vertex": mark.fsavg6_vertex,
        "vertex_surface": mark.vertex_surface,
        "box_dist_mm": mark.box_dist_mm,
    }


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    names = [str(name) for name in frame.columns]
    return [
        {name: _plain(value) for name, value in zip(names, values, strict=True)}
        for values in frame.itertuples(index=False, name=None)
    ]


def _true(value: object) -> bool:
    """A flag of a catalog frame: missing reads as false."""
    if value is None or value is pd.NA:
        return False
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


def _epi_counts(runs: Sequence[Mapping[str, Any]], state: Mapping[str, str]) -> dict[str, int]:
    """The counts of :class:`EpiIndexSessionOut` of the present runs of a session; ``state``
    is the status of the pack of each run that has one under the present settings."""
    packable = [r for r in runs if _true(r.get("packable")) and _true(r.get("selected"))]
    packed = sum(state.get(str(r["run_uid"])) == "done" for r in packable)
    failed = sum(state.get(str(r["run_uid"])) == "failed" for r in packable)
    return {
        "n_runs": len(runs),
        "n_packable": len(packable),
        "n_packed": packed,
        "n_pending": len(packable) - packed - failed,
        "n_failed": failed,
        "n_flagged": sum(_true(r.get("flagged")) for r in runs),
    }


def _review_rating(row: Mapping[str, Any]) -> schemas.ReviewRating:
    return schemas.ReviewRating(
        assignment_id=row["assignment_id"],
        rater=row["rater"],
        kind=row["kind"],
        render_key=row["render_key"],
        render_current=bool(row["render_current"]),
        A=row["A"],
        B=row["B"],
        C=row["C"],
        unratable_reason=row["unratable_reason"],
        other=bool(row["other"]),
        note=row["note"] or "",
        revision=row["revision"],
        server_ts=row["server_ts"],
        ms_on_subject=row["ms_on_subject"],
    )


def _review_mark(row: Mapping[str, Any]) -> schemas.ReviewMark:
    return schemas.ReviewMark.model_validate(
        {
            "mark_id": row["mark_id"],
            "assignment_id": row["assignment_id"],
            "rater": row["rater"],
            "class": row["class"],
            "panel": row["panel"],
            "slice": row["slice"],
            "px": row["px"],
            "py": row["py"],
            "vox": (row["vox_i"], row["vox_j"], row["vox_k"]),
            "tkr": (row["tkr_x"], row["tkr_y"], row["tkr_z"]),
            "hemi": row["hemi"],
            "pial_vertex": row["pial_vertex"],
            "pial_dist_mm": row["pial_dist_mm"],
            "white_vertex": row["white_vertex"],
            "white_dist_mm": row["white_dist_mm"],
            "fsavg6_vertex": row["fsavg6_vertex"],
            "vertex_surface": row["vertex_surface"],
            "box_dist_mm": row["box_dist_mm"],
        }
    )


# --- the app ------------------------------------------------------------------------------------


def create_app(
    run_dir: Path | str,
    *,
    mode: AppMode = "local",
    rater: str | None = None,
    allowed_hosts: Collection[str] | None = None,
) -> FastAPI:
    """The app on a run directory.

    Reads ``protocol.yaml`` (installing the shipped one into a run without it) and
    ``calibration.yaml``, opens ``app.sqlite``, makes the local rater (``rater``, default
    ``app.rater_name``, role ``lead``) and gives failed renders their unratable label. It
    serves the routes of ``app.schemas.ROUTES`` and no other; an answer with a status of 400
    or above is an ``ErrorsOut``. ``allowed_hosts`` are the hosts whose name a request may
    carry in its ``Host`` header (default :data:`LOCAL_HOSTS`; the port is ignored).

    The catalog and the database stay open until the app is shut down (its lifespan) or
    ``app.state.close()`` is called.

    Raises ``ConfigError``: for a snap radius (``app.mark_max_dist_mm``) above the
    ``vertex_map_margin_mm`` of the run's render config, a protocol or calibration set that
    cannot be used, a database of another run or mode, and ``mode="team"`` (milestone 5).
    """
    state = _App(run_dir, mode=mode, rater=rater, allowed_hosts=allowed_hosts)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            state.close()

    app = FastAPI(
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
        # a path is a route or it is not: no answer sends a browser to another one
        redirect_slashes=False,
    )
    app.state.close = state.close
    app.add_middleware(
        _Guard,
        allowed_hosts=state.allowed_hosts,
        headers=_security_headers(state.config.app.media_base_url),
    )

    @app.exception_handler(StarletteHTTPException)
    async def _no_route(_request: Request, exc: StarletteHTTPException) -> Response:
        said = {404: "there is no such route", 405: "the route does not take that method"}
        return _errors(exc.status_code, [("request", said.get(exc.status_code, "refused"))])

    @app.exception_handler(RequestValidationError)
    async def _not_valid(_request: Request, exc: RequestValidationError) -> Response:
        problems = [
            (".".join(str(part) for part in error["loc"]) or "request", str(error["msg"]))
            for error in exc.errors()
        ]
        return _errors(422, problems or [("request", "the request is not valid")])

    async def posted(request: Request, work: Callable[[bytes], Response]) -> Response:
        try:
            body = await _body(request)
        except _Refused as refused:
            return _errors(refused.status, refused.problems)
        return await run_in_threadpool(state.answer, work, body)

    @app.get("/")
    def rate_page() -> Response:
        return state.answer(state.page, "/")

    @app.get("/review")
    def review_page(request: Request) -> Response:
        reader = "text/html" in request.headers.get("accept", "")
        return state.answer(state.page, "/review", reader, unblinded=True)

    @app.get(STATIC_URL + "/{path:path}")
    def static(path: str) -> Response:
        return state.answer(state.static, path)

    @app.get("/media/{subject_uid}/{path:path}")
    def media(subject_uid: str, path: str, request: Request) -> Response:
        return state.answer(state.media, subject_uid, path, request.query_params.get("v"))

    @app.get("/healthz")
    def health() -> Response:
        return _json(schemas.HealthOut(ok=True, version=state.version))

    @app.get("/api/session")
    def session() -> Response:
        return state.answer(state.get_session)

    @app.get("/api/next")
    def next_() -> Response:
        return state.answer(state.get_next)

    @app.get("/api/back")
    def back() -> Response:
        return state.answer(state.get_back)

    @app.post("/api/rating")
    async def rating(request: Request) -> Response:
        return await posted(request, state.post_rating)

    @app.post("/api/skip")
    async def skip(request: Request) -> Response:
        return await posted(request, state.post_skip)

    @app.post("/api/events")
    async def events(request: Request) -> Response:
        return await posted(request, state.post_events)

    @app.get("/api/review")
    def review(request: Request) -> Response:
        return state.answer(state.get_review, dict(request.query_params), unblinded=True)

    @app.get("/api/review/{subject_uid}")
    def subject(subject_uid: str) -> Response:
        return state.answer(state.get_subject, subject_uid, unblinded=True)

    @app.get("/api/stats")
    def stats() -> Response:
        return state.answer(state.get_stats, unblinded=True)

    @app.get("/api/export")
    def export() -> Response:
        return state.answer(state.get_export, unblinded=True)

    # the EPI pages of a run directory: unblinded, and routes that do not exist (404) where
    # the run has none
    @app.get(EPI_INDEX_URL)
    def epi_index_page(request: Request) -> Response:
        reader = "text/html" in request.headers.get("accept", "")
        return state.answer(state.epi_page, None, reader, unblinded=True)

    @app.get(EPI_API)
    def epi_index() -> Response:
        return state.answer(state.get_epi_index, unblinded=True)

    @app.get(EPI_INDEX_URL + "/{participant_uid}")
    def epi_page(participant_uid: str, request: Request) -> Response:
        reader = "text/html" in request.headers.get("accept", "")
        return state.answer(state.epi_page, participant_uid, reader, unblinded=True)

    @app.get(EPI_API + "/{participant_uid}")
    def epi_stream(participant_uid: str) -> Response:
        return state.answer(state.get_epi_stream, participant_uid, unblinded=True)

    @app.get(EPI_MEDIA_URL + "/{participant_uid}/{session_uid}/{path:path}")
    def epi_media(participant_uid: str, session_uid: str, path: str, request: Request) -> Response:
        return state.answer(
            state.epi_media,
            participant_uid,
            session_uid,
            path,
            request.query_params.get("v"),
            unblinded=True,
        )

    return app


# --- starting and stopping ----------------------------------------------------------------------


@dataclass
class ServerHandle:
    """A server running in a thread of this process."""

    url: str
    """``http://127.0.0.1:<port>/``."""
    port: int
    """The port it got, which is not always the one asked for."""

    def __post_init__(self) -> None:
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._close: Callable[[], None] | None = None

    @property
    def running(self) -> bool:
        """Whether the thread of the server is alive."""
        return self._thread is not None and self._thread.is_alive()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the server and wait for its thread, at most ``timeout`` seconds. A second
        call does nothing."""
        server, thread, close = self._server, self._thread, self._close
        self._server = self._thread = self._close = None
        if server is None or thread is None:
            return
        server.should_exit = True
        thread.join(timeout)
        if thread.is_alive():
            # connections that do not end: stop without waiting for them
            server.force_exit = True
            thread.join(1.0)
        if close is not None:
            close()


def _listen(host: str, port: int) -> socket.socket:
    """A listening socket on ``host`` and ``port``, or on the first free one of the
    :data:`PORT_TRIES` ports after it; ``port=0`` takes what the system gives."""
    if not 0 <= port <= 65535:
        raise ConfigError("invalid", f"a port lies between 0 and 65535, got {port}")
    last = min(65535, port + PORT_TRIES)
    for candidate in [0] if port == 0 else range(port, last + 1):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if os.name != "nt":
                # a port that a server left a moment ago; a port in use stays refused
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, candidate))
            sock.listen(128)
        except OSError as exc:
            sock.close()
            if exc.errno == errno.EADDRINUSE:
                continue
            raise ConfigError(
                "invalid", f"the server cannot listen on {host}:{candidate}: {exc.strerror or exc}"
            ) from exc
        return sock
    raise ConfigError(
        "invalid", f"the ports {port} to {last} of {host} are in use; name another with --port"
    )


def start_server(
    run_dir: Path | str,
    *,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    mode: AppMode = "local",
    rater: str | None = None,
) -> ServerHandle:
    """Start the app in a thread and return once the port accepts connections.

    A taken port: the next :data:`PORT_TRIES` are tried; ``port=0`` asks the system for a
    free one. A local server binds to one of :data:`LOCAL_HOSTS` only
    (``ConfigError("invalid", ...)`` otherwise).
    """
    if mode == "local" and host not in LOCAL_HOSTS:
        raise ConfigError(
            "invalid",
            f"a local server listens on {' or '.join(LOCAL_HOSTS)} only, not on {host}: it "
            "has no login",
        )
    app = create_app(run_dir, mode=mode, rater=rater)
    try:
        sock = _listen(host, port)
    except BaseException:
        app.state.close()
        raise
    bound = int(sock.getsockname()[1])
    config = uvicorn.Config(
        app,
        log_level="warning",
        access_log=False,
        lifespan="on",
        # what uvicorn itself depends on, so that a server is the same one whether or not
        # httptools and uvloop happen to be installed beside it
        http="h11",
        loop="asyncio",
        # no proxy stands before a local server: what a request says of one is not believed
        proxy_headers=False,
        server_header=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [sock]}, name="surfcensus-app", daemon=True
    )
    handle = ServerHandle(url=f"http://{host}:{bound}/", port=bound)
    handle._server, handle._thread = server, thread

    def close() -> None:
        with suppress(OSError):
            sock.close()
        app.state.close()

    handle._close = close
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            handle.stop()
            raise ConfigError("invalid", f"the server did not start on {host}:{bound}")
        time.sleep(0.01)
    return handle


def serve(
    run_dir: Path | str,
    *,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    mode: AppMode = "local",
    rater: str | None = None,
    open_browser: bool = True,
    stop: StopRequest | None = None,
    stopped: Callable[[], None] | None = None,
) -> None:
    """:func:`start_server`, print the URL, open the browser, wait for Ctrl-C or SIGTERM in
    the main thread, stop. With ``stop`` (a ``stopping.StopFlag``, or a ``threading.Event``
    of a caller) the call sets no SIGTERM handler and serves until it is set (or Ctrl-C):
    the caller sets it from a handler of its own; the command line ignores SIGTERM from
    the first one on, so that a SIGTERM that comes after the server has stopped cannot
    kill a command that stopped in order. Without ``stop``, in the main thread, SIGTERM
    sets a flag for the length of the call; the handler takes no lock. ``stopped`` is called
    once the server has stopped (the command line ignores both signals from there). Refuses to start
    (``ConfigError("invalid", ...)``) when renders of the selection are out of date:
    ``N renders are out of date; run surfcensus render --run-dir DIR``. Says on stderr when
    selected subjects are not rendered yet, and when the queue is empty because the run
    has no subject to rate."""
    run_dir = Path(run_dir)
    config = load_config(run_dir)
    with PublicCatalog.open(run_dir) as public:
        census = public.mode() == "census"
        # the EPI pages, offered as the app offers them (``epi_offered``), and whether a pack
        # can be viewed there
        epi = not census and epi_settings(config).enabled and public.has_func()
        packs = public.packs() if epi else None
        viewable = packs is not None and bool((packs["status"] == "done").any())
        plan = build_plan(
            public,
            queue_scores(run_dir),
            config.app,
            "local",
            render_config_hash=config.render.config_hash(),
        )
    if plan.outdated > 0:
        renders = "render is" if plan.outdated == 1 else "renders are"
        raise ConfigError(
            "invalid",
            f"{plan.outdated} {renders} out of date; run surfcensus render --run-dir {run_dir}",
        )
    if not plan.order:
        # the app would say that everything is rated
        why = (
            "no subject of the run is selected"
            if census
            else f"the run has no subject; run surfcensus inventory --run-dir {run_dir}"
        )
        print(f"There is nothing to rate: {why}", file=sys.stderr, flush=True)
    if plan.pending > 0:
        # nothing renders behind this server: the queue would wait for them for good
        subjects = "subject is" if plan.pending == 1 else "subjects are"
        print(
            f"{plan.pending} selected {subjects} not rendered yet and will not be offered; "
            f"run surfcensus render --run-dir {run_dir}",
            file=sys.stderr,
            flush=True,
        )
    handle = start_server(run_dir, host=host, port=port, mode=mode, rater=rater)
    waits = StopFlag() if stop is None else stop
    # the caller's event: the caller's handler. Otherwise SIGTERM for the length of the call.
    in_main = stop is None and threading.current_thread() is threading.main_thread()
    earlier = signal.signal(signal.SIGTERM, lambda *_: waits.set()) if in_main else None
    try:
        print(f"Rating app: {handle.url} (Ctrl-C to stop)", file=sys.stderr, flush=True)
        epi_url = f"{handle.url.rstrip('/')}{EPI_INDEX_URL}" if epi else None
        if epi_url is not None:
            print(f"EPI movies: {epi_url}", file=sys.stderr, flush=True)
        # nothing to rate and EPI movies to view: the browser goes to the movies
        opened = epi_url if epi_url is not None and viewable and not plan.render_keys else None
        if opened is not None:
            print("No subject can be rated; the browser opens the EPI movies.", file=sys.stderr)
        if open_browser:
            with suppress(webbrowser.Error):
                webbrowser.open(opened or handle.url)
        while handle.running and not waits.wait(0.2):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        handle.stop()
        if stopped is not None:
            stopped()
        if in_main:
            signal.signal(signal.SIGTERM, earlier)
