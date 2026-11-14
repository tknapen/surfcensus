"""The EPI page served from a directory of packs, without a run directory (work package E1.D;
docs/epi_interface.md, The app, and step 1 of its build order).

``surfcensus epi-session ... --serve`` packs the sessions of one subject into ``--out`` and
calls :func:`serve_packs` on it. The server reads ``<out>/stream.json`` for the sessions and
their order (decision D38: one continuous stream of the subject), ``<out>/<session>/session.json``
and, of each packed run, ``pack.json`` and ``traces.json``; it serves the page, one answer of
:class:`app.epi_schemas.EpiStreamOut` built from them, and the files of the packs. It writes
nothing.

What it holds against is what the local rater server holds against (``app.server``, whose
guard and security headers it uses as they are): it binds to 127.0.0.1 or localhost only; a
request whose ``Host`` names another host is refused, as is a request for ``/api`` from another
origin; every answer carries ``Content-Security-Policy: default-src 'self'`` and forbids
framing; no answer is a redirect. ``/epi-media/<session>/<path>`` serves a file of a pack only
when its name matches the patterns of ``func`` (``func.is_pack_path``) and the session lists it
(a contour file of a grid, a still or chunk of a packed run whose pack is whole); a path that
climbs, a link that leads out of the session's directory and any other name are 404, and the
JSON files of a pack are never served.

Caching. A file asked for with ``?v=<version>`` of the pack it belongs to now is cached as
immutable, any other request is ``no-cache``. The version is 16 hex characters of the SHA-256
over the SHA-256 digests of the run's served files (or of the grid's contour files), which the
page receives in the same answer: it is a digest of what is served and of nothing else. Not
``PackInfo.content_digest``: that digest covers the defining JSON of ``pack.json``, which holds
the sizes and SHA-256 of the source files, and an unkeyed hash of those may not leave the run
(decision D20).
"""

from __future__ import annotations

import hashlib
import re
import signal
import sys
import threading
import time
import webbrowser
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, TypeVar

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

import surfcensus
from surfcensus.app import epi_schemas as shapes
from surfcensus.app.schemas import HealthOut
from surfcensus.app.server import (
    DEFAULT_PORT,
    LOCAL_HOSTS,
    MEDIA_CACHE,
    NO_CACHE,
    NO_STORE,
    STATIC_DIR,
    STATIC_URL,
    TEMPLATES_DIR,
    ServerHandle,
    _errors,
    _Guard,
    _listen,
    _security_headers,
)
from surfcensus.config import ConfigError
from surfcensus.func import (
    CONTOUR_UNIT,
    ID_PATTERN,
    PACK_JSON,
    PACK_VERSION,
    PLANES,
    SESSION_JSON,
    STILLS,
    STREAM_JSON,
    STREAMS,
    TRACES_JSON,
    is_pack_path,
)
from surfcensus.func.model import (
    READING,
    RULES,
    GridInfo,
    PackInfo,
    SessionInfo,
    StreamInfo,
    TracesInfo,
    read_info,
)
from surfcensus.stopping import StopFlag, StopRequest

EPI_PAGE = "epi.html"
API_STREAM = "/api/epi/stream"
MEDIA_URL = "/epi-media"
DEFAULT_MEMORY_MIB = 512

_ID = re.compile(ID_PATTERN)
_Info = TypeVar("_Info", SessionInfo, PackInfo, TracesInfo, StreamInfo)


class _Missing(Exception):
    """A request that is answered with 404."""


def _version(digests: list[str]) -> str:
    """The cache version of a set of served files: their SHA-256 digests, joined, hashed."""
    return hashlib.sha256("\n".join(digests).encode()).hexdigest()[:16]


def _run_files(info: PackInfo) -> list[tuple[str, str]]:
    """``(file, sha256)`` of every file of a run that the page is given, in a fixed order."""
    out = []
    for stream in STREAMS:
        files = info.streams[stream]
        out += [(getattr(files, still).file, getattr(files, still).sha256) for still in STILLS]
        out += [(chunk.file, chunk.sha256) for chunk in files.chunks]
    return out


def _grid_files(grid: GridInfo) -> list[tuple[str, str]]:
    if grid.contours is None:
        return []
    return [(grid.contours[plane].file, grid.contours[plane].sha256) for plane in PLANES]


class Packs:
    """A directory of packs, read as it is on each request. Parsed files are kept, keyed by
    their path, size and modification time, so a file that is written again is read again."""

    def __init__(self, out_dir: Path | str, *, memory_mib: int = DEFAULT_MEMORY_MIB) -> None:
        self.out_dir = Path(out_dir)
        if memory_mib < 1:
            raise ConfigError("invalid", f"a memory limit of {memory_mib} MiB is too small")
        self.memory_mib = int(memory_mib)
        self._cache: dict[tuple[str, type], tuple[tuple[int, int], Any]] = {}
        self._lock = threading.Lock()

    def _read(self, path: Path, model: type[_Info]) -> _Info | None:
        try:
            st = path.stat()
        except OSError:
            return None
        stamp = (st.st_size, st.st_mtime_ns)
        key = (str(path), model)
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None and hit[0] == stamp:
            return hit[1]
        info = read_info(path, model)
        if info is not None and getattr(info, "version", PACK_VERSION) != PACK_VERSION:
            info = None
        with self._lock:
            if len(self._cache) > 512:
                self._cache.clear()
            self._cache[key] = (stamp, info)
        return info

    def stream_info(self) -> StreamInfo | None:
        return self._read(self.out_dir / STREAM_JSON, StreamInfo)

    def session(self, session_id: str) -> SessionInfo | None:
        """``session.json`` of a session of the stream, or None."""
        stream = self.stream_info()
        if stream is None or _ID.fullmatch(session_id) is None or session_id not in stream.sessions:
            return None
        info = self._read(self.out_dir / session_id / SESSION_JSON, SessionInfo)
        return info if info is not None and info.session == session_id else None

    def pack(self, session_id: str, run: str) -> PackInfo | None:
        """``pack.json`` of a run that ``session.json`` lists as packed, when every file it
        lists is there (``packread.pack_on_disk``'s rule, by directory listings)."""
        session = self.session(session_id)
        if session is None or _ID.fullmatch(run) is None:
            return None
        listed = next((r for r in session.runs if r.run == run), None)
        if listed is None or listed.reason is not None or listed.n_volumes is None:
            return None
        directory = self.out_dir / session_id / run
        info = self._read(directory / PACK_JSON, PackInfo)
        if info is None or info.run != run or info.session != session_id:
            return None
        if set(info.streams) != set(STREAMS):
            return None
        present: dict[str, set[str]] = {}
        for name in [*(f for f, _ in _run_files(info)), info.traces.file]:
            folder, _, base = name.rpartition("/")
            if folder not in present:
                try:
                    present[folder] = {
                        p.name for p in (self.out_dir / session_id / folder).iterdir()
                    }
                except OSError:
                    return None
            if base not in present[folder]:
                return None
        return info

    def traces(self, session_id: str, run: str) -> TracesInfo | None:
        return self._read(self.out_dir / session_id / run / TRACES_JSON, TracesInfo)

    # --- the shape ------------------------------------------------------------------------

    def shape(self) -> shapes.EpiStreamOut:
        """``EpiStreamOut`` of the directory. Raises ``_Missing`` without ``stream.json``."""
        stream = self.stream_info()
        if stream is None:
            raise _Missing(f"{STREAM_JSON} is not there or does not read as one of this version")
        sessions = []
        rules: dict[str, str] = dict(RULES)
        for session_id in stream.sessions:
            info = self.session(session_id)
            if info is None:
                continue
            rules = dict(info.rules)
            sessions.append(self._session_shape(info))
        return shapes.EpiStreamOut(
            label=stream.label,
            order=stream.order,
            reading=READING,
            memory_mib=self.memory_mib,
            rules=rules,
            contour_unit=CONTOUR_UNIT,
            sessions=tuple(sessions),
        )

    def _session_shape(self, info: SessionInfo) -> shapes.EpiSessionOut:
        grids = tuple(
            shapes.GridOut(
                grid_key=g.grid_key,
                shape=g.shape,
                voxel_mm=g.voxel_mm,
                oblique_deg=g.oblique_deg,
                edge_letters=g.edge_letters,
                contours=None
                if g.contours is None
                else {p: shapes.ContourOut(**c.model_dump()) for p, c in g.contours.items()},
                slice_normal=g.slice_normal,
                slice_mm=g.slice_mm,
                token=_version([sha for _, sha in _grid_files(g)]),
            )
            for g in info.grids
        )
        runs = []
        for r in info.runs:
            pack = None if r.reason is not None else self._pack_shape(info.session, r.run)
            runs.append(
                shapes.EpiRunOut(
                    run_uid=r.run,
                    name=r.name,
                    acquisition_order=r.acquisition_order,
                    anchor=r.anchor,
                    grid_key=r.grid_key,
                    frame=r.frame,
                    reason=r.reason,
                    n_volumes=r.n_volumes,
                    tr_s=r.tr_s,
                    first=r.first,
                    window=r.window,
                    sd_window=r.sd_window,
                    detail=r.detail,
                    pack=pack,
                )
            )
        registration = None
        if info.registration is not None:
            registration = shapes.RegistrationOut(**info.registration.model_dump())
        return shapes.EpiSessionOut(
            session_uid=info.session,
            label=info.label,
            media_url=f"{MEDIA_URL}/{info.session}/",
            fd_threshold_mm=info.fd_threshold_mm,
            registration=registration,
            grids=grids,
            runs=tuple(runs),
            contours_from=info.contours_from,
        )

    def _pack_shape(self, session_id: str, run: str) -> shapes.EpiPackOut | None:
        info = self.pack(session_id, run)
        traces = self.traces(session_id, run)
        if info is None or traces is None:
            return None

        def stream(name: str) -> shapes.StreamOut:
            files = info.streams[name]
            return shapes.StreamOut(
                ref=shapes.FileOut(**files.ref.model_dump()),
                mean=shapes.FileOut(**files.mean.model_dump()),
                sd=shapes.FileOut(**files.sd.model_dump()),
                chunks=tuple(shapes.ChunkOut(**c.model_dump()) for c in files.chunks),
            )

        return shapes.EpiPackOut(
            raw_affine=info.raw_affine,
            reference_image=info.reference_image,
            rigid=info.rigid,
            corrections=info.corrections,
            chunk_volumes=info.chunk_volumes,
            stills_volumes=info.stills_volumes,
            streams={name: stream(name) for name in STREAMS},
            traces=shapes.TracesOut(**traces.model_dump()),
            token=_version([sha for _, sha in _run_files(info)]),
        )

    # --- the files ------------------------------------------------------------------------

    def media(self, session_id: str, path: str, version: str | None) -> tuple[Path, bool]:
        """The file a request names, and whether it was asked for with the present version.
        Raises ``_Missing`` for anything that is not a file the session lists."""
        session = self.session(session_id)
        if session is None or not is_pack_path(path):
            raise _Missing("no such file")
        listed: dict[str, str] = {}
        current = ""
        first = path.split("/", 1)[0]
        if first == "contours":
            for grid in session.grids:
                files = _grid_files(grid)
                if any(f == path for f, _ in files):
                    listed = dict(files)
                    current = _version([sha for _, sha in files])
        else:
            info = self.pack(session_id, first)
            if info is not None:
                files = _run_files(info)
                listed = dict(files)
                current = _version([sha for _, sha in files])
        if path not in listed:
            raise _Missing("no such file")
        try:
            # the session directory may itself be a link (to a larger disk); below it, the name
            # has to lead to a file of this session, links followed
            root = (self.out_dir / session_id).resolve(strict=True)
            target = (root / path).resolve(strict=True)
        except (OSError, ValueError, RuntimeError):
            raise _Missing("no such file") from None
        if not target.is_relative_to(root) or not target.is_file():
            raise _Missing("no such file")
        if not is_pack_path(target.relative_to(root).as_posix()):
            raise _Missing("no such file")
        return target, version is not None and version == current


def _static(path: str) -> Path:
    try:
        root = STATIC_DIR.resolve(strict=True)
        target = (root / path).resolve(strict=True)
    except (OSError, ValueError, RuntimeError):
        raise _Missing("no such file") from None
    if not target.is_relative_to(root) or not target.is_file():
        raise _Missing("no such file")
    if any(part.startswith(".") for part in target.relative_to(root).parts):
        raise _Missing("no such file")
    return target


def _json(model: BaseModel) -> Response:
    return JSONResponse(model.model_dump(mode="json"), headers={"Cache-Control": NO_STORE})


def create_epi_app(
    out_dir: Path,
    *,
    memory_mib: int = DEFAULT_MEMORY_MIB,
    allowed_hosts: tuple[str, ...] | None = None,
) -> FastAPI:
    """The pack-only app on ``out_dir`` (a directory with ``stream.json``). It serves the
    routes of :data:`app.epi_schemas.EPI_ROUTES` and no other. ``memory_mib`` is the limit of
    decoded chunks the page holds for the whole stream. Raises ``ConfigError("missing", ...)``
    when ``out_dir`` has no ``stream.json`` of this version."""
    packs = Packs(out_dir, memory_mib=memory_mib)
    if packs.stream_info() is None:
        raise ConfigError(
            "missing", f"{Path(out_dir) / STREAM_JSON} is not there or is not one of this version"
        )
    version = surfcensus.__version__
    pages = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR), undefined=StrictUndefined, autoescape=True
    )
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, redirect_slashes=False)
    hosts = frozenset(LOCAL_HOSTS if allowed_hosts is None else allowed_hosts)
    app.add_middleware(_Guard, allowed_hosts=hosts, headers=_security_headers(None))
    app.state.packs = packs

    @app.exception_handler(StarletteHTTPException)
    async def _no_route(_request: Request, exc: StarletteHTTPException) -> Response:
        said = {404: "there is no such route", 405: "the route does not take that method"}
        return _errors(exc.status_code, [("request", said.get(exc.status_code, "refused"))])

    def page() -> Response:
        text = pages.get_template(EPI_PAGE).render(
            static=STATIC_URL, version=version, mode="local", api=API_STREAM
        )
        return HTMLResponse(text, headers={"Cache-Control": NO_CACHE})

    def guarded(work: Callable[[], Response]) -> Response:
        try:
            return work()
        except _Missing:
            return _errors(404, [("path", "there is no such file")])

    @app.get("/")
    def root_page() -> Response:
        return page()

    @app.get("/epi")
    def epi_page() -> Response:
        return page()

    @app.get(API_STREAM)
    def stream() -> Response:
        try:
            return _json(packs.shape())
        except _Missing as missing:
            return _errors(503, [("stream", str(missing))])

    @app.get(MEDIA_URL + "/{session_id}/{path:path}")
    def media(session_id: str, path: str, request: Request) -> Response:
        def answer() -> Response:
            target, current = packs.media(session_id, path, request.query_params.get("v"))
            return FileResponse(
                target,
                media_type="application/octet-stream",
                headers={"Cache-Control": MEDIA_CACHE if current else NO_CACHE},
            )

        return guarded(answer)

    @app.get(STATIC_URL + "/{path:path}")
    def static(path: str) -> Response:
        return guarded(lambda: FileResponse(_static(path), headers={"Cache-Control": NO_CACHE}))

    @app.get("/healthz")
    def health() -> Response:
        return _json(HealthOut(ok=True, version=version))

    return app


def start_pack_server(
    out_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    memory_mib: int = DEFAULT_MEMORY_MIB,
) -> ServerHandle:
    """Start the pack-only app in a thread and return once the port accepts connections. A
    taken port: the next ones are tried; ``port=0`` asks the system for a free one."""
    if host not in LOCAL_HOSTS:
        raise ConfigError(
            "invalid",
            f"the EPI page listens on {' or '.join(LOCAL_HOSTS)} only, not on {host}: it has no "
            "login",
        )
    app = create_epi_app(Path(out_dir), memory_mib=memory_mib)
    sock = _listen(host, DEFAULT_PORT if port is None else port)
    bound = int(sock.getsockname()[1])
    config = uvicorn.Config(
        app,
        log_level="warning",
        access_log=False,
        lifespan="off",
        http="h11",
        loop="asyncio",
        proxy_headers=False,
        server_header=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [sock]}, name="surfcensus-epi", daemon=True
    )
    handle = ServerHandle(url=f"http://{host}:{bound}/", port=bound)
    handle._server, handle._thread = server, thread

    def close() -> None:
        with suppress(OSError):
            sock.close()

    handle._close = close
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            handle.stop()
            raise ConfigError("invalid", f"the server did not start on {host}:{bound}")
        time.sleep(0.01)
    return handle


def serve_packs(
    out_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    open_browser: bool = True,
    stop: StopRequest | None = None,
    memory_mib: int = DEFAULT_MEMORY_MIB,
) -> None:
    """Serve the EPI page on the packs in ``out_dir`` until Ctrl-C, SIGTERM or ``stop``, as
    ``app.server.serve`` serves the rater app: with ``stop`` (a ``stopping.StopFlag``, or a
    ``threading.Event`` of a caller) the call sets no SIGTERM handler and serves until it is
    set; without it, in the main thread, SIGTERM sets a flag for the length of the call (the
    handler takes no lock) and the earlier handler is put back. Prints the address on stderr
    and opens the browser unless ``open_browser`` is false. Raises ``ConfigError`` for a
    directory without ``stream.json`` and for a host that is not local."""
    handle = start_pack_server(Path(out_dir), host=host, port=port, memory_mib=memory_mib)
    waits = StopFlag() if stop is None else stop
    in_main = stop is None and threading.current_thread() is threading.main_thread()
    earlier = signal.signal(signal.SIGTERM, lambda *_: waits.set()) if in_main else None
    try:
        print(f"EPI page: {handle.url} (Ctrl-C to stop)", file=sys.stderr, flush=True)
        if open_browser:
            with suppress(webbrowser.Error):
                webbrowser.open(handle.url)
        while handle.running and not waits.wait(0.2):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        handle.stop()
        if in_main:
            signal.signal(signal.SIGTERM, earlier)
