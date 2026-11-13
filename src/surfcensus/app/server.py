"""The rater app as a web server: the routes of docs/rater_app.md, and starting and
stopping it.

The app opens the catalog with ``PublicCatalog.open(run_dir)``, read-only, so it cannot
reach an accession, a subject path or an input signature. Local mode has one rater per
server process and no login.

Interface of milestone 4; everything here is written by package 4.G.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI

from surfcensus.app.schemas import AppMode

_OWNER = "package 4.G (server and events)"

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
everything else arrives through the API."""


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
    ``app.rater_name``, role ``lead``) and gives failed renders their unratable label.

    Raises ``ConfigError``: for a snap radius (``app.mark_max_dist_mm``) above the
    ``vertex_map_margin_mm`` of the run's render config, a protocol or calibration set that
    cannot be used, a database of another run or mode, and ``mode="team"`` (milestone 5).
    """
    raise NotImplementedError(f"app.server.create_app: {_OWNER}")


@dataclass
class ServerHandle:
    """A server running in a thread of this process."""

    url: str
    """``http://127.0.0.1:<port>/``."""
    port: int
    """The port it got, which is not always the one asked for."""

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the server and wait for its thread, at most ``timeout`` seconds. A second
        call does nothing."""
        raise NotImplementedError(f"app.server.ServerHandle.stop: {_OWNER}")


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
    raise NotImplementedError(f"app.server.start_server: {_OWNER}")


def serve(
    run_dir: Path | str,
    *,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    mode: AppMode = "local",
    rater: str | None = None,
    open_browser: bool = True,
) -> None:
    """:func:`start_server`, print the URL, open the browser, wait for Ctrl-C or SIGTERM in
    the main thread, stop. Refuses to start (``ConfigError("invalid", ...)``) when renders
    of the selection are out of date: ``N renders are out of date; run surfcensus render
    --run-dir DIR``."""
    raise NotImplementedError(f"app.server.serve: {_OWNER}")
