"""Tests of surfcensus/app/server.py and surfcensus/app/events.py: the routes of the rater
app on a run, what guards them, and starting and stopping a server.

Most tests talk to the app through Starlette's test client on a copy of ``small_run`` (three
synthetic subjects of one anatomy, panels of 64 px, a vertex map with fsaverage6
counterparts). What the client cannot say, a path that it would normalize, goes to the app
as a raw ASGI request. One test renders the cached real subject, when it is there, and
checks a mark against the surface file; the last one drives the pages in a browser against
a server in a thread (marked ``browser``).
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import http.client
import io
import itertools
import json
import os
import signal
import socket
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import nibabel.freesurfer as fs
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel
from sqlalchemy.exc import OperationalError
from sqlmodel import Session, select

import surfcensus
from surfcensus.app import LABEL_REASONS_KEY, schemas, server
from surfcensus.app.db import get_or_create_rater, open_app_db
from surfcensus.app.events import CONFIG_CHANGE, record_run_event, record_setting, store_events
from surfcensus.app.models import Assignment, Event
from surfcensus.app.server import (
    DEFAULT_PORT,
    LOCAL_HOSTS,
    MEDIA_CACHE,
    PORT_TRIES,
    RETRY_AFTER_S,
    ServerHandle,
    create_app,
    serve,
    start_server,
)
from surfcensus.calibration import (
    CalibrationExample,
    CalibrationSet,
    CalibrationSubject,
    write_calibration,
)
from surfcensus.config import ConfigError, load_config, with_changes, write_config
from surfcensus.export.tables import frame_csv, ratings_frame
from surfcensus.protocol import PROTOCOL_FILE, Protocol, install_protocol
from surfcensus.render import geometry
from surfcensus.render.model import PANEL_ORDER, RENDER_JSON, RenderInfo, read_render
from surfcensus.render.vertexmap import read_vertex_map
from surfcensus.run import Run
from surfcensus.schemas import KEYS, SCHEMAS, SCORE_FLAGS
from surfcensus.sources.local import local
from surfcensus.store import (
    CATALOG_FILE,
    PublicCatalog,
    clusters_with_native,
    media_dir,
    primary_scores,
    read_table,
    table_path,
    write_table,
)

BASE = "http://127.0.0.1"
NONE = {"A": 0, "B": 0, "C": 0}
SECURITY_HEADERS = (
    "content-security-policy",
    "x-content-type-options",
    "x-frame-options",
    "referrer-policy",
    "cross-origin-resource-policy",
    "cross-origin-opener-policy",
)
UNBLINDED = ("/review", "/api/review", "/api/stats", "/api/export")

_clock = itertools.count()


def client_ts() -> str:
    """A client time that no other request of the test session has."""
    n = next(_clock)
    return f"2026-11-13T10:{n // 60 % 60:02d}:{n % 60:02d}.{n // 3600:03d}Z"


# --- a server on a run, and a rater who talks to it ---------------------------------------------


@contextmanager
def serving(run_dir: Path, **options: Any) -> Iterator[TestClient]:
    app = create_app(run_dir, **options)
    with TestClient(app, base_url=BASE) as client:
        yield client


@pytest.fixture
def client(app_run: Path) -> Iterator[TestClient]:
    with serving(app_run) as started:
        yield started


def rows(run_dir: Path, sql: str, *args: object) -> list[tuple[Any, ...]]:
    """Rows of ``app.sqlite``, through a connection of their own."""
    connection = sqlite3.connect(run_dir / "app.sqlite")
    try:
        return connection.execute(sql, args).fetchall()
    finally:
        connection.close()


def in_catalog(run_dir: Path, sql: str, *args: object) -> None:
    """Change the catalog of a run copy behind the app, as a renderer would."""
    connection = sqlite3.connect(run_dir / CATALOG_FILE)
    try:
        with connection:
            connection.execute(sql, args)
    finally:
        connection.close()


def set_app(run_dir: Path, **changes: object) -> None:
    config = load_config(run_dir)
    write_config(run_dir, with_changes(config, app=config.app.model_dump() | changes))


def shaped(response: Any, shape: type[BaseModel], status: int = 200) -> dict[str, Any]:
    """The JSON of a response, which has that status and that shape."""
    assert response.status_code == status, response.text
    body = response.json()
    shape.model_validate(body)
    return body


def refused(response: Any, status: int) -> list[tuple[str, str]]:
    """The (field, message) pairs of an answer that refuses with that status."""
    body = shaped(response, schemas.ErrorsOut, status)
    assert body["errors"]
    return [(error["field"], error["message"]) for error in body["errors"]]


def fields(response: Any, status: int) -> list[str]:
    return [where for where, _message in refused(response, status)]


def get_next(client: TestClient) -> dict[str, Any]:
    return shaped(client.get("/api/next"), schemas.NextOut)


def shown(client: TestClient) -> dict[str, Any]:
    """The assignment the rater is given now."""
    body = get_next(client)
    assert body["status"] == "assignment"
    return body["assignment"]


def rating(assignment: dict[str, Any] | int, **changes: Any) -> dict[str, Any]:
    assignment_id = assignment if isinstance(assignment, int) else assignment["assignment_id"]
    body = {
        "assignment_id": assignment_id,
        "protocol_version": "0.1-draft",
        "values": dict(NONE),
        "unratable": None,
        "other": False,
        "note": "",
        "marks": [],
        "client_ts": client_ts(),
        "ms_on_subject": 1200,
    }
    return body | changes


def rate(client: TestClient, assignment: dict[str, Any], **changes: Any) -> dict[str, Any]:
    return shaped(client.post("/api/rating", json=rating(assignment, **changes)), schemas.RatingOut)


def skip_body(assignment: dict[str, Any] | int, ms: int = 900) -> dict[str, Any]:
    assignment_id = assignment if isinstance(assignment, int) else assignment["assignment_id"]
    return {"assignment_id": assignment_id, "client_ts": client_ts(), "ms_on_subject": ms}


def skip(client: TestClient, assignment: dict[str, Any]) -> dict[str, Any]:
    return shaped(client.post("/api/skip", json=skip_body(assignment)), schemas.NextOut)


def rate_all(client: TestClient) -> int:
    done = 0
    while (body := get_next(client))["status"] == "assignment":
        rate(client, body["assignment"])
        done += 1
    assert body["status"] == "done"
    return done


def queue(run_dir: Path) -> list[str]:
    """The subjects of the run in the order a rater gets them: by ``primary_rank``."""
    return [str(uid) for uid in primary_scores(run_dir)["subject_uid"]]


def n_rows(run_dir: Path, table: str) -> int:
    return rows(run_dir, f"SELECT count(*) FROM {table}")[0][0]


def stored(run_dir: Path) -> dict[str, int]:
    """How many rows the tables that a rating writes hold."""
    return {
        table: n_rows(run_dir, table)
        for table in ("submissions", "ratings", "marks", "final_labels", "final_marks", "events")
    }


def mark(panel: str = "sagittal", **changes: Any) -> dict[str, Any]:
    return {"client_id": "m1", "class": "A", "panel": panel, "slice": 3, "px": 30.0, "py": 31.0} | (
        changes
    )


def asgi_get(
    app: Any, path: str, headers: list[tuple[str, str]] | None = None
) -> tuple[int, bytes]:
    """A GET with exactly that path and those headers, which a client library would
    normalize first."""
    sent: list[dict[str, Any]] = []
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [
            (name.lower().encode(), value.encode())
            for name, value in (headers or [("host", "127.0.0.1")])
        ],
        "server": ("127.0.0.1", 80),
        "client": ("127.0.0.1", 50000),
    }

    async def call() -> None:
        asked = False
        answered = asyncio.Event()

        async def receive() -> dict[str, Any]:
            # the request once; then nothing until the answer is out, as a connection that
            # stays open (a response that sends a file listens for the client leaving)
            nonlocal asked
            if not asked:
                asked = True
                return {"type": "http.request", "body": b"", "more_body": False}
            await answered.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)
            if message["type"] == "http.response.body" and not message.get("more_body"):
                answered.set()

        await app(scope, receive, send)

    asyncio.run(call())
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, body


def nearest(positions: np.ndarray, point: np.ndarray) -> tuple[int, float]:
    distances = np.linalg.norm(positions.astype(np.float64) - point, axis=1)
    return int(distances.argmin()), float(distances.min())


def mark_on_surface(run_dir: Path, subject_uid: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """A mark of class A that lies on a pial vertex of the subject, and what it has to
    resolve to, worked out here from the arrays of the vertex map without ``app.marks``."""
    directory = media_dir(run_dir, subject_uid)
    info = read_render(directory)
    assert info is not None and info.vertex_map is not None
    vmap = read_vertex_map(directory / info.vertex_map.file)
    affine = np.array(info.volume.vox2ras_tkr)
    best: tuple[float, str, int, float, float] | None = None
    for name in PANEL_ORDER:
        panel = info.panels[name]
        for hemi in ("lh", "rh"):
            for xyz in vmap[f"{hemi}_pial"][:: max(1, len(vmap[f"{hemi}_pial"]) // 200)]:
                px, py = geometry.tkr_to_pixel(panel, affine, xyz)
                if not (1 <= px <= panel.width - 1 and 1 <= py <= panel.height - 1):
                    continue
                for position in range(len(panel.slices)):
                    vox = geometry.pixel_to_voxel(panel, position, float(px), float(py))
                    off = float(np.linalg.norm(geometry.apply_affine(affine, vox) - xyz))
                    if best is None or off < best[0]:
                        best = (off, name, position, float(px), float(py))
    assert best is not None and best[0] < 1.0, "no pial vertex lies near a slice of the render"
    _off, name, position, px, py = best
    posted = {"client_id": "m1", "class": "A", "panel": name, "slice": position, "px": px, "py": py}
    point = geometry.apply_affine(
        affine, geometry.pixel_to_voxel(info.panels[name], position, px, py)
    )
    found = {hemi: nearest(vmap[f"{hemi}_pial"], point) for hemi in ("lh", "rh")}
    hemi = min(found, key=lambda h: found[h][1])
    row, pial_dist = found[hemi]
    white_row, white_dist = nearest(vmap[f"{hemi}_white"], point)
    expected = {
        "hemi": hemi,
        "pial_vertex": int(vmap[f"{hemi}_index"][row]),
        "pial_dist_mm": pial_dist,
        "white_vertex": int(vmap[f"{hemi}_index"][white_row]) if white_dist <= 5.0 else None,
        "fsavg6_vertex": int(vmap[f"{hemi}_fsavg6"][row]),
        "vertex_surface": "pial",
        "tkr": [float(value) for value in point],
    }
    return posted, expected


# --- the routes there are -----------------------------------------------------------------------


def test_the_app_serves_the_routes_of_the_schemas_and_no_other(client: TestClient) -> None:
    served = {
        (method, route.path.replace("{path:path}", "{path}"))
        for route in client.app.routes  # type: ignore[attr-defined]
        for method in route.methods
    }
    assert served == {(route.method, route.path) for route in schemas.ROUTES}
    # no documentation pages, and nothing else by another name
    for path in ("/docs", "/redoc", "/openapi.json", "/api", "/api/", "/media", "/favicon.ico"):
        assert fields(client.get(path), 404) == ["request"]
    assert fields(client.post("/api/next"), 405) == ["request"]
    assert fields(client.get("/api/rating"), 405) == ["request"]
    assert fields(client.put("/api/rating", json={}), 405) == ["request"]


def test_health_says_the_version(client: TestClient) -> None:
    body = shaped(client.get("/healthz"), schemas.HealthOut)
    assert body == {"ok": True, "version": surfcensus.__version__}


def test_the_pages_are_the_templates_with_three_variables_and_no_subject(
    client: TestClient, app_run: Path, small_run_uids: dict[str, str]
) -> None:
    for route, script in (("/", "js/rate.js"), ("/review", "js/review.js")):
        response = client.get(route)
        assert response.status_code == 200 and response.headers["cache-control"] == "no-cache"
        assert response.headers["content-type"].startswith("text/html")
        text = response.text
        assert f'src="/static/{script}?v={surfcensus.__version__}"' in text
        assert 'data-mode="local"' in text
        for label, uid in small_run_uids.items():
            assert label not in text and uid not in text
        assert str(app_run) not in text


def test_the_session_is_the_protocol_the_rater_and_the_settings(
    client: TestClient, app_run: Path, shipped_protocol: Protocol
) -> None:
    body = shaped(client.get("/api/session"), schemas.SessionOut)
    assert body["mode"] == "local" and body["app_version"] == surfcensus.__version__
    assert body["rater"] == {"id": 1, "name": "local", "role": "lead"}
    assert Protocol.model_validate(body["protocol"]) == shipped_protocol
    digest = hashlib.sha256((app_run / PROTOCOL_FILE).read_bytes()).hexdigest()
    assert body["protocol_sha256"] == digest
    # the shipped protocol asks for marks; a local run does not unless it says so
    assert shipped_protocol.require_marks_for_positive and body["require_marks"] is False
    assert body["settings"] == {"mark_show_mm": 3.0, "blind": False}
    assert body["examples"] == []
    assert rows(app_run, "SELECT name, role, token_hash FROM raters") == [("local", "lead", None)]


def test_the_session_follows_the_settings_of_the_run_and_the_rater_named(app_run: Path) -> None:
    set_app(app_run, blind=True, require_marks=True, mark_show_mm=1.5, rater_name="Ada")
    with serving(app_run) as client:
        body = shaped(client.get("/api/session"), schemas.SessionOut)
        assert body["rater"]["name"] == "Ada" and body["require_marks"] is True
        assert body["settings"] == {"mark_show_mm": 1.5, "blind": True}
    with serving(app_run, rater="Grace") as client:
        assert client.get("/api/session").json()["rater"] == {
            "id": 2,
            "name": "Grace",
            "role": "lead",
        }


# --- next ---------------------------------------------------------------------------------------


def test_next_gives_the_first_subject_of_the_queue_with_its_render_and_what_follows(
    client: TestClient,
    app_run: Path,
    small_run_uids: dict[str, str],
    small_run_renders: dict[str, RenderInfo],
) -> None:
    order = queue(app_run)
    assert order[0] == small_run_uids["sub-001"]
    body = get_next(client)
    assignment = body["assignment"]
    with PublicCatalog.open(app_run) as public:
        key = public.render(order[0]).render_key  # type: ignore[union-attr]
    assert {name: value for name, value in assignment.items() if name != "render"} == {
        "assignment_id": 1,
        "kind": "main",
        "subject_uid": order[0],
        "media_base": f"/media/{order[0]}/",
        "render_key": key,
        "has_vertex_map": True,
        "previous": None,
    }
    # the render as a page gets it: render.json without the digests of its sources
    info = small_run_renders["sub-001"]
    assert assignment["render"] == schemas.render_out(info).model_dump(mode="json")
    assert "inputs" not in assignment["render"] and "template" not in assignment["render"]
    assert [entry["subject_uid"] for entry in body["prefetch"]] == order[1:]
    assert body["prefetch"][0] == {
        "subject_uid": order[1],
        "media_base": f"/media/{order[1]}/",
        "render_key": key,
        "n_slices": {"sagittal": 7, "coronal": 6, "axial": 7},
        "tiles": info.tiles.model_dump(),
    }
    assert body["progress"] == {
        "done_today": 0,
        "done_total": 0,
        "remaining": 3,
        "skipped": 0,
        "rendered": 3,
        "n_subjects": 3,
    }
    assert body["retry_after_s"] is None
    assert client.get("/api/next").headers["cache-control"] == "no-store"


def test_next_is_the_same_assignment_until_it_is_rated(client: TestClient, app_run: Path) -> None:
    first = shown(client)
    assert shown(client) == first and shown(client) == first
    assert rows(app_run, "SELECT assignment_id, seq, kind, completed_at FROM assignments") == [
        (1, 1, "main", None)
    ]
    following = rate(client, first)["next"]
    assert following["assignment"]["assignment_id"] == 2
    assert shown(client) == following["assignment"]
    assert n_rows(app_run, "assignments") == 2


def test_the_queue_ends_with_done_and_counts_what_was_rated(
    client: TestClient, app_run: Path
) -> None:
    assert rate_all(client) == 3
    body = get_next(client)
    assert body["status"] == "done" and body["assignment"] is None and body["prefetch"] == []
    assert body["progress"] == {
        "done_today": 3,
        "done_total": 3,
        "remaining": 0,
        "skipped": 0,
        "rendered": 3,
        "n_subjects": 3,
    }
    # done today is by the clock: a rating of another day is done in total only
    connection = sqlite3.connect(app_run / "app.sqlite")
    with connection:
        connection.execute(
            "UPDATE assignments SET completed_at = '2026-01-02T12:00:00.000Z' WHERE seq = 1"
        )
    connection.close()
    progress = get_next(client)["progress"]
    assert (progress["done_today"], progress["done_total"]) == (2, 3)


def test_while_a_render_is_awaited_the_rater_is_told_to_ask_again(
    client: TestClient, app_run: Path
) -> None:
    last = queue(app_run)[-1]
    in_catalog(app_run, "DELETE FROM renders WHERE subject_uid = ?", last)
    first = get_next(client)
    assert first["progress"] | {"remaining": 3, "rendered": 2, "n_subjects": 3} == first["progress"]
    assert [entry["subject_uid"] for entry in first["prefetch"]] != [] and last not in [
        entry["subject_uid"] for entry in first["prefetch"]
    ]
    rate(client, first["assignment"])
    rate(client, shown(client))
    waiting = get_next(client)
    assert waiting["status"] == "rendering" and waiting["retry_after_s"] == RETRY_AFTER_S
    assert waiting["assignment"] is None
    assert waiting["progress"] == {
        "done_today": 2,
        "done_total": 2,
        "remaining": 1,
        "skipped": 0,
        "rendered": 2,
        "n_subjects": 3,
    }
    assert n_rows(app_run, "assignments") == 2


def test_a_render_that_fails_while_the_server_runs_gets_its_label_and_is_not_waited_for(
    client: TestClient, app_run: Path
) -> None:
    order = queue(app_run)
    assert rows(app_run, "SELECT * FROM final_labels") == []
    in_catalog(
        app_run,
        "UPDATE renders SET status = 'failed', reason = 'wrong orientation', render_key = NULL "
        "WHERE subject_uid = ?",
        order[1],
    )
    body = get_next(client)
    assert body["assignment"]["subject_uid"] == order[0]
    assert body["progress"] | {"remaining": 2, "rendered": 2, "n_subjects": 2} == body["progress"]
    # "wrong orientation" is the second unratable reason of the shipped protocol
    assert rows(app_run, "SELECT * FROM final_labels") == [(order[1], "unratable", 1, "single")]
    rate(client, body["assignment"])
    assert shown(client)["subject_uid"] == order[2]
    assert fields(client.get(f"/media/{order[1]}/poster.webp"), 404) == ["path"]


def test_a_run_of_the_phase_commands_is_served_without_a_selection(app_run: Path) -> None:
    # `inventory`, `metrics` and `render` write no selection; only `review` does. The
    # selection of a local run is then every present subject, as `render` reads it
    order = queue(app_run)
    in_catalog(app_run, "DELETE FROM selection")
    in_catalog(
        app_run,
        "UPDATE renders SET status = 'failed', reason = 'missing surface', render_key = NULL "
        "WHERE subject_uid = ?",
        order[1],
    )
    with serving(app_run) as client:
        body = get_next(client)
        assert body["status"] == "assignment"
        assert body["assignment"]["subject_uid"] == order[0]
        expected = {"remaining": 2, "rendered": 2, "n_subjects": 2}
        assert body["progress"] | expected == body["progress"]
        # the subject whose render failed has its label, as in a run that `review` made
        assert rows(app_run, "SELECT * FROM final_labels") == [(order[1], "unratable", 2, "single")]
        assert rate_all(client) == 2
        assert n_rows(app_run, "final_labels") == 7
    # a frozen calibration set on such a run does not become its queue
    in_catalog(
        app_run,
        "INSERT INTO selection (subject_uid, reason, stratum, inclusion_prob, seed) "
        "VALUES (?, 'calibration', 'calibration', 1.0, 0)",
        order[2],
    )
    with serving(app_run, rater="Grace") as grace:
        progress = get_next(grace)["progress"]
        assert progress | expected == progress


def test_a_render_the_run_directory_has_lost_is_no_assignment(
    client: TestClient, app_run: Path
) -> None:
    first = queue(app_run)[0]
    (media_dir(app_run, first) / RENDER_JSON).unlink()
    where, message = refused(client.get("/api/next"), 503)[0]
    assert where == "request" and "surfcensus render" in message
    # nothing was assigned for a subject that cannot be shown
    assert n_rows(app_run, "assignments") == 0


# --- a rating -----------------------------------------------------------------------------------


def test_a_rating_is_stored_with_its_marks_and_answered_with_the_next_subject(
    client: TestClient, app_run: Path
) -> None:
    first = shown(client)
    posted, expected = mark_on_surface(app_run, first["subject_uid"])
    sent = rating(
        first,
        values={"A": 2, "B": 1, "C": 0},
        marks=[posted | {"frame": None, "mode": None, "ts": "2026-11-13T09:59:58.000Z"}],
        note="a note",
        ms_on_subject=14200,
    )
    body = shaped(client.post("/api/rating", json=sent), schemas.RatingOut)
    saved = body["saved"]
    assert (saved["assignment_id"], saved["submission_id"], saved["revision"]) == (1, 1, 1)
    (stored_mark,) = saved["marks"]
    assert (stored_mark["client_id"], stored_mark["mark_id"]) == ("m1", 1)
    # the vertices are those nearest in the vertex map, found here without the app
    for name in ("hemi", "pial_vertex", "white_vertex", "fsavg6_vertex", "vertex_surface"):
        assert stored_mark[name] == expected[name], name
    assert stored_mark["pial_dist_mm"] == pytest.approx(expected["pial_dist_mm"], abs=1e-6)
    assert stored_mark["fsavg6_vertex"] >= 0
    assert body["next"]["assignment"]["assignment_id"] == 2
    assert body["next"]["progress"]["done_total"] == 1

    assert rows(
        app_run,
        "SELECT assignment_id, revision, current, unratable_reason, other, note, protocol_version,"
        " protocol_sha256, calibration_sha256, app_version, client_ts, ms_on_subject"
        " FROM submissions",
    ) == [
        (
            1,
            1,
            1,
            None,
            0,
            "a note",
            "0.1-draft",
            hashlib.sha256((app_run / PROTOCOL_FILE).read_bytes()).hexdigest(),
            None,
            surfcensus.__version__,
            sent["client_ts"],
            14200,
        )
    ]
    assert rows(app_run, "SELECT criterion, value FROM ratings ORDER BY criterion") == [
        ("A", 2),
        ("B", 1),
        ("C", 0),
    ]
    row = rows(
        app_run,
        "SELECT class, panel, slice, px, py, tkr_x, tkr_y, tkr_z, hemi, pial_vertex, created_ts,"
        " deleted_ts FROM marks",
    )[0]
    assert row[:3] == ("A", posted["panel"], posted["slice"])
    assert row[3:5] == (posted["px"], posted["py"])
    assert row[5:8] == pytest.approx(expected["tkr"], abs=1e-9)
    assert row[8:] == (expected["hemi"], expected["pial_vertex"], "2026-11-13T09:59:58.000Z", None)
    assert rows(app_run, "SELECT completed_at IS NOT NULL FROM assignments WHERE seq = 1") == [(1,)]
    # one rater: the rating is the final label, and its mark the final mark
    uid = first["subject_uid"]
    assert rows(app_run, "SELECT * FROM final_labels ORDER BY criterion") == [
        (uid, "A", 2, "single"),
        (uid, "B", 1, "single"),
        (uid, "C", 0, "single"),
    ]
    assert rows(
        app_run, "SELECT subject_uid, class, hemi, pial_vertex, fsavg6_vertex FROM final_marks"
    ) == [(uid, "A", expected["hemi"], expected["pial_vertex"], expected["fsavg6_vertex"])]


def test_an_unratable_rating_stores_the_reason_and_neither_values_nor_marks(
    client: TestClient, app_run: Path
) -> None:
    first = shown(client)
    body = rate(
        client,
        first,
        values={},
        unratable="image too poor to judge",
        marks=[mark(slice=99)],  # ignored, so not looked at either
    )
    assert body["saved"]["marks"] == []
    assert rows(app_run, "SELECT unratable_reason FROM submissions") == [
        ("image too poor to judge",)
    ]
    assert stored(app_run) | {"ratings": 0, "marks": 0, "final_marks": 0} == stored(app_run)
    assert rows(app_run, "SELECT criterion, value, method FROM final_labels") == [
        ("unratable", 3, "single")
    ]


def test_a_mark_far_from_every_vertex_is_kept_as_a_point(client: TestClient, app_run: Path) -> None:
    first = shown(client)
    corner = mark(panel="coronal", slice=0, px=0.0, py=0.0, **{"class": "X"})
    body = rate(client, first, marks=[corner])
    (saved,) = body["saved"]["marks"]
    assert saved["hemi"] is None and saved["pial_vertex"] is None and saved["fsavg6_vertex"] is None
    assert saved["vertex_surface"] is None and saved["box_dist_mm"] > 0
    assert rows(app_run, "SELECT hemi, pial_vertex, tkr_x IS NOT NULL FROM marks") == [
        (None, None, 1)
    ]


def test_marks_are_stored_as_points_when_the_vertex_map_cannot_be_read(
    client: TestClient, app_run: Path
) -> None:
    first = shown(client)
    posted, expected = mark_on_surface(app_run, first["subject_uid"])
    vertex_map = media_dir(app_run, first["subject_uid"]) / "vertex_map.npz"
    vertex_map.write_bytes(vertex_map.read_bytes()[:500])
    body = rate(client, first, values={"A": 1, "B": 0, "C": 0}, marks=[posted])
    (saved,) = body["saved"]["marks"]
    assert saved["hemi"] is None and saved["pial_vertex"] is None
    assert rows(app_run, "SELECT tkr_x, tkr_y, tkr_z FROM marks")[0] == pytest.approx(
        expected["tkr"], abs=1e-9
    )
    # a render whose vertex map is gone says so, and its marks are points as well
    second = body["next"]["assignment"]
    assert second["has_vertex_map"] is True
    posted, expected = mark_on_surface(app_run, second["subject_uid"])
    (media_dir(app_run, second["subject_uid"]) / "vertex_map.npz").unlink()
    second = shown(client)
    assert second["has_vertex_map"] is False
    (saved,) = rate(client, second, marks=[posted | {"class": "X"}])["saved"]["marks"]
    assert saved["hemi"] is None and saved["pial_vertex"] is None
    assert rows(app_run, "SELECT tkr_x, tkr_y, tkr_z FROM marks WHERE class = 'X'")[
        0
    ] == pytest.approx(expected["tkr"], abs=1e-9)


@pytest.mark.parametrize(
    ("changes", "where"),
    [
        # rule 2: unratable for a reason of the protocol, or a value for each criterion
        ({"unratable": "the rater was tired", "values": {}}, ["unratable"]),
        ({"values": {"A": 0, "B": 0}}, ["values"]),
        ({"values": {}}, ["values"]),
        ({"values": {"A": 0, "B": 0, "C": 0, "D": 0}}, ["values"]),
        ({"values": {"A": 0, "B": 3, "C": -1}}, ["values.B", "values.C"]),
        # rule 3: every mark lies on a tile of the render
        ({"marks": [mark(slice=7)]}, ["marks.0.slice"]),
        ({"marks": [mark(), mark(client_id="m2", px=64.5)]}, ["marks.1.px"]),
        (
            {"marks": [mark(py=-0.5), mark(client_id="m2", slice=-1)]},
            ["marks.0.py", "marks.1.slice"],
        ),
        ({"marks": [mark(frame=0, mode="sync")]}, ["marks.0.frame"]),
        # rule 5: another problem says what it is
        ({"other": True}, ["note"]),
        ({"other": True, "note": "  \n"}, ["note"]),
        ({"other": True, "unratable": "render failure", "values": {}}, ["note"]),
        # the first rule that fails is the one reported
        ({"values": {"A": 9}, "marks": [mark(slice=7)], "other": True}, ["values"]),
        ({"marks": [mark(slice=7)], "other": True}, ["marks.0.slice"]),
        # what the shape itself refuses
        ({"ms_on_subject": -1}, ["ms_on_subject"]),
        ({"marks": [mark(), mark()]}, ["request"]),
        ({"marks": [mark(panel="oblique")]}, ["marks.0.panel"]),
        ({"marks": [mark(px="NaN")]}, ["marks.0.px"]),
        ({"note": "x" * 10_001}, ["note"]),
        ({"subject_label": "sub-001"}, ["subject_label"]),
    ],
)
def test_a_rating_that_breaks_a_rule_names_the_field_and_stores_nothing(
    client: TestClient, app_run: Path, changes: dict[str, Any], where: list[str]
) -> None:
    first = shown(client)
    before = stored(app_run)
    response = client.post("/api/rating", json=rating(first, **changes))
    found = fields(response, 422)
    if where == ["request"]:
        assert len(found) == 1
    else:
        assert found == where
    assert stored(app_run) == before
    assert rows(app_run, "SELECT completed_at FROM assignments") == [(None,)]
    assert shown(client) == first


@pytest.mark.parametrize("number", [2**63, 2**70, -(2**63) - 1])
def test_a_number_that_no_column_can_hold_is_refused_and_no_error(
    client: TestClient, app_run: Path, number: int
) -> None:
    # JSON has no largest integer; SQLite has
    first = shown(client)
    before = stored(app_run)
    events = {"assignment_id": number, "events": [{"ts": "t", "type": "play"}]}
    for path, body, where in (
        ("/api/rating", rating(number), "assignment_id"),
        ("/api/rating", rating(first, ms_on_subject=number), "ms_on_subject"),
        ("/api/rating", rating(first, marks=[mark(frame=number)]), "marks.0.frame"),
        ("/api/rating", rating(first, marks=[mark(slice=number)]), "marks.0.slice"),
        ("/api/rating", rating(first, values=NONE | {"B": number}), "values.B"),
        ("/api/skip", skip_body(number), "assignment_id"),
        ("/api/skip", skip_body(first, ms=number), "ms_on_subject"),
        ("/api/events", events, "assignment_id"),
    ):
        assert fields(client.post(path, json=body), 422) == [where], (path, where)
    assert stored(app_run) == before
    assert rows(app_run, "SELECT completed_at, skip_count FROM assignments") == [(None, 0)]
    # the largest that fits names no assignment; a payload is text, whatever it holds
    assert fields(client.post("/api/skip", json=skip_body(2**63 - 1)), 404) == ["assignment_id"]
    kept = {"assignment_id": 1, "events": [{"ts": "t", "type": "zoom", "payload": {"n": number}}]}
    assert client.post("/api/events", json=kept).json() == {"stored": 1}
    assert rows(app_run, "SELECT payload_json FROM events WHERE type = 'zoom'") == [
        (f'{{"n":{number}}}',)
    ]


def test_marks_are_required_for_a_positive_value_when_the_run_says_so(app_run: Path) -> None:
    set_app(app_run, require_marks=True)
    with serving(app_run) as client:
        first = shown(client)
        sent = rating(first, values={"A": 2, "B": 1, "C": 0}, marks=[mark(**{"class": "B"})])
        assert refused(client.post("/api/rating", json=sent), 422) == [
            ("marks", "criterion A is rated major but has no mark of class A")
        ]
        sent = rating(first, values={"A": 2, "B": 1, "C": 0}, marks=[mark(**{"class": "X"})])
        assert refused(client.post("/api/rating", json=sent), 422) == [
            ("marks", "criterion A is rated major but has no mark of class A"),
            ("marks", "criterion B is rated minor but has no mark of class B"),
        ]
        assert stored(app_run)["submissions"] == 0
        # none needs no mark, unratable needs none, and a mark of each class passes
        rate(client, first, unratable="render failure", values={})
        second = shown(client)
        both = [mark(), mark(client_id="m2", **{"class": "B"})]
        rate(client, second, values={"A": 2, "B": 1, "C": 0}, marks=both)
        rate(client, shown(client))
    assert stored(app_run)["submissions"] == 3


def test_a_rating_of_no_assignment_or_of_another_rater_is_refused(app_run: Path) -> None:
    with serving(app_run, rater="Grace") as other:
        theirs = shown(other)
    with serving(app_run) as client:
        assert fields(client.post("/api/rating", json=rating(999)), 404) == ["assignment_id"]
        assert fields(client.post("/api/rating", json=rating(theirs)), 403) == ["assignment_id"]
        assert fields(client.post("/api/skip", json=skip_body(theirs)), 403) == ["assignment_id"]
        events = {"assignment_id": theirs["assignment_id"], "events": []}
        assert fields(client.post("/api/events", json=events), 403) == ["assignment_id"]
        assert stored(app_run)["submissions"] == 0
        assert rows(app_run, "SELECT skip_count FROM assignments") == [(0,)]


def test_a_page_with_another_protocol_version_is_told_to_load_again(
    client: TestClient, app_run: Path
) -> None:
    first = shown(client)
    response = client.post("/api/rating", json=rating(first, protocol_version="0.2"))
    assert fields(response, 409) == ["protocol_version"]
    assert stored(app_run)["submissions"] == 0
    rate(client, first)


def new_render_key(run_dir: Path, subject_uid: str, key: str = "f" * 16) -> None:
    """Record that the subject was rendered again from other files."""
    with PublicCatalog.open(run_dir, readonly=False) as public:
        row = public.render(subject_uid)
        assert row is not None
        public.record_renders([dataclasses.replace(row, render_key=key, rendered_at=None)])


def test_a_rating_or_a_skip_on_tiles_that_were_rendered_again_is_refused(
    client: TestClient, app_run: Path
) -> None:
    first = shown(client)
    new_render_key(app_run, first["subject_uid"])
    assert fields(client.post("/api/rating", json=rating(first)), 409) == ["assignment_id"]
    assert fields(client.post("/api/skip", json=skip_body(first)), 409) == ["assignment_id"]
    assert stored(app_run)["submissions"] == 0
    assert rows(app_run, "SELECT skip_count, skipped_at FROM assignments") == [(0, None)]
    # the subject is offered again on its present tiles, under a new assignment
    again = shown(client)
    assert again["subject_uid"] == first["subject_uid"] and again["assignment_id"] == 2
    assert again["render_key"] == "f" * 16
    following = rate(client, again)["next"]["assignment"]
    assert following["assignment_id"] == 3 and following["subject_uid"] != first["subject_uid"]
    # the assignment of the earlier tiles stays as it was left: open, and never offered
    assert rows(app_run, "SELECT assignment_id, completed_at IS NULL FROM assignments") == [
        (1, 1),
        (2, 0),
        (3, 1),
    ]


def test_a_rating_that_arrives_twice_is_one_submission(client: TestClient, app_run: Path) -> None:
    first = shown(client)
    posted, _expected = mark_on_surface(app_run, first["subject_uid"])
    sent = rating(first, values={"A": 1, "B": 0, "C": 0}, marks=[posted])
    one = shaped(client.post("/api/rating", json=sent), schemas.RatingOut)
    two = shaped(client.post("/api/rating", json=sent), schemas.RatingOut)
    assert one["saved"] == two["saved"] and one["next"] == two["next"]
    assert stored(app_run) | {"submissions": 1, "marks": 1} == stored(app_run)
    # the same press of Enter is known by its time: another time is a revision
    three = shaped(
        client.post("/api/rating", json=sent | {"client_ts": client_ts()}), schemas.RatingOut
    )
    assert three["saved"]["revision"] == 2 and three["saved"]["marks"][0]["mark_id"] == 2


RATED: dict[str, Any] = {"values": {"A": 1, "B": 0, "C": 0}, "note": "n"}
UNRATABLE: dict[str, Any] = {"unratable": "image too poor to judge", "note": "n"}


@pytest.mark.parametrize(
    ("base", "marked", "change"),
    [
        (RATED, True, {"other": True}),
        (RATED, True, {"note": "changed"}),
        (RATED, True, {"values": {"A": 1, "B": 1, "C": 0}}),
        (RATED, True, {"unratable": "image too poor to judge"}),
        (UNRATABLE, False, {"unratable": "wrong orientation"}),
        (RATED, True, {"marks": []}),
        (RATED, True, {"marks": "class"}),
        (RATED, True, {"marks": "panel"}),
        (RATED, True, {"marks": "slice"}),
        (RATED, True, {"marks": "px"}),
        (RATED, True, {"marks": "py"}),
    ],
    ids=[
        "other",
        "note",
        "values",
        "unratable",
        "reason",
        "no_marks",
        "mark_class",
        "mark_panel",
        "mark_slice",
        "mark_px",
        "mark_py",
    ],
)
def test_a_resent_time_with_another_body_is_a_revision(
    client: TestClient, app_run: Path, base: dict[str, Any], marked: bool, change: dict[str, Any]
) -> None:
    """One press of ``Enter`` is known by its client time and its body. A body that differs
    in one field under the same time is another press, so a new revision, and is not
    answered as the submission that is stored."""
    first = shown(client)
    posted, _expected = mark_on_surface(app_run, first["subject_uid"])
    sent = rating(first, marks=[posted] if marked else [], **base)
    one = shaped(client.post("/api/rating", json=sent), schemas.RatingOut)
    assert one["saved"]["revision"] == 1
    if isinstance(part := change.get("marks"), str):
        panels = first["render"]["panels"]
        other_panel = next(
            name
            for name in PANEL_ORDER
            if name != posted["panel"] and posted["slice"] < len(panels[name]["slices"])
        )
        moved = {
            "class": "X",
            "panel": other_panel,
            "slice": posted["slice"] + (1 if posted["slice"] == 0 else -1),
            "px": posted["px"] + 0.25,
            "py": posted["py"] + 0.25,
        }
        change = {"marks": [posted | {part: moved[part]}]}
    two = shaped(client.post("/api/rating", json=sent | change), schemas.RatingOut)
    assert two["saved"]["revision"] == 2
    assert rows(app_run, "SELECT revision, current FROM submissions ORDER BY revision") == [
        (1, 0),
        (2, 1),
    ]


# --- back, and a revision -----------------------------------------------------------------------


def test_back_is_a_404_before_anything_was_left(client: TestClient) -> None:
    assert fields(client.get("/api/back"), 404) == ["request"]
    shown(client)
    assert fields(client.get("/api/back"), 404) == ["request"]


def test_back_shows_the_rating_and_a_second_submission_is_a_revision(
    client: TestClient, app_run: Path
) -> None:
    first = shown(client)
    posted, _expected = mark_on_surface(app_run, first["subject_uid"])
    sent = rating(
        first, values={"A": 2, "B": 0, "C": 0}, marks=[posted | {"ts": "t-click"}], note="first"
    )
    one = shaped(client.post("/api/rating", json=sent), schemas.RatingOut)
    second = one["next"]["assignment"]

    back = shaped(client.get("/api/back"), schemas.NextOut)
    assert back["status"] == "assignment" and back["prefetch"] == []
    again = back["assignment"]
    assert {key: again[key] for key in first if key != "previous"} == {
        key: first[key] for key in first if key != "previous"
    }
    previous = again["previous"]
    assert (previous["values"], previous["note"]) == ({"A": 2, "B": 0, "C": 0}, "first")
    assert (previous["unratable"], previous["other"]) == (None, False)
    (old_mark,) = previous["marks"]
    saved = one["saved"]["marks"][0]
    assert old_mark == posted | saved | {
        "client_id": f"m{saved['mark_id']}",
        "frame": None,
        "mode": None,
        "ts": "t-click",
    }
    # going back assigns nothing and moves nothing: the open subject is still the next one
    assert shown(client) == second and n_rows(app_run, "assignments") == 2

    revised = rate(client, again, values={"A": 1, "B": 0, "C": 0}, marks=[], note="second")
    assert revised["saved"] | {"submission_id": 2, "revision": 2, "marks": []} == revised["saved"]
    assert revised["next"]["assignment"] == second
    assert rows(app_run, "SELECT revision, current, note FROM submissions ORDER BY revision") == [
        (1, 0, "first"),
        (2, 1, "second"),
    ]
    # nothing is overwritten: the values and the mark of revision 1 are still there
    assert rows(
        app_run, "SELECT submission_id, criterion, value FROM ratings WHERE criterion = 'A'"
    ) == [
        (1, "A", 2),
        (2, "A", 1),
    ]
    assert rows(app_run, "SELECT submission_id, deleted_ts IS NOT NULL FROM marks") == [(1, 1)]
    assert rows(app_run, "SELECT value FROM final_labels WHERE criterion = 'A'") == [(1,)]
    assert n_rows(app_run, "final_marks") == 0
    # what comes back for the next revision is the current submission, not the one before
    previous = shaped(client.get("/api/back"), schemas.NextOut)["assignment"]["previous"]
    assert (previous["values"], previous["note"]) == ({"A": 1, "B": 0, "C": 0}, "second")
    assert previous["marks"] == []
    assert revised["next"]["progress"]["done_total"] == 1
    third = rate(client, again, values={"A": 2, "B": 0, "C": 0}, marks=[posted])
    (new_mark,) = third["saved"]["marks"]
    previous = shaped(client.get("/api/back"), schemas.NextOut)["assignment"]["previous"]
    assert [found["mark_id"] for found in previous["marks"]] == [new_mark["mark_id"]]
    assert new_mark["mark_id"] != saved["mark_id"]


def test_only_the_subject_left_last_can_be_rated_again(client: TestClient, app_run: Path) -> None:
    first = shown(client)
    rate(client, first)
    second = shown(client)
    rate(client, second, note="of the second")
    back = shaped(client.get("/api/back"), schemas.NextOut)["assignment"]
    assert back["assignment_id"] == second["assignment_id"]
    assert back["previous"]["note"] == "of the second"
    before = stored(app_run)
    assert fields(client.post("/api/rating", json=rating(first, note="late")), 409) == [
        "assignment_id"
    ]
    assert stored(app_run) == before
    rate(client, second, note="again")
    assert rows(app_run, "SELECT assignment_id, max(revision) FROM submissions GROUP BY 1") == [
        (1, 1),
        (2, 2),
    ]


def test_back_is_refused_when_the_tiles_of_that_subject_changed(
    client: TestClient, app_run: Path
) -> None:
    first = shown(client)
    rate(client, first)
    new_render_key(app_run, first["subject_uid"])
    assert fields(client.get("/api/back"), 409) == ["request"]


# --- skip ---------------------------------------------------------------------------------------


def test_a_skip_stores_no_rating_and_the_subject_comes_back_after_the_others(
    client: TestClient, app_run: Path
) -> None:
    first = shown(client)
    before = stored(app_run)
    sent = skip_body(first, ms=9400)
    body = shaped(client.post("/api/skip", json=sent), schemas.NextOut)
    second = body["assignment"]
    assert second["assignment_id"] == 2 and second["subject_uid"] != first["subject_uid"]
    assert body["progress"] == {
        "done_today": 0,
        "done_total": 0,
        "remaining": 3,
        "skipped": 1,
        "rendered": 3,
        "n_subjects": 3,
    }
    # what is fetched ahead: the subject not shown yet, then the one that waits
    assert [entry["subject_uid"] for entry in body["prefetch"]][-1] == first["subject_uid"]
    assert stored(app_run) == before | {"events": before["events"] + 1}
    assert rows(
        app_run, "SELECT completed_at, skipped_at IS NOT NULL, skip_count FROM assignments"
    ) == [(None, 1, 1), (None, 0, 0)]
    ((assignment_id, ts, payload),) = rows(
        app_run, "SELECT assignment_id, ts, payload_json FROM events WHERE type = 'skip'"
    )
    assert (assignment_id, ts) == (1, sent["client_ts"])
    said = json.loads(payload)
    assert (said["ms_on_subject"], said["n"]) == (9400, 1) and set(said) == {
        "ms_on_subject",
        "n",
        "server_ts",
    }
    # a reload does not bring the skipped subject back: the next one is on the page
    assert shown(client) == second
    rate(client, second)
    third = shown(client)
    assert third["subject_uid"] not in (first["subject_uid"], second["subject_uid"])
    last = rate(client, third)["next"]
    # everything else is rated: now the skipped one is here again, under its assignment
    assert last["assignment"] == first
    assert last["progress"] | {"remaining": 1, "skipped": 0, "done_total": 2} == last["progress"]
    done = rate(client, first)["next"]
    assert done["status"] == "done" and done["progress"]["remaining"] == 0
    assert rows(app_run, "SELECT skip_count, completed_at IS NOT NULL FROM assignments") == [
        (1, 1),
        (0, 1),
        (0, 1),
    ]
    assert n_rows(app_run, "assignments") == 3


def test_a_skip_request_that_arrives_twice_is_one_skip(client: TestClient, app_run: Path) -> None:
    first = shown(client)
    sent = skip_body(first)
    one = shaped(client.post("/api/skip", json=sent), schemas.NextOut)
    two = shaped(client.post("/api/skip", json=sent), schemas.NextOut)
    assert one == two
    assert rows(app_run, "SELECT skip_count FROM assignments WHERE seq = 1") == [(1,)]
    assert rows(app_run, "SELECT count(*) FROM events WHERE type = 'skip'") == [(1,)]
    assert n_rows(app_run, "assignments") == 2
    # the same assignment at another time is a second skip
    three = shaped(client.post("/api/skip", json=skip_body(first)), schemas.NextOut)
    assert three["assignment"] == one["assignment"]
    assert rows(app_run, "SELECT skip_count FROM assignments WHERE seq = 1") == [(2,)]
    payloads = rows(app_run, "SELECT payload_json FROM events WHERE type = 'skip' ORDER BY 1")
    assert sorted(json.loads(text)["n"] for (text,) in payloads) == [1, 2]


def test_a_skip_of_an_assignment_that_is_not_open_is_refused(
    client: TestClient, app_run: Path
) -> None:
    first = shown(client)
    rate(client, first)
    before = stored(app_run)
    assert refused(client.post("/api/skip", json=skip_body(first)), 409)[0][0] == "assignment_id"
    assert fields(client.post("/api/skip", json=skip_body(999)), 404) == ["assignment_id"]
    assert stored(app_run) == before
    assert rows(app_run, "SELECT skip_count, skipped_at FROM assignments WHERE seq = 1") == [
        (0, None)
    ]
    # the shape of a skip has nothing of a rating in it
    extra = skip_body(shown(client)) | {"values": NONE}
    assert fields(client.post("/api/skip", json=extra), 422) == ["values"]
    without_time = {"assignment_id": 2, "ms_on_subject": 5}
    assert fields(client.post("/api/skip", json=without_time), 422) == ["client_ts"]
    assert stored(app_run) == before


def test_backspace_after_a_skip_is_the_skipped_subject_as_it_was_left(
    client: TestClient, app_run: Path
) -> None:
    first = shown(client)
    rate(client, first, note="rated")
    second = shown(client)
    third = skip(client, second)["assignment"]
    back = shaped(client.get("/api/back"), schemas.NextOut)
    # the one left last is the skipped one, not the one rated before it; it has no rating
    assert back["assignment"] == second and back["assignment"]["previous"] is None
    # it is on the page, so it does not wait
    assert back["progress"] | {"skipped": 0, "remaining": 2} == back["progress"]
    assert rows(app_run, "SELECT skip_count FROM assignments WHERE seq = 2") == [(1,)]
    # it can be rated from there, and then it is the one to go back to, with its rating
    assert rate(client, second, note="after all")["next"]["assignment"] == third
    back = shaped(client.get("/api/back"), schemas.NextOut)["assignment"]
    assert back["assignment_id"] == second["assignment_id"]
    assert back["previous"]["note"] == "after all"
    # and the first can no longer be rated again
    assert fields(client.post("/api/rating", json=rating(first)), 409) == ["assignment_id"]


def test_skipping_goes_round_the_skipped_subjects_and_the_last_one_stays(
    client: TestClient, app_run: Path
) -> None:
    first = shown(client)
    second = skip(client, first)["assignment"]
    third = skip(client, second)["assignment"]
    body = skip(client, third)
    # all three were shown: the one skipped longest ago is first
    assert body["assignment"] == first
    assert body["progress"] | {"skipped": 2, "remaining": 3} == body["progress"]
    assert skip(client, first)["assignment"] == second
    rate(client, second)
    rate(client, shown(client))
    only = shown(client)
    again = skip(client, only)
    assert again["assignment"] == only and again["progress"]["skipped"] == 0
    assert n_rows(app_run, "assignments") == 3 and n_rows(app_run, "submissions") == 2
    # the first was skipped three times, the two others once each
    assert rows(app_run, "SELECT skip_count FROM assignments ORDER BY seq") == [(3,), (1,), (1,)]


# --- events -------------------------------------------------------------------------------------


def test_events_are_stored_as_they_were_sent(client: TestClient, app_run: Path) -> None:
    first = shown(client)
    batch = {
        "assignment_id": first["assignment_id"],
        "events": [
            {"ts": "2026-11-13T10:00:01.000Z", "type": "play", "payload": {}},
            {
                "ts": "2026-11-13T10:00:02.000Z",
                "type": "frames",
                "payload": {"visits": [[0, 120], [1, 80]], "mode": "sync"},
            },
            {"ts": "2026-11-13T10:00:03.000Z", "type": "tour_open", "payload": {"how": "first"}},
            {"ts": "later", "type": "a type of a later page", "payload": {"b": 1, "a": None}},
        ],
    }
    assert shaped(client.post("/api/events", json=batch), schemas.EventsOut) == {"stored": 4}
    found = rows(
        app_run,
        "SELECT assignment_id, ts, type, payload_json FROM events WHERE assignment_id = 1 "
        "ORDER BY event_id",
    )
    assert found == [
        (1, "2026-11-13T10:00:01.000Z", "play", "{}"),
        (1, "2026-11-13T10:00:02.000Z", "frames", '{"mode":"sync","visits":[[0,120],[1,80]]}'),
        (1, "2026-11-13T10:00:03.000Z", "tour_open", '{"how":"first"}'),
        (1, "later", "a type of a later page", '{"a":null,"b":1}'),
    ]
    empty = {"assignment_id": first["assignment_id"], "events": []}
    assert shaped(client.post("/api/events", json=empty), schemas.EventsOut) == {"stored": 0}
    # events of a rated assignment still arrive: the page sends its last ones when it leaves
    rate(client, first)
    assert client.post("/api/events", json=batch).json() == {"stored": 4}


def test_a_batch_the_shapes_refuse_stores_nothing(client: TestClient, app_run: Path) -> None:
    first = shown(client)
    before = n_rows(app_run, "events")

    def post(events: list[dict[str, Any]], assignment_id: int = first["assignment_id"]) -> Any:
        return client.post("/api/events", json={"assignment_id": assignment_id, "events": events})

    good = {"ts": "t", "type": "pause", "payload": {"frame": 3}}
    # a skip is the server's to write
    assert fields(post([good, {"ts": "t", "type": "skip", "payload": {}}]), 422) == [
        "events.1.type"
    ]
    assert fields(post([good] * 501), 422) == ["events"]
    large = {"ts": "t", "type": "frames", "payload": {"visits": "x" * 16_000}}
    assert fields(post([good, good, large]), 422) == ["events.2.payload"]
    assert fields(post([good], 999), 404) == ["assignment_id"]
    assert fields(client.post("/api/events", json={"events": [good]}), 422) == ["assignment_id"]
    # numbers that JSON has not: stored, the payload would be no JSON
    for number in ("NaN", "1e400", "-Infinity"):
        body = (
            f'{{"assignment_id": {first["assignment_id"]}, "events": [{json.dumps(good)}, '
            f'{{"ts": "t", "type": "frames", "payload": {{"visits": [[0, {number}]]}}}}]}}'
        )
        sent = client.post(
            "/api/events", content=body, headers={"Content-Type": "application/json"}
        )
        assert fields(sent, 422) == ["events.1.payload"], number
    assert n_rows(app_run, "events") == before
    assert post([good] * 500).json() == {"stored": 500}


def test_the_event_functions_write_rows_and_commit_nothing(app_engine: Any) -> None:
    with Session(app_engine) as session:
        rater = get_or_create_rater(session, "local")
        assignment = Assignment(
            rater_id=rater.rater_id, subject_uid="a" * 16, seq=1, render_key="k", assigned_at="t"
        )
        with pytest.raises(ValueError, match="no id yet"):
            store_events(session, assignment, [])
        session.add(assignment)
        session.flush()
        sent = [
            schemas.EventIn(ts="t1", type="zoom", payload={"zoom": 2, "at": [1.5, 2]}),
            schemas.EventIn(ts="t0", type="unknown"),
        ]
        assert store_events(session, assignment, sent) == 2
        assert store_events(session, assignment, []) == 0
        event = record_run_event(session, "config_change", {"setting": "blind"}, now="t9")
        assert (event.assignment_id, event.ts, event.type) == (None, "t9", "config_change")
        assert event.event_id is not None and event.payload_json == '{"setting":"blind"}'
        assert record_run_event(session, "note", {}).ts.endswith("Z")
        for wrong in ("", "skip"):
            with pytest.raises(ValueError):
                record_run_event(session, wrong, {})
        found = session.exec(select(Event).order_by(Event.event_id)).all()  # type: ignore[arg-type]
        assert [(e.assignment_id, e.ts, e.type, e.payload_json) for e in found[:2]] == [
            (assignment.assignment_id, "t1", "zoom", '{"at":[1.5,2],"zoom":2}'),
            (assignment.assignment_id, "t0", "unknown", "{}"),
        ]
        session.rollback()
    with Session(app_engine) as session:
        assert session.exec(select(Event)).all() == []


def test_a_setting_is_recorded_when_it_is_not_what_the_events_say(app_engine: Any) -> None:
    with Session(app_engine) as session:
        first = record_setting(session, "blind", False, now="t1")
        assert first is not None and json.loads(first.payload_json) == {
            "setting": "blind",
            "old": None,
            "new": False,
        }
        assert record_setting(session, "blind", False) is None
        assert record_setting(session, "require_marks", False) is not None
        changed = record_setting(session, "blind", True, now="t2")
        assert changed is not None and (changed.type, changed.ts) == (CONFIG_CHANGE, "t2")
        assert json.loads(changed.payload_json) == {"setting": "blind", "old": False, "new": True}
        assert record_setting(session, "blind", True) is None
        assert record_setting(session, "blind", False) is not None
        session.commit()


def test_the_server_records_the_settings_a_rating_is_made_under(app_run: Path) -> None:
    def settings() -> list[tuple[str, Any, Any]]:
        found = rows(
            app_run,
            "SELECT payload_json FROM events WHERE assignment_id IS NULL ORDER BY event_id",
        )
        said = [json.loads(text) for (text,) in found]
        return [(entry["setting"], entry["old"], entry["new"]) for entry in said]

    with serving(app_run):
        pass
    assert settings() == [
        ("blind", None, False),
        ("require_marks", None, False),
        ("mark_max_dist_mm", None, 5.0),
    ]
    with serving(app_run):
        pass
    assert len(settings()) == 3
    set_app(app_run, blind=True)
    with serving(app_run):
        pass
    assert settings()[3:] == [("blind", False, True)]


def test_the_export_says_under_which_queue_each_rating_was_made(app_run: Path) -> None:
    """The record the server makes at its start is what `ratings.blind` of the export reads."""
    with serving(app_run) as scored:
        first = shown(scored)
        rate(scored, first)
    set_app(app_run, blind=True)
    with serving(app_run) as shuffled:
        second = shown(shuffled)
        rate(shuffled, second)
    frame = ratings_frame(app_run).set_index("assignment_id")
    assert frame["blind"].to_dict() == {
        first["assignment_id"]: False,
        second["assignment_id"]: True,
    }
    assert frame["seq"].to_dict() == {first["assignment_id"]: 1, second["assignment_id"]: 2}


# --- files --------------------------------------------------------------------------------------


def test_media_are_the_tiles_and_the_poster_of_a_ready_render(
    client: TestClient, app_run: Path
) -> None:
    first = shown(client)
    uid, key = first["subject_uid"], first["render_key"]
    directory = media_dir(app_run, uid)
    for name in ("poster.webp", "tiles/sagittal_000.webp", "tiles/coronal_005_c.webp"):
        response = client.get(f"/media/{uid}/{name}?v={key}")
        assert response.status_code == 200, name
        assert response.content == (directory / name).read_bytes()
        assert response.headers["content-type"] == "image/webp"
        assert response.headers["cache-control"] == MEDIA_CACHE
    # without the key of the present render the name may hold other bytes tomorrow
    for query in ("", "?v=0000000000000000", f"?w={key}"):
        response = client.get(f"/media/{uid}/poster.webp{query}")
        assert (response.status_code, response.headers["cache-control"]) == (200, "no-cache")


@pytest.mark.parametrize(
    "name",
    [
        "render.json",
        "vertex_map.npz",
        "anim.mp4",  # a name of a render, and this render has no video
        "tiles/sagittal_999.webp",
        "tiles",
        "tiles/",
        "",
        "tiles/sagittal_000.webp.bak",
        "tiles/sagittal_0.webp",
        "tiles/sagittal_0000.webp",
        "tiles/sagittal_\u0660\u0660\u0660.webp",
        "tiles/SAGITTAL_000.webp",
        "tiles/oblique_000.webp",
        "tiles/%2e%2e/render.json",
        "..%2f..%2fcatalog.sqlite",
        "../../catalog.sqlite",
        "poster.webp/../render.json",
        "poster.webp%00.json",
        "/etc/passwd",
    ],
)
def test_media_serve_no_other_name(client: TestClient, app_run: Path, name: str) -> None:
    uid = shown(client)["subject_uid"]
    assert client.get(f"/media/{uid}/{name}").status_code == 404
    status, body = asgi_get(client.app, f"/media/{uid}/{name}")
    assert status == 404 and b"errors" in body


def test_a_name_with_dot_segments_is_no_name_of_a_render(client: TestClient) -> None:
    # as a server hands the path over when a client did not resolve the dots; the test
    # client would, and ask for the poster
    uid = shown(client)["subject_uid"]
    assert asgi_get(client.app, f"/media/{uid}/poster.webp")[0] == 200
    for name in ("tiles/../poster.webp", "./poster.webp", "tiles/./sagittal_000.webp"):
        status, body = asgi_get(client.app, f"/media/{uid}/{name}")
        assert status == 404 and b"errors" in body, name
    assert asgi_get(client.app, f"/media/{uid}/../{uid}/poster.webp")[0] == 404
    assert asgi_get(client.app, f"/media/./{uid}/poster.webp")[0] == 404


def test_media_of_no_subject_or_of_a_render_that_is_not_ready_are_not_served(
    client: TestClient, app_run: Path
) -> None:
    order = queue(app_run)
    for uid in ("0" * 16, order[0].upper(), order[0][:15], order[0] + "0", "..", "tiles"):
        assert client.get(f"/media/{uid}/poster.webp").status_code == 404, uid
    # a render of another configuration, a failed one, a subject without a render row
    in_catalog(
        app_run, "UPDATE renders SET config_hash = 'dff40fe66869' WHERE subject_uid = ?", order[0]
    )
    in_catalog(app_run, "UPDATE renders SET status = 'failed' WHERE subject_uid = ?", order[1])
    in_catalog(app_run, "DELETE FROM renders WHERE subject_uid = ?", order[2])
    for uid in order:
        assert (media_dir(app_run, uid) / "poster.webp").is_file()
        assert fields(client.get(f"/media/{uid}/poster.webp"), 404) == ["path"]


def test_a_link_in_the_media_directory_leads_nowhere_else(
    client: TestClient, app_run: Path, tmp_path: Path
) -> None:
    first, second = queue(app_run)[:2]
    directory = media_dir(app_run, first)
    outside = tmp_path / "secret.webp"
    outside.write_bytes(b"not of the run")

    def link(name: str, target: Path) -> None:
        (directory / name).unlink(missing_ok=True)
        (directory / name).symlink_to(target)

    link("tiles/sagittal_000.webp", app_run / CATALOG_FILE)
    link("tiles/sagittal_001.webp", directory / RENDER_JSON)
    link("tiles/sagittal_002.webp", directory / "vertex_map.npz")
    link("tiles/sagittal_003.webp", outside)
    link("tiles/sagittal_004.webp", media_dir(app_run, second) / "poster.webp")
    link("poster.webp", app_run / "config.yaml")
    for name in (*(f"tiles/sagittal_00{n}.webp" for n in range(5)), "poster.webp"):
        assert fields(client.get(f"/media/{first}/{name}"), 404) == ["path"], name
    # a link to another tile of the same render is a tile of that render
    link("tiles/sagittal_005.webp", directory / "tiles" / "sagittal_006.webp")
    response = client.get(f"/media/{first}/tiles/sagittal_005.webp")
    assert response.content == (directory / "tiles" / "sagittal_006.webp").read_bytes()
    # a subject directory that is a link is not followed out of the media directory
    elsewhere = tmp_path / "elsewhere"
    media_dir(app_run, second).rename(elsewhere)
    media_dir(app_run, second).symlink_to(elsewhere, target_is_directory=True)
    assert (media_dir(app_run, second) / "poster.webp").is_file()
    assert fields(client.get(f"/media/{second}/poster.webp"), 404) == ["path"]


def test_the_media_directory_of_a_run_may_be_a_link(app_run: Path, tmp_path: Path) -> None:
    (app_run / "media").rename(tmp_path / "disk")
    (app_run / "media").symlink_to(tmp_path / "disk", target_is_directory=True)
    with serving(app_run) as client:
        first = shown(client)
        response = client.get(f"/media/{first['subject_uid']}/poster.webp")
        assert response.status_code == 200 and response.content[:4] == b"RIFF"


def test_media_are_not_served_by_a_run_whose_media_lie_elsewhere(app_run: Path) -> None:
    set_app(app_run, media_base_url="https://media.example.org/run-7/")
    with serving(app_run) as client:
        first = shown(client)
        uid = first["subject_uid"]
        assert first["media_base"] == f"https://media.example.org/run-7/{uid}/"
        assert fields(client.get(f"/media/{uid}/poster.webp"), 404) == ["path"]
        policy = client.get("/").headers["content-security-policy"]
        assert "img-src 'self' https://media.example.org data:" in policy


def test_static_files_are_those_of_the_package_and_nothing_beside_them(client: TestClient) -> None:
    for name, kind in (
        ("app.css", "text/css"),
        ("js/rate.js", "javascript"),
        ("js/tour.js", "javascript"),
    ):
        response = client.get(f"/static/{name}?v=0.0.1")
        assert response.status_code == 200 and kind in response.headers["content-type"]
        assert response.content == (server.STATIC_DIR / name).read_bytes()
        assert response.headers["cache-control"] == "no-cache"
    for path in (
        "/static/",
        "/static/js",
        "/static/js/",
        "/static/absent.js",
        "/static/../server.py",
        "/static/js/../../server.py",
        "/static/%2e%2e/server.py",
        "/static/..%2fserver.py",
        "/static/js/..%2f..%2f..%2fprotocol.py",
        "/static//etc/passwd",
        "/static/%2fetc%2fpasswd",
        "/static/app.css%00",
        "/static/../templates/rate.html",
    ):
        assert client.get(path).status_code == 404, path
        status, body = asgi_get(client.app, path)
        assert status == 404 and b"errors" in body, path
    # the same paths, as they are, find the files when they are meant
    assert (server.STATIC_DIR / "../server.py").resolve().is_file()


def test_a_link_or_a_hidden_file_in_the_static_directory_is_not_served(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    static = tmp_path / "static"
    (static / "js").mkdir(parents=True)
    (static / "app.css").write_text("body {}")
    (static / ".secret").write_text("hidden")
    (static / ".git").mkdir()
    (static / ".git" / "config.js").write_text("hidden")
    (tmp_path / "outside.js").write_text("outside")
    (static / "js" / "linked.js").symlink_to(tmp_path / "outside.js")
    (static / "js" / "inner.js").symlink_to(static / "app.css")
    monkeypatch.setattr(server, "STATIC_DIR", static)
    assert client.get("/static/app.css").text == "body {}"
    assert client.get("/static/js/inner.js").text == "body {}"
    for name in (".secret", ".git/config.js", "js/linked.js"):
        assert fields(client.get(f"/static/{name}"), 404) == ["path"], name


@pytest.mark.parametrize(
    "path",
    [
        "/catalog.sqlite",
        "/config.yaml",
        "/app.sqlite",
        "/protocol.yaml",
        "/media/catalog.sqlite",
        "/media/../catalog.sqlite",
        "/media/../config.yaml",
        "/media/../app.sqlite",
        "/static/../../../../config.yaml",
        "/tables/scores.parquet",
        "/media/../tables/scores.parquet",
    ],
)
def test_the_files_of_the_run_directory_are_not_served(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 404
    status, _body = asgi_get(client.app, path)
    assert status == 404


def raw_status(port: int, path: str, headers: dict[str, str] | None = None) -> int:
    """The status of a GET whose path and headers go over the socket as they are given."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.putrequest("GET", path, skip_host=True, skip_accept_encoding=True)
        sent = {"Host": f"127.0.0.1:{port}"} if headers is None else headers
        for name, value in sent.items():
            connection.putheader(name, value)
        connection.endheaders()
        response = connection.getresponse()
        response.read()
        return response.status
    finally:
        connection.close()


def test_a_path_that_climbs_is_a_404_however_it_is_written_on_the_socket(app_run: Path) -> None:
    # the HTTP server decodes %2e and %2f before the app sees the path, which neither the
    # test client nor a raw ASGI call shows
    uid = queue(app_run)[0]
    directory = media_dir(app_run, uid)
    climbing = {
        f"/media/{uid}/tiles/../{RENDER_JSON}": directory / RENDER_JSON,
        f"/media/{uid}/tiles/%2e%2e/{RENDER_JSON}": directory / RENDER_JSON,
        f"/media/{uid}/tiles/..%2f{RENDER_JSON}": directory / RENDER_JSON,
        f"/media/{uid}/tiles/%2e%2e/poster.webp": directory / "poster.webp",
        f"/media/{uid}/../../{CATALOG_FILE}": app_run / CATALOG_FILE,
        f"/media/{uid}/%2e%2e/%2e%2e/{CATALOG_FILE}": app_run / CATALOG_FILE,
        f"/media/{uid}/..%2f..%2fconfig.yaml": app_run / "config.yaml",
        f"/media/%2e%2e/{CATALOG_FILE}": app_run / CATALOG_FILE,
        "/static/../server.py": server.STATIC_DIR.parent / "server.py",
        "/static/%2e%2e/server.py": server.STATIC_DIR.parent / "server.py",
        "/static/..%2fserver.py": server.STATIC_DIR.parent / "server.py",
        "/static/js/%2e%2e/%2e%2e/templates/rate.html": server.TEMPLATES_DIR / "rate.html",
    }
    handle = start_server(app_run, port=0)
    try:
        assert raw_status(handle.port, f"/media/{uid}/poster.webp") == 200
        assert raw_status(handle.port, "/static/js/rate.js") == 200
        for path, target in climbing.items():
            # each names a file that is there: the 404 is the server's doing
            assert target.is_file(), path
            assert raw_status(handle.port, path) == 404, path
        for path in ("//etc/passwd", "/static//etc/passwd", "/static/%2fetc%2fpasswd", "/media/"):
            assert raw_status(handle.port, path) == 404, path
        # a request without a Host header is no HTTP/1.1 request: it does not reach the app,
        # which would refuse it as well (the raw ASGI call in the test of the hosts)
        assert raw_status(handle.port, "/healthz", {}) == 400
        assert raw_status(handle.port, "/healthz", {"Host": "evil.example"}) == 403
        assert raw_status(handle.port, "/healthz") == 200
    finally:
        handle.stop()


# --- who may ask --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host",
    [
        "evil.example",
        "evil.example:8765",
        "127.0.0.1.evil.example",
        "localhost.evil.example",
        "127.0.0.2",
        "0.0.0.0",
        "[::1]",
        "127.0.0.1:80:80",
        "127.0.0.1:port",
        "127.0.0.1@evil.example",
        "",
        " ",
    ],
)
def test_a_request_for_another_host_is_refused_on_every_route(
    client: TestClient, app_run: Path, host: str
) -> None:
    uid = queue(app_run)[0]
    for path in ("/", "/healthz", "/api/session", "/api/next", "/static/app.css", "/nowhere"):
        assert fields(client.get(path, headers={"Host": host}), 403) == ["request"], path
    assert client.get(f"/media/{uid}/poster.webp", headers={"Host": host}).status_code == 403
    response = client.post("/api/events", json={}, headers={"Host": host})
    assert response.status_code == 403
    assert n_rows(app_run, "assignments") == 0


def test_the_local_hosts_are_answered_whatever_the_port(client: TestClient) -> None:
    assert LOCAL_HOSTS == ("127.0.0.1", "localhost")
    for host in ("127.0.0.1", "127.0.0.1:8765", "localhost", "localhost:41007", "LOCALHOST:1"):
        assert client.get("/healthz", headers={"Host": host}).status_code == 200, host
    # two Host headers, or none: which host would a proxy have meant?
    two = [("host", "127.0.0.1"), ("host", "evil.example")]
    assert asgi_get(client.app, "/healthz", two)[0] == 403
    assert (
        asgi_get(client.app, "/healthz", [("host", "127.0.0.1"), ("host", "127.0.0.1")])[0] == 403
    )
    assert asgi_get(client.app, "/healthz", [("accept", "*/*")])[0] == 403
    assert asgi_get(client.app, "/healthz", [("host", "127.0.0.1")])[0] == 200


def test_the_hosts_a_server_answers_for_can_be_named(app_run: Path) -> None:
    with serving(app_run, allowed_hosts=["Rater.Lab.example"]) as client:
        assert client.get("/healthz", headers={"Host": "rater.lab.example:443"}).status_code == 200
        assert client.get("/healthz").status_code == 403
    with pytest.raises(ConfigError, match="at least one host"):
        create_app(app_run, allowed_hosts=[])


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "http://evil.example"},
        {"Origin": "http://127.0.0.1:3000"},  # another program on this machine
        {"Origin": "http://localhost"},  # another name is another origin
        {"Origin": "null"},
        {"Origin": "https://127.0.0.1.evil.example"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Sec-Fetch-Site": "same-site"},
        {"Origin": "http://127.0.0.1", "Sec-Fetch-Site": "cross-site"},
    ],
)
def test_a_request_of_another_origin_changes_and_reads_nothing(
    client: TestClient, app_run: Path, headers: dict[str, str]
) -> None:
    first = shown(client)
    before = (stored(app_run), n_rows(app_run, "assignments"))
    posts = {
        "/api/rating": rating(first),
        "/api/skip": skip_body(first),
        "/api/events": {"assignment_id": 1, "events": [{"ts": "t", "type": "play"}]},
    }
    for path, body in posts.items():
        assert fields(client.post(path, json=body, headers=headers), 403) == ["request"], path
    for path in ("/api/session", "/api/next", "/api/back", "/api/review", "/api/export"):
        assert fields(client.get(path, headers=headers), 403) == ["request"], path
    assert (stored(app_run), n_rows(app_run, "assignments")) == before
    assert rows(app_run, "SELECT skip_count FROM assignments") == [(0,)]
    # a link from elsewhere may open a page; what the page then asks for is its own
    assert client.get("/", headers=headers).status_code == 200


def test_the_pages_own_requests_pass(client: TestClient, app_run: Path) -> None:
    own = {"Origin": "http://127.0.0.1", "Sec-Fetch-Site": "same-origin"}
    first = shaped(
        client.get("/api/next", headers={"Sec-Fetch-Site": "same-origin"}), schemas.NextOut
    )
    response = client.post("/api/rating", json=rating(first["assignment"]), headers=own)
    assert response.status_code == 200
    # typed into the address bar
    assert client.get("/api/export", headers={"Sec-Fetch-Site": "none"}).status_code == 200
    with_port = {"Host": "localhost:8765", "Origin": "http://localhost:8765"}
    events = {"assignment_id": 1, "events": []}
    assert client.post("/api/events", json=events, headers=with_port).status_code == 200
    assert fields(
        client.post(
            "/api/events", json=events, headers=with_port | {"Origin": "http://localhost:8766"}
        ),
        403,
    ) == ["request"]


def test_a_post_is_json_of_a_bounded_size(client: TestClient, app_run: Path) -> None:
    first = shown(client)
    text = json.dumps(skip_body(first))
    # a form of another site can post text without asking; it cannot post JSON
    for kind in ("text/plain", "application/x-www-form-urlencoded", "multipart/form-data", ""):
        response = client.post("/api/skip", content=text, headers={"Content-Type": kind})
        assert fields(response, 415) == ["request"], kind
    assert rows(app_run, "SELECT skip_count FROM assignments") == [(0,)]
    response = client.post(
        "/api/skip", content=text, headers={"Content-Type": "application/json; charset=utf-8"}
    )
    assert response.status_code == 200
    large = json.dumps({"assignment_id": 1, "events": [], "pad": "x" * server.MAX_BODY_BYTES})
    response = client.post(
        "/api/events", content=large, headers={"Content-Type": "application/json"}
    )
    assert fields(response, 413) == ["request"]

    def unannounced() -> Iterator[bytes]:
        yield b'{"assignment_id": 1, "events": [], "pad": "'
        for _ in range(server.MAX_BODY_BYTES // 65_536 + 1):
            yield b"x" * 65_536
        yield b'"}'

    # a body that does not say how long it is is counted as it arrives
    before = n_rows(app_run, "events")
    response = client.post(
        "/api/events", content=unannounced(), headers={"Content-Type": "application/json"}
    )
    assert "content-length" not in response.request.headers
    assert fields(response, 413) == ["request"]
    assert n_rows(app_run, "events") == before
    for wrong in ("", "{", "[]", '"text"', "null"):
        response = client.post(
            "/api/rating", content=wrong, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422, wrong
        schemas.ErrorsOut.model_validate(response.json())


def test_every_answer_carries_the_headers_that_keep_other_origins_out(
    client: TestClient, app_run: Path
) -> None:
    uid = queue(app_run)[0]
    answers = [
        client.get("/"),
        client.get("/static/app.css"),
        client.get(f"/media/{uid}/poster.webp"),
        client.get("/api/session"),
        client.get("/api/export"),
        client.get("/nowhere"),
        client.get("/healthz", headers={"Host": "evil.example"}),
        client.post("/api/rating", json={}),
        client.post("/api/skip", content="x", headers={"Content-Type": "text/plain"}),
    ]
    assert [response.status_code for response in answers] == [
        200,
        200,
        200,
        200,
        200,
        404,
        403,
        422,
        415,
    ]
    for response in answers:
        for name in SECURITY_HEADERS:
            assert name in response.headers, (response.url, name)
        assert response.headers["x-frame-options"] == "DENY"
        assert response.headers["cross-origin-resource-policy"] == "same-origin"
        assert "access-control-allow-origin" not in response.headers
    policy = dict(
        part.strip().split(" ", 1)
        for part in answers[0].headers["content-security-policy"].split(";")
    )
    # nothing from another server, no inline script, no frame around the page
    assert policy["default-src"] == "'self'" and policy["frame-ancestors"] == "'none'"
    assert policy["img-src"] == "'self' data:" and "script-src" not in policy
    assert "unsafe" not in answers[0].headers["content-security-policy"]


# --- the unblinded routes, and when they open ---------------------------------------------------


def unblinded_paths(run_dir: Path) -> list[str]:
    uid = queue(run_dir)[0]
    return [*UNBLINDED, f"/api/review/{uid}"]


def test_the_unblinded_routes_are_those_of_the_schemas() -> None:
    marked = {route.path for route in schemas.ROUTES if route.unblinded}
    assert marked == {*UNBLINDED, "/api/review/{subject_uid}"}


def test_a_researcher_alone_with_their_own_data_can_look_at_any_time(
    client: TestClient, app_run: Path
) -> None:
    shown(client)
    for path in unblinded_paths(app_run):
        assert client.get(path).status_code == 200, path


def test_a_blind_run_opens_the_unblinded_routes_only_when_nothing_is_left(app_run: Path) -> None:
    set_app(app_run, blind=True)
    with serving(app_run) as client:
        paths = unblinded_paths(app_run)
        for path in paths:
            assert fields(client.get(path), 403) == ["request"], path
        first = shown(client)
        rate(client, first)
        skipped = shown(client)
        rate(client, skip(client, skipped)["assignment"])
        # one subject is left, and it is a skipped one: still closed
        assert get_next(client)["assignment"] == skipped
        for path in paths:
            assert fields(client.get(path), 403) == ["request"], path
        rate(client, skipped)
        for path in paths:
            assert client.get(path).status_code == 200, path
        # the rating routes were open all along
        assert client.get("/").status_code == 200


def test_a_browser_that_opens_a_closed_review_gets_a_page_that_says_why(app_run: Path) -> None:
    """The refusal of ``/review`` is the ``ErrorsOut`` of every route; a request that asks
    for a page, as a browser's address bar does, gets the same words as a page."""
    set_app(app_run, blind=True)
    with serving(app_run) as client:
        page = client.get("/review", headers={"Accept": "text/html,application/xhtml+xml"})
        assert page.status_code == 403
        assert page.headers["content-type"].startswith("text/html")
        assert page.headers["cache-control"] == "no-store"
        assert "you still have subjects to rate; the review opens when none is left" in page.text
        assert '<a href="/">Back to the rating screen</a>' in page.text
        # the page of the review itself, its scripts and anything of a subject are not in it
        assert "<script" not in page.text and "review.js" not in page.text
        for uid in queue(app_run):
            assert uid not in page.text
        # the routes of the API answer a browser as they answer a script
        api = client.get("/api/review", headers={"Accept": "text/html"})
        assert fields(api, 403) == ["request"]
        rate_all(client)
        opened = client.get("/review", headers={"Accept": "text/html"})
        assert opened.status_code == 200 and "review.js" in opened.text


def test_a_blind_run_stays_closed_while_a_render_is_awaited(app_run: Path) -> None:
    set_app(app_run, blind=True)
    last = queue(app_run)[-1]
    in_catalog(app_run, "DELETE FROM renders WHERE subject_uid = ?", last)
    with serving(app_run) as client:
        rate(client, shown(client))
        rate(client, shown(client))
        assert get_next(client)["status"] == "rendering"
        assert fields(client.get("/api/review"), 403) == ["request"]


def test_with_two_raters_each_sees_the_review_only_after_their_own_ratings(app_run: Path) -> None:
    with serving(app_run, rater="Ada") as ada:
        assert ada.get("/api/review").status_code == 200  # alone so far
        assert rate_all(ada) == 3
        assert n_rows(app_run, "final_labels") == 9
    with serving(app_run, rater="Grace") as grace:
        paths = unblinded_paths(app_run)
        for path in paths:
            assert fields(grace.get(path), 403) == ["request"], path
        # each rater walks the whole queue on their own
        assert (
            get_next(grace)["progress"] | {"remaining": 3, "done_total": 0}
            == get_next(grace)["progress"]
        )
        assert rate_all(grace) == 3
        for path in paths:
            assert grace.get(path).status_code == 200, path
    # with a second rater the labels are for `surfcensus finalize` to write
    assert n_rows(app_run, "final_labels") == 0
    assert rows(app_run, "SELECT name, role FROM raters ORDER BY rater_id") == [
        ("Ada", "lead"),
        ("Grace", "lead"),
    ]
    with serving(app_run, rater="Ada") as ada:
        assert get_next(ada)["status"] == "done" and ada.get("/api/stats").status_code == 200


def test_an_app_opened_without_a_name_beside_other_raters_says_that_it_made_a_rater(
    app_run: Path, caplog: pytest.LogCaptureFixture
) -> None:
    def warned(**options: Any) -> list[str]:
        caplog.clear()
        with caplog.at_level("WARNING", logger="surfcensus"), serving(app_run, **options):
            pass
        return [record.getMessage() for record in caplog.records if "rater" in record.msg]

    # the first rater of a run, named or not, and a second one who is named: nothing to say
    assert warned(rater="Grace") == []
    assert warned(rater="Ada") == []
    # opened without a name, the app rates as `local`: a third rater, who has every subject
    # left, so the review is closed to them and `finalize` lists them from now on
    (message,) = warned()
    assert message == (
        "no rater was named, so the app rates as a new rater, local, beside Ada, Grace: "
        "stop it and name the rater (--rater NAME) to go on as one of them"
    )
    assert [name for (name,) in rows(app_run, "SELECT name FROM raters ORDER BY name")] == [
        "Ada",
        "Grace",
        "local",
    ]
    # the rater is there now: opening it again is no news
    assert warned() == []


def test_a_run_whose_first_rater_has_no_name_is_opened_without_a_warning(
    app_run: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING", logger="surfcensus"), serving(app_run):
        pass
    assert [record for record in caplog.records if "rater" in record.msg] == []


# --- review, subject, statistics, export --------------------------------------------------------


def test_the_review_table_has_a_row_per_subject_in_the_order_of_the_queue(
    client: TestClient, app_run: Path, small_run_uids: dict[str, str]
) -> None:
    rate(client, shown(client), values={"A": 2, "B": 0, "C": 1})
    body = shaped(client.get("/api/review"), schemas.ReviewOut)
    columns = body["columns"]
    assert columns[:6] == [
        "subject_uid",
        "subject_label",
        "dataset",
        "fs_version",
        "render_status",
        "render_reason",
    ]
    assert columns[-7:] == [
        "final_A",
        "final_B",
        "final_C",
        "final_method",
        "unratable_reason",
        "n_ratings",
        "n_marks",
    ]
    scores = [name for name in SCHEMAS["scores"].names if name not in ("subject_uid", "dataset")]
    assert columns[6:-7] == scores and len(set(columns)) == len(columns)
    assert body["total"] == 3 and [list(row) for row in body["rows"]] == [columns] * 3
    order = queue(app_run)
    assert [row["subject_uid"] for row in body["rows"]] == order
    assert [row["primary_rank"] for row in body["rows"]] == [1, 2, 3]
    first = body["rows"][0]
    assert first["subject_label"] == "sub-001" and first["render_status"] == "done"
    assert (first["final_A"], first["final_B"], first["final_C"]) == (2, 0, 1)
    assert (first["final_method"], first["n_ratings"], first["n_marks"]) == ("single", 1, 0)
    assert body["rows"][1]["final_A"] is None and body["rows"][1]["n_ratings"] == 0
    assert isinstance(first["composite"], float) and first["primary"] is True


@pytest.mark.parametrize("edit", ["reordered", "removed"])
def test_an_unratable_label_keeps_its_reason_when_the_protocol_is_edited(
    app_run: Path,
    shipped_protocol: Protocol,
    write_reasons: Callable[[Path, Sequence[str]], None],
    edit: str,
) -> None:
    def reason_shown(client: TestClient, uid: str) -> object:
        body = shaped(client.get("/api/review"), schemas.ReviewOut)
        return next(row for row in body["rows"] if row["subject_uid"] == uid)["unratable_reason"]

    reasons = list(shipped_protocol.unratable_reasons)
    with serving(app_run) as client:
        uid = shown(client)["subject_uid"]
        rate(client, shown(client), values={}, unratable="image too poor to judge")
        assert rows(app_run, "SELECT value FROM final_labels") == [(3,)]
        # the file is edited while the server runs, and only the order of its list, or its
        # length, changes: the label is read with the list it was written with
        write_reasons(app_run, list(reversed(reasons)) if edit == "reordered" else reasons[:3])
        assert reason_shown(client, uid) == "image too poor to judge"
    # a server that starts with the edited protocol moves the label to where the reason is
    # now; a reason that is gone stays behind the reasons of the protocol
    with serving(app_run) as client:
        assert reason_shown(client, uid) == "image too poor to judge"
    assert rows(app_run, "SELECT value FROM final_labels") == [(0 if edit == "reordered" else 3,)]
    (kept,) = rows(app_run, "SELECT value FROM app_meta WHERE key = ?", LABEL_REASONS_KEY)
    assert json.loads(kept[0]) == (list(reversed(reasons)) if edit == "reordered" else reasons)


def test_the_review_table_follows_the_ranks_not_the_uids(app_run: Path) -> None:
    # the uids of the fixture follow from a random salt: in one session of six their order
    # is that of the ranks. Here the ranks run against the uids, so the two always differ.
    path = table_path(app_run, "scores")
    scores = read_table(path, SCHEMAS["scores"])
    against = sorted(scores["subject_uid"], reverse=True)
    assert scores["primary"].all() and len(against) == 3
    scores["primary_rank"] = [against.index(uid) + 1 for uid in scores["subject_uid"]]
    write_table(path, scores, SCHEMAS["scores"], sort_by=KEYS["scores"])
    with serving(app_run) as client:
        body = shaped(client.get("/api/review"), schemas.ReviewOut)
        assert [row["subject_uid"] for row in body["rows"]] == against
        tied = shaped(client.get("/api/review", params={"sort": "fs_version"}), schemas.ReviewOut)
        assert [row["subject_uid"] for row in tied["rows"]] == against
        # the queue is in the same order
        assert shown(client)["subject_uid"] == against[0]


def test_the_review_table_sorts_filters_and_pages(client: TestClient, app_run: Path) -> None:
    rate(client, shown(client), values={"A": 2, "B": 0, "C": 0})

    def table(**query: Any) -> dict[str, Any]:
        return shaped(client.get("/api/review", params=query), schemas.ReviewOut)

    labels = lambda body: [row["subject_label"] for row in body["rows"]]  # noqa: E731
    assert labels(table(sort="subject_label")) == ["sub-000", "sub-001", "sub-002"]
    assert labels(table(sort="subject_label", order="desc")) == ["sub-002", "sub-001", "sub-000"]
    by_score = table(sort="composite", order="desc")
    assert [row["composite"] for row in by_score["rows"]] == sorted(
        (row["composite"] for row in by_score["rows"]), reverse=True
    )
    # a subject without a value comes last in either direction
    assert labels(table(sort="final_A"))[0] == "sub-001"
    assert labels(table(sort="final_A", order="desc"))[0] == "sub-001"
    # rows that are equal in the column keep the order of the queue
    assert labels(table(sort="render_status")) == labels(table())
    assert (
        labels(table(filter="final_A:2")) == ["sub-001"] and table(filter="final_A:2")["total"] == 1
    )
    assert labels(table(filter="final_A:2.0")) == ["sub-001"]
    assert table(filter="final_A:1")["rows"] == []
    assert table(filter="final_A:")["total"] == 2  # the subjects without a label
    assert table(filter="render_status:done,subject_label:sub-002")["total"] == 1
    assert table(filter="primary:true")["total"] == 3
    paged = table(sort="subject_label", offset=1, limit=1)
    assert labels(paged) == ["sub-001"] and paged["total"] == 3
    assert table(offset=5)["rows"] == []
    for query, where in (
        ({"sort": "accession"}, "sort"),
        ({"sort": "age"}, "sort"),  # a column of the subjects, not of this table
        ({"filter": "nothing:1"}, "filter"),
        ({"filter": "final_A"}, "filter"),
        ({"order": "up"}, "order"),
        ({"limit": 0}, "limit"),
        ({"limit": 1001}, "limit"),
        ({"offset": -1}, "offset"),
        ({"page": 2}, "page"),
    ):
        assert fields(client.get("/api/review", params=query), 422) == [where], query


def test_a_subject_of_the_review_has_its_render_scores_ratings_marks_and_clusters(
    client: TestClient, app_run: Path, small_run_uids: dict[str, str]
) -> None:
    first = shown(client)
    uid = first["subject_uid"]
    posted, expected = mark_on_surface(app_run, uid)
    saved = rate(client, first, values={"A": 2, "B": 0, "C": 0}, marks=[posted], note="here")
    body = shaped(client.get(f"/api/review/{uid}"), schemas.SubjectOut)
    assert (body["subject_uid"], body["media_base"]) == (uid, f"/media/{uid}/")
    assert body["render_key"] == first["render_key"] and body["render"] == first["render"]
    assert body["scores"]["subject_uid"] == uid and body["scores"]["primary_rank"] == 1
    assert set(body["scores"]) == set(SCHEMAS["scores"].names)
    (rated,) = body["ratings"]
    assert (
        rated
        | {
            "assignment_id": 1,
            "rater": "local",
            "kind": "main",
            "render_current": True,
            "A": 2,
            "B": 0,
            "C": 0,
            "unratable_reason": None,
            "other": False,
            "note": "here",
            "revision": 1,
            "ms_on_subject": 1200,
        }
        == rated
    )
    (seen,) = body["marks"]
    assert seen | saved["saved"]["marks"][0] | {"client_id": "m1"} == seen | {"client_id": "m1"}
    assert (seen["class"], seen["panel"], seen["slice"], seen["rater"]) == (
        "A",
        posted["panel"],
        posted["slice"],
        "local",
    )
    assert seen["tkr"] == pytest.approx(expected["tkr"], abs=1e-9)
    # sub-001 carries the patch: its clusters with a native position are listed, no other
    clusters = clusters_with_native(app_run)
    own = clusters[(clusters["subject_uid"] == uid) & clusters["tkr_x"].notna()]
    assert uid == small_run_uids["sub-001"] and 0 < len(own) < int(clusters["tkr_x"].notna().sum())
    assert len(own) < int((clusters["subject_uid"] == uid).sum())
    assert [
        (found["reference_id"], found["hemi"], found["region"], found["rank"])
        for found in body["clusters"]
    ] == list(zip(own["reference_id"], own["hemi"], own["region"], own["rank"], strict=True))
    for cluster, tkr_x, peak_z in zip(body["clusters"], own["tkr_x"], own["peak_z"], strict=True):
        assert cluster["tkr"][0] == pytest.approx(tkr_x) and cluster["peak_z"] > 3
        assert cluster["peak_z"] == pytest.approx(peak_z)
    # a subject nobody rated; a subject without a ready render; no subject
    other = shaped(client.get(f"/api/review/{small_run_uids['sub-000']}"), schemas.SubjectOut)
    assert other["ratings"] == [] and other["marks"] == [] and other["render"] is not None
    in_catalog(app_run, "DELETE FROM renders WHERE subject_uid = ?", small_run_uids["sub-000"])
    other = shaped(client.get(f"/api/review/{small_run_uids['sub-000']}"), schemas.SubjectOut)
    assert (other["media_base"], other["render_key"], other["render"]) == (None, None, None)
    assert other["scores"] is not None
    for unknown in ("0" * 16, "sub-001", "x"):
        assert fields(client.get(f"/api/review/{unknown}"), 404) == ["subject_uid"]


def test_statistics_count_what_each_rater_completed(app_run: Path) -> None:
    with serving(app_run, rater="Ada") as ada:
        empty = shaped(ada.get("/api/stats"), schemas.StatsOut)
        assert empty == {
            "raters": [{"rater": "Ada", "n_completed": 0, "median_s": None, "by_day": []}],
            "agreement": None,
            "drift": None,
        }
        for ms in (1000, 4000, 2000):
            rate(ada, shown(ada), ms_on_subject=ms)
        # a revision is the same assignment, with the time of its latest visit
        back = shaped(ada.get("/api/back"), schemas.NextOut)["assignment"]
        rate(ada, back, ms_on_subject=3000)
        body = shaped(ada.get("/api/stats"), schemas.StatsOut)
    (stats,) = body["raters"]
    assert stats["n_completed"] == 3 and stats["median_s"] == 3.0
    assert sum(day["n"] for day in stats["by_day"]) == 3
    # days are those of UTC; the test may run across midnight
    recent = {time.strftime("%Y-%m-%d", time.gmtime(time.time() - back)) for back in (0, 600)}
    assert {day["day"] for day in stats["by_day"]} <= recent


def test_the_export_is_the_ratings_table_as_csv(client: TestClient, app_run: Path) -> None:
    rate(client, shown(client), values={"A": 1, "B": 0, "C": 0}, note='a "quoted", note')
    response = client.get("/api/export")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "ratings.csv" in response.headers["content-disposition"]
    assert response.text == frame_csv(ratings_frame(app_run))
    frame = pd.read_csv(io.StringIO(response.text))
    assert (
        len(frame) == 1 and frame["note"][0] == 'a "quoted", note' and frame["rater"][0] == "local"
    )


# --- blindness ----------------------------------------------------------------------------------


def keys_and_strings(value: Any, keys: set[str], strings: list[str]) -> None:
    if isinstance(value, dict):
        for key, inner in value.items():
            keys.add(str(key))
            keys_and_strings(inner, keys, strings)
    elif isinstance(value, list):
        for inner in value:
            keys_and_strings(inner, keys, strings)
    elif isinstance(value, str):
        strings.append(value)


def test_nothing_a_rating_page_receives_names_a_dataset_a_label_or_a_score(
    app_run: Path, small_run_uids: dict[str, str]
) -> None:
    set_app(app_run, blind=True, repeat_rate=0.5, repeat_gap=0)
    answers: list[Any] = []
    with serving(app_run) as client:

        def ask(method: str, path: str, **kw: Any) -> Any:
            response = client.request(method, path, **kw)
            answers.append(response.json())
            return response

        ask("GET", "/api/session")
        first = ask("GET", "/api/next").json()["assignment"]
        posted, _expected = mark_on_surface(app_run, first["subject_uid"])
        ask(
            "POST",
            "/api/events",
            json={"assignment_id": 1, "events": [{"ts": "t", "type": "play"}]},
        )
        ask("POST", "/api/rating", json=rating(first, values={"A": 9}))
        ask("POST", "/api/rating", json=rating(first, marks=[mark(slice=99)]))
        ask(
            "POST",
            "/api/rating",
            json=rating(first, values={"A": 1, "B": 0, "C": 0}, marks=[posted]),
        )
        ask("GET", "/api/back")
        second = ask("GET", "/api/next").json()["assignment"]
        ask("POST", "/api/skip", json=skip_body(second))
        ask("POST", "/api/skip", json=skip_body(first))
        ask("POST", "/api/rating", json=rating(999))
        ask("GET", "/api/back")
        ask("GET", "/api/review")  # closed: its refusal is read by the rating page too
        for _ in range(8):
            body = ask("GET", "/api/next").json()
            if body["status"] != "assignment":
                break
            ask("POST", "/api/rating", json=rating(body["assignment"], unratable="render failure"))
        ask("GET", "/api/next")
    keys: set[str] = set()
    strings: list[str] = []
    keys_and_strings(answers, keys, strings)
    config = load_config(app_run)
    with PublicCatalog.open(app_run) as public:
        subjects = public.subjects()
        datasets = set(public.datasets()["pseudonym"])
    score_columns = set(SCHEMAS["scores"].names) - {"subject_uid"}
    subject_columns = set(subjects.columns) - {"subject_uid"}
    assert keys & score_columns == set()
    assert keys & subject_columns == set()
    assert keys & {"inputs", "template", "scores", "rater_id", "detail", "sha256"} == set()
    text = "\n".join(strings)
    for label in small_run_uids:
        assert label not in text
    for private in (
        *datasets,
        str(app_run),
        str(config.sources[0].root),
        "sub-0",
        "D-",
        *SCORE_FLAGS,
    ):
        assert private and private not in text, private
    # no digest of a source file: nothing of 64 hex digits but the protocol's own
    digests = {word for word in text.replace('"', " ").split() if len(word) == 64}
    assert digests <= {hashlib.sha256((app_run / PROTOCOL_FILE).read_bytes()).hexdigest()}


def test_a_second_rater_is_shown_nothing_of_what_the_first_one_rated(app_run: Path) -> None:
    with serving(app_run, rater="Ottoline") as first_rater:
        first = shown(first_rater)
        posted, _expected = mark_on_surface(app_run, first["subject_uid"])
        rate(
            first_rater,
            first,
            values={"A": 2, "B": 1, "C": 0},
            marks=[posted],
            note="nota bene qzx",
        )
        assert rate_all(first_rater) == 2
    with serving(app_run, rater="Grace") as grace:
        answers: list[Any] = [shaped(grace.get("/api/session"), schemas.SessionOut)]
        # nobody's rating is hers to go back to
        assert fields(grace.get("/api/back"), 404) == ["request"]
        body = get_next(grace)
        answers.append(body)
        # the same subject, under an assignment of her own and as a new one
        assert body["assignment"]["subject_uid"] == first["subject_uid"]
        assert body["assignment"]["assignment_id"] != first["assignment_id"]
        assert body["assignment"]["previous"] is None
        assert body["progress"] | {"done_total": 0, "remaining": 3} == body["progress"]
        stored_body = rate(grace, body["assignment"])
        answers.append(stored_body)
        assert stored_body["saved"]["revision"] == 1 and stored_body["saved"]["marks"] == []
        back = shaped(grace.get("/api/back"), schemas.NextOut)
        answers.append(back)
        assert back["assignment"]["previous"] == {
            "values": NONE,
            "unratable": None,
            "other": False,
            "note": "",
            "marks": [],
        }
        answers.append(skip(grace, shown(grace)))
    text = json.dumps(answers)
    assert "Ottoline" not in text and "qzx" not in text
    # and what the first rater stored is as she left it
    assert rows(app_run, "SELECT note FROM submissions WHERE note != '' AND current") == [
        ("nota bene qzx",)
    ]


def masked(body: dict[str, Any]) -> dict[str, Any]:
    """An answer of ``next`` without what tells one subject from another."""
    assignment = dict(body["assignment"])
    for name in ("assignment_id", "subject_uid", "media_base"):
        assignment[name] = "…"
    return {**body, "assignment": assignment, "prefetch": len(body["prefetch"]), "progress": None}


def test_a_repeat_looks_like_a_new_subject(app_run: Path) -> None:
    set_app(app_run, repeat_rate=1.0, repeat_gap=0)
    with serving(app_run) as client:
        new = get_next(client)
        following = rate(client, new["assignment"], values={"A": 2, "B": 0, "C": 0})["next"]
        repeat = following["assignment"]
        # the rater has it again, and nothing in the answer says so
        assert repeat["subject_uid"] == new["assignment"]["subject_uid"]
        assert rows(app_run, "SELECT kind FROM assignments ORDER BY seq") == [
            ("main",),
            ("repeat",),
        ]
        assert (repeat["kind"], repeat["previous"]) == ("main", None)
        assert masked(following) == masked(new) | {"prefetch": masked(following)["prefetch"]}
        assert set(following) == set(new) and set(repeat) == set(new["assignment"])
        # it counts as an ordinary subject that was done, and is not one that remains
        assert following["progress"] | {"done_total": 1, "remaining": 2} == following["progress"]
        after = rate(client, repeat)["next"]
        assert after["progress"] | {"done_total": 2, "remaining": 2} == after["progress"]
        # a repeat writes no final label: the first rating stands
        assert rows(app_run, "SELECT value FROM final_labels WHERE criterion = 'A'") == [(2,)]


# --- reference examples -------------------------------------------------------------------------


def calibration_of(*subjects: CalibrationSubject) -> CalibrationSet:
    return CalibrationSet(version="1", subjects=subjects)


def test_the_examples_of_the_calibration_set_are_shown_after_its_subjects_are_rated(
    app_run: Path,
) -> None:
    order = queue(app_run)
    examples = (
        CalibrationExample(criterion="A", value=2, panel="sagittal", slice=3, caption="the sinus"),
        CalibrationExample(criterion="B", value=1, panel="coronal", slice=5),
    )
    write_calibration(
        app_run,
        calibration_of(
            CalibrationSubject(
                subject_uid=order[1], values={"A": 2, "B": 1, "C": 0}, examples=examples
            ),
            CalibrationSubject(subject_uid=order[2], values=NONE),
        ),
    )
    with serving(app_run) as client:
        session = lambda: shaped(client.get("/api/session"), schemas.SessionOut)  # noqa: E731
        assert session()["examples"] == []
        rate(client, shown(client))
        rate(client, shown(client))
        # one subject of the set is still to be rated: nobody is shown a consensus before
        assert session()["examples"] == []
        last = shown(client)
        skip(client, last)
        assert session()["examples"] == []
        rate(client, last)
        found = session()["examples"]
        key = last["render_key"]
        assert found == [
            {
                "criterion": "A",
                "value": 2,
                "panel": "sagittal",
                "slice": 3,
                "media_base": f"/media/{order[1]}/",
                "render_key": key,
                "caption": "the sinus",
                "tiles": last["render"]["tiles"],
            },
            {
                "criterion": "B",
                "value": 1,
                "panel": "coronal",
                "slice": 5,
                "media_base": f"/media/{order[1]}/",
                "render_key": key,
                "caption": "",
                "tiles": last["render"]["tiles"],
            },
        ]
    # the hash of the set is for submissions of calibration assignments, which a local
    # queue does not make
    assert rows(app_run, "SELECT DISTINCT calibration_sha256 FROM submissions") == [(None,)]


def test_a_calibration_set_the_run_cannot_use_stops_the_server(app_run: Path) -> None:
    order = queue(app_run)

    def refuse(subject: CalibrationSubject, said: str) -> None:
        (app_run / "calibration.yaml").unlink(missing_ok=True)
        write_calibration(app_run, calibration_of(subject))
        with pytest.raises(ConfigError, match=said) as caught:
            create_app(app_run)
        assert caught.value.reason == "invalid"

    refuse(CalibrationSubject(subject_uid="0" * 16, values=NONE), "not a subject of this run")
    refuse(CalibrationSubject(subject_uid=order[0], values={"A": 0}), "values are needed")
    refuse(
        CalibrationSubject(subject_uid=order[0], values={"A": 0, "B": 0, "C": 5}),
        "not on the scale",
    )
    bad_slice = CalibrationExample(criterion="A", value=1, panel="coronal", slice=6)
    refuse(
        CalibrationSubject(subject_uid=order[0], values=NONE, examples=(bad_slice,)),
        "slice 6 of the coronal panel",
    )
    (app_run / "calibration.yaml").write_text("version: 1\nsubjects: nobody\n")
    with pytest.raises(ConfigError, match=r"calibration\.yaml"):
        create_app(app_run)


def test_examples_of_a_subject_that_is_not_rendered_yet_wait_for_its_render(app_run: Path) -> None:
    order = queue(app_run)
    in_catalog(
        app_run, "UPDATE renders SET config_hash = 'dff40fe66869' WHERE subject_uid = ?", order[2]
    )
    example = CalibrationExample(criterion="A", value=1, panel="coronal", slice=6)
    wrong_value = CalibrationExample(criterion="A", value=7, panel="coronal", slice=0)
    late = CalibrationSubject(subject_uid=order[2], values=NONE, examples=(wrong_value,))
    write_calibration(app_run, calibration_of(late))
    # what needs no render is checked whether the subject is rendered or not
    with pytest.raises(ConfigError, match="not on the scale"):
        create_app(app_run)
    (app_run / "calibration.yaml").unlink()
    late = CalibrationSubject(subject_uid=order[2], values=NONE, examples=(example,))
    write_calibration(app_run, calibration_of(late))
    with serving(app_run) as client:
        rate(client, shown(client))
        rate(client, shown(client))
        assert get_next(client)["status"] == "rendering"
        assert client.get("/api/session").json()["examples"] == []
        # the render arrives, and its coronal panel has no slice 6: the example is left out
        in_catalog(
            app_run,
            "UPDATE renders SET config_hash ="
            " (SELECT config_hash FROM renders WHERE subject_uid = ?) WHERE subject_uid = ?",
            order[0],
            order[2],
        )
        rate(client, shown(client))
        assert client.get("/api/session").json()["examples"] == []


# --- what stops a server from starting ----------------------------------------------------------


def test_team_mode_is_not_served_yet(app_run: Path) -> None:
    with pytest.raises(ConfigError, match="milestone 5") as caught:
        create_app(app_run, mode="team")
    assert caught.value.reason == "invalid"
    assert not (app_run / "app.sqlite").exists()


def test_a_snap_radius_beyond_the_vertex_map_stops_the_server(app_run: Path) -> None:
    config = load_config(app_run)
    margin = config.render.vertex_map_margin_mm
    set_app(app_run, mark_max_dist_mm=margin + 0.5)
    with pytest.raises(ConfigError, match="vertex_map_margin_mm") as caught:
        create_app(app_run)
    assert caught.value.reason == "invalid" and not (app_run / "app.sqlite").exists()
    set_app(app_run, mark_max_dist_mm=margin)
    with serving(app_run) as client:
        assert client.get("/healthz").status_code == 200


def test_a_run_without_a_protocol_gets_the_shipped_one(run_copy: Path) -> None:
    assert not (run_copy / PROTOCOL_FILE).exists()
    with serving(run_copy) as client:
        assert client.get("/api/session").json()["protocol"]["version"] == "0.1-draft"
    edited = (run_copy / PROTOCOL_FILE).read_text().replace("0.1-draft", "0.2-pilot")
    (run_copy / PROTOCOL_FILE).write_text(edited)
    # a protocol that is there is the run's own
    with serving(run_copy) as client:
        first = shown(client)
        assert client.get("/api/session").json()["protocol"]["version"] == "0.2-pilot"
        assert fields(client.post("/api/rating", json=rating(first)), 409) == ["protocol_version"]
        rate(client, first, protocol_version="0.2-pilot")
    assert rows(run_copy, "SELECT protocol_version, protocol_sha256 FROM submissions") == [
        ("0.2-pilot", hashlib.sha256(edited.encode()).hexdigest())
    ]


def test_what_cannot_be_served_is_a_config_error(app_run: Path, tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        create_app(tmp_path / "no run")
    with pytest.raises(ConfigError):
        create_app(tmp_path)
    (app_run / PROTOCOL_FILE).write_text("version: 3\nscale: none\n")
    with pytest.raises(ConfigError, match=r"protocol\.yaml"):
        create_app(app_run)
    assert not (app_run / "app.sqlite").exists()
    (app_run / PROTOCOL_FILE).unlink()
    install_protocol(app_run)
    for name in ("", " Ada", "Ada\n"):
        with pytest.raises(ConfigError, match="name") as caught:
            create_app(app_run, rater=name)
        assert caught.value.reason == "invalid"
    assert rows(app_run, "SELECT count(*) FROM raters") == [(0,)]
    # the database of another run
    (app_run / "app.sqlite").unlink()
    open_app_db(app_run, run_id="another run", mode="local").dispose()
    with pytest.raises(ConfigError, match="another run"):
        create_app(app_run)
    # after each refusal the files are free: a server starts once the cause is gone
    (app_run / "app.sqlite").unlink()
    with serving(app_run) as client:
        assert shown(client)["assignment_id"] == 1


def test_a_failed_render_has_its_label_from_the_start(app_run: Path) -> None:
    order = queue(app_run)
    in_catalog(
        app_run,
        "UPDATE renders SET status = 'failed', reason = 'missing surface', render_key = NULL "
        "WHERE subject_uid = ?",
        order[0],
    )
    with serving(app_run) as client:
        assert rows(app_run, "SELECT * FROM final_labels") == [(order[0], "unratable", 2, "single")]
        assert shown(client)["subject_uid"] == order[1]
        assert rate_all(client) == 2
        table = client.get("/api/review").json()["rows"]
        assert (table[0]["render_status"], table[0]["unratable_reason"]) == (
            "failed",
            "missing surface",
        )


# --- when something goes wrong inside -----------------------------------------------------------


def test_an_error_inside_a_route_says_nothing_of_what_it_was(
    client: TestClient,
    app_run: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    first = shown(client)

    def fail(*_args: object, **_kw: object) -> None:
        raise RuntimeError(f"sub-001 of ds000102 at {app_run}/catalog.sqlite")

    # the last thing every one of the three routes does before it commits
    monkeypatch.setattr(server, "subjects_left", fail)
    before = stored(app_run)
    with caplog.at_level("ERROR", logger="surfcensus"):
        for response in (
            client.get("/api/next"),
            client.post("/api/rating", json=rating(first)),
            client.post("/api/skip", json=skip_body(first)),
        ):
            where, message = refused(response, 500)[0]
            assert where == "request"
            for private in ("sub-001", "ds000102", "catalog", str(app_run), "RuntimeError"):
                assert private not in message
            assert all(name in response.headers for name in SECURITY_HEADERS)
    # and nothing of the request was stored: the rating and the skip were rolled back
    assert stored(app_run) == before
    assert rows(app_run, "SELECT completed_at, skip_count FROM assignments") == [(None, 0)]
    assert "RuntimeError" in caplog.text and str(app_run) not in caplog.text

    def busy(*_args: object, **_kw: object) -> None:
        raise OperationalError(
            "BEGIN IMMEDIATE", {}, sqlite3.OperationalError("database is locked")
        )

    monkeypatch.setattr(server, "subjects_left", busy)
    where, message = refused(client.get("/api/next"), 503)[0]
    assert "try again" in message and "BEGIN" not in message
    monkeypatch.undo()
    assert shown(client) == first


def test_an_error_of_an_unblinded_route_says_what_the_operator_needs(
    client: TestClient, app_run: Path
) -> None:
    rate(client, shown(client))
    (app_run / "tables" / "scores.parquet").write_bytes(b"not parquet")
    where, message = refused(client.get("/api/review"), 500)[0]
    assert where == "request" and "scores" in message and str(app_run) not in message
    # the rating goes on without the scores: the queue is then in the order of the uids
    assert get_next(client)["status"] == "assignment"


# --- several requests at once -------------------------------------------------------------------


def test_requests_that_arrive_together_are_answered_one_after_the_other(app_run: Path) -> None:
    handle = start_server(app_run, port=0)
    try:

        def call(path: str, body: dict[str, Any] | None = None) -> tuple[int, Any]:
            data = None if body is None else json.dumps(body).encode()
            request = urllib.request.Request(
                handle.url + path, data=data, headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    return response.status, json.load(response)
            except urllib.error.HTTPError as error:
                return error.code, json.load(error)

        status, first = call("api/next")
        assert status == 200
        assignment_id = first["assignment"]["assignment_id"]
        results: list[tuple[int, Any]] = []
        lock = threading.Lock()

        def work(number: int) -> None:
            if number % 3 == 0:
                out = call("api/next")
            elif number % 3 == 1:
                events = [{"ts": f"t{number}", "type": "play", "payload": {"n": number}}]
                out = call("api/events", {"assignment_id": assignment_id, "events": events})
            else:
                out = call("api/session")
            with lock:
                results.append(out)

        threads = [threading.Thread(target=work, args=(number,)) for number in range(24)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
        assert [status for status, _body in results] == [200] * 24
        assert n_rows(app_run, "assignments") == 1 and n_rows(app_run, "events") == 8 + 3
        status, body = call("api/rating", rating(assignment_id))
        assert status == 200 and body["saved"]["revision"] == 1
    finally:
        handle.stop()


# --- starting and stopping ----------------------------------------------------------------------


def fetch(url: str, host: str | None = None) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(url, headers={} if host is None else {"Host": host})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def accepts(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(1)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def test_a_started_server_answers_on_the_port_it_says_and_stops(app_run: Path) -> None:
    handle = start_server(app_run, port=0)
    try:
        assert isinstance(handle, ServerHandle) and handle.running
        assert handle.port > 0 and handle.url == f"http://127.0.0.1:{handle.port}/"
        assert fetch(handle.url + "healthz") == (
            200,
            {"ok": True, "version": surfcensus.__version__},
        )
        assert fetch(handle.url + "api/next")[1]["status"] == "assignment"
        # over a real connection as well: another name for this machine is refused
        assert fetch(handle.url + "healthz", host="evil.example")[0] == 403
        assert fetch(f"http://localhost:{handle.port}/healthz")[0] == 200
    finally:
        handle.stop()
    assert not handle.running and not accepts(handle.port)
    handle.stop()  # a second call does nothing
    # and the run can be served again
    with serving(app_run) as client:
        assert shown(client)["assignment_id"] == 1


def test_a_taken_port_moves_the_server_to_the_next_free_one(app_run: Path) -> None:
    assert (DEFAULT_PORT, PORT_TRIES) == (8765, 20)
    taken = socket.socket()
    taken.bind(("127.0.0.1", 0))
    taken.listen(1)
    port = taken.getsockname()[1]
    try:
        handle = start_server(app_run, port=port)
        try:
            assert port < handle.port <= port + PORT_TRIES
            assert fetch(handle.url + "healthz")[0] == 200
        finally:
            handle.stop()
    finally:
        taken.close()


def test_no_free_port_and_no_local_host_are_config_errors(
    app_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for host in ("0.0.0.0", "192.168.1.5", "example.org", "::"):
        with pytest.raises(ConfigError, match="local server") as caught:
            start_server(app_run, host=host, port=0)
        assert caught.value.reason == "invalid"
    for port in (-1, 65536):
        with pytest.raises(ConfigError, match="port"):
            start_server(app_run, port=port)
    monkeypatch.setattr(server, "PORT_TRIES", 1)
    held = []
    try:
        first = socket.socket()
        first.bind(("127.0.0.1", 0))
        first.listen(1)
        held.append(first)
        port = first.getsockname()[1]
        second = socket.socket()
        try:
            second.bind(("127.0.0.1", port + 1))
            second.listen(1)
            held.append(second)
        except OSError:
            pytest.skip("the port after a free one is taken by something else")
        with pytest.raises(ConfigError, match="in use"):
            start_server(app_run, port=port)
    finally:
        for sock in held:
            sock.close()
    # a server that could not start leaves the database free
    with serving(app_run) as client:
        assert client.get("/healthz").status_code == 200


def test_serve_refuses_a_run_whose_renders_are_out_of_date(
    app_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def started(*_args: object, **_kw: object) -> None:
        # serve would then wait for a signal: fail here, not by a test that never ends
        raise AssertionError("the server was started")

    monkeypatch.setattr(server, "start_server", started)
    order = queue(app_run)
    said = ["1 render is out of date", "2 renders are out of date"]
    for uid, expected in zip(order[:2], said, strict=True):
        in_catalog(
            app_run, "UPDATE renders SET config_hash = 'dff40fe66869' WHERE subject_uid = ?", uid
        )
        with pytest.raises(ConfigError) as caught:
            serve(app_run, port=0, open_browser=False)
        assert caught.value.reason == "invalid"
        assert caught.value.detail == f"{expected}; run surfcensus render --run-dir {app_run}"
    assert not (app_run / "app.sqlite").exists()


def test_serve_says_when_there_is_nothing_to_rate(
    app_run: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Started(Exception):
        """In place of a server, which would wait for a signal."""

    def started(*_args: object, **_kw: object) -> None:
        raise Started

    monkeypatch.setattr(server, "start_server", started)

    def said() -> str:
        with pytest.raises(Started):
            serve(app_run, port=0, open_browser=False)
        return capsys.readouterr().err

    assert said() == ""
    # a run of the phase commands: every present subject is in the queue, and nothing is said
    in_catalog(app_run, "DELETE FROM selection")
    assert said() == ""
    # no subject at all: the app would only say that everything is rated
    in_catalog(app_run, "UPDATE subjects SET present = 0")
    assert said() == (
        "There is nothing to rate: the run has no subject; "
        f"run surfcensus inventory --run-dir {app_run}\n"
    )


def test_serve_listens_on_this_machine_only(app_run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def listened(*_args: object, **_kw: object) -> None:
        raise AssertionError("the server went on to listen")

    monkeypatch.setattr(server, "_listen", listened)
    with pytest.raises(ConfigError, match="local server"):
        serve(app_run, host="0.0.0.0", port=0, open_browser=False)


@contextmanager
def interrupted_if_it_waits_on(seconds: float = 20.0) -> Iterator[list[bool]]:
    """Around a call of ``serve`` that is meant to return. When it has not after
    ``seconds``, an interrupt ends it and the list says so: the test fails and does not
    hang."""
    waited_on: list[bool] = []

    def interrupt() -> None:
        waited_on.append(True)
        os.kill(os.getpid(), signal.SIGINT)

    earlier = signal.signal(signal.SIGINT, signal.default_int_handler)
    timer = threading.Timer(seconds, interrupt)
    timer.start()
    try:
        yield waited_on
    finally:
        timer.cancel()
        signal.signal(signal.SIGINT, earlier)


def test_serve_prints_the_url_opens_the_browser_and_stops_at_sigterm(
    app_run: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    opened: list[str] = []
    answers: list[Any] = []

    def browser(url: str) -> bool:
        opened.append(url)

        def visit() -> None:
            answers.append(fetch(url + "healthz"))
            os.kill(os.getpid(), signal.SIGTERM)

        threading.Thread(target=visit).start()
        return True

    monkeypatch.setattr(server.webbrowser, "open", browser)
    earlier = signal.getsignal(signal.SIGTERM)
    with interrupted_if_it_waits_on() as waited_on:
        serve(app_run, port=0)  # returns because of the signal
    assert not waited_on
    assert signal.getsignal(signal.SIGTERM) is earlier
    (url,) = opened
    assert answers == [(200, {"ok": True, "version": surfcensus.__version__})]
    assert f"Rating app: {url} (Ctrl-C to stop)" in capsys.readouterr().err
    assert not accepts(int(url.rsplit(":", 1)[1].strip("/")))
    # without a browser nothing is opened
    opened.clear()

    def stop_when_it_listens() -> None:
        # not before serve has taken the signal: it would end the tests
        while signal.getsignal(signal.SIGTERM) is earlier:
            time.sleep(0.01)
        time.sleep(0.2)
        os.kill(os.getpid(), signal.SIGTERM)

    stopper = threading.Thread(target=stop_when_it_listens)
    stopper.start()
    with interrupted_if_it_waits_on() as waited_on:
        serve(app_run, port=0, open_browser=False)
    stopper.join()
    assert not waited_on
    assert opened == [] and signal.getsignal(signal.SIGTERM) is earlier


def test_serve_stops_at_ctrl_c(app_run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ports: list[int] = []

    def browser(url: str) -> bool:
        # called once the server listens and serve waits: the interrupt arrives in there
        ports.append(int(url.rsplit(":", 1)[1].strip("/")))
        assert accepts(ports[0])
        threading.Thread(target=os.kill, args=(os.getpid(), signal.SIGINT)).start()
        return True

    monkeypatch.setattr(server.webbrowser, "open", browser)
    # Python's own handler, whatever the process was started with
    earlier = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        serve(app_run, port=0)
    except KeyboardInterrupt:
        pytest.fail("serve let the interrupt through")
    finally:
        signal.signal(signal.SIGINT, earlier)
    assert len(ports) == 1 and not accepts(ports[0])


# --- on the real subject ------------------------------------------------------------------------


def test_a_mark_on_the_real_subject_is_the_nearest_vertex_of_its_surface_file(
    cached_subjects: Path, tmp_path: Path
) -> None:
    subject = cached_subjects / "ds000102-sub-01"
    template = cached_subjects / "ds000102-fsaverage"
    if not (subject / "surf" / "lh.pial").is_file():
        pytest.skip("the real subject is not in the pytest cache (run the network tests once)")
    subjects_dir = tmp_path / "subjects"
    subjects_dir.mkdir()
    (subjects_dir / "sub-01").symlink_to(subject, target_is_directory=True)
    has_template = (template / "surf" / "lh.sphere.reg").is_file()
    if has_template:
        (subjects_dir / "fsaverage").symlink_to(template, target_is_directory=True)
    run_dir = tmp_path / "run"
    with Run(run_dir, local(subjects_dir)) as run:
        run.inventory()
        (uid,) = run.public().subject_uids()
        selection = pd.DataFrame(
            {
                "subject_uid": [uid],
                "reason": "all",
                "stratum": "all",
                "inclusion_prob": 1.0,
                "seed": 0,
            }
        )
        run.public().replace_selection(selection, ["all"])
        report = run.render(video=False)
        assert (report.done, report.failed) == (1, 0)
    info = read_render(media_dir(run_dir, uid))
    assert info is not None
    pial, _faces = fs.read_geometry(subject / "surf" / "lh.pial")
    white, _faces = fs.read_geometry(subject / "surf" / "lh.white")
    affine = np.array(info.volume.vox2ras_tkr)
    # a pial vertex of the left hemisphere on a sagittal slice plane, off the midline
    panel = info.panels["sagittal"]
    position = len(panel.slices) // 4
    plane = geometry.apply_affine(affine, geometry.pixel_to_voxel(panel, position, 0.0, 0.0))[0]
    px_all, py_all = geometry.tkr_to_pixel(panel, affine, pial)
    on_tile = (px_all > 5) & (px_all < panel.width - 5) & (py_all > 5) & (py_all < panel.height - 5)
    candidates = np.flatnonzero(on_tile & (np.abs(pial[:, 0] - plane) < 0.2))
    assert len(candidates) > 0
    vertex = int(candidates[len(candidates) // 2])
    posted = {
        "client_id": "m1",
        "class": "A",
        "panel": "sagittal",
        "slice": position,
        "px": float(px_all[vertex]),
        "py": float(py_all[vertex]),
    }
    point = geometry.apply_affine(
        affine, geometry.pixel_to_voxel(panel, position, posted["px"], posted["py"])
    )
    with serving(run_dir) as client:
        first = shown(client)
        assert first["subject_uid"] == uid and first["has_vertex_map"] is True
        for name in ("poster.webp", "tiles/sagittal_030.webp", "tiles/axial_060_c.webp"):
            assert client.get(f"/media/{uid}/{name}?v={first['render_key']}").status_code == 200
        body = rate(client, first, values={"A": 2, "B": 0, "C": 0}, marks=[posted])
    (saved,) = body["saved"]["marks"]
    # the vertex map of the render gives what the surface files of FreeSurfer give
    expected_pial, pial_dist = nearest(pial, point)
    expected_white, white_dist = nearest(white, point)
    assert pial_dist < 0.25 and saved["hemi"] == "lh"
    assert saved["pial_vertex"] == expected_pial
    assert saved["pial_dist_mm"] == pytest.approx(pial_dist, abs=1e-4)
    if white_dist <= 5.0:
        assert saved["white_vertex"] == expected_white
        assert saved["white_dist_mm"] == pytest.approx(white_dist, abs=1e-4)
    else:
        assert saved["white_vertex"] is None
    assert saved["vertex_surface"] == ("pial" if has_template else None)
    if has_template:
        assert 0 <= saved["fsavg6_vertex"] < 40_962
    else:
        assert saved["fsavg6_vertex"] is None


# --- the pages on the server, in a browser ------------------------------------------------------


@pytest.mark.browser
def test_the_pages_work_on_the_server_in_a_browser(app_run: Path) -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    from browser_launch import launch_browser  # pyrefly: ignore[missing-import]

    handle = start_server(app_run, port=0)
    problems: list[str] = []
    elsewhere: list[str] = []
    try:
        with sync_api.sync_playwright() as playwright:
            browser = launch_browser(playwright)
            page = browser.new_page(viewport={"width": 1400, "height": 900})
            page.add_init_script(
                "try { localStorage.setItem('surfcensus.tour.seen:local', '1'); } catch (e) {}"
            )
            page.on("console", lambda m: problems.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda error: problems.append(str(error)))
            page.on(
                "request",
                lambda r: (
                    None if r.url.startswith((handle.url, "data:")) else elsewhere.append(r.url)
                ),
            )

            def ready() -> str:
                page.wait_for_selector("#viewer[data-ready='true']")
                return page.evaluate(
                    "document.querySelector('.tile-gray').src.match(/media\\/([0-9a-f]{16})/)[1]"
                )

            def wait_for(uid: str) -> None:
                page.wait_for_function(
                    "uid => document.querySelector('.tile-gray').src.includes(uid)"
                    " && document.getElementById('viewer').dataset.ready === 'true'",
                    arg=uid,
                )

            def pause() -> None:
                if page.evaluate("document.getElementById('viewer').dataset.playing") == "true":
                    page.keyboard.press(" ")

            page.goto(handle.url)
            first = ready()
            pause()
            # rate with a mark, submit
            page.keyboard.press("3")
            box = page.locator(".panel[data-panel='sagittal'] .stage").bounding_box()
            page.mouse.click(box["x"] + box["width"] * 0.4, box["y"] + box["height"] * 0.5)
            page.keyboard.press("4")
            page.keyboard.press("7")
            page.keyboard.press("Enter")
            page.wait_for_function(
                "uid => !document.querySelector('.tile-gray').src.includes(uid)", arg=first
            )
            second = ready()
            assert rows(app_run, "SELECT revision, current FROM submissions") == [(1, 1)]
            assert rows(app_run, "SELECT class, panel FROM marks") == [("A", "sagittal")]
            # in the middle of a subject: reload, and it is the same one, as it was left
            pause()
            page.keyboard.press("2")
            page.reload()
            assert ready() == second and n_rows(app_run, "assignments") == 2
            assert page.locator(".level[aria-pressed='true']").count() == 1
            # skip: the third, and the line says that one waits
            pause()
            page.keyboard.press("k")
            page.wait_for_function(
                "uid => !document.querySelector('.tile-gray').src.includes(uid)", arg=second
            )
            third = ready()
            assert third not in (first, second)
            assert "1 skipped, waiting" in page.inner_text("#progress-line")
            assert rows(app_run, "SELECT skip_count FROM assignments ORDER BY seq") == [
                (0,),
                (1,),
                (0,),
            ]
            # skip, then Backspace: the skipped one, with what was entered
            page.keyboard.press("Backspace")
            wait_for(second)
            assert page.locator(".level[aria-pressed='true']").count() == 1
            pause()
            page.keyboard.press("k")
            wait_for(third)
            # rate the third: the skipped one comes back after it
            pause()
            for key in ("1", "4", "7", "Enter"):
                page.keyboard.press(key)
            wait_for(second)
            # go back to the third and revise it
            page.keyboard.press("Backspace")
            wait_for(third)
            assert page.locator(".level[aria-pressed='true']").count() == 3
            pause()
            page.keyboard.press("5")
            page.keyboard.press("Enter")
            wait_for(second)
            assert rows(
                app_run, "SELECT assignment_id, revision, current FROM submissions ORDER BY 1, 2"
            ) == [(1, 1, 1), (3, 1, 0), (3, 2, 1)]
            pause()
            for key in ("4", "7", "Enter"):
                page.keyboard.press(key)
            page.wait_for_selector("#rate[data-state='done']")
            assert rows(app_run, "SELECT count(*) FROM assignments WHERE completed_at IS NULL") == [
                (0,)
            ]
            assert rows(app_run, "SELECT skip_count FROM assignments ORDER BY seq") == [
                (0,),
                (2,),
                (0,),
            ]
            page.wait_for_load_state("networkidle")
            page.goto(handle.url + "review")
            page.wait_for_selector("#review[data-state='open']")
            assert page.locator("#subjects tbody tr").count() == 3
            page.locator("#subjects tbody tr").first.click()
            page.wait_for_selector("#viewer[data-ready='true']")
            browser.close()
    finally:
        handle.stop()
    assert problems == [] and elsewhere == []
    types = {kind for (kind,) in rows(app_run, "SELECT DISTINCT type FROM events")}
    assert {"skip", "back", "load", "config_change"} <= types
