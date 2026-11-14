"""The EPI pages of the main app on a run directory (work package E2.D; docs/epi_interface.md,
"The app in a run directory"): the index of the participants, the page and the stream of one
participant, the files of its packs, and the blindness rules as rules of the routes.

The table of "Blindness", cell by cell: in a local run that is not blind and has one rater the
routes are open; in a blind run and in a run with several raters they are closed (403, and the
reason as a page for a browser) until the rater has nothing left to rate; in a census run, in
a run whose EPI interface is off and in a run without functional data they answer as a route
that does not exist, exactly as before step 2. The rating screen gets one link, ``/epi``, and
nothing else of the EPI interface.

The run directory is ``app_run`` (three rendered synthetic subjects) with the EPI interface of
``tests/epi_run_fixture.py`` written into it.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from epi_packs import PackedStream, make_stream  # pyrefly: ignore[missing-import]
from epi_run_fixture import FULL, HALF, LABELS, EpiRun, add_epi  # pyrefly: ignore[missing-import]
from fastapi.testclient import TestClient

from surfcensus.app import epi_schemas as shapes
from surfcensus.app import schemas
from surfcensus.app.epi import Packs
from surfcensus.app.server import create_app
from surfcensus.config import EpiConfig, load_config, with_changes, write_config
from surfcensus.func import PACK_JSON, SESSION_JSON, STREAM_JSON, TRACES_JSON
from surfcensus.func.model import SessionInfo, TracesInfo, read_info
from surfcensus.func.packread import read_pack
from surfcensus.store import EPI_DIR, PublicCatalog

BASE = "http://127.0.0.1"
PAGE = {"Accept": "text/html,application/xhtml+xml"}
NO_ROUTE = {"errors": [{"field": "request", "message": "there is no such route"}]}
EPI_ROUTES = tuple(
    (route.method, route.path) for route in schemas.ROUTES if route.path.startswith("/epi")
) + tuple(
    (route.method, route.path) for route in schemas.ROUTES if route.path.startswith("/api/epi")
)


@pytest.fixture(scope="module")
def packed(tmp_path_factory: pytest.TempPathFactory) -> PackedStream:
    return make_stream(tmp_path_factory.mktemp("epirun"), n_volumes=10)


@pytest.fixture
def epi_run(app_run: Path, packed: PackedStream, small_run_uids: dict[str, str]) -> EpiRun:
    return add_epi(app_run, packed, subject_uid=small_run_uids["sub-001"])


@contextmanager
def serving(run_dir: Path, **options: Any) -> Iterator[TestClient]:
    with TestClient(create_app(run_dir, **options), base_url=BASE) as client:
        yield client


@pytest.fixture
def client(epi_run: EpiRun) -> Iterator[TestClient]:
    with serving(epi_run.run_dir) as started:
        yield started


def set_app(run_dir: Path, **changes: object) -> None:
    config = load_config(run_dir)
    write_config(run_dir, with_changes(config, app=config.app.model_dump() | changes))


def set_epi(run_dir: Path, epi: EpiConfig) -> None:
    write_config(run_dir, with_changes(load_config(run_dir), epi=epi.model_dump()))


def ok(response: Any, shape: type | None = None) -> Any:
    assert response.status_code == 200, (response.status_code, response.text[:400])
    if shape is None:
        return response
    body = response.json()
    shape.model_validate(body)
    return body


def paths(epi: EpiRun) -> dict[str, str]:
    """One concrete path of every EPI route, by route."""
    session = epi.sessions[FULL][0]
    info = read_info(epi.run_dir / EPI_DIR / FULL / session / SESSION_JSON, SessionInfo)
    assert info is not None
    run = next(r.run for r in info.runs if r.reason is None)
    pack = read_pack(epi.run_dir / EPI_DIR / FULL / session, run)
    assert pack is not None
    return {
        "/epi": "/epi",
        "/api/epi": "/api/epi",
        "/epi/{participant_uid}": f"/epi/{FULL}",
        "/api/epi/{participant_uid}": f"/api/epi/{FULL}",
        "/epi-media/{participant_uid}/{session_uid}/{path}": (
            f"/epi-media/{FULL}/{session}/{pack.streams['raw'].ref.file}"
        ),
    }


def rating_body(client: TestClient, assignment_id: int, at: int) -> dict[str, Any]:
    version = client.get("/api/session").json()["protocol"]["version"]
    return {
        "assignment_id": assignment_id,
        "protocol_version": version,
        "values": {"A": 0, "B": 0, "C": 0},
        "unratable": None,
        "other": False,
        "note": "",
        "marks": [],
        "client_ts": f"2026-11-14T12:00:{at:02d}.000Z",
        "ms_on_subject": 1000,
    }


def rate_everything(client: TestClient) -> int:
    n = 0
    while True:
        answer = client.get("/api/next").json()
        if answer["status"] != "assignment":
            return n
        body = rating_body(client, answer["assignment"]["assignment_id"], n)
        assert client.post("/api/rating", json=body).status_code == 200, client.post
        n += 1


# --- the routes there are -------------------------------------------------------------------


def test_the_epi_routes_of_the_schemas_are_unblinded_and_answer_with_their_shapes() -> None:
    routes = {(r.method, r.path): r for r in schemas.ROUTES}
    assert {key for key in routes if "epi" in key[1]} == {
        ("GET", "/epi"),
        ("GET", "/api/epi"),
        ("GET", "/epi/{participant_uid}"),
        ("GET", "/api/epi/{participant_uid}"),
        ("GET", "/epi-media/{participant_uid}/{session_uid}/{path}"),
    }
    assert all(routes[key].unblinded for key in routes if "epi" in key[1])
    assert routes[("GET", "/api/epi")].response is shapes.EpiIndexOut
    assert routes[("GET", "/api/epi/{participant_uid}")].response is shapes.EpiStreamOut


def test_every_epi_route_answers_in_a_local_run_with_one_rater_that_is_not_blind(
    client: TestClient, epi_run: EpiRun
) -> None:
    for route, path in paths(epi_run).items():
        answer = ok(client.get(path, headers=PAGE if "api" not in route else {}))
        if route in ("/epi", "/epi/{participant_uid}"):
            assert answer.headers["content-type"].startswith("text/html")
            assert answer.headers["cache-control"] == "no-cache"
        csp = answer.headers["content-security-policy"]
        assert "default-src 'self'" in csp and "frame-ancestors 'none'" in csp
    session = ok(client.get("/api/session"), schemas.SessionOut)
    assert session["epi_url"] == "/epi"


# --- a run without the EPI interface answers as before ------------------------------------


def no_route(answer: Any) -> None:
    assert answer.status_code == 404
    assert answer.json() == NO_ROUTE
    assert answer.headers["content-type"].startswith("application/json")


def without_epi_answers(client: TestClient, run_dir: Path, sessions: tuple[str, ...]) -> None:
    """Every EPI route answers as a path that is no route, for a page and for a script, and
    the session carries no link."""
    unknown = client.get("/no-such-route")
    no_route(unknown)
    participant = FULL
    for path in (
        "/epi",
        "/api/epi",
        f"/epi/{participant}",
        f"/api/epi/{participant}",
        f"/epi-media/{participant}/{sessions[0] if sessions else '0' * 16}/contours/x/axial.bin",
        f"/epi/{'0' * 16}",
    ):
        for headers in ({}, PAGE):
            answer = client.get(path, headers=headers)
            no_route(answer)
            assert answer.headers["cache-control"] == unknown.headers["cache-control"]
    assert client.get("/api/session").json()["epi_url"] is None


def test_a_run_without_functional_data_answers_every_epi_route_as_no_route(app_run: Path) -> None:
    with serving(app_run) as client:
        without_epi_answers(client, app_run, ())
        # the rating routes answer as they did; the session has the field, null
        session = ok(client.get("/api/session"), schemas.SessionOut)
        assert set(session) == set(schemas.SessionOut.model_fields)
        assert ok(client.get("/api/next"), schemas.NextOut)["status"] == "assignment"


def test_a_census_run_answers_every_epi_route_as_no_route(epi_run: EpiRun) -> None:
    # the EPI phases never run in a census run; here a local run's catalog is made to say
    # census, with its functional rows in place, so that nothing but the mode closes them
    connection = sqlite3.connect(epi_run.run_dir / "catalog.sqlite")
    with connection:
        connection.execute("UPDATE meta SET value = 'census' WHERE key = 'mode'")
    connection.close()
    with PublicCatalog.open(epi_run.run_dir) as public:
        assert public.mode() == "census" and public.has_func()
    with serving(epi_run.run_dir) as client:
        without_epi_answers(client, epi_run.run_dir, epi_run.sessions[FULL])


def test_a_run_whose_epi_interface_is_off_answers_every_epi_route_as_no_route(
    epi_run: EpiRun,
) -> None:
    set_epi(epi_run.run_dir, EpiConfig(enabled=False))
    with serving(epi_run.run_dir) as client:
        without_epi_answers(client, epi_run.run_dir, epi_run.sessions[FULL])


# --- the blindness table ----------------------------------------------------------------------


def closed(client: TestClient, epi: EpiRun) -> None:
    for route, path in paths(epi).items():
        answer = client.get(path)
        assert answer.status_code == 403, path
        assert answer.json() == {
            "errors": [
                {
                    "field": "request",
                    "message": "you still have subjects to rate; the EPI movies open when none "
                    "is left",
                }
            ]
        }
        page = client.get(path, headers=PAGE)
        assert page.status_code == 403
        if route in ("/epi", "/epi/{participant_uid}"):
            assert page.headers["content-type"].startswith("text/html")
            assert page.headers["cache-control"] == "no-store"
            assert "the EPI movies open when none is left" in page.text
            assert '<a href="/">Back to the rating screen</a>' in page.text
            assert (
                "<script" not in page.text and "<title>surfcensus: EPI movies</title>" in page.text
            )
            for uid in (*epi.participants, *epi.sessions[FULL]):
                assert uid not in page.text
        else:
            assert page.headers["content-type"].startswith("application/json")
    assert client.get("/api/session").json()["epi_url"] is None


def opened(client: TestClient, epi: EpiRun) -> None:
    for path in paths(epi).values():
        assert client.get(path).status_code == 200, path
    assert client.get("/api/session").json()["epi_url"] == "/epi"


def test_a_blind_run_opens_the_epi_pages_only_when_nothing_is_left_to_rate(
    epi_run: EpiRun,
) -> None:
    set_app(epi_run.run_dir, blind=True)
    with serving(epi_run.run_dir) as client:
        closed(client, epi_run)
        # the rating routes are open all along, and the review is closed as before
        assert client.get("/").status_code == 200
        assert client.get("/api/review").status_code == 403
        assert rate_everything(client) == 3
        opened(client, epi_run)


def test_a_second_rater_closes_the_epi_pages_until_that_rater_is_done(epi_run: EpiRun) -> None:
    with serving(epi_run.run_dir, rater="ada") as client:
        opened(client, epi_run)  # one rater, not blind: open
        assert rate_everything(client) == 3
    with serving(epi_run.run_dir, rater="ben") as client:
        closed(client, epi_run)
        rate_everything(client)
        opened(client, epi_run)
    # the first rater has nothing left either: open to her as well
    with serving(epi_run.run_dir, rater="ada") as client:
        opened(client, epi_run)


def test_the_closed_review_page_is_as_it_was(epi_run: EpiRun) -> None:
    set_app(epi_run.run_dir, blind=True)
    with serving(epi_run.run_dir) as client:
        page = client.get("/review", headers=PAGE)
        assert page.status_code == 403
        assert "<title>surfcensus: review</title>" in page.text
        assert "you still have subjects to rate; the review opens when none is left" in page.text


# --- the index --------------------------------------------------------------------------------


def test_the_index_is_the_catalog_and_what_the_run_directory_shows(
    client: TestClient, epi_run: EpiRun, small_run_uids: dict[str, str], packed: PackedStream
) -> None:
    index = ok(client.get("/api/epi"), shapes.EpiIndexOut)
    assert [p["participant_uid"] for p in index["participants"]] == [FULL, HALF]  # by label
    with PublicCatalog.open(epi_run.run_dir) as public:
        runs = public.func_runs()
        sessions = public.func_sessions()
    for p in index["participants"]:
        uid = p["participant_uid"]
        assert p["label"] == LABELS[uid]
        assert p["url"] == f"/epi/{uid}"
        assert [s["session_uid"] for s in p["sessions"]] == list(epi_run.sessions[uid])
        assert [s["session_uid"] for s in p["sessions"] if s["shown"]] == list(epi_run.shown[uid])
        mine = sessions[sessions["participant_uid"] == uid]
        assert [s["label"] for s in p["sessions"]] == [str(v) for v in mine["session_label"]]
        for s in p["sessions"]:
            listed = runs[runs["session_uid"] == s["session_uid"]]
            packable = int(listed["packable"].astype(bool).sum())
            assert s["n_runs"] == len(listed) == len(epi_run.runs[s["session_uid"]])
            assert s["n_packable"] == packable
            assert s["n_packed"] == len(epi_run.packed[s["session_uid"]])
            assert s["n_failed"] == sum(r in epi_run.failed for r in epi_run.runs[s["session_uid"]])
            assert s["n_pending"] == packable - s["n_packed"] - s["n_failed"]
            assert s["n_flagged"] == int(listed["flagged"].fillna(False).astype(bool).sum())
        for name in ("n_runs", "n_packable", "n_packed", "n_pending", "n_failed", "n_flagged"):
            assert p[name] == sum(s[name] for s in p["sessions"]), name
        assert p["n_sessions"] == len(p["sessions"]) == 2
        assert p["can_open"] is True
    full, half = index["participants"]
    assert full["subject_uids"] == [small_run_uids["sub-001"]] and half["subject_uids"] == []
    flagged = sum(
        run.frame is not None and run.frame.flagged
        for info in packed.infos.values()
        for run in info.runs
    )
    assert flagged >= 1 and full["n_flagged"] == half["n_flagged"] == flagged
    assert full["n_pending"] == 0 and full["n_failed"] == 0
    assert half["n_failed"] == 1 and half["n_pending"] == half["sessions"][1]["n_packable"] - 1 > 0
    assert index["n_pending"] == half["n_pending"]


def test_the_index_page_is_its_template_with_the_address_of_its_api(client: TestClient) -> None:
    page = ok(client.get("/epi", headers=PAGE)).text
    assert 'data-api="/api/epi"' in page
    assert '<script type="module" src="/static/js/epi/index.js?v=' in page
    assert '<a id="rate-link" href="/">rating screen</a>' in page
    assert "raw data" in page and "preprocessing" in page
    # nothing about a participant: everything arrives through the API
    for uid in (FULL, HALF, *LABELS.values()):
        assert uid not in page
    assert ok(client.get("/static/js/epi/index.js")).text.startswith("// The index")


# --- the page and the stream of a participant --------------------------------------------------


def test_the_page_of_a_participant_plays_its_stream_from_the_run(
    client: TestClient, epi_run: EpiRun
) -> None:
    page = ok(client.get(f"/epi/{FULL}", headers=PAGE)).text
    assert f'data-api="/api/epi/{FULL}"' in page
    # nothing waits: the page asks at the slow interval, so that a page left open still
    # notices packs made later (finding F4 of the review of step 2)
    assert 'data-poll="30"' in page and 'data-waiting="false"' in page
    assert 'data-poll-waiting="5" data-poll-idle="30"' in page
    assert f'data-index-api="/api/epi" data-participant="{FULL}"' in page
    assert '<a id="index-link" href="/epi">all participants</a>' in page
    assert '<a id="rate-link" href="/">rating screen</a>' in page
    assert '<span id="packing" class="epi-packing" data-waiting="false">' in page
    body = ok(client.get(f"/api/epi/{FULL}"), shapes.EpiStreamOut)
    # the answer of step 1's server on the same directory, with only the media URLs changed
    alone = Packs(epi_run.run_dir / EPI_DIR / FULL).shape().model_dump(mode="json")
    for session in alone["sessions"]:
        session["media_url"] = session["media_url"].replace("/epi-media/", f"/epi-media/{FULL}/")
    assert body == alone
    assert [s["session_uid"] for s in body["sessions"]] == list(epi_run.sessions[FULL])
    assert body["label"] == LABELS[FULL]


def test_a_half_packed_participant_shows_what_is_packed_and_asks_again(
    client: TestClient, epi_run: EpiRun
) -> None:
    body = ok(client.get(f"/api/epi/{HALF}"), shapes.EpiStreamOut)
    assert [s["session_uid"] for s in body["sessions"]] == list(epi_run.shown[HALF])
    assert len(epi_run.sessions[HALF]) == 2 and len(epi_run.shown[HALF]) == 1
    page = ok(client.get(f"/epi/{HALF}", headers=PAGE)).text
    assert 'data-poll="5"' in page and 'data-waiting="true"' in page
    assert '<span id="packing" class="epi-packing" data-waiting="true">' in page
    # the second session is packed behind the app: the next answer holds it
    second = epi_run.sessions[HALF][1]
    source = epi_run.run_dir / EPI_DIR / FULL / epi_run.sessions[FULL][1]
    target = epi_run.run_dir / EPI_DIR / HALF / second
    shutil.copytree(source, target)
    for path in target.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        for old, new in zip(
            epi_run.runs[epi_run.sessions[FULL][1]], epi_run.runs[second], strict=True
        ):
            text = text.replace(old, new)
        path.write_text(text.replace(epi_run.sessions[FULL][1], second), encoding="utf-8")
    for old, new in zip(epi_run.runs[epi_run.sessions[FULL][1]], epi_run.runs[second], strict=True):
        if (target / old).is_dir():
            (target / old).rename(target / new)
    grown = ok(client.get(f"/api/epi/{HALF}"), shapes.EpiStreamOut)
    assert [s["session_uid"] for s in grown["sessions"]] == list(epi_run.sessions[HALF])
    # what the page held is as it was
    assert grown["sessions"][0] == body["sessions"][0]


def test_a_participant_without_any_pack_yet_has_an_empty_stream(
    client: TestClient, epi_run: EpiRun
) -> None:
    shutil.rmtree(epi_run.run_dir / EPI_DIR / HALF)
    body = ok(client.get(f"/api/epi/{HALF}"), shapes.EpiStreamOut)
    assert body["sessions"] == [] and body["label"] == LABELS[HALF]
    page = ok(client.get(f"/epi/{HALF}", headers=PAGE)).text
    assert 'data-poll="5"' in page
    index = ok(client.get("/api/epi"), shapes.EpiIndexOut)
    half = next(p for p in index["participants"] if p["participant_uid"] == HALF)
    assert half["can_open"] is False and not any(s["shown"] for s in half["sessions"])


def test_a_participant_that_the_catalog_does_not_have_is_404(client: TestClient) -> None:
    for uid in ("0" * 16, FULL.upper(), FULL[:-1], "stream", "../" + FULL):
        for path in (f"/epi/{uid}", f"/api/epi/{uid}", f"/epi-media/{uid}/{'1' * 16}/x.bin"):
            answer = client.get(path)
            assert answer.status_code == 404, path
    answer = client.get(f"/api/epi/{'0' * 16}")
    assert answer.json() == {
        "errors": [{"field": "participant_uid", "message": "there is no such participant"}]
    }


# --- the files of the packs ---------------------------------------------------------------------


def listed(body: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    """``(media_url, file, sha256, token)`` of every file the answer lists."""
    out = []
    for session in body["sessions"]:
        url = session["media_url"]
        for grid in session["grids"]:
            for entry in (grid["contours"] or {}).values():
                out.append((url, entry["file"], entry["sha256"], grid["token"]))
        for run in session["runs"]:
            if run["pack"] is None:
                continue
            for stream in run["pack"]["streams"].values():
                for still in ("ref", "mean", "sd"):
                    out.append(
                        (url, stream[still]["file"], stream[still]["sha256"], run["pack"]["token"])
                    )
                for chunk in stream["chunks"]:
                    out.append((url, chunk["file"], chunk["sha256"], run["pack"]["token"]))
    return out


def test_every_listed_file_is_served_under_the_participant_with_its_cache_rule(
    client: TestClient, epi_run: EpiRun
) -> None:
    for participant in epi_run.participants:
        body = ok(client.get(f"/api/epi/{participant}"), shapes.EpiStreamOut)
        files = listed(body)
        assert len(files) > 50
        for url, file, sha, token in files:
            assert url.startswith(f"/epi-media/{participant}/")
            answer = client.get(url + file, params={"v": token})
            assert answer.status_code == 200, url + file
            assert hashlib.sha256(answer.content).hexdigest() == sha
            assert answer.headers["content-type"] == "application/octet-stream"
            assert "immutable" in answer.headers["cache-control"]
        url, file, _, token = files[0]
        for asked in ({}, {"v": "0" * 16}, {"v": token[:-1]}):
            answer = client.get(url + file, params=asked)
            assert answer.status_code == 200 and answer.headers["cache-control"] == "no-cache"


def test_the_json_files_and_every_unlisted_name_are_404(
    client: TestClient, epi_run: EpiRun
) -> None:
    a, b = epi_run.sessions[FULL]
    run_a = epi_run.packed[a][0]
    run_b = epi_run.packed[b][0]
    half_a = epi_run.sessions[HALF][0]
    half_b = epi_run.sessions[HALF][1]
    not_packed = next(r for r in epi_run.runs[b] if r not in epi_run.packed[b])
    base = f"/epi-media/{FULL}"
    for path in (
        f"{base}/{a}/{SESSION_JSON}",
        f"{base}/{a}/{run_a}/{PACK_JSON}",
        f"{base}/{a}/{run_a}/{TRACES_JSON}",
        f"{base}/../{STREAM_JSON}",
        f"{base}/{a}/../{STREAM_JSON}",
        f"{base}/{a}/%2e%2e/{STREAM_JSON}",
        f"{base}/{a}/{run_a}/raw/../../{SESSION_JSON}",
        f"/epi-media/{FULL}/{STREAM_JSON}",
        f"/epi/{FULL}/{STREAM_JSON}",
        f"{base}/{a}/{run_a}/raw/c9999.bin",
        f"{base}/{a}/{run_a}/raw/c0000.bin.tmp",
        f"{base}/{a}/{run_a}/raw/c\u0660\u0660\u0660\u0660.bin",
        f"{base}/{a}/{run_b}/raw/ref.bin",  # a run of the other session
        f"{base}/{b}/{not_packed}/raw/ref.bin",  # a run that is not packed
        f"{base}/{half_a}/{run_a}/raw/ref.bin",  # a session of another participant
        f"/epi-media/{HALF}/{a}/{run_a}/raw/ref.bin",  # and the other way round
        f"/epi-media/{HALF}/{half_b}/contours/x/axial.bin",  # listed, not packed yet
        f"{base}/{a.upper()}/{run_a}/raw/ref.bin",
        f"{base}/{a}/{run_a}/raw/",
        f"{base}/{a}",
        f"{base}/",
        "/epi-media/",
        "/epi/",
        "/api/epi/",
        "/static/js/epi/missing.js",
        f"/media/{FULL}/{a}/{run_a}/raw/ref.bin",
    ):
        answer = client.get(path)
        assert answer.status_code == 404, path
        assert b'"format"' not in answer.content and b"epi-pack" not in answer.content


def test_a_link_that_leads_out_of_the_session_is_404(
    client: TestClient, epi_run: EpiRun, tmp_path: Path
) -> None:
    session = epi_run.sessions[FULL][0]
    run = epi_run.packed[session][0]
    folder = epi_run.run_dir / EPI_DIR / FULL / session
    info = read_pack(folder, run)
    assert info is not None
    name = info.streams["raw"].chunks[0].file
    outside = tmp_path / "elsewhere.bin"
    outside.write_bytes((folder / name).read_bytes())
    (folder / name).unlink()
    (folder / name).symlink_to(outside)
    assert client.get(f"/epi-media/{FULL}/{session}/{name}").status_code == 404
    # a session directory of another participant, linked in, is not this participant's
    other = epi_run.sessions[HALF][0]
    (epi_run.run_dir / EPI_DIR / FULL / other).symlink_to(epi_run.run_dir / EPI_DIR / HALF / other)
    half = read_pack(epi_run.run_dir / EPI_DIR / HALF / other, epi_run.packed[other][0])
    assert half is not None
    assert (
        client.get(f"/epi-media/{FULL}/{other}/{half.streams['raw'].ref.file}").status_code == 404
    )


def test_another_host_and_another_origin_are_refused_on_the_epi_routes(
    client: TestClient, epi_run: EpiRun
) -> None:
    for path in paths(epi_run).values():
        assert client.get(path, headers={"Host": "evil.example"}).status_code == 403
        assert client.get(path, headers={"Origin": "http://evil.example"}).status_code in (
            200,
            403,
        )
        if path.startswith("/api"):
            assert client.get(path, headers={"Origin": "http://evil.example"}).status_code == 403
            assert client.get(path, headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
        answer = client.get(path, follow_redirects=False)
        assert answer.status_code == 200 and "location" not in answer.headers
        assert client.post(path, json={}, headers={"Origin": BASE}).status_code == 405


# --- the rating routes stay blind -----------------------------------------------------------------


def test_nothing_the_rating_routes_answer_holds_anything_of_the_epi_interface(
    client: TestClient, epi_run: EpiRun
) -> None:
    secret: set[str] = {*epi_run.participants, *LABELS.values(), *epi_run.names}
    for participant in epi_run.participants:
        for session in epi_run.sessions[participant]:
            secret.add(session)
            secret.update(epi_run.runs[session])
    numbers: set[str] = set()
    for session in epi_run.shown[FULL]:
        info = read_info(epi_run.run_dir / EPI_DIR / FULL / session / SESSION_JSON, SessionInfo)
        assert info is not None
        for run in info.runs:
            if run.frame is not None:
                numbers.update(repr(x) for x in (run.frame.mean_mm, run.frame.max_mm))
            if run.reason is None:
                traces = read_info(
                    epi_run.run_dir / EPI_DIR / FULL / session / run.run / TRACES_JSON, TracesInfo
                )
                assert traces is not None
                numbers.update(repr(x) for x in traces.fd[1:4] if x is not None)
                numbers.update(repr(x) for x in traces.dvars[1:4] if x is not None)
    secret = {s for s in secret if len(s) >= 4}
    # a number of a trace or a frame check, written as JSON writes it, of 4 digits or more
    numbers = {n for n in numbers if len(n.replace(".", "").replace("-", "").strip("0")) >= 4}
    assert len(numbers) > 5
    answers = []
    session = ok(client.get("/api/session")).json()
    assert session.pop("epi_url") == "/epi"
    answers.append(json.dumps(session))
    first = ok(client.get("/api/next")).json()
    answers.append(json.dumps(first))
    skipped = client.post(
        "/api/skip",
        json={
            "assignment_id": first["assignment"]["assignment_id"],
            "client_ts": "2026-11-14T12:00:00.000Z",
            "ms_on_subject": 700,
        },
    )
    answers.append(ok(skipped).text)
    second = skipped.json()["assignment"]
    rated = client.post("/api/rating", json=rating_body(client, second["assignment_id"], 1))
    answers.append(ok(rated).text)
    answers.append(ok(client.get("/api/back")).text)
    answers.append(ok(client.get("/")).text)
    text = "\n".join(answers)
    for word in sorted(secret):
        assert word not in text, word
    for number in sorted(numbers):
        assert not re.search(rf"(?<![\d.]){re.escape(number)}(?!\d)", text), number
    for word in ("participant", "session_uid", "run_uid", "stream", "trace", "framewise"):
        assert word not in text, word
    # the one link of the rating page, which names no subject
    page = ok(client.get("/")).text
    assert '<a id="epi-link" class="epi-link" href="/epi" hidden>EPI movies</a>' in page


def test_a_session_that_stream_json_lists_without_its_session_json_is_waited_for(
    client: TestClient, epi_run: EpiRun
) -> None:
    # every pack of FULL is recorded as done; the pack phase is writing its second session
    # again, which stream.json lists and whose session.json is not there yet
    page = ok(client.get(f"/epi/{FULL}", headers=PAGE)).text
    assert 'data-poll="30"' in page
    (epi_run.run_dir / EPI_DIR / FULL / epi_run.sessions[FULL][1] / SESSION_JSON).unlink()
    page = ok(client.get(f"/epi/{FULL}", headers=PAGE)).text
    assert 'data-poll="5"' in page
    body = ok(client.get(f"/api/epi/{FULL}"), shapes.EpiStreamOut)
    assert [s["session_uid"] for s in body["sessions"]] == [epi_run.sessions[FULL][0]]


def test_the_page_asks_again_fast_while_packs_wait_and_slowly_when_none_do(
    client: TestClient, epi_run: EpiRun, packed: PackedStream
) -> None:
    """Finding F4 of the review of step 2: a page opened when nothing waited never asked
    again (data-poll 0). Now it is told both intervals and the interval of now, and the index
    it reads after each answer says which applies; the pack-only server of step 1, whose
    --out is written whole, still does not ask."""
    from surfcensus.app import server
    from surfcensus.app.epi import create_epi_app

    assert (server.EPI_POLL_S, server.EPI_POLL_IDLE_S) == (5, 30)
    for participant, waiting in ((FULL, False), (HALF, True)):
        page = ok(client.get(f"/epi/{participant}", headers=PAGE)).text
        now = server.EPI_POLL_S if waiting else server.EPI_POLL_IDLE_S
        assert (
            f'data-poll="{now}" data-poll-waiting="5" data-poll-idle="30" '
            f'data-waiting="{str(waiting).lower()}" data-index-api="/api/epi" '
            f'data-participant="{participant}"'
        ) in page
        # what the page reads to choose: the index entry of the participant
        index = ok(client.get("/api/epi"), shapes.EpiIndexOut)
        me = next(p for p in index["participants"] if p["participant_uid"] == participant)
        says = me["n_pending"] > 0 or any(
            not s["shown"] and s["n_packable"] for s in me["sessions"]
        )
        assert says is waiting
    alone = TestClient(create_epi_app(packed.out), base_url="http://127.0.0.1")
    page = ok(alone.get("/", headers=PAGE)).text
    assert 'data-poll="0"' in page and "data-poll-idle" not in page
    assert 'id="packing"' not in page


def test_every_element_the_index_script_looks_up_is_in_its_page() -> None:
    from surfcensus.app.server import STATIC_DIR, TEMPLATES_DIR

    source = (STATIC_DIR / "js" / "epi" / "index.js").read_text(encoding="utf-8")
    wanted = set(re.findall(r"\$\('([\w-]+)'\)", source))
    page = (TEMPLATES_DIR / "epi_index.html").read_text(encoding="utf-8")
    assert wanted and wanted <= set(re.findall(r'\bid="([\w-]+)"', page)), wanted
    # it writes nothing and asks nothing of another host
    for word in ("localStorage", "sessionStorage", "method:", "http:", "https:"):
        assert word not in source, word
