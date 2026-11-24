"""The subject pages of a run directory (step 3 of docs/epi_interface.md, work package E3.A):
``/subjects``, ``/subject/<key>`` and their API, the two review columns, and the link of the
rating screen to the page of the subject on screen.

The blindness table of "Step 3: the plan (E3.0)", row by row, is a test each. The rows of
``/api/subjects`` are compared with the catalog and ``func.batch.recorded_states``: every
present session on exactly one row, a FreeSurfer subject without functional data with 0
sessions, a participant whose sessions have no FreeSurfer subject on a row of its own. The
rating routes answer nothing of any of it (decision D6).

The run directory is ``app_run`` (three rendered synthetic subjects) with the EPI interface of
``tests/epi_run_fixture.py`` written into it.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from epi_packs import PackedStream, make_stream  # pyrefly: ignore[missing-import]
from epi_run_fixture import (  # pyrefly: ignore[missing-import]
    CORRECTIONS,
    FULL,
    HALF,
    LABELS,
    EpiRun,
    add_epi,
    session_ids,
)
from fastapi.testclient import TestClient

from surfcensus.app import epi_schemas as shapes
from surfcensus.app import schemas
from surfcensus.app.server import create_app
from surfcensus.config import EpiConfig, load_config, with_changes, write_config
from surfcensus.func.batch import recorded_states
from surfcensus.func.model import corrections_info
from surfcensus.records import PackRow
from surfcensus.store import EPI_DIR, PublicCatalog

BASE = "http://127.0.0.1"
PAGE = {"Accept": "text/html,application/xhtml+xml"}
NO_ROUTE = {"errors": [{"field": "request", "message": "there is no such route"}]}
CLOSED = "you still have subjects to rate; the subject pages open when none is left"
NO_SUBJECT = {"errors": [{"field": "key", "message": "there is no such subject"}]}
SUBJECT_LINK = (
    '<a id="subject-link" class="epi-link" href="/subjects" hidden>'
    "functional runs of this subject</a>"
)


@pytest.fixture(scope="module")
def packed(tmp_path_factory: pytest.TempPathFactory) -> PackedStream:
    return make_stream(tmp_path_factory.mktemp("subjects"), n_volumes=10)


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


def ok(response: Any, shape: type | None = None) -> Any:
    assert response.status_code == 200, (response.status_code, response.text[:400])
    if shape is None:
        return response
    body = response.json()
    shape.model_validate(body)
    return body


def paths(key: str) -> dict[str, str]:
    """One concrete path of every subject route, by route."""
    return {
        "/subjects": "/subjects",
        "/subject/{key}": f"/subject/{key}",
        "/api/subjects": "/api/subjects",
        "/api/subjects/{key}": f"/api/subjects/{key}",
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
        "client_ts": f"2026-11-14T13:00:{at:02d}.000Z",
        "ms_on_subject": 1000,
    }


def rate_everything(client: TestClient) -> int:
    n = 0
    while True:
        answer = client.get("/api/next").json()
        if answer["status"] != "assignment":
            return n
        body = rating_body(client, answer["assignment"]["assignment_id"], n)
        assert client.post("/api/rating", json=body).status_code == 200
        n += 1


def test_the_four_routes_are_among_the_routes_the_server_answers() -> None:
    assert all(route in schemas.ROUTES for route in schemas.SUBJECT_ROUTES)
    routes = {(r.method, r.path): r for r in schemas.ROUTES}
    assert routes[("GET", "/api/subjects")].response is shapes.SubjectsOut
    assert routes[("GET", "/api/subjects/{key}")].response is shapes.SubjectPageOut


# --- the blindness table, row by row -------------------------------------------------------------


def opened(client: TestClient, key: str) -> None:
    for route, path in paths(key).items():
        page = "api" not in route
        answer = ok(client.get(path, headers=PAGE if page else {}))
        if page:
            assert answer.headers["content-type"].startswith("text/html")
            assert answer.headers["cache-control"] == "no-cache"
        csp = answer.headers["content-security-policy"]
        assert "default-src 'self'" in csp and "frame-ancestors 'none'" in csp
    assert client.get("/api/session").json()["epi_url"] == "/epi"
    columns = ok(client.get("/api/review"), schemas.ReviewOut)["columns"]
    assert columns[-2:] == ["func_sessions", "func_packed"]


def closed(client: TestClient, key: str, secrets: tuple[str, ...]) -> None:
    for route, path in paths(key).items():
        answer = client.get(path)
        assert answer.status_code == 403, path
        assert answer.json() == {"errors": [{"field": "request", "message": CLOSED}]}
        page = client.get(path, headers=PAGE)
        assert page.status_code == 403
        if "api" not in route:
            assert page.headers["content-type"].startswith("text/html")
            assert page.headers["cache-control"] == "no-store"
            assert CLOSED in page.text
            assert '<a href="/rate">Back to the rating screen</a>' in page.text
            assert "<script" not in page.text
            assert "<title>surfcensus: subjects</title>" in page.text
            for uid in (key, *secrets):
                assert uid not in page.text
        else:
            assert page.headers["content-type"].startswith("application/json")
    assert client.get("/api/session").json()["epi_url"] is None
    assert client.get("/api/review").status_code == 403


def test_local_not_blind_one_rater_opens_the_subject_pages(
    client: TestClient, small_run_uids: dict[str, str]
) -> None:
    opened(client, small_run_uids["sub-001"])
    # the link is in the page, hidden until an assignment is open; rate.js shows it
    assert SUBJECT_LINK in ok(client.get("/rate")).text


def test_blind_with_work_left_closes_them_and_everything_rated_opens_them(
    epi_run: EpiRun, small_run_uids: dict[str, str]
) -> None:
    set_app(epi_run.run_dir, blind=True)
    key = small_run_uids["sub-001"]
    with serving(epi_run.run_dir) as client:
        closed(client, key, (FULL, HALF, *small_run_uids.values()))
        assert rate_everything(client) == 3
        opened(client, key)


def test_a_second_rater_with_work_left_closes_them_and_done_opens_them(
    epi_run: EpiRun, small_run_uids: dict[str, str]
) -> None:
    key = small_run_uids["sub-001"]
    with serving(epi_run.run_dir, rater="ada") as client:
        opened(client, key)
        rate_everything(client)
    with serving(epi_run.run_dir, rater="ben") as client:
        closed(client, key, (FULL, HALF))
        rate_everything(client)
        opened(client, key)


def no_route(client: TestClient, key: str) -> None:
    unknown = client.get("/no-such-route")
    for path in [*paths(key).values(), f"/subject/{'0' * 16}", f"/api/subjects/{FULL}"]:
        for headers in ({}, PAGE):
            answer = client.get(path, headers=headers)
            assert answer.status_code == 404, path
            assert answer.json() == NO_ROUTE
            assert answer.headers["cache-control"] == unknown.headers["cache-control"]
        other_methods_as_no_route(client, path)
    assert client.get("/api/session").json()["epi_url"] is None


def other_methods_as_no_route(client: TestClient, path: str) -> None:
    """POST, HEAD, OPTIONS, PUT and DELETE answer as on a path that is no route (they gave
    405 "the route does not take that method": review of step 3, F3)."""
    for method in ("POST", "HEAD", "OPTIONS", "PUT", "DELETE"):
        body = {"json": {}} if method in ("POST", "PUT") else {}
        unknown = client.request(method, "/no-such-route", **body)
        answer = client.request(method, path, **body)
        assert unknown.status_code == 404
        assert answer.status_code == 404, (method, path)
        if method != "HEAD":
            assert answer.json() == unknown.json() == NO_ROUTE, (method, path)


def test_a_census_run_answers_the_subject_routes_as_no_route(
    epi_run: EpiRun, small_run_uids: dict[str, str]
) -> None:
    connection = sqlite3.connect(epi_run.run_dir / "catalog.sqlite")
    with connection:
        connection.execute("UPDATE meta SET value = 'census' WHERE key = 'mode'")
    connection.close()
    with serving(epi_run.run_dir) as client:
        no_route(client, small_run_uids["sub-001"])
        review = client.get("/api/review")
        if review.status_code == 200:
            assert "func_sessions" not in review.json()["columns"]


def test_a_run_whose_epi_interface_is_off_answers_them_as_no_route(
    epi_run: EpiRun, small_run_uids: dict[str, str]
) -> None:
    write_config(
        epi_run.run_dir,
        with_changes(load_config(epi_run.run_dir), epi=EpiConfig(enabled=False).model_dump()),
    )
    with serving(epi_run.run_dir) as client:
        no_route(client, small_run_uids["sub-001"])
        columns = ok(client.get("/api/review"), schemas.ReviewOut)["columns"]
        assert "func_sessions" not in columns and "func_packed" not in columns


def test_a_run_without_functional_data_answers_them_as_no_route_and_reviews_as_before(
    app_run: Path, small_run_uids: dict[str, str]
) -> None:
    with serving(app_run) as client:
        before = ok(client.get("/api/review"), schemas.ReviewOut)
        no_route(client, small_run_uids["sub-001"])
        assert ok(client.get("/api/review")).json() == before
    assert "func_sessions" not in before["columns"] and "func_packed" not in before["columns"]
    assert before["columns"][-2:] == ["n_ratings", "n_marks"]


# --- the rows of /api/subjects ------------------------------------------------------------------


def catalog_view(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    with PublicCatalog.open(run_dir) as public:
        sessions = [s for s in public.func_sessions().to_dict("records") if s["present"]]
        runs = public.func_runs().to_dict("records")
        states = recorded_states(public, EpiConfig().pack_hash())
    runs_of: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        if run["present"]:
            runs_of.setdefault(run["session_uid"], []).append(run | {"rec": states[run["run_uid"]]})
    return sessions, runs_of


def expected_counts(own: list[dict[str, Any]], runs_of: dict[str, list[dict[str, Any]]]) -> dict:
    runs = [r for s in own for r in runs_of.get(s["session_uid"], [])]
    usable = [r for r in runs if r["packable"] and r["selected"]]
    packed = sum(r["rec"].state == "packed" for r in usable)
    failed = sum(r["rec"].state == "failed" for r in usable)
    return {
        "n_sessions": len(own),
        "n_runs": len(runs),
        "n_packable": len(usable),
        "n_packed": packed,
        "n_failed": failed,
        "n_pending": len(usable) - packed - failed,
        "n_flagged": sum(bool(r["flagged"]) for r in runs if r["flagged"] is not None),
    }


def test_every_present_session_is_on_exactly_one_row(
    app_run: Path, packed: PackedStream, small_run_uids: dict[str, str]
) -> None:
    ids = session_ids(packed)
    # FULL: its first session under sub-001, its second under sub-000; HALF: its first under
    # sub-002, its second without a FreeSurfer subject
    epi = add_epi(
        app_run,
        packed,
        subject_uid=small_run_uids["sub-001"],
        subject_uids={
            ids[FULL][1]: small_run_uids["sub-000"],
            ids[HALF][0]: small_run_uids["sub-002"],
        },
    )
    with serving(epi.run_dir) as client:
        listing = ok(client.get("/api/subjects"), shapes.SubjectsOut)
        pages = {
            row["key"]: ok(client.get(f"/api/subjects/{row['key']}"), shapes.SubjectPageOut)
            for row in listing["subjects"]
        }
    sessions, runs_of = catalog_view(epi.run_dir)
    rows = listing["subjects"]
    # the subject rows by label, then the participant rows by label
    assert [(r["kind"], r["label"]) for r in rows] == [
        ("subject", "sub-000"),
        ("subject", "sub-001"),
        ("subject", "sub-002"),
        ("participant", LABELS[HALF]),
    ]
    by_subject = {
        uid: [s for s in sessions if s["subject_uid"] == uid] for uid in small_run_uids.values()
    }
    for row in rows:
        if row["kind"] == "subject":
            own = by_subject[row["key"]]
            assert row["subject_uid"] == row["key"] and row["surface"] == "rendered"
        else:
            own = [
                s for s in sessions if s["participant_uid"] == row["key"] and not s["subject_uid"]
            ]
            assert row["subject_uid"] is None and row["surface"] == "none"
            assert row["participant_uid"] == row["key"] == HALF
        assert {k: row[k] for k in expected_counts(own, runs_of)} == expected_counts(own, runs_of)
        assert row["url"] == f"/subject/{row['key']}"
        participant = own[0]["participant_uid"]
        assert row["participant_uid"] == participant
        assert row["epi_url"] == f"/epi/{participant}"
    placed = [s["session_uid"] for page in pages.values() for s in page["sessions"]]
    assert sorted(placed) == sorted(s["session_uid"] for s in sessions)
    assert len(placed) == len(set(placed)) == 4
    assert listing["n_pending"] == sum(r["n_pending"] for r in rows) > 0


def test_a_subject_without_functional_data_has_a_row_with_no_session(
    client: TestClient, epi_run: EpiRun, small_run_uids: dict[str, str]
) -> None:
    listing = ok(client.get("/api/subjects"), shapes.SubjectsOut)
    rows = {r["key"]: r for r in listing["subjects"]}
    for label in ("sub-000", "sub-002"):
        row = rows[small_run_uids[label]]
        assert row["n_sessions"] == row["n_runs"] == 0
        assert row["participant_uid"] is None and row["epi_url"] is None
        page = ok(client.get(f"/api/subjects/{row['key']}"), shapes.SubjectPageOut)
        assert page["sessions"] == []
        assert page["review_url"] == f"/review#subject={row['key']}"
    # FULL's second session and both of HALF have no FreeSurfer subject: participant rows
    assert [r["key"] for r in listing["subjects"] if r["kind"] == "participant"] == [FULL, HALF]
    assert rows[FULL]["n_sessions"] == 1 and rows[HALF]["n_sessions"] == 2


def test_the_surface_state_follows_the_ratings_and_the_renders(
    client: TestClient, small_run_uids: dict[str, str]
) -> None:
    def surfaces() -> dict[str, str]:
        rows = ok(client.get("/api/subjects"), shapes.SubjectsOut)["subjects"]
        return {r["key"]: r["surface"] for r in rows}

    assert {surfaces()[uid] for uid in small_run_uids.values()} == {"rendered"}
    answer = client.get("/api/next").json()["assignment"]
    body = rating_body(client, answer["assignment_id"], 1)
    body["values"] = {}
    body["unratable"] = client.get("/api/session").json()["protocol"]["unratable_reasons"][0]
    assert client.post("/api/rating", json=body).status_code == 200
    assert surfaces()[answer["subject_uid"]] == "unratable"
    rows = ok(client.get("/api/subjects"))
    row = next(r for r in rows.json()["subjects"] if r["key"] == answer["subject_uid"])
    assert row["surface_reason"] == body["unratable"]
    second = client.get("/api/next").json()["assignment"]
    rated = client.post("/api/rating", json=rating_body(client, second["assignment_id"], 2))
    assert rated.status_code == 200
    assert surfaces()[second["subject_uid"]] == "rated"


# --- the page of one subject --------------------------------------------------------------------


def test_the_page_of_a_subject_has_every_session_and_run_with_what_the_catalog_says(
    client: TestClient, epi_run: EpiRun, small_run_uids: dict[str, str]
) -> None:
    key = small_run_uids["sub-001"]
    page = ok(client.get(f"/api/subjects/{key}"), shapes.SubjectPageOut)
    assert page["review_url"] == f"/review#subject={key}"
    assert (page["rate_url"], page["index_url"], page["epi_index_url"]) == (
        "/rate",
        "/subjects",
        "/epi",
    )
    sessions, runs_of = catalog_view(epi_run.run_dir)
    (listed,) = page["sessions"]
    (session,) = [s for s in sessions if s["subject_uid"] == key]
    assert listed["session_uid"] == session["session_uid"] == epi_run.sessions[FULL][0]
    assert listed["participant"] == LABELS[FULL] and listed["participant_uid"] == FULL
    assert listed["stream_rule"] == "session label" and listed["stream_order"] == 0
    # FULL has its second session on a row of its own: the stream still has two (F1)
    assert listed["stream_sessions"] == len(epi_run.sessions[FULL]) == 2
    assert listed["shown"] is True
    runs = runs_of[session["session_uid"]]
    assert [r["run_uid"] for r in listed["runs"]] == [
        r["run_uid"] for r in sorted(runs, key=lambda r: (r["acquisition_order"], r["run_uid"]))
    ]
    anchors = [r for r in listed["runs"] if r["anchor"]]
    assert len(anchors) == 1 and anchors[0]["run_uid"] == listed["anchor_run_uid"]
    assert listed["anchor_name"] == anchors[0]["name"]
    assert listed["anchor_packed"] is (anchors[0]["state"] == "packed")
    wanted = corrections_info(CORRECTIONS).model_dump(mode="json")
    for out, run in zip(
        listed["runs"], sorted(runs, key=lambda r: r["acquisition_order"]), strict=True
    ):
        assert out["state"] == run["rec"].state and out["reason"] == run["rec"].reason
        assert out["name"] == run["name"] and out["n_volumes"] == run["n_volumes"]
        assert out["corrections"] == wanted
        assert out["raw_affine"] == run["raw_affine"]
        if run["flagged"] is not None:
            assert out["frame"]["flagged"] == run["flagged"]
            assert out["frame"]["max_mm"] == run["frame_max_mm"]
        if out["state"] == "packed":
            assert out["epi_url"] == f"/epi/{FULL}#run={out['run_uid']}"
        else:
            assert out["epi_url"] is None
        if run["packable"]:
            assert out["grid_shape"] is not None and len(out["voxel_mm"]) == 3


def test_a_run_of_a_session_the_stream_does_not_show_has_no_link(
    client: TestClient, epi_run: EpiRun
) -> None:
    page = ok(client.get(f"/api/subjects/{HALF}"), shapes.SubjectPageOut)
    assert page["review_url"] is None and page["subject"]["kind"] == "participant"
    first, second = page["sessions"]
    assert [first["session_uid"], second["session_uid"]] == list(epi_run.sessions[HALF])
    assert first["shown"] is True and second["shown"] is False
    assert any(r["epi_url"] for r in first["runs"])
    assert all(r["epi_url"] is None for r in second["runs"])
    failed = [r for r in second["runs"] if r["state"] == "failed"]
    assert [r["run_uid"] for r in failed] == list(epi_run.failed)
    assert failed[0]["reason"] == "pack failure" and failed[0]["detail"] == "OSError"
    assert {r["state"] for r in second["runs"]} >= {"failed", "pending"}


@pytest.mark.parametrize(
    "key",
    [
        "F0F0F0F0F0F0F0F0",  # not lower case
        "f0f0f0f0f0f0f0f",  # 15 characters
        "f0f0f0f0f0f0f0f0a",
        "g" * 16,
        "0" * 16,  # no row
        "..%2f..%2fcatalog",
    ],
)
def test_a_key_that_is_no_row_is_404(client: TestClient, key: str) -> None:
    for path in (f"/api/subjects/{key}", f"/subject/{key}"):
        for headers in ({}, PAGE):
            answer = client.get(path, headers=headers)
            assert answer.status_code == 404, (path, answer.text)
            if answer.headers["content-type"].startswith("application/json"):
                assert answer.json() in (NO_SUBJECT, NO_ROUTE)


def test_a_session_and_a_run_are_no_keys(client: TestClient, epi_run: EpiRun) -> None:
    for key in (epi_run.sessions[FULL][0], epi_run.runs[epi_run.sessions[FULL][0]][0]):
        assert client.get(f"/api/subjects/{key}").json() == NO_SUBJECT
        assert client.get(f"/subject/{key}", headers=PAGE).status_code == 404


def test_the_pages_are_their_templates_with_their_addresses(
    client: TestClient, small_run_uids: dict[str, str]
) -> None:
    listing = ok(client.get("/subjects", headers=PAGE)).text
    assert 'data-api="/api/subjects"' in listing and 'data-poll="5"' in listing
    assert '<script type="module" src="/static/js/subjects.js?v=' in listing
    key = small_run_uids["sub-001"]
    one = ok(client.get(f"/subject/{key}", headers=PAGE)).text
    assert f'data-key="{key}"' in one and f'data-api="/api/subjects/{key}"' in one
    for text in (listing, one):
        for href in ('href="/"', 'href="/review"', 'href="/epi"'):
            assert href in text
        assert "raw data" in text and "preprocessing" in text
    assert 'href="/subjects"' in one
    # the EPI pages link to the subjects page
    assert (
        '<a id="subjects-link" href="/subjects">subjects</a>'
        in ok(client.get("/epi", headers=PAGE)).text
    )
    assert (
        '<a id="subjects-link" href="/subjects">subjects</a>'
        in ok(client.get(f"/epi/{FULL}", headers=PAGE)).text
    )


def test_the_page_of_epi_session_serve_has_no_link_to_subjects(packed: PackedStream) -> None:
    from surfcensus.app.epi import create_epi_app

    page = ok(TestClient(create_epi_app(packed.out), base_url=BASE).get("/", headers=PAGE)).text
    assert "subjects-link" not in page and "/subjects" not in page


# --- the review columns -------------------------------------------------------------------------


def test_the_review_columns_count_the_sessions_and_packed_runs_of_each_subject(
    client: TestClient, epi_run: EpiRun, small_run_uids: dict[str, str]
) -> None:
    review = ok(client.get("/api/review"), schemas.ReviewOut)
    assert review["columns"][-4:] == ["n_ratings", "n_marks", "func_sessions", "func_packed"]
    got = {r["subject_uid"]: (r["func_sessions"], r["func_packed"]) for r in review["rows"]}
    assert got[small_run_uids["sub-001"]] == (1, len(epi_run.packed[epi_run.sessions[FULL][0]]))
    assert got[small_run_uids["sub-000"]] == got[small_run_uids["sub-002"]] == (0, 0)
    # sorted and filtered like any other column
    by = ok(client.get("/api/review", params={"sort": "func_sessions", "order": "desc"}))
    assert by.json()["rows"][0]["subject_uid"] == small_run_uids["sub-001"]
    only = ok(client.get("/api/review", params={"filter": "func_sessions:1"})).json()
    assert [r["subject_uid"] for r in only["rows"]] == [small_run_uids["sub-001"]]


# --- the rating routes stay blind ---------------------------------------------------------------


def test_the_rating_routes_answer_nothing_of_the_subject_pages(
    client: TestClient, epi_run: EpiRun, small_run_uids: dict[str, str]
) -> None:
    listing = ok(client.get("/api/subjects")).json()
    words = {
        "func_sessions",
        "func_packed",
        "n_packed",
        "n_sessions",
        "surface_reason",
        "stream_rule",
        "acquisition",
        "/subject/",
        "/api/subjects",
        "functional",
        *LABELS.values(),
        *epi_run.names,
        FULL,
        HALF,
    }
    for participant in epi_run.participants:
        for session in epi_run.sessions[participant]:
            words.add(session)
            words.update(epi_run.runs[session])
    words.update(r["url"] for r in listing["subjects"])
    words = {w for w in words if len(w) >= 4}
    answers = []
    session = ok(client.get("/api/session")).json()
    assert set(session) == set(schemas.SessionOut.model_fields)
    answers.append(json.dumps(session))
    first = ok(client.get("/api/next")).json()
    assert set(first["assignment"]) == set(schemas.AssignmentOut.model_fields)
    answers.append(json.dumps(first))
    skipped = client.post(
        "/api/skip",
        json={
            "assignment_id": first["assignment"]["assignment_id"],
            "client_ts": "2026-11-14T13:30:00.000Z",
            "ms_on_subject": 700,
        },
    )
    answers.append(ok(skipped).text)
    second = skipped.json()["assignment"]
    rated = client.post("/api/rating", json=rating_body(client, second["assignment_id"], 9))
    answers.append(ok(rated).text)
    answers.append(ok(client.get("/api/back")).text)
    events = client.post(
        "/api/events",
        json={
            "assignment_id": second["assignment_id"],
            "events": [{"ts": "x", "type": "play", "payload": {}}],
        },
    )
    answers.append(ok(events).text)
    text = "\n".join(answers)
    for word in sorted(words):
        assert word not in text, word
    # the rating page holds the link and nothing of a subject: the href is a path of the
    # route table, which rate.js replaces with the subject's page while one is open
    page = ok(client.get("/rate")).text
    assert SUBJECT_LINK in page
    assert not re.search(r"\b[0-9a-f]{16}\b", page)
    for word in ("/subject/", "func_sessions", *LABELS.values()):
        assert word not in page


def test_nothing_is_written_into_the_packs_or_the_catalog_by_the_pages(
    client: TestClient, epi_run: EpiRun, small_run_uids: dict[str, str]
) -> None:
    before = {
        p: p.stat().st_mtime_ns for p in (epi_run.run_dir / EPI_DIR).rglob("*") if p.is_file()
    }
    catalog = (epi_run.run_dir / "catalog.sqlite").read_bytes()
    for key in (*small_run_uids.values(), FULL, HALF):
        client.get(f"/api/subjects/{key}")
        client.get(f"/subject/{key}", headers=PAGE)
    client.get("/api/subjects")
    after = {p: p.stat().st_mtime_ns for p in (epi_run.run_dir / EPI_DIR).rglob("*") if p.is_file()}
    assert after == before
    assert (epi_run.run_dir / "catalog.sqlite").read_bytes() == catalog


def test_a_render_that_fails_while_the_app_runs_is_render_failed_until_the_queue_moves(
    client: TestClient, epi_run: EpiRun, small_run_uids: dict[str, str]
) -> None:
    """When "render failed" is seen (review of step 3, F7): the server gives a failed render
    its unratable label when it starts and when the queue next moves (``sync_unratable``).
    A render that fails behind the running app is "render failed" on the subjects page until
    then, and "unratable" with the reason of the render after it."""
    key = sorted(small_run_uids.items())[-1][1]

    def row() -> dict[str, Any]:
        rows = ok(client.get("/api/subjects"), shapes.SubjectsOut)["subjects"]
        return next(r for r in rows if r["key"] == key)

    assert row()["surface"] == "rendered"
    connection = sqlite3.connect(epi_run.run_dir / "catalog.sqlite")
    with connection:
        connection.execute(
            "UPDATE renders SET status = 'failed', reason = 'render failure', "
            "detail = 'unreadable input (T1.mgz: truncated)' WHERE subject_uid = ?",
            (key,),
        )
    connection.close()
    assert (row()["surface"], row()["surface_reason"]) == ("render failed", "render failure")
    assert client.get("/api/next").status_code == 200
    assert (row()["surface"], row()["surface_reason"]) == ("unratable", "render failure")


# --- what broken copies of step 3 got past (review of step 3, F4) -------------------------------


def test_the_counts_and_links_on_a_fixture_where_they_differ(
    app_run: Path, packed: PackedStream, small_run_uids: dict[str, str]
) -> None:
    """A fixture on which the counts that the review's broken copies confused are not equal:
    sub-001 has a packed session and HALF's second session (one failed run, the others
    pending, not in the stream); HALF's first session names a FreeSurfer subject that the run
    does not list; and one run of HALF's second session is recorded as packed, although the
    stream does not hold its session yet."""
    ids = session_ids(packed)
    gone = small_run_uids["sub-002"]  # made absent below: a FreeSurfer subject that left
    epi = add_epi(
        app_run,
        packed,
        subject_uid=small_run_uids["sub-001"],
        subject_uids={ids[HALF][1]: small_run_uids["sub-001"], ids[HALF][0]: gone},
    )
    second = epi.sessions[HALF][1]
    with PublicCatalog.open(epi.run_dir) as public:
        packable = {r["run_uid"] for r in public.func_runs().to_dict("records") if r["packable"]}
    waiting = [r for r in epi.runs[second] if r in packable and r not in epi.failed]
    assert len(waiting) >= 2
    connection = sqlite3.connect(epi.run_dir / "catalog.sqlite")
    with connection:
        connection.execute("UPDATE subjects SET present = 0 WHERE subject_uid = ?", (gone,))
    connection.close()
    # a run of a session the stream does not hold, recorded as packed
    with PublicCatalog.open(epi.run_dir, readonly=False) as public:
        public.record_packs(
            [
                PackRow(
                    run_uid=waiting[-1],
                    session_uid=second,
                    config_hash=EpiConfig().pack_hash(),
                    input_signature="0" * 16,
                    status="done",
                    pack_key="1" * 16,
                    path=f"{EPI_DIR}/{HALF}/{second}/{waiting[-1]}",
                )
            ]
        )
    with serving(epi.run_dir) as client:
        listing = ok(client.get("/api/subjects"), shapes.SubjectsOut)
        rows = {r["key"]: r for r in listing["subjects"]}
        mine = ok(client.get(f"/api/subjects/{small_run_uids['sub-001']}"), shapes.SubjectPageOut)
        half = ok(client.get(f"/api/subjects/{HALF}"), shapes.SubjectPageOut)
        review = ok(client.get("/api/review"), schemas.ReviewOut)
    sessions, runs_of = catalog_view(epi.run_dir)

    # a session whose FreeSurfer subject is not present is on its participant's row
    assert [s["session_uid"] for s in half["sessions"]] == [epi.sessions[HALF][0]]
    assert rows[HALF]["n_sessions"] == 1 and gone not in rows

    # the counts of sub-001: packed, failed and pending differ, and none is the number of runs
    own = [s for s in sessions if s["subject_uid"] == small_run_uids["sub-001"]]
    counts = expected_counts(own, runs_of)
    row = rows[small_run_uids["sub-001"]]
    assert {k: row[k] for k in counts} == counts
    assert counts["n_failed"] == 1 and counts["n_pending"] > 0
    assert len({counts["n_packed"], counts["n_pending"], counts["n_runs"]}) == 3
    func = {r["subject_uid"]: (r["func_sessions"], r["func_packed"]) for r in review["rows"]}
    assert func[small_run_uids["sub-001"]] == (2, counts["n_packed"])

    # a packed run of a session that the stream does not show has no link into it, and the
    # anchor of a session whose anchor is not packed is said not to be
    (listed,) = [s for s in mine["sessions"] if s["session_uid"] == second]
    assert listed["shown"] is False
    states = {r["run_uid"]: (r["state"], r["epi_url"]) for r in listed["runs"]}
    assert states[waiting[-1]] == ("packed", None)
    anchor = next(r for r in listed["runs"] if r["anchor"])
    assert anchor["state"] != "packed" and listed["anchor_packed"] is False
