"""The start page of the app (``/`` and ``/api/home``): one page that leads to both
interfaces, the surfaces and the EPI movies, and says no more than the rating screen does."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from epi_packs import PackedStream, make_stream  # pyrefly: ignore[missing-import]
from epi_run_fixture import LABELS, EpiRun, add_epi  # pyrefly: ignore[missing-import]
from fastapi.testclient import TestClient

from surfcensus.app import schemas
from surfcensus.app.server import create_app
from surfcensus.config import EpiConfig, load_config, with_changes, write_config

BASE = "http://127.0.0.1"


@pytest.fixture(scope="module")
def packed(tmp_path_factory: pytest.TempPathFactory) -> PackedStream:
    return make_stream(tmp_path_factory.mktemp("home"), n_volumes=10)


@pytest.fixture
def epi_run(app_run: Path, packed: PackedStream, small_run_uids: dict[str, str]) -> EpiRun:
    return add_epi(app_run, packed, subject_uid=small_run_uids["sub-001"])


@contextmanager
def serving(run_dir: Path, **options: Any) -> Iterator[TestClient]:
    with TestClient(create_app(run_dir, **options), base_url=BASE) as client:
        yield client


def set_app(run_dir: Path, **changes: object) -> None:
    config = load_config(run_dir)
    write_config(run_dir, with_changes(config, app=config.app.model_dump() | changes))


def home(client: TestClient) -> dict[str, Any]:
    answer = client.get("/api/home")
    assert answer.status_code == 200, answer.text[:400]
    body = answer.json()
    schemas.HomeOut.model_validate(body)
    return body


def rate_everything(client: TestClient) -> int:
    version = client.get("/api/session").json()["protocol"]["version"]
    n = 0
    while True:
        answer = client.get("/api/next").json()
        if answer["status"] != "assignment":
            return n
        body = {
            "assignment_id": answer["assignment"]["assignment_id"],
            "protocol_version": version,
            "values": {"A": 0, "B": 0, "C": 0},
            "unratable": None,
            "other": False,
            "note": "",
            "marks": [],
            "client_ts": f"2026-11-24T12:00:{n:02d}.000Z",
            "ms_on_subject": 1000,
        }
        assert client.post("/api/rating", json=body).status_code == 200
        n += 1


def assignments(run_dir: Path) -> int:
    with sqlite3.connect(run_dir / "app.sqlite") as db:
        return db.execute("SELECT count(*) FROM assignments").fetchone()[0]


def test_the_start_page_leads_to_both_interfaces_and_names_no_subject(
    app_run: Path, small_run_uids: dict[str, str]
) -> None:
    with serving(app_run) as client:
        page = client.get("/")
        assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
        assert 'src="/static/js/home.js?v=' in page.text
        assert 'href="/rate"' in page.text and 'href="/epi"' in page.text
        for label, uid in small_run_uids.items():
            assert label not in page.text and uid not in page.text
        # the rating screen moved to /rate and is what it was
        rate = client.get("/rate")
        assert rate.status_code == 200 and 'src="/static/js/rate.js?v=' in rate.text


def test_without_functional_data_the_start_page_says_so_and_shows_the_surfaces(
    app_run: Path,
) -> None:
    with serving(app_run) as client:
        body = home(client)
        assert body["functional"] == {
            "offered": False,
            "url": None,
            "subjects_url": None,
            "closed": None,
            "n_participants": 0,
            "n_runs": 0,
            "n_packable": 0,
            "n_packed": 0,
            "n_pending": 0,
            "n_failed": 0,
        }
        surfaces = body["surfaces"]
        assert surfaces["url"] == "/rate" and surfaces["review_url"] == "/review"
        assert body["rater"] == "local" and body["blind"] is False
        # the numbers of the progress line of the rating screen
        progress = client.get("/api/next").json()["progress"]
        after = home(client)["surfaces"]
        assert (after["n_subjects"], after["rendered"], after["done_total"]) == (
            progress["n_subjects"],
            progress["rendered"],
            progress["done_total"],
        )
        assert after["remaining"] == progress["remaining"] == 3
        rate_everything(client)
        done = home(client)["surfaces"]
        assert (done["done_total"], done["remaining"]) == (3, 0)


def test_asking_the_start_page_opens_no_assignment(app_run: Path) -> None:
    with serving(app_run) as client:
        for _ in range(3):
            home(client)
            client.get("/")
        assert assignments(app_run) == 0
        client.get("/api/next")
        assert assignments(app_run) == 1


def test_with_functional_data_the_start_page_counts_what_the_epi_index_counts(
    epi_run: EpiRun,
) -> None:
    with serving(epi_run.run_dir) as client:
        functional = home(client)["functional"]
        index = client.get("/api/epi").json()
        participants = index["participants"]
        assert functional["offered"] is True
        assert (functional["url"], functional["subjects_url"], functional["closed"]) == (
            "/epi",
            "/subjects",
            None,
        )
        assert functional["n_participants"] == len(participants) > 0
        for name in ("n_runs", "n_packable", "n_packed", "n_pending", "n_failed"):
            assert functional[name] == sum(p[name] for p in participants), name
        assert functional["n_packed"] > 0
        # packs wait in the fixture, so the page asks again
        assert home(client)["poll_s"] == (5.0 if index["n_pending"] else None)


def test_a_blind_run_closes_the_epi_card_and_the_review_until_everything_is_rated(
    epi_run: EpiRun,
) -> None:
    set_app(epi_run.run_dir, blind=True)
    with serving(epi_run.run_dir) as client:
        body = home(client)
        assert body["blind"] is True
        assert body["surfaces"]["review_url"] is None
        functional = body["functional"]
        assert functional["offered"] is True and functional["closed"]
        assert functional["url"] is None and functional["subjects_url"] is None
        assert all(functional[name] == 0 for name in functional if name.startswith("n_"))
        # nothing of the functional side reaches a rater who is blind to it
        text = client.get("/api/home").text
        for word in (*LABELS.values(), *epi_run.sessions, "participant_uid", "/epi/"):
            assert word not in text, word
        assert not re.search(r"\b[0-9a-f]{16}\b", text)
        rate_everything(client)
        opened = home(client)
        assert opened["surfaces"]["review_url"] == "/review"
        assert opened["functional"]["url"] == "/epi" and opened["functional"]["closed"] is None
        assert opened["functional"]["n_participants"] > 0


def test_the_epi_interface_turned_off_is_not_offered(epi_run: EpiRun) -> None:
    config = load_config(epi_run.run_dir)
    write_config(epi_run.run_dir, with_changes(config, epi=EpiConfig(enabled=False).model_dump()))
    with serving(epi_run.run_dir) as client:
        assert home(client)["functional"]["offered"] is False


def test_every_page_links_back_to_the_start_page(epi_run: EpiRun) -> None:
    with serving(epi_run.run_dir) as client:
        for path in ("/rate", "/review", "/epi", "/subjects"):
            page = client.get(path, headers={"Accept": "text/html"})
            assert page.status_code == 200, path
            assert '<a id="home-link" href="/">start</a>' in page.text, path
