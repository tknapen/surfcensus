"""The subject pages of a run directory in a real browser (work package E3.A; marked
``browser``, not part of the default run or of CI): from the rating screen by its link to the
page of the subject on screen, from there into the EPI page at one run and back with the draft
of the open rating kept; from the review screen to a subject page and back to the review of
that subject; the list of every subject; the EPI page asked for a run that is not in its
stream. At 1400 by 900 and 400 by 800, with no horizontal scroll of a page, no request to
another host and an empty console.

The run is ``app_run`` with the EPI interface of ``tests/epi_run_fixture.py``. Screenshots go
where ``SURFCENSUS_SHOTS`` names (outside the repository), none otherwise."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from browser_launch import launch_browser  # pyrefly: ignore[missing-import]
from epi_packs import PackedStream, make_stream  # pyrefly: ignore[missing-import]
from epi_run_fixture import FULL, HALF, LABELS, EpiRun, add_epi  # pyrefly: ignore[missing-import]

from surfcensus.app.server import start_server

sync_api = pytest.importorskip("playwright.sync_api")

pytestmark = pytest.mark.browser

_SHOTS = os.environ.get("SURFCENSUS_SHOTS", "")
SHOTS = Path(_SHOTS) if _SHOTS else None
SEEN_TOUR = "try { localStorage.setItem('surfcensus.tour.seen:local', '1'); } catch (e) {}"
SIZES = [(1400, 900), (400, 800)]


@pytest.fixture(scope="module")
def packed(tmp_path_factory: pytest.TempPathFactory) -> PackedStream:
    return make_stream(tmp_path_factory.mktemp("subjectsbrowser"), n_volumes=12)


@pytest.fixture(scope="module")
def browser() -> Iterator[Any]:
    with sync_api.sync_playwright() as playwright:
        started = launch_browser(playwright)
        yield started
        started.close()


@pytest.fixture
def epi_run(app_run: Path, packed: PackedStream, small_run_uids: dict[str, str]) -> EpiRun:
    return add_epi(app_run, packed, subject_uid=small_run_uids["sub-001"])


@pytest.fixture
def server(epi_run: EpiRun) -> Iterator[Any]:
    handle = start_server(epi_run.run_dir, port=0)
    yield handle
    handle.stop()


class Tab:
    """A browser tab on the app, with what went wrong on it and what it asked for."""

    def __init__(self, browser: Any, url: str, *, width: int, height: int) -> None:
        self.url = url
        self.width = width
        self.context = browser.new_context(
            viewport={"width": width, "height": height}, device_scale_factor=1
        )
        self.context.add_init_script(SEEN_TOUR)
        self.page = self.context.new_page()
        self.problems: list[str] = []
        self.elsewhere: list[str] = []
        page = self.page
        page.on("console", lambda m: self.problems.append(f"{m.type}: {m.text}"))
        page.on("pageerror", lambda error: self.problems.append(str(error)))
        page.on(
            "requestfailed",
            lambda r: (
                None
                if "ERR_ABORTED" in (r.failure or "")
                else self.problems.append(f"failed: {r.url}: {r.failure}")
            ),
        )
        page.on(
            "response",
            lambda r: self.problems.append(f"{r.status}: {r.url}") if r.status >= 400 else None,
        )
        page.on(
            "request",
            lambda r: None if r.url.startswith((url, "data:")) else self.elsewhere.append(r.url),
        )

    def shot(self, name: str) -> None:
        if SHOTS is not None:
            SHOTS.mkdir(parents=True, exist_ok=True)
            self.page.screenshot(path=str(SHOTS / f"{name}_{self.width}.png"), full_page=True)

    def close(self) -> None:
        assert self.problems == []
        assert self.elsewhere == []
        self.context.close()


def no_horizontal_scroll(tab: Tab) -> None:
    assert tab.page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
    main = tab.page.locator("main").first
    assert main.evaluate("m => m.scrollWidth <= m.clientWidth + 1")


def rating_screen(tab: Tab) -> str:
    tab.page.wait_for_selector("#rate[data-state='rating']")
    tab.page.wait_for_selector("#viewer[data-ready='true']")
    return tab.page.evaluate(
        "document.querySelector('.tile-gray').src.match(/media\\/([0-9a-f]{16})/)[1]"
    )


def pressed(tab: Tab) -> list[list[str]]:
    return tab.page.eval_on_selector_all(
        ".level[aria-pressed='true']",
        "levels => levels.map(l => [l.dataset.criterion, l.dataset.value])",
    )


def run_on_screen(tab: Tab) -> str:
    return tab.page.evaluate(
        "(() => { const e = window.__epi; const t = e.timeline; const g = e.state.g;"
        " const seg = t.segments.find((s) => s.first <= g && g < s.first + s.n);"
        " return e.stream.sessions[seg.session].runs[seg.run].run_uid; })()"
    )


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_from_the_rating_screen_to_a_run_of_the_subject_and_back_with_the_draft_kept(
    browser: Any,
    server: Any,
    epi_run: EpiRun,
    small_run_uids: dict[str, str],
    width: int,
    height: int,
) -> None:
    tab = Tab(browser, server.url, width=width, height=height)
    page = tab.page
    page.goto(server.url + "rate")
    uid = rating_screen(tab)
    assert uid == small_run_uids["sub-001"]  # the first of the queue, which has a session
    page.keyboard.press("2")
    page.keyboard.press("n")
    page.fill("#note", "seen before the subject page")
    page.keyboard.press("Escape")
    link = page.locator("#subject-link")
    assert link.is_visible() and link.get_attribute("href") == f"/subject/{uid}"
    # it sits in the progress line, below the rating bar
    bar = page.locator("#rating").bounding_box()
    box = link.bounding_box()
    assert bar is not None and box is not None and box["y"] >= bar["y"] + bar["height"] - 1
    no_horizontal_scroll(tab)
    tab.shot("rate")

    link.click()
    page.wait_for_url(f"{server.url}subject/{uid}")
    page.wait_for_selector("#subjects-page[data-state='ready']")
    assert page.inner_text("#subject-name") == "sub-001"
    sections = page.eval_on_selector_all(".subject-session", "s => s.map((e) => e.dataset.session)")
    session = epi_run.sessions[FULL][0]
    assert sections == [session]
    text = page.inner_text("#subjects-page")
    for words in (
        "raw data",
        "preprocessing",
        "anchor",
        "frame check",
        "recorded",
        "summary report",
    ):
        assert words in text, words
    runs = page.eval_on_selector_all(
        f".subject-session[data-session='{session}'] tbody tr",
        "rows => rows.map((r) => [r.dataset.run, r.dataset.state])",
    )
    assert [r for r, _ in runs] == list(epi_run.runs[session])
    assert {s for _, s in runs} == {"packed"}
    # the design review of 2026-11-24: the run name is the link into the EPI page at the run,
    # "(recorded)" is said once, in the heading of the states, a flagged run has a badge and an
    # amber edge, and the column of the corrections is left out when no run has its own
    info = page.evaluate(f"fetch('/api/subjects/{uid}').then((r) => r.json())")
    for run in info["sessions"][0]["runs"]:
        row = f".subject-session[data-session='{session}'] tr[data-run='{run['run_uid']}']"
        assert page.inner_text(f"{row} .run-name") == (run["name"] or run["run_uid"])
        if run["epi_url"]:
            assert (page.get_attribute(f"{row} a.run-link", "href") or "").startswith(
                run["epi_url"]
            )
        flagged = bool(run["frame"] and run["frame"]["flagged"])
        assert page.locator(f"{row} .flag-badge").count() == int(flagged)
        assert ("flagged-row" in (page.get_attribute(row, "class") or "")) == flagged
    headings = page.eval_on_selector_all(
        f".subject-session[data-session='{session}'] thead th", "h => h.map((e) => e.textContent)"
    )
    assert "state (recorded)" in headings and "preprocessing applied" not in headings
    assert "open the EPI page at this run" not in text and "packed (recorded)" not in text
    assert page.locator("#review-link").get_attribute("href") == f"/review#subject={uid}"
    no_horizontal_scroll(tab)
    tab.shot("subject")

    target = epi_run.runs[session][1]  # not the first run of the stream
    page.click(f"tr[data-run='{target}'] a")
    page.wait_for_url(f"{server.url}epi/{FULL}#run={target}&subject={uid}")
    page.wait_for_function("() => window.__epi && window.__epi.shown", timeout=20000)
    assert page.evaluate("window.__epi.page.startRun") == target
    assert run_on_screen(tab) == target
    assert page.evaluate("window.__epi.state.g") > 0
    assert "not in the stream" not in page.evaluate("window.__epi.texts().status")
    assert page.is_visible("#subjects-link")
    # a link back to the subject page the run link came from (review of step 3, F5)
    back = page.locator("#subject-back")
    assert back.is_visible() and back.get_attribute("href") == f"/subject/{uid}"
    no_horizontal_scroll(tab)
    tab.shot("epi_at_run")

    page.click("#subjects-link")
    page.wait_for_url(f"{server.url}subjects")
    page.wait_for_selector("#subjects-page[data-state='ready']")
    keys = page.eval_on_selector_all("#subject-rows tbody tr", "r => r.map((e) => e.dataset.key)")
    labels = ("sub-000", "sub-001", "sub-002")
    assert keys == [*(small_run_uids[label] for label in labels), FULL, HALF]
    assert page.is_visible("#subjects-packing")  # HALF still waits for packs
    # each count under its heading, as the API gives it (review of step 3, F4)
    answer = page.evaluate("fetch('/api/subjects').then((r) => r.json())")
    fields = {"packed": "n_packed", "waiting": "n_pending", "failed": "n_failed"}
    headings = page.eval_on_selector_all("#subject-rows thead th", "h => h.map((e) => e.innerText)")
    for row in answer["subjects"]:
        cells = page.eval_on_selector_all(
            f"#subject-rows tr[data-key='{row['key']}'] td", "c => c.map((e) => e.innerText)"
        )
        for heading, field in fields.items():
            assert cells[headings.index(heading)] == str(row[field]), (row["key"], heading)
    half = next(r for r in answer["subjects"] if r["key"] == HALF)
    assert half["n_pending"] != half["n_failed"]
    no_horizontal_scroll(tab)
    tab.shot("subjects")

    page.click("#rate-link")
    page.wait_for_url(server.url + "rate")
    assert rating_screen(tab) == uid
    assert pressed(tab) == [["A", "1"]]
    assert page.input_value("#note") == "seen before the subject page"
    tab.close()


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_from_the_review_screen_to_a_subject_page_and_back_to_its_review(
    browser: Any, server: Any, small_run_uids: dict[str, str], width: int, height: int
) -> None:
    tab = Tab(browser, server.url, width=width, height=height)
    page = tab.page
    page.goto(f"{server.url}review")
    page.wait_for_selector("#review[data-state='open']")
    uid = small_run_uids["sub-001"]
    assert page.is_visible("#subjects-link")
    # and the EPI movies, as every other page links to them (review of step 3, F5)
    assert page.locator("#epi-index-link").is_visible()
    assert page.locator("#epi-index-link").get_attribute("href") == "/epi"
    cell = page.locator(f"#subjects tbody tr[data-subject-uid='{uid}'] a")
    assert cell.inner_text() == "1"
    cell.click()
    # the link opens the subject page and not the row of the review
    page.wait_for_url(f"{server.url}subject/{uid}")
    page.wait_for_selector("#subjects-page[data-state='ready']")
    no_horizontal_scroll(tab)
    page.click("#review-link")
    page.wait_for_url(f"{server.url}review#subject={uid}")
    page.wait_for_selector("#subject:not([hidden])")
    page.wait_for_function(
        f"() => document.querySelector(\"#subjects tr[data-subject-uid='{uid}']\")"
        ".getAttribute('aria-selected') === 'true'"
    )
    assert page.inner_text("#subject-title") == "sub-001"
    tab.shot("review_from_subject")
    # a subject without functional data: its page says so
    other = small_run_uids["sub-000"]
    page.goto(f"{server.url}subject/{other}")
    page.wait_for_selector("#subjects-page[data-state='ready']")
    assert "No functional runs" in page.inner_text("#surface")
    assert page.locator(".subject-session").count() == 0
    no_horizontal_scroll(tab)
    tab.shot("subject_without_runs")
    # a participant without a FreeSurfer subject has a page of its own, without a review link
    page.goto(f"{server.url}subject/{HALF}")
    page.wait_for_selector("#subjects-page[data-state='ready']")
    assert page.inner_text("#subject-name") == LABELS[HALF]
    assert not page.is_visible("#review-link")
    assert page.locator(".subject-session").count() == 2
    assert page.locator("td.state-failed").count() == 1
    no_horizontal_scroll(tab)
    tab.shot("participant")
    tab.close()


def test_the_epi_page_asked_for_a_run_that_is_not_in_its_stream_starts_at_the_beginning(
    browser: Any, server: Any
) -> None:
    tab = Tab(browser, server.url, width=1400, height=900)
    page = tab.page
    page.goto(f"{server.url}epi/{FULL}#run={'0' * 16}")
    page.wait_for_function("() => window.__epi && window.__epi.shown", timeout=20000)
    assert page.evaluate("window.__epi.page.startRun") is None
    assert not page.is_visible("#subject-back")  # opened without a subject page behind it
    assert page.evaluate("window.__epi.state.g") == 0
    page.wait_for_function(
        "() => window.__epi.texts().status.includes('the run asked for is not in the stream now')"
    )
    tab.close()
