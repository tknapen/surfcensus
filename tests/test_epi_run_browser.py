"""The EPI pages of a run directory in a real browser (work package E2.D; marked ``browser``,
not part of the default run or of CI): from the rating screen by its one link to the index of
the participants, to a participant's stream and back, with the draft of the open rating kept;
the stream playing through its boundaries under the run's URLs; a participant whose packs are
still being made; no request to another host and an empty console, at 1400 and 400 px.

The run is ``app_run`` with the EPI interface of ``tests/epi_run_fixture.py``; the page of the
stream is driven through the hooks of ``tests/test_epi_browser.py`` (``window.__epi``)."""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from browser_launch import launch_browser  # pyrefly: ignore[missing-import]
from epi_packs import PackedStream, make_stream  # pyrefly: ignore[missing-import]
from epi_run_fixture import FULL, HALF, LABELS, EpiRun, add_epi  # pyrefly: ignore[missing-import]

from surfcensus.app.server import start_server
from surfcensus.store import EPI_DIR

sync_api = pytest.importorskip("playwright.sync_api")

pytestmark = pytest.mark.browser

_SHOTS = os.environ.get("SURFCENSUS_SHOTS", "")
SHOTS = Path(_SHOTS) if _SHOTS else None
"""Where the tests leave screenshots for a reader, when ``SURFCENSUS_SHOTS`` names a place
(outside the repository); none otherwise."""
SEEN_TOUR = "try { localStorage.setItem('surfcensus.tour.seen:local', '1'); } catch (e) {}"


@pytest.fixture(scope="module")
def packed(tmp_path_factory: pytest.TempPathFactory) -> PackedStream:
    return make_stream(tmp_path_factory.mktemp("epirunbrowser"), n_volumes=12)


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

    def __init__(self, browser: Any, url: str, *, width: int, height: int = 900) -> None:
        self.url = url
        self.context = browser.new_context(viewport={"width": width, "height": height})
        self.context.add_init_script(SEEN_TOUR)
        self.page = self.context.new_page()
        self.problems: list[str] = []
        self.elsewhere: list[str] = []
        page = self.page
        page.on("console", lambda m: self.problems.append(f"{m.type}: {m.text}"))
        page.on("pageerror", lambda error: self.problems.append(str(error)))
        # a request that the browser cancels because the page is left (a prefetched tile) is
        # no problem; any other failure is
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
            self.page.screenshot(path=str(SHOTS / f"{name}.png"))

    def close(self) -> None:
        assert self.problems == []
        assert self.elsewhere == []
        self.context.close()


def no_horizontal_scroll(tab: Tab) -> None:
    assert tab.page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")


def rating_screen(tab: Tab) -> str:
    """Wait for a subject on the rating screen; its uid."""
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


@pytest.mark.parametrize("width", [1400, 400])
def test_from_the_rating_screen_to_a_stream_and_back_with_the_draft_kept(
    browser: Any, server: Any, epi_run: EpiRun, width: int
) -> None:
    tab = Tab(browser, server.url, width=width)
    page = tab.page
    page.goto(server.url)
    uid = rating_screen(tab)
    page.keyboard.press("2")  # A minor: a draft of the open subject
    page.keyboard.press("n")
    page.fill("#note", "seen before the EPI movies")
    page.keyboard.press("Escape")
    assert pressed(tab) == [["A", "1"]]
    link = page.locator("#epi-link")
    assert link.is_visible() and link.get_attribute("href") == "/epi"
    # the link sits in the progress line, below the rating bar, and moves nothing of it
    bar = page.locator("#rating").bounding_box()
    box = link.bounding_box()
    assert bar is not None and box is not None and box["y"] >= bar["y"] + bar["height"] - 1
    no_horizontal_scroll(tab)
    tab.shot(f"rate_{width}")

    link.click()
    page.wait_for_url(f"{server.url}epi")
    page.wait_for_selector("#epi-index[data-state='ready']")
    rows = page.eval_on_selector_all(
        "#participants tbody tr", "rows => rows.map(r => r.dataset.participant)"
    )
    assert rows == [FULL, HALF]
    text = page.inner_text("#participants")
    assert LABELS[FULL] in text and LABELS[HALF] in text and "ses-1" in text
    assert page.is_visible("#index-packing")  # HALF waits for its second session
    assert "still being made" in page.inner_text("#index-packing")
    no_horizontal_scroll(tab)
    tab.shot(f"index_{width}")

    page.click(f"tr[data-participant='{FULL}'] a")
    page.wait_for_url(f"{server.url}epi/{FULL}")
    page.wait_for_function("() => window.__epi && window.__epi.shown", timeout=20000)
    page.wait_for_function("() => window.__epi.allLoaded()", timeout=30000)
    assert page.inner_text("#subject") == LABELS[FULL]
    sessions = page.evaluate("window.__epi.stream.sessions.map(s => s.media_url)")
    assert sessions == [f"/epi-media/{FULL}/{s}/" for s in epi_run.sessions[FULL]]
    assert page.is_visible("#index-link") and not page.is_visible("#packing")
    no_horizontal_scroll(tab)
    tab.shot(f"stream_{width}")

    page.click("#rate-link")
    page.wait_for_url(server.url)
    assert rating_screen(tab) == uid
    assert pressed(tab) == [["A", "1"]]
    assert page.input_value("#note") == "seen before the EPI movies"
    tab.close()


def test_the_stream_of_a_run_plays_through_its_boundaries_under_the_runs_urls(
    browser: Any, server: Any, epi_run: EpiRun
) -> None:
    tab = Tab(browser, server.url, width=1400)
    page = tab.page
    page.goto(f"{server.url}epi/{FULL}")
    page.wait_for_function("() => window.__epi && window.__epi.shown", timeout=20000)
    page.wait_for_function("() => window.__epi.allLoaded()", timeout=30000)
    timeline = page.evaluate("window.__epi.timeline")
    firsts = [seg["first"] for seg in timeline["segments"]]
    total = timeline["total"]
    sessions = {seg["session"] for seg in timeline["segments"]}
    assert len(firsts) >= 6 and len(sessions) == 2
    session_start = next(s["first"] for s in timeline["segments"] if s["session"] == 1)
    start = session_start - 4
    count = 8
    page.evaluate("(c) => window.__epi.set(c)", {"g": start, "rate": 30, "view": "side"})
    page.evaluate("window.__epi.page.log.length = 0")
    page.evaluate("(c) => window.__epi.set(c)", {"playing": True})
    page.wait_for_function(
        "(n) => window.__epi.page.log.filter((e) => e.reason === 'play').length >= n",
        arg=count,
        timeout=60000,
    )
    page.evaluate("(c) => window.__epi.set(c)", {"playing": False})
    log = [e for e in page.evaluate("window.__epi.page.log") if e["reason"] == "play"]
    gs = [e["g"] for e in log][:count]
    assert gs == [(start + 1 + k) % total for k in range(count)]
    assert session_start in gs  # the session boundary was crossed
    assert all(e["nz"] > 0 for e in log)
    # every file came from the run's EPI media of this participant
    asked = page.evaluate(
        "performance.getEntriesByType('resource').map(e => new URL(e.name).pathname)"
    )
    media = [p for p in asked if p.startswith("/epi-media/")]
    assert media and all(p.startswith(f"/epi-media/{FULL}/") for p in media)
    assert not any(p.startswith("/api/epi/stream") for p in asked)
    tab.close()


def test_a_stream_that_grows_under_an_open_page_says_so_and_plays_on(
    browser: Any, server: Any, epi_run: EpiRun
) -> None:
    tab = Tab(browser, server.url, width=1400)
    page = tab.page
    page.goto(f"{server.url}epi/{HALF}")
    page.wait_for_function("() => window.__epi && window.__epi.shown", timeout=20000)
    assert page.is_visible("#packing")
    assert page.evaluate("window.__epi.stream.sessions.length") == 1
    tab.shot("stream_half")
    # the pack phase writes the second session behind the app
    second = epi_run.sessions[HALF][1]
    full_second = epi_run.sessions[FULL][1]
    source = epi_run.run_dir / EPI_DIR / FULL / full_second
    target = epi_run.run_dir / EPI_DIR / HALF / ("." + second)
    shutil.copytree(source, target)
    pairs = list(zip(epi_run.runs[full_second], epi_run.runs[second], strict=True))
    for path in target.rglob("*.json"):
        text = path.read_text(encoding="utf-8").replace(full_second, second)
        for old, new in pairs:
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")
    for old, new in pairs:
        if (target / old).is_dir():
            (target / old).rename(target / new)
    target.rename(epi_run.run_dir / EPI_DIR / HALF / second)
    page.evaluate("(c) => window.__epi.set(c)", {"playing": True})
    page.wait_for_function("() => window.__epi.texts().notice !== ''", timeout=20000)
    notice = page.evaluate("window.__epi.texts().notice")
    assert "More of this stream has been packed" in notice and "plays on" in notice
    # what was on screen plays on: the store was not stopped
    before = page.evaluate("window.__epi.state.g")
    page.wait_for_function("(g) => window.__epi.state.g !== g", arg=before, timeout=10000)
    tab.shot("stream_half_grown")
    page.click("#notice-reload")
    page.wait_for_function(
        "() => window.__epi && window.__epi.stream && window.__epi.stream.sessions.length === 2",
        timeout=20000,
    )
    tab.close()


def test_a_closed_epi_page_is_a_page_with_a_way_back(
    browser: Any, epi_run: EpiRun, tmp_path: Path
) -> None:
    from surfcensus.config import load_config, with_changes, write_config

    config = load_config(epi_run.run_dir)
    write_config(
        epi_run.run_dir, with_changes(config, app=config.app.model_dump() | {"blind": True})
    )
    handle = start_server(epi_run.run_dir, port=0)
    try:
        tab = Tab(browser, handle.url, width=1400)
        page = tab.page
        page.goto(handle.url)
        rating_screen(tab)
        assert page.locator("#epi-link").is_hidden()
        tab.problems.clear()
        answer = page.goto(f"{handle.url}epi")
        assert answer is not None and answer.status == 403
        assert "the EPI movies open when none is left" in page.inner_text("body")
        tab.shot("closed")
        tab.problems = [p for p in tab.problems if not p.startswith("403:")]
        page.click("text=Back to the rating screen")
        rating_screen(tab)
        tab.problems = [p for p in tab.problems if "403" not in p]
        tab.close()
    finally:
        handle.stop()


PLACES = (
    "Array.from(document.querySelectorAll('#views .epi-panel, #timeline, #run-strip'))"
    ".map((e) => { const b = e.getBoundingClientRect();"
    " return [b.x, b.y, b.width, b.height]; })"
)
"""The boxes of the panels, the timeline and the strip, in CSS px."""


@pytest.mark.parametrize("participant", [FULL, HALF])
def test_a_run_packed_later_into_the_session_on_screen_is_noticed_and_plays_on(
    browser: Any,
    server: Any,
    epi_run: EpiRun,
    participant: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding F4 of the review of step 2. ``FULL`` has no pack to be made when its page
    opens, so the page asks at the slow interval (before, it never asked again); ``HALF``
    waits for its second session and asks at the fast one. In both, a run of the session on
    screen is packed while the page plays: the page says that the stream grew, not that the
    packs were made again, plays on without a jump, and after the reload it offers, the
    timeline holds the run in its place."""
    from surfcensus.app import server as app_server

    monkeypatch.setattr(app_server, "EPI_POLL_S", 1)
    monkeypatch.setattr(app_server, "EPI_POLL_IDLE_S", 2)
    session = epi_run.shown[participant][-1]
    runs = epi_run.packed[session]
    held = runs[1]  # a run in the middle of the session: packed "later"
    where = epi_run.run_dir / EPI_DIR / participant / session
    (where / held).rename(where / f".later-{held}")
    tab = Tab(browser, server.url, width=1400)
    page = tab.page
    page.goto(f"{server.url}epi/{participant}")
    page.wait_for_function("() => window.__epi && window.__epi.shown", timeout=20000)
    page.wait_for_function("() => window.__epi.allLoaded()", timeout=30000)
    main = page.locator("main#views")
    assert main.get_attribute("data-poll") == ("1" if participant == HALF else "2")
    assert page.is_visible("#packing") is (participant == HALF)
    before = page.evaluate("window.__epi.timeline")
    uids = [r["run_uid"] for s in page.evaluate("window.__epi.stream.sessions") for r in s["runs"]]
    assert held in uids  # listed, without its pack
    first = next(
        seg["first"]
        for seg in before["segments"]
        if seg["session"] == len(epi_run.shown[participant]) - 1
    )
    page.evaluate("(c) => window.__epi.set(c)", {"g": first, "rate": 12, "view": "side"})
    boxes = page.evaluate(PLACES)
    page.evaluate("window.__epi.page.log.length = 0")
    page.evaluate("(c) => window.__epi.set(c)", {"playing": True})
    page.wait_for_function("() => window.__epi.page.log.length >= 3", timeout=10000)
    asked = page.evaluate("window.__epi.poll.asks")
    # the pack phase of another process puts the run's pack in place
    at = page.evaluate("performance.now()")
    (where / f".later-{held}").rename(where / held)
    page.wait_for_function("() => window.__epi.texts().notice !== ''", timeout=20000)
    notice = page.evaluate("window.__epi.texts().notice")
    assert "More of this stream has been packed" in notice and "plays on" in notice
    assert "made again" not in notice
    took = (page.evaluate("window.__epi.page.grownAt") - at) / 1000
    assert took <= (1 if participant == HALF else 2) + 5, took
    assert page.evaluate("window.__epi.poll.asks") > asked
    # it plays on, every volume in order, nothing moved; it does not reload by itself
    count = page.evaluate("window.__epi.page.log.length")
    page.wait_for_function("(n) => window.__epi.page.log.length >= n + 6", arg=count, timeout=20000)
    page.evaluate("(c) => window.__epi.set(c)", {"playing": False})
    gs = [e["g"] for e in page.evaluate("window.__epi.page.log") if e["reason"] == "play"]
    assert gs == [(first + 1 + k) % before["total"] for k in range(len(gs))], gs
    assert page.evaluate("window.__epi.timeline") == before
    assert page.evaluate(PLACES) == boxes
    tab.shot(f"grown_in_session_{LABELS[participant]}")
    page.click("#notice-reload")
    page.wait_for_function(
        "(t) => window.__epi && window.__epi.timeline && window.__epi.timeline.total > t",
        arg=before["total"],
        timeout=20000,
    )
    page.wait_for_function("() => window.__epi.shown", timeout=20000)
    after = page.evaluate("window.__epi.timeline")
    sessions = page.evaluate("window.__epi.stream.sessions")
    # the run is in its place in acquisition order, and the timeline adds up
    added = [
        seg
        for seg in after["segments"]
        if sessions[seg["session"]]["runs"][seg["run"]]["run_uid"] == held
    ]
    assert len(added) == 1 and len(after["segments"]) == len(before["segments"]) + 1
    upto = 0
    for seg in after["segments"]:
        assert seg["first"] == upto
        upto += seg["n"]
    assert upto == after["total"] == before["total"] + added[0]["n"]
    order = [
        sessions[s["session"]]["runs"][s["run"]]["acquisition_order"]
        for s in after["segments"]
        if s["session"] == added[0]["session"]
    ]
    assert order == sorted(order)
    assert page.evaluate("window.__epi.texts().notice") == ""
    tab.close()
