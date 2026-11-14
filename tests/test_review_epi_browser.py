"""The whole path of step 2 of the EPI interface in a real browser (work package E2.Z; marked
``browser``, not part of the default run or of CI): ``surfcensus review`` as a process on one
folder with both kinds of data (``synth_func.write_review_folder``), the rating screen, its link
to the EPI index, a participant's stream made by the real phases playing through a run
boundary and a session boundary, and back to the rating screen with the draft kept; no request
to another host and an empty console.

The tab and its checks are those of ``tests/test_epi_run_browser.py``.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from browser_launch import launch_browser  # pyrefly: ignore[missing-import]
from synth_func import write_review_folder  # pyrefly: ignore[missing-import]
from test_epi_run_browser import Tab, no_horizontal_scroll, pressed, rating_screen  # pyrefly: ignore[missing-import]

sync_api = pytest.importorskip("playwright.sync_api")

pytestmark = pytest.mark.browser


@pytest.fixture(scope="module")
def browser() -> Iterator[Any]:
    with sync_api.sync_playwright() as playwright:
        started = launch_browser(playwright)
        yield started
        started.close()


@pytest.fixture
def reviewing(tmp_path: Path) -> Iterator[tuple[str, subprocess.Popen[str]]]:
    folder = write_review_folder(tmp_path / "data")
    command = [sys.executable, "-c", "from surfcensus.cli import app; app()", "review"]
    command += [str(folder.raw_root), "--no-browser", "--port", "0", "--jobs", "1"]
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ | {"PYTHONUNBUFFERED": "1"},
    )
    assert process.stderr is not None
    said, url = "", None
    while "All EPI packs are made" not in said:
        line = process.stderr.readline()
        assert line, f"the command ended early:\n{said}"
        said += line
        found = re.match(r"Rating app: (http://127\.0\.0\.1:\d+/)", line)
        url = found.group(1) if found else url
    assert url is not None and f"EPI movies: {url}epi" in said
    yield url, process
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=60) == 0
    process.stderr.close()


def test_from_the_rating_screen_to_a_stream_made_by_review_and_back(
    browser: Any, reviewing: tuple[str, subprocess.Popen[str]]
) -> None:
    url, _ = reviewing
    tab = Tab(browser, url, width=1400)
    page = tab.page
    page.goto(url)
    uid = rating_screen(tab)
    page.keyboard.press("2")
    page.keyboard.press("n")
    page.fill("#note", "before the EPI movies")
    page.keyboard.press("Escape")
    link = page.locator("#epi-link")
    assert link.is_visible() and link.get_attribute("href") == "/epi"
    link.click()
    page.wait_for_url(f"{url}epi")
    page.wait_for_selector("#epi-index[data-state='ready']")
    text = page.inner_text("#participants")
    assert "sub-01" in text and "sub-02" in text
    assert not page.is_visible("#index-packing")  # every pack is made
    no_horizontal_scroll(tab)
    first = page.eval_on_selector_all(
        "#participants tbody tr", "rows => rows.map(r => r.dataset.participant)"
    )[0]
    page.click(f"tr[data-participant='{first}'] a")
    page.wait_for_url(f"{url}epi/{first}")
    page.wait_for_function("() => window.__epi && window.__epi.shown", timeout=20000)
    page.wait_for_function("() => window.__epi.allLoaded()", timeout=30000)
    assert page.inner_text("#subject") == "sub-01"
    timeline = page.evaluate("window.__epi.timeline")
    segments, total = timeline["segments"], timeline["total"]
    # sub-01: two sessions, 4 and 3 packed runs, one stream
    assert len(segments) == 7 and len({s["session"] for s in segments}) == 2
    boundary = next(s["first"] for s in segments if s["session"] == 1)
    start, count = boundary - 3, 6
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
    assert boundary in gs and all(e["nz"] > 0 for e in log)
    asked = page.evaluate(
        "performance.getEntriesByType('resource').map(e => new URL(e.name).pathname)"
    )
    media = [p for p in asked if p.startswith("/epi-media/")]
    assert media and all(p.startswith(f"/epi-media/{first}/") for p in media)

    page.click("#rate-link")
    page.wait_for_url(url)
    assert rating_screen(tab) == uid
    assert pressed(tab) == [["A", "1"]]
    assert page.input_value("#note") == "before the EPI movies"
    tab.close()
