"""The random order of play of the EPI page (D48) and its surface control, in a real browser on
the pack-only server (marked ``browser``, not part of the default run or of CI), on the
synthetic stream of ``tests/test_epi_browser.py``: two sessions, eight packed runs of 24
volumes, two grids.

What is checked: with the random order on, playing at 12 and 60 volumes per second paints
exactly the order in force over one loop, every volume once, without a stall or a blank frame,
for both scopes; at 300 per second the painted volumes are a subsequence of the order; the
canvas still holds the decoded values of the volume painted; the key ``x``, the checkbox, the
scope and ``Shift`` + ``x``; switching off or on while playing goes on from the volume on
screen; the arrow keys, Home, End and ``r`` stay in time; nothing on the page moves when the
order is switched or while it plays; the surface control by its label and its key, in every
view and on the stills."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from browser_launch import launch_browser  # pyrefly: ignore[missing-import]
from epi_packs import PackedStream, make_stream  # pyrefly: ignore[missing-import]
from test_epi_browser import (  # pyrefly: ignore[missing-import]
    N_VOLUMES,
    Page,
    canvas,
    expected,
    held_rects,
    moved,
    play,
    play_for,
    segments,
)

from surfcensus.app.epi import start_pack_server

sync_api = pytest.importorskip("playwright.sync_api")

pytestmark = pytest.mark.browser

TOTAL = 8 * N_VOLUMES


@pytest.fixture(scope="module")
def packed(tmp_path_factory: pytest.TempPathFactory) -> PackedStream:
    return make_stream(tmp_path_factory.mktemp("epirandom"), n_volumes=N_VOLUMES)


@pytest.fixture(scope="module")
def server(packed: PackedStream) -> Iterator[Any]:
    handle = start_pack_server(packed.out, port=0)
    yield handle
    handle.stop()


@pytest.fixture(scope="module")
def browser() -> Iterator[Any]:
    with sync_api.sync_playwright() as playwright:
        started = launch_browser(playwright)
        yield started
        started.close()


@pytest.fixture
def page(browser: Any, server: Any) -> Iterator[Page]:
    opened = Page(browser, server.url)
    yield opened
    opened.clean()


def order(page: Page) -> list[int]:
    return page.js("window.__epi.order()")


def random_on(page: Page, scope: str = "stream", seed: int = 1) -> list[int]:
    page.set({"random": True, "scope": scope, "seed": seed})
    info = page.js("window.__epi.page.random")
    assert info == {"on": True, "scope": scope, "seed": seed}
    got = order(page)
    assert sorted(got) == list(range(TOTAL))
    return got


def run_of(segs: list[dict[str, Any]], g: int) -> int:
    return max(k for k, s in enumerate(segs) if s["first"] <= g)


# --- what is played ---------------------------------------------------------------------------


@pytest.mark.parametrize("scope", ["stream", "run"])
@pytest.mark.parametrize("rate", [12, 60])
def test_random_play_paints_the_order_in_force_every_volume_once_per_loop(
    page: Page, packed: PackedStream, scope: str, rate: int
) -> None:
    segs = segments(packed)
    page.set({"view": "raw", "normalized": False})
    got = random_on(page, scope)
    assert got != list(range(TOTAL))
    if scope == "run":
        for seg in segs:  # the runs in the order of acquisition, permuted within each
            block = got[seg["first"] : seg["first"] + seg["n"]]
            assert sorted(block) == list(range(seg["first"], seg["first"] + seg["n"]))
    else:
        assert len({run_of(segs, g) for g in got[:24]}) > 2  # the runs are mixed
    stalls = page.js("window.__epi.page.stalls + window.__epi.page.holds")
    # from the last place of the order: the loop that follows is the order itself
    log = play(page, got[-1], rate, TOTAL)
    gs = [e["g"] for e in log][:TOTAL]
    assert gs == got
    assert [e["n"] for e in log[:TOTAL]] == [1] * TOTAL
    assert all(e["nz"] > 0 for e in log), "a blank frame was painted"
    assert page.js("window.__epi.page.stalls + window.__epi.page.holds") == stalls
    span = (log[TOTAL - 1]["ms"] - log[0]["ms"]) / 1000
    assert (TOTAL - 1) / span == pytest.approx(rate, rel=0.06)
    gaps = np.diff([e["ms"] for e in log[:TOTAL]])
    assert gaps.max() < 3 * 1000 / rate, gaps.max()
    # the slider follows the volume painted, not the place in the order
    assert page.js("Number(document.getElementById('time').value)") == page.js(
        "window.__epi.shown.g"
    )


@pytest.mark.parametrize("scope", ["stream", "run"])
def test_at_300_the_painted_volumes_are_a_subsequence_of_the_order(page: Page, scope: str) -> None:
    page.set({"view": "pre"})
    got = random_on(page, scope, seed=3)
    place = {g: p for p, g in enumerate(got)}
    log = play_for(page, got[-1], 300, 3.0)
    places = [place[e["g"]] for e in log]
    loops = 0
    for a, b in itertools.pairwise(places):
        if b <= a:
            assert b == 0, (a, b)  # a loop starts again at the first place; none twice in a loop
            loops += 1
    assert loops >= 2
    assert max(e["n"] for e in log) > 1  # volumes between two paints were left out
    assert all(e["nz"] > 0 for e in log)
    assert (
        "at most once" in page.js("window.__epi.texts().random")
        or page.js("window.__epi.state.playing") is False
    )


def test_the_canvas_holds_the_decoded_volume_painted_in_a_random_order(
    page: Page, packed: PackedStream
) -> None:
    segs = segments(packed)
    random_on(page, "stream", seed=7)
    checked = set()
    for view in ("raw", "pre"):
        page.set({"view": view})
        for _ in range(4):
            page.set({"rate": 30, "playing": True})
            page.page.wait_for_timeout(230)
            shown = page.set({"playing": False})
            checked.add(shown["g"])
            want = expected(segs, shown["g"], shown["slices"], view, False)
            for plane in range(3):
                assert np.array_equal(canvas(page, view, plane), want[plane]), (view, shown["g"])
    assert len(checked) >= 6


# --- the controls -------------------------------------------------------------------------------


def test_the_key_the_checkbox_the_scope_and_the_seed(page: Page) -> None:
    assert page.js("window.__epi.page.random") == {"on": False, "scope": "stream", "seed": 1}
    assert order(page) == list(range(TOTAL))
    assert page.js("window.__epi.texts().random") == "seed 1"
    page.page.keyboard.press("x")
    page.page.wait_for_function("() => window.__epi.state.random === true")
    first = order(page)
    assert first != list(range(TOTAL)) and page.js("document.getElementById('random').checked")
    assert page.js("window.__epi.texts().random") == "seed 1: every volume once per loop"
    page.page.keyboard.press("Shift+X")
    page.page.wait_for_function("() => window.__epi.state.seed === 2")
    second = order(page)
    assert second != first and sorted(second) == list(range(TOTAL))
    assert page.js("window.__epi.texts().random").startswith("seed 2:")
    page.page.select_option("#random-scope", "run")
    page.page.wait_for_function("() => window.__epi.state.scope === 'run'")
    by_run = order(page)
    assert by_run not in (first, second)
    assert all(by_run[k] // N_VOLUMES == k // N_VOLUMES for k in range(TOTAL))
    page.page.click("#random")
    page.page.wait_for_function("() => window.__epi.state.random === false")
    assert order(page) == list(range(TOTAL))
    assert page.js("window.__epi.page.random") == {"on": False, "scope": "run", "seed": 2}
    # the same seed and scope give the same order again
    page.page.select_option("#random-scope", "stream")
    page.set({"random": True, "seed": 2})
    assert order(page) == second


def test_switching_while_playing_goes_on_from_the_volume_on_screen(page: Page) -> None:
    page.set({"g": 10, "rate": 12})
    page.js("window.__epi.page.log.length = 0")
    page.set({"playing": True})
    wait = "(n) => window.__epi.page.log.filter((e) => e.reason === 'play').length >= n"
    page.page.wait_for_function(wait, arg=4)
    # switched in one evaluation, so that the log is split exactly at the switch
    switch = (
        "(on) => { const n = window.__epi.page.log.length;"
        " window.__epi.set({ random: on }); return n; }"
    )
    at = page.page.evaluate(switch, True)
    got = order(page)
    page.page.wait_for_function(wait, arg=12)
    off = page.page.evaluate(switch, False)
    page.page.wait_for_function(wait, arg=20)
    page.set({"playing": False})
    log = page.js("window.__epi.page.log")
    place = {g: p for p, g in enumerate(got)}

    def played(part: list[dict[str, Any]], before: int) -> list[int]:
        return [before] + [e["g"] for e in part if e["reason"] == "play"]

    first = played(log[:at], 10)
    on_part = played(log[at:off], first[-1])
    off_part = played(log[off:], on_part[-1])
    assert len(first) > 3 and len(on_part) > 3 and len(off_part) > 3
    # before the switch on: time order; after it: the next places of the order from the volume
    # on screen; after the switch off: time order again from the volume on screen
    assert all(b == (a + 1) % TOTAL for a, b in itertools.pairwise(first))
    assert all(place[b] == (place[a] + 1) % TOTAL for a, b in itertools.pairwise(on_part))
    assert all(b == (a + 1) % TOTAL for a, b in itertools.pairwise(off_part))
    assert any(b != (a + 1) % TOTAL for a, b in itertools.pairwise(on_part))


def test_the_arrow_keys_home_end_and_r_stay_in_time(page: Page, packed: PackedStream) -> None:
    segs = segments(packed)
    random_on(page)
    page.set({"g": 50})
    for key, want in (
        ("ArrowRight", 51),
        ("ArrowRight", 52),
        ("ArrowLeft", 51),
        ("Shift+ArrowRight", 61),
        ("Home", 0),
        ("End", TOTAL - 1),
        ("r", 0),
        ("r", segs[1]["first"]),
        ("Shift+R", 0),
    ):
        page.page.keyboard.press(key)
        page.page.wait_for_function("(g) => window.__epi.shown.g === g", arg=want)
    assert page.js("window.__epi.state.random") is True


@pytest.mark.parametrize("size", [(1400, 900), (400, 800)])
def test_nothing_moves_when_the_order_is_switched_or_while_it_plays(
    browser: Any, server: Any, size: tuple[int, int]
) -> None:
    opened = Page(browser, server.url, *size)
    try:
        for view in ("raw", "pre"):
            opened.set({"view": view, "g": 40})
            held0 = held_rects(opened)
            opened.page.keyboard.press("x")
            opened.page.wait_for_function("() => window.__epi.state.random === true")
            assert moved(held0, held_rects(opened)) == {}
            for scope in ("stream", "run"):
                opened.set({"scope": scope})
                # random play across runs, grids and the session boundary
                log = play_for(opened, 40, 60, 2.0)
                assert len({e["grid"] for e in log}) == 2 or scope == "run"
                assert moved(held0, held_rects(opened)) == {}, (view, scope)
            opened.set({"rate": 300, "playing": True})
            opened.page.wait_for_timeout(1500)
            assert moved(held0, held_rects(opened)) == {}
            opened.set({"playing": False, "rate": 12})
            opened.page.keyboard.press("Shift+X")
            opened.page.keyboard.press("x")
            opened.page.wait_for_function("() => window.__epi.state.random === false")
            opened.set({"g": 40, "scope": "stream", "seed": 1})
            assert moved(held0, held_rects(opened)) == {}, view
        assert opened.js("document.scrollingElement.scrollWidth") <= size[0]
    finally:
        opened.clean()


# --- the surface control ---------------------------------------------------------------------


@pytest.mark.parametrize("size", [(1400, 900), (400, 800)])
def test_the_surface_control_is_named_and_visible_and_acts_in_every_view(
    browser: Any, server: Any, size: tuple[int, int]
) -> None:
    opened = Page(browser, server.url, *size)
    try:
        label = opened.js("document.getElementById('contours').closest('label').textContent")
        assert " ".join(label.split()) == "surfaces s"
        # in the window, and the row of controls does not scroll sideways
        box = opened.js(
            "(() => { const r = document.getElementById('contours').closest('label')"
            ".getBoundingClientRect(); return [r.left, r.right]; })()"
        )
        assert box[0] >= 0 and box[1] <= size[0]
        assert opened.js(
            "(() => { const c = document.getElementById('controls');"
            " return c.scrollWidth <= c.clientWidth; })()"
        )
        assert opened.js("document.scrollingElement.scrollWidth") <= size[0]
        drawn = "window.__epi.pixels(arg, 1, 'over').drawn"
        opened.set({"crosshair": False, "g": 10, "slices": [8, 9, 7]})
        for view, group in (("raw", "raw"), ("pre", "pre")):
            for kind in ("series", "mean", "sd"):
                opened.set({"view": view, "kind": kind})
                assert opened.js(drawn, group) > 0, (view, kind)
                opened.page.keyboard.press("s")
                opened.page.wait_for_function("() => window.__epi.state.contours === false")
                assert not opened.js("document.getElementById('contours').checked")
                assert opened.js(drawn, group) == 0, (view, kind)
                opened.page.keyboard.press("s")
                opened.page.wait_for_function("() => window.__epi.state.contours === true")
        # the help says the same word
        help_text = opened.js("document.getElementById('help').textContent")
        assert "surfaces (the white and pial contours) on, off" in " ".join(help_text.split())
    finally:
        opened.clean()


def test_what_the_page_says_about_the_random_order(page: Page) -> None:
    title = "document.getElementById('random-note').title"
    assert "order of acquisition" in page.js(title)
    page.set({"random": True})
    said = page.js(title)
    assert "each once per loop" in said and "nothing else changed" in said
    assert "x for the order of acquisition" in said
    help_text = " ".join(page.js("document.getElementById('help').textContent").split())
    assert "Random order (x), off by default" in help_text
    assert "slow drift" in help_text and "show as flicker" in help_text
    for text in (said, help_text):
        assert "\u2014" not in text
