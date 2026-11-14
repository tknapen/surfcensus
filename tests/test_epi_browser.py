"""The EPI page in a real browser, on the pack-only server (work package E1.D; ``just
test-browser``, marked ``browser``, not part of the default run or of CI). Modelled on stage
1's ``check_player.py``.

The packs are those of ``tests/epi_packs.py``: one subject, two sessions, eight packed runs of
24 volumes (192 in the stream), a run on a grid of its own (7 mm on a shifted field of view),
a brighter run, a flagged run, a session without contours. What is checked: the canvas pixels
and the voxel centres of a screenshot against the values decoded in Python with
``func.packread``, under intensity normalization off and on; that playing paints every volume
of the stream once and in order at 12, 30 and 60 volumes per second through a run boundary, a
grid change, a session boundary and the end of the stream, and nothing blank; that at 120, 200
and 300 per second the stream keeps the rate, leaves out the volumes between two paints and
says so (D43); jumps to the
exact volume; the slices after a grid change by the nearest-mm rule; the contours of the grid
on screen; the memory limit; the warnings; what the page says about the run on screen; the
page at 1400 and 400 px; no request to another host and an empty console."""

from __future__ import annotations

import io
import itertools
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from browser_launch import launch_browser  # pyrefly: ignore[missing-import]
from epi_packs import PackedStream, make_stream  # pyrefly: ignore[missing-import]
from PIL import Image

from surfcensus.app.epi import start_pack_server
from surfcensus.func import PLANES
from surfcensus.func.packread import common_scale, decode_still, decode_volume, read_pack

sync_api = pytest.importorskip("playwright.sync_api")

pytestmark = pytest.mark.browser

N_VOLUMES = 24
ZERO = (24, 24, 24)
POS = ((214, 48, 39), (255, 224, 144))
NEG = ((33, 102, 172), (166, 226, 255))


@pytest.fixture(scope="module")
def packed(tmp_path_factory: pytest.TempPathFactory) -> PackedStream:
    return make_stream(tmp_path_factory.mktemp("epibrowser"), n_volumes=N_VOLUMES)


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


class Page:
    """A page of the app with what it said on its console and every URL it asked for."""

    def __init__(self, browser: Any, url: str, width: int = 1400, height: int = 900) -> None:
        self.page = browser.new_page(viewport={"width": width, "height": height})
        self.url = url
        self.console: list[str] = []
        self.requests: list[str] = []
        self.page.on("console", lambda m: self.console.append(f"{m.type}: {m.text}"))
        self.page.on("pageerror", lambda e: self.console.append(f"pageerror: {e}"))
        self.page.on("request", lambda r: self.requests.append(r.url))
        self.page.goto(url)
        self.page.wait_for_function("() => window.__epi && window.__epi.shown", timeout=20000)
        self.page.wait_for_function("() => window.__epi.allLoaded()", timeout=30000)

    def js(self, body: str, arg: Any = None) -> Any:
        return self.page.evaluate(f"(arg) => {body}", arg)

    def set(self, change: dict[str, Any]) -> dict[str, Any]:
        return self.page.evaluate("(c) => window.__epi.set(c)", change)

    def clean(self) -> None:
        assert self.console == []
        assert [u for u in self.requests if not u.startswith(self.url)] == []
        self.page.close()


@pytest.fixture
def page(browser: Any, server: Any) -> Iterator[Page]:
    opened = Page(browser, server.url)
    yield opened
    opened.clean()


# --- what the page should show, computed here ---------------------------------------------------


def segments(packed: PackedStream) -> list[dict[str, Any]]:
    """Per packed run in stream order: session dir, pack, grid, first, window."""
    out = []
    at = 0
    for sid in packed.sessions:
        info = packed.infos[sid]
        grids = {g.grid_key: g for g in info.grids}
        for run in info.runs:
            if run.n_volumes is None:
                continue
            pack = read_pack(packed.out / sid, run.run)
            assert pack is not None
            out.append(
                {
                    "dir": packed.out / sid,
                    "pack": pack,
                    "grid": grids[pack.grid_key],
                    "first": at,
                    "n": run.n_volumes,
                    "window": run.window,
                    "sd_window": run.sd_window,
                    "run": run,
                }
            )
            at += run.n_volumes
    return out


def segment_of(segs: list[dict[str, Any]], g: int) -> dict[str, Any]:
    return next(s for s in reversed(segs) if s["first"] <= g)


def table(
    segs: list[dict[str, Any]], seg: dict[str, Any], normalized: bool, sd: bool = False
) -> np.ndarray:
    if normalized:
        return np.arange(256, dtype=np.uint8)
    key = "sd_window" if sd else "window"
    top = max(s[key][1] for s in segs)
    return common_scale(seg[key], top)


def panel_rows(volume: np.ndarray, plane: int, index: int) -> np.ndarray:
    """A slice as the page draws it: top row first, the horizontal of coronal and axial against
    its voxel axis (logic.SPEC)."""
    rows = np.take(volume, index, axis=plane).T[::-1]
    return rows[:, ::-1] if plane > 0 else rows


def diff_colour(d: int, scale: int) -> tuple[int, int, int]:
    a = min(abs(d), scale)
    if a == 0:
        return ZERO
    mid, end = POS if d > 0 else NEG
    c0, c1, num = (ZERO, mid, 2 * a) if 2 * a <= scale else (mid, end, 2 * a - scale)
    return tuple(c0[k] + ((c1[k] - c0[k]) * num * 2 + scale) // (2 * scale) for k in range(3))  # type: ignore[return-value]


def expected(
    segs: list[dict[str, Any]],
    g: int,
    slices: list[int],
    view: str,
    normalized: bool,
    scale: int = 32,
) -> list[np.ndarray]:
    """Per plane, the RGB the canvas of the first group of ``view`` holds."""
    seg = segment_of(segs, g)
    lut = table(segs, seg, normalized)
    t = g - seg["first"]
    raw = lut[decode_volume(seg["dir"], seg["pack"], "raw", t)]
    pre = lut[decode_volume(seg["dir"], seg["pack"], "pre", t)]
    out = []
    for plane in range(3):
        if view == "diff":
            a = panel_rows(raw, plane, slices[plane]).astype(int)
            b = panel_rows(pre, plane, slices[plane]).astype(int)
            colours = np.array(
                [
                    [diff_colour(int(x - y), scale) for x, y in zip(ra, rb, strict=True)]
                    for ra, rb in zip(a, b, strict=True)
                ]
            )
            out.append(colours.astype(np.uint8))
        else:
            gray = panel_rows(raw if view == "raw" else pre, plane, slices[plane])
            out.append(np.repeat(gray[..., None], 3, axis=2))
    return out


def canvas(page: Page, group: str, plane: int) -> np.ndarray:
    got = page.js("window.__epi.pixels(arg[0], arg[1])", [group, plane])
    return np.asarray(got["rgb"], np.uint8).reshape(got["h"], got["w"], 3)


def nearest(grid: Any, x: np.ndarray) -> list[int]:
    out = []
    for plane in PLANES:
        target = float(np.dot(grid.slice_normal[plane], x))
        out.append(int(np.argmin(np.abs(np.asarray(grid.slice_mm[plane]) - target))))
    return out


def meet(grid: Any, slices: list[int]) -> np.ndarray:
    normals = np.asarray([grid.slice_normal[p] for p in PLANES])
    distances = np.asarray([grid.slice_mm[p][k] for p, k in zip(PLANES, slices, strict=True)])
    return np.linalg.solve(normals, distances)


# --- exactness ----------------------------------------------------------------------------------


@pytest.mark.parametrize("normalized", [False, True])
def test_the_canvas_holds_the_decoded_values_through_the_table_of_the_setting(
    page: Page, packed: PackedStream, normalized: bool
) -> None:
    segs = segments(packed)
    page.set({"normalized": normalized})
    rng = np.random.default_rng(11)
    cases = [0, 23, 24, 30, 95, 96, 110, 119, 120, 191] + [int(x) for x in rng.integers(0, 192, 6)]
    for view in ("raw", "pre", "diff"):
        page.set({"view": view})
        for g in cases:
            shown = page.set({"g": g})
            assert shown["g"] == g and shown["normalized"] is normalized
            want = expected(segs, g, shown["slices"], view, normalized)
            for plane in range(3):
                assert np.array_equal(canvas(page, view, plane), want[plane]), (view, g, plane)


@pytest.mark.parametrize("normalized", [False, True])
def test_the_screenshot_shows_the_values_at_every_voxel_centre(
    page: Page, packed: PackedStream, normalized: bool
) -> None:
    segs = segments(packed)
    page.set({"normalized": normalized, "contours": False, "crosshair": False, "view": "pre"})
    for g in (5, 100, 150):  # grid A, grid C, the second session
        shown = page.set({"g": g})
        want = expected(segs, g, shown["slices"], "pre", normalized)
        for plane in range(3):
            rect = page.js("window.__epi.rect('pre', arg)", plane)
            x0, y0 = int(rect["x"]), int(rect["y"])
            shot = page.page.screenshot(
                clip={"x": x0, "y": y0, "width": int(rect["w"]) + 2, "height": int(rect["h"]) + 2}
            )
            image = np.asarray(Image.open(io.BytesIO(shot)).convert("RGB"))
            h, w = want[plane].shape[:2]
            assert (rect["cw"], rect["ch"]) == (w, h)
            # the edge letters lie in the margin of the box, outside the picture: every voxel
            # centre is compared, none is passed over (review of the page, finding 10)
            letters = page.js(
                'Array.from(document.querySelectorAll(`#views .epi-group[data-group="pre"] '
                '.epi-panel[data-plane="${arg}"] .epi-letter`)).map((e) => {'
                " const r = e.getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom,"
                " e.textContent]; })",
                plane,
            )
            assert len(letters) == 4 and all(t for *_, t in letters)
            for x1, y1, x2, y2, _ in letters:
                apart = (
                    x2 <= rect["x"]
                    or x1 >= rect["x"] + rect["w"]
                    or y2 <= rect["y"]
                    or y1 >= rect["y"] + rect["h"]
                )
                assert apart, (plane, (x1, y1, x2, y2), rect)
            for r in range(h):
                for c in range(w):
                    px = rect["x"] - x0 + (c + 0.5) * rect["w"] / w
                    py = rect["y"] - y0 + (r + 0.5) * rect["h"] / h
                    got = image[int(py), int(px)]
                    assert tuple(got) == tuple(want[plane][r, c]), (g, plane, r, c)


def test_the_stills_and_their_sd_window(page: Page, packed: PackedStream) -> None:
    segs = segments(packed)
    # the brighter run, and the run with an offset, whose value table and SD table differ when
    # normalization is off (a mean drawn through the SD table, or the SD through the value
    # table, shows there)
    offset = segs[6]
    assert offset["window"][1] / segs[1]["window"][1] != pytest.approx(
        offset["sd_window"][1] / segs[1]["sd_window"][1], rel=1e-3
    )
    assert not np.array_equal(table(segs, offset, False), table(segs, offset, False, sd=True))
    for g in (40, offset["first"] + 3):
        for normalized in (False, True):
            for kind, still in (("mean", "mean"), ("sd", "sd")):
                shown = page.set({"g": g, "view": "raw", "kind": kind, "normalized": normalized})
                seg = segment_of(segs, g)
                lut = table(segs, seg, normalized, sd=kind == "sd")
                volume = lut[decode_still(seg["dir"], seg["pack"], "raw", still)]
                for plane in range(3):
                    gray = panel_rows(volume, plane, shown["slices"][plane])
                    assert np.array_equal(canvas(page, "raw", plane)[..., 0], gray), (g, kind)
            legend = page.js("window.__epi.texts().legend")
            assert ("normalization on" if normalized else "normalization off") in legend
            assert ("temporal SD" in legend) is (kind == "sd")


# --- one continuous stream ----------------------------------------------------------------------


def play(page: Page, start: int, rate: int, count: int) -> list[dict[str, Any]]:
    page.set({"g": start, "rate": rate})
    page.js("window.__epi.page.log.length = 0")
    page.set({"playing": True})
    page.page.wait_for_function(
        "(n) => window.__epi.page.log.filter((e) => e.reason === 'play').length >= n",
        arg=count,
        timeout=60000,
    )
    page.set({"playing": False})
    return [e for e in page.js("window.__epi.page.log") if e["reason"] == "play"]


@pytest.mark.parametrize(
    ("rate", "start", "count"),
    [(12, 14, 24), (30, 86, 50), (60, 90, 120), (60, 170, 60)],
)
def test_playing_paints_every_volume_once_in_order_through_every_boundary(
    page: Page, packed: PackedStream, rate: int, start: int, count: int
) -> None:
    segs = segments(packed)
    total = sum(s["n"] for s in segs)
    page.set({"view": "side", "normalized": False})
    stalls = page.js("window.__epi.page.stalls")
    log = play(page, start, rate, count)
    gs = [e["g"] for e in log]
    assert gs[:count] == [(start + 1 + k) % total for k in range(count)]
    assert all(e["nz"] > 0 for e in log), "a blank frame was painted"
    crossed = {s["first"] for s in segs[1:]} & set(gs)
    assert crossed, "no boundary was crossed"
    # the painted rate over the whole stretch, boundaries included
    span = (log[count - 1]["ms"] - log[0]["ms"]) / 1000
    painted = (count - 1) / span
    assert painted == pytest.approx(rate, rel=0.06), painted
    gaps = np.diff([e["ms"] for e in log[:count]])
    assert gaps.max() < 3 * 1000 / rate, gaps.max()
    assert page.js("window.__epi.page.stalls") == stalls
    # nothing but the pictures: the view, the slices at the place, the layout stay
    assert {e["view"] for e in log} == {"side"}
    boxes = {tuple(page.js("window.__epi.boxRect('raw', arg)", p).values()) for p in range(3)}
    assert len(boxes) == 3


def play_for(page: Page, start: int, rate: int, seconds: float) -> list[dict[str, Any]]:
    """Play from ``start`` at ``rate`` for ``seconds`` and return what was painted."""
    page.set({"g": start, "rate": rate})
    page.js("window.__epi.page.log.length = 0")
    page.set({"playing": True})
    page.page.wait_for_timeout(int(seconds * 1000))
    page.set({"playing": False})
    return [e for e in page.js("window.__epi.page.log") if e["reason"] == "play"]


NOTE = re.compile(r"^(every \d+(st|nd|rd|th) volume shown|\d0% of volumes shown)$")


@pytest.mark.parametrize("rate", [120, 200, 300])
def test_above_the_refresh_rate_the_stream_keeps_the_rate_and_says_what_it_leaves_out(
    page: Page, packed: PackedStream, rate: int
) -> None:
    """D43: above the frame rate the page achieves, the volume that is due is painted and the
    ones in between are not; none out of order, none twice, none blank, no stall at a run, grid
    or session boundary, the stream at the rate asked, and the page says so beside the rate
    without anything moving."""
    segs = segments(packed)
    total = sum(s["n"] for s in segs)
    page.set({"view": "side", "normalized": False, "g": 0})
    held0 = held_rects(page)
    assert page.js("document.getElementById('rate-note').textContent") == ""
    stalls = page.js("window.__epi.page.stalls + window.__epi.page.holds")
    log = play_for(page, 0, rate, 3.5)
    fps = page.js("window.__epi.page.fps")
    gs = [e["g"] for e in log]
    # in order: strictly increasing within a loop, and a loop starts again at the first volume
    for a, b in itertools.pairwise(gs):
        assert b > a or b == 0, (a, b)
    assert gs.count(0) >= 2, "the stream did not loop"
    assert all(e["nz"] > 0 for e in log), "a blank frame was painted"
    assert page.js("window.__epi.page.stalls + window.__epi.page.holds") == stalls
    # every run of every session and grid was on screen in every full loop
    loops: list[list[int]] = [[]]
    for g in gs:
        if g == 0 and loops[-1]:
            loops.append([])
        loops[-1].append(g)
    for loop in loops[1:-1]:
        assert {segs.index(segment_of(segs, g)) for g in loop} == set(range(len(segs)))
    # volumes were left out, at most one paint per frame, and the stream kept the rate: the
    # tolerance of 10 per cent holds on a shared machine under load (measured: within 2)
    assert max(e["n"] for e in log) > 1
    span = (log[-1]["ms"] - log[0]["ms"]) / 1000
    stream_rate = sum(e["n"] for e in log[1:]) / span
    painted = (len(log) - 1) / span
    assert stream_rate == pytest.approx(rate, rel=0.10), stream_rate
    assert painted <= fps * 1.15, (painted, fps)
    # what the page says, in a box that keeps its place and size
    page.set({"g": 0, "rate": rate})
    page.set({"playing": True})
    page.page.wait_for_function(
        "() => document.getElementById('rate-note').textContent !== ''", timeout=5000
    )
    note = page.js("document.getElementById('rate-note').textContent")
    assert NOTE.match(note), note
    assert moved(held0, held_rects(page)) == {}
    page.set({"playing": False})
    page.page.wait_for_function(
        "() => document.getElementById('rate-note').textContent === ''", timeout=3000
    )
    assert total == 192


@pytest.mark.parametrize("rate", [30, 60])
def test_at_60_and_below_no_volume_is_left_out(page: Page, packed: PackedStream, rate: int) -> None:
    """D43 keeps D38 at the rates of before: every volume once, in order, and nothing said."""
    total = sum(s["n"] for s in segments(packed))
    log = play_for(page, 100, rate, 2.5)
    assert [e["n"] for e in log] == [1] * len(log)
    assert [e["g"] for e in log] == [(101 + k) % total for k in range(len(log))]
    assert page.js("window.__epi.page.skipping") is False
    assert page.js("document.getElementById('rate-note').textContent") == ""


def test_the_slices_on_another_grid_are_the_nearest_in_mm_and_come_back(
    page: Page, packed: PackedStream
) -> None:
    segs = segments(packed)
    grid_a, grid_c = segs[0]["grid"], segs[4]["grid"]
    assert grid_a.grid_key != grid_c.grid_key
    for slices in ([3, 5, 10], [8, 9, 7], [0, 17, 0], [16, 0, 13]):
        shown = page.set({"g": 10, "slices": slices})
        assert shown["slices"] == slices
        on_c = page.set({"g": 100})
        assert on_c["grid"] == grid_c.grid_key
        assert on_c["slices"] == nearest(grid_c, meet(grid_a, slices))
        back = page.set({"g": 130})  # the second session, grid A again
        assert back["slices"] == slices


def test_the_contours_drawn_are_those_of_the_grid_on_screen(
    page: Page, packed: PackedStream
) -> None:
    segs = segments(packed)
    count = "window.__epi.pixels('raw', 1, 'over').drawn"
    page.set({"view": "raw", "crosshair": False, "contours": True})
    drawn = {}
    for g in (10, 100, 150):
        shown = page.set({"g": g, "slices": [8, 9, 7]} if g == 10 else {"g": g})
        state = page.js("window.__epi.contoursDrawn()")
        assert state["grid"] == shown["grid"] == segment_of(segs, g)["grid"].grid_key
        assert state["key"].startswith(shown["grid"])
        drawn[g] = page.js(count)
    assert drawn[10] > 0 and drawn[100] > 0
    assert drawn[150] == 0  # the second session has no contours


def test_jumps_go_to_the_exact_volume_anywhere_in_the_stream(
    page: Page, packed: PackedStream
) -> None:
    segs = segments(packed)
    for g in (0, 191, 23, 24, 96, 119, 120, 57):
        assert page.set({"g": g})["g"] == g
        assert page.js("Number(document.getElementById('time').value)") == g
    # a run button jumps to the first volume of its run
    for k, seg in enumerate(segs):
        page.page.click(f'#segments button[data-seg="{k}"]')
        page.page.wait_for_function(
            "(g) => window.__epi.shown && window.__epi.shown.g === g", arg=seg["first"]
        )
    page.page.keyboard.press("r")
    page.page.wait_for_function("() => window.__epi.shown.g === 0")
    page.page.keyboard.press("Shift+R")
    page.page.wait_for_function("(g) => window.__epi.shown.g === g", arg=segs[-1]["first"])
    page.page.keyboard.press("ArrowLeft")
    page.page.wait_for_function("(g) => window.__epi.shown.g === g", arg=segs[-1]["first"] - 1)


def test_what_the_page_says_follows_the_run_on_screen(page: Page, packed: PackedStream) -> None:
    segs = segments(packed)
    said = {}
    for k, seg in enumerate(segs):
        page.set({"g": seg["first"]})
        texts = page.js("window.__epi.texts()")
        said[k] = texts
        assert seg["run"].name in texts["runNote"] and seg["run"].name in texts["readout"]
        assert seg["run"].name in texts["corrections"]
        flagged = seg["run"].frame is not None and seg["run"].frame.flagged
        assert (f"{seg['run'].name} is flagged" in texts["warning"]) is flagged
        assert ("flagged:" in texts["badges"]) is flagged
        # the corrections: each statement with its own source, the dof once (finding 7)
        assert texts["corrections"].count("6 dof") == 1, texts["corrections"]
        assert "head-motion correction (stated by no file)" in texts["corrections"]
    assert (
        "unchecked" in said[3]["warning"] and "distortion correction, FLB" in said[3]["corrections"]
    )
    assert (
        'distortion correction, FLB (fieldmap-less, SyN-based) (summary report: "FLB")'
        in (said[3]["corrections"])
    )
    # a statement about the raw data, said as such and not as a fault of the page
    assert "statement about the raw data" in said[3]["warning"]
    assert "failed" not in said[3]["warning"]
    assert "odd one" in said[5]["warning"]  # the anchor of the second session is the outlier
    assert "no raw series" in said[5]["session"]
    assert "not in the stream" in said[5]["badges"]
    assert page.js("document.querySelector('#segments button.flagged') !== null")
    # nothing that changes while playing is announced as an alert
    assert page.js("document.querySelectorAll('.epi [role=alert]').length") == 0
    assert page.js("getComputedStyle(document.querySelector('#badges .badge.flag')).color") != (
        "rgb(255, 82, 82)"
    )


def test_the_normalization_key_and_what_the_legend_says(page: Page) -> None:
    assert "normalization off" in page.js("window.__epi.texts().legend")
    page.page.keyboard.press("n")
    page.page.wait_for_function("() => window.__epi.state.normalized === true")
    assert "normalization on" in page.js("window.__epi.texts().legend")
    assert page.js("document.getElementById('normalize').getAttribute('aria-pressed')") == "true"
    page.set({"view": "diff"})
    assert "units of signal" in page.js("window.__epi.texts().legend")


@pytest.mark.parametrize("width", [1400, 400])
def test_the_page_fits_the_window(browser: Any, server: Any, width: int) -> None:
    opened = Page(browser, server.url, width=width, height=900)
    try:
        assert opened.js("document.scrollingElement.scrollWidth") <= width
        for group in ("raw",):
            for plane in range(3):
                box = opened.js("window.__epi.boxRect(arg[0], arg[1])", [group, plane])
                assert box["w"] > 50 and box["x"] >= 0 and box["x"] + box["w"] <= width + 0.5
        opened.page.keyboard.press("3")
        opened.page.wait_for_function("() => window.__epi.shown.view === 'side'")
        assert opened.js("document.scrollingElement.scrollWidth") <= width
    finally:
        opened.clean()


def test_the_memory_limit_holds_for_the_whole_stream(browser: Any, packed: PackedStream) -> None:
    handle = start_pack_server(packed.out, port=0, memory_mib=1)
    try:
        opened = Page(browser, handle.url)
        try:
            stats = opened.js("window.__epi.stats()")
            assert stats["limitBytes"] == 1024 * 1024
            assert stats["chunksHeld"] < stats["chunksInStream"]
            log = play(opened, 0, 60, 200)
            gs = [e["g"] for e in log][:200]
            assert gs == [(1 + k) % 192 for k in range(200)]
            stats = opened.js("window.__epi.stats()")
            assert stats["peakDecoded"] <= stats["limitBytes"]
            assert stats["evictions"] > 0
        finally:
            opened.clean()
    finally:
        handle.stop()


def test_time_to_first_picture_is_recorded(page: Page) -> None:
    timing = page.js("window.__epi.page.timing")
    assert timing["firstFrame"] is not None and timing["firstFrame"] - timing["opened"] < 10000


def test_screenshots_for_a_reader(
    page: Page, packed: PackedStream, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Writes screenshots to look at; asserts only that they were written."""
    where = Path(tmp_path_factory.mktemp("shots"))
    for name, change in (
        ("start", {"g": 0}),
        ("boundary", {"g": 24}),
        ("grid_change", {"g": 96}),
        ("normalized_off", {"g": 30, "view": "side", "normalized": False}),
        ("normalized_on", {"g": 30, "view": "side", "normalized": True}),
        ("difference", {"g": 30, "view": "diff"}),
        ("flagged", {"g": 75, "view": "raw"}),
        ("session_two", {"g": 140}),
    ):
        page.set(change)
        page.page.screenshot(path=str(where / f"{name}.png"))
        assert (where / f"{name}.png").stat().st_size > 1000


# --- the layout (review of the page, findings 1, 3 and 10) ----------------------------------

BLOCKS = ("bar", "timeline", "run-strip", "controls", "traces", "status", "about")


def rects(page: Page) -> dict[str, list[list[float]]]:
    """Where the panel boxes, the canvases and the blocks of the page lie, in CSS px."""
    return page.js(
        "(() => { const r = (e) => { const b = e.getBoundingClientRect();"
        " return [b.x, b.y, b.width, b.height]; };"
        " return { boxes: Array.from(document.querySelectorAll('#views .epi-panel')).map(r),"
        " canvases: Array.from(document.querySelectorAll('#views canvas')).map(r),"
        " blocks: arg.map((id) => document.getElementById(id)).filter(Boolean).map(r) }; })()",
        list(BLOCKS),
    )


HELD = ("bar", "run-strip", "timeline", "controls")
"""The parts of the page whose every element with an id keeps its box (finding F7 of the
review of step 2): not a list of elements, so that one added later is measured too."""


def held_rects(page: Page) -> dict[str, list[float]]:
    """The box of every element with an id inside the bar, the strip, the timeline and the
    controls, and of those four, in CSS px, by id."""
    return page.js(
        "(() => { const out = {}; for (const id of arg) { const root = document.getElementById(id);"
        " if (!root) continue; for (const e of [root, ...root.querySelectorAll('[id]')]) {"
        " const b = e.getBoundingClientRect(); out[e.id] = [b.x, b.y, b.width, b.height]; } }"
        " return out; })()",
        list(HELD),
    )


def moved(r0: dict[str, list[float]], r1: dict[str, list[float]]) -> dict[str, list[float]]:
    """The elements whose box differs, with the largest change of x, y, width and height."""
    assert r0.keys() == r1.keys()
    return {
        k: [round(b - a, 2) for a, b in zip(r0[k], r1[k], strict=True)]
        for k in r0
        if r0[k] != r1[k]
    }


@pytest.mark.parametrize("size", [(1400, 900), (1920, 1080), (400, 800)])
@pytest.mark.parametrize("view", ["raw", "side", "diff"])
def test_nothing_moves_at_any_boundary_of_the_stream(
    browser: Any, server: Any, packed: PackedStream, size: tuple[int, int], view: str
) -> None:
    """D38: at a run or session boundary nothing changes but the pictures. Every panel box and
    every block of the page keeps its place and size; so does every canvas where the grid
    stays, and where the grid changes the canvas keeps the scale in px per mm and its centre
    in its box."""
    segs = segments(packed)
    total = sum(s["n"] for s in segs)
    opened = Page(browser, server.url, *size)
    try:
        # normalization on as well: the legend says the window of the run on screen
        for k, (seg, normalized) in enumerate(
            (s, n) for n in (False, True) for s in [*segs[1:], segs[0]]
        ):
            opened.set({"view": view, "normalized": normalized})
            after = seg["first"]
            before = (after - 1) % total
            prev = segment_of(segs, before)
            opened.set({"g": before})
            r0 = rects(opened)
            held0 = held_rects(opened)
            opened.set({"g": after})
            r1 = rects(opened)
            held1 = held_rects(opened)
            assert r0["boxes"] == r1["boxes"], (k, after)
            assert r0["blocks"] == r1["blocks"], (k, after)
            assert moved(held0, held1) == {}, (k, after)
            if prev["grid"].grid_key == seg["grid"].grid_key:
                assert r0["canvases"] == r1["canvases"], (k, after)
            else:
                for (x0, y0, w0, h0), (x1, y1, w1, h1) in zip(
                    r0["canvases"], r1["canvases"], strict=True
                ):
                    assert abs((x0 + w0 / 2) - (x1 + w1 / 2)) <= 1
                    assert abs((y0 + h0 / 2) - (y1 + h1 / 2)) <= 1
        # and while playing through a run boundary, the flagged run and the session boundary
        opened.set({"g": 40, "normalized": False})
        r0 = rects(opened)
        held0 = held_rects(opened)
        log = play(opened, 40, 60, 80)
        assert {e["g"] for e in log} >= {48, 72, 96, 119, 120}
        opened.set({"g": 125})
        assert rects(opened)["boxes"] == r0["boxes"]
        assert rects(opened)["canvases"] == r0["canvases"]
        assert rects(opened)["blocks"] == r0["blocks"]
        assert moved(held0, held_rects(opened)) == {}
        # play and pause: the button keeps its width for both labels, and nothing beside it
        # moves (the rate, the track of the timeline)
        opened.set({"g": 40, "rate": 2})
        held0 = held_rects(opened)
        opened.set({"playing": True})
        opened.page.wait_for_function(
            "() => document.getElementById('play-text').textContent === 'pause'"
        )
        held1 = held_rects(opened)
        opened.set({"playing": False})
        assert moved(held0, held1) == {}
        # a change of rate while playing, to one that leaves volumes out: the text beside the
        # rate appears in its own box, and nothing moves (D43)
        opened.set({"playing": True})
        opened.set({"rate": 300})
        opened.page.wait_for_function(
            "() => document.getElementById('rate-note').textContent !== ''", timeout=5000
        )
        held2 = held_rects(opened)
        opened.set({"playing": False, "rate": 2})
        assert moved(held0, held2) == {}
        assert moved(held0, held_rects(opened)) == {}
    finally:
        opened.clean()


@pytest.mark.parametrize("size", [(1400, 900), (1920, 1080)])
def test_the_page_does_not_overflow_the_window(
    browser: Any, server: Any, size: tuple[int, int]
) -> None:
    opened = Page(browser, server.url, *size)
    try:
        assert opened.js("window.__epi.state.traces") is True
        for view in ("raw", "pre", "side", "diff"):
            for g in (0, 75, 100, 150):  # an anchor, the flagged run, grid C, session B
                opened.set({"view": view, "g": g})
                height = opened.js("document.scrollingElement.scrollHeight")
                assert height <= size[1], (view, g, height)
                assert opened.js("document.scrollingElement.scrollWidth") <= size[0]
    finally:
        opened.clean()


def picture_share(page: Page) -> float:
    return page.js(
        "(() => { let a = 0; for (const c of document.querySelectorAll('#views canvas.gray')) {"
        " const r = c.getBoundingClientRect();"
        " a += Math.max(0, Math.min(r.right, innerWidth) - Math.max(r.left, 0))"
        " * Math.max(0, Math.min(r.bottom, innerHeight) - Math.max(r.top, 0)); }"
        " return a / (innerWidth * innerHeight); })()"
    )


def test_the_pictures_come_first_and_take_most_of_the_window(browser: Any, server: Any) -> None:
    """Finding 3: the views start under one bar, before any text about the run, and take the
    largest share of the window the controls leave."""
    opened = Page(browser, server.url, 1400, 900)
    try:
        for view, least in (("raw", 0.3), ("side", 0.2), ("diff", 0.3)):
            opened.set({"view": view, "g": 75})
            top = opened.js("document.getElementById('views').getBoundingClientRect().top")
            bar = opened.js("document.getElementById('bar').getBoundingClientRect().bottom")
            assert top <= bar + 8, (view, top, bar)
            for block in ("timeline", "run-strip", "controls", "about"):
                below = opened.js(f"document.getElementById('{block}').getBoundingClientRect().top")
                assert below > top
            share = picture_share(opened)
            assert share >= least, (view, share)
            # the views take what the rest leaves: the page fills the window, it does not
            # leave a stretch empty under it (height-limited views)
            if view != "side":
                height = opened.js("document.querySelector('.epi').getBoundingClientRect().height")
                assert height >= 900 - 30, (view, height)
    finally:
        opened.clean()
    narrow = Page(browser, server.url, 400, 900)
    try:
        first = narrow.js("window.__epi.rect('raw', 0)")
        assert first["y"] + first["h"] <= 900 and first["w"] > 300
    finally:
        narrow.clean()


# --- the boundary marks (finding 4) ---------------------------------------------------------


def test_the_cursor_and_the_boundaries_are_marked_apart_on_the_slider_and_in_the_traces(
    page: Page, packed: PackedStream
) -> None:
    segs = segments(packed)
    total = sum(s["n"] for s in segs)
    page.set({"g": 30})
    ticks = page.js("document.getElementById('time').style.getPropertyValue('--ticks')")
    # one tick per boundary: thin ones between runs, one thick one between sessions
    assert ticks.count("#c9d1db calc(") == 2 * (len(segs) - 2)
    assert ticks.count("#ffffff calc(") == 2
    # the segments are the runs, each a jump to its first volume, the session start marked
    buttons = page.js(
        "Array.from(document.querySelectorAll('#segments button')).map((b) => [b.textContent,"
        " b.classList.contains('session-start'), b.getBoundingClientRect().width])"
    )
    assert [b[0] for b in buttons] == [s["run"].name for s in segs]
    assert [b[1] for b in buttons] == [k == 5 for k in range(len(segs))]
    assert all(w > 10 for *_, w in buttons)
    # hiding the traces leaves the slider and its marks as they are
    before = rects(page)["blocks"][BLOCKS.index("timeline")]
    page.page.keyboard.press("t")
    page.page.wait_for_function("() => window.__epi.state.traces === false")
    assert page.js("document.getElementById('time').style.getPropertyValue('--ticks')") == ticks
    assert page.js("document.getElementById('segments').children.length") == len(segs)
    page.page.keyboard.press("t")
    page.page.wait_for_function("() => window.__epi.state.traces === true")
    assert rects(page)["blocks"][BLOCKS.index("timeline")][2:] == before[2:]
    # in the traces: the cursor in a colour of its own, unlike a run or a session boundary
    page.set({"g": 50})
    shot = Image.open(io.BytesIO(page.page.locator("#traces").screenshot())).convert("RGB")
    pixels = {tuple(p) for p in np.asarray(shot).reshape(-1, 3)}
    assert (255, 61, 240) in pixels  # the cursor, magenta
    assert (255, 255, 255) in pixels  # the session boundary, white
    assert (201, 209, 219) in pixels  # a run boundary, light gray, dashed
    assert total == 192


# --- focus (finding 11) ---------------------------------------------------------------------


def test_the_help_keeps_the_focus_inside_and_gives_it_back(page: Page) -> None:
    page.js("document.getElementById('normalize').focus()")
    page.page.keyboard.press("?")
    page.page.wait_for_function("() => !document.getElementById('help').hidden")
    for key in ("Tab", "Tab", "Tab", "Shift+Tab", "Tab"):
        page.page.keyboard.press(key)
        assert page.js("document.getElementById('help').contains(document.activeElement)"), key
    page.page.keyboard.press("Escape")
    page.page.wait_for_function("() => document.getElementById('help').hidden")
    assert page.js("document.activeElement && document.activeElement.id") == "normalize"
    # opened from nothing, it gives the focus back to nothing: Space then plays, as before
    page.js("document.activeElement.blur()")
    page.page.keyboard.press("?")
    page.page.wait_for_function("() => !document.getElementById('help').hidden")
    page.page.keyboard.press("Escape")
    assert page.js("document.activeElement === document.body")


# --- the place in the anatomy (finding 12) --------------------------------------------------


def test_a_slice_step_on_another_grid_leaves_the_other_planes_where_they_were(
    page: Page, packed: PackedStream
) -> None:
    for slices in ([10, 17, 5], [3, 5, 10], [8, 9, 7]):
        page.set({"g": 10, "slices": slices})
        on_c = page.set({"g": 100})
        for plane in range(3):
            page.set({"g": 100, "slices": on_c["slices"]})
            page.set({"g": 10, "slices": slices})
            page.set({"g": 100})
            # one step up of the plane with the arrow key, as a viewer does it
            page.js("window.__epi.state.active = arg", plane)
            page.page.keyboard.press("ArrowUp")
            page.page.wait_for_function(
                "(a) => window.__epi.shown.slices[a[0]] === a[1]",
                arg=[plane, min(on_c["slices"][plane] + 1, packed.runs[4].grid.shape[plane] - 1)],
            )
            back = page.set({"g": 130})
            others = [q for q in range(3) if q != plane]
            assert [back["slices"][q] for q in others] == [slices[q] for q in others], (
                slices,
                plane,
            )


# --- errors and packs made again (finding 2, the page's part) -------------------------------


def test_an_error_that_passes_leaves_no_error_on_the_page(browser: Any, server: Any) -> None:
    opened = Page.__new__(Page)
    opened.page = browser.new_page(viewport={"width": 1400, "height": 900})
    opened.url = server.url
    opened.console = []
    opened.requests = []
    opened.page.on("request", lambda r: opened.requests.append(r.url))
    failed: list[str] = []

    def once(route: Any) -> None:
        if not failed:
            failed.append(route.request.url)
            route.fulfill(status=404, body="")
        else:
            route.continue_()

    opened.page.route("**/c0001.bin*", once)
    opened.page.goto(server.url)
    opened.page.wait_for_function("() => window.__epi && window.__epi.shown", timeout=20000)
    opened.page.wait_for_function("() => window.__epi.page.errors.length > 0", timeout=20000)
    opened.page.wait_for_function("() => window.__epi.allLoaded()", timeout=30000)
    opened.set({"g": 2})
    assert failed
    assert "ERROR" not in opened.js("window.__epi.texts().status")
    assert opened.js("window.__epi.texts().notice") == ""
    opened.page.close()


def test_packs_made_again_under_an_open_page_are_said_with_a_reload(
    browser: Any, server: Any
) -> None:
    """A file whose bytes are not those the page was given, while the server now lists other
    files: the page says the packs were made again, stops asking, and offers to reload."""
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    calls: list[int] = []

    def stream(route: Any) -> None:
        response = route.fetch()
        body = response.json()
        if calls:  # every answer after the first lists other files
            body["sessions"][0]["runs"][0]["pack"]["token"] = "0" * 16
        calls.append(1)
        route.fulfill(response=response, json=body)

    page.route("**/api/epi/stream", stream)
    page.route("**/c0002.bin*", lambda route: route.fulfill(status=200, body=b"not the bytes"))
    page.goto(server.url)
    page.wait_for_function("() => window.__epi && window.__epi.shown", timeout=20000)
    page.wait_for_function("() => window.__epi.texts().notice !== ''", timeout=20000)
    notice = page.evaluate("window.__epi.texts().notice")
    assert "made again" in notice and "Reload" in notice
    assert page.is_visible("#notice-reload")
    assert len(calls) >= 2
    page.close()
