"""The EPI page in a real browser, on the pack-only server (work package E1.D; ``just
test-browser``, marked ``browser``, not part of the default run or of CI). Modelled on stage
1's ``check_player.py``.

The packs are those of ``tests/epi_packs.py``: one subject, two sessions, eight packed runs of
24 volumes (192 in the stream), a run on a grid of its own (7 mm on a shifted field of view),
a brighter run, a flagged run, a session without contours. What is checked: the canvas pixels
and the voxel centres of a screenshot against the values decoded in Python with
``func.packread``, under intensity normalization off and on; that playing paints every volume
of the stream once and in order at 12, 30 and 60 volumes per second through a run boundary, a
grid change, a session boundary and the end of the stream, and nothing blank; jumps to the
exact volume; the slices after a grid change by the nearest-mm rule; the contours of the grid
on screen; the memory limit; the warnings; what the page says about the run on screen; the
page at 1400 and 400 px; no request to another host and an empty console."""

from __future__ import annotations

import io
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
            # the edge letters lie over the picture; a voxel centre under one is not compared
            letters = page.js(
                'Array.from(document.querySelectorAll(`#views .epi-group[data-group="pre"] '
                '.epi-panel[data-plane="${arg}"] .epi-letter`)).map((e) => {'
                " const r = e.getBoundingClientRect(); return [r.left - 2, r.top - 2, r.right + 2,"
                " r.bottom + 2]; })",
                plane,
            )
            compared = 0
            for r in range(h):
                for c in range(w):
                    px = rect["x"] - x0 + (c + 0.5) * rect["w"] / w
                    py = rect["y"] - y0 + (r + 0.5) * rect["h"] / h
                    ax, ay = px + x0, py + y0
                    if any(x1 <= ax <= x2 and y1 <= ay <= y2 for x1, y1, x2, y2 in letters):
                        continue
                    got = image[int(py), int(px)]
                    assert tuple(got) == tuple(want[plane][r, c]), (g, plane, r, c)
                    compared += 1
            assert compared > 0.9 * w * h


def test_the_stills_and_their_sd_window(page: Page, packed: PackedStream) -> None:
    segs = segments(packed)
    for normalized in (False, True):
        for kind, still in (("mean", "mean"), ("sd", "sd")):
            shown = page.set({"g": 40, "view": "raw", "kind": kind, "normalized": normalized})
            seg = segment_of(segs, 40)
            lut = table(segs, seg, normalized, sd=kind == "sd")
            volume = lut[decode_still(seg["dir"], seg["pack"], "raw", still)]
            for plane in range(3):
                gray = panel_rows(volume, plane, shown["slices"][plane])
                assert np.array_equal(canvas(page, "raw", plane)[..., 0], gray)
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
        page.page.click(f'#runs button[data-seg="{k}"]')
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
        assert ("FLAGGED" in texts["warning"]) is flagged
    assert (
        "unchecked" in said[3]["warning"] and "distortion correction, FLB" in said[3]["corrections"]
    )
    assert "odd one" in said[5]["warning"]  # the anchor of the second session is the outlier
    assert "no raw series" in said[5]["session"]
    assert page.js("document.querySelector('#runs button.flagged') !== null")
    assert page.js("document.querySelectorAll('#runs button:disabled').length") == 1


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
