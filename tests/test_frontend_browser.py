"""The pages of the rater app in a real browser, against a mock server (``just test-browser``;
marked ``browser``, not part of the default run or of CI).

The server here is ``tests/frontend_mock.py``: it serves the pages, the static files and the
tiles of renders on disk, answers with the shapes of ``app/schemas.py`` and validates every
body it is sent. The subjects are three synthetic 1 mm brains rendered at 320 px. To run the
same tests on other renders, name their media directories (each with ``render.json`` and
``tiles/``) in ``SURFCENSUS_FRONTEND_MEDIA``, separated by the path separator; at least
three are needed.

What is tested is what a rater does: every key of the keyboard map of docs/rater_app.md,
click marks and where they land, the note field, the unratable reasons, the help screen,
submit, skip, back, the tour of the screen, a server that is away, and what the page asks
the server for. The test of the page on the real server is ``tests/test_browser.py``.

A page of these tests belongs to a rater who was shown the tour before (``SEEN_TOUR``), so
that it does not open over every test; the tests of the tour say otherwise.
"""

from __future__ import annotations

import io
import os
import re
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from browser_launch import launch_browser  # pyrefly: ignore[missing-import]
from PIL import Image
from synth import write_synthetic_subject  # pyrefly: ignore[missing-import]

from surfcensus.app import schemas
from surfcensus.render import geometry
from surfcensus.render.api import render_subject
from surfcensus.render.model import PANEL_ORDER, RenderConfig, RenderInfo
from surfcensus.sources.patterns import resolve_subject_files

sync_api = pytest.importorskip("playwright.sync_api")
pytest.importorskip("uvicorn")
from frontend_mock import (  # noqa: E402  # pyrefly: ignore[missing-import]
    FIRST_ASSIGNMENT,
    MockHandle,
    MockState,
    MockSubject,
    start_mock,
    subject_from_media,
)

pytestmark = pytest.mark.browser

MEDIA_ENV = "SURFCENSUS_FRONTEND_MEDIA"
WIDE = {"width": 1400, "height": 900}
NARROW = {"width": 400, "height": 800}
SEEN_TOUR = "try { localStorage.setItem('surfcensus.tour.seen:local', '1'); } catch (e) {}"
"""Run before the scripts of a page: the rater of the mock server has seen the tour."""
TOUR = ("views", "rating", "marks", "moving", "help")


@pytest.fixture(scope="module")
def playwright() -> Iterator[Any]:
    with sync_api.sync_playwright() as started:
        yield started


@pytest.fixture(scope="module")
def browser(playwright: Any) -> Iterator[Any]:
    started = launch_browser(playwright)
    yield started
    started.close()


@pytest.fixture(scope="module")
def subjects(tmp_path_factory: pytest.TempPathFactory) -> list[MockSubject]:
    named = os.environ.get(MEDIA_ENV)
    if named:
        media = [Path(part) for part in named.split(os.pathsep) if part]
    else:
        root = tmp_path_factory.mktemp("frontend")
        media = []
        for position in range(3):
            subject = write_synthetic_subject(
                root / "data",
                name=f"sub-{position:03d}",
                subdivisions=4,
                midline_x=-4.0 + 1.5 * position,
            )
            out = root / "media" / f"{position}"
            render_subject(resolve_subject_files(subject.path), out, RenderConfig(), video=False)
            media.append(out)
    assert len(media) >= 3, f"{MEDIA_ENV} names fewer than three media directories"
    return [subject_from_media(path, position) for position, path in enumerate(media)]


@pytest.fixture
def state(subjects: list[MockSubject]) -> MockState:
    return MockState(list(subjects))


@pytest.fixture
def server(state: MockState) -> Iterator[MockHandle]:
    handle = start_mock(state)
    yield handle
    handle.stop()


class Rater:
    """A page on the mock server, and what a test asks of it."""

    def __init__(self, page: Any, server: MockHandle, *, seen_tour: bool = True) -> None:
        self.page = page
        self.server = server
        self.state = server.state
        if seen_tour:
            page.add_init_script(SEEN_TOUR)
        self.console: list[str] = []
        self.failed: list[str] = []
        self.requested: list[str] = []
        page.on("console", lambda m: self.console.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda error: self.console.append(str(error)))
        page.on("requestfailed", lambda request: self.failed.append(request.url))
        page.on("request", lambda request: self.requested.append(request.url))

    def open(self, path: str = "") -> Rater:
        self.page.goto(self.server.url + path)
        return self

    def ready(self, position: int | None = None) -> Rater:
        """Wait for a subject with all its tiles; with ``position``, for that subject."""
        self.page.wait_for_selector("#viewer[data-ready='true']")
        if position is not None:
            uid = self.state.subjects[position].subject_uid
            self.page.wait_for_function(
                "uid => document.querySelector('.tile-gray').src.includes(uid)"
                " && document.getElementById('viewer').dataset.ready === 'true'",
                arg=uid,
            )
        return self

    def paused(self, position: int | None = None) -> Rater:
        self.ready(position)
        if self.viewer["playing"] == "true":
            self.press(" ")
        return self

    def press(self, *keys: str) -> None:
        for key in keys:
            self.page.keyboard.press(key)

    @property
    def viewer(self) -> dict[str, str]:
        return self.page.evaluate("({ ...document.getElementById('viewer').dataset })")

    @property
    def frame(self) -> int:
        return int(self.viewer["frame"])

    @property
    def hint(self) -> str:
        return self.page.inner_text("#hint")

    @property
    def page_state(self) -> str:
        return self.page.get_attribute("#rate", "data-state")

    @property
    def refusal(self) -> str:
        """What the page says beside the submit button after a refused Enter; empty when it
        says nothing."""
        return self.page.inner_text("#refusal") if self.page.is_visible("#refusal") else ""

    def lacking(self) -> dict[str, str]:
        """Criterion -> what its row is marked as lacking."""
        found = self.page.eval_on_selector_all(
            ".criterion[data-lacking]:not([data-lacking=''])",
            "all => all.map((row) => [row.dataset.criterion, row.dataset.lacking])",
        )
        return dict(found)

    @property
    def tour_step(self) -> str | None:
        """The step the tour shows, or None when it is closed."""
        if self.page.is_hidden("#tour"):
            return None
        return self.page.get_attribute("#tour", "data-step")

    def box(self, selector: str) -> dict[str, float]:
        return self.page.locator(selector).first.bounding_box()

    @property
    def progress(self) -> str:
        return self.page.inner_text("#progress-line")

    def wait_skips(self, n: int) -> None:
        for _ in range(200):
            if len(self.state.skips) >= n:
                return
            self.page.wait_for_timeout(25)
        raise AssertionError(f"{len(self.state.skips)} skips arrived, {n} were expected")

    def events_of(self, kind: str) -> list[dict[str, Any]]:
        return [
            event.payload
            for batch in self.state.events
            for event in batch.events
            if event.type == kind
        ]

    def seek(self, frame: int) -> None:
        self.page.evaluate(
            "frame => { const slider = document.getElementById('frame'); slider.value = frame;"
            " slider.dispatchEvent(new Event('input')); }",
            frame,
        )

    def pressed(self) -> dict[str, int]:
        """Criterion -> the value whose button is pressed."""
        found = self.page.eval_on_selector_all(
            ".level[aria-pressed='true']",
            "all => all.map((button) => [button.dataset.criterion, Number(button.dataset.value)])",
        )
        return dict(found)

    def stage(self, panel: str) -> dict[str, float]:
        return self.page.locator(f".panel[data-panel='{panel}'] .stage").bounding_box()

    def click_tile(
        self, panel: str, px: float, py: float, *, tile: int = 320, zoom: int = 1
    ) -> None:
        """Click where the tile pixel (px, py) of a panel is on screen."""
        box = self.stage(panel)
        scale = zoom * box["width"] / tile
        x = box["x"] + box["width"] / 2 + (px - tile / 2) * scale
        y = box["y"] + box["height"] / 2 + (py - tile / 2) * scale
        self.page.mouse.click(x, y)

    def screen_pixel(self, *, tile: int = 320) -> float:
        """The width of a screen pixel in tile pixels. The browser reports a click at a whole
        screen pixel, so that is how far a stored mark can lie from where a test aimed."""
        return tile / self.stage("sagittal")["width"] + 1e-6

    def marks(self, panel: str) -> list[dict[str, Any]]:
        return self.page.eval_on_selector_all(
            f".panel[data-panel='{panel}'] g.mark",
            """all => all.map((mark) => {
                const numbers = mark.getAttribute('transform').match(/-?[\\d.]+(?:e-?\\d+)?/g);
                const [x, y] = numbers.map(Number);
                return { id: mark.dataset.clientId, cls: mark.dataset.class,
                         own: mark.dataset.own === 'true',
                         distance: Number(mark.dataset.distance),
                         opacity: Number(mark.getAttribute('opacity')),
                         px: x, py: y, chosen: mark.classList.contains('mark-selected') };
            })""",
        )

    def counts(self) -> dict[str, int]:
        found = self.page.eval_on_selector_all(
            ".mark-count",
            "all => all.map((count) => [count.dataset.class, Number(count.dataset.count)])",
        )
        return dict(found)

    def wait_ratings(self, n: int) -> None:
        self.page.wait_for_function("() => true")
        for _ in range(200):
            if len(self.state.ratings) >= n:
                return
            self.page.wait_for_timeout(25)
        raise AssertionError(f"{len(self.state.ratings)} ratings arrived, {n} were expected")

    def wait_events(self, wanted: Callable[[list[str]], bool]) -> list[str]:
        for _ in range(200):
            types = self.state.event_types()
            if wanted(types):
                return types
            self.page.wait_for_timeout(25)
        raise AssertionError(f"the events did not arrive: {self.state.event_types()}")


@pytest.fixture
def rater(browser: Any, server: MockHandle) -> Iterator[Rater]:
    page = browser.new_page(viewport=WIDE)
    yield Rater(page, server)
    page.close()


def info_of(state: MockState, position: int = 0) -> RenderInfo:
    return state.subjects[position].info


def tiles_of(info: RenderInfo) -> int:
    return 2 * sum(len(panel.slices) for panel in info.panels.values())


# --- loading ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "viewport", [WIDE, {"width": 700, "height": 600}, {"width": 2200, "height": 1100}]
)
def test_the_page_shows_three_square_panels_side_by_side(
    browser: Any, server: MockHandle, viewport: dict[str, int]
) -> None:
    page = browser.new_page(viewport=viewport)
    # paused: while it plays, the image elements of a panel change with every frame
    rater = Rater(page, server).open().paused()
    info = info_of(server.state)
    boxes = [rater.stage(name) for name in PANEL_ORDER]
    side = boxes[0]["width"]
    assert side >= 150
    for box in boxes:
        assert box["width"] == box["height"] == side
        assert box["y"] == boxes[0]["y"]
        assert 0 <= box["x"] and box["x"] + side <= viewport["width"]
    assert boxes[0]["x"] + side <= boxes[1]["x"] and boxes[1]["x"] + side <= boxes[2]["x"]
    # as large as the window allows: the three share its width, or the height is used up
    gaps = viewport["width"] - 3 * side
    assert (
        gaps < 60
        or page.evaluate("document.documentElement.scrollHeight") <= viewport["height"] + 1
    )
    for name in PANEL_ORDER:
        sizes = page.eval_on_selector_all(
            f".panel[data-panel='{name}'] img",
            "all => all.map((image) => [image.className, image.naturalWidth, image.naturalHeight,"
            " image.getBoundingClientRect().width])",
        )
        panel = info.panels[name]
        assert sorted(sizes) == [
            ["tile tile-contours", panel.width, panel.height, side],
            ["tile tile-gray", panel.width, panel.height, side],
        ]
    assert rater.console == [] and rater.failed == []
    page.close()


def test_the_page_asks_its_own_server_and_nothing_else(rater: Rater, state: MockState) -> None:
    rater.open().ready()
    rater.press("?")
    rater.press("Escape", "1", "4", "7", "Enter")
    rater.ready(1)
    # leaving a page ends the requests that are on their way: let the tiles ahead arrive first
    rater.page.wait_for_load_state("networkidle")
    rater.open("review")
    rater.page.wait_for_selector("#review[data-state='open']")
    rater.page.locator("#subjects tbody tr").first.click()
    rater.ready()
    assert rater.console == [] and rater.failed == []
    assert rater.requested and all(url.startswith(rater.server.url) for url in rater.requested)
    assert state.hosts == {f"127.0.0.1:{rater.server.port}"}
    # the routes of the schemas, and no other path
    patterns = [
        re.compile(
            re.sub(
                r"\\\{path\\\}",
                ".+",
                re.sub(r"\\\{subject_uid\\\}", "[0-9a-f]{16}", re.escape(route.path)),
            )
        )
        for route in schemas.ROUTES
    ]
    for _, path in state.requests:
        path = path.split("?")[0]
        assert any(pattern.fullmatch(path) for pattern in patterns), path
    assert not any(
        path.endswith(("render.json", "vertex_map.npz", "favicon.ico"))
        for _, path in state.requests
    )


@pytest.mark.parametrize("cache_media", [True, False])
def test_no_tile_is_requested_twice(rater: Rater, state: MockState, cache_media: bool) -> None:
    state.cache_media = cache_media
    rater.open().paused(0)
    info = info_of(state)
    # every frame of both modes, then the next subject and back again
    for frame in range(0, len(info.modes["sync"].frames), 3):
        rater.seek(frame)
    rater.press("m")
    for frame in range(0, len(info.modes["seq"].frames), 5):
        rater.seek(frame)
    rater.press(" ")
    rater.page.wait_for_timeout(300)
    rater.press("1", "4", "7", "Enter")
    rater.paused(1)
    rater.press("Backspace")
    rater.paused(0)
    rater.press("Enter")
    rater.ready(1)
    rater.page.wait_for_load_state("networkidle")
    media = Counter(path for _, path in state.requests if path.startswith("/media/"))
    assert max(media.values()) == 1, [path for path, n in media.items() if n > 1][:5]
    # the subject on screen and the two that come next, every tile of each, and nothing else
    assert len(media) == sum(tiles_of(info_of(state, position)) for position in range(3))
    for position in range(3):
        subject = state.subjects[position]
        assert all(
            path.endswith(f"?v={subject.render_key}")
            for path in media
            if subject.subject_uid in path
        )
    assert rater.console == [] and rater.failed == []


def test_tiles_of_the_next_subjects_are_there_before_they_are_shown(
    rater: Rater, state: MockState
) -> None:
    rater.open().paused(0)
    rater.page.wait_for_load_state("networkidle")
    before = len(state.paths("/media/"))
    assert before == sum(tiles_of(info_of(state, position)) for position in range(3))
    rater.press("1", "4", "7", "Enter")
    rater.ready(1)
    rater.page.wait_for_load_state("networkidle")
    # the second subject needed no request for a tile; its successor has none left to load
    assert len(state.paths("/media/")) == before


# --- playback --------------------------------------------------------------------------------


def test_space_plays_and_pauses_and_the_slider_follows(rater: Rater, state: MockState) -> None:
    rater.open().ready()
    assert rater.viewer["playing"] == "true" and rater.viewer["mode"] == "sync"
    rater.page.wait_for_function("document.getElementById('viewer').dataset.frame >= 3")
    rater.press(" ")
    assert rater.viewer["playing"] == "false"
    frame = rater.frame
    rater.page.wait_for_timeout(300)
    assert rater.frame == frame
    assert int(rater.page.input_value("#frame")) == frame
    assert (
        rater.page.inner_text("#frame-label")
        == f"frame {frame + 1} / {len(info_of(state).modes['sync'].frames)}"
    )
    rater.press(" ")
    assert rater.viewer["playing"] == "true"
    rater.page.wait_for_function(
        "frame => document.getElementById('viewer').dataset.frame != frame", arg=str(frame)
    )


def test_playback_runs_at_the_frame_rate_of_the_render_and_changes_with_the_speed(
    rater: Rater, state: MockState
) -> None:
    rater.open().ready()
    fps = info_of(state).config.fps

    def frames_in(ms: int) -> int:
        count = len(info_of(state).modes["sync"].frames)
        start = rater.frame
        rater.page.wait_for_timeout(ms)
        return (rater.frame - start) % count

    at_1x = frames_in(1500)
    assert 0.6 * 1.5 * fps <= at_1x <= 1.25 * 1.5 * fps
    rater.press("]")
    assert rater.viewer["speed"] == "2" and rater.page.inner_text("#speed") == "2x"
    assert frames_in(1500) > 1.4 * at_1x
    rater.press("[", "[", "[")
    assert rater.viewer["speed"] == "0.25"
    assert frames_in(1500) <= 0.5 * at_1x
    # the ends of the range
    rater.press("[", "[")
    assert rater.viewer["speed"] == "0.25"
    rater.press("]", "]", "]", "]", "]", "]")
    assert rater.viewer["speed"] == "4"


def test_the_arrows_step_one_frame_and_ten_with_shift(rater: Rater, state: MockState) -> None:
    rater.open().ready()
    count = len(info_of(state).modes["sync"].frames)
    # an arrow pauses a playing subject
    rater.press("ArrowRight")
    assert rater.viewer["playing"] == "false"
    rater.seek(20)
    rater.press("ArrowRight")
    assert rater.frame == 21
    rater.press("ArrowLeft", "ArrowLeft")
    assert rater.frame == 19
    rater.press("Shift+ArrowRight")
    assert rater.frame == 29
    rater.press("Shift+ArrowLeft", "Shift+ArrowLeft")
    assert rater.frame == 9
    # the loop goes round
    rater.seek(0)
    rater.press("ArrowLeft")
    assert rater.frame == count - 1
    rater.press("ArrowRight")
    assert rater.frame == 0
    rater.press("Shift+ArrowLeft")
    assert rater.frame == count - 10


def test_a_focused_control_does_not_take_a_key_twice(rater: Rater) -> None:
    rater.open().paused()
    rater.seek(20)
    # the slider keeps the focus after a click on it; an arrow still moves one frame
    rater.page.focus("#frame")
    rater.press("ArrowRight")
    assert rater.frame == 21
    rater.press("Shift+ArrowRight")
    assert rater.frame == 31
    # Space on the focused play button plays once, and does not toggle twice
    rater.page.focus("#play")
    rater.press(" ")
    assert rater.viewer["playing"] == "true"
    rater.page.focus("#play")
    rater.press(" ")
    assert rater.viewer["playing"] == "false"
    # Enter on a focused button is submit (here refused: no values), not a press of the button
    rater.page.focus("#mode")
    rater.press("Enter")
    assert rater.viewer["mode"] == "sync" and "Not submitted" in rater.refusal
    # the buttons do what their keys do
    rater.page.click("#mode")
    assert rater.viewer["mode"] == "seq"
    rater.page.click("#play")
    assert rater.viewer["playing"] == "true"
    rater.page.click("#faster")
    assert rater.viewer["speed"] == "2"
    rater.page.click("#zoom")
    assert rater.viewer["zoom"] == "2"


def test_m_switches_between_the_two_modes_and_keeps_the_place(
    rater: Rater, state: MockState
) -> None:
    rater.open().paused()
    info = info_of(state)
    sync, seq = info.modes["sync"].frames, info.modes["seq"].frames
    rater.seek(40)
    rater.press("m")
    assert rater.viewer["mode"] == "seq"
    assert rater.page.inner_text("#mode").endswith("one at a time")
    assert rater.page.input_value("#frame") == str(rater.frame)
    assert rater.page.get_attribute("#frame", "max") == str(len(seq) - 1)
    # the frame of the other mode whose slices are nearest those on screen
    distance = np.abs(np.array(seq) - np.array(sync[40])).sum(axis=1)
    assert rater.frame == int(distance.argmin())
    shown = [
        int(rater.page.get_attribute(f".panel[data-panel='{name}']", "data-slice"))
        for name in PANEL_ORDER
    ]
    assert shown == list(seq[rater.frame])
    # one plane at a time: the next frames move one panel and hold the other two
    rater.seek(5)
    rater.press("ArrowRight")
    assert seq[6][1:] == seq[5][1:] and seq[6][0] != seq[5][0]
    assert rater.page.get_attribute(".panel[data-panel='sagittal']", "data-slice") == str(seq[6][0])
    rater.press("m")
    assert rater.viewer["mode"] == "sync" and rater.page.inner_text("#mode").endswith(
        "synchronized"
    )


def test_the_tiles_on_screen_are_those_of_the_frame_table(rater: Rater, state: MockState) -> None:
    rater.open().paused()
    info = info_of(state)
    subject = state.subjects[0]
    for mode in ("sync", "seq"):
        frames = info.modes[mode].frames
        for frame in (0, 17, len(frames) // 2, len(frames) - 1):
            rater.seek(frame)
            for position, name in enumerate(PANEL_ORDER):
                shown = rater.page.eval_on_selector_all(
                    f".panel[data-panel='{name}'] img",
                    "all => all.map((image) => image.getAttribute('src'))",
                )
                index = frames[frame][position]
                base = schemas.media_base(subject.subject_uid)
                assert sorted(shown) == sorted(
                    [
                        f"{base}{info.tiles.gray_path(name, index)}?v={subject.render_key}",
                        f"{base}{info.tiles.contours_path(name, index)}?v={subject.render_key}",
                    ]
                )
        rater.press("m")


def test_z_zooms_two_times_about_the_centre_of_the_tile(rater: Rater) -> None:
    rater.open().paused()
    box = rater.stage("coronal")
    inner = rater.page.locator(".panel[data-panel='coronal'] .zoomable")
    assert inner.bounding_box() == box
    rater.press("z")
    assert rater.viewer["zoom"] == "2"
    zoomed = inner.bounding_box()
    assert zoomed["width"] == pytest.approx(2 * box["width"]) and zoomed["height"] == pytest.approx(
        2 * box["height"]
    )
    assert zoomed["x"] + zoomed["width"] / 2 == pytest.approx(box["x"] + box["width"] / 2)
    assert zoomed["y"] + zoomed["height"] / 2 == pytest.approx(box["y"] + box["height"] / 2)
    # the panel itself keeps its place and cuts off what is outside
    assert rater.stage("coronal") == box
    rater.press("z")
    assert rater.viewer["zoom"] == "1" and inner.bounding_box() == box


# --- crosshairs and letters --------------------------------------------------------------------


def _blueness(image: np.ndarray) -> np.ndarray:
    return image[..., 2] - (image[..., 0] + image[..., 1]) / 2


def test_crosshairs_and_edge_letters_are_those_of_the_render(
    browser: Any, server: MockHandle
) -> None:
    # a window in which a panel is as large as its tile, so a screenshot is the tile's pixels
    page = browser.new_page(viewport={"width": 996, "height": 1000})
    rater = Rater(page, server).open().paused()
    info = info_of(server.state)
    panels = dict(info.panels)
    assert rater.stage("sagittal")["width"] == info.panels["sagittal"].width == 320
    for mode in ("sync", "seq"):
        frames = info.modes[mode].frames
        for frame in (0, 7, len(frames) // 4, len(frames) // 2 + 3, len(frames) - 1):
            rater.seek(frame)
            expected = geometry.crosshair_pixels(panels, frames[frame])
            for name in PANEL_ORDER:
                column, row = (int(np.floor(value + 1e-6)) for value in expected[name])
                lines = page.eval_on_selector_all(
                    f".panel[data-panel='{name}'] .crosshair line",
                    "all => all.map((line) => ['x1', 'x2', 'y1', 'y2']"
                    ".map((key) => Number(line.getAttribute(key))))",
                )
                assert lines == [
                    [column + 0.5, column + 0.5, 0, 320],
                    [0, 320, row + 0.5, row + 0.5],
                ]
                # and on screen: the bluest pixel column and row of the panel
                shot = page.locator(f".panel[data-panel='{name}'] .stage").screenshot()
                image = np.asarray(Image.open(io.BytesIO(shot)).convert("RGB")).astype(float)
                assert image.shape == (320, 320, 3)
                blue = _blueness(image)
                assert int(blue.mean(axis=0).argmax()) == column, (mode, frame, name)
                assert int(blue.mean(axis=1).argmax()) == row, (mode, frame, name)
        rater.press("m")
    for name in PANEL_ORDER:
        edges = info.panels[name].edges
        letters = page.eval_on_selector_all(
            f".panel[data-panel='{name}'] .edge",
            """all => all.map((edge) => {
                const box = edge.getBoundingClientRect();
                const stage = edge.closest('.stage').getBoundingClientRect();
                return [edge.textContent, (box.x + box.width / 2 - stage.x) / stage.width,
                        (box.y + box.height / 2 - stage.y) / stage.height];
            })""",
        )
        places = {letter: (x, y) for letter, x, y in letters}
        assert set(places) == {edges.left, edges.right, edges.top, edges.bottom}
        assert places[edges.left][0] < 0.08 and places[edges.left][1] == pytest.approx(
            0.5, abs=0.02
        )
        assert places[edges.right][0] > 0.92 and places[edges.right][1] == pytest.approx(
            0.5, abs=0.02
        )
        assert places[edges.top][1] < 0.08 and places[edges.top][0] == pytest.approx(0.5, abs=0.02)
        assert places[edges.bottom][1] > 0.92 and places[edges.bottom][0] == pytest.approx(
            0.5, abs=0.02
        )
    colour = page.evaluate("getComputedStyle(document.querySelector('.crosshair line')).stroke")
    red, green, blue_ = (int(info.config.crosshair_colour[k : k + 2], 16) for k in (1, 3, 5))
    assert colour == f"rgb({red}, {green}, {blue_})"
    opacity = page.evaluate(
        "getComputedStyle(document.querySelector('.crosshair line')).strokeOpacity"
    )
    assert float(opacity) == pytest.approx(info.config.crosshair_alpha)
    page.close()


# --- the keys that act while held ---------------------------------------------------------------


def test_s_hides_the_contours_while_it_is_held(rater: Rater) -> None:
    rater.open().paused()
    contours = ".panel[data-panel='sagittal'] .tile-contours"
    gray = ".panel[data-panel='sagittal'] .tile-gray"
    visibility = "image => getComputedStyle(image).visibility"
    assert rater.page.eval_on_selector(contours, visibility) == "visible"
    rater.page.keyboard.down("s")
    assert rater.viewer["contours"] == "hidden"
    assert rater.page.eval_on_selector(contours, visibility) == "hidden"
    assert rater.page.eval_on_selector(gray, visibility) == "visible"
    # the key repeats while it is down; the contours stay hidden, and a step still works
    rater.page.keyboard.down("s")
    rater.page.keyboard.down("s")
    frame = rater.frame
    rater.press("ArrowRight")
    assert rater.viewer["contours"] == "hidden" and rater.frame == frame + 1
    assert rater.page.eval_on_selector(contours, visibility) == "hidden"
    rater.page.keyboard.up("s")
    assert rater.viewer["contours"] == "shown"
    assert rater.page.eval_on_selector(contours, visibility) == "visible"
    # Ctrl+S is the browser's, not the page's
    rater.page.keyboard.down("Control")
    rater.page.keyboard.down("s")
    assert rater.viewer["contours"] == "shown"
    rater.page.keyboard.up("s")
    rater.page.keyboard.up("Control")
    # a window that loses the focus while the key is down does not keep the contours hidden
    rater.page.keyboard.down("s")
    assert rater.viewer["contours"] == "hidden"
    rater.page.evaluate("window.dispatchEvent(new Event('blur'))")
    assert rater.viewer["contours"] == "shown"
    rater.page.keyboard.up("s")


def test_h_hides_the_marks_while_it_is_held(rater: Rater) -> None:
    rater.open().paused()
    rater.press("a")
    rater.click_tile("sagittal", 150, 160)
    assert len(rater.marks("sagittal")) == 1
    rater.page.keyboard.down("h")
    assert rater.marks("sagittal") == [] and rater.viewer["marks"] == "hidden"
    # a hidden mark cannot be selected or covered by accident: the count stays
    assert rater.counts()["A"] == 1
    rater.page.keyboard.up("h")
    assert len(rater.marks("sagittal")) == 1 and rater.viewer["marks"] == "shown"


# --- rating ----------------------------------------------------------------------------------


def test_the_rating_bar_is_the_protocol_with_its_keys(rater: Rater, state: MockState) -> None:
    rater.open().ready()
    protocol = state.protocol
    rows = rater.page.eval_on_selector_all(
        ".criterion:not(.criterion-other)",
        """all => all.map((row) => [row.dataset.criterion,
            row.querySelector('.criterion-label').textContent,
            row.querySelector('.criterion-label').title,
            [...row.querySelectorAll('.level')].map((level) => [
                level.querySelector('kbd').textContent,
                level.textContent.trim(), Number(level.dataset.value), level.title]),
            row.querySelector('.class-key kbd').textContent])""",
    )
    assert rows == [
        [
            criterion.id,
            f"{criterion.id} {criterion.label}",
            criterion.question,
            [
                [key, f"{key} {level.label}", level.value, level.help]
                for key, level in zip(criterion.keys, protocol.scale, strict=True)
            ],
            criterion.mark_key,
        ]
        for criterion in protocol.criteria
    ]
    assert rater.page.inner_text(".criterion-other .class-key kbd") == protocol.other_mark_key
    assert rater.page.is_disabled("#submit")
    # nothing about the subject is on the page
    text = rater.page.inner_text("body")
    subject = state.subjects[0]
    assert subject.subject_uid not in text and "D-" not in text and "sub-" not in text


def test_the_criterion_keys_set_values_and_enter_submits_when_all_are_set(
    rater: Rater, state: MockState
) -> None:
    rater.open().paused()
    assert rater.page.is_disabled("#submit")
    assert "no value yet for A, B, C" in rater.hint
    rater.press("Enter")
    rater.page.wait_for_timeout(150)
    assert state.ratings == [] and "Not submitted" in rater.refusal
    rater.press("3")
    assert rater.pressed() == {"A": 2} and rater.page.is_disabled("#submit")
    assert "no value yet for B, C" in rater.hint
    rater.press("2")
    assert rater.pressed() == {"A": 1}
    rater.press("4", "9")
    assert rater.pressed() == {"A": 1, "B": 0, "C": 2}
    assert rater.page.is_enabled("#submit") and rater.hint == "Ready: Enter submits."
    progress = rater.page.inner_text("#progress-line")
    assert progress == "0 done today · 0 done in total · 3 remaining"
    rater.press("Enter")
    rater.ready(1)
    rating = state.ratings[0]
    assert rating.assignment_id == FIRST_ASSIGNMENT and rating.values == {"A": 1, "B": 0, "C": 2}
    assert rating.protocol_version == state.protocol.version
    assert (
        rating.unratable is None and not rating.other and rating.note == "" and rating.marks == ()
    )
    assert rating.client_ts is not None and 0 < rating.ms_on_subject < 60_000
    # the next subject starts clean and plays
    assert rater.pressed() == {} and rater.page.is_disabled("#submit")
    assert rater.viewer["playing"] == "true" and rater.frame >= 0
    assert rater.page.inner_text("#progress-line") == "1 done today · 1 done in total · 2 remaining"
    assert rater.page.get_attribute("#progress-line", "data-done-total") == "1"
    assert rater.console == []


def test_the_buttons_of_the_rating_bar_do_what_the_keys_do(rater: Rater, state: MockState) -> None:
    rater.open().paused()
    rater.page.click(".level[data-criterion='A'][data-value='0']")
    rater.page.click(".level[data-criterion='B'][data-value='1']")
    rater.page.click(".level[data-criterion='C'][data-value='0']")
    assert rater.pressed() == {"A": 0, "B": 1, "C": 0}
    assert rater.page.get_attribute("#active-class", "data-class") == "B"
    rater.page.click(".class-key[data-class='X']")
    assert rater.page.get_attribute("#active-class", "data-class") == "X"
    # a pressed button does not keep the focus: Space still plays
    rater.press(" ")
    assert rater.viewer["playing"] == "true"
    rater.page.click("#submit")
    rater.ready(1)
    assert state.ratings[0].values == {"A": 0, "B": 1, "C": 0}


def test_a_positive_value_and_the_class_keys_set_the_active_mark_class(rater: Rater) -> None:
    rater.open().paused()
    active = lambda: rater.page.get_attribute("#active-class", "data-class")  # noqa: E731
    assert active() == "" and rater.viewer["activeClass"] == ""
    rater.press("1")  # none: no class
    assert active() == ""
    rater.press("5")  # B minor
    assert active() == "B" and rater.viewer["activeClass"] == "B"
    assert rater.page.inner_text("#active-class") == "marking B"
    rater.press("9")
    assert active() == "C"
    rater.press("4")  # B back to none: the class stays what it was
    assert active() == "C"
    for key, wanted in (("a", "A"), ("b", "B"), ("c", "C"), ("x", "X"), ("A", "A")):
        rater.press(key)
        assert active() == wanted
        assert (
            rater.page.get_attribute(f".class-key[data-class='{wanted}']", "aria-pressed") == "true"
        )
    cursor = rater.page.eval_on_selector(".stage", "stage => getComputedStyle(stage).cursor")
    assert cursor.startswith('url("data:image/svg+xml,') and cursor.endswith("crosshair")


# --- marks -----------------------------------------------------------------------------------


@pytest.mark.parametrize("viewport", [WIDE, {"width": 700, "height": 600}])
def test_a_click_puts_a_mark_where_it_was_clicked(
    browser: Any, server: MockHandle, state: MockState, viewport: dict[str, int]
) -> None:
    page = browser.new_page(viewport=viewport)
    rater = Rater(page, server).open().ready()
    info = info_of(state)
    assert rater.viewer["playing"] == "true"
    # without a class a click pauses and places nothing
    rater.click_tile("coronal", 100, 100)
    assert rater.viewer["playing"] == "false"
    assert rater.marks("coronal") == [] and "Choose what to mark first" in rater.hint
    rater.press(" ")
    rater.page.wait_for_function("document.getElementById('viewer').dataset.frame >= 6")
    rater.press("3")
    # a click on a playing subject pauses it and marks the frame that was showing
    rater.click_tile("sagittal", 131.5, 88.25)
    assert rater.viewer["playing"] == "false"
    frame = rater.frame
    slices = info.modes["sync"].frames[frame]
    rater.press("5")
    rater.click_tile("coronal", 250.0, 40.5)
    rater.press("x")
    rater.click_tile("axial", 12.25, 300.0)
    # 2x zoom: the tile pixel under a click is still the one that is stored
    rater.press("c", "z")
    rater.click_tile("axial", 200.5, 170.25, zoom=2)
    rater.press("z")
    # the edge of a tile is still on the tile
    rater.press("a")
    box = rater.stage("coronal")
    page.mouse.click(box["x"] + 0.4, box["y"] + box["height"] - 0.4)
    assert rater.counts() == {"A": 2, "B": 1, "C": 1, "X": 1}
    rater.press("7", "Enter")
    rater.ready(1)
    marks = state.ratings[0].marks
    tolerance = rater.screen_pixel()
    wanted = [
        ("n1", "A", "sagittal", 131.5, 88.25),
        ("n2", "B", "coronal", 250.0, 40.5),
        ("n3", "X", "axial", 12.25, 300.0),
        ("n4", "C", "axial", 200.5, 170.25),
    ]
    for mark, (client_id, mark_class, panel, px, py) in zip(marks[:4], wanted, strict=True):
        assert (mark.client_id, mark.mark_class, mark.panel) == (client_id, mark_class, panel)
        assert mark.px == pytest.approx(px, abs=tolerance) and mark.py == pytest.approx(
            py, abs=tolerance
        )
        assert mark.slice == slices[PANEL_ORDER.index(panel)]
        assert (mark.frame, mark.mode) == (frame, "sync") and mark.ts is not None
    edge = marks[4]
    assert 0 <= edge.px <= 2 * tolerance and 320 - 2 * tolerance <= edge.py <= 320
    assert state.ratings[0].values == {"A": 2, "B": 1, "C": 0}
    page.close()


def test_a_mark_is_drawn_at_its_point_in_its_class_and_the_last_one_is_selected(
    rater: Rater,
) -> None:
    rater.open().paused()
    rater.seek(30)
    rater.press("b")
    rater.click_tile("coronal", 111.0, 222.0)
    rater.press("a")
    rater.click_tile("coronal", 200.0, 100.0)
    drawn = rater.marks("coronal")
    assert [(mark["id"], mark["cls"], mark["own"], mark["chosen"]) for mark in drawn] == [
        ("n1", "B", True, False),
        ("n2", "A", True, True),
    ]
    near = rater.screen_pixel()
    for mark, (px, py) in zip(drawn, [(111.0, 222.0), (200.0, 100.0)], strict=True):
        assert mark["px"] == pytest.approx(px, abs=near) and mark["py"] == pytest.approx(
            py, abs=near
        )
        assert mark["distance"] == pytest.approx(0, abs=1e-6) and mark["opacity"] == 1
    # on screen the ring is centred on the clicked point, in the colour of its class
    box = rater.stage("coronal")
    ring = rater.page.locator(".panel[data-panel='coronal'] g.mark[data-client-id='n1'] .mark-ring")
    ring_box = ring.bounding_box()
    scale = box["width"] / 320
    assert ring_box["x"] + ring_box["width"] / 2 == pytest.approx(box["x"] + 111.0 * scale, abs=1.5)
    assert ring_box["y"] + ring_box["height"] / 2 == pytest.approx(
        box["y"] + 222.0 * scale, abs=1.5
    )
    colours = rater.page.eval_on_selector_all(
        ".panel[data-panel='coronal'] g.mark .mark-ring",
        "all => all.map((ring) => getComputedStyle(ring).stroke)",
    )
    assert len(set(colours)) == 2
    labels = rater.page.eval_on_selector_all(
        ".panel[data-panel='coronal'] g.mark text", "all => all.map((label) => label.textContent)"
    )
    assert labels == ["B", "A"]


def test_a_mark_shows_in_the_other_panels_within_3_mm_and_fades_with_distance(
    rater: Rater, state: MockState
) -> None:
    state.mark_show_mm = 3.0
    rater.open().paused()
    info = info_of(state)
    panels = dict(info.panels)
    sync = info.modes["sync"].frames
    rater.seek(30)
    rater.press("a")
    px, py = 171.0, 141.0
    rater.click_tile("sagittal", px, py)
    placed = rater.marks("sagittal")[0]
    slice_pos = sync[30][0]
    vox = geometry.pixel_to_voxel(panels["sagittal"], slice_pos, placed["px"], placed["py"])
    tkr = geometry.apply_affine(np.array(info.volume.vox2ras_tkr), vox)
    rater.press("m")  # one plane at a time: each panel passes the point on its own
    seq = info.modes["seq"].frames
    seen: dict[str, set[float]] = {name: set() for name in PANEL_ORDER}
    for frame in range(len(seq)):
        rater.seek(frame)
        for position, name in enumerate(PANEL_ORDER):
            panel = panels[name]
            distance = abs(float(tkr[panel.world_axis]) - panel.slice_mm[seq[frame][position]])
            drawn = rater.marks(name)
            if distance > 3.0:
                assert drawn == [], (frame, name, distance)
                continue
            assert len(drawn) == 1, (frame, name, distance)
            mark = drawn[0]
            where = geometry.voxel_to_pixel(panel, vox)
            assert mark["px"] == pytest.approx(float(where[0]), abs=1e-6)
            assert mark["py"] == pytest.approx(float(where[1]), abs=1e-6)
            assert mark["distance"] == pytest.approx(distance, abs=2e-3)
            assert mark["opacity"] == pytest.approx(1 - 0.8 * distance / 3.0, abs=2e-3)
            assert mark["own"] == (name == "sagittal")
            seen[name].add(round(distance, 3))
    # every panel passed the point, at several distances, and fainter the farther away
    for name in PANEL_ORDER:
        assert len(seen[name]) >= 3 and min(seen[name]) < 0.51 and max(seen[name]) > 2.0, (
            name,
            seen[name],
        )


def test_the_distance_a_mark_shows_within_is_the_setting_of_the_session(
    rater: Rater, state: MockState
) -> None:
    state.mark_show_mm = 1.0
    rater.open().paused()
    rater.seek(30)
    rater.press("a")
    rater.click_tile("sagittal", 171.0, 141.0)
    shown = 0
    for _ in range(8):
        distances = [mark["distance"] for mark in rater.marks("sagittal")]
        assert all(distance <= 1.0 + 1e-6 for distance in distances)
        shown += len(distances)
        rater.press("ArrowRight")
    # on its slice and one millimetre from it, and no farther
    assert shown == 2


def test_a_click_on_a_mark_selects_it_and_delete_removes_it(rater: Rater, state: MockState) -> None:
    rater.open().paused()
    rater.seek(30)
    rater.press("a")
    rater.click_tile("sagittal", 100.0, 100.0)
    rater.click_tile("sagittal", 200.0, 200.0)
    rater.click_tile("coronal", 150.0, 150.0)
    assert rater.counts()["A"] == 3
    # Delete without a selection... the last mark placed is the selected one
    chosen = [mark["id"] for name in PANEL_ORDER for mark in rater.marks(name) if mark["chosen"]]
    assert set(chosen) == {"n3"}
    # a click on an existing mark selects it and places no new one
    rater.click_tile("sagittal", 100.0, 100.0)
    assert rater.counts()["A"] == 3
    assert [mark["id"] for mark in rater.marks("sagittal") if mark["chosen"]] == ["n1"]
    rater.press("Delete")
    assert rater.counts()["A"] == 2
    assert [mark["id"] for mark in rater.marks("sagittal") if mark["own"]] == ["n2"]
    # nothing is selected now: Delete removes nothing more
    rater.press("Delete")
    assert rater.counts()["A"] == 2
    # Escape drops a selection
    rater.click_tile("sagittal", 200.0, 200.0)
    assert [mark["id"] for mark in rater.marks("sagittal") if mark["chosen"]] == ["n2"]
    rater.press("Escape", "Delete")
    assert rater.counts()["A"] == 2
    rater.press("1", "4", "7", "Enter")
    rater.wait_ratings(1)
    assert [mark.client_id for mark in state.ratings[0].marks] == ["n2", "n3"]


@pytest.mark.parametrize("modifier", ["Control", "Meta"])
def test_ctrl_z_undoes_the_last_mark(rater: Rater, modifier: str) -> None:
    rater.open().paused()
    rater.seek(30)
    rater.press("a")
    rater.click_tile("sagittal", 100.0, 100.0)
    rater.press("b")
    rater.click_tile("coronal", 150.0, 150.0)
    rater.click_tile("axial", 50.0, 250.0)
    assert rater.counts() == {"A": 1, "B": 2, "C": 0, "X": 0}
    rater.press(f"{modifier}+z")
    assert rater.counts() == {"A": 1, "B": 1, "C": 0, "X": 0}
    assert rater.marks("axial") == [] or all(not mark["own"] for mark in rater.marks("axial"))
    rater.press(f"{modifier}+z", f"{modifier}+z", f"{modifier}+z")
    assert rater.counts() == {"A": 0, "B": 0, "C": 0, "X": 0}
    # z alone is zoom, and undoes nothing
    rater.click_tile("sagittal", 100.0, 100.0)
    rater.press("z")
    assert rater.counts()["B"] == 1 and rater.viewer["zoom"] == "2"


def test_marks_are_required_for_a_positive_value_when_the_session_says_so(
    rater: Rater, state: MockState
) -> None:
    state.require_marks = True
    rater.open().paused()
    rater.press("3", "5", "7")
    assert rater.page.is_disabled("#submit")
    assert "criterion A is rated major but has no mark of class A" in rater.hint
    assert "criterion B is rated minor but has no mark of class B" in rater.hint
    rater.press("Enter")
    rater.page.wait_for_timeout(150)
    assert state.ratings == []
    rater.press("a")
    rater.click_tile("sagittal", 120.0, 130.0)
    assert (
        rater.page.is_disabled("#submit")
        and "class B" in rater.hint
        and "class A" not in rater.hint
    )
    # a mark of another class does not count
    rater.press("x")
    rater.click_tile("coronal", 60.0, 60.0)
    assert rater.page.is_disabled("#submit")
    rater.press("b")
    rater.click_tile("coronal", 160.0, 160.0)
    assert rater.page.is_enabled("#submit")
    # taking the mark away takes Enter away again
    rater.press("Control+z")
    assert rater.page.is_disabled("#submit")
    rater.press("4")  # B is none after all: no mark is needed for it
    assert rater.page.is_enabled("#submit")
    rater.press("Enter")
    rater.wait_ratings(1)
    assert [mark.mark_class for mark in state.ratings[0].marks] == ["A", "X"]
    assert rater.console == []


def test_marks_are_optional_when_the_session_does_not_require_them(
    rater: Rater, state: MockState
) -> None:
    rater.open().paused()
    rater.press("3", "6", "9", "Enter")
    rater.wait_ratings(1)
    assert state.ratings[0].values == {"A": 2, "B": 2, "C": 2} and state.ratings[0].marks == ()


# --- unratable, other, note ----------------------------------------------------------------------


def test_u_asks_for_a_reason_and_the_digits_then_choose_it(rater: Rater, state: MockState) -> None:
    rater.open().ready()
    reasons = list(state.protocol.unratable_reasons)
    rater.press("u")
    assert rater.page.is_visible("#unratable-picker")
    assert rater.viewer["playing"] == "false"
    listed = rater.page.eval_on_selector_all(
        ".reason", "all => all.map((reason) => reason.textContent.trim())"
    )
    assert listed == [f"{number + 1} {reason}" for number, reason in enumerate(reasons)]
    # the keys of the page are not the page's while the reasons are open
    rater.press(" ", "s", "m", "z", "a", "Backspace", "9")
    assert (
        rater.viewer["playing"] == "false"
        and rater.viewer["mode"] == "sync"
        and rater.viewer["zoom"] == "1"
    )
    assert rater.pressed() == {} and state.paths("/api/back") == []
    assert rater.page.is_visible("#unratable-picker")
    rater.press("Escape")
    assert rater.page.is_hidden("#unratable-picker") and rater.page.is_disabled("#submit")
    rater.press("u", "2")
    assert rater.page.is_hidden("#unratable-picker")
    assert rater.pressed() == {}, "the digit chose a reason, not a value"
    assert rater.page.inner_text("#unratable").endswith(f"unratable: {reasons[1]}")
    assert rater.page.is_enabled("#submit")
    # u again takes it back
    rater.press("u")
    assert rater.page.is_hidden("#unratable-picker") and rater.page.is_disabled("#submit")
    # arrows and Enter
    rater.press("u", "ArrowDown", "ArrowDown", "ArrowUp", "ArrowUp", "ArrowUp", "Enter")
    assert rater.page.inner_text("#unratable").endswith(f"unratable: {reasons[-1]}")
    rater.press("Enter")
    rater.ready(1)
    rating = state.ratings[0]
    assert rating.unratable == reasons[-1] and rating.values == {} and rating.marks == ()
    # the next subject is not unratable
    assert rater.page.get_attribute("#unratable", "aria-pressed") == "false"


def test_a_value_takes_unratable_back_and_marks_of_an_unratable_subject_are_not_sent(
    rater: Rater, state: MockState
) -> None:
    rater.open().paused()
    rater.press("a")
    rater.click_tile("sagittal", 120.0, 130.0)
    rater.press("u", "1")
    assert rater.page.is_enabled("#submit")
    rater.press("3")
    assert rater.page.get_attribute("#unratable", "aria-pressed") == "false"
    assert rater.page.is_disabled("#submit") and rater.pressed() == {"A": 2}
    rater.press("u", "4", "Enter")
    rater.wait_ratings(1)
    assert state.ratings[0].unratable == state.protocol.unratable_reasons[3]
    assert state.ratings[0].values == {} and state.ratings[0].marks == ()


def test_the_note_field_takes_every_key_until_escape(rater: Rater, state: MockState) -> None:
    rater.open().ready()
    assert rater.page.is_hidden("#note-box")
    rater.press("n")
    assert (
        rater.page.is_visible("#note")
        and rater.page.evaluate("document.activeElement.id") == "note"
    )
    before = rater.viewer
    typed = "s3 u?ab mzho[] 159"
    rater.page.keyboard.type(typed)
    rater.press("Backspace", "Enter")
    rater.page.keyboard.type("x")
    rater.press("Delete", "ArrowLeft", "ArrowRight", " ")
    # every key went into the note, and none did anything else
    assert rater.page.input_value("#note") == typed[:-1] + "\nx "
    after = rater.viewer
    for key in ("playing", "mode", "zoom", "contours", "speed", "activeClass"):
        assert after[key] == before[key], key
    assert rater.pressed() == {} and rater.counts() == {"A": 0, "B": 0, "C": 0, "X": 0}
    assert rater.page.is_hidden("#help") and rater.page.is_hidden("#unratable-picker")
    assert rater.page.get_attribute("#other", "aria-pressed") == "false"
    assert state.paths("/api/back") == [] and state.ratings == []
    # Escape gives the keys back
    rater.press("Escape")
    assert rater.page.evaluate("document.activeElement.tagName") == "BODY"
    assert rater.page.is_visible("#note-box")
    rater.press("3", "4", "7")
    assert rater.pressed() == {"A": 2, "B": 0, "C": 0}
    assert rater.page.input_value("#note") == typed[:-1] + "\nx "
    rater.press("Enter")
    rater.wait_ratings(1)
    assert state.ratings[0].note == typed[:-1] + "\nx"
    rater.ready(1)
    assert rater.page.is_hidden("#note-box") and rater.page.input_value("#note") == ""


def test_an_empty_note_closes_again_and_n_goes_back_into_an_open_one(rater: Rater) -> None:
    rater.open().paused()
    rater.press("n")
    rater.press("Escape")
    assert rater.page.is_hidden("#note-box")
    rater.press("n")
    rater.page.keyboard.type("dura")
    rater.press("Escape")
    assert rater.page.is_visible("#note-box")
    rater.press("n")
    assert rater.page.evaluate("document.activeElement.id") == "note"
    rater.page.keyboard.type("!")
    assert rater.page.input_value("#note") == "dura!"


def test_o_flags_another_problem_and_wants_a_note(rater: Rater, state: MockState) -> None:
    rater.open().paused()
    rater.press("1", "4", "7")
    assert rater.page.is_enabled("#submit")
    rater.press("o")
    assert rater.page.get_attribute("#other", "aria-pressed") == "true"
    assert rater.page.evaluate("document.activeElement.id") == "note"
    rater.press("Escape")
    assert rater.page.is_disabled("#submit") and "needs a note" in rater.hint
    rater.press("Enter")
    rater.page.wait_for_timeout(150)
    assert state.ratings == []
    rater.press("n")
    rater.page.keyboard.type("skull in the box")
    rater.press("Escape")
    assert rater.page.is_enabled("#submit")
    rater.press("Enter")
    rater.wait_ratings(1)
    assert state.ratings[0].other and state.ratings[0].note == "skull in the box"
    # o again takes the flag back
    rater.paused(1)
    rater.press("o", "Escape", "o")
    assert rater.page.get_attribute("#other", "aria-pressed") == "false"


# --- help ------------------------------------------------------------------------------------


def test_the_help_screen_has_the_scale_the_criteria_and_the_keys(
    rater: Rater, state: MockState
) -> None:
    rater.open().paused()
    assert rater.page.is_hidden("#help")
    rater.press("?")
    assert rater.page.is_visible("#help")
    text = rater.page.inner_text("#help")
    for level in state.protocol.scale:
        assert f"{level.value} {level.label}" in text and level.help in text
    for criterion in state.protocol.criteria:
        assert criterion.question in text and criterion.label in text
    for reason in state.protocol.unratable_reasons:
        assert reason in text
    keys = rater.page.eval_on_selector_all(
        "#help .help-keys kbd", "all => all.map((key) => key.textContent)"
    )
    for key in (
        "Space",
        "←",
        "→",
        "[",
        "]",
        "m",
        "z",
        "Delete",
        "h",
        "s",
        "u",
        "o",
        "n",
        "Enter",
        "k",
        "Backspace",
        "?",
        "t",
    ):
        assert key in keys, key
    for criterion in state.protocol.criteria:
        assert set(criterion.keys) <= set(keys) and criterion.mark_key in keys
    assert state.protocol.other_mark_key in keys
    assert "Reference examples" not in text
    # the page behind it does not react
    rater.press(" ", "3", "u", "m", "Enter", "Backspace")
    assert rater.viewer["playing"] == "false" and rater.viewer["mode"] == "sync"
    assert rater.pressed() == {} and rater.page.is_hidden("#unratable-picker")
    assert state.paths("/api/back") == []
    rater.press("?")
    assert rater.page.is_hidden("#help")
    rater.press("?", "Escape")
    assert rater.page.is_hidden("#help")
    rater.page.click("#help-toggle")
    assert rater.page.is_visible("#help")
    rater.press("Escape", "3")
    assert rater.pressed() == {"A": 2}


def test_the_help_screen_shows_the_reference_examples_of_the_session(
    rater: Rater, state: MockState
) -> None:
    subject = state.subjects[2]
    state.examples = (
        schemas.ExampleOut(
            criterion="A",
            value=2,
            panel="coronal",
            slice=11,
            media_base=schemas.media_base(subject.subject_uid),
            render_key=subject.render_key,
            caption="The pial surface encloses the sinus.",
            tiles=subject.info.tiles,
        ),
    )
    rater.open().paused()
    rater.press("?")
    rater.page.wait_for_function(
        "(() => { const images = [...document.querySelectorAll('#help .example img')];"
        " return images.length === 2 && images.every((image) => image.naturalWidth > 0); })()"
    )
    sources = rater.page.eval_on_selector_all(
        "#help .example img", "all => all.map((image) => image.getAttribute('src'))"
    )
    base = schemas.media_base(subject.subject_uid)
    assert sources == [
        f"{base}tiles/coronal_011.webp?v={subject.render_key}",
        f"{base}tiles/coronal_011_c.webp?v={subject.render_key}",
    ]
    caption = rater.page.inner_text("#help .example figcaption")
    assert caption == "A: major The pial surface encloses the sinus."
    assert rater.console == []


# --- back, and what survives a reload -------------------------------------------------------------


def test_backspace_goes_back_to_the_last_subject_with_its_values_and_marks(
    rater: Rater, state: MockState
) -> None:
    rater.open().paused()
    rater.press("Backspace")
    rater.page.wait_for_function(
        "document.getElementById('hint').textContent.includes('no earlier subject')"
    )
    rater.seek(30)
    rater.press("3")
    rater.click_tile("sagittal", 120.5, 130.25)
    rater.press("5")
    rater.click_tile("coronal", 60.0, 200.0)
    rater.press("7", "n")
    rater.page.keyboard.type("first look")
    rater.press("Escape", "Enter")
    rater.paused(1)
    # something is entered for the second subject, then the rater looks back
    rater.press("2", "b")
    rater.click_tile("axial", 99.0, 77.0)
    rater.press("Backspace")
    rater.ready(0)
    assert rater.viewer["playing"] == "false", (
        "a subject that is shown again does not start playing"
    )
    assert rater.pressed() == {"A": 2, "B": 1, "C": 0}
    assert rater.page.input_value("#note") == "first look" and rater.page.is_visible("#note-box")
    assert rater.counts() == {"A": 1, "B": 1, "C": 0, "X": 0}
    assert "rated before" in rater.hint
    rater.seek(30)
    own = [
        (mark["id"], mark["cls"])
        for name in PANEL_ORDER
        for mark in rater.marks(name)
        if mark["own"]
    ]
    first_id = state.completed[0].marks[0].mark_id
    assert own == [(f"m{first_id}", "A"), (f"m{first_id + 1}", "B")]
    where = rater.marks("sagittal")[0]
    near = rater.screen_pixel()
    assert where["px"] == pytest.approx(120.5, abs=near) and where["py"] == pytest.approx(
        130.25, abs=near
    )
    # revise: B is major after all, one more mark, one mark less
    rater.press("6")
    rater.click_tile("coronal", 200.0, 100.0)
    rater.click_tile("sagittal", 120.5, 130.25)
    rater.press("Delete", "Enter")
    rater.wait_ratings(2)
    revision = state.ratings[1]
    assert revision.assignment_id == FIRST_ASSIGNMENT and revision.values == {
        "A": 2,
        "B": 2,
        "C": 0,
    }
    assert [(mark.client_id, mark.mark_class) for mark in revision.marks] == [
        (f"m{first_id + 1}", "B"),
        ("n1", "B"),
    ]
    assert revision.note == "first look"
    # and on to the second subject, with what had been entered for it
    rater.ready(1)
    assert rater.pressed() == {"A": 1} and rater.counts()["B"] == 1
    assert rater.page.get_attribute("#active-class", "data-class") == ""
    assert state.paths("/api/back") == ["/api/back"] * 2  # the first found no earlier subject
    assert "back" in rater.wait_events(lambda types: "back" in types)


def test_what_is_entered_survives_a_reload_of_the_page(rater: Rater, state: MockState) -> None:
    rater.open().paused()
    rater.seek(30)
    rater.press("3", "4")
    rater.click_tile("sagittal", 120.0, 130.0)
    rater.press("n")
    rater.page.keyboard.type("to be continued")
    rater.press("Escape")
    rater.page.reload()
    rater.ready(0)
    assert rater.pressed() == {"A": 2, "B": 0}
    assert rater.counts()["A"] == 1 and rater.page.input_value("#note") == "to be continued"
    rater.press("7", "Enter")
    rater.wait_ratings(1)
    rating = state.ratings[0]
    assert (
        rating.values == {"A": 2, "B": 0, "C": 0}
        and len(rating.marks) == 1
        and rating.note == "to be continued"
    )
    # the draft is gone with the submission: the next subject is empty after a reload too
    rater.ready(1)
    rater.page.reload()
    rater.ready(1)
    assert rater.pressed() == {} and rater.counts()["A"] == 0


# --- a server that refuses, or is away ---------------------------------------------------------


def test_a_rating_is_sent_again_while_the_server_is_away(rater: Rater, state: MockState) -> None:
    state.fail_ratings = 2
    rater.open().paused()
    rater.press("1", "4", "7", "Enter")
    rater.page.wait_for_function(
        "document.getElementById('hint').textContent.includes('Not saved yet')"
    )
    assert rater.page_state == "saving" and state.ratings == []
    # the keys do nothing to the rating while it is on its way
    rater.press("3", "u")
    assert rater.pressed() == {"A": 0, "B": 0, "C": 0} and rater.page.is_hidden("#unratable-picker")
    rater.ready(1)
    assert len(state.ratings) == 1 and state.ratings[0].values == {"A": 0, "B": 0, "C": 0}
    assert len(state.paths("/api/rating")) == 3
    assert rater.page_state == "rating"


def test_a_rating_that_was_never_answered_is_sent_by_the_reloaded_page(
    rater: Rater, state: MockState
) -> None:
    state.fail_ratings = 1000
    rater.open().paused()
    rater.press("1", "5", "7", "Enter")
    rater.page.wait_for_function(
        "document.getElementById('hint').textContent.includes('Not saved yet')"
    )
    state.fail_ratings = 0
    rater.page.reload()
    rater.ready(1)
    assert [rating.values for rating in state.ratings] == [{"A": 0, "B": 1, "C": 0}]
    assert rater.pressed() == {}


def test_a_refused_rating_stays_on_the_page_with_what_the_server_said(
    rater: Rater, state: MockState
) -> None:
    state.refuse_rating = (422, "marks.0.px", "the point is outside the tile")
    rater.open().paused()
    rater.press("1", "4", "7", "Enter")
    rater.page.wait_for_function(
        "document.getElementById('hint').textContent.includes('Not stored')"
    )
    assert rater.hint == "Not stored: marks.0.px: the point is outside the tile"
    assert rater.page_state == "rating" and rater.pressed() == {"A": 0, "B": 0, "C": 0}
    assert state.ratings == []
    rater.press("Enter")
    rater.ready(1)
    assert len(state.ratings) == 1


def test_after_a_conflict_the_page_asks_for_the_session_and_the_subject_again(
    rater: Rater, state: MockState
) -> None:
    state.refuse_rating = (409, "render_key", "the render of the subject changed")
    rater.open().paused()
    rater.press("1", "4", "7", "Enter")
    rater.page.wait_for_function(
        "document.getElementById('hint').textContent.includes('loaded again')"
    )
    assert "the render of the subject changed" in rater.hint
    assert state.ratings == [] and rater.page_state == "rating"
    assert state.paths("/api/session") == ["/api/session"] * 2
    assert state.paths("/api/next") == ["/api/next"] * 2


# --- rendering, done ---------------------------------------------------------------------------


def test_the_page_waits_while_nothing_is_rendered_and_goes_on_by_itself(
    rater: Rater, state: MockState
) -> None:
    state.rendering_polls = 3
    rater.open()
    rater.page.wait_for_selector("#rate[data-state='rendering']")
    message = rater.page.inner_text("#message")
    assert message == "Rendering: 0 of 3 subjects are ready. The next one opens by itself."
    assert "0 of 3 rendered" in rater.page.inner_text("#progress-line")
    assert rater.page.is_disabled("#submit")
    rater.ready(0)
    assert rater.page_state == "rating" and rater.page.is_hidden("#message")
    assert len(state.paths("/api/next")) == 4
    assert "rendered" not in rater.page.inner_text("#progress-line")


def test_when_nothing_is_left_the_page_says_so_and_backspace_still_goes_back(
    rater: Rater, state: MockState
) -> None:
    del state.subjects[1:]
    state.blind = True
    rater.open().paused()
    assert rater.page.is_hidden("#review-link")
    rater.press("1", "4", "7", "Enter")
    rater.page.wait_for_selector("#rate[data-state='done']")
    assert rater.page.inner_text("#message").startswith("Nothing is left to rate: 1 done.")
    assert (
        rater.page.is_visible("#review-link")
        and rater.page.get_attribute("#review-link", "href") == "/review"
    )
    assert rater.page.is_disabled("#submit") and rater.pressed() == {}
    # the keys of a subject do nothing when there is none
    rater.press(" ", "3", "Enter", "u")
    assert rater.page.is_hidden("#unratable-picker") and len(state.ratings) == 1
    rater.press("Backspace")
    rater.ready(0)
    assert rater.page_state == "rating" and rater.pressed() == {"A": 0, "B": 0, "C": 0}
    rater.press("2", "Enter")
    rater.page.wait_for_selector("#rate[data-state='done']")
    assert [rating.values["A"] for rating in state.ratings] == [0, 1]


def test_the_review_link_is_there_from_the_start_when_the_run_is_not_blind(
    rater: Rater, state: MockState
) -> None:
    assert not state.blind
    rater.open().ready()
    assert rater.page.is_visible("#review-link")


# --- events ----------------------------------------------------------------------------------


def test_what_the_rater_does_is_logged_and_posted_before_the_rating(
    rater: Rater, state: MockState
) -> None:
    rater.open().ready()
    rater.page.wait_for_function("document.getElementById('viewer').dataset.frame >= 4")
    rater.press(" ", "m", "m", "]", "z", "z")
    rater.page.keyboard.down("s")
    rater.page.keyboard.up("s")
    rater.page.keyboard.down("h")
    rater.page.keyboard.up("h")
    rater.press("a")
    rater.click_tile("sagittal", 100.0, 100.0)
    rater.click_tile("coronal", 100.0, 100.0)
    rater.press("Control+z")
    rater.click_tile("sagittal", 100.0, 100.0)
    rater.press("Delete", "?", "?", "ArrowRight", "ArrowRight", "1", "4", "7", "Enter")
    rater.ready(1)
    order = [path for _, path in state.requests if path in ("/api/events", "/api/rating")]
    assert order[-2:] == ["/api/events", "/api/rating"]
    first = [
        event
        for batch in state.events
        if batch.assignment_id == FIRST_ASSIGNMENT
        for event in batch.events
    ]
    types = [event.type for event in first]
    assert set(types) <= set(schemas.EVENT_TYPES)
    for wanted in (
        "load",
        "play",
        "pause",
        "frames",
        "mode",
        "speed",
        "zoom",
        "contours",
        "marks_hidden",
        "mark_add",
        "mark_delete",
        "mark_undo",
        "help",
    ):
        assert wanted in types, wanted
    assert (
        types.count("mark_add") == 2 and types.count("mode") == 2 and types.count("contours") == 2
    )
    by_type = {event.type: event.payload for event in first}
    assert by_type["speed"] == {"speed": 2} and by_type["mark_undo"] == {"client_id": "n2"}
    assert by_type["mark_delete"] == {"client_id": "n1"}
    load = next(event.payload for event in first if event.type == "load")
    assert load["tiles"] == tiles_of(info_of(state)) and load["failed"] == 0
    # the frames that were on screen, in order, with how long each was
    visits = [
        visit for event in first if event.type == "frames" for visit in event.payload["visits"]
    ]
    assert all(ms >= 0 for _, ms in visits)
    sync_frames = [
        frame
        for event in first
        if event.type == "frames" and event.payload["mode"] == "sync"
        for frame, _ in event.payload["visits"]
    ]
    assert sync_frames[:5] == [0, 1, 2, 3, 4]
    total = sum(ms for _, ms in visits)
    assert (
        0.5 * state.ratings[0].ms_on_subject <= total <= 1.2 * state.ratings[0].ms_on_subject + 200
    )


def test_events_are_posted_while_the_rater_looks_and_when_the_page_is_left(
    rater: Rater, state: MockState
) -> None:
    rater.open().ready()
    # within a few seconds, without a submission
    types = rater.wait_events(lambda found: "load" in found and "frames" in found)
    assert state.ratings == [] and "play" in types
    before = len(state.events)
    rater.press(" ", "z")
    rater.page.goto("about:blank")
    for _ in range(200):
        if len(state.events) > before:
            break
        rater.page.wait_for_timeout(25)
    late = [event.type for batch in state.events[before:] for event in batch.events]
    assert "zoom" in late and "pause" in late


# --- the review page ---------------------------------------------------------------------------


def _rate_first(rater: Rater) -> None:
    rater.open().paused()
    rater.seek(30)
    rater.press("3")
    rater.click_tile("sagittal", 120.0, 130.0)
    rater.press("4", "7", "Enter")
    rater.ready(1)


def test_the_review_page_lists_the_subjects_and_sorts_and_filters(
    rater: Rater, state: MockState
) -> None:
    _rate_first(rater)
    rater.open("review")
    rater.page.wait_for_selector("#review[data-state='open']")
    head = rater.page.eval_on_selector_all(
        "#subjects thead th", "all => all.map((cell) => cell.textContent)"
    )
    assert (
        head[:2] == ["subject_uid", "subject_label"] and "composite" in head and "final_A" in head
    )
    rows = rater.page.eval_on_selector_all(
        "#subjects tbody tr",
        "all => all.map((row) => [...row.cells].map((cell) => cell.textContent))",
    )
    assert [row[1] for row in rows] == ["sub-001", "sub-002", "sub-003"]
    first = dict(zip(head, rows[0], strict=True))
    assert first["final_A"] == "2" and first["final_B"] == "0" and first["n_marks"] == "1"
    assert dict(zip(head, rows[1], strict=True))["final_A"] == ""
    assert rater.page.inner_text("#review-count") == "3 subjects"
    assert len(rater.page.query_selector_all("#scatter .scatter-dot")) == 3
    labels = rater.page.eval_on_selector_all(
        "#scatter text", "all => all.map((label) => label.textContent)"
    )
    for level in state.protocol.scale:
        assert level.label in labels
    # a click on a column sorts by it, a second one the other way round
    rater.page.click("#subjects thead button[data-column='composite']")
    rater.page.wait_for_function(
        "document.querySelector('#subjects tbody tr td:nth-child(2)').textContent === 'sub-003'"
    )
    rater.page.click("#subjects thead button[data-column='composite']")
    rater.page.wait_for_function(
        "document.querySelector('#subjects tbody tr td:nth-child(2)').textContent === 'sub-001'"
    )
    queries = [path for path in state.paths("/api/review?")]
    assert "sort=composite&order=asc" in queries[-2] and "sort=composite&order=desc" in queries[-1]
    # typing in the filter does not move a viewer or anything else; Enter applies it
    rater.page.fill("#filter", "final_A:2")
    rater.page.press("#filter", "Enter")
    rater.page.wait_for_function("document.querySelectorAll('#subjects tbody tr').length === 1")
    assert "filter=final_A%3A2" in state.paths("/api/review?")[-1]
    assert rater.page.inner_text("#review-count") == "1 subjects"
    assert rater.page.get_attribute("#export-link", "href") == "/api/export"
    assert rater.console == []


def test_a_subject_of_the_review_shows_marks_and_cluster_centroids(
    rater: Rater, state: MockState
) -> None:
    _rate_first(rater)
    rater.open("review")
    rater.page.wait_for_selector("#review[data-state='open']")
    rater.page.locator("#subjects tbody tr").first.click()
    rater.ready(0)
    assert rater.page.inner_text("#subject-title") == "sub-001"
    assert rater.viewer["playing"] == "false"
    info = info_of(state)
    mark = state.ratings[0].marks[0]
    # the viewer opens on the tile the first mark was placed on
    assert rater.page.get_attribute(".panel[data-panel='sagittal']", "data-slice") == str(
        mark.slice
    )
    drawn = rater.marks("sagittal")
    assert len(drawn) == 1 and drawn[0]["cls"] == "A"
    assert drawn[0]["px"] == pytest.approx(mark.px) and drawn[0]["py"] == pytest.approx(mark.py)
    label = rater.page.text_content(".panel[data-panel='sagittal'] g.mark text")
    assert label == "A local"
    detail = rater.page.inner_text("#subject-detail")
    assert (
        "Ratings (1)" in detail
        and "Marks (1)" in detail
        and "Outlier clusters with a native position (1)" in detail
    )
    # the centroid of the cluster: a diamond at the pial position the server gave
    centre = np.array(info.box.centre)
    vox = geometry.apply_affine(np.linalg.inv(np.array(info.volume.vox2ras_tkr)), centre)
    rater.page.locator("#subject-detail tr", has_text="set-cecaad3b4478").click()
    for name in PANEL_ORDER:
        panel = info.panels[name]
        shown = int(rater.page.get_attribute(f".panel[data-panel='{name}']", "data-slice"))
        if abs(panel.slice_mm[shown] - centre[panel.world_axis]) > 3.0:
            continue
        where = rater.page.eval_on_selector(
            f".panel[data-panel='{name}'] g.centroid",
            "group => group.getAttribute('transform').match(/-?[\\d.]+/g).map(Number)",
        )
        px, py, _ = geometry.voxel_to_pixel(panel, vox)
        assert where == pytest.approx([float(px), float(py)], abs=1e-6)
    assert rater.page.query_selector_all("g.centroid"), "no panel shows the centroid"
    # the keys of the viewer work here, the keys of a rating do not exist
    rater.press("ArrowRight", "m", "3", "Enter")
    assert rater.viewer["mode"] == "seq" and len(state.ratings) == 1
    assert rater.console == []


def test_the_review_page_says_in_words_that_it_is_closed(rater: Rater, state: MockState) -> None:
    rater.open("review")
    rater.page.wait_for_selector("#review[data-state='open']")
    state.review_closed = True
    rater.page.click("#subjects thead button[data-column='composite']")
    rater.page.wait_for_selector("#review[data-state='closed']")
    message = rater.page.inner_text("#review-message")
    assert message.startswith("The review is closed while you still have subjects to rate")
    assert "you still have subjects to rate" in message
    assert rater.page.is_hidden("#review-body")
    assert rater.page.get_attribute("#rate-link", "href") == "/"


# --- skip ------------------------------------------------------------------------------------


def _enter_something(rater: Rater) -> None:
    """A value, a mark and a note for the subject on screen."""
    rater.seek(30)
    rater.press("3")
    rater.click_tile("sagittal", 120.0, 130.0)
    rater.press("n")
    rater.page.keyboard.type("come back to this")
    rater.press("Escape")


def test_k_skips_and_the_subject_comes_back_after_the_others_with_what_was_entered(
    rater: Rater, state: MockState
) -> None:
    rater.open().paused(0)
    assert rater.page.inner_text("#skip") == "k skip"
    # the skip button stands next to the back button, in the rating bar
    back, skip = rater.box("#back"), rater.box("#skip")
    assert skip["y"] == back["y"] and 0 < skip["x"] - (back["x"] + back["width"]) < 20
    assert rater.page.locator("#rating #skip").count() == 1
    _enter_something(rater)
    rater.press("k")
    rater.paused(1)
    rater.wait_skips(1)
    skipped = state.skips[0]
    assert skipped.assignment_id == FIRST_ASSIGNMENT and 0 < skipped.ms_on_subject < 60_000
    # nothing of what was entered went to the server: no rating, and the skip has no place for it
    assert state.ratings == [] and state.completed == []
    assert state.paths("/api/rating") == [] and state.paths("/api/skip") == ["/api/skip"]
    # the next subject starts clean, and the page says what happened
    assert rater.pressed() == {} and rater.counts()["A"] == 0
    assert rater.page.input_value("#note") == "" and rater.page.is_hidden("#note-box")
    assert "Skipped" in rater.hint and "nothing was stored" in rater.hint
    assert rater.progress == "0 done today · 0 done in total · 3 remaining · 1 skipped, waiting"
    assert rater.page.get_attribute("#progress-line", "data-skipped") == "1"
    rater.press("1", "4", "7", "Enter")
    rater.paused(2)
    assert "1 skipped, waiting" in rater.progress
    rater.press("1", "4", "7", "Enter")
    # every other subject was shown: the skipped one is back, as it was left
    rater.paused(0)
    assert rater.pressed() == {"A": 2} and rater.counts()["A"] == 1
    assert rater.page.input_value("#note") == "come back to this"
    assert rater.page.is_visible("#note-box")
    assert rater.progress == "2 done today · 2 done in total · 1 remaining"
    rater.seek(30)
    mark = rater.marks("sagittal")[0]
    near = rater.screen_pixel()
    assert mark["px"] == pytest.approx(120.0, abs=near) and mark["cls"] == "A"
    rater.press("4", "7", "Enter")
    rater.page.wait_for_selector("#rate[data-state='done']")
    assert [rating.assignment_id for rating in state.ratings] == [
        FIRST_ASSIGNMENT + 1,
        FIRST_ASSIGNMENT + 2,
        FIRST_ASSIGNMENT,
    ]
    last = state.ratings[-1]
    assert last.values == {"A": 2, "B": 0, "C": 0} and last.note == "come back to this"
    assert len(last.marks) == 1 and len(state.skips) == 1
    # the time of the rating is of the visit that ended with it, not of the page since it loaded
    assert 0 < last.ms_on_subject < 60_000
    assert rater.console == [] and rater.failed == []


def test_the_skip_button_does_what_the_key_does_and_skip_goes_round(
    rater: Rater, state: MockState
) -> None:
    rater.open().paused(0)
    rater.press("2")
    rater.page.click("#skip")
    rater.paused(1)
    rater.page.click("#skip")
    rater.paused(2)
    assert "2 skipped, waiting" in rater.progress
    rater.press("k")
    # only skipped subjects are left: skip goes round them, in the order of skipping
    rater.paused(0)
    assert rater.pressed() == {"A": 1}
    assert "2 skipped, waiting" in rater.progress and "3 remaining" in rater.progress
    rater.press("k")
    rater.paused(1)
    rater.press("k")
    rater.paused(2)
    rater.wait_skips(5)
    assert [skip.assignment_id - FIRST_ASSIGNMENT for skip in state.skips] == [0, 1, 2, 0, 1]
    assert state.ratings == [] and rater.page_state == "rating"
    # the page never says that nothing is left while a subject has no rating
    assert rater.page.is_hidden("#message")
    assert rater.console == []


def test_the_last_subject_without_a_rating_cannot_be_skipped_away(
    rater: Rater, state: MockState
) -> None:
    del state.subjects[1:]
    rater.open().paused(0)
    rater.press("5", "k")
    rater.wait_skips(1)
    rater.page.wait_for_function(
        "document.getElementById('hint').textContent.includes('No other subject is left')"
    )
    rater.ready(0)
    assert rater.page_state == "rating" and rater.pressed() == {"B": 1}
    assert "skipped" not in rater.progress and "1 remaining" in rater.progress
    rater.press("1", "7", "Enter")
    rater.page.wait_for_selector("#rate[data-state='done']")
    assert state.ratings[0].values == {"A": 0, "B": 1, "C": 0}


def test_backspace_after_a_skip_goes_back_to_the_skipped_subject(
    rater: Rater, state: MockState
) -> None:
    rater.open().paused(0)
    _enter_something(rater)
    rater.press("k")
    rater.paused(1)
    rater.press("6", "Backspace")
    rater.ready(0)
    # the skipped subject, with what was entered, and nothing that says it was rated
    assert rater.pressed() == {"A": 2} and rater.counts()["A"] == 1
    assert "was skipped" in rater.hint and "rated before" not in rater.hint
    assert rater.page.input_value("#note") == "come back to this"
    assert "skipped, waiting" not in rater.progress  # the one that waited is on the page
    assert len(state.skips) == 1  # going back takes no skip away and adds none
    rater.press("4", "7", "Enter")
    rater.wait_ratings(1)
    assert state.ratings[0].assignment_id == FIRST_ASSIGNMENT
    assert state.ratings[0].values == {"A": 2, "B": 0, "C": 0}
    # and on with the subject that was on the page, with what was entered for it
    rater.ready(1)
    assert rater.pressed() == {"B": 2} and "skipped" not in rater.progress
    rater.press("1", "7", "Enter")
    rater.ready(2)
    assert [rating.assignment_id - FIRST_ASSIGNMENT for rating in state.ratings] == [0, 1]


def test_a_skip_is_sent_again_while_the_server_is_away_and_counted_once(
    rater: Rater, state: MockState
) -> None:
    state.fail_skips = 2
    rater.open().paused(0)
    rater.press("k")
    rater.page.wait_for_function(
        "document.getElementById('hint').textContent.includes('Not skipped yet')"
    )
    assert rater.page_state == "saving" and state.skips == []
    rater.ready(1)
    # three requests with one body, which the server takes as one skip
    assert len(state.paths("/api/skip")) == 3 and len(state.skips) == 1
    assert rater.page_state == "rating" and "Skipped" in rater.hint


def test_a_skip_that_the_server_refuses_loads_the_subject_again(
    rater: Rater, state: MockState
) -> None:
    rater.open().paused(0)
    # the subject was rated in another window meanwhile
    state.queue.remove(0)
    rater.press("k")
    rater.ready(1)
    assert "Not skipped" in rater.hint and state.skips == []


def test_skip_on_a_subject_that_is_shown_again_leaves_its_rating_as_it_was(
    rater: Rater, state: MockState
) -> None:
    rater.open().paused(0)
    rater.press("3", "4", "7", "Enter")
    rater.paused(1)
    rater.press("Backspace")
    rater.ready(0)
    assert rater.pressed() == {"A": 2, "B": 0, "C": 0}
    rater.press("1", "k")
    rater.ready(1)
    assert "Left as it was" in rater.hint
    # no skip of a rated subject, and no second rating
    assert state.skips == [] and state.paths("/api/skip") == []
    assert len(state.ratings) == 1 and state.ratings[0].values["A"] == 2


# --- a refused Enter -------------------------------------------------------------------------


@pytest.mark.parametrize("viewport", [WIDE, NARROW])
def test_a_refused_enter_says_why_beside_the_submit_button_and_marks_the_rows(
    browser: Any, server: MockHandle, state: MockState, viewport: dict[str, int]
) -> None:
    page = browser.new_page(viewport=viewport)
    rater = Rater(page, server).open().paused(0)
    assert rater.refusal == "" and rater.lacking() == {}
    rater.press("Enter")
    page.wait_for_selector("#refusal:not([hidden])")
    assert state.ratings == []
    text = rater.refusal
    assert text.startswith("Not submitted.")
    assert "A, B and C have no value yet" in text
    # the same message names the way out: the skip key, and what a skip does
    assert re.search(r"press\s+k\s*:", text) and "nothing is stored" in text
    assert re.search(r"press\s+u\s+if the subject cannot be rated", text)
    assert page.get_attribute("#refusal", "role") == "alert"
    assert rater.lacking() == {"A": "value", "B": "value", "C": "value"}
    labels = page.eval_on_selector_all(".criterion-need", "all => all.map((e) => e.textContent)")
    assert labels == ["needs a value"] * 3
    # where it is: on screen, directly under the submit button, its arrow pointing at it
    box, button = rater.box("#refusal"), rater.box("#submit")
    assert 0 <= box["x"] and box["x"] + box["width"] <= viewport["width"]
    assert 0 <= box["y"] and box["y"] + box["height"] <= viewport["height"]
    assert 0 <= box["y"] - (button["y"] + button["height"]) <= 16
    arrow = box["x"] + float(
        page.evaluate(
            "parseFloat(getComputedStyle(document.getElementById('refusal'))"
            ".getPropertyValue('--arrow'))"
        )
    )
    assert button["x"] <= arrow <= button["x"] + button["width"]
    # it cannot be overlooked: larger than the status line, in a colour of its own
    look = page.evaluate(
        """() => {
            const style = (id) => getComputedStyle(document.getElementById(id));
            const framed = document.querySelectorAll('.criterion[data-lacking=value] .levels');
            return { size: parseFloat(style('refusal').fontSize),
                     hint: parseFloat(style('hint').fontSize),
                     back: style('refusal').backgroundColor,
                     rating: style('rating').backgroundColor,
                     frames: [...framed].map((e) => getComputedStyle(e).outlineStyle) };
        }"""
    )
    assert look["size"] > look["hint"] and look["back"] != look["rating"]
    assert look["frames"] == ["solid"] * 3
    assert box["height"] >= 60 and box["width"] >= min(360, viewport["width"] - 40)

    # a row that gets its value loses its mark, and the message follows
    rater.press("3")
    assert rater.lacking() == {"B": "value", "C": "value"}
    assert "B and C have no value yet" in rater.refusal
    rater.press("4")
    assert "C has no value yet. Give it a value" in rater.refusal
    rater.press("9")
    assert rater.refusal == "" and rater.lacking() == {}
    assert rater.hint == "Ready: Enter submits."
    rater.press("Enter")
    rater.ready(1)
    assert len(state.ratings) == 1 and rater.refusal == "" and rater.lacking() == {}
    assert rater.console == []
    page.close()


def test_a_click_on_a_submit_that_is_held_back_is_answered_too(
    rater: Rater, state: MockState
) -> None:
    rater.open().paused(0)
    # the button can be pressed while the rating is incomplete, and says so to a screen reader
    assert rater.page.get_attribute("#submit", "aria-disabled") == "true"
    assert rater.page.evaluate("document.getElementById('submit').disabled") is False
    rater.page.locator("#submit").click(force=True)
    rater.page.wait_for_selector("#refusal:not([hidden])")
    assert "Not submitted." in rater.refusal and state.ratings == []
    # skipping from there: the next subject starts without the refusal
    rater.press("k")
    rater.ready(1)
    assert rater.refusal == "" and rater.lacking() == {}
    rater.press("1", "4", "7")
    assert rater.page.get_attribute("#submit", "aria-disabled") == "false"


def test_a_refusal_names_the_mark_and_the_note_that_are_missing(
    rater: Rater, state: MockState
) -> None:
    state.require_marks = True
    rater.open().paused(0)
    rater.press("3", "4", "7", "o", "Escape", "Enter")
    rater.page.wait_for_selector("#refusal:not([hidden])")
    text = rater.refusal
    assert "A has no mark yet. Click on the error" in text
    assert re.search(r"needs a note: press\s+n", text) and "no value yet" not in text
    assert rater.lacking() == {"A": "mark"}
    rater.click_tile("sagittal", 120.0, 130.0)
    assert rater.lacking() == {} and "no mark yet" not in rater.refusal
    assert "needs a note" in rater.refusal
    rater.press("o")
    assert rater.refusal == ""


# --- the tour --------------------------------------------------------------------------------


@pytest.fixture
def newcomer(browser: Any, server: MockHandle) -> Iterator[Rater]:
    """A rater who was never shown the tour, at a wide window."""
    context = browser.new_context(viewport=WIDE)
    yield Rater(context.new_page(), server, seen_tour=False)
    context.close()


def test_the_tour_opens_on_a_first_visit_and_not_on_the_second(
    newcomer: Rater, state: MockState
) -> None:
    rater = newcomer
    rater.open().ready(0)
    rater.page.wait_for_selector("#tour:not([hidden])")
    assert rater.tour_step == "views"
    # playback waits behind it, and nothing of the rating was touched
    assert rater.viewer["playing"] == "false" and rater.pressed() == {}
    assert rater.page.evaluate("localStorage.getItem('surfcensus.tour.seen:local')") == "1"
    rater.press("Escape")
    assert rater.tour_step is None
    # what was to play plays now
    rater.page.wait_for_function("document.getElementById('viewer').dataset.playing === 'true'")
    rater.page.reload()
    rater.ready(0)
    rater.page.wait_for_function("document.getElementById('viewer').dataset.playing === 'true'")
    assert rater.tour_step is None
    # nor on the next subject, nor in a new window of the same browser
    rater.press("1", "4", "7", "Enter")
    rater.ready(1)
    assert rater.tour_step is None
    second = rater.page.context.new_page()
    second.goto(rater.server.url)
    second.wait_for_selector("#viewer[data-ready='true']")
    second.wait_for_function("document.getElementById('viewer').dataset.playing === 'true'")
    assert second.is_hidden("#tour")
    second.close()
    opened = rater.wait_events(lambda types: "tour_close" in types)
    assert opened.count("tour_open") == 1
    assert rater.events_of("tour_open") == [{"how": "first"}]
    assert rater.console == []


def test_without_storage_the_tour_opens_once_per_load_and_the_page_works(
    browser: Any, server: MockHandle, state: MockState
) -> None:
    page = browser.new_page(viewport=WIDE)
    page.add_init_script(
        """for (const name of ['localStorage', 'sessionStorage']) {
             Object.defineProperty(window, name, {
               get() { throw new DOMException('denied', 'SecurityError'); } });
           }"""
    )
    rater = Rater(page, server, seen_tour=False).open().ready(0)
    page.wait_for_selector("#tour:not([hidden])")
    rater.press("Escape")
    rater.paused(0)
    # once per load: not again for the next subject
    rater.press("k")
    rater.paused(1)
    assert rater.tour_step is None
    rater.press("1", "4", "7", "Enter")
    rater.ready(2)
    assert rater.tour_step is None and len(state.ratings) == 1 and len(state.skips) == 1
    page.reload()
    rater.ready(2)
    page.wait_for_selector("#tour:not([hidden])")
    assert rater.console == []
    page.close()


@pytest.mark.parametrize(
    "viewport",
    [
        WIDE,
        NARROW,
        # windows too low for the page, which scrolls: a phone on its side, a small laptop
        {"width": 400, "height": 440},
        {"width": 1366, "height": 560},
    ],
)
def test_every_step_of_the_tour_lights_a_part_that_is_on_screen_with_its_text_beside_it(
    browser: Any, server: MockHandle, state: MockState, viewport: dict[str, int]
) -> None:
    page = browser.new_page(viewport=viewport)
    rater = Rater(page, server, seen_tour=False).open().ready(0)
    page.wait_for_selector("#tour:not([hidden])")
    width, height = viewport["width"], viewport["height"]
    seen = []
    for number, step in enumerate(TOUR, start=1):
        assert rater.tour_step == step
        seen.append(step)
        assert page.inner_text(".tour-count") == f"{number} of {len(TOUR)}"
        measured = page.evaluate(
            """() => {
                const rect = (element) => {
                  const box = element.getBoundingClientRect();
                  return [box.left, box.top, box.right, box.bottom];
                };
                return { spot: rect(document.querySelector('.tour-spot')),
                         card: rect(document.querySelector('.tour-card')),
                         scroll: [document.documentElement.scrollWidth, window.innerWidth] };
            }"""
        )
        spot, card = measured["spot"], measured["card"]
        # the part exists and is on screen, all of it
        assert spot[2] - spot[0] >= 20 and spot[3] - spot[1] >= 20, step
        # (the frame around it is 6 px wider than the part, which may touch an edge)
        assert -6 <= spot[0] and spot[2] <= width + 6 and -6 <= spot[1] and spot[3] <= height + 6, (
            step,
            spot,
        )
        # the text is inside the window whatever its size, and beside the part: over none
        # of it where there is room, and never over all of it
        assert 0 <= card[0] and card[2] <= width and 0 <= card[1] and card[3] <= height, (
            step,
            card,
        )
        over = max(0.0, min(spot[2], card[2]) - max(spot[0], card[0])) * max(
            0.0, min(spot[3], card[3]) - max(spot[1], card[1])
        )
        share = over / ((spot[2] - spot[0]) * (spot[3] - spot[1]))
        assert share <= 0.5, (step, share)
        if viewport == WIDE:
            assert share == 0, (step, share)
        # the window does not scroll sideways for it
        assert measured["scroll"][0] <= measured["scroll"][1], step
        # what is lit is what the step is about
        lit = page.evaluate(
            """([x, y]) => {
                const tour = document.getElementById('tour');
                tour.style.visibility = 'hidden';
                const under = document.elementFromPoint(x, y);
                tour.style.visibility = '';
                return under ? [under.closest('#viewer') ? 'viewer' : '',
                                under.closest('#criteria') ? 'criteria' : '',
                                under.closest('.actions') ? 'actions' : ''].join('') : '';
            }""",
            [(spot[0] + spot[2]) / 2, (spot[1] + spot[3]) / 2],
        )
        assert (
            lit
            == {
                "views": "viewer",
                "rating": "criteria",
                "marks": "criteria",
                "moving": "actions",
                "help": "actions",
            }[step]
        )
        text = page.inner_text(".tour-card")
        assert len(text) < 420 and "Esc closes" in text
        rater.press("Enter")
    assert seen == list(TOUR) and rater.tour_step is None
    assert rater.console == []
    page.close()


def test_the_tour_says_how_one_gets_from_subject_to_subject_with_the_keys_of_the_session(
    newcomer: Rater, state: MockState
) -> None:
    rater = newcomer
    rater.open().ready(0)
    rater.page.wait_for_selector("#tour:not([hidden])")
    texts = {}
    keys = {}
    for step in TOUR:
        texts[step] = " ".join(rater.page.inner_text(".tour-text").split())
        keys[step] = rater.page.eval_on_selector_all(
            ".tour-text kbd", "all => all.map((key) => key.textContent)"
        )
        rater.press("ArrowRight")
    assert keys["views"] == ["Space", "←", "→", "s"]
    protocol = state.protocol
    assert keys["rating"] == [key for criterion in protocol.criteria for key in criterion.keys]
    assert keys["marks"] == [
        *(criterion.mark_key for criterion in protocol.criteria),
        protocol.other_mark_key,
    ]
    assert keys["moving"] == ["Enter", "u", "k", "Backspace"] and keys["help"] == ["?", "t"]
    assert "Enter stores the rating and shows the next subject" in texts["moving"]
    assert "once every row has a value" in texts["moving"]
    assert "k skips: nothing is stored, and the subject comes back later" in texts["moving"]
    assert "Backspace goes back" in texts["moving"]
    assert "this tour" in texts["help"]
    # nothing about the subject is in it
    whole = " ".join(texts.values())
    for subject in state.subjects:
        assert subject.subject_uid not in whole and subject.render_key not in whole
    assert not re.search(r"[0-9a-f]{16}|sub-|D-[0-9a-f]", whole)


def test_no_key_and_no_click_reaches_the_rating_screen_while_the_tour_is_open(
    newcomer: Rater, state: MockState
) -> None:
    rater = newcomer
    rater.open().ready(0)
    rater.page.wait_for_selector("#tour:not([hidden])")
    before = rater.frame
    # (Space is the focused button of the card: a step of the tour, tested with its focus)
    rater.press("3", "5", "u", "o", "n", "m", "z", "k", "Backspace", "a", "?", "]", "Delete")
    rater.press("Control+z")
    rater.page.keyboard.down("s")
    assert rater.viewer["contours"] == "shown"
    rater.page.keyboard.up("s")
    # a click on a panel places no mark, and the buttons under the tour are not pressed
    box = rater.stage("sagittal")
    rater.page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    for control in ("#skip", "#submit", "#unratable", "#play", "#help-toggle"):
        target = rater.box(control)
        rater.page.mouse.click(
            target["x"] + target["width"] / 2, target["y"] + target["height"] / 2
        )
    rater.page.wait_for_timeout(300)
    assert rater.tour_step == "views"
    viewer = rater.viewer
    assert viewer["playing"] == "false" and viewer["mode"] == "sync" and viewer["zoom"] == "1"
    assert viewer["speed"] == "1" and rater.frame == before
    assert rater.pressed() == {} and rater.counts() == {"A": 0, "B": 0, "C": 0, "X": 0}
    assert rater.page.is_hidden("#unratable-picker") and rater.page.is_hidden("#help")
    assert rater.page.is_hidden("#note-box") and rater.refusal == ""
    assert rater.page.get_attribute("#other", "aria-pressed") == "false"
    assert rater.page.get_attribute("#active-class", "data-class") == ""
    assert state.skips == [] and state.ratings == []
    assert state.paths("/api/back") == [] and state.paths("/api/skip") == []
    # closed, the same keys work
    rater.press("Escape", "3", "k")
    rater.wait_skips(1)
    rater.ready(1)
    assert rater.console == []


def test_the_keys_of_the_tour_and_its_close_button(newcomer: Rater, state: MockState) -> None:
    rater = newcomer
    rater.open().ready(0)
    rater.page.wait_for_selector("#tour:not([hidden])")
    assert rater.page.is_disabled(".tour-back")
    rater.press("ArrowLeft")
    assert rater.tour_step == "views"  # there is no step before the first
    rater.press("Enter")
    assert rater.tour_step == "rating"
    rater.press("ArrowRight")
    assert rater.tour_step == "marks"
    rater.press("ArrowLeft")
    assert rater.tour_step == "rating"
    # the buttons of the card do the same
    rater.page.click(".tour-next")
    rater.page.click(".tour-next")
    assert rater.tour_step == "moving"
    rater.page.click(".tour-back")
    assert rater.tour_step == "marks"
    rater.page.click(".tour-close")
    assert rater.tour_step is None
    closed = rater.wait_events(lambda types: "tour_close" in types)
    assert closed.count("tour_open") == closed.count("tour_close") == 1
    (read,) = rater.events_of("tour_close")
    # how far it was read: the fourth step was the last one shown, and it was not finished
    assert (read["step"], read["steps"], read["finished"]) == (4, 5, False)
    assert isinstance(read["ms"], int) and 0 <= read["ms"] < 60_000

    # t opens it again, from the first step; Enter on the last step ends it
    rater.press("t")
    assert rater.tour_step == "views"
    for _ in TOUR[:-1]:
        rater.press("Enter")
    assert rater.tour_step == "help"
    assert "done" in rater.page.inner_text(".tour-next")
    rater.press("Enter")
    assert rater.tour_step is None
    # t closes it too, and holding Enter does not run through it
    rater.press("t")
    rater.page.keyboard.down("Enter")
    for _ in range(4):
        rater.page.keyboard.down("Enter")  # the repeats of a key that is held
    rater.page.keyboard.up("Enter")
    assert rater.tour_step == "rating"
    rater.press("t")
    assert rater.tour_step is None
    rater.wait_events(lambda types: types.count("tour_close") == 3)
    assert rater.events_of("tour_open") == [{"how": "first"}, {"how": "key"}, {"how": "key"}]
    reads = [(read["step"], read["finished"]) for read in rater.events_of("tour_close")]
    assert reads == [(4, False), (5, True), (2, False)]
    assert state.ratings == [] and rater.refusal == ""  # no Enter of the tour was a submit


def test_the_help_screen_opens_the_tour_again(rater: Rater, state: MockState) -> None:
    rater.open().paused(0)
    assert rater.tour_step is None  # this rater has seen it
    rater.press("?")
    assert rater.page.is_visible("#help-tour")
    assert "t show the tour" in rater.page.inner_text("#help-tour")
    rater.page.click("#help-tour")
    assert rater.tour_step == "views" and rater.page.is_hidden("#help")
    rater.press("Escape")
    # and its key works on the help screen as well
    rater.press("?", "t")
    assert rater.tour_step == "views" and rater.page.is_hidden("#help")
    rater.press("Escape")
    assert rater.tour_step is None and rater.page.is_hidden("#help")
    rater.wait_events(lambda types: types.count("tour_close") == 2)
    assert rater.events_of("tour_open") == [{"how": "help"}, {"how": "help"}]
    # while a reason is being chosen it does not open over the reasons
    rater.press("u", "t")
    assert rater.tour_step is None and rater.page.is_visible("#unratable-picker")


def test_the_tour_is_a_dialog_that_keeps_the_focus_and_gives_it_back(
    rater: Rater, state: MockState
) -> None:
    rater.open().paused(0)
    rater.page.focus("#frame")
    rater.press("t")
    card = rater.page.locator(".tour-card")
    assert card.get_attribute("role") == "dialog" and card.get_attribute("aria-modal") == "true"
    title = rater.page.locator(f"#{card.get_attribute('aria-labelledby')}")
    assert title.inner_text() == "Three views of one brain"
    assert rater.page.locator(f"#{card.get_attribute('aria-describedby')}").count() == 1
    assert rater.page.get_attribute(".tour-close", "aria-label") == "Close the tour"
    focused = "document.activeElement.className"
    inside = "document.querySelector('.tour-card').contains(document.activeElement)"
    assert rater.page.evaluate(inside) and "tour-next" in rater.page.evaluate(focused)
    # Tab goes round the buttons of the card and never leaves it
    visited = []
    for _ in range(7):
        rater.press("Tab")
        assert rater.page.evaluate(inside)
        visited.append(rater.page.evaluate(focused).split()[-1])
    assert set(visited) == {"tour-close", "tour-next"}  # the back button is off on step one
    rater.press("Enter", "Shift+Tab")
    assert rater.page.evaluate(inside)
    rater.press("Shift+Tab", "Shift+Tab")
    assert rater.page.evaluate(inside)
    # focus that is taken away comes back
    rater.page.evaluate("document.activeElement.blur()")
    rater.page.wait_for_function(inside)
    rater.press("Escape")
    assert rater.page.evaluate("document.activeElement.id") == "frame"
    # Space on the button that has the focus is one step, not two
    rater.press("t", " ")
    assert rater.tour_step == "rating"


def test_time_in_the_tour_is_not_time_on_the_subject(newcomer: Rater, state: MockState) -> None:
    rater = newcomer
    rater.open().ready(0)
    rater.page.wait_for_selector("#tour:not([hidden])")
    rater.page.wait_for_timeout(2500)
    rater.press("Escape", "1", "4", "7", "Enter")
    rater.wait_ratings(1)
    # the tour was open for 2.5 s of the subject's time on screen
    assert state.ratings[0].ms_on_subject < 1500
    (read,) = rater.events_of("tour_close")
    assert read["ms"] >= 2400
    # and the frame under the tour was not being looked at for that long
    rater.wait_events(lambda types: "frames" in types)
    viewed = sum(
        ms
        for batch in state.events
        if batch.assignment_id == FIRST_ASSIGNMENT
        for event in batch.events
        if event.type == "frames"
        for _, ms in event.payload["visits"]
    )
    assert viewed < 1500
    # the same for a skip: its time is of the visit, without the tour
    rater.ready(1)
    rater.press("t")
    rater.page.wait_for_timeout(1500)
    rater.press("Escape", "k")
    rater.wait_skips(1)
    assert state.skips[0].ms_on_subject < 1200


def test_the_tour_does_not_open_where_there_is_no_subject(rater: Rater, state: MockState) -> None:
    del state.subjects[1:]
    rater.open().paused(0)
    rater.press("1", "4", "7", "Enter")
    rater.page.wait_for_selector("#rate[data-state='done']")
    rater.press("t")
    assert rater.tour_step is None
    rater.press("?")
    assert rater.page.is_disabled("#help-tour")
    rater.press("t")
    assert rater.tour_step is None and rater.page.is_visible("#help")


def test_a_rater_who_presses_only_enter_learns_from_the_screen_what_to_do(
    newcomer: Rater, state: MockState
) -> None:
    rater = newcomer
    rater.open().ready(0)
    rater.page.wait_for_selector("#tour:not([hidden])")
    read = []
    for _ in TOUR:
        read.append(" ".join(rater.page.inner_text(".tour-card").split()))
        rater.press("Enter")
    assert rater.tour_step is None
    # the tour said it: Enter needs a value in every row, and k skips
    told = " ".join(read)
    assert "works once every row has a value" in told and "k skips" in told
    # the next Enter is refused, and the screen says so where it cannot be missed
    rater.press("Enter")
    rater.page.wait_for_selector("#refusal:not([hidden])")
    assert state.ratings == [] and "Not submitted." in rater.refusal
    assert "have no value yet" in rater.refusal and re.search(r"press\s+k", rater.refusal)
    assert rater.lacking() == {"A": "value", "B": "value", "C": "value"}
    # one more Enter changes nothing but is answered again
    rater.press("Enter")
    assert rater.page.evaluate(
        "document.getElementById('refusal').classList.contains('refusal-again')"
    )
    # doing what it says leads on: the app has more than one subject
    rater.press("k")
    rater.ready(1)
    assert "Skipped" in rater.hint and "1 skipped, waiting" in rater.progress
    assert rater.console == []
