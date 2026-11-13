"""The rater app in a real browser (``just test-browser``; marked ``browser``, not part of the
default run or of CI).

First that the browser the tests are given starts, runs a script of the kind the frontend
is made of and gives a screenshot. Then the pages themselves, on the server, on a run that
``review`` made of two synthetic subjects (1 mm, 160^3 voxels, panels of 320 px): what a
rater does with the keyboard and the mouse, and what is in ``app.sqlite`` afterwards
(docs/dev/contract.md, section 9.2). The frontend's own browser tests
(``test_frontend_browser.py``) run against a stand-in for the server; these do not.
"""

from __future__ import annotations

import io
import shutil
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import browser_launch  # pyrefly: ignore[missing-import]
import nibabel as nib
import nibabel.freesurfer as fs
import numpy as np
import pytest
from browser_launch import launch_browser  # pyrefly: ignore[missing-import]
from PIL import Image
from synth import write_synthetic_subject  # pyrefly: ignore[missing-import]

from surfcensus.review import review
from surfcensus.store import PublicCatalog

sync_api = pytest.importorskip("playwright.sync_api")

pytestmark = pytest.mark.browser

PAGE = """<!doctype html>
<title>surfcensus</title>
<div id="panel" style="width: 320px; height: 320px; background: #FFD400"></div>
<script type="module">
  const panel = document.getElementById("panel");
  panel.addEventListener("click", (event) => {
    const box = panel.getBoundingClientRect();
    panel.dataset.px = String((event.clientX - box.left) * 64 / box.width);
  });
  document.title = `ready ${[1, 2, 3].at(-1)}`;
</script>
"""


@pytest.fixture(scope="module")
def playwright() -> Iterator[Any]:
    with sync_api.sync_playwright() as started:
        yield started


@pytest.fixture(scope="module")
def browser(playwright: Any) -> Iterator[Any]:
    started = launch_browser(playwright)
    yield started
    started.close()


@pytest.mark.parametrize("width", [1280, 700])
def test_the_browser_runs_a_module_script_and_gives_a_screenshot(browser: Any, width: int) -> None:
    page = browser.new_page(viewport={"width": width, "height": 600})
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.set_content(PAGE)
    page.wait_for_function("document.title.startsWith('ready')")
    assert page.title() == "ready 3" and errors == []
    # a click at a known place of the element arrives there: 80 of 320 px is 16 of 64
    page.click("#panel", position={"x": 80, "y": 40})
    assert float(page.eval_on_selector("#panel", "panel => panel.dataset.px")) == pytest.approx(16)
    shot = Image.open(io.BytesIO(page.screenshot()))
    assert shot.size == (width, 600)
    assert shot.convert("RGB").getpixel((100, 100)) == (0xFF, 0xD4, 0x00)
    page.close()


def test_without_a_named_browser_and_without_chrome_the_test_is_skipped(
    playwright: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a channel that is not installed stands in for a machine without Chrome
    monkeypatch.delenv(browser_launch.BROWSER_ENV, raising=False)
    monkeypatch.setattr(browser_launch, "CHANNEL", "chrome-canary")
    with pytest.raises(pytest.skip.Exception, match="SURFCENSUS_BROWSER"):
        launch_browser(playwright).close()


def test_a_named_browser_that_is_no_file_is_an_error(
    playwright: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(browser_launch.BROWSER_ENV, "/nonexistent/chromium")
    with pytest.raises(FileNotFoundError, match="not a file"):
        launch_browser(playwright).close()


# --- the pages on the server --------------------------------------------------------------------

SEEN_TOUR = "try { localStorage.setItem('surfcensus.tour.seen:local', '1'); } catch (e) {}"
LABELS = ("sub-01", "sub-02")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Two synthetic subjects and the run that ``review(..., serve=False,
    require_marks=True)`` makes of them: the subjects directory and the run directory."""
    pytest.importorskip("fastapi")
    root = tmp_path_factory.mktemp("browser")
    data = root / "subjects"
    data.mkdir()
    for position, label in enumerate(LABELS):
        # the midline moves from subject to subject, so no two have the same files
        write_synthetic_subject(data, name=label, subdivisions=4, midline_x=-2.0 + 3.0 * position)
    review(data, run_dir=root / "run", serve=False, require_marks=True, jobs=1).close()
    return data, root / "run"


@pytest.fixture
def served(prepared: tuple[Path, Path], tmp_path: Path) -> Iterator[tuple[Any, Path, Path]]:
    """A server on a copy of the prepared run: its handle, the run directory, the data."""
    from surfcensus.app.server import start_server

    data, run = prepared
    run_dir = Path(shutil.copytree(run, tmp_path / "run"))
    handle = start_server(run_dir, port=0)
    yield handle, run_dir, data
    handle.stop()


class Rater:
    """A page of the app and what went wrong on it."""

    def __init__(self, browser: Any, url: str, *, width: int, seen_tour: bool = True) -> None:
        self.url = url
        self.context = browser.new_context(viewport={"width": width, "height": 800})
        if seen_tour:
            self.context.add_init_script(SEEN_TOUR)
        self.page = self.context.new_page()
        self.problems: list[str] = []
        self.elsewhere: list[str] = []
        page = self.page
        page.on("console", lambda m: self.problems.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda error: self.problems.append(str(error)))
        page.on("requestfailed", lambda r: self.problems.append(f"failed: {r.url}"))
        page.on(
            "response",
            lambda r: self.problems.append(f"{r.status}: {r.url}") if r.status >= 400 else None,
        )
        page.on(
            "request",
            lambda r: None if r.url.startswith((url, "data:")) else self.elsewhere.append(r.url),
        )

    def close(self) -> None:
        self.context.close()

    def viewer(self, name: str) -> str:
        return self.page.get_attribute("#viewer", f"data-{name}")

    def subject(self) -> str:
        """The uid of the subject on screen, once its tiles are loaded."""
        self.page.wait_for_selector("#rate[data-state='rating']")
        self.page.wait_for_selector("#viewer[data-ready='true']")
        return self.page.evaluate(
            "document.querySelector('.tile-gray').src.match(/media\\/([0-9a-f]{16})/)[1]"
        )

    def wait_for_another(self, uid: str) -> str:
        self.page.wait_for_function(
            "uid => { const tile = document.querySelector('.tile-gray');"
            " return tile && !tile.src.includes(uid)"
            " && document.getElementById('viewer').dataset.ready === 'true'; }",
            arg=uid,
        )
        return self.subject()

    def pause(self) -> None:
        if self.viewer("playing") == "true":
            self.page.keyboard.press("Space")
        self.page.wait_for_selector("#viewer[data-playing='false']")

    def frame(self) -> int:
        return int(self.viewer("frame"))

    def slice_of(self, panel: str) -> int:
        return int(self.page.get_attribute(f".panel[data-panel='{panel}']", "data-slice"))

    def submit_refused(self) -> bool:
        return self.page.get_attribute("#submit", "aria-disabled") == "true"

    def marks(self, panel: str) -> list[dict[str, Any]]:
        return self.page.eval_on_selector_all(
            f".panel[data-panel='{panel}'] g.mark",
            "marks => marks.map(m => ({id: m.dataset.clientId, cls: m.dataset.class,"
            " own: m.dataset.own, at: m.getAttribute('transform')}))",
        )


def app_rows(run_dir: Path, sql: str) -> list[tuple[Any, ...]]:
    connection = sqlite3.connect(run_dir / "app.sqlite")
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


def labels_by_uid(run_dir: Path) -> dict[str, str]:
    with PublicCatalog.open(run_dir) as public:
        subjects = public.subjects()
    return dict(zip(subjects["subject_uid"], subjects["subject_label"], strict=True))


@pytest.mark.parametrize("width", [1280, 800])
def test_a_rater_rates_marks_goes_back_and_reviews_on_the_real_pages(
    browser: Any, served: tuple[Any, Path, Path], width: int
) -> None:
    handle, run_dir, data = served
    rater = Rater(browser, handle.url, width=width)
    page = rater.page
    page.goto(handle.url)
    first = rater.subject()
    label = labels_by_uid(run_dir)[first]

    # 1 and 2: the page loaded what it asked for; three panels, each with its two layers
    assert rater.problems == [] and rater.elsewhere == []
    for panel in ("sagittal", "coronal", "axial"):
        sizes = page.eval_on_selector_all(
            f".panel[data-panel='{panel}'] img",
            "imgs => imgs.map(i => [i.className, i.naturalWidth])",
        )
        assert sorted(sizes) == [["tile tile-contours", 320], ["tile tile-gray", 320]], panel

    # 3: a subject plays when it is loaded; Space stops and starts, an arrow is one frame
    assert rater.viewer("playing") == "true"
    page.wait_for_function("() => Number(document.getElementById('viewer').dataset.frame) > 2")
    assert int(page.input_value("#frame")) > 0  # the slider moves with the frames
    rater.pause()
    at = rater.frame()
    page.wait_for_timeout(300)
    assert rater.frame() == at
    page.keyboard.press("ArrowRight")
    assert rater.frame() == at + 1 and int(page.input_value("#frame")) == at + 1
    page.keyboard.press("ArrowLeft")
    assert rater.frame() == at
    assert rater.viewer("mode") == "sync"
    page.keyboard.press("m")
    assert rater.viewer("mode") == "seq"
    page.keyboard.press("m")
    assert rater.viewer("mode") == "sync"

    # 4: the contours are hidden while s is held
    contours = ".panel[data-panel='sagittal'] img.tile-contours"
    assert page.is_visible(contours)
    page.keyboard.down("s")
    assert rater.viewer("contours") == "hidden" and not page.is_visible(contours)
    page.keyboard.up("s")
    assert rater.viewer("contours") == "shown" and page.is_visible(contours)

    # 5: nothing to submit before a rating; 3 is A major and makes class A active
    assert rater.submit_refused()
    assert page.get_attribute("#active-class", "data-class") == ""
    page.keyboard.press("3")
    pressed = page.eval_on_selector_all(
        ".level[aria-pressed='true']",
        "levels => levels.map(l => [l.dataset.criterion, l.dataset.value])",
    )
    assert pressed == [["A", "2"]]
    assert page.get_attribute("#active-class", "data-class") == "A"
    assert rater.submit_refused()

    # 6: a click on the place of a known pial vertex, computed from the surface file and the
    # geometry the page was given
    subject = data / label
    pial = {h: fs.read_geometry(str(subject / "surf" / f"{h}.pial"))[0] for h in ("lh", "rh")}
    t1 = nib.load(str(subject / "mri" / "T1.mgz"))
    tkr2vox = np.linalg.inv(t1.header.get_vox2ras_tkr())  # pyrefly: ignore[missing-attribute]
    render = page.evaluate(
        "() => fetch('/api/next').then(r => r.json()).then(b => b.assignment.render)"
    )
    sagittal = render["panels"]["sagittal"]
    index = rater.slice_of("sagittal")
    # the lh vertex nearest the shown slice plane among those well inside the tile
    voxels = (tkr2vox @ np.c_[pial["lh"], np.ones(len(pial["lh"]))].T).T[:, :3]
    px = sagittal["vox2pix"][0][0] * voxels[:, sagittal["h_voxel_axis"]] + sagittal["vox2pix"][0][2]
    py = sagittal["vox2pix"][1][1] * voxels[:, sagittal["v_voxel_axis"]] + sagittal["vox2pix"][1][2]
    off_plane = np.abs(voxels[:, sagittal["slice_voxel_axis"]] - sagittal["slices"][index])
    inside = (px > 40) & (px < 280) & (py > 40) & (py < 280)
    vertex = int(np.flatnonzero(inside)[np.argmin(off_plane[inside])])
    box = page.locator(".panel[data-panel='sagittal'] .stage").bounding_box()
    scale = box["width"] / 320
    page.mouse.click(box["x"] + px[vertex] * scale, box["y"] + py[vertex] * scale)
    (mark,) = rater.marks("sagittal")
    assert (mark["cls"], mark["own"]) == ("A", "true")
    drawn = [float(v) for v in mark["at"].removeprefix("translate(").removesuffix(")").split()]
    # a click lands on a whole screen pixel: within one of it, in tile pixels
    assert abs(drawn[0] - px[vertex]) <= 1 / scale and abs(drawn[1] - py[vertex]) <= 1 / scale
    # the mark is a point in the volume: it shows in another panel when its slice comes by
    seen_elsewhere = False
    for _ in range(len(render["modes"]["sync"]["frames"])):
        if rater.marks("coronal") or rater.marks("axial"):
            seen_elsewhere = True
            break
        page.keyboard.press("ArrowRight")
    assert seen_elsewhere
    assert all(m["own"] == "false" for m in rater.marks("coronal") + rater.marks("axial"))
    # marks are required in this run: A has its mark, B and C have no value yet
    assert rater.submit_refused()
    page.keyboard.press("4")
    assert rater.submit_refused()
    page.keyboard.press("7")
    assert not rater.submit_refused()

    # 7: Enter submits; the line counts one; the other subject is there
    page.keyboard.press("Enter")
    second = rater.wait_for_another(first)
    assert page.get_attribute("#progress-line", "data-done-total") == "1"
    assert app_rows(run_dir, "SELECT revision, current FROM submissions") == [(1, 1)]

    # 9: the stored mark is at the vertex that was aimed at
    stored = app_rows(
        run_dir,
        "SELECT class, panel, slice, px, py, hemi, pial_vertex, pial_dist_mm, tkr_x, tkr_y, tkr_z "
        "FROM marks",
    )
    assert len(stored) == 1
    cls, panel, at_slice, spx, spy, hemi, pial_vertex, dist, *tkr = stored[0]
    assert (cls, panel, at_slice, hemi) == ("A", "sagittal", index, "lh")
    assert abs(spx - px[vertex]) <= 1 / scale and abs(spy - py[vertex]) <= 1 / scale
    # the nearest vertex of the stored point, over both surfaces as they are on disk
    nearest = {h: np.linalg.norm(pial[h] - np.array(tkr), axis=1) for h in ("lh", "rh")}
    assert (
        pial_vertex == int(np.argmin(nearest["lh"])) and nearest["lh"].min() < nearest["rh"].min()
    )
    assert dist == pytest.approx(float(nearest["lh"].min()), abs=1e-6)
    assert (
        np.linalg.norm(pial["lh"][vertex] - np.array(tkr)) < 0.5 + 0.25 / scale + off_plane[vertex]
    )

    # 8: Backspace shows the first subject again, as it was rated
    rater.pause()
    page.keyboard.press("Backspace")
    assert rater.wait_for_another(second) == first
    pressed = page.eval_on_selector_all(
        ".level[aria-pressed='true']",
        "levels => levels.map(l => l.dataset.criterion + l.dataset.value)",
    )
    assert sorted(pressed) == ["A2", "B0", "C0"]
    assert page.get_attribute(".mark-count[data-class='A']", "data-count") == "1"
    # k on a subject that is rated leaves the rating as it is and goes on
    page.keyboard.press("k")
    assert rater.wait_for_another(first) == second
    assert app_rows(run_dir, "SELECT COUNT(*) FROM submissions") == [(1,)]
    assert app_rows(run_dir, "SELECT SUM(skip_count) FROM assignments") == [(0,)]

    # 10: one rater and not blind: the review lists the rated subject with its values
    page.goto(handle.url + "review")
    page.wait_for_selector("#review[data-state='open']")
    table = page.inner_text("#subjects")
    assert label in table and first in table
    headers = page.eval_on_selector_all(
        "#subjects thead th", "ths => ths.map(th => th.textContent.trim())"
    )
    row = page.eval_on_selector_all(
        "#subjects tbody tr",
        "(rows, uid) => [...rows.find(r => r.textContent.includes(uid)).cells]"
        ".map(c => c.textContent.trim())",
        first,
    )
    shown = dict(zip(headers, row, strict=True))
    assert (shown["final_A"], shown["final_B"], shown["final_C"]) == ("2", "0", "0")
    assert rater.problems == [] and rater.elsewhere == []
    rater.close()


def test_a_first_time_rater_learns_from_the_screen_how_to_get_to_the_next_subject(
    browser: Any, served: tuple[Any, Path, Path]
) -> None:
    """Nobody told this rater anything: the tour says how one moves on, a bare Enter is
    answered beside the button with the reason and the key that skips, and that key shows
    the other subject, with the skip on record and nothing stored as a rating."""
    handle, run_dir, _ = served
    rater = Rater(browser, handle.url, width=1280, seen_tour=False)
    page = rater.page
    page.goto(handle.url)
    page.wait_for_selector("#tour:not([hidden])")
    said = []
    for _ in range(6):
        if page.is_hidden("#tour"):
            break
        said.append((page.get_attribute("#tour", "data-step"), page.inner_text("#tour-text")))
        # while the tour is open a key of the rating screen does nothing
        page.keyboard.press("3")
        page.keyboard.press("Enter")
    assert [step for step, _ in said] == ["views", "rating", "marks", "moving", "help"]
    moving = dict(said)["moving"]
    assert "Enter" in moving and "k" in moving and "next subject" in moving
    assert page.is_hidden("#tour")
    first = rater.subject()
    assert page.locator(".level[aria-pressed='true']").count() == 0
    assert app_rows(run_dir, "SELECT COUNT(*) FROM submissions") == [(0,)]

    # the rater presses Enter, the one key everybody tries
    assert page.is_hidden("#refusal")
    page.keyboard.press("Enter")
    page.wait_for_selector("#refusal:not([hidden])")
    refusal = page.inner_text("#refusal")
    assert refusal.startswith("Not submitted.") and "A, B and C have no value yet" in refusal
    assert "comes back" in refusal
    # the box is where the eye is: within reach of the submit button, on screen
    box, button = page.locator("#refusal").bounding_box(), page.locator("#submit").bounding_box()
    assert box["y"] >= button["y"] and box["y"] - (button["y"] + button["height"]) < 40
    assert box["y"] + box["height"] <= 800
    # the key it names for leaving the subject is on the screen as a key
    keys = page.eval_on_selector_all("#refusal kbd", "kbds => kbds.map(k => k.textContent)")
    assert keys == ["u", "k"]
    assert rater.subject() == first  # nothing moved

    page.keyboard.press(keys[-1])
    second = rater.wait_for_another(first)
    assert second != first and page.is_hidden("#refusal")
    assert "1 skipped, waiting" in page.inner_text("#progress-line")
    assert app_rows(
        run_dir, "SELECT subject_uid, skip_count, completed_at FROM assignments ORDER BY seq"
    ) == [
        (first, 1, None),
        (second, 0, None),
    ]
    assert app_rows(run_dir, "SELECT COUNT(*) FROM submissions") == [(0,)]
    assert app_rows(run_dir, "SELECT COUNT(*) FROM events WHERE type = 'skip'") == [(1,)]
    # skipped twice round, the first subject is there again: it was deferred, not dropped
    page.keyboard.press("k")
    assert rater.wait_for_another(second) == first
    # a second visit of this rater's browser gets no tour
    page.reload()
    assert rater.subject() == first and page.is_hidden("#tour")
    # the only answers of 400 and above would be listed here
    assert rater.problems == [] and rater.elsewhere == []
    rater.close()
