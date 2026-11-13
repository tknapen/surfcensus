"""The rater app in a real browser (``just test-browser``; marked ``browser``, not part of the
default run or of CI).

Only the first step is here: the browser that the tests are given starts, runs a script of
the kind the frontend is made of and gives a screenshot. The test of the rating page itself
is added with the page (docs/dev/contract.md, section 9.2).
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from typing import Any

import pytest
from browser_launch import open_browser  # pyrefly: ignore[missing-import]
from PIL import Image

pytest.importorskip("playwright.sync_api")

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
def browser() -> Iterator[Any]:
    with open_browser() as started:
        yield started


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
