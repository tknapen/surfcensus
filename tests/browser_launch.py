"""How the browser tests of the rater app get their browser.

They drive a browser that is installed already; nothing is downloaded. The executable is
the one named by the environment variable ``SURFCENSUS_BROWSER`` (a path to a Chromium or
Chrome binary), else the installed Chrome (Playwright's channel ``chrome``). With the
variable unset and no Chrome, the test is skipped. A variable that names no file is an
error, not a skip: whoever set it wanted the test to run.

The Python package ``playwright`` comes from the dependency group ``browser``
(``just test-browser``); a module that uses this one starts with
``pytest.importorskip("playwright.sync_api")``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

BROWSER_ENV = "SURFCENSUS_BROWSER"
CHANNEL = "chrome"


def launch_options(environ: dict[str, str] | None = None) -> dict[str, str]:
    """The keyword arguments of ``chromium.launch`` that name the browser."""
    path = (os.environ if environ is None else environ).get(BROWSER_ENV)
    if not path:
        return {"channel": CHANNEL}
    if not Path(path).is_file():
        raise FileNotFoundError(f"{BROWSER_ENV} names {path}, which is not a file")
    return {"executable_path": path}


@contextmanager
def open_browser(**launch: Any) -> Iterator[Any]:
    """A headless browser (``playwright.sync_api.Browser``), closed on exit. Skips the test
    when no executable is named and Chrome is not installed."""
    sync_api = import_module("playwright.sync_api")
    options = launch_options()
    with sync_api.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(**options, **launch)
        except sync_api.Error as exc:
            if "channel" in options and "is not found" in str(exc):
                pytest.skip(f"no browser: set {BROWSER_ENV} to an executable, or install Chrome")
            raise
        try:
            yield browser
        finally:
            browser.close()
