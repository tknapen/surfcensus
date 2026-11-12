"""Three-plane occipital renders: tiles, composite video, render.json. See docs/rendering.md."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from surfcensus.render.model import RenderConfig, RenderInfo

if TYPE_CHECKING:
    from surfcensus.render.api import (
        OutputInSourceError,
        RenderError,
        render_subject,
        three_plane,
    )

__all__ = [
    "OutputInSourceError",
    "RenderConfig",
    "RenderError",
    "RenderInfo",
    "render_subject",
    "three_plane",
]


def __getattr__(name: str) -> object:
    # the renderer (and with it matplotlib) is imported on first use
    if name in ("OutputInSourceError", "RenderError", "render_subject", "three_plane"):
        return getattr(import_module("surfcensus.render.api"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
