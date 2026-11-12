"""Where subjects come from: local directories and public buckets. See docs/architecture.md.

``surfcensus.sources.local(path)`` makes a local source. Nothing is imported until it is
used: ``config.py`` imports ``sources/base.py``, which runs this file first, so this file
must not import anything that imports ``config.py``.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from surfcensus.sources.local import LocalSource, local

__all__ = ["LocalSource", "local"]


def __getattr__(name: str) -> object:
    # `local` is both the submodule and the function in it. Once the submodule is imported
    # the name is bound to the module, which is callable and makes the source (see the end
    # of sources/local.py), so `sources.local(path)` works whatever was imported first.
    if name == "local":
        return import_module("surfcensus.sources.local")
    if name == "LocalSource":
        return import_module("surfcensus.sources.local").LocalSource
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
