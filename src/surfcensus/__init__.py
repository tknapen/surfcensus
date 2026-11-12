"""Score, render and rate FreeSurfer surfaces around the occipital pole.

See docs/ for the specification.
"""

from importlib import import_module

__version__ = "0.0.1"

# name -> module, or "module:attribute". Importing surfcensus loads the standard library
# only; `import surfcensus as sc; sc.render.three_plane(...)` and `sc.Run(...)` import what
# they need on first use.
_LAZY = {
    "render": "surfcensus.render",
    "sources": "surfcensus.sources",
    "Run": "surfcensus.run:Run",
}


def __getattr__(name: str) -> object:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, _, attribute = _LAZY[name].partition(":")
    loaded = import_module(module)
    return getattr(loaded, attribute) if attribute else loaded
