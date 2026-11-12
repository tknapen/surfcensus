"""Score, render and rate FreeSurfer surfaces around the occipital pole.

See docs/ for the specification.
"""

from importlib import import_module

__version__ = "0.0.1"


def __getattr__(name: str) -> object:
    # `import surfcensus as sc; sc.render.three_plane(...)` works without importing the
    # renderer's dependencies until they are needed
    if name == "render":
        return import_module("surfcensus.render")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
