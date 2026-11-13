"""FreeSurfer per-vertex data (``surf/?h.thickness``).

Interface of milestone 3; the function is written by package 3.B.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from jaxtyping import Float

_OWNER = "package 3.B (annotation and morph data)"


def read_morph(path: Path | str) -> Float[np.ndarray, " n"]:
    """Read a morphometry file with nibabel: float32 in native byte order (the file stores
    big-endian values)."""
    raise NotImplementedError(f"fsio.morph.read_morph: {_OWNER}")
