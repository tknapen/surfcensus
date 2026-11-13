"""FreeSurfer per-vertex data (``surf/?h.thickness``).

:func:`read_morph` raises ``OSError`` for a file that is absent or may not be opened and
``ValueError`` for one that is damaged. A ``ValueError`` names the file, never its path.
"""

from __future__ import annotations

from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np
from jaxtyping import Float

_NEW_FORMAT = 0xFFFFFF
"""The first three bytes of a file that stores float32 values and its count as an int32."""


def read_morph(path: Path | str) -> Float[np.ndarray, " n"]:
    """Read a morphometry file with nibabel: float32 in native byte order (the file stores
    big-endian values).

    Raises ValueError when the file does not hold the number of values its header declares.
    nibabel 5.5.0 does not compare the two: it returns the 8 values that are left of a file
    of 10 that was cut short, and reads a text file as a few numbers.

    Two headers exist, as ``nibabel.freesurfer.read_morph_data`` reads them. The one
    recon-all writes starts with the bytes ff ff ff, followed by the count as a big-endian
    int32 (seen in ``?h.thickness`` of FreeSurfer 6.0.1 and 7.3.2). Any other file is taken
    for the older form, whose first three bytes are the count and whose values are int16
    hundredths; no real file of that form was looked at.
    """
    path = Path(path)
    declared = _declared_count(path)
    data = fs.read_morph_data(str(path))
    if data.size != declared:
        raise ValueError(f"{path.name}: declares {declared} values and holds {data.size}")
    return np.ascontiguousarray(data, dtype=np.float32)


def _declared_count(path: Path) -> int:
    with open(path, "rb") as handle:
        head = handle.read(7)
    if len(head) < 3:
        raise ValueError(f"{path.name}: not a morphometry file (no header)")
    magic = int.from_bytes(head[:3], "big")
    if magic != _NEW_FORMAT:
        return magic
    if len(head) < 7:
        raise ValueError(f"{path.name}: not a morphometry file (the header ends early)")
    count = int.from_bytes(head[3:], "big", signed=True)
    if count < 0:
        # nibabel would pass a negative count on to numpy, which reads to the end of the file
        raise ValueError(f"{path.name}: not a morphometry file (declares {count} values)")
    return count
