"""Reading a pack back, in Python: for the tests and the checks, which compare what a pack
holds with values computed from the NIfTI files. The page has its own reader in JavaScript.
Work package E1.C.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from jaxtyping import UInt8

from surfcensus.func.model import PackInfo, SessionInfo


def read_session(session_dir: Path) -> SessionInfo | None:
    """``session.json`` of a session directory, or None when there is none of this version."""
    raise NotImplementedError("work package E1.C: func/packread.py")


def read_pack(session_dir: Path, run: str) -> PackInfo | None:
    """``pack.json`` of a run, or None when there is none of this version."""
    raise NotImplementedError("work package E1.C: func/packread.py")


def pack_on_disk(session_dir: Path, run: str) -> bool:
    """Whether the pack of the run is whole: ``pack.json`` of this version and a file under
    every name it lists, by two directory listings (the skip rule of a render)."""
    raise NotImplementedError("work package E1.C: func/packread.py")


def decode_volume(
    session_dir: Path, info: PackInfo, stream: str, t: int
) -> UInt8[np.ndarray, "i j k"]:
    """Volume ``t`` of a stream as the page shows it: the chunk inflated, the reference added
    back mod 256. Raises ``ValueError`` when a file's SHA-256 is not the one listed."""
    raise NotImplementedError("work package E1.C: func/packread.py")


def decode_still(
    session_dir: Path, info: PackInfo, stream: str, still: str
) -> UInt8[np.ndarray, "i j k"]:
    """``ref``, ``mean`` or ``sd`` of a stream, decoded."""
    raise NotImplementedError("work package E1.C: func/packread.py")


def decode_contours(session_dir: Path, grid_key: str, plane: str) -> bytes:
    """The inflated file of one plane's contours (``func.contours.read_contours`` reads it)."""
    raise NotImplementedError("work package E1.C: func/packread.py")
