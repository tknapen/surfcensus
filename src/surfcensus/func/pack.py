"""The pack of a session: both streams of every packable run as 8-bit volumes that the page
reslices, the stills, the traces, the contours and the frame check (format ``epi-pack``
version 2; ``func`` has the layout and ``func.model`` the JSON files). Work package E1.C.

The formats of chunks, stills and contours are those of stage 1's version 1
(``pack_run.py``): a volume is ``nx * ny * nz`` bytes, voxel ``(i, j, k)`` at
``i + nx * (j + ny * k)``; a chunk holds its volumes as residuals ``(value - ref) mod 256``;
one 8-bit window per run for both streams, 0 to the 99.5th percentile of the preprocessed
series inside the run's brain mask over the volumes 0, 10, 20, ...; the stills over the
steady-state volumes in float64; every ``.bin`` file is a zlib stream.

What changes against the prototype: the preprocessed series is read one volume at a time
through the array proxy, never whole; a pack names nothing and is traced to its inputs in
``pack.json``; its identity is :func:`content_digest`, over the decoded values, so that the
compressed bytes may differ between zlib builds. A pack directory is written under a temporary
name and renamed into place.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np
from jaxtyping import Float, UInt8

from surfcensus.config import EpiConfig
from surfcensus.func.model import ContourFile, PackInfo, SessionInfo
from surfcensus.records import PlaneContours, RunPackInput, SessionPlan

ZLIB_LEVEL = 6
MAX_CHUNK = 32
"""At most this many volumes per chunk, whatever ``chunk_bytes`` allows."""
WINDOW_STEP = 10
"""The window and the reference volume are taken over the volumes 0, 10, 20, ..."""
PERCENTILE = 99.5


class PackError(Exception):
    """A run cannot be packed. ``reason`` is one of ``records.PACK_REASONS``; ``detail`` names
    files by their role or name, never by path."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail

    def __reduce__(self) -> tuple[type[PackError], tuple[str, str]]:
        # pickle would call __init__ with the one argument of Exception.args
        return type(self), (self.reason, self.detail)


def quantise(
    volume: Float[np.ndarray, "i j k"], lo: float, hi: float
) -> UInt8[np.ndarray, "i j k"]:
    """``clip(floor((v - lo) * 255 / (hi - lo) + 0.5), 0, 255)`` in float32; NaN is 0."""
    raise NotImplementedError("work package E1.C: func/pack.py")


def volume_bytes(volume: UInt8[np.ndarray, "i j k"]) -> bytes:
    """A volume in the byte order of the pack (Fortran order)."""
    raise NotImplementedError("work package E1.C: func/pack.py")


def chunk_volumes(n_voxels: int, chunk_bytes: int) -> int:
    """Volumes per chunk: as many as fit ``chunk_bytes`` decoded, at least 1, at most
    :data:`MAX_CHUNK`."""
    raise NotImplementedError("work package E1.C: func/pack.py")


def series_volumes(
    series: Path, volumes: Sequence[int] | None = None
) -> Iterator[tuple[int, Float[np.ndarray, "i j k"]]]:
    """``(t, float32 volume)`` of a stored series, scaled per volume, read one volume at a time
    through the array proxy. ``volumes``: indices, all when None."""
    raise NotImplementedError("work package E1.C: func/pack.py")


def pack_run(session_dir: Path, run: RunPackInput, config: EpiConfig) -> PackInfo:
    """Write ``<session_dir>/<run.run>/`` (the streams, the stills, ``traces.json`` and
    ``pack.json`` last) and return what ``pack.json`` holds. Raises :class:`PackError` for a
    run that turns out not to be packable (``volume counts differ``, ``no brain mask`` for an
    empty mask or one off the grid)."""
    raise NotImplementedError("work package E1.C: func/pack.py")


def write_contours(
    session_dir: Path, grid_key: str, planes: Sequence[PlaneContours]
) -> Mapping[str, ContourFile]:
    """Write ``contours/<grid_key>/<plane>.bin`` and return, per plane, the
    ``func.model.ContourFile`` of it."""
    raise NotImplementedError("work package E1.C: func/pack.py")


def pack_session(
    out_dir: Path,
    session_id: str,
    plan: SessionPlan,
    runs: Sequence[RunPackInput],
    contours: Mapping[str, Sequence[PlaneContours]] | None,
    config: EpiConfig,
) -> SessionInfo:
    """Pack a whole session into ``<out_dir>/<session_id>/``: the contours of each grid
    (``contours`` by grid key; None without a FreeSurfer subject), every run of ``runs``, then
    ``session.json`` last. A run that raises :class:`PackError` is listed in ``session.json``
    with its reason and the others are packed."""
    raise NotImplementedError("work package E1.C: func/pack.py")


def content_digest(session_dir: Path, info: PackInfo) -> str:
    """SHA-256 over the decoded 8-bit volumes (raw then pre, in volume order), the references,
    the stills, the traces as written and the decoded contour bytes of its grid, in a fixed
    order: the identity of a pack whatever zlib wrote it."""
    raise NotImplementedError("work package E1.C: func/pack.py")
