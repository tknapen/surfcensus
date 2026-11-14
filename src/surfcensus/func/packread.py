"""Reading a pack back, in Python: for the tests and the checks, which compare what a pack
holds with values computed from the NIfTI files. The page has its own reader in JavaScript.
Work package E1.C.

Every compressed file is checked against the SHA-256 that ``pack.json`` or ``session.json``
lists for it before it is inflated, as the page's workers check it, and a file whose inflated
length is not what its entry implies is refused (``ValueError``). A decoded volume is
``(residual + ref) mod 256`` of the chunk that holds it, with ``ref`` the decoded ``ref.bin``
of the run and stream; the stills are stored as values. This module imports numpy and the
models only, not nibabel.
"""

from __future__ import annotations

import hashlib
import os
import zlib
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import numpy as np
from jaxtyping import UInt8

from surfcensus.func import (
    PACK_JSON,
    PACK_VERSION,
    PLANES,
    SESSION_JSON,
    STILLS,
    STREAMS,
    contour_file,
)
from surfcensus.func.model import PackFile, PackInfo, SessionInfo, read_info


def read_session(session_dir: Path) -> SessionInfo | None:
    """``session.json`` of a session directory, or None when there is none of this version."""
    info = read_info(Path(session_dir) / SESSION_JSON, SessionInfo)
    if info is None or info.version != PACK_VERSION:
        return None
    return info


def read_pack(session_dir: Path, run: str) -> PackInfo | None:
    """``pack.json`` of a run, or None when there is none of this version."""
    info = read_info(Path(session_dir) / run / PACK_JSON, PackInfo)
    if info is None or info.version != PACK_VERSION or info.run != run:
        return None
    return info


def _listed(info: PackInfo) -> list[str]:
    """Every file ``pack.json`` lists, relative to the session directory."""
    files = [info.traces.file]
    for stream in STREAMS:
        files += [getattr(info.streams[stream], still).file for still in STILLS]
        files += [chunk.file for chunk in info.streams[stream].chunks]
    return files


def pack_on_disk(session_dir: Path, run: str) -> bool:
    """Whether the pack of the run is whole: ``pack.json`` of this version and a file under
    every name it lists, by two directory listings (the skip rule of a render).

    The listings are those of ``<run>/raw`` and ``<run>/pre`` (and of ``<run>`` for
    ``traces.json``); a name that is there counts, whatever it holds, as for a render. A pack
    without both streams, or whose ``pack.json`` names another run, is not whole.
    """
    session_dir = Path(session_dir)
    info = read_pack(session_dir, run)
    if info is None or set(info.streams) != set(STREAMS):
        return False
    present: dict[str, set[str]] = {}
    for relative in _listed(info):
        folder, _, name = relative.rpartition("/")
        if folder not in present:
            try:
                present[folder] = set(os.listdir(session_dir / folder))
            except OSError:
                return False
        if name not in present[folder]:
            return False
    return True


@lru_cache(maxsize=4)
def _inflated(path: str, sha256: str | None, mtime_ns: int, size: int) -> bytes:
    blob = Path(path).read_bytes()
    if sha256 is not None and hashlib.sha256(blob).hexdigest() != sha256:
        raise ValueError(f"{Path(path).name}: the SHA-256 is not the one listed")
    try:
        return zlib.decompress(blob)
    except zlib.error as err:
        raise ValueError(f"{Path(path).name} is not a zlib stream: {err}") from None


def inflate_checked(path: Path, sha256: str | None) -> bytes:
    """The inflated content of a file of a pack, after its compressed bytes were checked
    against ``sha256`` (None: not checked). Raises ``ValueError`` when the digest differs or
    the file is no zlib stream. The last four files read are kept, keyed by path, digest,
    modification time and size, so that decoding the volumes of one chunk inflates it once."""
    stat = Path(path).stat()
    return _inflated(str(path), sha256, stat.st_mtime_ns, stat.st_size)


def _file(session_dir: Path, entry: PackFile) -> bytes:
    return inflate_checked(Path(session_dir) / entry.file, entry.sha256)


def stream_bytes(session_dir: Path, info: PackInfo, stream: str) -> Iterator[bytes]:
    """The decoded 8-bit volumes of a stream in volume order, each as ``n_voxels`` bytes in the
    byte order of the pack. Needs no grid shape: a volume is as long as ``ref.bin``. Raises
    ``ValueError`` for a file whose digest or length is not what ``pack.json`` says, and for
    chunks that do not cover the volumes 0 to ``n_volumes - 1`` in order."""
    session_dir = Path(session_dir)
    files = info.streams[stream]
    ref = np.frombuffer(_file(session_dir, files.ref), np.uint8)
    expected = 0
    for chunk in files.chunks:
        if chunk.first != expected:
            raise ValueError(f"chunk {chunk.file} starts at {chunk.first}, not at {expected}")
        data = _file(session_dir, chunk)
        if len(data) != chunk.count * len(ref):
            raise ValueError(
                f"chunk {chunk.file} holds {len(data)} bytes, not {chunk.count} volumes"
            )
        residuals = np.frombuffer(data, np.uint8).reshape(chunk.count, len(ref))
        values = residuals + ref  # uint8 arithmetic: mod 256
        for row in values:
            yield row.tobytes()
        expected += chunk.count
    if expected != info.n_volumes:
        raise ValueError(f"the chunks of {stream} hold {expected} of {info.n_volumes} volumes")


def _shape(session_dir: Path, info: PackInfo) -> tuple[int, int, int]:
    session = read_session(session_dir)
    if session is not None:
        for grid in session.grids:
            if grid.grid_key == info.grid_key:
                return grid.shape
    raise ValueError(
        f"no {SESSION_JSON} that lists the grid {info.grid_key}: the shape of a volume of run"
        f" {info.run} is not known"
    )


def _volume(data: bytes, shape: tuple[int, int, int]) -> UInt8[np.ndarray, "i j k"]:
    if len(data) != shape[0] * shape[1] * shape[2]:
        raise ValueError(f"a volume of {len(data)} bytes is not one of the grid {shape}")
    return np.frombuffer(data, np.uint8).reshape(shape, order="F").copy()


def decode_volume(
    session_dir: Path, info: PackInfo, stream: str, t: int
) -> UInt8[np.ndarray, "i j k"]:
    """Volume ``t`` of a stream as the page shows it: the chunk inflated, the reference added
    back mod 256. Raises ``ValueError`` when a file's SHA-256 is not the one listed.

    The shape comes from the grid that ``session.json`` lists under the pack's ``grid_key``
    (``pack.json`` holds no shape), so a run's pack is decoded once its session is written.
    """
    if stream not in STREAMS:
        raise ValueError(f"no stream {stream!r}")
    if not 0 <= t < info.n_volumes:
        raise IndexError(f"volume {t} of a pack of {info.n_volumes}")
    shape = _shape(session_dir, info)
    files = info.streams[stream]
    ref = np.frombuffer(_file(session_dir, files.ref), np.uint8)
    for chunk in files.chunks:
        if chunk.first <= t < chunk.first + chunk.count:
            data = _file(session_dir, chunk)
            n = len(ref)
            if len(data) != chunk.count * n:
                raise ValueError(f"chunk {chunk.file} is not {chunk.count} volumes long")
            at = (t - chunk.first) * n
            residual = np.frombuffer(data, np.uint8, n, at)
            return _volume((residual + ref).tobytes(), shape)
    raise ValueError(f"no chunk of {stream} holds volume {t}")


def decode_still(
    session_dir: Path, info: PackInfo, stream: str, still: str
) -> UInt8[np.ndarray, "i j k"]:
    """``ref``, ``mean`` or ``sd`` of a stream, decoded."""
    if stream not in STREAMS or still not in STILLS:
        raise ValueError(f"no still {stream}/{still}")
    entry = getattr(info.streams[stream], still)
    return _volume(_file(session_dir, entry), _shape(session_dir, info))


def decode_contours(session_dir: Path, grid_key: str, plane: str) -> bytes:
    """The inflated file of one plane's contours (``func.contours.read_contours`` reads it).

    Checked against the SHA-256 that ``session.json`` lists for it when the session lists the
    grid with contours; inflated unchecked otherwise (a session being written)."""
    if plane not in PLANES:
        raise ValueError(f"no plane {plane!r}")
    sha = None
    session = read_session(session_dir)
    if session is not None:
        for grid in session.grids:
            if grid.grid_key == grid_key and grid.contours is not None:
                sha = grid.contours[plane].sha256
    return inflate_checked(Path(session_dir) / contour_file(grid_key, plane), sha)


def byte_values(window: tuple[float, float]) -> np.ndarray:
    """What each stored value 0 to 255 of a run with this window stands for, float64: the
    middle of the values that quantise to it, ``lo + b * (hi - lo) / 255`` (``RULES["value"]``
    of ``func.model``). 0 also holds everything below ``lo`` and 255 everything above ``hi``."""
    lo, hi = float(window[0]), float(window[1])
    return lo + np.arange(256, dtype=np.float64) * ((hi - lo) / 255.0)


def common_scale(window: tuple[float, float], top: float) -> np.ndarray:
    """The table by which the page shows a run under one window ``(0, top)`` for the whole
    stream, with intensity normalization off (``RULES["common_scale"]``): stored value ``b``
    of a run with window ``(0, hi)`` is shown as ``floor(b * (hi / top) + 0.5)``, clipped to
    0..255, uint8. ``top`` is the largest upper limit of the runs of the stream; a run with
    that limit keeps its values."""
    if not window[0] == 0.0:
        raise ValueError("the windows of a pack start at 0")
    if not top >= window[1] > 0:
        raise ValueError("top is the largest upper limit of the stream")
    scaled = np.floor(np.arange(256, dtype=np.float64) * (float(window[1]) / float(top)) + 0.5)
    return np.clip(scaled, 0, 255).astype(np.uint8)
