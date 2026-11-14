"""The pack of a session: both streams of every packable run as 8-bit volumes that the page
reslices, the stills, the traces, the contours and the frame check (format ``epi-pack``
version 2; ``func`` has the layout and ``func.model`` the JSON files). Work package E1.C.

The formats of chunks, stills and contours are those of stage 1's version 1
(``pack_run.py``): a volume is ``nx * ny * nz`` bytes, voxel ``(i, j, k)`` at
``i + nx * (j + ny * k)``; a chunk holds its volumes as residuals ``(value - ref) mod 256``;
one 8-bit window per run for both streams, 0 to the 99.5th percentile of the preprocessed
series inside the run's brain mask over the volumes 0, 10, 20, ...; the stills over the
steady-state volumes in float64; every ``.bin`` file is a zlib stream.

What changes against the prototype: the series are read one volume at a time through the
array proxy, never whole; a pack names nothing and is traced to its inputs in ``pack.json``
(role, base, size and SHA-256, never a name or a path); its identity is
:func:`content_digest`, over the decoded values and the JSON that defines the pack, so that
the compressed bytes may differ between zlib builds. A run's pack is written into a hidden
temporary directory beside it, ``pack.json`` last, and renamed into place under the ``flock``
of the session directory; ``session.json`` is written last of a session. So the presence of
``pack.json`` marks a complete pack and that of ``session.json`` a complete session, as
``render.json`` marks a render.

The pack is the tool's display cache and nothing more: 8-bit values of the files that exist,
on the grid of each preprocessed series, kept in the run directory (or the ``--out`` directory
of ``epi-session``) so that a browser can play them, and made again from the data at any time.
It writes no image volume, no resampled series and nothing into or beside the data, and builds
no grid common to runs or sessions. Each run is stored under its own window and the window is
recorded, so that the page can show the stream of a subject either way (decision D38).

The preprocessed stream is the series as the pipeline wrote it, scaled per volume and
quantised; the raw stream comes from ``func.geometry.raw_stream`` (the one placement of D36)
and is only quantised. Nothing else is done to either.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import secrets
import shutil
import stat
import time
import zlib
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from jaxtyping import Bool, Float, UInt8

from surfcensus import __version__
from surfcensus._io import atomic_path, canonical_json, sha256_file
from surfcensus.config import EpiConfig
from surfcensus.func import (
    CONTOUR_UNIT,
    CONTOURS_DIR,
    ID_PATTERN,
    PACK_JSON,
    PACK_VERSION,
    PLANES,
    SESSION_JSON,
    STILLS,
    STREAMS,
    TRACES_JSON,
    chunk_name,
    contour_file,
    geometry,
    stream_file,
)
from surfcensus.func.model import (
    ChunkFile,
    ContourFile,
    FrameInfo,
    GridInfo,
    PackFile,
    PackInfo,
    PackInput,
    PackSettings,
    RegistrationInfo,
    SessionInfo,
    SessionRunInfo,
    StreamFiles,
    TracesInfo,
    corrections_info,
    info_text,
    read_info,
)
from surfcensus.func.packread import inflate_checked, pack_on_disk, read_pack, stream_bytes
from surfcensus.records import (
    IDENTITY4,
    NO_BRAIN_MASK,
    NO_PREPROCESSED_SERIES,
    NO_RAW_SERIES,
    PACK_FAILURE,
    VOLUME_COUNTS_DIFFER,
    FrameDisplacement,
    Grid,
    Matrix4,
    PlaneContours,
    RunPackInput,
    SessionPlan,
)
from surfcensus.sources.base import FUNC_RAW_ROLES, FuncRole

try:
    import fcntl
except ImportError:  # Windows has no flock: the swap of a run directory is not serialized there
    fcntl = None

ZLIB_LEVEL = 6
MAX_CHUNK = 32
"""At most this many volumes per chunk, whatever ``chunk_bytes`` allows."""
WINDOW_STEP = 10
"""The window and the reference volume are taken over the volumes 0, 10, 20, ..."""
PERCENTILE = 99.5

DIGEST_FORMAT = "epi-pack content digest 1"
"""The first line of the text that :func:`content_digest` hashes; it changes when the rule of
the digest changes."""

_GRID_TOLERANCE = 1e-4
"""A brain mask lies on the grid when its shape is the grid's and its affine agrees to this
(``geometry.GRID_TOLERANCE``)."""

_UNREADABLE = (OSError, ValueError, EOFError, zlib.error, nib.filebasedimages.ImageFileError)


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


# --- the values ---------------------------------------------------------------------------


def quantise(
    volume: Float[np.ndarray, "i j k"], lo: float, hi: float
) -> UInt8[np.ndarray, "i j k"]:
    """``clip(floor((v - lo) * 255 / (hi - lo) + 0.5), 0, 255)`` in float32; NaN is 0.

    The expression of stage 1's ``pack_run.quantise``, operation for operation: the factor
    ``255 / (hi - lo)`` is computed in float64 and rounded to float32 once, the product and the
    sum are float32, and the cast after the clip truncates, which for values of 0 and more is
    the floor. An infinity is clipped like a large number.
    """
    k = np.float32(255.0 / (hi - lo))
    v = np.nan_to_num(np.asarray(volume, np.float32), nan=0.0)
    return np.clip((v - np.float32(lo)) * k + np.float32(0.5), 0, 255).astype(np.uint8)


def volume_bytes(volume: UInt8[np.ndarray, "i j k"]) -> bytes:
    """A volume in the byte order of the pack (Fortran order)."""
    return np.asarray(volume, np.uint8).tobytes(order="F")


def chunk_volumes(n_voxels: int, chunk_bytes: int) -> int:
    """Volumes per chunk: as many as fit ``chunk_bytes`` decoded, at least 1, at most
    :data:`MAX_CHUNK`."""
    if n_voxels < 1:
        raise ValueError("a grid has at least one voxel")
    return max(1, min(MAX_CHUNK, int(chunk_bytes) // int(n_voxels)))


def _open_series(series: Path) -> Any:
    # keep_file_open: a gzip stream is read front to back once, not reopened per volume
    return nib.load(str(series), keep_file_open=True)


def _n_volumes(img: Any) -> int:
    shape = tuple(img.shape)
    return int(shape[3]) if len(shape) > 3 else 1


def series_volumes(
    series: Path, volumes: Sequence[int] | None = None
) -> Iterator[tuple[int, Float[np.ndarray, "i j k"]]]:
    """``(t, float32 volume)`` of a stored series, scaled per volume, read one volume at a time
    through the array proxy. ``volumes``: indices, all when None.

    The values are ``stored[..., t].astype(float32) * slope + inter``, the expression of stage
    1 (``pack_run.scaled``), with ``slope`` and ``inter`` as nibabel's proxy holds them (1 and
    0 for a series without scaling). Only the bytes of volume ``t`` are read (nibabel's
    ``_get_unscaled`` with a slicer); a gzip stream is read forward through one open file. A
    series of three dimensions is one volume, ``t`` 0.
    """
    img = _open_series(Path(series))
    proxy = img.dataobj
    slope, inter = float(proxy.slope), float(proxy.inter)
    four_d = len(proxy.shape) > 3
    n = _n_volumes(img)
    wanted = range(n) if volumes is None else volumes
    try:
        for t in wanted:
            t = int(t)
            if not 0 <= t < n:
                raise IndexError(f"volume {t} of a series of {n}")
            slicer = (slice(None), slice(None), slice(None), t) if four_d else (slice(None),) * 3
            stored = proxy._get_unscaled(slicer)
            yield t, np.asarray(stored).astype(np.float32) * slope + inter
    finally:
        # the proxy keeps its file open (keep_file_open); close it with the generator
        opener = getattr(proxy, "_opener", None)
        if opener is not None:
            opener.close_if_mine()


def _raw_volumes(
    run: RunPackInput, volumes: Sequence[int] | None, portable: bool
) -> Iterator[tuple[int, Float[np.ndarray, "i j k"]]]:
    """The raw stream of a run on its display grid: the one source of raw volumes of the pack
    (``func.geometry.raw_stream``, decision D36)."""
    return iter(
        geometry.raw_stream(
            run.files[FuncRole.RAW_BOLD],
            run.raw_affine.matrix,
            run.grid,
            run.anchor_matrix,
            run.display_to_anat,
            volumes,
            portable,
        )
    )


def _guarded(
    volumes: Iterator[tuple[int, Float[np.ndarray, "i j k"]]],
    role: FuncRole,
    wanted: Sequence[int],
) -> Iterator[Float[np.ndarray, "i j k"]]:
    """The volumes ``wanted`` of a series, in that order, with an error of reading turned into
    :class:`PackError`, and a series that yields another volume, or fewer or more, refused."""
    count = 0
    while True:
        try:
            item = next(volumes)
        except StopIteration:
            break
        except _UNREADABLE as err:
            raise PackError(
                PACK_FAILURE, f"unreadable input ({role.value}: {_no_path(err)})"
            ) from None
        if count >= len(wanted) or item[0] != wanted[count]:
            due = wanted[count] if count < len(wanted) else "none"
            raise PackError(PACK_FAILURE, f"{role.value}: volume {item[0]} where {due} was due")
        count += 1
        yield item[1]
    if count != len(wanted):
        raise PackError(PACK_FAILURE, f"{role.value}: {count} volumes of {len(wanted)}")


def _no_path(err: BaseException) -> str:
    """The text of an error with every absolute path replaced by its last component."""
    text = str(err.strerror) if isinstance(err, OSError) and err.strerror else str(err)
    return re.sub(r"(?:/[^/\s'\"]+)+/([^/\s'\"]+)", r"\1", text)


# --- one run --------------------------------------------------------------------------------


def _read_mask(
    path: Path, grid_shape: tuple[int, int, int], grid_affine: np.ndarray
) -> Bool[np.ndarray, "i j k"]:
    try:
        img = nib.load(str(path))
        data = np.asarray(img.dataobj)
    except _UNREADABLE as err:
        raise PackError(PACK_FAILURE, f"unreadable input (pre_mask: {_no_path(err)})") from None
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]
    if tuple(data.shape) != tuple(grid_shape):
        raise PackError(NO_BRAIN_MASK, "the brain mask is not on the grid of the series")
    if not np.allclose(np.asarray(img.affine, float), grid_affine, atol=_GRID_TOLERANCE):
        raise PackError(NO_BRAIN_MASK, "the brain mask is not on the grid of the series")
    mask = data > 0
    if not mask.any():
        raise PackError(NO_BRAIN_MASK, "the brain mask is empty")
    return mask


def _header_volumes(role: FuncRole, path: Path) -> tuple[Any, int]:
    try:
        img = nib.load(str(path))
        return img, _n_volumes(img)
    except _UNREADABLE as err:
        raise PackError(PACK_FAILURE, f"unreadable input ({role.value}: {_no_path(err)})") from None


_DIGESTS: dict[tuple[str, int, int, int], str] = {}
"""SHA-256 of files already read in this process, by (path, device, inode, mtime_ns): the
FreeSurfer files and the anchor's transform are inputs of every run of a session."""


def _file_digest(path: Path) -> tuple[int, str]:
    st = path.stat()
    key = (str(path), st.st_dev, st.st_ino, st.st_mtime_ns)
    digest = _DIGESTS.get(key)
    if digest is None:
        digest = sha256_file(path)
        if len(_DIGESTS) > 256:
            _DIGESTS.clear()
        _DIGESTS[key] = digest
    return st.st_size, digest


def _inputs(
    files: Mapping[FuncRole, Path],
    session_inputs: Sequence[tuple[str, str, Path]] = (),
) -> tuple[PackInput, ...]:
    """Every file the pack was given, in ``FuncRole`` order (raw roles come first there), then
    the files outside the run that it depends on (``RunPackInput.session_inputs``), in the
    order given."""
    items: list[tuple[str, str, Path]] = [
        (role.value, "raw" if role in FUNC_RAW_ROLES else "deriv", Path(files[role]))
        for role in FuncRole
        if role in files
    ]
    items += [(role, base, Path(path)) for role, base, path in session_inputs]
    out = []
    for role, base, path in items:
        try:
            size, digest = _file_digest(path)
        except OSError as err:
            raise PackError(PACK_FAILURE, f"unreadable input ({role}: {_no_path(err)})") from None
        out.append(PackInput(role=role, base=base, size=size, sha256=digest))  # pyrefly: ignore[bad-argument-type]
    return tuple(out)


class _Writer:
    """Writes the compressed files of one run under a temporary session-like directory and
    keeps what ``pack.json`` lists of them."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.total = 0

    def put(self, relative: str, data: bytes) -> PackFile:
        blob = zlib.compress(data, ZLIB_LEVEL)
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        self.total += len(blob)
        return PackFile(file=relative, bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())


def _steady(n: int, non_steady: Sequence[int]) -> np.ndarray:
    steady = np.ones(n, bool)
    steady[[t for t in non_steady if 0 <= t < n]] = False
    if steady.sum() < 2:  # stage 1's rule: then the stills are over every volume
        steady[:] = True
    return steady


@dataclass(frozen=True)
class _Checked:
    """What the headers and the small files of a run say, checked before any volume is read."""

    n: int
    mask: np.ndarray
    non_steady: tuple[int, ...]
    steady: np.ndarray


def _check(run: RunPackInput) -> _Checked:
    """The refusals that need no volume: missing roles, grids, volume counts, the mask."""
    if not re.fullmatch(ID_PATTERN, run.run):
        raise ValueError(f"not a run id: {run.run!r}")
    files = run.files
    for role, reason in (
        (FuncRole.PRE_BOLD, NO_PREPROCESSED_SERIES),
        (FuncRole.RAW_BOLD, NO_RAW_SERIES),
        (FuncRole.PRE_MASK, NO_BRAIN_MASK),
    ):
        if role not in files:
            raise PackError(reason, f"the run has no {role.value} file")
    shape = tuple(int(x) for x in run.grid.shape)
    pre_img, n = _header_volumes(FuncRole.PRE_BOLD, files[FuncRole.PRE_BOLD])
    if tuple(int(x) for x in pre_img.shape[:3]) != shape:
        raise PackError(PACK_FAILURE, "the preprocessed series is not on the grid of the run")
    _, n_raw = _header_volumes(FuncRole.RAW_BOLD, files[FuncRole.RAW_BOLD])
    if n_raw != n:
        raise PackError(
            VOLUME_COUNTS_DIFFER, f"the raw series has {n_raw} volumes, the preprocessed {n}"
        )
    motion = run.motion
    if motion is not None and (len(motion.fd) != n or len(motion.dvars) != n):
        raise PackError(
            VOLUME_COUNTS_DIFFER,
            f"the confounds table has {len(motion.fd)} rows, the series {n} volumes",
        )
    mask = _read_mask(files[FuncRole.PRE_MASK], shape, np.asarray(run.grid.affine, float))
    non_steady = tuple(t for t in motion.non_steady if 0 <= t < n) if motion else ()
    return _Checked(n=n, mask=mask, non_steady=non_steady, steady=_steady(n, non_steady))


def _pre(run: RunPackInput, volumes: Sequence[int]) -> Iterator[np.ndarray]:
    path = run.files[FuncRole.PRE_BOLD]
    return _guarded(series_volumes(path, volumes), FuncRole.PRE_BOLD, volumes)


def _raw(run: RunPackInput, volumes: Sequence[int], portable: bool) -> Iterator[np.ndarray]:
    return _guarded(_raw_volumes(run, volumes, portable), FuncRole.RAW_BOLD, volumes)


def _sd(sums: np.ndarray, squares: np.ndarray, n_steady: int) -> np.ndarray:
    """The temporal SD (n - 1) of stage 1, from float64 sums and sums of squares."""
    if n_steady < 2:
        return np.zeros(sums.shape, np.float64)
    return np.sqrt(np.maximum(squares - sums**2 / n_steady, 0.0) / (n_steady - 1))


def pack_run(session_dir: Path, run: RunPackInput, config: EpiConfig) -> PackInfo:
    """Write ``<session_dir>/<run.run>/`` (the streams, the stills, ``traces.json`` and
    ``pack.json`` last) and return what ``pack.json`` holds. Raises :class:`PackError` for a
    run that turns out not to be packable (``volume counts differ``, ``no brain mask`` for an
    empty mask or one off the grid).

    The run is stored under its own window, ``(0, the 99.5th percentile of its preprocessed
    series inside its brain mask over the volumes 0, 10, 20, ...)``, for both streams, and its
    SD stills under their own (0 to the 99.5th percentile of the temporal SD of the
    preprocessed series inside the mask): the most of each run in 8 bits. Both are recorded
    exactly (``PackInfo.window``, ``sd_window``), so that the page can show the stream of a
    subject under one common scale as well (``func.packread.byte_values``; decision D38).

    Two passes over the volumes, as in stage 1. The first reads the volumes 0, 10, 20, ... of
    the preprocessed series twice (once for the window, once for the reference volume of each
    stream, which is quantised under the window) and those of the raw stream once. The second
    reads every volume of both streams once, writes the chunks as they fill and adds the
    steady-state volumes to the sums of the stills. What is held: one volume of each stream,
    the float64 sums and squares (32 bytes per voxel and stream), the chunk being filled (at
    most ``chunk_bytes`` per stream), and the masked values of the volumes 0, 10, 20, ... of
    the preprocessed series for the percentile (0.4 bytes per masked voxel and volume).

    The pack is the tool's display cache: 8-bit values of the existing files on the grid of
    the preprocessed series, in the run directory or the ``--out`` directory. It writes no
    image volume, and nothing into or beside the data, and can be deleted and made again.

    Everything is written into a hidden directory ``.<run>.<16 hex>.tmp`` in ``session_dir``,
    ``pack.json`` last, and moved to ``<session_dir>/<run>`` under the ``flock`` of
    ``session_dir``; an earlier pack of the run is moved aside first and removed. An
    interrupted call leaves the earlier pack, or none, and removes its temporary directory.
    The contours of the run's grid (``contours/<grid_key>/``) are read from ``session_dir``
    for the digest when they are there, so :func:`pack_session` writes them first.
    """
    session_dir = Path(session_dir)
    checked = _check(run)
    n, mask, steady, non_steady = checked.n, checked.mask, checked.steady, checked.non_steady
    motion = run.motion
    shape = tuple(int(x) for x in run.grid.shape)
    inputs = _inputs(run.files, run.session_inputs)
    k_chunk = chunk_volumes(int(np.prod(shape)), config.chunk_bytes)
    portable = config.portable_voxel_map

    # pass A, the volumes 0, 10, 20, ...: the window of the run and the reference volumes
    sub = list(range(0, n, WINDOW_STEP))
    values = np.empty((len(sub), int(mask.sum())), np.float32)
    for row, vol in enumerate(_pre(run, sub)):
        values[row] = vol[mask]
    # the values of stage 1's np.stack, partitioned in place instead of copied
    hi = float(np.percentile(values, PERCENTILE, overwrite_input=True))
    del values
    if not hi > 0:
        raise PackError(PACK_FAILURE, "the preprocessed series has no signal inside the mask")
    window = (0.0, hi)
    # the volumes 0, 10, 20, ...: the reference volume of each stream
    sub = list(range(0, n, WINDOW_STEP))
    ref_sum = {s: np.zeros(shape, np.float64) for s in STREAMS}
    for vol in _pre(run, sub):
        ref_sum["pre"] += quantise(vol, *window)
    for vol in _raw(run, sub, portable):
        ref_sum["raw"] += quantise(vol, *window)
    ref = {s: np.floor(ref_sum[s] / len(sub) + 0.5).astype(np.uint8) for s in STREAMS}
    del ref_sum

    session_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = session_dir / f".{run.run}.{secrets.token_hex(8)}.tmp"
    try:
        writer = _Writer(tmp_root)
        sums = {s: np.zeros(shape, np.float64) for s in STREAMS}
        squares = {s: np.zeros(shape, np.float64) for s in STREAMS}
        chunks: dict[str, list[ChunkFile]] = {s: [] for s in STREAMS}
        pending: dict[str, list[bytes]] = {s: [] for s in STREAMS}
        hashers = {s: hashlib.sha256() for s in STREAMS}

        def flush(stream: str, first: int) -> None:
            name = chunk_name(len(chunks[stream]))
            entry = writer.put(stream_file(run.run, stream, name), b"".join(pending[stream]))
            chunks[stream].append(
                ChunkFile(**entry.model_dump(), first=first, count=len(pending[stream]))
            )
            pending[stream].clear()

        # pass B, every volume: the chunks, and the sums for the stills
        every = range(n)
        sources = {"raw": _raw(run, every, portable), "pre": _pre(run, every)}
        for t in range(n):
            for stream in STREAMS:
                vol = next(sources[stream])
                if steady[t]:
                    v64 = vol.astype(np.float64)
                    sums[stream] += v64
                    squares[stream] += v64 * v64
                q = quantise(vol, *window)
                hashers[stream].update(volume_bytes(q))
                pending[stream].append(volume_bytes(q - ref[stream]))  # uint8: mod 256
                if len(pending[stream]) == k_chunk or t == n - 1:
                    flush(stream, t + 1 - len(pending[stream]))
        for stream in STREAMS:
            for _ in sources[stream]:  # the guard refuses a volume beyond the last
                pass

        # the stills
        n_steady = int(steady.sum())
        mean = {s: sums[s] / n_steady for s in STREAMS}
        sd = {s: _sd(sums[s], squares[s], n_steady) for s in STREAMS}
        del sums, squares
        sd_hi = float(np.percentile(sd["pre"][mask], PERCENTILE))
        sd_window = (0.0, sd_hi if sd_hi > 0 else 1.0)
        stills: dict[str, dict[str, bytes]] = {
            s: {
                "ref": volume_bytes(ref[s]),
                "mean": volume_bytes(quantise(mean[s], *window)),
                "sd": volume_bytes(quantise(sd[s], *sd_window)),
            }
            for s in STREAMS
        }
        streams = {
            s: StreamFiles(
                **{
                    still: writer.put(stream_file(run.run, s, still), stills[s][still])
                    for still in STILLS
                },
                chunks=tuple(chunks[s]),
            )
            for s in STREAMS
        }

        traces = TracesInfo(
            fd=tuple(motion.fd) if motion else (),
            dvars=tuple(motion.dvars) if motion else (),
            non_steady=non_steady,
        )
        traces_bytes = info_text(traces).encode()
        traces_rel = f"{run.run}/{TRACES_JSON}"
        (tmp_root / traces_rel).write_bytes(traces_bytes)
        traces_file = PackFile(
            file=traces_rel,
            bytes=len(traces_bytes),
            sha256=hashlib.sha256(traces_bytes).hexdigest(),
        )

        info = PackInfo(
            surfcensus_version=__version__,
            run=run.run,
            session=session_dir.name,
            config_hash=config.pack_hash(),
            settings=_settings(config),
            inputs=inputs,
            grid_key=run.grid.key,
            anchor_run=run.anchor_run,
            n_volumes=n,
            tr_s=run.tr_s,
            window=window,
            sd_window=sd_window,
            chunk_volumes=k_chunk,
            stills_volumes=n_steady,
            raw_affine=run.raw_affine.choice,
            reference_image=run.raw_affine.reference_image,
            rigid=run.rigid.rigid,
            corrections=corrections_info(run.corrections),
            streams=streams,
            traces=traces_file,
            content_digest="0" * 64,
        )
        parts = _digest_parts(
            info,
            {s: hashers[s].hexdigest() for s in STREAMS},
            {s: {still: _sha(stills[s][still]) for still in STILLS} for s in STREAMS},
            _sha(traces_bytes),
            _contour_digests(session_dir, info.grid_key),
        )
        info = info.model_copy(update={"content_digest": _digest(parts)})
        (tmp_root / run.run / PACK_JSON).write_text(info_text(info))
        _swap_into_place(session_dir, tmp_root, run.run)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    return info


def _settings(config: EpiConfig) -> PackSettings:
    return PackSettings(
        portable_voxel_map=config.portable_voxel_map,
        chunk_bytes=config.chunk_bytes,
        window_step=WINDOW_STEP,
        percentile=PERCENTILE,
        zlib_level=ZLIB_LEVEL,
        contour_unit=CONTOUR_UNIT,
    )


@contextmanager
def _locked(directory: Path) -> Iterator[bool]:
    """``flock`` on the directory itself, as a render's media directory is locked; yields
    whether a lock is held. Without ``fcntl``, or where the file system refuses it, none."""
    descriptor = -1
    locked = False
    try:
        if fcntl is not None:
            try:
                descriptor = os.open(directory, os.O_RDONLY)
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                locked = True
            except OSError:
                pass
        yield locked
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _swap_into_place(session_dir: Path, tmp_root: Path, run: str) -> None:
    """Move ``tmp_root/run`` to ``session_dir/run``, the earlier one aside first. Between the
    two renames, and only then, the run's directory is not there; an exception between them
    (an interrupt) moves the earlier one back."""
    final = session_dir / run
    with _locked(session_dir) as locked:
        if locked:  # what a killed call of this run left; under the lock no other call is in it
            mine = tmp_root.name
            for entry in session_dir.iterdir():
                name = entry.name
                if name != mine and re.fullmatch(rf"\.{run}\.[0-9a-f]{{16}}\.(tmp|old)", name):
                    shutil.rmtree(entry, ignore_errors=True)
        old = None
        try:
            if final.is_symlink() or final.exists():
                old = session_dir / f".{run}.{secrets.token_hex(8)}.old"
                os.replace(final, old)
            os.replace(tmp_root / run, final)
        except BaseException:
            # interrupted between the two renames (a Ctrl-C, a SIGTERM taken as one): the
            # earlier pack goes back, so that a session.json that lists the run finds it
            if old is not None and not (final.is_symlink() or final.exists()):
                os.replace(old, final)
            raise
    if old is not None:
        if old.is_symlink() or old.is_file():
            old.unlink()
        else:
            shutil.rmtree(old, ignore_errors=True)


# --- the digest -----------------------------------------------------------------------------


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _defining_json(info: PackInfo) -> str:
    """The JSON that defines a pack: ``pack.json`` without the content digest, without the
    version of the package that wrote it, without the ids of the run, its session and its
    anchor (keyed names, drawn anew per call in step 1; what the anchor is shows in the raw
    values), and without the name, compressed size and SHA-256 of every file entry (an object
    with a ``file`` key): names hold the run id, sizes and digests depend on the zlib build.
    What stays of a file entry is what it says about the values (``first`` and ``count`` of a
    chunk)."""

    def strip(value: Any) -> Any:
        if isinstance(value, dict):
            drop = {"file", "bytes", "sha256"} if "file" in value else set()
            return {k: strip(v) for k, v in value.items() if k not in drop}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value

    data = info.model_dump(mode="json")
    for key in ("content_digest", "surfcensus_version", "run", "session", "anchor_run"):
        data.pop(key)
    return canonical_json(strip(data))


def _contour_digests(session_dir: Path, grid_key: str) -> dict[str, str] | None:
    """The SHA-256 of the decoded contour file of every plane of a grid, or None when the
    session directory holds no contours of that grid. A plane whose file is missing while
    others are there is an error: contours are written whole by :func:`write_contours`."""
    folder = Path(session_dir) / CONTOURS_DIR / grid_key
    if not folder.is_dir():
        return None
    out = {}
    for plane in PLANES:
        path = Path(session_dir) / contour_file(grid_key, plane)
        out[plane] = _sha(inflate_checked(path, None))
    return out


def _digest_parts(
    info: PackInfo,
    volumes: Mapping[str, str],
    stills: Mapping[str, Mapping[str, str]],
    traces: str,
    contours: Mapping[str, str] | None,
) -> list[str]:
    lines = [DIGEST_FORMAT, f"json {_sha(_defining_json(info).encode())}"]
    lines += [f"volumes {s} {volumes[s]}" for s in STREAMS]
    lines += [f"still {s} {still} {stills[s][still]}" for s in STREAMS for still in STILLS]
    lines.append(f"traces {traces}")
    if contours is None:
        lines.append("contours none")
    else:
        lines += [f"contours {plane} {contours[plane]}" for plane in PLANES]
    return lines


def _digest(lines: Sequence[str]) -> str:
    return _sha(("\n".join(lines) + "\n").encode())


def content_digest(session_dir: Path, info: PackInfo) -> str:
    """SHA-256 over the decoded 8-bit volumes (raw then pre, in volume order), the references,
    the stills, the traces as written and the decoded contour bytes of its grid, in a fixed
    order: the identity of a pack whatever zlib wrote it.

    Precisely: the SHA-256 of the lines of :func:`_digest_parts` joined by newlines, which are
    the rule's name, the SHA-256 of the defining JSON (``pack.json`` without the digest, the
    package version and the compressed sizes and digests of its files; :func:`_defining_json`),
    then per stream the SHA-256 of its decoded volumes one after the other, per stream and
    still the SHA-256 of the decoded still, the SHA-256 of ``traces.json``, and per plane the
    SHA-256 of the inflated contour file of the grid, or ``contours none``. Each compressed
    file is checked against the SHA-256 that ``info`` lists for it before it is inflated
    (``ValueError`` otherwise); the contour files are checked by their content only, since
    ``pack.json`` does not list them. :func:`pack_run` computes the same lines while it writes,
    from the values in memory.
    """
    session_dir = Path(session_dir)
    volumes = {}
    for stream in STREAMS:
        hasher = hashlib.sha256()
        for data in stream_bytes(session_dir, info, stream):
            hasher.update(data)
        volumes[stream] = hasher.hexdigest()
    stills = {
        s: {
            still: _sha(
                inflate_checked(
                    session_dir / getattr(info.streams[s], still).file,
                    getattr(info.streams[s], still).sha256,
                )
            )
            for still in STILLS
        }
        for s in STREAMS
    }
    traces_path = session_dir / info.traces.file
    traces = traces_path.read_bytes()
    if _sha(traces) != info.traces.sha256:
        raise ValueError(f"{TRACES_JSON} of run {info.run} is not the file pack.json lists")
    contours = _contour_digests(session_dir, info.grid_key)
    return _digest(_digest_parts(info, volumes, stills, _sha(traces), contours))


# --- contours and the session ---------------------------------------------------------------


def write_contours(
    session_dir: Path, grid_key: str, planes: Sequence[PlaneContours]
) -> Mapping[str, ContourFile]:
    """Write ``contours/<grid_key>/<plane>.bin`` and return, per plane, the
    ``func.model.ContourFile`` of it.

    Each file is the plane's data compressed with zlib at :data:`ZLIB_LEVEL`, written
    atomically, and only when the file there does not hold those bytes already (a second call
    into the same directory leaves unchanged contours as they are, under a page that serves
    them). ``planes`` must hold each of the three planes once; the result is in the
    order of ``func.PLANES``.
    """
    by_plane = {p.plane: p for p in planes}
    if len(by_plane) != len(planes) or set(by_plane) != set(PLANES):
        raise ValueError("contours need each of the planes sagittal, coronal and axial once")
    out: dict[str, ContourFile] = {}
    for plane in PLANES:
        item = by_plane[plane]
        relative = contour_file(grid_key, plane)
        blob = zlib.compress(item.data, ZLIB_LEVEL)
        target = Path(session_dir) / relative
        try:
            same = target.is_file() and not target.is_symlink() and target.read_bytes() == blob
        except OSError:
            same = False
        if not same:  # a file that a page may be reading is replaced only when it changes
            with atomic_path(target) as tmp:
                tmp.write_bytes(blob)
        out[plane] = ContourFile(
            file=relative,
            bytes=len(blob),
            sha256=_sha(blob),
            slices=item.slices,
            lines=item.lines,
            points=item.points,
        )
    return out


def _frame_info(d: FrameDisplacement) -> FrameInfo:
    def r(x: float) -> float:
        return round(float(x), 4)

    return FrameInfo(
        mean_mm=r(d.mean_mm),
        max_mm=r(d.max_mm),
        rotation_deg=r(d.rotation_deg),
        translation_mm=r(d.translation_mm),
        translation_ras=(
            r(d.translation_ras[0]),
            r(d.translation_ras[1]),
            r(d.translation_ras[2]),
        ),
        shear_scale=r(d.shear_scale),
        flagged=bool(d.flagged),
    )


def _remove_run_dir(session_dir: Path, run: str) -> None:
    path = session_dir / run
    with _locked(session_dir):
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def slice_planes(
    grid: Grid, display_to_anat: Matrix4 = IDENTITY4
) -> tuple[dict[str, tuple[float, float, float]], dict[str, tuple[float, ...]]]:
    """Per plane of ``grid``, the unit normal of its slices in the anatomy's world (RAS,
    pointing the way the slice index grows) and, for each slice, the signed distance of the
    slice plane from the origin of the anatomy's world along that normal, in mm
    (``GridInfo.slice_normal`` and ``GridInfo.slice_mm``), rounded to 1e-6.

    Plane ``n`` is the plane of voxel coordinate ``n`` equal to ``k``. With ``A = P . G``
    (display voxel to the anatomy's world) and ``g`` row ``n`` of the 3x3 part of ``A^-1``
    and ``c`` its translation, the plane is ``g . x = k - c``; the normal is ``g / |g|`` and the
    distance ``(k - c) / |g|``. The origin is a point of the anatomy, the same for every grid
    and every session of a subject that share the anatomy's world, so a plane on one grid and a
    plane on another, of another run or session, are compared by these numbers alone. For a
    grid along the world axes the normal is an axis and the distance the plane's coordinate
    (negated for an axis that runs backwards); for an oblique grid the normal is tilted and the
    numbers stay exact: the three planes of the slices on screen meet in one point ``x`` with
    ``normal_p . x = distance_p``, and on another grid the slice of plane ``p`` nearest to
    that place is the one whose distance is nearest ``normal'_p . x``. Computed with
    ``geometry.compose`` and ``geometry.inverse``, in Python floats."""
    inv = geometry.inverse(geometry.compose(display_to_anat, grid.affine))
    normals: dict[str, tuple[float, float, float]] = {}
    distances: dict[str, tuple[float, ...]] = {}
    for n, plane in enumerate(PLANES):
        g = inv[n][:3]
        c = inv[n][3]
        length = math.sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2])
        normals[plane] = (
            round(g[0] / length, 6) + 0.0,
            round(g[1] / length, 6) + 0.0,
            round(g[2] / length, 6) + 0.0,
        )
        distances[plane] = tuple(
            round((k - c) / length, 6) + 0.0 for k in range(int(grid.shape[n]))
        )
    return normals, distances


def pack_session(
    out_dir: Path,
    session_id: str,
    plan: SessionPlan,
    runs: Sequence[RunPackInput],
    contours: Mapping[str, Sequence[PlaneContours]] | None,
    config: EpiConfig,
    progress: Callable[[str, PackInfo | None, str | None, float, bool], None] | None = None,
    *,
    force: bool = False,
    keep: Collection[str] | None = None,
) -> SessionInfo:
    """Pack a whole session into ``<out_dir>/<session_id>/``: the contours of each grid
    (``contours`` by grid key; None without a FreeSurfer subject), every run of ``runs``, then
    ``session.json`` last. A run that raises :class:`PackError` is listed in ``session.json``
    with its reason and the others are packed. ``progress`` is called after each run of
    ``plan.runs`` without a reason, with its id, its pack (None when it failed), the reason
    of a failure, the seconds it took and whether its earlier pack was kept.

    A run whose pack in the directory is current is kept and not packed again
    (:func:`current_pack`: whole on disk, made by this version of the package and format
    from the same inputs, by role, base, size and SHA-256, the session's inputs included,
    with the same settings, anchor and grid); ``force`` packs every run again.

    ``keep`` is for a run directory (step 2), whose skip decision comes from the catalog and
    not from the files (``func.batch.pack_plan``): the runs it names are taken from their
    ``pack.json`` as they are, without hashing their inputs, when the pack is whole on disk
    (``func.packread.pack_on_disk``) and of this session and config hash, else packed; every
    other run of ``runs`` is packed, and :func:`current_pack` is not asked. A kept run need not
    be among ``runs``. ``force`` packs every run again here as well. Without ``keep`` (None)
    the rule of :func:`current_pack` holds, as in step 1.

    What a reader of the directory sees while this runs: the earlier ``session.json`` stays
    until the new one is renamed over it, last, so a page that is served from the directory
    keeps a whole session throughout. A run that is packed again is moved into place by two
    renames under the ``flock`` of the session directory (the earlier pack aside, the new one
    in), so between those two renames, and only then, the run's directory is not there. The
    run directories that the new ``session.json`` does not list as packed (runs that are no
    longer packed, runs that left the session, runs not selected) and the contours of grids
    it no longer lists are removed after it is written, never before. An interrupted call
    leaves the earlier ``session.json`` with every pack it lists whole, earlier or new.

    The session is one stretch of the subject's stream (decision D38): its packed runs one
    after the other in the order of ``plan.runs`` (acquisition order), each
    listed with its volumes, its TR, the index of its first volume within the session, and its
    own windows, from which the page makes the common scale of the stream. Runs on different
    grids are packed as groups, one per grid of ``plan.grids``, each with its own contours,
    and each grid is listed with its slice planes as normals and distances from the origin of
    the anatomy (:func:`slice_planes`), so that the page keeps one place in the anatomy across
    grids and sessions without the affine. A run of ``plan.runs`` without a reason that
    ``runs`` does not hold is listed as ``pack failure``. A run that is not packed loses a
    pack directory that an earlier call left, after the new ``session.json`` is written.
    ``session.json`` is written last, so it marks a session whose every pack is in place.

    Like :func:`pack_run`, this writes the tool's display cache only, under ``out_dir``: no
    image volume, and nothing into or beside the data.
    """
    if not re.fullmatch(ID_PATTERN, session_id):
        raise ValueError(f"not a session id: {session_id!r}")
    session_dir = Path(out_dir) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    grid_keys = list(dict.fromkeys(plan.grid_of[r.run] for r in plan.runs if r.run in plan.grid_of))
    grid_keys += [k for k in plan.grids if k not in grid_keys]
    contour_files: dict[str, Mapping[str, ContourFile] | None] = {}
    for key in grid_keys:
        if contours is not None and key in contours:
            contour_files[key] = write_contours(session_dir, key, contours[key])
        else:
            contour_files[key] = None

    given = {r.run: r for r in runs}
    reasons: dict[str, str] = {}
    details: dict[str, str] = dict(plan.details)
    packs: dict[str, PackInfo] = {}
    for item in plan.runs:
        if item.reason is not None:
            reasons[item.run] = item.reason
            continue
        started = time.perf_counter()
        kept = None
        if keep is not None and not force and item.run in keep:
            kept = _kept_pack(session_dir, item.run, config)
        if kept is None and item.run not in given:
            reasons[item.run] = PACK_FAILURE
        else:
            if kept is None and keep is None and not force:
                kept = current_pack(session_dir, given[item.run], config)
            try:
                packs[item.run] = (
                    kept if kept is not None else pack_run(session_dir, given[item.run], config)
                )
            except PackError as err:
                reasons[item.run] = err.reason
                details[item.run] = err.detail
            if progress is not None:
                progress(
                    item.run,
                    packs.get(item.run),
                    reasons.get(item.run),
                    time.perf_counter() - started,
                    kept is not None,
                )

    listed: list[SessionRunInfo] = []
    first = 0
    for order, item in enumerate(plan.runs):
        frame = None
        if item.reason is None and plan.frame is not None and item.run in plan.frame.runs:
            frame = _frame_info(plan.frame.runs[item.run])
        pack = packs.get(item.run)
        listed.append(
            SessionRunInfo(
                run=item.run,
                name=item.name,
                acquisition_order=order,
                anchor=plan.anchor is not None and item.run == plan.anchor.run,
                grid_key=plan.grid_of.get(item.run) if item.reason is None else None,
                frame=frame,
                reason=reasons.get(item.run),
                detail=details.get(item.run) if item.run in reasons else None,
                session_order=(plan.order.index(item.run) if item.run in plan.order else None),
                n_volumes=None if pack is None else pack.n_volumes,
                tr_s=None if pack is None else pack.tr_s,
                first=None if pack is None else first,
                window=None if pack is None else pack.window,
                sd_window=None if pack is None else pack.sd_window,
            )
        )
        if pack is not None:
            first += pack.n_volumes

    registration = None
    if plan.anchor is not None and plan.frame is not None:
        registration = RegistrationInfo(
            anchor=plan.anchor.run,
            anchor_packed=plan.anchor.run in packs,
            note=plan.anchor_note,
            rule=plan.anchor.rule,
            limit_mm=plan.frame.limit_mm,
            shared_frame_plausible=plan.frame.shared_frame_plausible,
            anchor_is_outlier=plan.frame.anchor_is_outlier,
            portable_voxel_map=config.portable_voxel_map,
        )
    keys = [k for k in grid_keys if k in plan.grids]
    to_anat = {g: given[r].display_to_anat for r, g in plan.grid_of.items() if r in given}
    grids = []
    for key in keys:
        grid = plan.grids[key]
        normals, distances = slice_planes(grid, to_anat.get(key, IDENTITY4))
        files = contour_files[key]
        grids.append(
            GridInfo(
                grid_key=key,
                shape=grid.shape,
                voxel_mm=geometry.voxel_mm(grid),
                oblique_deg=geometry.oblique_deg(grid),
                edge_letters=dict(geometry.edge_letters(grid)),
                contours=None if files is None else dict(files),
                slice_normal=normals,
                slice_mm=distances,
            )
        )
    label = plan.participant if plan.session is None else f"{plan.participant} ses-{plan.session}"
    info = SessionInfo(
        surfcensus_version=__version__,
        session=session_id,
        label=label,
        config_hash=config.session_hash(),
        fd_threshold_mm=config.fd_threshold_mm,
        registration=registration,
        grids=tuple(grids),
        runs=tuple(listed),
        contours_from=(
            plan.fs_subject if any(v is not None for v in contour_files.values()) else None
        ),
    )
    with atomic_path(session_dir / SESSION_JSON) as tmp:
        tmp.write_text(info_text(info))
    _remove_unlisted(session_dir, set(packs), {g.grid_key for g in grids})
    return info


def _remove_unlisted(session_dir: Path, packed: set[str], grid_keys: set[str]) -> None:
    """Remove the run directories of ``session_dir`` that are not among ``packed`` and the
    contour folders of grids not among ``grid_keys``: what an earlier call left that the new
    ``session.json`` does not list. Only names that a pack writes are touched."""
    try:
        entries = list(os.scandir(session_dir))
    except OSError:
        return
    for entry in entries:
        if re.fullmatch(ID_PATTERN, entry.name) and entry.name not in packed:
            _remove_run_dir(session_dir, entry.name)
    folder = session_dir / CONTOURS_DIR
    try:
        grids = list(os.scandir(folder))
    except OSError:
        return
    for entry in grids:
        if re.fullmatch(r"[0-9a-f]{12}", entry.name) and entry.name not in grid_keys:
            shutil.rmtree(entry.path, ignore_errors=True)


def _kept_pack(session_dir: Path, run: str, config: EpiConfig) -> PackInfo | None:
    """The pack of ``run`` as ``pack.json`` holds it, when the catalog says it is current
    (``pack_session(..., keep=...)``): whole on disk by directory listings, of this session
    and of this config hash. Its inputs are not hashed; the catalog's input signature stands
    for them. None otherwise, and the run is packed."""
    session_dir = Path(session_dir)
    if not pack_on_disk(session_dir, run):
        return None
    info = read_pack(session_dir, run)
    if info is None or info.session != session_dir.name or info.config_hash != config.pack_hash():
        return None
    return info


def _whole(session_dir: Path, info: PackInfo) -> bool:
    """Every file ``pack.json`` lists is a regular file of the size it lists."""
    files: list[PackFile] = [info.traces]
    for stream in STREAMS:
        if stream not in info.streams:
            return False
        files += [getattr(info.streams[stream], still) for still in STILLS]
        files += list(info.streams[stream].chunks)
    for entry in files:
        try:
            st = os.stat(session_dir / entry.file)
        except OSError:
            return False
        if not stat.S_ISREG(st.st_mode) or st.st_size != entry.bytes:
            return False
    return True


def current_pack(session_dir: Path, run: RunPackInput, config: EpiConfig) -> PackInfo | None:
    """The pack of ``run`` in ``session_dir`` when it is what :func:`pack_run` would make now,
    else None.

    Current means: ``pack.json`` reads as a pack of this format and version, of this run and
    session, written by this version of the package; every file it lists is there with the
    size it lists; its config hash and settings are those of ``config``; its anchor and grid
    are those of ``run``; its inputs (role, base, size and SHA-256 of every file, the anchor's
    transform, the fsnative transform and the FreeSurfer files included) are those of the
    files now; and what it states from them (raw affine, reference image, rigidity, TR,
    corrections) is what they state now. The inputs are hashed (some 100 MB a run, seconds),
    which is cheap beside packing; no volume is read. The contours are not compared: they are
    made from inputs that are compared and from the settings."""
    session_dir = Path(session_dir)
    info = read_info(session_dir / run.run / PACK_JSON, PackInfo)
    if (
        info is None
        or info.version != PACK_VERSION
        or info.surfcensus_version != __version__
        or info.run != run.run
        or info.session != session_dir.name
        or info.config_hash != config.pack_hash()
        or info.settings != _settings(config)
        or info.anchor_run != run.anchor_run
        or info.grid_key != run.grid.key
        or not _whole(session_dir, info)
    ):
        return None
    try:
        inputs = _inputs(run.files, run.session_inputs)
    except PackError:
        return None
    if (
        info.inputs != inputs
        or info.raw_affine != run.raw_affine.choice
        or info.reference_image != run.raw_affine.reference_image
        or info.rigid != run.rigid.rigid
        or info.tr_s != run.tr_s
        or info.corrections != corrections_info(run.corrections)
    ):
        return None
    return info
