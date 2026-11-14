"""The EPI movie interface: the raw and the preprocessed EPI series of every functional run of
a session, as moving pictures on the same slices with the FreeSurfer white and pial surfaces
as contours (docs/epi_interface.md; decisions D36 and D37).

Two uses, named wherever a view is described: quality control of the raw data (motion,
artefacts in single volumes, distortion and dropout as acquired, movement between runs) and
quality control of the preprocessing (what was corrected, what remains, how the preprocessed
series fits the white and pial contours). The raw stream is the raw BIDS series placed once per
session into the anatomy by the anchor run's stored transform and nothing else; the
preprocessed stream is the series as the pipeline wrote it. Their difference is the signal.

The modules of step 1 ("one session, view only"):

* ``func.bids``: BIDS entities, run keys, name order, sidecar times, the raw listing (E1.A).
* ``func.itk``: ITK text transforms as RAS point maps (E1.A).
* ``func.connectors``: what is specific to a pipeline, behind one protocol; ``fmriprep`` (E1.A).
* ``func.session``: the anchor, the frame check, the raw affine, the plan of a session (E1.B).
* ``func.geometry``: the display grid, the voxel map, the resampling of the raw stream (E1.B).
* ``func.contours``: the white and pial contours in the voxels of a grid (E1.B).
* ``func.model``: the contents of ``pack.json``, ``session.json`` and ``traces.json`` (E1.0).
* ``func.pack``: the pack, format ``epi-pack`` version 2 (E1.C); ``func.packread`` reads one
  back for tests (E1.C).
* ``func.api``: ``epi_session``, the function behind ``surfcensus epi-session`` (E1.Z).

This file holds the names of the pack's files and the patterns by which the app serves them;
it imports the standard library only, so that the server can match a path without loading
numpy or nibabel.

Where a pack lives::

    <out>/<session_id>/
      session.json                       registration, anchor, frame check, grids
      contours/<grid_key>/<plane>.bin    one set per grid
      <run_id>/
        pack.json                        inputs (sizes, SHA-256), settings, content digest
        raw/ref.bin raw/mean.bin raw/sd.bin raw/c0000.bin ...
        pre/ref.bin pre/mean.bin pre/sd.bin pre/c0000.bin ...
        traces.json                      framewise displacement, DVARS, non-steady-state volumes

``<out>`` is ``epi/<participant_uid>/`` of a run directory from step 2, and the ``--out``
directory of ``surfcensus epi-session`` in step 1. ``session.json`` and ``pack.json`` are never served as
files; the page receives a shape built from them.
"""

from __future__ import annotations

import hashlib
import hmac
import re

PACK_FORMAT = "epi-pack"
PACK_VERSION = 2
"""The container of the pack. The formats of chunks, stills and contours are those of stage
1's version 1 (``func-viewer-pack``); the container is version 2 because the manifest is split
into ``session.json`` and ``pack.json`` and runs are named by id."""

SESSION_JSON = "session.json"
PACK_JSON = "pack.json"
TRACES_JSON = "traces.json"
STREAM_JSON = "stream.json"
"""At the root of an ``--out`` directory: the sessions of one subject in the order of the
subject's one stream (decision D38); ``func.model.StreamInfo``."""
CONTOURS_DIR = "contours"

STREAMS: tuple[str, str] = ("raw", "pre")
STILLS: tuple[str, str, str] = ("ref", "mean", "sd")
"""The reference volume of the residuals, the temporal mean and the temporal SD."""
PLANES: tuple[str, str, str] = ("sagittal", "coronal", "axial")
"""The slice axis of plane n is voxel axis n of the display grid."""
SURFACES: tuple[str, str, str, str] = ("lh.white", "rh.white", "lh.pial", "rh.pial")
"""In the order of the contour files: white first, so that pial is drawn over it."""
CONTOUR_UNIT = 64
"""Contour points are stored in 1/64 voxel."""

ID_PATTERN = r"[0-9a-f]{16}"
"""What a session id and a run id look like: the form of a ``subject_uid``."""
GRID_KEY_PATTERN = r"[0-9a-f]{12}"
"""What ``records.Grid.key`` looks like."""

_SESSION_PATH = re.compile(rf"(?:{CONTOURS_DIR}/({GRID_KEY_PATTERN})/({'|'.join(PLANES)})\.bin)")
# ASCII digits only: in Python ``\d`` matches every Unicode decimal digit (decision D19)
_RUN_PATH = re.compile(
    rf"({ID_PATTERN})/({'|'.join(STREAMS)})/(ref|mean|sd|c(?:[0-9]{{4}}|[1-9][0-9]{{4,}}))\.bin"
)


def chunk_name(index: int) -> str:
    """The file name of chunk ``index`` of a stream: ``c0000.bin``; five digits and more, with
    no leading zero, from 10,000 on."""
    if index < 0:
        raise ValueError("a chunk index is not negative")
    return f"c{index:04d}.bin"


def stream_file(run: str, stream: str, name: str) -> str:
    """The path of a still or a chunk of a run, relative to the session directory:
    ``<run>/raw/mean.bin``. ``name`` is ``ref``, ``mean``, ``sd`` or a chunk name."""
    if stream not in STREAMS:
        raise ValueError(f"no stream {stream!r}")
    file = f"{name}.bin" if name in STILLS else name
    path = f"{run}/{stream}/{file}"
    if not _RUN_PATH.fullmatch(path):
        raise ValueError(f"not a file of a pack: {path}")
    return path


def contour_file(grid_key: str, plane: str) -> str:
    """The path of the contours of one plane of a grid, relative to the session directory."""
    path = f"{CONTOURS_DIR}/{grid_key}/{plane}.bin"
    if not _SESSION_PATH.fullmatch(path):
        raise ValueError(f"not a contour file: {path}")
    return path


def is_pack_path(path: str) -> bool:
    """Whether ``path`` (relative to a session directory, POSIX) names a file the app may
    serve: a contour file or a still or chunk of a run. ``session.json``, ``pack.json``,
    ``traces.json`` and every other name are not; ``traces.json`` reaches the page through the
    session shape (docs/epi_interface.md, Serving packs)."""
    return bool(_SESSION_PATH.fullmatch(path) or _RUN_PATH.fullmatch(path))


def func_id(key: bytes, *parts: str) -> str:
    """16 hex characters of HMAC-SHA256 under ``key`` over ``parts`` joined by a NUL.

    The form of a session id and a run id. In step 2 ``privacy.py`` owns ``session_uid`` and
    ``run_uid`` and keys them with the run's key or salt (docs/epi_interface.md, Identifiers);
    step 1 has no run directory and keys them with what ``func.api.epi_session`` is given.
    """
    if not key:
        raise ValueError("an id needs a key")
    message = "\0".join(parts).encode()
    return hmac.new(key, message, hashlib.sha256).hexdigest()[:16]
