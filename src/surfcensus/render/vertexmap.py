"""vertex_map.npz: the native vertices near the view box, for resolving click marks.

Per hemisphere ``h`` in ``lh``, ``rh`` (docs/rendering.md):

* ``{h}_index``  int32 (k,): native vertex indices, ascending;
* ``{h}_pial``   float32 (k, 3): their pial positions in tkr RAS;
* ``{h}_white``  float32 (k, 3): their white positions in tkr RAS;
* ``{h}_fsavg6`` int32 (k,): fsaverage6 counterparts, -1 until the sphere mapping exists.

A vertex is included when its pial or its white position lies inside the box grown by the
margin on every side (boundary included).
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from surfcensus.fsio.surfaces import Surface
from surfcensus.render.encode import atomic_path
from surfcensus.render.model import BoxInfo

HEMIS = ("lh", "rh")
_DTYPES = {"index": "<i4", "pial": "<f4", "white": "<f4", "fsavg6": "<i4"}
KEYS = tuple(f"{hemi}_{field}" for hemi in HEMIS for field in _DTYPES)


def build_vertex_map(
    surfaces: Mapping[str, Surface], box: BoxInfo, margin_mm: float
) -> dict[str, np.ndarray]:
    """Vertex map from surfaces keyed ``lh_pial``, ``lh_white``, ``rh_pial``, ``rh_white``."""
    low, high = np.array(box.low) - margin_mm, np.array(box.high) + margin_mm
    vmap: dict[str, np.ndarray] = {}
    for hemi in HEMIS:
        pial, white = surfaces[f"{hemi}_pial"].vertices, surfaces[f"{hemi}_white"].vertices
        if pial.shape != white.shape:
            raise ValueError(
                f"{hemi} pial and white differ in vertex count: {len(pial)} and {len(white)}"
            )
        inside = np.zeros(len(pial), dtype=bool)
        for xyz in (pial, white):
            inside |= ((xyz >= low) & (xyz <= high)).all(axis=1)
        index = np.flatnonzero(inside)
        vmap[f"{hemi}_index"] = index.astype("<i4")
        vmap[f"{hemi}_pial"] = pial[index].astype("<f4")
        vmap[f"{hemi}_white"] = white[index].astype("<f4")
        vmap[f"{hemi}_fsavg6"] = np.full(len(index), -1, dtype="<i4")
    return vmap


def write_vertex_map(path: Path | str, vmap: Mapping[str, np.ndarray]) -> None:
    """Write an ``.npz`` that ``numpy.load`` reads, byte-identical for identical input.

    ``numpy.savez`` stamps every zip entry with the current time, so the archive is written
    here: entries in the order of ``KEYS``, stored uncompressed, dated 1980-01-01. Atomic.
    """
    if set(vmap) != set(KEYS):
        raise ValueError(f"a vertex map has exactly the keys {KEYS}, got {sorted(vmap)}")
    with atomic_path(path) as tmp, zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as archive:
        for key in KEYS:
            buffer = io.BytesIO()
            dtype = _DTYPES[key.split("_")[1]]
            np.lib.format.write_array(buffer, np.ascontiguousarray(vmap[key], dtype=dtype))
            entry = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3  # Unix on every platform, with fixed permissions
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, buffer.getvalue())


def read_vertex_map(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in KEYS}
