"""vertex_map.npz: the native vertices a click mark can resolve to.

Per hemisphere ``h`` in ``lh``, ``rh`` (docs/rendering.md):

* ``{h}_index``   int32 (k,): native vertex indices, ascending;
* ``{h}_pial``    float32 (k, 3): their pial positions in tkr RAS;
* ``{h}_white``   float32 (k, 3): their white positions in tkr RAS;
* ``{h}_fsavg6``  int32 (k,): fsaverage6 counterparts, -1 until the sphere mapping exists;
* ``{h}_visible`` bool (k,): whether the pial position lies in the region itself, without the
  margin.

The region is a union of boxes; the renderer passes the clickable region of the tiles
(``geometry.clickable_region``). A vertex is included when its pial or its white position
lies in a box of the region grown by the margin on every side (boundary included). A point
of the region then has the same nearest pial vertex, and the same nearest white vertex, in
the map as on the full surface, whenever that vertex is no farther away than the margin.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from jaxtyping import Float

from surfcensus._io import read_npz, write_npz
from surfcensus.fsio.surfaces import Surface
from surfcensus.render.geometry import in_region
from surfcensus.sources.base import HEMIS

_DTYPES = {"index": "<i4", "pial": "<f4", "white": "<f4", "fsavg6": "<i4", "visible": "|b1"}
KEYS = tuple(f"{hemi}_{field}" for hemi in HEMIS for field in _DTYPES)


def build_vertex_map(
    surfaces: Mapping[str, Surface], region: Float[np.ndarray, "boxes 2 3"], margin_mm: float
) -> dict[str, np.ndarray]:
    """Vertex map from surfaces keyed ``lh_pial``, ``lh_white``, ``rh_pial``, ``rh_white``.

    ``region[b]`` is the low and the high corner of box ``b`` in tkr RAS.
    """
    vmap: dict[str, np.ndarray] = {}
    for hemi in HEMIS:
        pial, white = surfaces[f"{hemi}_pial"].vertices, surfaces[f"{hemi}_white"].vertices
        if pial.shape != white.shape:
            raise ValueError(
                f"{hemi} pial and white differ in vertex count: {len(pial)} and {len(white)}"
            )
        near = in_region(pial, region, margin_mm) | in_region(white, region, margin_mm)
        index = np.flatnonzero(near)
        vmap[f"{hemi}_index"] = index.astype("<i4")
        vmap[f"{hemi}_pial"] = pial[index].astype("<f4")
        vmap[f"{hemi}_white"] = white[index].astype("<f4")
        vmap[f"{hemi}_fsavg6"] = np.full(len(index), -1, dtype="<i4")
        vmap[f"{hemi}_visible"] = in_region(pial[index], region)
    return vmap


def write_vertex_map(path: Path | str, vmap: Mapping[str, np.ndarray]) -> None:
    """Write an ``.npz`` that ``numpy.load`` reads, byte-identical for identical input.

    The entries are written in the order of ``KEYS``, each cast to the dtype of its field,
    by the byte-stable writer of ``surfcensus._io``. Atomic.
    """
    if set(vmap) != set(KEYS):
        raise ValueError(f"a vertex map has exactly the keys {KEYS}, got {sorted(vmap)}")
    write_npz(
        path,
        {key: np.ascontiguousarray(vmap[key], dtype=_DTYPES[key.split("_")[1]]) for key in KEYS},
    )


def read_vertex_map(path: Path | str) -> dict[str, np.ndarray]:
    archive = read_npz(path)
    return {key: archive[key] for key in KEYS}
