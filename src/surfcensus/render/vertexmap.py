"""vertex_map.npz: the native vertices a click mark can resolve to.

Per hemisphere ``h`` in ``lh``, ``rh`` (docs/rendering.md):

* ``{h}_index``   int32 (k,): native vertex indices, ascending;
* ``{h}_pial``    float32 (k, 3): their pial positions in tkr RAS;
* ``{h}_white``   float32 (k, 3): their white positions in tkr RAS;
* ``{h}_fsavg6``  int32 (k,): for each, the vertex of the template mesh nearest its place on
  the registered sphere (:func:`fill_fsaverage`); -1 throughout a hemisphere that was
  rendered without a template or without a usable ``sphere.reg``. The key has this name
  whatever the mesh; ``render.json`` names the mesh;
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
from jaxtyping import Float, Int

from surfcensus._io import read_npz, write_npz
from surfcensus.fsio.surfaces import Surface
from surfcensus.render.geometry import in_region
from surfcensus.sources.base import HEMIS, Hemi

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


def fill_fsaverage(
    vmap: Mapping[str, np.ndarray], nearest_target: Mapping[Hemi, Int[np.ndarray, " _"]]
) -> dict[str, np.ndarray]:
    """``vmap`` with the template counterparts of its vertices; ``vmap`` is not changed.

    ``nearest_target[h][v]`` is the template vertex nearest native vertex ``v`` of
    hemisphere ``h`` on the registered sphere (``SphereMap.nearest_target``), for all
    vertices of the hemisphere: ``{h}_fsavg6`` becomes ``nearest_target[h][{h}_index]``. A
    hemisphere that the mapping lacks gets -1 throughout, also where ``vmap`` held
    counterparts, so the result does not depend on what was filled before.

    Raises ValueError when a mapping does not reach the vertices of the map.
    """
    filled = dict(vmap)
    for hemi in HEMIS:
        index = np.asarray(vmap[f"{hemi}_index"])
        if hemi not in nearest_target:
            filled[f"{hemi}_fsavg6"] = np.full(len(index), -1, dtype="<i4")
            continue
        nearest = np.asarray(nearest_target[hemi])
        if index.size and (index.min() < 0 or index.max() >= len(nearest)):
            raise ValueError(
                f"the {hemi} mapping covers {len(nearest)} vertices, the vertex map names "
                f"vertices {int(index.min())} to {int(index.max())}"
            )
        filled[f"{hemi}_fsavg6"] = nearest[index].astype("<i4")
    return filled


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
